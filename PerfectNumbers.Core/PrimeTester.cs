using System;
using System.Buffers;
using System.Collections.Concurrent;
using Open.Numeric.Primes;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core;

public sealed class PrimeTester(bool useInternal = false)
{
    // TODO: Don't use these tiny arrays. Hard-code the values / checks instead
    private static readonly ulong[] GroupAConstantDivisors = [3UL, 7UL, 11UL, 13UL];
    private static readonly byte[] GroupAIncrementPattern = [20, 10];
    private static readonly byte[] GroupBEndingOrderMod1 = [9, 1];
    private static readonly byte[] GroupBEndingOrderMod3 = [9, 7];
    private static readonly byte[] GroupBEndingOrderMod7 = [7, 1];
    private static readonly byte[] GroupBEndingOrderMod9 = [9, 7, 1];

    // TODO: Would it speed up things if we implement something like ExponentRemainderStepper,
    // MersenneDivisorResidueStepper, or CycleRemainderStepper work with these?
    private static readonly ushort[] Wheel210ResiduesEnding1 = [1, 11, 31, 41, 61, 71, 101, 121, 131, 151, 181, 191];
    private static readonly ushort[] Wheel210ResiduesEnding7 = [17, 37, 47, 67, 97, 107, 127, 137, 157, 167, 187, 197];
    private static readonly ushort[] Wheel210ResiduesEnding9 = [19, 29, 59, 79, 89, 109, 139, 149, 169, 179, 199, 209];
    private const ulong Wheel210 = 210UL;
    internal const int MaxGroupBSequences = 36;

    private const bool UseHeuristicGroupBTrialDivision = false; // Temporary fallback gate for Group B.

    private readonly bool _useLegacyPrimeTester = useInternal;

    private static readonly ulong[] HeuristicSmallCycleSnapshot = MersenneDivisorCycles.Shared.ExportSmallCyclesSnapshot();

    internal enum HeuristicDivisorGroup : byte
    {
        None = 0,
        GroupAConstant = 1,
        GroupAWheel = 2,
        GroupB = 3,
    }

    internal readonly struct HeuristicDivisorCandidate(ulong value, HeuristicDivisorGroup group, byte ending, byte priorityIndex, ushort wheelResidue)
    {
        public readonly ulong Value = value;
        public readonly HeuristicDivisorGroup Group = group;
        public readonly byte Ending = ending;
        public readonly byte PriorityIndex = priorityIndex;
        public readonly ushort WheelResidue = wheelResidue;
    }

    internal readonly struct HeuristicDivisorPreparation(
        in HeuristicDivisorCandidate candidate,
        in MontgomeryDivisorData divisorData,
        ulong cycleLengthHint,
        bool hasCycleLengthHint)
    {
        public readonly HeuristicDivisorCandidate Candidate = candidate;
        public readonly MontgomeryDivisorData DivisorData = divisorData;
        public readonly ulong CycleLengthHint = cycleLengthHint;
        public readonly bool HasCycleLengthHint = hasCycleLengthHint;
        public readonly bool RequiresCycleComputation = !hasCycleLengthHint;
    }

    public bool IsPrime(ulong n, CancellationToken ct)
    {
        if (_useLegacyPrimeTester)
        {
            return LegacyIsPrimeInternal(n, ct);
        }

        return IsPrimeInternal(n, ct);
    }

    [ThreadStatic]
    private static PrimeTester? _tester;

    public static PrimeTester Exclusive => _tester ??= new();

    public static bool IsPrimeGpu(ulong n)
    {
        return Exclusive.IsPrimeGpu(n, CancellationToken.None);
    }

    public static bool IsPrimeGpu(ulong n, ulong limit, byte nMod10)
    {
        return Exclusive.HeuristicIsPrimeGpu(n, limit, nMod10);
    }

    // Optional GPU-assisted primality: batched small-prime sieve on device.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool IsPrimeGpu(ulong n, CancellationToken ct)
    {
        if (_useLegacyPrimeTester || !EnableHeuristicPrimeTesting)
        {
            return LegacyIsPrimeGpu(n, ct);
        }

        return HeuristicIsPrimeGpu(n);
    }

    public static bool LegacyIsPrimeGpu(ulong n, CancellationToken ct)
    {
        // EvenPerfectBitScanner never enqueues exponents <= 3 on the GPU path; keep the legacy guard commented out
        // so ad-hoc callers that expect that behavior must re-enable it intentionally.
        // if (n <= 3UL)
        // {
        //     return n >= 2UL;
        // }

        // Candidate generation (Add/AddPrimes transforms plus ModResidueTracker) already strips even values and multiples of five.
        // The downstream small-factor sweep and Pollard routines factor divisors q = 2kp + 1 with k >= 1 without feeding new
        // exponents back into this path, so the GPU sees the same odd sequence that already passed the CPU residues.

        // EvenPerfectBitScanner seeds p at 136,279,841 and advances monotonically, so production scans never hit the GPU with
        // exponents below 31. Leave this CPU redirect commented out to keep the wrapper branchless while documenting the guard.
        // The ternary fallback below still routes sub-31 values through LegacyIsPrimeInternal for correctness when ad-hoc callers bypass
        // the scanner pipeline.
        // if (n < 31UL)
        // {
        //     return LegacyIsPrimeInternal(n, ct);
        // }

        bool forceCpu = GpuContextPool.ForceCpu;
        Span<ulong> one = stackalloc ulong[1];
        Span<byte> outFlags = stackalloc byte[1];
        // TODO: Inline the single-value GPU sieve fast path from GpuModularArithmeticBenchmarks so this wrapper
        // can skip stackalloc buffers and reuse the pinned upload span the benchmark identified as fastest.
        one[0] = n;
        outFlags[0] = 0;

        if (!forceCpu)
        {
            IsPrimeBatchGpu(one, outFlags);
        }

        bool belowGpuRange = n < 31UL;
        bool gpuReportedPrime = !forceCpu && !belowGpuRange && outFlags[0] != 0;
        bool requiresCpuFallback = forceCpu || belowGpuRange || !gpuReportedPrime;

        return requiresCpuFallback ? LegacyIsPrimeInternal(n, ct) : true;
    }

    public bool HeuristicIsPrimeGpu(ulong n)
    {
        byte nMod10 = (byte)n.Mod10();
        ulong sqrtLimit = ComputeHeuristicSqrt(n);
        return HeuristicIsPrimeGpu(n, sqrtLimit, nMod10);
    }

    private bool HeuristicIsPrimeGpu(ulong n, ulong sqrtLimit, byte nMod10)
    {
        // TODO: Is this condition ever met on the execution path in EvenPerfectBitScanner?
        if (sqrtLimit < 3UL)
        {
            return EvaluateWithOpenNumericFallback(n);
        }

        if (!UseHeuristicGroupBTrialDivision)
        {
            return HeuristicTrialDivisionCpu(n, sqrtLimit, nMod10, includeGroupB: false);
        }

        if (GpuContextPool.ForceCpu)
        {
            return HeuristicIsPrimeCpu(n, sqrtLimit, nMod10);
        }

        bool compositeDetected = HeuristicTrialDivisionGpuDetectsDivisor(n, sqrtLimit, nMod10);
        return !compositeDetected;
    }

    internal static bool HeuristicIsPrimeCpu(ulong n)
    {
        return HeuristicIsPrimeCpu(n, 0UL, (byte)n.Mod10());
    }

    internal static bool HeuristicIsPrimeCpu(ulong n, ulong sqrtLimit, byte nMod10)
    {
        if (sqrtLimit == 0UL)
        {
            sqrtLimit = ComputeHeuristicSqrt(n);
        }

        if (sqrtLimit < 3UL)
        {
            return EvaluateWithOpenNumericFallback(n);
        }

        bool includeGroupB = UseHeuristicGroupBTrialDivision;
        return HeuristicTrialDivisionCpu(n, sqrtLimit, nMod10, includeGroupB);
    }

    private static bool HeuristicTrialDivisionCpu(ulong n, ulong sqrtLimit, byte nMod10, bool includeGroupB)
    {
        Span<HeuristicGroupBSequenceState> groupBBuffer = stackalloc HeuristicGroupBSequenceState[MaxGroupBSequences];
        var enumerator = new HeuristicDivisorEnumerator(sqrtLimit, nMod10, groupBBuffer);

        while (enumerator.TryGetNext(out HeuristicDivisorCandidate candidate))
        {
            if (!includeGroupB && candidate.Group == HeuristicDivisorGroup.GroupB)
            {
                return EvaluateWithOpenNumericFallback(n);
            }

            ulong divisor = candidate.Value;
            if (divisor <= 1UL)
            {
                continue;
            }

            if (n % divisor == 0UL)
            {
                return false;
            }
        }

        return includeGroupB ? true : EvaluateWithOpenNumericFallback(n);
    }

    private bool HeuristicTrialDivisionGpuDetectsDivisor(ulong n, ulong sqrtLimit, byte nMod10)
    {
        Span<HeuristicGroupBSequenceState> groupBBuffer = stackalloc HeuristicGroupBSequenceState[MaxGroupBSequences];
        var enumerator = new HeuristicDivisorEnumerator(sqrtLimit, nMod10, groupBBuffer);

        int batchCapacity = Math.Max(1, HeuristicGpuDivisorBatchSize);
        var candidatePool = ArrayPool<HeuristicDivisorCandidate>.Shared;
        var divisorPool = ThreadStaticPools.UlongPool;
        var hitPool = ThreadStaticPools.BytePool;

        HeuristicDivisorCandidate[]? candidateArray = null;
        ulong[]? divisorArray = null;
        byte[]? hitFlags = null;

        var limiter = GpuPrimeWorkLimiter.Acquire();
        var gpu = PrimeTesterGpuContextPool.Rent();

        bool compositeDetected = false;

        try
        {
            candidateArray = candidatePool.Rent(batchCapacity);
            divisorArray = divisorPool.Rent(batchCapacity);
            hitFlags = hitPool.Rent(batchCapacity);

            var accelerator = gpu.Accelerator;
            var state = GpuKernelState.GetOrCreate(accelerator);

            lock (gpu.ExecutionLock)
            {
                var scratch = state.RentScratch(batchCapacity, accelerator);
                try
                {
                    int count = 0;

                    bool ProcessBatch(int length)
                    {
                        scratch.Input.View.CopyFromCPU(ref divisorArray![0], length);
                        state.HeuristicTrialDivisionKernel(length, scratch.Input.View, n, scratch.Output.View);
                        accelerator.Synchronize();
                        scratch.Output.View.CopyToCPU(ref hitFlags![0], length);

                        for (int i = 0; i < length; i++)
                        {
                            if (hitFlags![i] == 0)
                            {
                                continue;
                            }

                            ulong divisor = candidateArray![i].Value;
                            if (divisor > 1UL && n % divisor == 0UL)
                            {
                                return true;
                            }
                        }

                        return false;
                    }

                    while (enumerator.TryGetNext(out HeuristicDivisorCandidate candidate))
                    {
                        ulong divisor = candidate.Value;
                        if (divisor <= 1UL)
                        {
                            continue;
                        }

                        candidateArray[count] = candidate;
                        divisorArray[count] = divisor;
                        count++;

                        if (count == batchCapacity)
                        {
                            if (ProcessBatch(count))
                            {
                                compositeDetected = true;
                                break;
                            }

                            count = 0;
                        }
                    }

                    if (!compositeDetected && count > 0)
                    {
                        if (ProcessBatch(count))
                        {
                            compositeDetected = true;
                        }
                    }
                }
                finally
                {
                    state.ReturnScratch(scratch);
                }
            }
        }
        finally
        {
            if (hitFlags is not null)
            {
                hitPool.Return(hitFlags);
            }

            if (divisorArray is not null)
            {
                divisorPool.Return(divisorArray);
            }

            if (candidateArray is not null)
            {
                candidatePool.Return(candidateArray);
            }

            gpu.Dispose();
            limiter.Dispose();
        }

        return compositeDetected;
    }

    private static bool EvaluateWithOpenNumericFallback(ulong n)
    {
        return Prime.Numbers.IsPrime(n);
    }

    internal static ulong ComputeHeuristicSqrt(ulong n)
    {
        ulong sqrt = (ulong)Math.Sqrt(n);
        UInt128 square = (UInt128)sqrt * sqrt;

        while (square > n)
        {
            sqrt--;
            square = (UInt128)sqrt * sqrt;
        }

        ulong next = sqrt + 1UL;
        UInt128 nextSquare = (UInt128)next * next;
        while (nextSquare <= n)
        {
            sqrt = next;
            next++;
            nextSquare = (UInt128)next * next;
        }

        return sqrt;
    }

    private static ReadOnlySpan<byte> GetGroupBEndingOrder(byte nMod10) => nMod10 switch
    {
        1 => GroupBEndingOrderMod1,
        3 => GroupBEndingOrderMod3,
        7 => GroupBEndingOrderMod7,
        9 => GroupBEndingOrderMod9,
        _ => ReadOnlySpan<byte>.Empty,
    };

    private static ReadOnlySpan<ushort> GetWheel210ResiduesForEnding(byte ending) => ending switch
    {
        1 => Wheel210ResiduesEnding1,
        7 => Wheel210ResiduesEnding7,
        9 => Wheel210ResiduesEnding9,
        _ => ReadOnlySpan<ushort>.Empty,
    };

    private static int InitializeGroupBStates(byte nMod10, Span<HeuristicGroupBSequenceState> buffer)
    {
        ReadOnlySpan<byte> endings = GetGroupBEndingOrder(nMod10);
        int count = 0;

        for (int priorityIndex = 0; priorityIndex < endings.Length; priorityIndex++)
        {
            ReadOnlySpan<ushort> residues = GetWheel210ResiduesForEnding(endings[priorityIndex]);
            for (int residueIndex = 0; residueIndex < residues.Length; residueIndex++)
            {
                buffer[count++] = new HeuristicGroupBSequenceState(residues[residueIndex], endings[priorityIndex], (byte)priorityIndex);
            }
        }

        return count;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static HeuristicDivisorEnumerator CreateHeuristicDivisorEnumerator(ulong sqrtLimit, byte nMod10, Span<HeuristicGroupBSequenceState> groupBBuffer)
    {
        return new HeuristicDivisorEnumerator(sqrtLimit, nMod10, groupBBuffer);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static MersenneHeuristicDivisorEnumerator CreateMersenneDivisorEnumerator(ulong exponent, ulong maxDivisor)
    {
        return new MersenneHeuristicDivisorEnumerator(exponent, maxDivisor);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static HeuristicDivisorPreparation PrepareHeuristicDivisor(in HeuristicDivisorCandidate candidate)
    {
        ulong divisor = candidate.Value;
        MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(divisor);
        bool hasCycleHint = TryGetCycleLengthHint(divisor, out ulong cycleLength);
        return new HeuristicDivisorPreparation(candidate, divisorData, cycleLength, hasCycleHint);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static ulong ResolveHeuristicCycleLength(
        ulong exponent,
        in HeuristicDivisorPreparation preparation,
        out bool cycleFromHint,
        out bool cycleComputed,
        out bool primeOrderFailed)
    {
        if (preparation.HasCycleLengthHint && preparation.CycleLengthHint != 0UL)
        {
            cycleFromHint = true;
            cycleComputed = true;
            primeOrderFailed = false;
            return preparation.CycleLengthHint;
        }

        ulong divisor = preparation.Candidate.Value;
        MontgomeryDivisorData divisorData = preparation.DivisorData;

        bool trySuccess = MersenneDivisorCycles.TryCalculateCycleLengthForExponentCpu(
            divisor,
            exponent,
            divisorData,
            out ulong computedCycle,
            out bool primeOrderFailedLocal);

        if (trySuccess && computedCycle != 0UL)
        {
            cycleFromHint = false;
            cycleComputed = true;
            primeOrderFailed = primeOrderFailedLocal;
            return computedCycle;
        }

        primeOrderFailed = primeOrderFailedLocal || !trySuccess || computedCycle == 0UL;
        ulong resolvedCycle = MersenneDivisorCycles.CalculateCycleLength(divisor, divisorData, skipPrimeOrderHeuristic: primeOrderFailed);
        cycleFromHint = false;
        cycleComputed = resolvedCycle != 0UL;
        return resolvedCycle;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool TryGetCycleLengthHint(ulong divisor, out ulong cycleLength)
    {
        ulong[] snapshot = HeuristicSmallCycleSnapshot;
        if (divisor < (ulong)snapshot.Length)
        {
            cycleLength = snapshot[(int)divisor];
            if (cycleLength != 0UL)
            {
                return true;
            }
        }

        cycleLength = 0UL;
        return false;
    }

    internal ref struct HeuristicDivisorEnumerator
    {
        private readonly ulong sqrtLimit;
        private int groupAConstantIndex;
        private ulong groupAWheelCandidate;
        private int groupAIncrementIndex;
        private bool groupAWheelActive;
        private Span<HeuristicGroupBSequenceState> groupBStates;

        public HeuristicDivisorEnumerator(ulong sqrtLimit, byte nMod10, Span<HeuristicGroupBSequenceState> groupBBuffer)
        {
            this.sqrtLimit = sqrtLimit;
            groupAConstantIndex = 0;
            groupAWheelCandidate = 23UL;
            groupAIncrementIndex = 0;
            groupAWheelActive = sqrtLimit >= 23UL;

            int count = InitializeGroupBStates(nMod10, groupBBuffer);
            groupBStates = count == 0 ? Span<HeuristicGroupBSequenceState>.Empty : groupBBuffer[..count];

            for (int i = 0; i < groupBStates.Length; i++)
            {
                groupBStates[i].TryNormalize();
            }
        }

        public bool TryGetNext(out HeuristicDivisorCandidate candidate)
		{
			// TODO: Hard-code constant A divisor checks here instead of using an array
            while (groupAConstantIndex < GroupAConstantDivisors.Length)
            {
                ulong value = GroupAConstantDivisors[groupAConstantIndex++];
				if (value > sqrtLimit)
				{
					groupAConstantIndex = GroupAConstantDivisors.Length;
					break;
				}

				// TODO: We're constantly dividing constants mod 10 which is constant too.
                candidate = new HeuristicDivisorCandidate(value, HeuristicDivisorGroup.GroupAConstant, (byte)(value % 10UL), 0, (ushort)value);
                return true;
            }

            if (groupAWheelActive)
            {
                ulong value = groupAWheelCandidate;
                if (value <= sqrtLimit)
                {
                    candidate = new HeuristicDivisorCandidate(value, HeuristicDivisorGroup.GroupAWheel, 3, 0, (ushort)(value % Wheel210));
                    byte increment = GroupAIncrementPattern[groupAIncrementIndex];
                    groupAIncrementIndex ^= 1;

                    ulong next = value + increment;
                    if (next <= value)
                    {
                        groupAWheelActive = false;
                    }
                    else
                    {
                        groupAWheelCandidate = next;
                    }

                    return true;
                }

                groupAWheelActive = false;
            }

            ulong bestCandidate = ulong.MaxValue;
            int bestIndex = -1;

            for (int i = 0; i < groupBStates.Length; i++)
            {
                ref HeuristicGroupBSequenceState state = ref groupBStates[i];
                ulong value = state.Candidate;
                if (value == ulong.MaxValue)
                {
                    continue;
                }

                if (value > sqrtLimit)
                {
                    state.Candidate = ulong.MaxValue;
                    continue;
                }

                if (bestIndex == -1 || value < bestCandidate ||
                    (value == bestCandidate && state.PriorityIndex < groupBStates[bestIndex].PriorityIndex))
                {
                    bestCandidate = value;
                    bestIndex = i;
                }
            }

            if (bestIndex == -1)
            {
                candidate = default;
                return false;
            }

            ref HeuristicGroupBSequenceState bestState = ref groupBStates[bestIndex];
            var result = new HeuristicDivisorCandidate(bestState.Candidate, HeuristicDivisorGroup.GroupB, bestState.Ending, bestState.PriorityIndex, bestState.Residue);
            bestState.Advance();
            candidate = result;
            return true;
        }
    }

    internal struct HeuristicGroupBSequenceState(ushort residue, byte ending, byte priorityIndex)
	{
        public ulong Candidate = residue;
        public byte PriorityIndex = priorityIndex;
        public byte Ending = ending;
        public ushort Residue = residue;

		public void TryNormalize()
        {
            ulong candidate = Candidate;
            if (candidate == ulong.MaxValue)
            {
                return;
            }

            while (candidate <= 13UL)
            {
                ulong next = candidate + Wheel210;
                if (next <= candidate)
                {
                    Candidate = ulong.MaxValue;
                    return;
                }

                candidate = next;
            }

            Candidate = candidate;
        }

        public void Advance()
        {
            ulong candidate = Candidate;
            if (candidate == ulong.MaxValue)
            {
                return;
            }

            ulong next = candidate + Wheel210;
            Candidate = next > candidate ? next : ulong.MaxValue;
        }


    }

    internal ref struct MersenneHeuristicDivisorEnumerator
    {
        private readonly GpuUInt128 step;
        private readonly GpuUInt128 limit;
        private GpuUInt128 current;
        private MersenneDivisorResidueStepper residueStepper;
        private bool active;
        private ulong processedCount;
        private ulong lastDivisor;

        public MersenneHeuristicDivisorEnumerator(ulong exponent, ulong maxDivisor)
        {
            var stepLocal = new GpuUInt128(exponent);
            stepLocal.ShiftLeft(1);
            step = stepLocal;

            var limitLocal = new GpuUInt128(maxDivisor);
            limit = limitLocal;

            var firstDivisor = stepLocal;
            firstDivisor.Add(1UL);

            bool hasCandidates = !stepLocal.IsZero && firstDivisor.CompareTo(limitLocal) <= 0;

            current = hasCandidates ? firstDivisor : GpuUInt128.Zero;
            residueStepper = hasCandidates ? new MersenneDivisorResidueStepper(exponent, stepLocal, firstDivisor) : default;
            active = hasCandidates;
            processedCount = 0UL;
            lastDivisor = 0UL;
        }

        public bool TryGetNext(out HeuristicDivisorCandidate candidate)
        {
            while (active)
            {
                ulong value = current.Low;
                processedCount++;
                lastDivisor = value;

                bool admissible = residueStepper.IsAdmissible();

                Advance();

                if (admissible)
                {
                    candidate = CreateCandidate(value);
                    return true;
                }
            }

            candidate = default;
            return false;
        }

        private void Advance()
        {
            GpuUInt128 next = current;
            next.Add(step);

            if (next.CompareTo(limit) > 0 || next.CompareTo(current) <= 0)
            {
                active = false;
                return;
            }

            current = next;
            residueStepper.Advance();
        }

        public ulong ProcessedCount => processedCount;

        public ulong LastDivisor => lastDivisor;

        public bool Exhausted => !active;

        private static HeuristicDivisorCandidate CreateCandidate(ulong value)
        {
            byte ending = (byte)(value % 10UL);
            HeuristicDivisorGroup group;

            if (value == 3UL || value == 7UL || value == 11UL || value == 13UL)
            {
                group = HeuristicDivisorGroup.GroupAConstant;
            }
            else if (ending == 3)
            {
                group = HeuristicDivisorGroup.GroupAWheel;
            }
            else
            {
                group = HeuristicDivisorGroup.GroupB;
            }

            ushort residue = (ushort)(value % Wheel210);
            return new HeuristicDivisorCandidate(value, group, ending, 0, residue);
        }
    }


    internal static bool IsPrimeInternal(ulong n, CancellationToken ct)
    {
        if (!EnableHeuristicPrimeTesting)
        {
            return LegacyIsPrimeInternal(n, ct);
        }

        return HeuristicIsPrimeCpu(n);
    }

    public static bool LegacyIsPrimeInternal(ulong n, CancellationToken ct)
    {
        // EvenPerfectBitScanner streams monotonically increasing odd exponents that start at 136,279,841 and already exclude
        // multiples of five. Factorization helpers (PrimeOrderCalculator, Pollard routines, and divisor-cycle warmups) reuse
        // this path for arbitrary residues and cofactors, so keep the boolean guards even though the scanner never trips them.
        bool isTwo = n == 2UL;
        bool isOdd = (n & 1UL) != 0UL;
        // TODO: Replace this modulo check with ULongExtensions.Mod5 so the CPU hot path reuses the benchmarked helper instead of `%`.
        ulong mod5 = n % 5UL;
        bool divisibleByFive = n > 5UL && mod5 == 0UL;

        bool result = n >= 2UL && (isTwo || isOdd) && !divisibleByFive;
        bool requiresTrialDivision = result && n >= 7UL && !isTwo;

        if (requiresTrialDivision)
        {
            bool sharesMaxExponentFactor = n.Mod10() == 1UL && SharesFactorWithMaxExponent(n);
            result &= !sharesMaxExponentFactor;

            if (result)
            {
                var smallPrimeDivisorsLength = PrimesGenerator.SmallPrimes.Length;
                uint[] smallPrimeDivisors = PrimesGenerator.SmallPrimes;
                ulong[] smallPrimeDivisorsMul = PrimesGenerator.SmallPrimesPow2;
                for (int i = 0; i < smallPrimeDivisorsLength; i++)
                {
                    if (smallPrimeDivisorsMul[i] > n)
                    {
                        break;
                    }

                    // TODO: Route this small-prime filtering through the shared divisor-cycle cache once
                    // PrimeTester can consult it directly; the cached cycles avoid the repeated `%` work
                    // that slows these hot loops when sieving hundreds of millions of candidates.
                    if (n % smallPrimeDivisors[i] == 0)
                    {
                        result = false;
                        break;
                    }
                }
            }
        }

        return result;
    }

    public static bool EnableHeuristicPrimeTesting { get; set; } = true;

    public static int GpuBatchSize { get; set; } = 262_144;

    public static int HeuristicGpuDivisorBatchSize { get; set; } = 4_096;


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void IsPrimeBatchGpu(ReadOnlySpan<ulong> values, Span<byte> results)
    {
        // Limit concurrency and declare variables outside loops for performance and reuse
        var limiter = GpuPrimeWorkLimiter.Acquire();
        var gpu = PrimeTesterGpuContextPool.Rent();

        try
        {
            var accelerator = gpu.Accelerator;

            // Get per-accelerator cached kernel and device primes buffer.
            var state = GpuKernelState.GetOrCreate(accelerator);
            int totalLength = values.Length;
            int batchSize = Math.Max(1, GpuBatchSize);

            lock (gpu.ExecutionLock)
            {
                var scratch = state.RentScratch(batchSize, accelerator);
                var input = scratch.Input;
                var output = scratch.Output;
                ulong[] temp = ArrayPool<ulong>.Shared.Rent(batchSize);
                // TODO: Replace this ad-hoc ArrayPool buffer with the pinned span cache from
                // PrimeSieveGpuBenchmarks so batch uploads reuse preallocated GPU-friendly
                // memory and avoid the extra copy before every kernel launch.

                try
                {
                    int pos = 0;
                    while (pos < totalLength)
                    {
                        int remaining = totalLength - pos;
                        int count = remaining > batchSize ? batchSize : remaining;

                        values.Slice(pos, count).CopyTo(temp);
						input.View.CopyFromCPU(ref temp[0], count);

                        state.Kernel(count, input.View, state.DevicePrimes.View, output.View);
                        accelerator.Synchronize();
                        output.View.CopyToCPU(ref results[pos], count);

                        pos += count;
                    }
                }
                finally
                {
                    ArrayPool<ulong>.Shared.Return(temp);
                    state.ReturnScratch(scratch);
                }
            }
        }
        finally
        {
            gpu.Dispose();
            limiter.Dispose();
        }
    }

    private static class PrimeTesterGpuContextPool
    {
        internal sealed class PooledContext : IDisposable
        {
            private bool _disposed;

            public Context Context { get; }

            public Accelerator Accelerator { get; }

            public object ExecutionLock { get; } = new();

            public PooledContext()
            {
                Context = Context.CreateDefault();
                Accelerator = Context.GetPreferredDevice(false).CreateAccelerator(Context);
            }

            public void Dispose()
            {
                if (_disposed)
                {
                    return;
                }

                PrimeTester.ClearGpuCaches(Accelerator);
                Accelerator.Dispose();
                Context.Dispose();
                _disposed = true;
            }
        }

        private static readonly ConcurrentQueue<PooledContext> Pool = new();
        private static readonly object CreationLock = new();

        internal static PrimeTesterGpuContextLease Rent()
        {
            if (Pool.TryDequeue(out var ctx))
            {
                return new PrimeTesterGpuContextLease(ctx);
            }

            lock (CreationLock)
            {
                return new PrimeTesterGpuContextLease(new PooledContext());
            }
        }

        internal static void DisposeAll()
        {
            while (Pool.TryDequeue(out var ctx))
            {
                ctx.Dispose();
            }
        }

        private static void Return(PooledContext ctx)
        {
            ctx.Accelerator.Synchronize();
            Pool.Enqueue(ctx);
        }

        internal struct PrimeTesterGpuContextLease : IDisposable
        {
            private PooledContext? _ctx;

            internal PrimeTesterGpuContextLease(PooledContext ctx)
            {
                _ctx = ctx;
            }

            public Accelerator Accelerator => _ctx!.Accelerator;

            public object ExecutionLock => _ctx!.ExecutionLock;

            public void Dispose()
            {
                if (_ctx is { } ctx)
                {
                    Return(ctx);
                    _ctx = null;
                }
            }
        }
    }

    // Per-accelerator GPU state for prime sieve (kernel + uploaded primes).
    private sealed class KernelState
    {
        public Action<Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<byte>> Kernel { get; }
        public Action<Index1D, ArrayView<ulong>, ulong, ArrayView<byte>> HeuristicTrialDivisionKernel { get; }
        public MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimes { get; }
        private readonly Accelerator _accel;
        private readonly System.Collections.Concurrent.ConcurrentBag<ScratchBuffers> _scratchPool = [];
        // TODO: Replace this ConcurrentBag with the lock-free ring buffer variant validated in
        // GpuModularArithmeticBenchmarks so renting scratch buffers stops contending on the bag's internal locks when
        // thousands of GPU batches execute per second.

        public KernelState(Accelerator accelerator)
        {
            _accel = accelerator;
            // Compile once per accelerator and upload primes once.
            Kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<byte>>(PrimeTesterKernels.SmallPrimeSieveKernel);
            HeuristicTrialDivisionKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ulong, ArrayView<byte>>(PrimeTesterKernels.HeuristicTrialDivisionKernel);

            var primes = PrimesGenerator.SmallPrimes;
            DevicePrimes = accelerator.Allocate1D<uint>(primes.Length);
            DevicePrimes.View.CopyFromCPU(primes);
        }

        internal sealed class ScratchBuffers : IDisposable
        {
            public MemoryBuffer1D<ulong, Stride1D.Dense> Input { get; private set; }
            public MemoryBuffer1D<byte, Stride1D.Dense> Output { get; private set; }
            public int Capacity { get; private set; }

            public ScratchBuffers(Accelerator accel, int capacity)
            {
                Capacity = Math.Max(1, capacity);
                Input = accel.Allocate1D<ulong>(Capacity);
                Output = accel.Allocate1D<byte>(Capacity);
            }

            public void Dispose()
            {
                Output.Dispose();
                Input.Dispose();
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ScratchBuffers RentScratch(int minCapacity, Accelerator accel)
        {
            minCapacity = Math.Max(1, minCapacity);
            while (_scratchPool.TryTake(out var sb))
            {
                if (sb.Capacity >= minCapacity)
                {
                    return sb;
                }

                sb.Dispose();
            }

            return new ScratchBuffers(accel, minCapacity);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void ReturnScratch(ScratchBuffers scratch)
        {
            _scratchPool.Add(scratch);
        }

        public void Clear()
        {
            while (_scratchPool.TryTake(out var sb))
            {
                sb.Dispose();
            }

            DevicePrimes.Dispose();
        }
    }

    private static class GpuKernelState
    {
        // Map accelerator to cached state; use Lazy to serialize kernel creation
        private static readonly System.Collections.Concurrent.ConcurrentDictionary<Accelerator, Lazy<KernelState>> States = new();
        // TODO: Prewarm this per-accelerator cache during startup (and reuse a simple array keyed by accelerator index)
        // once the kernel pool exposes deterministic ordering; the Lazy wrappers showed measurable overhead in the
        // GpuModularArithmeticBenchmarks hot path.

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static KernelState GetOrCreate(Accelerator accelerator)
        {
            var lazy = States.GetOrAdd(
                accelerator,
                acc => new Lazy<KernelState>(() => new KernelState(acc), System.Threading.LazyThreadSafetyMode.ExecutionAndPublication));
            return lazy.Value;
        }

        public static void Clear(Accelerator accelerator)
        {
            if (States.TryRemove(accelerator, out var lazy))
            {
                if (lazy.IsValueCreated)
                {
                    var state = lazy.Value;
                    state.Clear();
                }
            }
        }
    }

    // Expose cache clearing for accelerator disposal coordination
    public static void ClearGpuCaches(Accelerator accelerator)
    {
        GpuKernelState.Clear(accelerator);
    }

    internal static void DisposeGpuContexts()
    {
        PrimeTesterGpuContextPool.DisposeAll();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool SharesFactorWithMaxExponent(ulong n)
    {
        // TODO: Replace this on-the-fly GCD probe with the cached factor table derived from
        // ResidueComputationBenchmarks so divisor-cycle metadata can short-circuit the test
        // instead of recomputing binary GCD for every candidate.
        ulong m = (ulong)BitOperations.Log2(n);
        return BinaryGcd(n, m) != 1UL;
    }

    internal static void SharesFactorWithMaxExponentBatch(ReadOnlySpan<ulong> values, Span<byte> results)
    {
        // TODO: Route this batch helper through the shared GPU kernel pool from
        // GpuUInt128BinaryGcdBenchmarks so we reuse cached kernels, pinned host buffers,
        // and divisor-cycle staging instead of allocating new device buffers per call.
        var gpu = PrimeTesterGpuContextPool.Rent();
        var accelerator = gpu.Accelerator;
        var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ArrayView<byte>>(PrimeTesterKernels.SharesFactorKernel);

        int length = values.Length;
        var inputBuffer = accelerator.Allocate1D<ulong>(length);
        var resultBuffer = accelerator.Allocate1D<byte>(length);

        ulong[] temp = ArrayPool<ulong>.Shared.Rent(length);
        values.CopyTo(temp);
        inputBuffer.View.CopyFromCPU(ref temp[0], length);
        kernel(length, inputBuffer.View, resultBuffer.View);
        accelerator.Synchronize();
        resultBuffer.View.CopyToCPU(ref results[0], length);
        ArrayPool<ulong>.Shared.Return(temp);
        resultBuffer.Dispose();
        inputBuffer.Dispose();
        gpu.Dispose();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong BinaryGcd(ulong u, ulong v)
    {
        // TODO: Swap this handwritten binary GCD for the optimized helper measured in
        // GpuUInt128BinaryGcdBenchmarks so CPU callers share the faster subtract-less
        // ladder once the common implementation is promoted into PerfectNumbers.Core.
        if (u == 0UL)
        {
            return v;
        }

        if (v == 0UL)
        {
            return u;
        }

        int shift = BitOperations.TrailingZeroCount(u | v);
        u >>= BitOperations.TrailingZeroCount(u);

        do
        {
            v >>= BitOperations.TrailingZeroCount(v);
            if (u > v)
            {
                (u, v) = (v, u);
            }

            v -= u;
        }
        while (v != 0UL);

        return u << shift;
    }
    }
