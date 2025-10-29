using System;
using System.Buffers;
using System.Collections.Concurrent;
using Open.Numeric.Primes;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core;

public sealed class HeuristicPrimeTester
{
    static HeuristicPrimeTester()
    {
        HeuristicPrimeSieves.EnsureInitialized();
    }

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

    private static readonly bool UseHeuristicGroupBTrialDivision = false; // Temporary fallback gate for Group B.


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


    public static int HeuristicGpuDivisorBatchSize { get; set; } = 4_096;

    public static int HeuristicDivisorInterleaveBatchSize { get; set; } = 64;

    [ThreadStatic]
    private static HeuristicPrimeTester? _tester;

    public static HeuristicPrimeTester Exclusive => _tester ??= new();

    public bool IsPrimeCpu(ulong n, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        byte nMod10 = (byte)n.Mod10();
        ulong sqrtLimit = ComputeHeuristicSqrt(n);

        if (sqrtLimit < 3UL)
        {
            return EvaluateWithOpenNumericFallback(n);
        }

        bool includeGroupB = UseHeuristicGroupBTrialDivision;
        return HeuristicTrialDivisionCpu(n, sqrtLimit, nMod10, includeGroupB);
    }

    public bool IsPrimeCpu(ulong n)
    {
        return IsPrimeCpu(n, CancellationToken.None);
    }

    public bool IsPrimeGpu(ulong n)
    {
        byte nMod10 = (byte)n.Mod10();
        ulong sqrtLimit = ComputeHeuristicSqrt(n);
        return IsPrimeGpu(n, sqrtLimit, nMod10);
    }

    public bool IsPrimeGpu(ulong n, ulong sqrtLimit, byte nMod10)
    {
        return HeuristicIsPrimeGpuCore(n, sqrtLimit, nMod10);
    }

    private bool HeuristicIsPrimeGpuCore(ulong n, ulong sqrtLimit, byte nMod10)
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

        bool compositeDetected = HeuristicTrialDivisionGpuDetectsDivisor(n, sqrtLimit, nMod10);
        return !compositeDetected;
    }

    private static bool HeuristicTrialDivisionCpu(ulong n, ulong sqrtLimit, byte nMod10, bool includeGroupB)
    {
        ReadOnlySpan<int> groupADivisors = HeuristicPrimeSieves.GroupADivisors;
        int interleaveBatchSize = Math.Max(1, HeuristicDivisorInterleaveBatchSize);
        int groupAIndex = 0;

        if (!includeGroupB)
        {
            while (groupAIndex < groupADivisors.Length)
            {
                int processed = 0;
                while (processed < interleaveBatchSize && groupAIndex < groupADivisors.Length)
                {
                    ulong divisor = (ulong)groupADivisors[groupAIndex];
                    if (divisor > sqrtLimit)
                    {
                        groupAIndex = groupADivisors.Length;
                        break;
                    }

                    if (n % divisor == 0UL)
                    {
                        return false;
                    }

                    groupAIndex++;
                    processed++;
                }

                if (groupAIndex >= groupADivisors.Length)
                {
                    break;
                }

                if ((ulong)groupADivisors[groupAIndex] > sqrtLimit)
                {
                    groupAIndex = groupADivisors.Length;
                    break;
                }
            }

            return EvaluateWithOpenNumericFallback(n);
        }

        ReadOnlySpan<byte> endingOrder = GetGroupBEndingOrder(nMod10);
        if (endingOrder.IsEmpty)
        {
            return EvaluateWithOpenNumericFallback(n);
        }

        Span<int> indices = endingOrder.Length <= 8
            ? stackalloc int[endingOrder.Length]
            : new int[endingOrder.Length];

        bool groupAHasMore = groupAIndex < groupADivisors.Length && (ulong)groupADivisors[groupAIndex] <= sqrtLimit;
        bool groupBHasMore = HasGroupBCandidates(endingOrder, indices, sqrtLimit);

        while (groupAHasMore || groupBHasMore)
        {
            if (groupAHasMore)
            {
                int processed = 0;
                while (processed < interleaveBatchSize && groupAIndex < groupADivisors.Length)
                {
                    ulong divisor = (ulong)groupADivisors[groupAIndex];
                    if (divisor > sqrtLimit)
                    {
                        groupAIndex = groupADivisors.Length;
                        groupAHasMore = false;
                        break;
                    }

                    if (n % divisor == 0UL)
                    {
                        return false;
                    }

                    groupAIndex++;
                    processed++;
                }

                if (groupAIndex >= groupADivisors.Length)
                {
                    groupAHasMore = false;
                }
                else
                {
                    groupAHasMore = (ulong)groupADivisors[groupAIndex] <= sqrtLimit;
                    if (!groupAHasMore)
                    {
                        groupAIndex = groupADivisors.Length;
                    }
                }
            }

            if (groupBHasMore)
            {
                int processed = 0;
                while (processed < interleaveBatchSize)
                {
                    if (!TrySelectNextGroupBDivisor(endingOrder, indices, sqrtLimit, out ulong divisor))
                    {
                        groupBHasMore = false;
                        break;
                    }

                    if (n % divisor == 0UL)
                    {
                        return false;
                    }

                    processed++;
                }

                if (groupBHasMore)
                {
                    groupBHasMore = HasGroupBCandidates(endingOrder, indices, sqrtLimit);
                }
            }
        }

        return true;
    }




    private bool HeuristicTrialDivisionGpuDetectsDivisor(ulong n, ulong sqrtLimit, byte nMod10)
    {
        int batchCapacity = Math.Max(1, HeuristicGpuDivisorBatchSize);
        int interleaveBatchSize = Math.Max(1, HeuristicDivisorInterleaveBatchSize);
        var divisorPool = ThreadStaticPools.UlongPool;
        var hitPool = ThreadStaticPools.BytePool;

        ulong[] divisorArray = divisorPool.Rent(batchCapacity);
        byte[] hitFlags = hitPool.Rent(batchCapacity);

        var limiter = GpuPrimeWorkLimiter.Acquire();
        var gpu = PrimeTester.PrimeTesterGpuContextPool.Rent();
        var accelerator = gpu.Accelerator;
        var state = PrimeTester.GpuKernelState.GetOrCreate(accelerator);
        var scratch = state.RentScratch(batchCapacity, accelerator);

        bool compositeDetected = false;
        int count = 0;

        lock (gpu.ExecutionLock)
        {
            var input = scratch.Input;
            var output = scratch.Output;

            bool ProcessBatch(int length)
            {
                if (length <= 0)
                {
                    return false;
                }

                input.View.CopyFromCPU(ref divisorArray[0], length);
                state.HeuristicTrialDivisionKernel(length, input.View, n, output.View);
                accelerator.Synchronize();
                output.View.CopyToCPU(ref hitFlags[0], length);

                for (int i = 0; i < length; i++)
                {
                    if (hitFlags[i] == 0)
                    {
                        continue;
                    }

                    ulong divisor = divisorArray[i];
                    if (divisor > 1UL && n % divisor == 0UL)
                    {
                        return true;
                    }
                }

                return false;
            }

            ReadOnlySpan<int> groupADivisors = HeuristicPrimeSieves.GroupADivisors;
            int groupAIndex = 0;
            ReadOnlySpan<byte> endingOrder = GetGroupBEndingOrder(nMod10);
            Span<int> indices = endingOrder.Length <= 8
                ? stackalloc int[endingOrder.Length]
                : new int[endingOrder.Length];

            bool groupAHasMore = groupAIndex < groupADivisors.Length && (ulong)groupADivisors[groupAIndex] <= sqrtLimit;
            bool groupBHasMore = !endingOrder.IsEmpty && HasGroupBCandidates(endingOrder, indices, sqrtLimit);

            while (!compositeDetected && (groupAHasMore || groupBHasMore))
            {
                if (groupAHasMore)
                {
                    int processed = 0;
                    while (processed < interleaveBatchSize && groupAIndex < groupADivisors.Length)
                    {
                        ulong divisor = (ulong)groupADivisors[groupAIndex];
                        if (divisor > sqrtLimit)
                        {
                            groupAIndex = groupADivisors.Length;
                            groupAHasMore = false;
                            break;
                        }

                        divisorArray[count] = divisor;
                        count++;

                        if (count == batchCapacity)
                        {
                            if (ProcessBatch(count))
                            {
                                compositeDetected = true;
                                goto Cleanup;
                            }

                            count = 0;
                        }

                        groupAIndex++;
                        processed++;
                    }

                    if (groupAIndex >= groupADivisors.Length)
                    {
                        groupAHasMore = false;
                    }
                    else
                    {
                        groupAHasMore = (ulong)groupADivisors[groupAIndex] <= sqrtLimit;
                        if (!groupAHasMore)
                        {
                            groupAIndex = groupADivisors.Length;
                        }
                    }
                }

                if (!compositeDetected && groupBHasMore)
                {
                    int processed = 0;
                    while (processed < interleaveBatchSize)
                    {
                        if (!TrySelectNextGroupBDivisor(endingOrder, indices, sqrtLimit, out ulong divisor))
                        {
                            groupBHasMore = false;
                            break;
                        }

                        divisorArray[count] = divisor;
                        count++;

                        if (count == batchCapacity)
                        {
                            if (ProcessBatch(count))
                            {
                                compositeDetected = true;
                                goto Cleanup;
                            }

                            count = 0;
                        }

                        processed++;
                    }

                    if (groupBHasMore)
                    {
                        groupBHasMore = HasGroupBCandidates(endingOrder, indices, sqrtLimit);
                    }
                }
            }

            if (!compositeDetected && count > 0)
            {
                if (ProcessBatch(count))
                {
                    compositeDetected = true;
                    goto Cleanup;
                }

                count = 0;
            }
        }

Cleanup:
        state.ReturnScratch(scratch);
        divisorPool.Return(divisorArray);
        hitPool.Return(hitFlags);
        gpu.Dispose();
        limiter.Dispose();
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


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool TrySelectNextGroupBDivisor(ReadOnlySpan<byte> endingOrder, Span<int> indices, ulong sqrtLimit, out ulong divisor)
    {
        ulong bestCandidate = ulong.MaxValue;
        int bestEndingIndex = -1;

        for (int i = 0; i < endingOrder.Length; i++)
        {
            ReadOnlySpan<uint> divisors = GetGroupBDivisors(endingOrder[i]);
            int index = indices[i];
            if ((uint)index >= (uint)divisors.Length)
            {
                continue;
            }

            ulong candidate = (ulong)divisors[index];
            if (candidate > sqrtLimit)
            {
                indices[i] = divisors.Length;
                continue;
            }

            if (bestEndingIndex == -1 || candidate < bestCandidate)
            {
                bestCandidate = candidate;
                bestEndingIndex = i;
            }
        }

        if (bestEndingIndex == -1)
        {
            divisor = 0UL;
            return false;
        }

        divisor = bestCandidate;
        indices[bestEndingIndex]++;
        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool HasGroupBCandidates(ReadOnlySpan<byte> endingOrder, Span<int> indices, ulong sqrtLimit)
    {
        for (int i = 0; i < endingOrder.Length; i++)
        {
            ReadOnlySpan<uint> divisors = GetGroupBDivisors(endingOrder[i]);
            int index = indices[i];
            if ((uint)index >= (uint)divisors.Length)
            {
                continue;
            }

            ulong candidate = (ulong)divisors[index];
            if (candidate > sqrtLimit)
            {
                indices[i] = divisors.Length;
                continue;
            }

            return true;
        }

        return false;
    }

    private static ReadOnlySpan<uint> GetGroupBDivisors(byte ending) => ending switch
    {
        1 => DivisorGenerator.SmallPrimesLastOneWithoutLastThree,
        3 => ReadOnlySpan<uint>.Empty,
        7 => DivisorGenerator.SmallPrimesLastSevenWithoutLastThree,
        9 => DivisorGenerator.SmallPrimesLastNineWithoutLastThree,
        _ => ReadOnlySpan<uint>.Empty,
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
}
