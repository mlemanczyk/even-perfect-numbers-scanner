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

public sealed class HeuristicCombinedPrimeTester
{
    static HeuristicCombinedPrimeTester()
    {
        HeuristicPrimeSieves.EnsureInitialized();
    }

    private const ulong Wheel210 = 210UL;

    private static readonly bool UseHeuristicGroupBTrialDivision = false; // Temporary fallback gate for Group B.

    private static readonly Lazy<uint[]> CombinedDivisorsEnding1 = new(() => BuildCombinedDivisors(1), LazyThreadSafetyMode.ExecutionAndPublication);
    private static readonly Lazy<uint[]> CombinedDivisorsEnding3 = new(() => BuildCombinedDivisors(3), LazyThreadSafetyMode.ExecutionAndPublication);
    private static readonly Lazy<uint[]> CombinedDivisorsEnding7 = new(() => BuildCombinedDivisors(7), LazyThreadSafetyMode.ExecutionAndPublication);
    private static readonly Lazy<uint[]> CombinedDivisorsEnding9 = new(() => BuildCombinedDivisors(9), LazyThreadSafetyMode.ExecutionAndPublication);

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
    private static HeuristicCombinedPrimeTester? _tester;

    public static HeuristicCombinedPrimeTester Exclusive => _tester ??= new();

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
        ReadOnlySpan<uint> combinedDivisors = GetCombinedDivisors(nMod10);
        if (combinedDivisors.IsEmpty)
        {
            return EvaluateWithOpenNumericFallback(n);
        }

        int targetGroupACount = CountGroupAEntriesUpTo(sqrtLimit);
        int targetGroupBCount = includeGroupB ? CountGroupBEntriesUpTo(sqrtLimit, nMod10) : 0;

        if (targetGroupACount == 0 && (!includeGroupB || targetGroupBCount == 0))
        {
            return EvaluateWithOpenNumericFallback(n);
        }

        int processedA = 0;
        int processedB = 0;

        for (int i = 0; i < combinedDivisors.Length; i++)
        {
            uint entry = combinedDivisors[i];
            HeuristicDivisorGroup group = ResolveGroup(entry);

            if (group == HeuristicDivisorGroup.GroupB)
            {
                if (!includeGroupB)
                {
                    continue;
                }

                if (processedB >= targetGroupBCount)
                {
                    if (processedA >= targetGroupACount)
                    {
                        break;
                    }

                    continue;
                }

                processedB++;
            }
            else
            {
                if (processedA >= targetGroupACount)
                {
                    if (!includeGroupB || processedB >= targetGroupBCount)
                    {
                        break;
                    }

                    continue;
                }

                processedA++;
            }

            ulong divisor = entry;
            if (divisor > sqrtLimit)
            {
                continue;
            }

            if (n % divisor == 0UL)
            {
                return false;
            }

            if (processedA >= targetGroupACount && (!includeGroupB || processedB >= targetGroupBCount))
            {
                break;
            }
        }

        return EvaluateWithOpenNumericFallback(n);
    }

    private bool HeuristicTrialDivisionGpuDetectsDivisor(ulong n, ulong sqrtLimit, byte nMod10)
    {
        int batchCapacity = Math.Max(1, HeuristicGpuDivisorBatchSize);
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

            ReadOnlySpan<uint> combinedDivisors = GetCombinedDivisors(nMod10);
            if (combinedDivisors.IsEmpty)
            {
                compositeDetected = !EvaluateWithOpenNumericFallback(n);
                goto Cleanup;
            }

            int targetGroupACount = CountGroupAEntriesUpTo(sqrtLimit);
            int targetGroupBCount = CountGroupBEntriesUpTo(sqrtLimit, nMod10);
            bool includeGroupB = targetGroupBCount > 0;

            if (targetGroupACount == 0 && !includeGroupB)
            {
                compositeDetected = !EvaluateWithOpenNumericFallback(n);
                goto Cleanup;
            }

            int processedA = 0;
            int processedB = 0;

            for (int i = 0; i < combinedDivisors.Length; i++)
            {
                uint entry = combinedDivisors[i];
                HeuristicDivisorGroup group = ResolveGroup(entry);

                if (group == HeuristicDivisorGroup.GroupB)
                {
                    if (!includeGroupB)
                    {
                        continue;
                    }

                    if (processedB >= targetGroupBCount)
                    {
                        if (processedA >= targetGroupACount)
                        {
                            break;
                        }

                        continue;
                    }

                    processedB++;
                }
                else
                {
                    if (processedA >= targetGroupACount)
                    {
                        if (!includeGroupB || processedB >= targetGroupBCount)
                        {
                            break;
                        }

                        continue;
                    }

                    processedA++;
                }

                ulong divisor = entry;
                if (divisor > sqrtLimit)
                {
                    continue;
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

                if (processedA >= targetGroupACount && (!includeGroupB || processedB >= targetGroupBCount))
                {
                    break;
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

    private static ReadOnlySpan<uint> GetCombinedDivisors(byte nMod10) => nMod10 switch
    {
        1 => CombinedDivisorsEnding1.Value,
        3 => CombinedDivisorsEnding3.Value,
        7 => CombinedDivisorsEnding7.Value,
        9 => CombinedDivisorsEnding9.Value,
        _ => ReadOnlySpan<uint>.Empty,
    };

    private static uint[] BuildCombinedDivisors(byte nMod10)
    {
        ReadOnlySpan<int> groupA = HeuristicPrimeSieves.GroupADivisors;
        ReadOnlySpan<uint> groupB = GetGroupBDivisors(nMod10);

        var combined = new List<uint>(groupA.Length + groupB.Length);

        int groupAIndex = 0;
        while (groupAIndex < groupA.Length && groupA[groupAIndex] <= 19)
        {
            combined.Add((uint)groupA[groupAIndex]);
            groupAIndex++;
        }

        int groupBIndex = GetGroupBStartIndex(groupB);

        while (groupAIndex < groupA.Length || groupBIndex < groupB.Length)
        {
            int addedA = 0;
            while (addedA < 2 && groupAIndex < groupA.Length)
            {
                combined.Add((uint)groupA[groupAIndex]);
                groupAIndex++;
                addedA++;
            }

            if (groupBIndex < groupB.Length)
            {
                combined.Add(groupB[groupBIndex]);
                groupBIndex++;
            }
        }

        return combined.ToArray();
    }

    private static int CountGroupAEntriesUpTo(ulong sqrtLimit)
    {
        ReadOnlySpan<int> groupA = HeuristicPrimeSieves.GroupADivisors;
        int count = 0;

        while (count < groupA.Length && (ulong)groupA[count] <= sqrtLimit)
        {
            count++;
        }

        return count;
    }

    private static int CountGroupBEntriesUpTo(ulong sqrtLimit, byte nMod10)
    {
        ReadOnlySpan<uint> divisors = GetGroupBDivisors(nMod10);
        if (divisors.IsEmpty)
        {
            return 0;
        }

        int index = GetGroupBStartIndex(divisors);
        int count = 0;

        for (int i = index; i < divisors.Length; i++)
        {
            if ((ulong)divisors[i] > sqrtLimit)
            {
                break;
            }

            count++;
        }

        return count;
    }

    private static int GetGroupBStartIndex(ReadOnlySpan<uint> divisors)
    {
        int index = 0;
        while (index < divisors.Length && IsGroupAPrefixValue(divisors[index]))
        {
            index++;
        }

        return index;
    }

    private static bool IsGroupAPrefixValue(uint value) => value is 3U or 7U or 11U or 13U;

    private static HeuristicDivisorGroup ResolveGroup(uint value) => value switch
    {
        3U or 7U or 11U or 13U => HeuristicDivisorGroup.GroupAConstant,
        _ when value % 10U == 3U => HeuristicDivisorGroup.GroupAWheel,
        _ => HeuristicDivisorGroup.GroupB,
    };

    private static ReadOnlySpan<uint> GetGroupBDivisors(byte ending) => ending switch
    {
        1 => DivisorGenerator.SmallPrimesLastOneWithoutLastThree,
        3 => ReadOnlySpan<uint>.Empty,
        7 => DivisorGenerator.SmallPrimesLastSevenWithoutLastThree,
        9 => DivisorGenerator.SmallPrimesLastNineWithoutLastThree,
        _ => ReadOnlySpan<uint>.Empty,
    };

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static HeuristicDivisorEnumerator CreateHeuristicDivisorEnumerator(ulong sqrtLimit, byte nMod10, Span<HeuristicGroupBSequenceState> groupBBuffer)
    {
        return new HeuristicDivisorEnumerator(sqrtLimit, nMod10, groupBBuffer);
    }

    internal struct HeuristicGroupBSequenceState
    {
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
        private readonly ReadOnlySpan<uint> combinedDivisors;
        private readonly bool includeGroupB;
        private readonly int targetGroupACount;
        private readonly int targetGroupBCount;
        private int index;
        private int processedA;
        private int processedB;

        public HeuristicDivisorEnumerator(ulong sqrtLimit, byte nMod10, Span<HeuristicGroupBSequenceState> groupBBuffer)
        {
            this.sqrtLimit = sqrtLimit;
            _ = groupBBuffer;
            combinedDivisors = GetCombinedDivisors(nMod10);
            targetGroupACount = CountGroupAEntriesUpTo(sqrtLimit);
            targetGroupBCount = CountGroupBEntriesUpTo(sqrtLimit, nMod10);
            includeGroupB = targetGroupBCount > 0;
            index = 0;
            processedA = 0;
            processedB = 0;
        }

        public bool TryGetNext(out HeuristicDivisorCandidate candidate)
        {
            while (index < combinedDivisors.Length)
            {
                uint entry = combinedDivisors[index++];
                HeuristicDivisorGroup group = ResolveGroup(entry);

                if (group == HeuristicDivisorGroup.GroupB)
                {
                    if (!includeGroupB)
                    {
                        continue;
                    }

                    if (processedB >= targetGroupBCount)
                    {
                        if (processedA >= targetGroupACount)
                        {
                            break;
                        }

                        continue;
                    }

                    processedB++;
                }
                else
                {
                    if (processedA >= targetGroupACount)
                    {
                        if (!includeGroupB || processedB >= targetGroupBCount)
                        {
                            break;
                        }

                        continue;
                    }

                    processedA++;
                }

                ulong value = entry;
                if (value > sqrtLimit)
                {
                    continue;
                }

                byte ending = (byte)(value % 10UL);
                ushort residue = (ushort)(value % Wheel210);
                candidate = new HeuristicDivisorCandidate(value, group, ending, 0, residue);
                return true;
            }

            candidate = default;
            return false;
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
