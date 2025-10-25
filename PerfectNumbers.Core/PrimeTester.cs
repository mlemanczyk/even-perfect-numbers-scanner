using System;
using System.Buffers;
using System.Collections.Concurrent;
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
    private static readonly ulong[] GroupAConstantDivisors = { 3UL, 7UL, 11UL, 13UL };
    private static readonly byte[] GroupAIncrementPattern = { 20, 10 };
    private static readonly byte[] GroupBEndingOrderMod1 = { 9, 1 };
    private static readonly byte[] GroupBEndingOrderMod3 = { 9, 7 };
    private static readonly byte[] GroupBEndingOrderMod7 = { 7, 1 };
    private static readonly byte[] GroupBEndingOrderMod9 = { 9, 7, 1 };
    private static readonly ushort[] Wheel210ResiduesEnding1 = { 1, 11, 31, 41, 61, 71, 101, 121, 131, 151, 181, 191 };
    private static readonly ushort[] Wheel210ResiduesEnding7 = { 17, 37, 47, 67, 97, 107, 127, 137, 157, 167, 187, 197 };
    private static readonly ushort[] Wheel210ResiduesEnding9 = { 19, 29, 59, 79, 89, 109, 139, 149, 169, 179, 199, 209 };
    private const ulong Wheel210 = 210UL;
    internal const int MaxGroupBSequences = 36;

    private readonly bool _useLegacyPrimeTester = useInternal;

    private static readonly ulong[] HeuristicSmallCycleSnapshot = MersenneDivisorCycles.Shared.ExportSmallCyclesSnapshot();

    internal enum HeuristicDivisorGroup : byte
    {
        None = 0,
        GroupAConstant = 1,
        GroupAWheel = 2,
        GroupB = 3,
    }

    internal readonly struct HeuristicDivisorCandidate
    {
        public ulong Value { get; }

        public HeuristicDivisorGroup Group { get; }

        public byte Ending { get; }

        public byte PriorityIndex { get; }

        public ushort WheelResidue { get; }

        public HeuristicDivisorCandidate(ulong value, HeuristicDivisorGroup group, byte ending, byte priorityIndex, ushort wheelResidue)
        {
            Value = value;
            Group = group;
            Ending = ending;
            PriorityIndex = priorityIndex;
            WheelResidue = wheelResidue;
        }
    }

    internal readonly struct HeuristicDivisorPreparation
    {
        public HeuristicDivisorCandidate Candidate { get; }

        public MontgomeryDivisorData DivisorData { get; }

        public ulong CycleLengthHint { get; }

        public bool HasCycleLengthHint { get; }

        public bool RequiresCycleComputation => !HasCycleLengthHint;

        public HeuristicDivisorPreparation(
            in HeuristicDivisorCandidate candidate,
            in MontgomeryDivisorData divisorData,
            ulong cycleLengthHint,
            bool hasCycleLengthHint)
        {
            Candidate = candidate;
            DivisorData = divisorData;
            CycleLengthHint = cycleLengthHint;
            HasCycleLengthHint = hasCycleLengthHint;
        }
    }

    internal struct HeuristicTrialDivisionSummary
    {
        public ulong MaxGroupADivisor;

        public ulong MinGroupBDivisor;

        public ulong LastTestedDivisor;

        public ulong HitDivisor;

        public HeuristicDivisorGroup HitGroup;

        public HeuristicDivisorPreparation HitPreparation;

        public ulong HitCycleLength;

        public bool HitCycleFromHint;

        public bool HitCycleComputed;

        public bool HitPrimeOrderFailed;

        public bool HitMontgomeryIsUnity;

        public bool HitConfirmsMersenne;

        public uint TotalDivisorsTested;

        public byte MinGroupBEnding;

        public byte MinGroupBPriorityIndex;

        public byte HitEnding;

        public byte HitPriorityIndex;

        public bool SawGroupAConstant;

        public bool SawGroupAWheel;

        public bool SawGroupB;

        public bool HasHitPreparation;

        public bool Hit;
    }

    internal readonly struct HeuristicDivisorHitResolution
    {
        public HeuristicDivisorPreparation Preparation { get; }

        public ulong CycleLength { get; }

        public bool CycleFromHint { get; }

        public bool CycleComputed { get; }

        public bool PrimeOrderFailed { get; }

        public bool MontgomeryIsUnity { get; }

        public bool ConfirmsMersenne { get; }

        public HeuristicDivisorHitResolution(
            in HeuristicDivisorPreparation preparation,
            ulong cycleLength,
            bool cycleFromHint,
            bool cycleComputed,
            bool primeOrderFailed,
            bool montgomeryIsUnity,
            bool confirmsMersenne)
        {
            Preparation = preparation;
            CycleLength = cycleLength;
            CycleFromHint = cycleFromHint;
            CycleComputed = cycleComputed;
            PrimeOrderFailed = primeOrderFailed;
            MontgomeryIsUnity = montgomeryIsUnity;
            ConfirmsMersenne = confirmsMersenne;
        }
    }

    [Flags]
    public enum HeuristicGpuConfirmationMode : byte
    {
        None = 0,
        OnHit = 1,
        OnPrime = 2,
        Always = OnHit | OnPrime,
    }


    internal delegate void HeuristicDivisorCandidateCallback(in HeuristicDivisorCandidate candidate, in HeuristicTrialDivisionSummary summary);

    internal delegate void HeuristicDivisorHitCallback(in HeuristicDivisorPreparation preparation, in HeuristicTrialDivisionSummary summary);

    internal delegate void HeuristicTrialDivisionCompletedCallback(in HeuristicTrialDivisionSummary summary, bool isPrime);

    internal readonly struct HeuristicTrialDivisionCallbacks
    {
        public HeuristicDivisorCandidateCallback? OnCandidate { get; }

        public HeuristicDivisorHitCallback? OnHit { get; }

        public HeuristicTrialDivisionCompletedCallback? OnCompleted { get; }

        public HeuristicTrialDivisionCallbacks(
            HeuristicDivisorCandidateCallback? onCandidate,
            HeuristicDivisorHitCallback? onHit,
            HeuristicTrialDivisionCompletedCallback? onCompleted)
        {
            OnCandidate = onCandidate;
            OnHit = onHit;
            OnCompleted = onCompleted;
        }

        public bool RequiresSummary => OnCandidate is not null || OnHit is not null || OnCompleted is not null;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Candidate(in HeuristicDivisorCandidate candidate, in HeuristicTrialDivisionSummary summary)
        {
            OnCandidate?.Invoke(in candidate, in summary);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Hit(in HeuristicDivisorPreparation preparation, in HeuristicTrialDivisionSummary summary)
        {
            OnHit?.Invoke(in preparation, in summary);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Completed(in HeuristicTrialDivisionSummary summary, bool isPrime)
        {
            OnCompleted?.Invoke(in summary, isPrime);
        }

        public bool HasCandidate => OnCandidate is not null;

        public bool HasHit => OnHit is not null;

        public bool HasCompleted => OnCompleted is not null;
    }


    public bool IsPrime(ulong n, CancellationToken ct)
    {
        if (_useLegacyPrimeTester)
        {
            return LegacyIsPrimeInternal(n, ct);
        }

        return IsPrimeInternal(n, ct);
    }

    public static bool IsPrimeGpu(ulong n) => new PrimeTester().IsPrimeGpu(n, CancellationToken.None);

    public static bool HeuristicIsPrimeGpu(ulong n) => new PrimeTester().HeuristicIsPrimeGpu(n, CancellationToken.None);

    // Optional GPU-assisted primality: batched small-prime sieve on device.
    public bool IsPrimeGpu(ulong n, CancellationToken ct)
    {
        if (_useLegacyPrimeTester || !EnableHeuristicPrimeTesting)
        {
            return LegacyIsPrimeGpu(n, ct);
        }

        return HeuristicIsPrimeGpu(n, ct);
    }

    private static bool LegacyIsPrimeGpu(ulong n, CancellationToken ct)
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

        // Defensive fallback: GPU sieve produced a composite verdict or execution was skipped.
        // Confirm with the legacy CPU logic to prevent false negatives observed on some accelerators.
        return requiresCpuFallback ? LegacyIsPrimeInternal(n, ct) : true;
    }

    public bool HeuristicIsPrimeGpu(ulong n, CancellationToken ct)
    {
        byte nMod10 = (byte)n.Mod10();
        var summary = default(HeuristicTrialDivisionSummary);
        return HeuristicIsPrimeGpu(n, 0UL, nMod10, ct, ref summary, collectSummary: false, callbacks: default);
    }

    internal bool HeuristicIsPrimeGpu(ulong n, CancellationToken ct, in HeuristicTrialDivisionCallbacks callbacks)
    {
        byte nMod10 = (byte)n.Mod10();
        var summary = default(HeuristicTrialDivisionSummary);
        return HeuristicIsPrimeGpu(n, 0UL, nMod10, ct, ref summary, collectSummary: false, in callbacks);
    }

    internal bool HeuristicIsPrimeGpu(ulong n, CancellationToken ct, ref HeuristicTrialDivisionSummary summary)
    {
        byte nMod10 = (byte)n.Mod10();
        return HeuristicIsPrimeGpu(n, 0UL, nMod10, ct, ref summary, collectSummary: true, callbacks: default);
    }

    internal bool HeuristicIsPrimeGpu(ulong n, ulong sqrtLimit, byte nMod10, CancellationToken ct)
    {
        var summary = default(HeuristicTrialDivisionSummary);
        return HeuristicIsPrimeGpu(n, sqrtLimit, nMod10, ct, ref summary, collectSummary: false, callbacks: default);
    }

    internal bool HeuristicIsPrimeGpu(ulong n, ulong sqrtLimit, byte nMod10, CancellationToken ct, in HeuristicTrialDivisionCallbacks callbacks)
    {
        var summary = default(HeuristicTrialDivisionSummary);
        return HeuristicIsPrimeGpu(n, sqrtLimit, nMod10, ct, ref summary, collectSummary: false, in callbacks);
    }

    internal bool HeuristicIsPrimeGpu(ulong n, ulong sqrtLimit, byte nMod10, CancellationToken ct, ref HeuristicTrialDivisionSummary summary)
    {
        return HeuristicIsPrimeGpu(n, sqrtLimit, nMod10, ct, ref summary, collectSummary: true, callbacks: default);
    }

    internal bool HeuristicIsPrimeGpu(ulong n, ulong sqrtLimit, byte nMod10, CancellationToken ct, ref HeuristicTrialDivisionSummary summary, in HeuristicTrialDivisionCallbacks callbacks)
    {
        return HeuristicIsPrimeGpu(n, sqrtLimit, nMod10, ct, ref summary, collectSummary: true, in callbacks);
    }


    // Note: GPU-backed primality path is implemented via IsPrimeGpu/IsPrimeBatchGpu and is routed
    // from EvenPerfectBitScanner based on --primes-device.

    internal static bool HeuristicIsPrimeCpu(ulong n, CancellationToken ct)
    {
        var summary = default(HeuristicTrialDivisionSummary);
        byte nMod10 = (byte)n.Mod10();
        return HeuristicIsPrimeCpu(n, 0UL, nMod10, ct, ref summary, collectSummary: false, callbacks: default);
    }

    internal static bool HeuristicIsPrimeCpu(ulong n, CancellationToken ct, in HeuristicTrialDivisionCallbacks callbacks)
    {
        var summary = default(HeuristicTrialDivisionSummary);
        byte nMod10 = (byte)n.Mod10();
        return HeuristicIsPrimeCpu(n, 0UL, nMod10, ct, ref summary, collectSummary: false, in callbacks);
    }

    internal static bool HeuristicIsPrimeCpu(ulong n, CancellationToken ct, ref HeuristicTrialDivisionSummary summary)
    {
        byte nMod10 = (byte)n.Mod10();
        return HeuristicIsPrimeCpu(n, 0UL, nMod10, ct, ref summary, collectSummary: true, callbacks: default);
    }

    internal static bool HeuristicIsPrimeCpu(ulong n, CancellationToken ct, ref HeuristicTrialDivisionSummary summary, in HeuristicTrialDivisionCallbacks callbacks)
    {
        byte nMod10 = (byte)n.Mod10();
        return HeuristicIsPrimeCpu(n, 0UL, nMod10, ct, ref summary, collectSummary: true, in callbacks);
    }

    internal static bool HeuristicIsPrimeCpu(ulong n, ulong sqrtLimit, byte nMod10, CancellationToken ct)
    {
        var summary = default(HeuristicTrialDivisionSummary);
        return HeuristicIsPrimeCpu(n, sqrtLimit, nMod10, ct, ref summary, collectSummary: false, callbacks: default);
    }

    internal static bool HeuristicIsPrimeCpu(ulong n, ulong sqrtLimit, byte nMod10, CancellationToken ct, in HeuristicTrialDivisionCallbacks callbacks)
    {
        var summary = default(HeuristicTrialDivisionSummary);
        return HeuristicIsPrimeCpu(n, sqrtLimit, nMod10, ct, ref summary, collectSummary: false, in callbacks);
    }

    internal static bool HeuristicIsPrimeCpu(ulong n, ulong sqrtLimit, byte nMod10, CancellationToken ct, ref HeuristicTrialDivisionSummary summary)
    {
        return HeuristicIsPrimeCpu(n, sqrtLimit, nMod10, ct, ref summary, collectSummary: true, callbacks: default);
    }

    internal static bool HeuristicIsPrimeCpu(ulong n, ulong sqrtLimit, byte nMod10, CancellationToken ct, ref HeuristicTrialDivisionSummary summary, in HeuristicTrialDivisionCallbacks callbacks)
    {
        return HeuristicIsPrimeCpu(n, sqrtLimit, nMod10, ct, ref summary, collectSummary: true, in callbacks);
    }

    private static bool HeuristicIsPrimeCpu(ulong n, ulong sqrtLimit, byte nMod10, CancellationToken ct, ref HeuristicTrialDivisionSummary summary, bool collectSummary, in HeuristicTrialDivisionCallbacks callbacks)
    {
        bool shouldResetSummary = collectSummary || callbacks.RequiresSummary;
        if (TryResolveHeuristicTrivialCases(n, nMod10, shouldResetSummary, ref summary, out bool earlyResult))
        {
            return earlyResult;
        }

        if (sqrtLimit == 0UL)
        {
            sqrtLimit = ComputeSqrtLimit(n);
        }

        return HeuristicTrialDivision(n, sqrtLimit, nMod10, ct, ref summary, collectSummary, in callbacks);
    }


    private static bool HeuristicIsPrimeGpu(ulong n, ulong sqrtLimit, byte nMod10, CancellationToken ct, ref HeuristicTrialDivisionSummary summary, bool collectSummary, in HeuristicTrialDivisionCallbacks callbacks)
    {
        if (GpuContextPool.ForceCpu)
        {
            return HeuristicIsPrimeCpu(n, sqrtLimit, nMod10, ct, ref summary, collectSummary, in callbacks);
        }

        bool shouldResetSummary = collectSummary || callbacks.RequiresSummary;
        if (TryResolveHeuristicTrivialCases(n, nMod10, shouldResetSummary, ref summary, out bool earlyResult))
        {
            return earlyResult;
        }

        if (sqrtLimit == 0UL)
        {
            sqrtLimit = ComputeSqrtLimit(n);
        }

        return HeuristicTrialDivisionGpu(n, sqrtLimit, nMod10, ct, ref summary, collectSummary, in callbacks);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool TryResolveHeuristicTrivialCases(ulong n, byte nMod10, bool resetSummary, ref HeuristicTrialDivisionSummary summary, out bool result)
    {
        if (n < 2UL)
        {
            if (resetSummary)
            {
                summary = default;
            }

            result = false;
            return true;
        }

        if (n == 2UL)
        {
            if (resetSummary)
            {
                summary = default;
            }

            result = true;
            return true;
        }

        if ((n & 1UL) == 0UL)
        {
            if (resetSummary)
            {
                summary = default;
            }

            result = false;
            return true;
        }

        if (n == 5UL)
        {
            if (resetSummary)
            {
                summary = default;
            }

            result = true;
            return true;
        }

        if (n > 5UL && nMod10 == 5)
        {
            if (resetSummary)
            {
                summary = default;
            }

            result = false;
            return true;
        }

        if (n <= 13UL)
        {
            if (resetSummary)
            {
                summary = default;
            }

            result = n == 3UL || n == 7UL || n == 11UL || n == 13UL;
            return true;
        }

        if (nMod10 == 1 && SharesFactorWithMaxExponent(n))
        {
            if (resetSummary)
            {
                summary = default;
            }

            result = false;
            return true;
        }

        result = false;
        return false;
    }

    private static bool HeuristicTrialDivision(ulong n, ulong sqrtLimit, byte nMod10, CancellationToken ct)
    {
        var summary = default(HeuristicTrialDivisionSummary);
        return HeuristicTrialDivision(n, sqrtLimit, nMod10, ct, ref summary, collectSummary: false, callbacks: default);
    }

    private static bool HeuristicTrialDivision(ulong n, ulong sqrtLimit, byte nMod10, CancellationToken ct, in HeuristicTrialDivisionCallbacks callbacks)
    {
        var summary = default(HeuristicTrialDivisionSummary);
        return HeuristicTrialDivision(n, sqrtLimit, nMod10, ct, ref summary, collectSummary: false, in callbacks);
    }

    private static bool HeuristicTrialDivision(ulong n, ulong sqrtLimit, byte nMod10, CancellationToken ct, ref HeuristicTrialDivisionSummary summary)
    {
        return HeuristicTrialDivision(n, sqrtLimit, nMod10, ct, ref summary, collectSummary: true, callbacks: default);
    }

    private static bool HeuristicTrialDivision(ulong n, ulong sqrtLimit, byte nMod10, CancellationToken ct, ref HeuristicTrialDivisionSummary summary, in HeuristicTrialDivisionCallbacks callbacks)
    {
        return HeuristicTrialDivision(n, sqrtLimit, nMod10, ct, ref summary, collectSummary: true, in callbacks);
    }

    private static bool HeuristicTrialDivision(ulong n, ulong sqrtLimit, byte nMod10, CancellationToken ct, ref HeuristicTrialDivisionSummary summary, bool collectSummary, in HeuristicTrialDivisionCallbacks callbacks)
    {
        bool shouldCollectSummary = collectSummary || callbacks.RequiresSummary;
        if (shouldCollectSummary)
        {
            summary = default;
        }

        Span<HeuristicGroupBSequenceState> groupBBuffer = stackalloc HeuristicGroupBSequenceState[MaxGroupBSequences];
        var enumerator = new HeuristicDivisorEnumerator(sqrtLimit, nMod10, groupBBuffer);

        while (enumerator.TryGetNext(out HeuristicDivisorCandidate candidate))
        {
            ulong divisor = candidate.Value;
            if (divisor <= 1UL)
            {
                continue;
            }

            if (shouldCollectSummary)
            {
                UpdateHeuristicSummaryForCandidate(ref summary, candidate);
            }

            if (callbacks.HasCandidate)
            {
                callbacks.Candidate(in candidate, in summary);
            }

            ct.ThrowIfCancellationRequested();

            if (n % divisor == 0UL)
            {
                HeuristicDivisorPreparation preparation;
                if (shouldCollectSummary)
                {
                    summary.Hit = true;
                    summary.HitDivisor = divisor;
                    summary.HitGroup = candidate.Group;
                    summary.HitEnding = candidate.Ending;
                    summary.HitPriorityIndex = candidate.PriorityIndex;
                    summary.HitPreparation = PrepareHeuristicDivisor(in candidate);
                    summary.HasHitPreparation = true;
                    ResolveHeuristicHit(n, ref summary);
                    preparation = summary.HitPreparation;
                }
                else
                {
                    preparation = PrepareHeuristicDivisor(in candidate);
                }

                if (callbacks.HasHit)
                {
                    callbacks.Hit(in preparation, in summary);
                }

                if (callbacks.HasCompleted)
                {
                    callbacks.Completed(in summary, isPrime: false);
                }

                return false;
            }
        }

        if (shouldCollectSummary)
        {
            summary.Hit = false;
            summary.HitDivisor = 0UL;
            summary.HitGroup = HeuristicDivisorGroup.None;
            summary.HitEnding = 0;
            summary.HitPriorityIndex = 0;
            summary.HitPreparation = default;
            summary.HitCycleLength = 0UL;
            summary.HitCycleFromHint = false;
            summary.HitCycleComputed = false;
            summary.HitPrimeOrderFailed = false;
            summary.HitMontgomeryIsUnity = false;
            summary.HitConfirmsMersenne = false;
            summary.HasHitPreparation = false;
        }

        if (callbacks.HasCompleted)
        {
            callbacks.Completed(in summary, isPrime: true);
        }

        return true;
    }

    private static bool HeuristicTrialDivisionGpu(ulong n, ulong sqrtLimit, byte nMod10, CancellationToken ct, ref HeuristicTrialDivisionSummary summary, bool collectSummary, in HeuristicTrialDivisionCallbacks callbacks)
    {
        bool shouldCollectSummary = collectSummary || callbacks.RequiresSummary;
        if (shouldCollectSummary)
        {
            summary = default;
        }

        HeuristicGpuConfirmationMode confirmationMode = HeuristicGpuConfirmation;
        bool confirmHits = (confirmationMode & HeuristicGpuConfirmationMode.OnHit) != 0;
        bool confirmPrimes = (confirmationMode & HeuristicGpuConfirmationMode.OnPrime) != 0;

        Span<HeuristicGroupBSequenceState> groupBBuffer = stackalloc HeuristicGroupBSequenceState[MaxGroupBSequences];
        var enumerator = new HeuristicDivisorEnumerator(sqrtLimit, nMod10, groupBBuffer);

        int batchCapacity = Math.Max(1, HeuristicGpuDivisorBatchSize);
        var candidatePool = ArrayPool<HeuristicDivisorCandidate>.Shared;
        var divisorPool = ArrayPool<ulong>.Shared;
        var hitPool = ArrayPool<byte>.Shared;

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

                    static bool ProcessBatch(
                        int length,
                        ulong n,
                        bool collectSummaryLocal,
                        bool confirmHitsLocal,
                        ref HeuristicTrialDivisionSummary summaryRef,
                        in HeuristicTrialDivisionCallbacks callbacksRef,
                        HeuristicDivisorCandidate[] candidateArrayLocal,
                        ulong[] divisorArrayLocal,
                        byte[] hitFlagsLocal,
                        KernelState.ScratchBuffers scratchLocal,
                        Accelerator acceleratorLocal,
                        Action<Index1D, ArrayView<ulong>, ulong, ArrayView<byte>> kernel)
                    {
                        scratchLocal.Input.View.CopyFromCPU(ref divisorArrayLocal[0], length);
                        kernel(length, scratchLocal.Input.View, n, scratchLocal.Output.View);
                        acceleratorLocal.Synchronize();
                        scratchLocal.Output.View.CopyToCPU(ref hitFlagsLocal[0], length);

                        for (int i = 0; i < length; i++)
                        {
                            if (hitFlagsLocal[i] == 0)
                            {
                                continue;
                            }

                            var candidate = candidateArrayLocal[i];

                            if (confirmHitsLocal && n % candidate.Value != 0UL)
                            {
                                throw new InvalidOperationException("GPU heuristic reported a divisor that CPU rejected.");
                            }

                            HeuristicDivisorPreparation preparation;
                            if (collectSummaryLocal)
                            {
                                summaryRef.Hit = true;
                                summaryRef.HitDivisor = candidate.Value;
                                summaryRef.HitGroup = candidate.Group;
                                summaryRef.HitEnding = candidate.Ending;
                                summaryRef.HitPriorityIndex = candidate.PriorityIndex;
                                summaryRef.HitPreparation = PrepareHeuristicDivisor(in candidate);
                                summaryRef.HasHitPreparation = true;
                                ResolveHeuristicHit(n, ref summaryRef);
                                preparation = summaryRef.HitPreparation;
                            }
                            else
                            {
                                preparation = PrepareHeuristicDivisor(in candidate);
                            }

                            if (callbacksRef.HasHit)
                            {
                                callbacksRef.Hit(in preparation, in summaryRef);
                            }

                            if (callbacksRef.HasCompleted)
                            {
                                callbacksRef.Completed(in summaryRef, isPrime: false);
                            }

                            return true;
                        }

                        return false;
                    }

                    while (enumerator.TryGetNext(out HeuristicDivisorCandidate candidate))
                    {
                        ct.ThrowIfCancellationRequested();

                        ulong divisor = candidate.Value;
                        if (divisor <= 1UL)
                        {
                            continue;
                        }

                        if (shouldCollectSummary)
                        {
                            UpdateHeuristicSummaryForCandidate(ref summary, in candidate);
                        }

                        if (callbacks.HasCandidate)
                        {
                            callbacks.Candidate(in candidate, in summary);
                        }

                        candidateArray[count] = candidate;
                        divisorArray[count] = divisor;
                        count++;

                        if (count == batchCapacity)
                        {
                            ct.ThrowIfCancellationRequested();
                            if (ProcessBatch(count, n, shouldCollectSummary, confirmHits, ref summary, in callbacks, candidateArray, divisorArray, hitFlags, scratch, accelerator, state.HeuristicTrialDivisionKernel))
                            {
                                compositeDetected = true;
                                break;
                            }

                            count = 0;
                        }
                    }

                    if (!compositeDetected && count > 0)
                    {
                        ct.ThrowIfCancellationRequested();
                        if (ProcessBatch(count, n, shouldCollectSummary, confirmHits, ref summary, in callbacks, candidateArray, divisorArray, hitFlags, scratch, accelerator, state.HeuristicTrialDivisionKernel))
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

        if (compositeDetected)
        {
            return false;
        }

        if (confirmPrimes)
        {
            ct.ThrowIfCancellationRequested();
            var confirmationSummary = default(HeuristicTrialDivisionSummary);
            bool cpuPrime = HeuristicTrialDivision(n, sqrtLimit, nMod10, ct, ref confirmationSummary, collectSummary: false, callbacks: default);
            if (!cpuPrime)
            {
                throw new InvalidOperationException("GPU heuristic reported a prime result that the CPU rejected.");
            }
        }

        if (shouldCollectSummary)
        {
            summary.Hit = false;
            summary.HitDivisor = 0UL;
            summary.HitGroup = HeuristicDivisorGroup.None;
            summary.HitEnding = 0;
            summary.HitPriorityIndex = 0;
            summary.HitPreparation = default;
            summary.HitCycleLength = 0UL;
            summary.HitCycleFromHint = false;
            summary.HitCycleComputed = false;
            summary.HitPrimeOrderFailed = false;
            summary.HitMontgomeryIsUnity = false;
            summary.HitConfirmsMersenne = false;
            summary.HasHitPreparation = false;
        }

        if (callbacks.HasCompleted)
        {
            callbacks.Completed(in summary, isPrime: true);
        }

        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static ulong ComputeSqrtLimit(ulong n)
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
    internal static void UpdateHeuristicSummaryForCandidate(ref HeuristicTrialDivisionSummary summary, in HeuristicDivisorCandidate candidate)
    {
        ulong divisor = candidate.Value;
        summary.TotalDivisorsTested++;
        summary.LastTestedDivisor = divisor;

        switch (candidate.Group)
        {
            case HeuristicDivisorGroup.GroupAConstant:
                summary.SawGroupAConstant = true;
                if (divisor > summary.MaxGroupADivisor)
                {
                    summary.MaxGroupADivisor = divisor;
                }

                break;
            case HeuristicDivisorGroup.GroupAWheel:
                summary.SawGroupAWheel = true;
                if (divisor > summary.MaxGroupADivisor)
                {
                    summary.MaxGroupADivisor = divisor;
                }

                break;
            case HeuristicDivisorGroup.GroupB:
                summary.SawGroupB = true;
                if (summary.MinGroupBDivisor == 0UL || divisor < summary.MinGroupBDivisor)
                {
                    summary.MinGroupBDivisor = divisor;
                    summary.MinGroupBEnding = candidate.Ending;
                    summary.MinGroupBPriorityIndex = candidate.PriorityIndex;
                }

                break;
        }
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
    internal static bool TryExportHeuristicHit(in HeuristicTrialDivisionSummary summary, out HeuristicDivisorHitResolution resolution)
    {
        if (!summary.Hit || !summary.HasHitPreparation)
        {
            resolution = default;
            return false;
        }

        HeuristicDivisorPreparation preparation = summary.HitPreparation;
        resolution = new HeuristicDivisorHitResolution(
            in preparation,
            summary.HitCycleLength,
            summary.HitCycleFromHint,
            summary.HitCycleComputed,
            summary.HitPrimeOrderFailed,
            summary.HitMontgomeryIsUnity,
            summary.HitConfirmsMersenne);
        return true;
    }


    private static void ResolveHeuristicHit(ulong exponent, ref HeuristicTrialDivisionSummary summary)
    {
        if (!summary.HasHitPreparation)
        {
            return;
        }

        HeuristicDivisorPreparation preparation = summary.HitPreparation;
        MontgomeryDivisorData divisorData = preparation.DivisorData;

        bool cycleFromHint;
        bool cycleComputed;
        bool primeOrderFailed;

        ulong cycleLength = ResolveHeuristicCycleLength(
            exponent,
            in preparation,
            out cycleFromHint,
            out cycleComputed,
            out primeOrderFailed);

        summary.HitCycleLength = cycleLength;
        summary.HitCycleFromHint = cycleFromHint;
        summary.HitCycleComputed = cycleComputed;
        summary.HitPrimeOrderFailed = primeOrderFailed;

        bool montgomeryUnity = false;
        bool confirmsMersenne = false;

        if (cycleComputed && cycleLength != 0UL)
        {
            ExponentRemainderStepperCpu stepper = ThreadStaticPools.RentExponentStepperCpu(divisorData);
            try
            {
                montgomeryUnity = stepper.InitializeCpuIsUnity(exponent);
            }
            finally
            {
                ThreadStaticPools.ReturnExponentStepperCpu(stepper);
            }

            confirmsMersenne = montgomeryUnity && cycleLength == exponent;
        }

        summary.HitMontgomeryIsUnity = montgomeryUnity;
        summary.HitConfirmsMersenne = confirmsMersenne;
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
            while (groupAConstantIndex < GroupAConstantDivisors.Length)
            {
                ulong value = GroupAConstantDivisors[groupAConstantIndex++];
                if (value > sqrtLimit)
                {
                    groupAConstantIndex = GroupAConstantDivisors.Length;
                    break;
                }

                candidate = new HeuristicDivisorCandidate(value, HeuristicDivisorGroup.GroupAConstant, (byte)(value % 10UL), 0, (ushort)(value % Wheel210));
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

    internal struct HeuristicGroupBSequenceState
    {
        public ulong Candidate;

        public byte PriorityIndex;

        public byte Ending;

        public ushort Residue;

        public HeuristicGroupBSequenceState(ushort residue, byte ending, byte priorityIndex)
        {
            Candidate = residue;
            Ending = ending;
            Residue = residue;
            PriorityIndex = priorityIndex;
        }

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

        return HeuristicIsPrimeCpu(n, ct);
    }

    internal static bool LegacyIsPrimeInternal(ulong n, CancellationToken ct)
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

    public static HeuristicGpuConfirmationMode HeuristicGpuConfirmation { get; set; } = HeuristicGpuConfirmationMode.None;

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
