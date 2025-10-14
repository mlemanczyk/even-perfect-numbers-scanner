using System.Buffers;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using Open.Collections;
using Open.Numeric.Primes;
using PerfectNumbers.Core;
using PerfectNumbers.Core.Cpu;
using PerfectNumbers.Core.Gpu;

namespace EvenPerfectBitScanner;

internal static class Program
{
    private static ThreadLocal<PrimeTester> PrimeTesters = null!;
    private static ThreadLocal<MersenneNumberTester> MersenneTesters = null!;
    private static ThreadLocal<ModResidueTracker> PResidue = null!;      // p mod d tracker (per-thread)
    private const ulong InitialP = PerfectNumberConstants.BiggestKnownEvenPerfectP;
    private const int WriteBatchSize = 100;
    private const string DefaultCyclesPath = "divisor_cycles.bin";
    private static string ResultsFileName = "even_perfect_bit_scan_results.csv";
    private const string PrimeFoundSuffix = " (FOUND VALID CANDIDATES)";
    private static StringBuilder? _outputBuilder;
    private static readonly object Sync = new();
    private static readonly object FileWriteSync = new();
    private static int _consoleCounter;
    private static int _writeIndex;
    private static int _primeCount;
    private static bool _primeFoundAfterInit;
    private static bool _useGcdFilter;
    private static bool _useDivisor;
    private static bool _useResidueMode;
    private static bool _useByDivisorMode;
    private static MersenneNumberDivisorGpuTester? _divisorTester;
    private static IMersenneNumberDivisorByDivisorTester? _byDivisorTester;
    private static Dictionary<ulong, (bool DetailedCheck, bool PassedAllTests)>? _byDivisorPreviousResults;
    private static ulong? _orderWarmupLimitOverride;
    private static unsafe delegate*<ulong, ref ulong, ulong> _transformP;
    private static long _state;
    private static bool _limitReached;
    private static string? _resultsDir;
    private static string? _resultsPrefix;
    private static readonly Optimized PrimeIterator = new();
    private static ulong _byDivisorStartPrime;
    private static TimeSpan? _primeTestLimit;

    [ThreadStatic]
    private static bool _lastCompositeP;

    // RLE/bit-pattern filtering options
    private static string? _rleBlacklistPath;
    private static ulong _rleHardMaxP = ulong.MaxValue;
    private static bool _rleOnlyLast7 = true;
    private static int _writeBatchSize = WriteBatchSize;
    private static double _zeroFracHard = -1.0;                 // disabled when < 0
    private static double _zeroFracConj = -1.0;                 // disabled when < 0
    private static int _maxZeroConj = -1;                       // disabled when < 0

    private static unsafe void Main(string[] args)
    {
        ulong currentP = InitialP;
        int threadCount = Environment.ProcessorCount;
        int blockSize = 1;
        int gpuPrimeThreads = 1;
        int gpuSmallCycleThreads = 512;
        int gpuPrimeBatch = 262_144;
        GpuKernelType kernelType = GpuKernelType.Incremental;
        bool useModuloWorkaround = false; // TODO: Remove once the runtime defaults to the ImmediateModulo path measured fastest in GpuUInt128NativeModuloBenchmarks.
                                          // removed: useModAutomaton
        bool useOrder = false;
        bool showHelp = false;
        bool useBitTransform = false;
        bool useLucas = false;
        bool useResidue = false;    // M_p test via residue divisors (replaces LL/incremental)
        bool useDivisor = false;     // M_p divisibility by specific divisor
        bool useByDivisor = false;   // Iterative divisor scan across primes		UInt128 divisor = UInt128.Zero;
        ByDivisorDeltasDevice byDivisorDeltasDevice = ByDivisorDeltasDevice.Cpu;
        ByDivisorMontgomeryDevice byDivisorMontgomeryDevice = ByDivisorMontgomeryDevice.Cpu;
                                     // Device routing
        bool useGpuCycles = true;
        bool mersenneOnGpu = true;   // controls Lucas/incremental/pow2mod device
        bool orderOnGpu = true;      // controls order computations device
        int scanBatchSize = 2_097_152, sliceSize = 32;
        ulong residueKMax = 5_000_000UL;
        string filterFile = string.Empty;
        string cyclesPath = DefaultCyclesPath;
        int cyclesBatchSize = 512;
        bool continueCyclesGeneration = false;
        ulong divisorCyclesSearchLimit = PerfectNumberConstants.ExtraDivisorCycleSearchLimit;
        int argIndex = 0;
        string arg = string.Empty;
        ReadOnlySpan<char> mersenneOption = default;
        ulong parsedLimit = 0UL;
        ulong parsedResidueMax = 0UL;
        ulong remainder = 0UL;
        bool startPrimeProvided = false;

        // NTT backend selection (GPU): reference vs staged
        for (; argIndex < args.Length; argIndex++)
        {
            arg = args[argIndex];
            if (arg.Equals("--?", StringComparison.OrdinalIgnoreCase) || arg.Equals("-?", StringComparison.OrdinalIgnoreCase) || arg.Equals("--help", StringComparison.OrdinalIgnoreCase) || arg.Equals("-help", StringComparison.OrdinalIgnoreCase) || arg.Equals("/?", StringComparison.OrdinalIgnoreCase))
            {
                showHelp = true;
                break;
            }
            else if (arg.StartsWith("--prime=", StringComparison.OrdinalIgnoreCase))
            {
                // TODO: Replace this ulong.Parse with the Utf8Parser-based span helper once CLI option parsing
                // adopts the zero-allocation fast path proven fastest in the parser benchmarks.
                currentP = ulong.Parse(arg.AsSpan(arg.IndexOf('=') + 1));
                startPrimeProvided = true;
            }
            else if (arg.Equals("--increment=bit", StringComparison.OrdinalIgnoreCase))
            {
                useBitTransform = true;
            }
            else if (arg.Equals("--increment=add", StringComparison.OrdinalIgnoreCase))
            {
                useBitTransform = false;
            }
            else if (arg.StartsWith("--threads=", StringComparison.OrdinalIgnoreCase))
            {
                // TODO: Swap int.Parse for the span-based Utf8Parser fast path to avoid transient strings when
                // processing CLI numeric options.
                threadCount = Math.Max(1, int.Parse(arg.AsSpan(arg.IndexOf('=') + 1)));
            }
            else if (arg.StartsWith("--mersenne=", StringComparison.OrdinalIgnoreCase))
            {
                mersenneOption = arg.AsSpan(arg.IndexOf('=') + 1);
                if (mersenneOption.Equals("pow2mod", StringComparison.OrdinalIgnoreCase))
                {
                    kernelType = GpuKernelType.Pow2Mod; // Dispatch to the ProcessEightBitWindows pow2mod kernel.
                }
                else if (mersenneOption.Equals("lucas", StringComparison.OrdinalIgnoreCase))
                {
                    useLucas = true;
                }
                else if (mersenneOption.Equals("residue", StringComparison.OrdinalIgnoreCase))
                {
                    useResidue = true;
                    useLucas = false;
                }
                else if (mersenneOption.Equals("divisor", StringComparison.OrdinalIgnoreCase))
                {
                    useDivisor = true;
                    useLucas = false;
                    useResidue = false;
                }
                else if (mersenneOption.Equals("bydivisor", StringComparison.OrdinalIgnoreCase))
                {
                    useByDivisor = true;
                    useLucas = false;
                    useResidue = false;
                    useDivisor = false;
                }
                else
                {
                    kernelType = GpuKernelType.Incremental;
                }
            }
            else if (arg.StartsWith("--divisor-cycles-limit=", StringComparison.OrdinalIgnoreCase))
            {
                // TODO: Replace ulong.TryParse with the Utf8Parser fast-path once CLI parsing utilities expose
                // span-based helpers for optional arguments.
                if (ulong.TryParse(arg.AsSpan(arg.IndexOf('=') + 1), out parsedLimit))
                {
                    divisorCyclesSearchLimit = parsedLimit;
                }
            }
            else if (arg.StartsWith("--residue-max-k=", StringComparison.OrdinalIgnoreCase))
            {
                // TODO: Swap this TryParse for the zero-allocation Utf8Parser helper to keep residue CLI option
                // parsing in the fast path.
                if (ulong.TryParse(arg.AsSpan(arg.IndexOf('=') + 1), out parsedResidueMax))
                {
                    residueKMax = parsedResidueMax;
                }
            }
            // Replaces --lucas=cpu|gpu; controls device for Lucas and for
            // incremental/pow2mod scanning
            else if (arg.StartsWith("--mersenne-device=", StringComparison.OrdinalIgnoreCase))
            {
                mersenneOnGpu = !arg.AsSpan(arg.IndexOf('=') + 1).Equals("cpu", StringComparison.OrdinalIgnoreCase);
            }
            else if (arg.StartsWith("--bydivisor-deltas-device=", StringComparison.OrdinalIgnoreCase))
            {
                ReadOnlySpan<char> deviceValue = arg.AsSpan(arg.IndexOf('=') + 1);
                if (deviceValue.Equals("gpu", StringComparison.OrdinalIgnoreCase))
                {
                    byDivisorDeltasDevice = ByDivisorDeltasDevice.Gpu;
                }
                else if (deviceValue.Equals("cpu", StringComparison.OrdinalIgnoreCase))
                {
                    byDivisorDeltasDevice = ByDivisorDeltasDevice.Cpu;
                }
                else
                {
                    Console.WriteLine("Invalid --bydivisor-deltas-device value. Expected cpu or gpu.");
                    return;
                }
            }
            else if (arg.StartsWith("--bydivisor-montgomery-device=", StringComparison.OrdinalIgnoreCase))
            {
                ReadOnlySpan<char> deviceValue = arg.AsSpan(arg.IndexOf('=') + 1);
                if (deviceValue.Equals("gpu", StringComparison.OrdinalIgnoreCase))
                {
                    byDivisorMontgomeryDevice = ByDivisorMontgomeryDevice.Gpu;
                }
                else if (deviceValue.Equals("cpu", StringComparison.OrdinalIgnoreCase))
                {
                    byDivisorMontgomeryDevice = ByDivisorMontgomeryDevice.Cpu;
                }
                else
                {
                    Console.WriteLine("Invalid --bydivisor-montgomery-device value. Expected cpu or gpu.");
                    return;
                }
            }
            else if (arg.StartsWith("--prime-test-limit=", StringComparison.OrdinalIgnoreCase))
            {
                ReadOnlySpan<char> value = arg.AsSpan(arg.IndexOf('=') + 1);
                if (!TryParsePrimeTestLimit(value, out TimeSpan parsedLimitValue))
                {
                    Console.WriteLine("Failed to parse --prime-test-limit value.");
                    return;
                }

                _primeTestLimit = parsedLimitValue;
            }
            else if (arg.Equals("--workaround-mod", StringComparison.OrdinalIgnoreCase))
            {
                useModuloWorkaround = true;
            }
            else if (arg.Equals("--use-order", StringComparison.OrdinalIgnoreCase))
            {
                useOrder = true;
            }
            else if (arg.StartsWith("--order-warmup-limit=", StringComparison.OrdinalIgnoreCase))
            {
                // stored and used below when initializing testers
                // Reusing parsedLimit to capture the --order-warmup-limit argument.
                // TODO: Use the Utf8Parser-based reader here to eliminate the temporary substring allocation
                // when parsing warm-up limits.
                if (ulong.TryParse(arg.AsSpan(arg.IndexOf('=') + 1), out parsedLimit))
                {
                    _orderWarmupLimitOverride = parsedLimit;
                }
            }
            else if (arg.Equals("--gcd-filter", StringComparison.OrdinalIgnoreCase))
            {
                _useGcdFilter = true;
            }
            else if (arg.StartsWith("--ntt=", StringComparison.OrdinalIgnoreCase))
            {
                ReadOnlySpan<char> value = arg.AsSpan(arg.IndexOf('=') + 1);
                if (value.Equals("staged", StringComparison.OrdinalIgnoreCase))
                {
                    NttGpuMath.GpuTransformBackend = NttBackend.Staged;
                }
                else
                {
                    NttGpuMath.GpuTransformBackend = NttBackend.Reference;
                }
            }
            else if (arg.StartsWith("--mod-reduction=", StringComparison.OrdinalIgnoreCase))
            {
                ReadOnlySpan<char> value = arg.AsSpan(arg.IndexOf('=') + 1);
                if (value.Equals("mont64", StringComparison.OrdinalIgnoreCase))
                {
                    NttGpuMath.ReductionMode = ModReductionMode.Mont64;
                }
                else if (value.Equals("barrett128", StringComparison.OrdinalIgnoreCase))
                {
                    NttGpuMath.ReductionMode = ModReductionMode.Barrett128;
                }
                else if (value.Equals("uint128", StringComparison.OrdinalIgnoreCase))
                {
                    NttGpuMath.ReductionMode = ModReductionMode.GpuUInt128;
                }
                else
                {
                    NttGpuMath.ReductionMode = ModReductionMode.Auto;
                }
            }
            else if (arg.StartsWith("--primes-device=", StringComparison.OrdinalIgnoreCase))
            {
                // Controls default device for library prime-related GPU kernels
                // that are not explicitly parameterized (backward compatibility).
                GpuContextPool.ForceCpu = arg.AsSpan(arg.IndexOf('=') + 1).Equals("cpu", StringComparison.OrdinalIgnoreCase);
            }
            // New: choose device for order computations (warm-ups and order scans)
            else if (arg.StartsWith("--order-device=", StringComparison.OrdinalIgnoreCase))
            {
                orderOnGpu = !arg.AsSpan(arg.IndexOf('=') + 1).Equals("cpu", StringComparison.OrdinalIgnoreCase);
            }
            // Optional: RLE blacklist for p
            else if (arg.StartsWith("--rle-blacklist=", StringComparison.OrdinalIgnoreCase))
            {
                _rleBlacklistPath = arg[(arg.IndexOf('=') + 1)..];
            }
            else if (arg.StartsWith("--rle-hard-max=", StringComparison.OrdinalIgnoreCase))
            {
                ulong rleMaxP;
                // TODO: Replace this TryParse with the Utf8Parser span helper once shared CLI parsing utilities
                // land, mirroring the other numeric option optimizations.
                if (ulong.TryParse(arg.AsSpan(arg.IndexOf('=') + 1), out rleMaxP))
                {
                    _rleHardMaxP = rleMaxP;
                }
            }
            else if (arg.StartsWith("--rle-only-last7=", StringComparison.OrdinalIgnoreCase))
            {
                ReadOnlySpan<char> value = arg.AsSpan(arg.IndexOf('=') + 1);
                _rleOnlyLast7 = !value.Equals("false", StringComparison.OrdinalIgnoreCase) && !value.Equals("0", StringComparison.OrdinalIgnoreCase);
            }
            else if (arg.StartsWith("--zero-hard=", StringComparison.OrdinalIgnoreCase))
            {
                double zeroFraction;
                // TODO: Swap to the Utf8Parser double fast-path so zero-hard parsing avoids culture-dependent
                // conversions in the hot CLI path.
                if (double.TryParse(arg.AsSpan(arg.IndexOf('=') + 1), out zeroFraction))
                {
                    _zeroFracHard = zeroFraction;
                }
            }
            else if (arg.StartsWith("--zero-conj=", StringComparison.OrdinalIgnoreCase))
            {
                // format: <zeroFrac>:<maxZeroBlock>
                ReadOnlySpan<char> value = arg.AsSpan(arg.IndexOf('=') + 1);
                int colon = value.IndexOf(':');
                if (colon > 0)
                {
                    double zeroFrac;
                    int maxZero;
                    // TODO: Replace these double/int TryParse calls with span-based Utf8Parser helpers once
                    // available so parsing zero-conj stays allocation-free.
                    if (double.TryParse(value[..colon], out zeroFrac) && int.TryParse(value[(colon + 1)..], out maxZero))
                    {
                        _zeroFracConj = zeroFrac;
                        _maxZeroConj = maxZero;
                    }
                }
            }
            // Backward-compat: accept deprecated device flags and map to new ones
            else if (arg.StartsWith("--lucas=", StringComparison.OrdinalIgnoreCase))
            {
                mersenneOnGpu = !arg.AsSpan(arg.IndexOf('=') + 1).Equals("cpu", StringComparison.OrdinalIgnoreCase);
            }
            else if (arg.StartsWith("--primes=", StringComparison.OrdinalIgnoreCase))
            {
                GpuContextPool.ForceCpu = arg.AsSpan(arg.IndexOf('=') + 1).Equals("cpu", StringComparison.OrdinalIgnoreCase);
            }
            else if (arg.StartsWith("--gpu-kernels=", StringComparison.OrdinalIgnoreCase) || arg.StartsWith("--accelerator=", StringComparison.OrdinalIgnoreCase))
            {
                // Legacy alias: treat as primes-device for backward compat
                GpuContextPool.ForceCpu = arg.AsSpan(arg.IndexOf('=') + 1).Equals("cpu", StringComparison.OrdinalIgnoreCase);
            }
            else if (arg.StartsWith("--results-dir=", StringComparison.OrdinalIgnoreCase))
            {
                _resultsDir = arg[(arg.IndexOf('=') + 1)..];
            }
            else if (arg.StartsWith("--results-prefix=", StringComparison.OrdinalIgnoreCase))
            {
                _resultsPrefix = arg[(arg.IndexOf('=') + 1)..];
            }

            else if (arg.StartsWith("--gpu-prime-threads=", StringComparison.OrdinalIgnoreCase))
            {
                // TODO: Convert this int.Parse to the Utf8Parser-based helper to keep CLI option parsing
                // allocation-free in the hot startup path.
                gpuPrimeThreads = Math.Max(1, int.Parse(arg.AsSpan(arg.IndexOf('=') + 1)));
            }
            else if (arg.StartsWith("--primes-with-smallcycles-threads=", StringComparison.OrdinalIgnoreCase))
            {
                gpuSmallCycleThreads = Math.Max(1, int.Parse(arg.AsSpan(arg.IndexOf('=') + 1)));
            }
            else if (arg.StartsWith("--gpu-prime-batch=", StringComparison.OrdinalIgnoreCase))
            {
                // TODO: Swap int.Parse for Utf8Parser to align with the faster CLI numeric parsing path.
                gpuPrimeBatch = Math.Max(1, int.Parse(arg.AsSpan(arg.IndexOf('=') + 1)));
            }
            else if (arg.StartsWith("--ll-slice=", StringComparison.OrdinalIgnoreCase))
            {
                // TODO: Replace int.Parse with Utf8Parser-based parsing to eliminate temporary strings when
                // reading slice configuration.
                sliceSize = Math.Max(1, int.Parse(arg.AsSpan(arg.IndexOf('=') + 1)));
            }
            else if (arg.StartsWith("--gpu-scan-batch=", StringComparison.OrdinalIgnoreCase))
            {
                // TODO: Use the Utf8Parser-based fast path here to match the other CLI numeric parsing
                // optimizations we plan to adopt.
                scanBatchSize = Math.Max(1, int.Parse(arg.AsSpan(arg.IndexOf('=') + 1)));
            }
            else if (arg.StartsWith("--block-size=", StringComparison.OrdinalIgnoreCase))
            {
                // TODO: Inline the Utf8Parser-based int reader here once shared CLI parsing utilities land so
                // we avoid repeated string allocations.
                blockSize = Math.Max(1, int.Parse(arg.AsSpan(arg.IndexOf('=') + 1)));
            }
            else if (arg.StartsWith("--filter-p=", StringComparison.OrdinalIgnoreCase))
            {
                filterFile = arg[(arg.IndexOf('=') + 1)..];
            }
            else if (arg.StartsWith("--write-batch-size=", StringComparison.OrdinalIgnoreCase))
            {
                // TODO: Replace int.Parse with the shared Utf8Parser fast-path to keep hot CLI option parsing
                // allocation-free.
                _writeBatchSize = Math.Max(1, int.Parse(arg.AsSpan(arg.IndexOf('=') + 1)));
            }
            if (arg.StartsWith("--divisor-cycles="))
            {
                cyclesPath = arg[(arg.IndexOf('=') + 1)..];
            }
            else if (arg.StartsWith("--divisor-cycles-device=", StringComparison.OrdinalIgnoreCase))
            {
                useGpuCycles = !arg.AsSpan(arg.IndexOf('=') + 1).Equals("cpu", StringComparison.OrdinalIgnoreCase);
            }
            else if (arg.StartsWith("--divisor-cycles-batch=", StringComparison.OrdinalIgnoreCase))
            {
                // TODO: Convert this int.Parse to Utf8Parser once the optimized CLI parsing helper is shared
                // across tools.
                cyclesBatchSize = Math.Max(1, int.Parse(arg.AsSpan(arg.IndexOf('=') + 1)));
            }
            else if (arg.Equals("--divisor-cycles-continue", StringComparison.OrdinalIgnoreCase))
            {
                continueCyclesGeneration = true;
            }
        }

        GpuKernelPool.CpuCycleUsesGpu = orderOnGpu;

        if (useByDivisor)
        {
            _byDivisorStartPrime = startPrimeProvided ? currentP : 0UL;
        }
        else
        {
            _byDivisorStartPrime = 0UL;
        }

        if (useByDivisor && string.IsNullOrEmpty(filterFile))
        {
            Console.WriteLine("--mersenne=bydivisor requires --filter-p=<path>.");
            return;
        }

        if (showHelp)
        {
            PrintHelp();
            return;
        }

        // Apply GPU prime sieve runtime configuration
        GpuPrimeWorkLimiter.SetLimit(gpuPrimeThreads);
        GpuSmallCycleKernelLimiter.SetLimit(gpuSmallCycleThreads);
        PrimeTester.GpuBatchSize = Math.Max(1, gpuPrimeBatch);

        bool warmGpuPow2Kernel = mersenneOnGpu
            && !GpuContextPool.ForceCpu
            && !useResidue
            && !useDivisor
            && !useByDivisor
            && !useLucas
            && kernelType == GpuKernelType.Pow2Mod;
        if (warmGpuPow2Kernel)
        {
            WarmPow2ModKernels(orderOnGpu, useOrder);
        }

        MersenneDivisorCycles mersenneDivisorCycles = new();
        if (!File.Exists(cyclesPath) || continueCyclesGeneration)
        {
            Console.WriteLine($"Generating divisor cycles '{cyclesPath}' for all p <= {PerfectNumberConstants.MaxQForDivisorCycles}...");

            long nextPosition = 0L, completeCount = 0L;
            if (continueCyclesGeneration && File.Exists(cyclesPath))
            {
                Console.WriteLine("Finding last divisor position...");
                (nextPosition, completeCount) = MersenneDivisorCycles.FindLast(cyclesPath);
            }

            if (useGpuCycles)
            {
                if (!continueCyclesGeneration || completeCount == 0L)
                {
                    Console.WriteLine("Starting generation...");
                }
                else
                {
                    Console.WriteLine("Resuming generation...");
                }

                // TODO: Wire GenerateGpu to the unrolled-hex kernel that led the MersenneDivisorCycleLengthGpuBenchmarks once it lands.
                MersenneDivisorCycles.GenerateGpu(cyclesPath, PerfectNumberConstants.MaxQForDivisorCycles, cyclesBatchSize, skipCount: completeCount, nextPosition: nextPosition);
            }
            else
            {
                MersenneDivisorCycles.Generate(cyclesPath, PerfectNumberConstants.MaxQForDivisorCycles);
            }
        }

        Console.WriteLine($"Loading divisor cycles into memory...");
        if (string.Equals(Path.GetExtension(cyclesPath), ".csv", StringComparison.OrdinalIgnoreCase))
        {
            MersenneDivisorCycles.Shared.LoadFrom(cyclesPath);
        }
        else
        {
            MersenneDivisorCycles.Shared.LoadFrom(cyclesPath);
        }

        DivisorCycleCache.SetDivisorCyclesBatchSize(cyclesBatchSize);
        DivisorCycleCache.Shared.ConfigureGeneratorDevice(useGpuCycles);
        // TODO: Keep a single cached block loaded from disk and honor the configured device when
        // computing ad-hoc cycles for divisors that fall outside that snapshot instead of queuing
        // generation of additional blocks.
        DivisorCycleCache.Shared.RefreshSnapshot();
        // TODO: Stop reloading the full snapshot once the ad-hoc path streams results straight from
        // the configured device without persisting them, so memory stays bounded while preserving
        // the single cached block strategy.

        Console.WriteLine("Divisor cycles are ready");

        if (useByDivisor)
        {
            blockSize = 1;
        }

        _useDivisor = useDivisor;
        _useResidueMode = useResidue;
        _useByDivisorMode = useByDivisor;
        if (useBitTransform)
        {
            _transformP = &TransformPBit;
        }
        else if (useResidue)
        {
            _transformP = &TransformPAdd;
        }
        else
        {
            _transformP = &TransformPAddPrimes;
        }
        PResidue = new ThreadLocal<ModResidueTracker>(() => new ModResidueTracker(ResidueModel.Identity, initialNumber: currentP, initialized: true), trackAllValues: true);
        PrimeTesters = new ThreadLocal<PrimeTester>(() => new PrimeTester(), trackAllValues: true);
        // Note: --primes-device controls default device for library kernels; p primality remains CPU here.
        // Initialize per-thread p residue tracker (Identity model) at currentP
        if (!useDivisor && !useByDivisor)
        {
            MersenneTesters = new ThreadLocal<MersenneNumberTester>(() =>
            {
                // ProcessEightBitWindows windowed pow2 ladder is the default kernel.
                var tester = new MersenneNumberTester(
                                            useIncremental: !useLucas,
                                            useOrderCache: false,
                                            kernelType: kernelType,
                                            useModuloWorkaround: useModuloWorkaround,
                                            useOrder: useOrder,
                                            useGpuLucas: mersenneOnGpu,
                                            useGpuScan: mersenneOnGpu,
                                            useGpuOrder: orderOnGpu,
                                            useResidue: useResidue,
                                            maxK: residueKMax);
                if (!useLucas)
                {
                    Console.WriteLine("Warming up orders");
                    tester.WarmUpOrders(currentP, _orderWarmupLimitOverride ?? 5_000_000UL);
                }

                return tester;
            }, trackAllValues: true);
        }
        else if (useDivisor)
        {
            MersenneNumberDivisorGpuTester.BuildDivisorCandidates();

            _divisorTester = new MersenneNumberDivisorGpuTester();
        }
        else if (useByDivisor)
        {
            _byDivisorTester = mersenneOnGpu
                    ? new MersenneNumberDivisorByDivisorGpuTester()
                    : new MersenneNumberDivisorByDivisorCpuTester();
            _byDivisorTester.BatchSize = scanBatchSize;
            if (_byDivisorTester is MersenneNumberDivisorByDivisorCpuTester cpuByDivisorTester)
            {
                cpuByDivisorTester.DeltasDevice = byDivisorDeltasDevice;
                cpuByDivisorTester.MontgomeryDevice = byDivisorMontgomeryDevice;
            }
        }

        // Load RLE blacklist (optional)
        if (!string.IsNullOrEmpty(_rleBlacklistPath))
        {
            RleBlacklist.Load(_rleBlacklistPath!);
        }

        // Mod6 lookup turned out slower (see Mod6ComparisonBenchmarks), so keep `%` here for large candidates.
        if (currentP == InitialP && string.IsNullOrEmpty(filterFile))
        {
            // bool passedAllTests = IsEvenPerfectCandidate(InitialP, out bool searchedMersenne, out bool detailedCheck);
            // Skip the already processed range below 138 million
            while (currentP < 138_000_000UL && !Volatile.Read(ref _limitReached))
            {
                currentP = _transformP(currentP, ref remainder);
            }

            if (Volatile.Read(ref _limitReached))
            {
                return;
            }
        }

        Console.WriteLine("Initialization...");
        // Compose a results file name that encodes configuration (before opening file)
        var builtName = BuildResultsFileName(
                        useBitTransform,
                        threadCount,
                        blockSize,
                        kernelType,
                        useLucas,
                        useDivisor,
                        useByDivisor,
                        mersenneOnGpu,
                        useOrder,
                        useModuloWorkaround,
                        _useGcdFilter,
                        NttGpuMath.GpuTransformBackend,
                        gpuPrimeThreads,
                        sliceSize,
                        scanBatchSize,
                        _orderWarmupLimitOverride ?? 5_000_000UL,
                        NttGpuMath.ReductionMode,
                        mersenneOnGpu ? "gpu" : "cpu",
                        (GpuContextPool.ForceCpu ? "cpu" : "gpu"),
                        orderOnGpu ? "gpu" : "cpu");

        if (!string.IsNullOrEmpty(_resultsPrefix))
        {
            builtName = _resultsPrefix + "_" + builtName;
        }
        ResultsFileName = string.IsNullOrEmpty(_resultsDir) ? builtName : Path.Combine(_resultsDir!, builtName);
        var dir = Path.GetDirectoryName(ResultsFileName);
        if (!string.IsNullOrEmpty(dir))
        {
            Directory.CreateDirectory(dir!);
        }


        Dictionary<ulong, (bool DetailedCheck, bool PassedAllTests)>? previousResults = useByDivisor ? new Dictionary<ulong, (bool DetailedCheck, bool PassedAllTests)>() : null;
        _byDivisorPreviousResults = previousResults;

        if (File.Exists(ResultsFileName))
        {
            Console.WriteLine("Processing previous results...");
            LoadResultsFile(ResultsFileName, (p, detailedCheck, passedAllTests) =>
            {
                if (detailedCheck && passedAllTests)
                {
                    _primeCount++;
                }

                if (previousResults is not null)
                {
                    previousResults[p] = (detailedCheck, passedAllTests);
                }
            });
        }
        else
        {
            // TODO: Replace this File.WriteAllText call with the pooled TextFileWriter pipeline once the
            // scanner's output flush adopts the benchmarked buffered writer so we avoid allocating the
            // entire header string and opening the file twice per run.
            File.WriteAllText(ResultsFileName, $"p,searchedMersenne,detailedCheck,passedAllTests{Environment.NewLine}");
        }

        bool useFilter = !string.IsNullOrEmpty(filterFile);
        HashSet<ulong> filter = [];
        List<ulong> byDivisorCandidates = [];
        ulong maxP = 0UL;
        ulong[] localFilter = Array.Empty<ulong>();
        int filterCount = 0;
        if (useFilter)
        {
            Console.WriteLine("Loading filter...");
            if (_useByDivisorMode)
            {
                byDivisorCandidates = LoadByDivisorCandidates(filterFile);
            }
            else
            {
                // TODO: Rent this filter buffer from ArrayPool<ulong> and reuse it across reload batches so
                // we do not allocate fresh arrays while replaying large result filters.
                localFilter = new ulong[1024];
                filterCount = 0;
                LoadResultsFile(filterFile, (p, detailedCheck, passedAllTests) =>
                {
                    if (passedAllTests)
                    {
                        localFilter[filterCount++] = p;
                        if (p > maxP)
                        {
                            maxP = p;
                        }

                        if (filterCount == 1024)
                        {
                            filter.AddRange(localFilter[..filterCount]);
                            filterCount = 0;
                            Console.WriteLine($"Added {p}");
                        }
                    }
                });

                if (filterCount > 0)
                {
                    filter.AddRange(localFilter[..filterCount]);
                }
            }
        }

        if (_primeCount >= 2)
        {
            _primeFoundAfterInit = true;
        }

        _outputBuilder = StringBuilderPool.Rent();

        // Limit GPU concurrency only for prime checks (LL/NTT & GPU order scans).
        GpuPrimeWorkLimiter.SetLimit(gpuPrimeThreads);
        GpuSmallCycleKernelLimiter.SetLimit(gpuSmallCycleThreads);
        // Configure batch size for GPU primality sieve
        PrimeTester.GpuBatchSize = gpuPrimeBatch;

        Console.WriteLine("Starting scan...");

        if (_useByDivisorMode)
        {
            if (_byDivisorTester is null)
            {
                throw new InvalidOperationException("By-divisor tester has not been initialized.");
            }

            MersenneNumberDivisorByDivisorTester.Run(
                    byDivisorCandidates,
                    _byDivisorTester,
                    _byDivisorPreviousResults,
                    _byDivisorStartPrime,
                    static () => _lastCompositeP = true,
                    static () => _lastCompositeP = false,
                    PrintResult,
                    threadCount,
                    _primeTestLimit);
            FlushBuffer();
            StringBuilderPool.Return(_outputBuilder!);
            return;
        }

        if (!useDivisor && !useByDivisor)
        {
            threadCount = Math.Max(1, threadCount);
            _ = MersenneTesters.Value;
            if (!useLucas)
            {
                Console.WriteLine("Warming up orders...");
                MersenneTesters.Value!.WarmUpOrders(currentP, _orderWarmupLimitOverride ?? 5_000_000UL);
            }
        }

        _state = ((long)currentP << 3) | (long)remainder;
        Task[] tasks = new Task[threadCount];

        int taskIndex = 0;
        for (; taskIndex < threadCount; taskIndex++)
        {
            tasks[taskIndex] = Task.Run(() =>
            {
                int count, j;
                ulong p;
                bool passedAllTests, searchedMersenne, detailedCheck;
                // TODO: Rent this buffer from ArrayPool once the pooled prime-block allocator lands so worker
                // threads stop allocating fresh arrays for every reservation.
                ulong[] buffer = new ulong[blockSize];
                bool reachedMax = false;
                while (!Volatile.Read(ref _limitReached))
                {
                    count = ReserveBlock(buffer, blockSize);
                    if (count == 0)
                    {
                        break;
                    }

                    if (!useFilter)
                    {
                        for (j = 0; j < count && !Volatile.Read(ref _limitReached); j++)
                        {
                            p = buffer[j];
                            passedAllTests = IsEvenPerfectCandidate(p, divisorCyclesSearchLimit, out searchedMersenne, out detailedCheck);
                            PrintResult(p, searchedMersenne, detailedCheck, passedAllTests);
                        }
                    }
                    else
                    {
                        reachedMax = false;

                        for (j = 0; j < count; j++)
                        {
                            p = buffer[j];

                            if (Volatile.Read(ref _limitReached) && p > maxP)
                            {
                                break;
                            }

                            if (useFilter && !filter.Contains(p))
                            {
                                continue;
                            }

                            passedAllTests = IsEvenPerfectCandidate(p, divisorCyclesSearchLimit, out searchedMersenne, out detailedCheck);
                            PrintResult(p, searchedMersenne, detailedCheck, passedAllTests);

                            if (p == maxP)
                            {
                                reachedMax = true;
                            }
                        }

                        if (reachedMax)
                        {
                            Volatile.Write(ref _limitReached, true);
                            break;
                        }
                    }
                }
            });
        }

        Task.WaitAll(tasks);
        FlushBuffer();
        StringBuilderPool.Return(_outputBuilder!);
    }

    private static List<ulong> LoadByDivisorCandidates(string candidateFile)
    {
        List<ulong> candidates = [];
        using FileStream readStream = new(candidateFile, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
        using StreamReader reader = new(readStream);
        string? line;
        ReadOnlySpan<char> span;
        while ((line = reader.ReadLine()) is not null)
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            span = line.AsSpan().Trim();
            if (span.IsEmpty || span[0] == '#')
            {
                continue;
            }

            // TODO: Swap this char-by-char parser for the benchmarked Utf8Parser-based fast-path once
            // the CLI plumbing accepts spans to avoid redundant range checks while collecting primes.
            int index = 0;
            while (index < span.Length)
            {
                if (!char.IsDigit(span[index]))
                {
                    index++;
                    continue;
                }

                int start = index;
                index++;
                while (index < span.Length && char.IsDigit(span[index]))
                {
                    index++;
                }

                // TODO: Swap ulong.TryParse for the Utf8Parser-based fast path once the shared span helpers
                // arrive so by-divisor candidate loading avoids per-value string allocations.
                if (ulong.TryParse(span[start..index], NumberStyles.None, CultureInfo.InvariantCulture, out ulong parsed))
                {
                    candidates.Add(parsed);
                }
            }
        }

        return candidates;
    }

    private static unsafe void LoadResultsFile(string resultsFileName, Action<ulong, bool, bool> lineProcessorAction)
    {
        using FileStream readStream = new(resultsFileName, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
        using StreamReader reader = new(readStream);
        string? line;
        bool headerSkipped = false;
        ReadOnlySpan<char> span;
        int first = 0;
        int second = 0;
        int third = 0;
        ulong parsedP = 0UL;
        ReadOnlySpan<char> detailedSpan = default;
        ReadOnlySpan<char> passedAllTestsSpan = default;
        bool detailed = false;
        bool passedAllTests = false;
        while ((line = reader.ReadLine()) != null)
        {
            if (!headerSkipped)
            {
                headerSkipped = true;
                continue;
            }

            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            span = line.AsSpan();
            first = span.IndexOf(',');
            if (first < 0)
            {
                continue;
            }

            // TODO: Replace this ulong.Parse call with the Utf8Parser-based span fast-path we benchmarked so results
            // reload skips the slower string allocation and culture-aware parsing.
            parsedP = ulong.Parse(span[..first]);
            span = span[(first + 1)..];
            second = span.IndexOf(',');
            if (second < 0)
            {
                continue;
            }

            span = span[(second + 1)..];
            third = span.IndexOf(',');
            if (third < 0)
            {
                continue;
            }

            detailedSpan = span[..third];
            passedAllTestsSpan = span[(third + 1)..];

            // TODO: Replace these bool.TryParse calls with the span-based fast path once we expose the
            // Utf8Parser-powered helpers so results reload avoids per-line string allocations for booleans.
            if (bool.TryParse(detailedSpan, out detailed) && bool.TryParse(passedAllTestsSpan, out passedAllTests))
            {
                lineProcessorAction(parsedP, detailed, passedAllTests);
            }
        }
    }

    private static bool TryParsePrimeTestLimit(ReadOnlySpan<char> value, out TimeSpan result)
    {
        result = TimeSpan.Zero;
        if (value.IsEmpty)
        {
            return false;
        }

        ReadOnlySpan<char> magnitudeSpan = value;
        double magnitude;

        if (value.Length >= 2 && value.EndsWith("ms", StringComparison.OrdinalIgnoreCase))
        {
            magnitudeSpan = value[..^2];
            if (!double.TryParse(magnitudeSpan, NumberStyles.Float, CultureInfo.InvariantCulture, out magnitude))
            {
                return false;
            }

            result = TimeSpan.FromMilliseconds(magnitude);
            return true;
        }

        char suffix = char.ToLowerInvariant(value[^1]);
        if (suffix == 's' || suffix == 'm')
        {
            magnitudeSpan = value[..^1];
            if (!double.TryParse(magnitudeSpan, NumberStyles.Float, CultureInfo.InvariantCulture, out magnitude))
            {
                return false;
            }

            result = suffix == 's' ? TimeSpan.FromSeconds(magnitude) : TimeSpan.FromMinutes(magnitude);
            return true;
        }

        if (!double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out magnitude))
        {
            return false;
        }

        result = TimeSpan.FromMilliseconds(magnitude);
        return true;
    }

    private static void PrintHelp()
    {
        Console.WriteLine("Usage: EvenPerfectBitScanner [options]");
        Console.WriteLine();
        Console.WriteLine("Options:");
        Console.WriteLine("  --prime=<value>        starting exponent (p)");
        Console.WriteLine("  --increment=bit|add    exponent increment method");
        Console.WriteLine("  --threads=<value>      number of worker threads");
        Console.WriteLine("  --block-size=<value>   values processed per thread batch");
        Console.WriteLine("  --mersenne=pow2mod|incremental|lucas|residue|divisor|bydivisor  Mersenne test method");
        Console.WriteLine("  --residue-max-k=<value>  max k for residue Mersenne test (q = 2*p*k + 1)");
        Console.WriteLine("  --mersenne-device=cpu|gpu  Device for Mersenne method (default gpu)");
        Console.WriteLine("  --primes-device=cpu|gpu    Device for prime-scan kernels (default gpu)");
        Console.WriteLine("  --gpu-prime-batch=<n>      Batch size for GPU primality sieve (default 262144)");
        Console.WriteLine("  --order-device=cpu|gpu     Device for order computations (default gpu)");
        Console.WriteLine("  --ntt=reference|staged GPU NTT backend (default staged)");
        Console.WriteLine("  --mod-reduction=auto|uint128|mont64|barrett128  staged NTT reduction (default auto)");
        Console.WriteLine("  --gpu-prime-threads=<value>  max concurrent GPU prime checks (default 1)");
        Console.WriteLine("  --primes-with-smallcycles-threads=<value>  max concurrent pow2 kernels that preload small cycles (default 512)");
        Console.WriteLine("  --ll-slice=<value>     Lucas–Lehmer iterations per slice (default 32)");
        Console.WriteLine("  --gpu-scan-batch=<value>  GPU q-scan batch size (default 2_097_152)");
        Console.WriteLine("  --order-warmup-limit=<value>  Warm-up order candidates (default 5_000_000)");
        Console.WriteLine("  --rle-blacklist=<path>  enable RLE blacklist for p (hard filter up to --rle-hard-max)");
        Console.WriteLine("  --rle-hard-max=<p>      apply RLE blacklist only for p <= this (default ulong.MaxValue = no limit)");
        Console.WriteLine("  --rle-only-last7=true|false  apply RLE only when p % 10 == 7 (default true)");
        Console.WriteLine("  --zero-hard=<f>         hard reject if zero_fraction(p) > f (default off)");
        Console.WriteLine("  --zero-conj=<f>:<r>     hard reject if zero_fraction(p) > f AND max_zero_block >= r (default off)");
        Console.WriteLine("  --results-dir=<path>   directory for results file");
        Console.WriteLine("  --results-prefix=<text> prefix to prepend to results filename");
        Console.WriteLine("  --divisor-cycles=<path>       divisor cycles data file");
        Console.WriteLine("  --divisor-cycles-device=cpu|gpu  device for cycles generation (default gpu)");
        Console.WriteLine("  --divisor-cycles-batch=<value> batch size for cycles generation (default 512)");
        Console.WriteLine("  --divisor-cycles-continue  continue divisor cycles generation");
        Console.WriteLine("  --divisor-cycles-limit=<value> cycle search iterations when --mersenne=divisor");
        Console.WriteLine("  --prime-test-limit=<value>   Time limit for --mersenne=bydivisor prime checks (e.g. 5s, 500ms)");
        Console.WriteLine("  --bydivisor-deltas-device=cpu|gpu  Device for --mersenne=bydivisor residue batches (default: cpu)");
        Console.WriteLine("  --bydivisor-montgomery-device=cpu|gpu  Device for Montgomery data when using --mersenne=bydivisor (default: cpu)");
        Console.WriteLine("  --use-order            test primality via q order");
        Console.WriteLine("  --workaround-mod       avoid '%' operator on the GPU");
        // mod-automaton removed
        Console.WriteLine("  --filter-p=<path>      process only p from previous run results (required for --mersenne=bydivisor; primes must be strictly increasing)");
        Console.WriteLine("  --write-batch-size=<value> overwrite frequency of disk writes (default 100 lines)");
        Console.WriteLine("  --gcd-filter           enable early sieve based on GCD");
        Console.WriteLine("  --help, -help, --?, -?, /?   show this help message");
    }

    private static string BuildResultsFileName(bool bitInc, int threads, int block, GpuKernelType kernelType, bool useLucasFlag, bool useDivisorFlag, bool useByDivisorFlag, bool mersenneOnGpu, bool useOrder, bool useModWorkaround, bool useGcd, NttBackend nttBackend, int gpuPrimeThreads, int llSlice, int gpuScanBatch, ulong warmupLimit, ModReductionMode reduction, string mersenneDevice, string primesDevice, string orderDevice)
    {
        string inc = bitInc ? "bit" : "add";
        string mers = useDivisorFlag
            ? "divisor"
            : (useLucasFlag
                ? "lucas"
                : (useByDivisorFlag
                    ? "bydivisor"
                    : (kernelType == GpuKernelType.Pow2Mod ? "pow2mod" : "incremental")));
        string ntt = nttBackend == NttBackend.Staged ? "staged" : "reference";
        string red = reduction switch { ModReductionMode.Mont64 => "mont64", ModReductionMode.Barrett128 => "barrett128", ModReductionMode.GpuUInt128 => "uint128", _ => "auto" };
        string order = useOrder ? "order-on" : "order-off";
        string gcd = useGcd ? "gcd-on" : "gcd-off";
        string work = useModWorkaround ? "modfix-on" : "modfix-off";
        // TODO: Replace this string interpolation with the pooled ValueStringBuilder pipeline from the
        // results-writer benchmarks so filename generation reuses the zero-allocation formatter once
        // we consolidate output paths.
        return $"even_perfect_bit_scan_inc-{inc}_thr-{threads}_blk-{block}_mers-{mers}_mersdev-{mersenneDevice}_ntt-{ntt}_red-{red}_primesdev-{primesDevice}_{order}_orderdev-{orderDevice}_{gcd}_{work}_gputh-{gpuPrimeThreads}_llslice-{llSlice}_scanb-{gpuScanBatch}_warm-{warmupLimit}.csv";
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe int ReserveBlock(ulong[] buffer, int blockSize)
    {
        Span<ulong> bufferSpan = new(buffer);
        ulong p, remainder;
        long state, newState, original;
        int count;
        while (true)
        {
            state = Volatile.Read(ref _state);
            p = (ulong)state >> 3;
            remainder = ((ulong)state) & 7UL;

            count = 0;
            while (count < blockSize && !Volatile.Read(ref _limitReached))
            {
                bufferSpan[count++] = p;
                // TODO: Inline the fast-path transform here once we collapse the delegate* indirection so
                // block reservations stop paying the extra jump highlighted in the prime-stepping benchmarks.
                p = _transformP(p, ref remainder);
            }

            if (count == 0)
            {
                return 0;
            }

            newState = ((long)p << 3) | (long)remainder;
            original = Interlocked.CompareExchange(ref _state, newState, state);
            if (original == state)
            {
                return count;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool TryAcquireConsoleSlot(bool primeCandidate, out bool primeFlag)
    {
        int consoleInterval = PerfectNumberConstants.ConsoleInterval;
        int counterValue = Interlocked.Increment(ref _consoleCounter);
        if (counterValue >= consoleInterval)
        {
            int previous = Interlocked.Exchange(ref _consoleCounter, 0);
            if (previous >= consoleInterval)
            {
                primeFlag = primeCandidate;
                return true;
            }
        }

        primeFlag = false;
        return false;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static StringBuilder? AppendToOutputBuffer(StringBuilder source)
    {
        StringBuilder? builderToFlush = null;

        lock (Sync)
        {
            StringBuilder outputBuilder = _outputBuilder!;
            _ = outputBuilder.Append(source);

            int nextIndex = _writeIndex + 1;
            if (nextIndex >= _writeBatchSize)
            {
                builderToFlush = outputBuilder;
                _outputBuilder = StringBuilderPool.Rent();
                _writeIndex = 0;
            }
            else
            {
                _writeIndex = nextIndex;
            }
        }

        return builderToFlush;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void PrintResult(ulong currentP, bool searchedMersenne, bool detailedCheck, bool passedAllTests)
    {
        if (passedAllTests)
        {
            int newCount = Interlocked.Increment(ref _primeCount);
            if (newCount >= 2)
            {
                Volatile.Write(ref _primeFoundAfterInit, true);
            }
        }

        bool lastWasComposite = _lastCompositeP;
        bool primeCandidate = passedAllTests && !lastWasComposite && Volatile.Read(ref _primeFoundAfterInit);

        // TODO: Switch this StringBuilder-based formatter to a span-friendly Utf8Formatter pipeline so
        // console/file output avoids intermediate builders in the hot logging loop.
        StringBuilder localBuilder = StringBuilderPool.Rent();
        _ = localBuilder
                        .Append(currentP).Append(',')
                        .Append(searchedMersenne).Append(',')
                        .Append(detailedCheck).Append(',')
                        .Append(passedAllTests).Append('\n');

        bool printToConsole = TryAcquireConsoleSlot(primeCandidate, out bool primeFlag);
        StringBuilder? builderToFlush = AppendToOutputBuffer(localBuilder);

        if (builderToFlush is not null)
        {
            FlushBuffer(builderToFlush);
        }

        if (printToConsole)
        {
            if (primeFlag)
            {
                _ = localBuilder
                                .Remove(localBuilder.Length - 1, 1)
                                .Append(PrimeFoundSuffix)
                                .Append('\n');
            }

            Console.WriteLine(localBuilder.Remove(localBuilder.Length - 1, 1).ToString());
        }

        localBuilder.Clear();
        StringBuilderPool.Return(localBuilder);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void FlushBuffer()
    {
        StringBuilder? builderToFlush = null;

        lock (Sync)
        {
            if (_writeIndex == 0 || _outputBuilder is null || _outputBuilder.Length == 0)
            {
                return;
            }

            builderToFlush = _outputBuilder;
            _outputBuilder = StringBuilderPool.Rent();
            _writeIndex = 0;
        }

        FlushBuffer(builderToFlush!);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void FlushBuffer(StringBuilder builderToFlush)
    {
        try
        {
            lock (FileWriteSync)
            {
                // TODO: Replace these per-flush FileStream/StreamWriter allocations with a pooled TextFileWriter-style
                // helper so batched result writes keep the hot path on the benchmarked persistent-handle pipeline.
                using FileStream stream = new(ResultsFileName, FileMode.Append, FileAccess.Write, FileShare.Read);
                using StreamWriter writer = new(stream) { AutoFlush = false };
                writer.Write(builderToFlush);
                writer.Flush();
            }
        }
        finally
        {
            builderToFlush.Clear();
            StringBuilderPool.Return(builderToFlush);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static ulong CountOnes(ulong value)
    {
        // TODO: Remove this wrapper once callers can invoke BitOperations.PopCount directly so the hot bit-stat path
        // avoids the extra call layer.
        return (ulong)BitOperations.PopCount(value);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong GetNextAddDiff(ulong remainder)
    {
        return remainder switch
        {
            0UL => 1UL,
            1UL => 4UL,
            2UL => 3UL,
            3UL => 2UL,
            4UL => 1UL,
            _ => 2UL,
        };
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static ulong TransformPBit(ulong value, ref ulong remainder)
    {
        ulong original = value;
        if (value > (ulong.MaxValue >> 1))
        {
            Volatile.Write(ref _limitReached, true);
            return value;
        }

        ulong next = (value << 1) | 1UL;
        remainder = (remainder << 1) + 1UL;
        // Mod6 lookup loses to `%` in the benches; stick with subtraction + modulo for correctness and speed.
        while (remainder >= 6UL)
        {
            remainder -= 6UL;
        }

        value = remainder switch
        {
            0UL => 1UL,
            1UL => 0UL,
            2UL => 5UL,
            3UL => 4UL,
            4UL => 3UL,
            _ => 2UL,
        }; // 'value' now holds diff

        if (next > ulong.MaxValue - value)
        {
            Volatile.Write(ref _limitReached, true);
            return original;
        }

        remainder += value;
        // `% 6` remains faster per Mod6ComparisonBenchmarks; keep this modulo fold and continue using subtraction.
        while (remainder >= 6UL)
        {
            remainder -= 6UL;
        }

        return next + value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static ulong TransformPAdd(ulong value, ref ulong remainder)
    {
        ulong next = value;
        ulong diff = GetNextAddDiff(remainder);

        if (next > ulong.MaxValue - diff)
        {
            Volatile.Write(ref _limitReached, true);
            return next;
        }

        remainder += diff;
        // Retain direct modulo because the Mod6 helper underperforms on 64-bit operands.
        while (remainder >= 6UL)
        {
            remainder -= 6UL;
        }

        return next + diff;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static ulong TransformPAddPrimes(ulong value, ref ulong remainder)
    {
        ulong originalRemainder = remainder;
        ulong addRemainder = remainder;
        ulong candidate = value;
        ulong primeCandidate = value;
        bool advanceAdd = true;
        bool advancePrime = true;
        ulong diff = 0UL;
        ulong nextPrime = 0UL;

        while (true)
        {
            if (advanceAdd)
            {
                diff = GetNextAddDiff(addRemainder);
                if (candidate > ulong.MaxValue - diff)
                {
                    Volatile.Write(ref _limitReached, true);
                    remainder = originalRemainder;
                    return candidate;
                }

                candidate += diff;
                addRemainder += diff;
                // Skip the Mod6 lookup: benchmarked `%` stays ahead for these prime increments.
                // benchmarked fastest remainder updates instead of looping subtraction.
                while (addRemainder >= 6UL)
                {
                    addRemainder -= 6UL;
                }
            }

            if (advancePrime)
            {
                try
                {
                    nextPrime = PrimeIterator.Next(in primeCandidate);
                }
                catch (InvalidOperationException)
                {
                    Volatile.Write(ref _limitReached, true);
                    remainder = originalRemainder;
                    return value;
                }

                if (nextPrime <= primeCandidate)
                {
                    Volatile.Write(ref _limitReached, true);
                    remainder = originalRemainder;
                    return value;
                }

                primeCandidate = nextPrime;
            }

            if (candidate == primeCandidate)
            {
                remainder = addRemainder;
                return candidate;
            }

            advanceAdd = candidate < primeCandidate;
            advancePrime = candidate > primeCandidate;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool IsEvenPerfectCandidate(ulong p, ulong divisorCyclesSearchLimit) =>
            // TODO: Remove this pass-through once callers can provide the full out parameters
            // directly; shaving this hop keeps the hot candidate filter on the shortest path.
            IsEvenPerfectCandidate(p, divisorCyclesSearchLimit, out _, out _);

    internal static bool IsEvenPerfectCandidate(ulong p, ulong divisorCyclesSearchLimit, out bool searchedMersenne, out bool detailedCheck)
    {
        searchedMersenne = false;
        detailedCheck = false;
        _lastCompositeP = false;

        if (_useGcdFilter && IsCompositeByGcd(p))
        {
            _lastCompositeP = true;
            return false;
        }

        // Optional: RLE blacklist and binary-threshold filters on p (safe only when configured)
        if (p <= _rleHardMaxP)
        {
            if (!_rleOnlyLast7 || p.Mod10() == 7UL)
            {
                if (RleBlacklist.IsLoaded() && RleBlacklist.Matches(p))
                {
                    return false;
                }
            }
        }

        if (_zeroFracHard >= 0 || (_zeroFracConj >= 0 && _maxZeroConj >= 0))
        {
            double zf;
            int bitLength = 0;
            int zeroCountValue = 0;
            int maxZeroBlockValue = 0;
            ComputeBitStats(p, out bitLength, out zeroCountValue, out maxZeroBlockValue);
            zf = (double)zeroCountValue / bitLength;
            if (_zeroFracHard >= 0)
            {
                if (zf > _zeroFracHard)
                {
                    return false;
                }
            }

            if (_zeroFracConj >= 0 && _maxZeroConj >= 0)
            {
                if (zf > _zeroFracConj && maxZeroBlockValue >= _maxZeroConj)
                {
                    return false;
                }
            }
        }

        // Fast residue-based composite check for p using small primes
        if (!_useDivisor && !_useByDivisorMode && IsCompositeByResidues(p))
        {
            searchedMersenne = false;
            detailedCheck = false;
            _lastCompositeP = true;
            return false;
        }

        // If primes-device=gpu, route p primality through GPU-assisted sieve with deterministic MR validation.
        if (!_useByDivisorMode)
        {
            if (!(GpuContextPool.ForceCpu
                                            ? PrimeTesters.Value!.IsPrime(p, CancellationToken.None)
                                            : PrimeTesters.Value!.IsPrimeGpu(p, CancellationToken.None)))
            {
                _lastCompositeP = true;
                return false;
            }
        }

        searchedMersenne = true;
        if (_useByDivisorMode)
        {
            // Windowed pow2mod kernel handles by-divisor scans.
            return _byDivisorTester!.IsPrime(p, out detailedCheck);
        }

        detailedCheck = MersenneTesters.Value!.IsMersennePrime(p);
        return detailedCheck;
    }

    // Use ModResidueTracker with a small set of primes to pre-filter composite p.
    private static bool IsCompositeByResidues(ulong p)
    {
        var tracker = PResidue.Value!;
        // TODO: Integrate the divisor-cycle cache here so the small-prime sweep reuses precomputed remainders instead of
        // running MergeOrAppend for every candidate and missing the cycle-accelerated early exits.
        tracker.BeginMerge(p);
        // Use the small prime list from PerfectNumbers.Core to the extent of sqrt(p)
        var primes = PrimesGenerator.SmallPrimes;
        var primesPow2 = PrimesGenerator.SmallPrimesPow2;
        int len = primes.Length;
        int primeIndex = 0;
        bool divisible = false;
        for (; primeIndex < len; primeIndex++)
        {
            if (primesPow2[primeIndex] > p)
            {
                break;
            }

            if (tracker.MergeOrAppend(p, primes[primeIndex], out divisible) && divisible)
            {
                return true;
            }
        }

        return false;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ComputeBitStats(ulong value, out int bitLen, out int zeroCount, out int maxZeroBlock)
    {
        bitLen = 64 - int.CreateChecked(ulong.LeadingZeroCount(value));
        zeroCount = 0;
        maxZeroBlock = 0;
        if (bitLen <= 0)
        {
            return;
        }

        int msbIndex = (bitLen - 1) >> 3;                 // 0..7
        int bitsInTopByte = ((bitLen - 1) & 7) + 1;        // 1..8
        int currentRun = 0;
        byte inspectedByte = 0;
        int zeroCountInByte = 0;
        int prefixZeros = 0;
        int suffixZeros = 0;
        int maxZeroRunInByte = 0;
        int candidate = 0;
        int byteIndex = msbIndex;

        for (; byteIndex >= 0; byteIndex--)
        {
            inspectedByte = (byte)(value >> (byteIndex * 8));
            if (byteIndex == msbIndex && bitsInTopByte < 8)
            {
                // Mask off leading unused bits by setting them to 1 (ignore)
                inspectedByte |= (byte)(0xFF << bitsInTopByte);
            }

            // TODO: Replace this byte-by-byte scan with the lookup-table based statistics collector validated in the
            // BitStats benchmarks so zero runs leverage cached results instead of recomputing per bit.

            zeroCountInByte = ByteZeroCount[inspectedByte];
            zeroCount += zeroCountInByte;

            if (zeroCountInByte == 8)
            {
                currentRun += 8;
                if (currentRun > maxZeroBlock)
                {
                    maxZeroBlock = currentRun;
                }

                continue;
            }

            prefixZeros = BytePrefixZero[inspectedByte];
            suffixZeros = ByteSuffixZero[inspectedByte];
            maxZeroRunInByte = ByteMaxZeroRun[inspectedByte];

            candidate = currentRun + prefixZeros;
            if (candidate > maxZeroBlock)
            {
                maxZeroBlock = candidate;
            }

            if (maxZeroRunInByte > maxZeroBlock)
            {
                maxZeroBlock = maxZeroRunInByte;
            }

            currentRun = suffixZeros;
        }

        if (currentRun > maxZeroBlock)
        {
            maxZeroBlock = currentRun;
        }
    }

    // LUTs for fast per-byte zero stats (MSB-first in each byte)
    private static readonly byte[] ByteZeroCount = new byte[256];
    private static readonly byte[] BytePrefixZero = new byte[256];
    private static readonly byte[] ByteSuffixZero = new byte[256];
    private static readonly byte[] ByteMaxZeroRun = new byte[256];

    private static void WarmPow2ModKernels(bool useGpuOrder, bool warmOrderKernel)
    {
        using var warmupLease = GpuKernelPool.GetKernel(useGpuOrder);
        var warmupScope = warmupLease.EnterExecutionScope();
        var accelerator = warmupLease.Accelerator;
        _ = GpuKernelPool.EnsureSmallPrimesOnDevice(accelerator);
        try
        {
            _ = warmupLease.Pow2ModWindowedKernel;
            if (warmOrderKernel)
            {
                _ = warmupLease.Pow2ModWindowedOrderKernel;
            }
        }
        finally
        {
            warmupScope.Dispose();
        }
    }

    static Program()
    {
        int valueIndex = 0;
        int zeros = 0;
        int pref = 0;
        int suff = 0;
        int maxIn = 0;
        int run = 0;
        int bit = 0;
        bool isZero = false;
        int boundaryCounter = 0;
        int boundaryBitIndex = 0;
        for (; valueIndex < 256; valueIndex++)
        {
            zeros = 0;
            pref = 0;
            suff = 0;
            maxIn = 0;
            run = 0;

            // MSB-first within byte: bit 7 .. bit 0
            for (bit = 7; bit >= 0; bit--)
            {
                isZero = ((valueIndex >> bit) & 1) == 0;
                if (isZero)
                {
                    zeros++;
                    run++;
                    if (run > maxIn)
                    {
                        maxIn = run;
                    }
                }
                else
                {
                    run = 0;
                }

                if (bit == 7)
                {
                    // leading zeros (prefix)
                    boundaryCounter = 0;
                    for (boundaryBitIndex = 7; boundaryBitIndex >= 0; boundaryBitIndex--)
                    {
                        if (((valueIndex >> boundaryBitIndex) & 1) == 0)
                        {
                            boundaryCounter++;
                        }
                        else
                        {
                            break;
                        }
                    }
                    pref = boundaryCounter;
                }

                if (bit == 0)
                {
                    // trailing zeros (suffix)
                    // Reusing boundaryCounter to count trailing zeros after prefix detection above.
                    boundaryCounter = 0;
                    for (boundaryBitIndex = 0; boundaryBitIndex < 8; boundaryBitIndex++)
                    {
                        if (((valueIndex >> boundaryBitIndex) & 1) == 0)
                        {
                            boundaryCounter++;
                        }
                        else
                        {
                            break;
                        }
                    }
                    suff = boundaryCounter;
                }
            }

            ByteZeroCount[valueIndex] = (byte)zeros;
            BytePrefixZero[valueIndex] = (byte)pref;
            ByteSuffixZero[valueIndex] = (byte)suff;
            ByteMaxZeroRun[valueIndex] = (byte)maxIn;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool IsCompositeByGcd(ulong p)
    {
        if (p < 2)
        {
            return true;
        }

        ulong m = (ulong)BitOperations.Log2(p);
        return BinaryGcd(p, m) > 1UL;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong BinaryGcd(ulong a, ulong b)
    {
        if (a == 0UL)
        {
            return b;
        }

        if (b == 0UL)
        {
            return a;
        }

        int shift = BitOperations.TrailingZeroCount(a | b);
        a >>= BitOperations.TrailingZeroCount(a);

        do
        {
            b >>= BitOperations.TrailingZeroCount(b);

            if (a > b)
            {
                (a, b) = (b, a);
            }

            b -= a;
        }
        while (b != 0UL);

        return a << shift;
    }
}

