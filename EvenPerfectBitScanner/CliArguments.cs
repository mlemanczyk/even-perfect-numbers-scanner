using System.Globalization;
using System.Numerics;
using PerfectNumbers.Core;
using PerfectNumbers.Core.Gpu;

namespace EvenPerfectBitScanner;

internal readonly struct CliArguments
{
    private const string DefaultCyclesPath = "divisor_cycles.bin";

    internal readonly ulong StartPrime;
    internal readonly bool StartPrimeProvided;
    internal readonly UInt128 MaxPrimeLimit;
    internal readonly bool MaxPrimeConfigured;
    internal readonly int ThreadCount;
    internal readonly int BlockSize;
    internal readonly int GpuPrimeThreads;
    internal readonly int GpuPrimeBatch;
    internal readonly GpuKernelType KernelType;
    internal readonly bool UseOrder;
    internal readonly bool ShowHelp;
    internal readonly bool UseBitTransform;
    internal readonly bool UseLucas;
    internal readonly bool UseResidue;
    internal readonly bool UseDivisor;
    internal readonly bool UseByDivisor;
    internal readonly bool UseGcdFilter;
    internal readonly bool TestMode;
    internal readonly bool UseGpuCycles;
    internal readonly bool UseMersenneOnGpu;
    internal readonly bool UseOrderOnGpu;
    internal readonly int ScanBatchSize;
    internal readonly int SliceSize;
    internal readonly ulong ResidueKMax;
    internal readonly string FilterFile;
    internal readonly bool ForcePrimeKernelsOnCpu;
    internal readonly ModReductionMode ModReductionMode;
    internal readonly NttBackend NttBackend;
    internal readonly string? ResultsDirectory;
    internal readonly string? ResultsPrefix;
    internal readonly int WriteBatchSize;
    internal readonly string CyclesPath;
    internal readonly bool ContinueCyclesGeneration;
    internal readonly int CyclesBatchSize;
    internal readonly ulong DivisorCyclesSearchLimit;
    internal readonly ulong? OrderWarmupLimitOverride;
    internal readonly string? RleBlacklistPath;
    internal readonly ulong RleHardMaxP;
    internal readonly bool RleOnlyLast7;
    internal readonly double ZeroFractionHard;
    internal readonly double ZeroFractionConjecture;
    internal readonly int MaxZeroConjecture;

    private CliArguments(
            ulong startPrime,
            bool startPrimeProvided,
            UInt128 maxPrimeLimit,
            bool maxPrimeConfigured,
            int threadCount,
            int blockSize,
            int gpuPrimeThreads,
            int gpuPrimeBatch,
            GpuKernelType kernelType,
            bool useOrder,
            bool showHelp,
            bool useBitTransform,
            bool useLucas,
            bool useResidue,
            bool useDivisor,
            bool useByDivisor,
            bool useGcdFilter,
            bool testMode,
            bool useGpuCycles,
            bool useMersenneOnGpu,
            bool useOrderOnGpu,
            int scanBatchSize,
            int sliceSize,
            ulong residueKMax,
            string filterFile,
            bool forcePrimeKernelsOnCpu,
            ModReductionMode modReductionMode,
            NttBackend nttBackend,
            string? resultsDirectory,
            string? resultsPrefix,
            int writeBatchSize,
            string cyclesPath,
            bool continueCyclesGeneration,
            int cyclesBatchSize,
            ulong divisorCyclesSearchLimit,
            ulong? orderWarmupLimitOverride,
            string? rleBlacklistPath,
            ulong rleHardMaxP,
            bool rleOnlyLast7,
            double zeroFractionHard,
            double zeroFractionConjecture,
            int maxZeroConjecture)
    {
        StartPrime = startPrime;
        StartPrimeProvided = startPrimeProvided;
        MaxPrimeLimit = maxPrimeLimit;
        MaxPrimeConfigured = maxPrimeConfigured;
        ThreadCount = threadCount;
        BlockSize = blockSize;
        GpuPrimeThreads = gpuPrimeThreads;
        GpuPrimeBatch = gpuPrimeBatch;
        KernelType = kernelType;
        UseOrder = useOrder;
        ShowHelp = showHelp;
        UseBitTransform = useBitTransform;
        UseLucas = useLucas;
        UseResidue = useResidue;
        UseDivisor = useDivisor;
        UseByDivisor = useByDivisor;
        UseGcdFilter = useGcdFilter;
        TestMode = testMode;
        UseGpuCycles = useGpuCycles;
        UseMersenneOnGpu = useMersenneOnGpu;
        UseOrderOnGpu = useOrderOnGpu;
        ScanBatchSize = scanBatchSize;
        SliceSize = sliceSize;
        ResidueKMax = residueKMax;
        FilterFile = filterFile;
        ForcePrimeKernelsOnCpu = forcePrimeKernelsOnCpu;
        ModReductionMode = modReductionMode;
        NttBackend = nttBackend;
        ResultsDirectory = resultsDirectory;
        ResultsPrefix = resultsPrefix;
        WriteBatchSize = writeBatchSize;
        CyclesPath = cyclesPath;
        ContinueCyclesGeneration = continueCyclesGeneration;
        CyclesBatchSize = cyclesBatchSize;
        DivisorCyclesSearchLimit = divisorCyclesSearchLimit;
        OrderWarmupLimitOverride = orderWarmupLimitOverride;
        RleBlacklistPath = rleBlacklistPath;
        RleHardMaxP = rleHardMaxP;
        RleOnlyLast7 = rleOnlyLast7;
        ZeroFractionHard = zeroFractionHard;
        ZeroFractionConjecture = zeroFractionConjecture;
        MaxZeroConjecture = maxZeroConjecture;
    }

    internal bool UseFilter => !TestMode && !string.IsNullOrEmpty(FilterFile);

    internal static CliArguments Parse(string[] args)
    {
        ulong startPrime = PerfectNumberConstants.BiggestKnownEvenPerfectP;
        bool startPrimeProvided = false;
        UInt128 maxPrimeLimit = UInt128.MaxValue;
        bool maxPrimeConfigured = false;
        int threadCount = Environment.ProcessorCount;
        int blockSize = 1;
        int gpuPrimeThreads = 1;
        int gpuPrimeBatch = 262_144;
        GpuKernelType kernelType = GpuKernelType.Incremental;
        bool useOrder = false;
        bool showHelp = false;
        bool useBitTransform = false;
        bool useLucas = false;
        bool useResidue = false;
        bool useDivisor = false;
        bool useByDivisor = false;
        bool useGcdFilter = false;
        bool testMode = false;
        bool useGpuCycles = true;
        bool useMersenneOnGpu = true;
        bool useOrderOnGpu = true;
        int scanBatchSize = 2_097_152;
        int sliceSize = 32;
        ulong residueKMax = 5_000_000UL;
        string filterFile = string.Empty;
        bool forcePrimeKernelsOnCpu = false;
        ModReductionMode modReductionMode = ModReductionMode.Auto;
        NttBackend nttBackend = NttBackend.Reference;
        string? resultsDirectory = null;
        string? resultsPrefix = null;
        int writeBatchSize = CalculationResultHandler.DefaultWriteBatchSize;
        string cyclesPath = DefaultCyclesPath;
        bool continueCyclesGeneration = false;
        int cyclesBatchSize = 512;
        ulong divisorCyclesSearchLimit = PerfectNumberConstants.ExtraDivisorCycleSearchLimit;
        ulong? orderWarmupLimitOverride = null;
        string? rleBlacklistPath = null;
        ulong rleHardMaxP = ulong.MaxValue;
        bool rleOnlyLast7 = true;
        double zeroFractionHard = -1.0;
        double zeroFractionConjecture = -1.0;
        int maxZeroConjecture = -1;

        foreach (string argument in args)
        {
            if (argument.Equals("--?", StringComparison.OrdinalIgnoreCase) ||
                argument.Equals("-?", StringComparison.OrdinalIgnoreCase) ||
                argument.Equals("--help", StringComparison.OrdinalIgnoreCase) ||
                argument.Equals("-help", StringComparison.OrdinalIgnoreCase) ||
                argument.Equals("/?", StringComparison.OrdinalIgnoreCase))
            {
                showHelp = true;
                break;
            }

            if (argument.StartsWith("--prime=", StringComparison.OrdinalIgnoreCase))
            {
                startPrime = Utf8CliParser.ParseUInt64(argument.AsSpan(argument.IndexOf('=') + 1));
                startPrimeProvided = true;
                continue;
            }

            if (argument.StartsWith("--max-prime=", StringComparison.OrdinalIgnoreCase))
            {
                ReadOnlySpan<char> value = argument.AsSpan(argument.IndexOf('=') + 1);
                if (UInt128.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out UInt128 parsedMaxPrime))
                {
                    maxPrimeLimit = parsedMaxPrime;
                    maxPrimeConfigured = true;
                }
                else
                {
                    Console.WriteLine("Invalid value for --max-prime.");
                    return default;
                }

                continue;
            }

            if (argument.Equals("--increment=bit", StringComparison.OrdinalIgnoreCase))
            {
                useBitTransform = true;
                continue;
            }

            if (argument.Equals("--increment=add", StringComparison.OrdinalIgnoreCase))
            {
                useBitTransform = false;
                continue;
            }

            if (argument.StartsWith("--threads=", StringComparison.OrdinalIgnoreCase))
            {
                threadCount = Math.Max(1, Utf8CliParser.ParseInt32(argument.AsSpan(argument.IndexOf('=') + 1)));
                continue;
            }

            if (argument.StartsWith("--mersenne=", StringComparison.OrdinalIgnoreCase))
            {
                ReadOnlySpan<char> value = argument.AsSpan(argument.IndexOf('=') + 1);
                if (value.Equals("pow2mod", StringComparison.OrdinalIgnoreCase))
                {
                    kernelType = GpuKernelType.Pow2Mod;
                }
                else if (value.Equals("lucas", StringComparison.OrdinalIgnoreCase))
                {
                    useLucas = true;
                }
                else if (value.Equals("residue", StringComparison.OrdinalIgnoreCase))
                {
                    useResidue = true;
                    useLucas = false;
                }
                else if (value.Equals("divisor", StringComparison.OrdinalIgnoreCase))
                {
                    useDivisor = true;
                    useLucas = false;
                    useResidue = false;
                }
                else if (value.Equals("bydivisor", StringComparison.OrdinalIgnoreCase))
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

                continue;
            }

            if (argument.StartsWith("--divisor-cycles-limit=", StringComparison.OrdinalIgnoreCase))
            {
                if (Utf8CliParser.TryParseUInt64(argument.AsSpan(argument.IndexOf('=') + 1), out ulong parsedLimit))
                {
                    divisorCyclesSearchLimit = parsedLimit;
                }

                continue;
            }

            if (argument.StartsWith("--residue-max-k=", StringComparison.OrdinalIgnoreCase))
            {
                if (Utf8CliParser.TryParseUInt64(argument.AsSpan(argument.IndexOf('=') + 1), out ulong parsedResidueMax))
                {
                    residueKMax = parsedResidueMax;
                }

                continue;
            }

            if (argument.StartsWith("--mersenne-device=", StringComparison.OrdinalIgnoreCase))
            {
                useMersenneOnGpu = !argument.AsSpan(argument.IndexOf('=') + 1).Equals("cpu", StringComparison.OrdinalIgnoreCase);
                continue;
            }

            if (argument.Equals("--test", StringComparison.OrdinalIgnoreCase))
            {
                testMode = true;
                continue;
            }

            if (argument.Equals("--use-order", StringComparison.OrdinalIgnoreCase))
            {
                useOrder = true;
                continue;
            }

            if (argument.StartsWith("--order-warmup-limit=", StringComparison.OrdinalIgnoreCase))
            {
                if (Utf8CliParser.TryParseUInt64(argument.AsSpan(argument.IndexOf('=') + 1), out ulong parsedLimit))
                {
                    orderWarmupLimitOverride = parsedLimit;
                }

                continue;
            }

            if (argument.Equals("--gcd-filter", StringComparison.OrdinalIgnoreCase))
            {
                useGcdFilter = true;
                continue;
            }

            if (argument.StartsWith("--ntt=", StringComparison.OrdinalIgnoreCase))
            {
                ReadOnlySpan<char> value = argument.AsSpan(argument.IndexOf('=') + 1);
                nttBackend = value.Equals("staged", StringComparison.OrdinalIgnoreCase) ? NttBackend.Staged : NttBackend.Reference;
                continue;
            }

            if (argument.StartsWith("--mod-reduction=", StringComparison.OrdinalIgnoreCase))
            {
                ReadOnlySpan<char> value = argument.AsSpan(argument.IndexOf('=') + 1);
                if (value.Equals("mont64", StringComparison.OrdinalIgnoreCase))
                {
                    modReductionMode = ModReductionMode.Mont64;
                }
                else if (value.Equals("barrett128", StringComparison.OrdinalIgnoreCase))
                {
                    modReductionMode = ModReductionMode.Barrett128;
                }
                else if (value.Equals("uint128", StringComparison.OrdinalIgnoreCase))
                {
                    modReductionMode = ModReductionMode.GpuUInt128;
                }
                else
                {
                    modReductionMode = ModReductionMode.Auto;
                }

                continue;
            }

            if (argument.StartsWith("--primes-device=", StringComparison.OrdinalIgnoreCase))
            {
                forcePrimeKernelsOnCpu = argument.AsSpan(argument.IndexOf('=') + 1).Equals("cpu", StringComparison.OrdinalIgnoreCase);
                continue;
            }

            if (argument.StartsWith("--order-device=", StringComparison.OrdinalIgnoreCase))
            {
                useOrderOnGpu = !argument.AsSpan(argument.IndexOf('=') + 1).Equals("cpu", StringComparison.OrdinalIgnoreCase);
                continue;
            }

            if (argument.StartsWith("--rle-blacklist=", StringComparison.OrdinalIgnoreCase))
            {
                rleBlacklistPath = argument[(argument.IndexOf('=') + 1)..];
                continue;
            }

            if (argument.StartsWith("--rle-hard-max=", StringComparison.OrdinalIgnoreCase))
            {
                if (Utf8CliParser.TryParseUInt64(argument.AsSpan(argument.IndexOf('=') + 1), out ulong parsedMax))
                {
                    rleHardMaxP = parsedMax;
                }

                continue;
            }

            if (argument.StartsWith("--rle-only-last7=", StringComparison.OrdinalIgnoreCase))
            {
                ReadOnlySpan<char> value = argument.AsSpan(argument.IndexOf('=') + 1);
                rleOnlyLast7 = !value.Equals("false", StringComparison.OrdinalIgnoreCase) && !value.Equals("0", StringComparison.OrdinalIgnoreCase);
                continue;
            }

            if (argument.StartsWith("--zero-hard=", StringComparison.OrdinalIgnoreCase))
            {
                if (Utf8CliParser.TryParseDouble(argument.AsSpan(argument.IndexOf('=') + 1), out double zeroFraction))
                {
                    zeroFractionHard = zeroFraction;
                }

                continue;
            }

            if (argument.StartsWith("--zero-conj=", StringComparison.OrdinalIgnoreCase))
            {
                ReadOnlySpan<char> value = argument.AsSpan(argument.IndexOf('=') + 1);
                int colon = value.IndexOf(':');
                if (colon > 0)
                {
                    if (Utf8CliParser.TryParseDouble(value[..colon], out double zeroFrac) &&
                        Utf8CliParser.TryParseInt32(value[(colon + 1)..], out int maxZero))
                    {
                        zeroFractionConjecture = zeroFrac;
                        maxZeroConjecture = maxZero;
                    }
                }

                continue;
            }

            if (argument.StartsWith("--lucas=", StringComparison.OrdinalIgnoreCase))
            {
                useMersenneOnGpu = !argument.AsSpan(argument.IndexOf('=') + 1).Equals("cpu", StringComparison.OrdinalIgnoreCase);
                continue;
            }

            if (argument.StartsWith("--primes=", StringComparison.OrdinalIgnoreCase) ||
                argument.StartsWith("--gpu-kernels=", StringComparison.OrdinalIgnoreCase) ||
                argument.StartsWith("--accelerator=", StringComparison.OrdinalIgnoreCase))
            {
                forcePrimeKernelsOnCpu = argument.AsSpan(argument.IndexOf('=') + 1).Equals("cpu", StringComparison.OrdinalIgnoreCase);
                continue;
            }

            if (argument.StartsWith("--results-dir=", StringComparison.OrdinalIgnoreCase))
            {
                resultsDirectory = argument[(argument.IndexOf('=') + 1)..];
                continue;
            }

            if (argument.StartsWith("--results-prefix=", StringComparison.OrdinalIgnoreCase))
            {
                resultsPrefix = argument[(argument.IndexOf('=') + 1)..];
                continue;
            }

            if (argument.StartsWith("--gpu-prime-threads=", StringComparison.OrdinalIgnoreCase))
            {
                gpuPrimeThreads = Math.Max(1, Utf8CliParser.ParseInt32(argument.AsSpan(argument.IndexOf('=') + 1)));
                continue;
            }

            if (argument.StartsWith("--gpu-prime-batch=", StringComparison.OrdinalIgnoreCase))
            {
                gpuPrimeBatch = Math.Max(1, Utf8CliParser.ParseInt32(argument.AsSpan(argument.IndexOf('=') + 1)));
                continue;
            }

            if (argument.StartsWith("--ll-slice=", StringComparison.OrdinalIgnoreCase))
            {
                sliceSize = Math.Max(1, Utf8CliParser.ParseInt32(argument.AsSpan(argument.IndexOf('=') + 1)));
                continue;
            }

            if (argument.StartsWith("--gpu-scan-batch=", StringComparison.OrdinalIgnoreCase))
            {
                scanBatchSize = Math.Max(1, Utf8CliParser.ParseInt32(argument.AsSpan(argument.IndexOf('=') + 1)));
                continue;
            }

            if (argument.StartsWith("--block-size=", StringComparison.OrdinalIgnoreCase))
            {
                blockSize = Math.Max(1, Utf8CliParser.ParseInt32(argument.AsSpan(argument.IndexOf('=') + 1)));
                continue;
            }

            if (argument.StartsWith("--filter-p=", StringComparison.OrdinalIgnoreCase))
            {
                filterFile = argument[(argument.IndexOf('=') + 1)..];
                continue;
            }

            if (argument.StartsWith("--write-batch-size=", StringComparison.OrdinalIgnoreCase))
            {
                writeBatchSize = Math.Max(1, Utf8CliParser.ParseInt32(argument.AsSpan(argument.IndexOf('=') + 1)));
                continue;
            }

            if (argument.StartsWith("--divisor-cycles=", StringComparison.OrdinalIgnoreCase))
            {
                cyclesPath = argument[(argument.IndexOf('=') + 1)..];
                continue;
            }

            if (argument.StartsWith("--divisor-cycles-device=", StringComparison.OrdinalIgnoreCase))
            {
                useGpuCycles = !argument.AsSpan(argument.IndexOf('=') + 1).Equals("cpu", StringComparison.OrdinalIgnoreCase);
                continue;
            }

            if (argument.StartsWith("--divisor-cycles-batch=", StringComparison.OrdinalIgnoreCase))
            {
                cyclesBatchSize = Math.Max(1, Utf8CliParser.ParseInt32(argument.AsSpan(argument.IndexOf('=') + 1)));
                continue;
            }

            if (argument.Equals("--divisor-cycles-continue", StringComparison.OrdinalIgnoreCase))
            {
                continueCyclesGeneration = true;
            }
        }

        return new CliArguments(
                startPrime,
                startPrimeProvided,
                maxPrimeLimit,
                maxPrimeConfigured,
                threadCount,
                blockSize,
                gpuPrimeThreads,
                gpuPrimeBatch,
                kernelType,
                useOrder,
                showHelp,
                useBitTransform,
                useLucas,
                useResidue,
                useDivisor,
                useByDivisor,
                useGcdFilter,
                testMode,
                useGpuCycles,
                useMersenneOnGpu,
                useOrderOnGpu,
                scanBatchSize,
                sliceSize,
                residueKMax,
                filterFile,
                forcePrimeKernelsOnCpu,
                modReductionMode,
                nttBackend,
                resultsDirectory,
                resultsPrefix,
                writeBatchSize,
                cyclesPath,
                continueCyclesGeneration,
                cyclesBatchSize,
                divisorCyclesSearchLimit,
                orderWarmupLimitOverride,
                rleBlacklistPath,
                rleHardMaxP,
                rleOnlyLast7,
                zeroFractionHard,
                zeroFractionConjecture,
                maxZeroConjecture);
    }
}
