using System.Buffers;
using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
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
    private static MersenneNumberDivisorGpuTester? _divisorTester;
    private static IMersenneNumberDivisorByDivisorTester? _byDivisorTester;
    private static Dictionary<ulong, (bool DetailedCheck, bool PassedAllTests)>? _byDivisorPreviousResults;
    private static long _state;
    private static bool _limitReached;
    private static readonly Optimized PrimeIterator = new();
    private static ulong _byDivisorStartPrime;
    private static CliArguments _cliArguments;
    private static int _testTargetPrimeCount;
    private static int _testProcessedPrimeCount;
    private const string TestTimeFileName = "even_perfect_test_time.txt";

    private enum PrimeTransformMode
    {
        Bit,
        Add,
        AddPrimes,
    }

    private static PrimeTransformMode TransformMode => _cliArguments.UseBitTransform
        ? PrimeTransformMode.Bit
        : (_cliArguments.UseResidue
            ? PrimeTransformMode.Add
            : PrimeTransformMode.AddPrimes);

    [ThreadStatic]
    private static bool _lastCompositeP;

    private static void Main(string[] args)
    {
        try
        {
            CliArguments parsedArguments = CliArguments.Parse(args);
            if (parsedArguments.ThreadCount == 0 && !parsedArguments.ShowHelp)
            {
                return;
            }

            _cliArguments = parsedArguments;

            if (_cliArguments.ShowHelp)
            {
                CliArguments.PrintHelp();
                return;
            }

            NttGpuMath.GpuTransformBackend = _cliArguments.NttBackend;
            NttGpuMath.ReductionMode = _cliArguments.ModReductionMode;
            GpuContextPool.ForceCpu = _cliArguments.ForcePrimeKernelsOnCpu;

            ulong currentP = _cliArguments.StartPrime;
            ulong remainder = currentP % 6UL;
            bool startPrimeProvided = _cliArguments.StartPrimeProvided;
            int threadCount = Math.Max(1, _cliArguments.ThreadCount);
            int blockSize = Math.Max(1, _cliArguments.BlockSize);
            int gpuPrimeThreads = Math.Max(1, _cliArguments.GpuPrimeThreads);
            int gpuPrimeBatch = Math.Max(1, _cliArguments.GpuPrimeBatch);
            GpuKernelType kernelType = _cliArguments.KernelType;
            bool useBitTransform = _cliArguments.UseBitTransform;
            bool useOrder = _cliArguments.UseOrder;
            bool useResidue = _cliArguments.UseResidue;
            bool useDivisor = _cliArguments.UseDivisor;
            bool useByDivisor = _cliArguments.UseByDivisor;
            bool useLucas = _cliArguments.UseLucas;
            bool testMode = _cliArguments.TestMode;
            bool useGpuCycles = _cliArguments.UseGpuCycles;
            bool mersenneOnGpu = _cliArguments.UseMersenneOnGpu;
            bool orderOnGpu = _cliArguments.UseOrderOnGpu;
            int scanBatchSize = Math.Max(1, _cliArguments.ScanBatchSize);
            int sliceSize = Math.Max(1, _cliArguments.SliceSize);
            ulong residueKMax = _cliArguments.ResidueKMax;
            string filterFile = _cliArguments.FilterFile;
            UInt128 maxPrimeLimit = _cliArguments.MaxPrimeLimit;
            bool maxPrimeConfigured = _cliArguments.MaxPrimeConfigured;
            string cyclesPath = _cliArguments.CyclesPath;
            int cyclesBatchSize = Math.Max(1, _cliArguments.CyclesBatchSize);
            bool continueCyclesGeneration = _cliArguments.ContinueCyclesGeneration;
            ulong divisorCyclesSearchLimit = _cliArguments.DivisorCyclesSearchLimit;
            ulong? orderWarmupLimitOverride = _cliArguments.OrderWarmupLimitOverride;
            int testPrimeCandidateLimit = 0;

            _limitReached = false;
            _testTargetPrimeCount = 0;
            _testProcessedPrimeCount = 0;

            if (testMode)
            {
                currentP = 31UL;
                remainder = currentP % 6UL;
                filterFile = string.Empty;
                maxPrimeLimit = UInt128.MaxValue;
                maxPrimeConfigured = false;
                startPrimeProvided = true;
                testPrimeCandidateLimit = Math.Max(1, threadCount) * 3;
            }

            if (useByDivisor)
            {
                _byDivisorStartPrime = startPrimeProvided ? currentP : 0UL;
            }
            else
            {
                _byDivisorStartPrime = 0UL;
            }

            if (useByDivisor && !testMode && string.IsNullOrEmpty(filterFile))
            {
                Console.WriteLine("--mersenne=bydivisor requires --filter-p=<path>.");
                return;
            }

            // Apply GPU prime sieve runtime configuration
                GpuPrimeWorkLimiter.SetLimit(gpuPrimeThreads);
                PrimeTester.GpuBatchSize = Math.Max(1, gpuPrimeBatch);
        
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
        
                PResidue = new ThreadLocal<ModResidueTracker>(() => new ModResidueTracker(ResidueModel.Identity, initialNumber: currentP, initialized: true), trackAllValues: true);
                PrimeTesters = new ThreadLocal<PrimeTester>(() => new PrimeTester(), trackAllValues: true);
                // Note: --primes-device controls default device for library kernels; p primality remains CPU here.
                // Initialize per-thread p residue tracker (Identity model) at currentP
                if (!useDivisor && !_cliArguments.UseByDivisor)
                {
                    MersenneTesters = new ThreadLocal<MersenneNumberTester>(() =>
                    {
                        // ProcessEightBitWindows windowed pow2 ladder is the default kernel.
                        var tester = new MersenneNumberTester(
                                                    useIncremental: !useLucas,
                                                    useOrderCache: false,
                                                    kernelType: kernelType,
                                                    useOrder: useOrder,
                                                    useGpuLucas: mersenneOnGpu,
                                                    useGpuScan: mersenneOnGpu,
                                                    useGpuOrder: orderOnGpu,
                                                    useResidue: useResidue,
                                                    maxK: residueKMax);
                        if (!useLucas)
                        {
                            Console.WriteLine("Warming up orders");
                            tester.WarmUpOrders(currentP, orderWarmupLimitOverride ?? 5_000_000UL);
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
                }
        
                // Load RLE blacklist (optional)
                if (!string.IsNullOrEmpty(_cliArguments.RleBlacklistPath))
                {
                    RleBlacklist.Load(_cliArguments.RleBlacklistPath!);
                }
        
                // Mod6 lookup turned out slower (see Mod6ComparisonBenchmarks), so keep `%` here for large candidates.
                if (currentP == InitialP && string.IsNullOrEmpty(filterFile))
                {
                    // bool passedAllTests = IsEvenPerfectCandidate(InitialP, out bool searchedMersenne, out bool detailedCheck);
                    // Skip the already processed range below 138 million
                    while (currentP < 138_000_000UL && !Volatile.Read(ref _limitReached))
                    {
                        currentP = AdvancePrime(currentP, ref remainder);
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
                _cliArguments.UseGcdFilter,
                NttGpuMath.GpuTransformBackend,
                gpuPrimeThreads,
                sliceSize,
                scanBatchSize,
                orderWarmupLimitOverride ?? 5_000_000UL,
                NttGpuMath.ReductionMode,
                mersenneOnGpu ? "gpu" : "cpu",
                GpuContextPool.ForceCpu ? "cpu" : "gpu",
                orderOnGpu ? "gpu" : "cpu");
        
                if (!string.IsNullOrEmpty(_cliArguments.ResultsPrefix))
                {
                    builtName = _cliArguments.ResultsPrefix + "_" + builtName;
                }

                string resultsFileName = string.IsNullOrEmpty(_cliArguments.ResultsDirectory)
                        ? builtName
                        : Path.Combine(_cliArguments.ResultsDirectory!, builtName);

                CalculationResultHandler.Initialize(resultsFileName, _cliArguments.WriteBatchSize);
                var dir = Path.GetDirectoryName(resultsFileName);
                if (!string.IsNullOrEmpty(dir))
                {
                    Directory.CreateDirectory(dir!);
                }
        
        
                Dictionary<ulong, (bool DetailedCheck, bool PassedAllTests)>? previousResults = useByDivisor ? new Dictionary<ulong, (bool DetailedCheck, bool PassedAllTests)>() : null;
                _byDivisorPreviousResults = previousResults;
        
                if (File.Exists(resultsFileName))
                {
                    Console.WriteLine("Processing previous results...");
                    LoadResultsFile(resultsFileName, (p, detailedCheck, passedAllTests) =>
                    {
                        CalculationResultHandler.RegisterExistingResult(detailedCheck, passedAllTests);

                        if (previousResults is not null)
                        {
                            previousResults[p] = (detailedCheck, passedAllTests);
                        }
                    });
                }
                else
                {
                    CalculationResultHandler.CreateResultsFileWithHeader();
                }
        
                bool useFilter = !testMode && !string.IsNullOrEmpty(filterFile);
                HashSet<ulong> filter = [];
                List<ulong> byDivisorCandidates = [];
                ulong maxP = 0UL;
                ulong[] localFilter = Array.Empty<ulong>();
                int filterCount = 0;
        
                if (useByDivisor)
                {
                    if (testMode)
                    {
                        byDivisorCandidates = BuildTestPrimeCandidates(testPrimeCandidateLimit);
                    }
                    else
                    {
                        Console.WriteLine("Loading filter...");
                        int skippedByLimit = 0;
                        byDivisorCandidates = LoadByDivisorCandidates(filterFile, maxPrimeLimit, maxPrimeConfigured, out skippedByLimit);
                        if (maxPrimeConfigured)
                        {
                            if (skippedByLimit > 0)
                            {
                                Console.WriteLine($"Skipped {skippedByLimit} by-divisor candidate(s) above --max-prime limit.");
                            }
        
                            if (byDivisorCandidates.Count == 0)
                            {
                                Console.WriteLine("No by-divisor candidates fall within the --max-prime limit.");
                                return;
                            }
                        }
                    }
                }
                else if (useFilter)
                {
                    Console.WriteLine("Loading filter...");
                    // TODO: Rent this filter buffer from ArrayPool<ulong> and reuse it across reload batches so
                    // we do not allocate fresh arrays while replaying large result filters.
                    localFilter = new ulong[1024];
                    filterCount = 0;
                    bool addedWithinLimit = false;
                    int skippedByLimit = 0;
                    LoadResultsFile(filterFile, (p, detailedCheck, passedAllTests) =>
                    {
                        if (!passedAllTests)
                        {
                            return;
                        }
        
                        if (maxPrimeConfigured && (UInt128)p > maxPrimeLimit)
                        {
                            skippedByLimit++;
                            return;
                        }
        
                        localFilter[filterCount++] = p;
                        addedWithinLimit = true;
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
                    });
        
                    if (filterCount > 0)
                    {
                        filter.AddRange(localFilter[..filterCount]);
                    }
        
                    if (maxPrimeConfigured)
                    {
                        if (!addedWithinLimit)
                        {
                            Console.WriteLine("No filter candidates fall within the --max-prime limit.");
                            return;
                        }
        
                        if (skippedByLimit > 0)
                        {
                            Console.WriteLine($"Skipped {skippedByLimit} filter candidate(s) above --max-prime limit.");
                        }
                    }
                }
        
        
                if (testMode)
                {
                    _testTargetPrimeCount = testPrimeCandidateLimit;
                }
                CalculationResultHandler.InitializeOutputBuffer();
        
                // Limit GPU concurrency only for prime checks (LL/NTT & GPU order scans).
                GpuPrimeWorkLimiter.SetLimit(gpuPrimeThreads);
                // Configure batch size for GPU primality sieve
                PrimeTester.GpuBatchSize = gpuPrimeBatch;
        
                Stopwatch? stopwatch = null;
                if (testMode)
                {
                    stopwatch = Stopwatch.StartNew();
                }
        
                Console.WriteLine("Starting scan...");
        
                if (useByDivisor)
                {
                    MersenneNumberDivisorByDivisorTester.Run(
                            byDivisorCandidates,
                            _byDivisorTester!,
                            _byDivisorPreviousResults,
                            _byDivisorStartPrime,
                            static () => _lastCompositeP = true,
                            static () => _lastCompositeP = false,
                            static (candidate, searchedMersenne, detailedCheck, passedAllTests) => CalculationResultHandler.HandleResult(candidate, searchedMersenne, detailedCheck, passedAllTests, _lastCompositeP),
                            threadCount);
                    CalculationResultHandler.FlushBuffer();
                    CalculationResultHandler.ReleaseOutputBuffer();
                    if (stopwatch is not null)
                    {
                        stopwatch.Stop();
                        ReportTestTime(stopwatch.Elapsed);
                    }
        
                    return;
                }
        
                if (!useDivisor && !_cliArguments.UseByDivisor)
                {
                    threadCount = Math.Max(1, threadCount);
                    _ = MersenneTesters.Value;
                    if (!useLucas)
                    {
                        Console.WriteLine("Warming up orders...");
                        MersenneTesters.Value!.WarmUpOrders(currentP, orderWarmupLimitOverride ?? 5_000_000UL);
                    }
                }
        
                if (testMode)
                {
                    testPrimeCandidateLimit = Math.Max(1, threadCount) * 3;
                    _testTargetPrimeCount = testPrimeCandidateLimit;
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
                        ArrayPool<ulong> pool = ThreadStaticPools.UlongPool;
                        ulong[] buffer = pool.Rent(blockSize);
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
									CalculationResultHandler.HandleResult(p, searchedMersenne, detailedCheck, passedAllTests, _lastCompositeP);
                                }
        
                                continue;
                            }
        
                            bool reachedMax = false;
        
                            for (j = 0; j < count; j++)
                            {
                                p = buffer[j];
        
                                if (Volatile.Read(ref _limitReached) && p > maxP)
                                {
                                    break;
                                }
        
                                if (!filter.Contains(p))
                                {
                                    continue;
                                }
        
                                passedAllTests = IsEvenPerfectCandidate(p, divisorCyclesSearchLimit, out searchedMersenne, out detailedCheck);
								CalculationResultHandler.HandleResult(p, searchedMersenne, detailedCheck, passedAllTests, _lastCompositeP);
        
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
        
                        pool.Return(buffer);
                    });
                }
        
                Task.WaitAll(tasks);
        
                TimeSpan testElapsed = default;
                bool reportTestTime = false;
                if (stopwatch is not null)
                {
                    stopwatch.Stop();
                    testElapsed = stopwatch.Elapsed;
                    reportTestTime = true;
                }
        
                CalculationResultHandler.FlushBuffer();
                CalculationResultHandler.ReleaseOutputBuffer();
        
                if (reportTestTime)
                {
                    ReportTestTime(testElapsed);
                }
        }
        finally
        {
            CalculationResultHandler.Dispose();
        }
    }

    private static List<ulong> BuildTestPrimeCandidates(int targetCount)
    {
        if (targetCount <= 0)
        {
            return [];
        }

        List<ulong> candidates = new(targetCount);
        ulong previous = 29UL;
        int remaining = targetCount;
        ulong nextPrime;

        while (remaining > 0)
        {
            try
            {
                nextPrime = PrimeIterator.Next(in previous);
            }
            catch (InvalidOperationException)
            {
                break;
            }

            if (nextPrime < 31UL)
            {
                previous = nextPrime;
                continue;
            }

            candidates.Add(nextPrime);
            remaining--;
            previous = nextPrime;
        }

        if (remaining > 0)
        {
            Console.WriteLine("Unable to populate the requested number of test primes before reaching the 64-bit limit.");
        }

        return candidates;
    }

    private static List<ulong> LoadByDivisorCandidates(string candidateFile, UInt128 maxPrimeLimit, bool maxPrimeConfigured, out int skippedByLimit)
    {
        List<ulong> candidates = [];
        skippedByLimit = 0;
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
                if (Utf8CliParser.TryParseUInt64(span[start..index], out ulong parsed))
                {
                    if (!maxPrimeConfigured || (UInt128)parsed <= maxPrimeLimit)
                    {
                        candidates.Add(parsed);
                    }
                    else
                    {
                        skippedByLimit++;
                    }
                }
            }
        }

        return candidates;
    }

    private static void LoadResultsFile(string resultsFileName, Action<ulong, bool, bool> lineProcessorAction)
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
            parsedP = Utf8CliParser.ParseUInt64(span[..first]);
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

            if (Utf8CliParser.TryParseBoolean(detailedSpan, out detailed) && Utf8CliParser.TryParseBoolean(passedAllTestsSpan, out passedAllTests))
            {
                lineProcessorAction(parsedP, detailed, passedAllTests);
            }
        }
    }


	    private static string BuildResultsFileName(bool bitInc, int threads, int block, GpuKernelType kernelType, bool useLucasFlag, bool useDivisorFlag, bool useByDivisorFlag, bool mersenneOnGpu, bool useOrder, bool useGcd, NttBackend nttBackend, int gpuPrimeThreads, int llSlice, int gpuScanBatch, ulong warmupLimit, ModReductionMode reduction, string mersenneDevice, string primesDevice, string orderDevice)
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
        Span<char> initialBuffer = stackalloc char[256];
        var builder = new PooledValueStringBuilder(initialBuffer);
        try
        {
            builder.Append("even_perfect_bit_scan_inc-");
            builder.Append(inc);
            builder.Append("_thr-");
            builder.Append(threads);
            builder.Append("_blk-");
            builder.Append(block);
            builder.Append("_mers-");
            builder.Append(mers);
            builder.Append("_mersdev-");
            builder.Append(mersenneDevice);
            builder.Append("_ntt-");
            builder.Append(ntt);
            builder.Append("_red-");
            builder.Append(red);
            builder.Append("_primesdev-");
            builder.Append(primesDevice);
            builder.Append('_');
            builder.Append(order);
            builder.Append("_orderdev-");
            builder.Append(orderDevice);
            builder.Append('_');
            builder.Append(gcd);
            builder.Append("_gputh-");
            builder.Append(gpuPrimeThreads);
            builder.Append("_llslice-");
            builder.Append(llSlice);
            builder.Append("_scanb-");
            builder.Append(gpuScanBatch);
            builder.Append("_warm-");
            builder.Append(warmupLimit);
            builder.Append(".csv");
            return builder.ToString();
        }
        finally
        {
            builder.Dispose();
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong AdvancePrime(ulong value, ref ulong remainder) => TransformMode switch
    {
        PrimeTransformMode.Bit => TransformPBit(value, ref remainder),
        PrimeTransformMode.Add => TransformPAdd(value, ref remainder),
        _ => TransformPAddPrimes(value, ref remainder),
    };

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int ReserveBlock(ulong[] buffer, int blockSize)
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
            switch (TransformMode)
            {
                case PrimeTransformMode.Bit:
                    while (count < blockSize && !Volatile.Read(ref _limitReached))
                    {
                        bufferSpan[count++] = p;
                        p = TransformPBit(p, ref remainder);
                    }

                    break;

                case PrimeTransformMode.Add:
                    while (count < blockSize && !Volatile.Read(ref _limitReached))
                    {
                        bufferSpan[count++] = p;
                        p = TransformPAdd(p, ref remainder);
                    }

                    break;

                default:
                    while (count < blockSize && !Volatile.Read(ref _limitReached))
                    {
                        bufferSpan[count++] = p;
                        p = TransformPAddPrimes(p, ref remainder);
                    }

                    break;
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
    private static void OnPrimeCandidateConfirmed()
    {
        if (!_cliArguments.TestMode)
        {
            return;
        }

        int processed = Interlocked.Increment(ref _testProcessedPrimeCount);
        if (processed == _testTargetPrimeCount)
        {
            Volatile.Write(ref _limitReached, true);
        }
    }

    private static void ReportTestTime(TimeSpan elapsed)
    {
        double elapsedSeconds = elapsed.TotalSeconds;
        double elapsedMilliseconds = elapsed.TotalMilliseconds;
        Console.WriteLine($"Test elapsed time: {elapsedSeconds:F3} s");

        string filePath = GetTestTimeFilePath();
        bool hasPrevious = TryReadTestTime(filePath, out double previousMilliseconds);
        string elapsedText = elapsedMilliseconds.ToString("F3", CultureInfo.InvariantCulture);

        if (!hasPrevious)
        {
            WriteTestTime(filePath, elapsedMilliseconds);
            Console.WriteLine($"FIRST TIME: {elapsedText} ms");
            return;
        }

        string previousText = previousMilliseconds.ToString("F3", CultureInfo.InvariantCulture);
        if (elapsedMilliseconds < previousMilliseconds)
        {
            WriteTestTime(filePath, elapsedMilliseconds);
            Console.WriteLine($"BETTER TIME: {elapsedText} ms (previous {previousText} ms)");
        }
        else
        {
            Console.WriteLine($"WORSE TIME: {elapsedText} ms (best {previousText} ms)");
        }
    }

    private static string GetTestTimeFilePath()
    {
        if (string.IsNullOrEmpty(_cliArguments.ResultsDirectory))
        {
            return TestTimeFileName;
        }

        return Path.Combine(_cliArguments.ResultsDirectory!, TestTimeFileName);
    }

    private static bool TryReadTestTime(string filePath, out double milliseconds)
    {
        if (!File.Exists(filePath))
        {
            milliseconds = 0.0;
            return false;
        }

        string content = File.ReadAllText(filePath).Trim();
        if (content.Length == 0)
        {
            milliseconds = 0.0;
            return false;
        }

        return double.TryParse(content, NumberStyles.Float, CultureInfo.InvariantCulture, out milliseconds);
    }

    private static void WriteTestTime(string filePath, double milliseconds)
    {
        string? directory = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        string formatted = milliseconds.ToString("F6", CultureInfo.InvariantCulture);
        File.WriteAllText(filePath, formatted);
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
        ulong diff;
        ulong nextPrime;

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

    internal static bool IsEvenPerfectCandidate(ulong p, ulong divisorCyclesSearchLimit, out bool searchedMersenne, out bool detailedCheck)
    {
        searchedMersenne = false;
        detailedCheck = false;
        _lastCompositeP = false;

        if (_cliArguments.UseGcdFilter && p.IsCompositeByGcd())
        {
            _lastCompositeP = true;
            return false;
        }

        // Optional: RLE blacklist and binary-threshold filters on p (safe only when configured)
        if (p <= _cliArguments.RleHardMaxP)
        {
            if (!_cliArguments.RleOnlyLast7 || p.Mod10() == 7UL)
            {
                if (RleBlacklist.IsLoaded() && RleBlacklist.Matches(p))
                {
                    return false;
                }
            }
        }

        if (_cliArguments.ZeroFractionHard >= 0 || (_cliArguments.ZeroFractionConjecture >= 0 && _cliArguments.MaxZeroConjecture >= 0))
        {
            double zf;
            int bitLength;
            int zeroCountValue;
            int maxZeroBlockValue;
            p.ComputeBitStats(out bitLength, out zeroCountValue, out maxZeroBlockValue);
            zf = (double)zeroCountValue / bitLength;
            if (_cliArguments.ZeroFractionHard >= 0)
            {
                if (zf > _cliArguments.ZeroFractionHard)
                {
                    return false;
                }
            }

            if (_cliArguments.ZeroFractionConjecture >= 0 && _cliArguments.MaxZeroConjecture >= 0)
            {
                if (zf > _cliArguments.ZeroFractionConjecture && maxZeroBlockValue >= _cliArguments.MaxZeroConjecture)
                {
                    return false;
                }
            }
        }

        // Fast residue-based composite check for p using small primes
        if (!_cliArguments.UseDivisor && !_cliArguments.UseByDivisor && IsCompositeByResidues(p))
        {
            searchedMersenne = false;
            detailedCheck = false;
            _lastCompositeP = true;
            return false;
        }

        // If primes-device=gpu, route p primality through GPU-assisted sieve with deterministic MR validation.
        if (!_cliArguments.UseByDivisor)
        {
            if (!(GpuContextPool.ForceCpu
                                            ? PrimeTesters.Value!.IsPrime(p, CancellationToken.None)
                                            : PrimeTesters.Value!.IsPrimeGpu(p, CancellationToken.None)))
            {
                _lastCompositeP = true;
                return false;
            }

            OnPrimeCandidateConfirmed();
        }

        searchedMersenne = true;
        if (_cliArguments.UseByDivisor)
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
        bool divisible;
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

}
