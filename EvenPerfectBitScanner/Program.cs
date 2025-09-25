using System.Buffers;
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
	private static bool _byDivisorPrecheckOnly;
	private static UInt128 _divisor;
	private static MersenneNumberDivisorGpuTester? _divisorTester;
	private static IMersenneNumberDivisorByDivisorTester? _byDivisorTester;
	private static ulong? _orderWarmupLimitOverride;
	private static unsafe delegate*<ulong, ref ulong, ulong> _transformP;
	private static long _state;
	private static bool _limitReached;
	private static string? _resultsDir;
	private static string? _resultsPrefix;
	private static readonly Optimized PrimeIterator = new();
	private static ulong _byDivisorStartPrime;

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

	private const int ByDivisorStateActive = 0;
	private const int ByDivisorStateComposite = 1;
	private const int ByDivisorStateCompleted = 2;
	private const int ByDivisorStateCompletedDetailed = 3;
	private const int DivisorAllocationBlockSize = 64;

	private struct ByDivisorPrimeState
	{
		internal ulong Prime;
		internal ulong AllowedMax;
		internal bool Completed;
		internal bool Composite;
		internal bool DetailedCheck;
	}

	private readonly struct PendingResult
	{
		internal PendingResult(ulong prime, bool detailedCheck, bool passedAllTests)
		{
			Prime = prime;
			DetailedCheck = detailedCheck;
			PassedAllTests = passedAllTests;
		}

		internal ulong Prime { get; }

		internal bool DetailedCheck { get; }

		internal bool PassedAllTests { get; }
	}

	private static unsafe void Main(string[] args)
	{
		ulong currentP = InitialP;
		int threadCount = Environment.ProcessorCount;
		int blockSize = 1;
		int gpuPrimeThreads = 1;
		int gpuPrimeBatch = 262_144;
		GpuKernelType kernelType = GpuKernelType.Incremental;
		bool useModuloWorkaround = false;
		// removed: useModAutomaton
		bool useOrder = false;
		bool showHelp = false;
		bool useBitTransform = false;
		bool useLucas = false;
		bool useResidue = false;    // M_p test via residue divisors (replaces LL/incremental)
		bool useDivisor = false;     // M_p divisibility by specific divisor
		bool useByDivisor = false;   // Iterative divisor scan across primes
		bool useDivisorCycles = false;
		UInt128 divisor = UInt128.Zero;
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
				threadCount = Math.Max(1, int.Parse(arg.AsSpan(arg.IndexOf('=') + 1)));
			}
			else if (arg.StartsWith("--mersenne=", StringComparison.OrdinalIgnoreCase))
			{
				mersenneOption = arg.AsSpan(arg.IndexOf('=') + 1);
				if (mersenneOption.Equals("pow2mod", StringComparison.OrdinalIgnoreCase))
				{
					kernelType = GpuKernelType.Pow2Mod;
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
			else if (arg.StartsWith("--divisor=", StringComparison.OrdinalIgnoreCase))
			{
				divisor = UInt128.Parse(arg.AsSpan(arg.IndexOf('=') + 1));
			}
			else if (arg.StartsWith("--use-divisor-cycles=", StringComparison.OrdinalIgnoreCase))
			{
				if (bool.TryParse(arg.AsSpan(arg.IndexOf('=') + 1), out bool parsedBool))
				{
					useDivisorCycles = parsedBool;
				}
			}
			else if (arg.StartsWith("--divisor-cycles-limit=", StringComparison.OrdinalIgnoreCase))
			{
				if (ulong.TryParse(arg.AsSpan(arg.IndexOf('=') + 1), out parsedLimit))
				{
					divisorCyclesSearchLimit = parsedLimit;
				}
			}
			else if (arg.StartsWith("--residue-max-k=", StringComparison.OrdinalIgnoreCase))
			{
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
				gpuPrimeThreads = Math.Max(1, int.Parse(arg.AsSpan(arg.IndexOf('=') + 1)));
			}
			else if (arg.StartsWith("--gpu-prime-batch=", StringComparison.OrdinalIgnoreCase))
			{
				gpuPrimeBatch = Math.Max(1, int.Parse(arg.AsSpan(arg.IndexOf('=') + 1)));
			}
			else if (arg.StartsWith("--ll-slice=", StringComparison.OrdinalIgnoreCase))
			{
				sliceSize = Math.Max(1, int.Parse(arg.AsSpan(arg.IndexOf('=') + 1)));
			}
			else if (arg.StartsWith("--gpu-scan-batch=", StringComparison.OrdinalIgnoreCase))
			{
				scanBatchSize = Math.Max(1, int.Parse(arg.AsSpan(arg.IndexOf('=') + 1)));
			}
			else if (arg.StartsWith("--block-size=", StringComparison.OrdinalIgnoreCase))
			{
				blockSize = Math.Max(1, int.Parse(arg.AsSpan(arg.IndexOf('=') + 1)));
			}
			else if (arg.StartsWith("--filter-p=", StringComparison.OrdinalIgnoreCase))
			{
				filterFile = arg[(arg.IndexOf('=') + 1)..];
			}
			else if (arg.StartsWith("--write-batch-size=", StringComparison.OrdinalIgnoreCase))
			{
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
			else if (arg.StartsWith("--divisor-cycles-batch-size=", StringComparison.OrdinalIgnoreCase))
			{
				cyclesBatchSize = Math.Max(1, int.Parse(arg.AsSpan(arg.IndexOf('=') + 1)));
			}
			else if (arg.Equals("--divisor-cycles-continue", StringComparison.OrdinalIgnoreCase))
			{
				continueCyclesGeneration = true;
			}
		}

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

                DivisorCycleCache.Shared.ConfigureGeneratorDevice(orderOnGpu);
                DivisorCycleCache.Shared.ReloadFromCurrentSnapshot();

		Console.WriteLine("Divisor cycles are ready");

		if (useByDivisor)
		{
			threadCount = 1;
			blockSize = 1;
		}

		_useDivisor = useDivisor;
		_useResidueMode = useResidue;
		_useByDivisorMode = useByDivisor;
		_divisor = divisor;
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
		tester.WarmUpOrders(currentP, _orderWarmupLimitOverride ?? 5_000_000UL);
	}

	return tester;
}, trackAllValues: true);
		}
		else if (useDivisor)
		{
			if (divisor == UInt128.Zero)
			{
				MersenneNumberDivisorGpuTester.BuildDivisorCandidates();
			}

			_divisorTester = new MersenneNumberDivisorGpuTester();
		}
		else if (useByDivisor)
		{
			_byDivisorTester = mersenneOnGpu
					? new MersenneNumberDivisorByDivisorGpuTester()
					: new MersenneNumberDivisorByDivisorCpuTester();
			_byDivisorTester.BatchSize = scanBatchSize;
			_byDivisorTester.UseDivisorCycles = useDivisorCycles;
		}

		// Load RLE blacklist (optional)
		if (!string.IsNullOrEmpty(_rleBlacklistPath))
		{
			RleBlacklist.Load(_rleBlacklistPath!);
		}

		ulong remainder = currentP % 6UL;
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


		if (File.Exists(ResultsFileName))
		{
			Console.WriteLine("Processing previous results...");
			LoadResultsFile(ResultsFileName, (p, detailedCheck, passedAllTests) =>
			{
				if (detailedCheck && passedAllTests)
				{
					_primeCount++;
				}
			});
		}
		else
		{
			File.WriteAllText(ResultsFileName, $"p,searchedMersenne,detailedCheck,passedAllTests{Environment.NewLine}");
		}

		bool useFilter = !string.IsNullOrEmpty(filterFile);
		HashSet<ulong> filter = [];
		List<ulong> filterList = [];
		ulong maxP = 0UL;
		ulong[] localFilter = Array.Empty<ulong>();
		int filterCount = 0;
		if (useFilter)
		{
			Console.WriteLine("Loading filter...");
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
			filterList = [.. filter];
		}

		if (useByDivisor)
		{
			if (maxP <= 1UL)
			{
				Console.WriteLine("The filter specified by --filter-p must contain at least one prime exponent greater than 1 for --mersenne=bydivisor.");
				return;
			}

			_byDivisorTester!.ConfigureFromMaxPrime(maxP);
		}

		if (_primeCount >= 2)
		{
			_primeFoundAfterInit = true;
		}

		_outputBuilder = StringBuilderPool.Rent();

		// Limit GPU concurrency only for prime checks (LL/NTT & GPU order scans).
		GpuPrimeWorkLimiter.SetLimit(gpuPrimeThreads);
		// Configure batch size for GPU primality sieve
		PrimeTester.GpuBatchSize = gpuPrimeBatch;

		Console.WriteLine("Starting scan...");

		if (_useByDivisorMode)
		{
			RunByDivisorMode(filterList, divisorCyclesSearchLimit, threadCount);
			FlushBuffer();
			StringBuilderPool.Return(_outputBuilder!);
			return;
		}

		if (!useDivisor && !useByDivisor)
		{
			Console.WriteLine("Warming up orders...");

			threadCount = Math.Max(1, threadCount);
			_ = MersenneTesters.Value;
			if (!useLucas)
			{
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

	// TODO: We should create a scanner class and move this there to keep functionality encapsulated and avoid too big files.
	private static void RunByDivisorMode(List<ulong> primes, ulong divisorCyclesSearchLimit, int threadCount)
	{
		if (primes.Count == 0)
		{
			return;
		}

		List<ByDivisorPrimeState> states = new(primes.Count);
		ByDivisorPrimeState stateToAdd;
		bool searched = false;
		bool detailed = false;
		ulong allowedMax = 0UL;

		_byDivisorPrecheckOnly = true;
		ulong startPrime = _byDivisorStartPrime;
		bool applyStartPrime = startPrime > 0UL;

		foreach (ulong prime in primes)
		{
			if (applyStartPrime && prime < startPrime)
			{
				continue;
			}

			if (!IsEvenPerfectCandidate(prime, divisorCyclesSearchLimit, out searched, out detailed))
			{
				PrintResult(prime, searched, detailed, false);
				continue;
			}

			allowedMax = _byDivisorTester!.GetAllowedMaxDivisor(prime);
			if (allowedMax < 3UL)
			{
				PrintResult(prime, searchedMersenne: true, detailedCheck: true, passedAllTests: true);
				continue;
			}

			stateToAdd = new ByDivisorPrimeState
			{
				Prime = prime,
				AllowedMax = allowedMax,
				Completed = false,
				Composite = false,
				DetailedCheck = false,
			};

			states.Add(stateToAdd);
		}

		_byDivisorPrecheckOnly = false;

		if (states.Count == 0)
		{
			if (applyStartPrime)
			{
				Console.WriteLine($"No primes greater than or equal to {startPrime.ToString(CultureInfo.InvariantCulture)} were found for --mersenne=bydivisor.");
			}

			return;
		}

		// TODO: The list of primes should be always in ascending order. Why do we need to separately sort it?
		states.Sort(static (left, right) =>
		{
			int compare = left.AllowedMax.CompareTo(right.AllowedMax);
			if (compare != 0)
			{
				return compare;
			}

			return left.Prime.CompareTo(right.Prime);
		});

		int stateCount = states.Count;
		ulong[] primeValues = new ulong[stateCount];
		ulong[] allowedMaxValues = new ulong[stateCount];
		int[] stateFlags = new int[stateCount];

		int stateIndex = 0;
		ByDivisorPrimeState currentState;
		for (; stateIndex < stateCount; stateIndex++)
		{
			currentState = states[stateIndex];
			primeValues[stateIndex] = currentState.Prime;
			allowedMaxValues[stateIndex] = currentState.AllowedMax;
			stateFlags[stateIndex] = ByDivisorStateActive;
		}

		states.Clear();

		ulong divisorLimit = _byDivisorTester!.DivisorLimit;
		ulong nextDivisor = 3UL;
		long finalDivisorBits = unchecked((long)3UL);
		int divisorsExhaustedFlag = 0;
		int finalizerState = 0;
		int finalizationCompleted = 0;
		int remainingStates = stateCount;
		int activeStartIndex = 0;
		long[] activeStateMask = new long[(stateCount + 63) >> 6];

		int maskStateIndex = 0;
		int wordIndex = 0;
		int bitIndex = 0;
		for (; maskStateIndex < stateCount; maskStateIndex++)
		{
			wordIndex = maskStateIndex >> 6;
			bitIndex = maskStateIndex & 63;
			activeStateMask[wordIndex] |= 1L << bitIndex;
		}

		Task[] workers = new Task[Math.Max(1, threadCount)];

		int workerIndex = 0;
		for (; workerIndex < workers.Length; workerIndex++)
		{
			int capturedStateCount = stateCount;
			// TODO: I don't like that we schedule tasks and don't control how many get actually executed at the same time. This results in not enough threads being created and only some tasks running in parallel.
			workers[workerIndex] = Task.Run(() =>
			{
				// TODO: Much of this should be probably moved to MersenneNumberDivisorByDivisorGpuTester / MersenneNumberDivisorByDivisorCpuTester and adjusted for the best performance depending on the device. The goal of CPU implementation is to avoid as much as possible of work caused only by GPU support.
				using var session = _byDivisorTester!.CreateDivisorSession();
				byte[] hitsBuffer = ArrayPool<byte>.Shared.Rent(capturedStateCount);
				ulong[] primeBuffer = ArrayPool<ulong>.Shared.Rent(capturedStateCount);
				int[] indexBuffer = ArrayPool<int>.Shared.Rent(capturedStateCount);
				// TODO: This is probably not necessary for the CPU path. We might be able to handle it better.
				PendingResult[] completionsBuffer = ArrayPool<PendingResult>.Shared.Rent(capturedStateCount);
				PendingResult[] compositesBuffer = ArrayPool<PendingResult>.Shared.Rent(capturedStateCount);
				int completionsCount = 0;
				int compositesCount = 0;
				ulong localDivisorCursor = 0UL;
				int localDivisorsRemaining = 0;
				bool exhausted = false;
				ulong divisor = 0UL;
				int activeCount = 0;
				// TODO: We probably shouldn't need hits array on the CPU path, as we can calculate & report things on the fly
				Span<byte> hitsSpan = default;
				int hitIndex = 0;
				int index = 0;
				// TODO: We should be consistent in naming things. This should be called useDivisorCycles.
				bool useCycleFilter = _byDivisorTester!.UseDivisorCycles;
				DivisorCycleCache.Lease cycleLease = default;
				ulong cycleLeaseStart = 0UL;
				ulong cycleLeaseEnd = 0UL;
				ulong[]? cycleValues = null;
				ulong divisorCycle = 0UL;

				try
				{
					while (true)
					{
						if (Volatile.Read(ref remainingStates) == 0)
						{
							if (Volatile.Read(ref finalizationCompleted) != 0 || Volatile.Read(ref divisorsExhaustedFlag) == 0)
							{
								break;
							}
						}

						// TODO: Can we assign block of divisors to tasks instead of giving them one by one to improve thread concurrency? Say - the next 10 divisors are yours
						divisor = AcquireNextDivisor(ref nextDivisor, divisorLimit, ref divisorsExhaustedFlag, ref finalDivisorBits, out exhausted, ref localDivisorCursor, ref localDivisorsRemaining);

						// TODO: This condition should be reversed as it should highly simplify the code. We currently have a gigantic if block.
						if (divisor == 0UL)
						{
							if (!exhausted)
							{
								if (Volatile.Read(ref remainingStates) == 0)
								{
									break;
								}

								Thread.Yield();
								continue;
							}

							if (Interlocked.CompareExchange(ref finalizerState, 1, 0) == 0)
							{
								FinalizeRemainingStates(primeValues, allowedMaxValues, stateFlags, ref remainingStates, ref finalDivisorBits, completionsBuffer, ref completionsCount, activeStateMask);

								if (completionsCount > 0)
								{
									FlushPendingResults(completionsBuffer, ref completionsCount);
								}

								Volatile.Write(ref finalizationCompleted, 1);
							}
							else
							{
								while (Volatile.Read(ref finalizationCompleted) == 0 && Volatile.Read(ref remainingStates) > 0)
								{
									Thread.Yield();
								}
							}

							if (Volatile.Read(ref remainingStates) == 0)
							{
								break;
							}

							continue;
						}

						if (useCycleFilter)
						{
							if (!cycleLease.IsValid || divisor < cycleLeaseStart || divisor > cycleLeaseEnd)
							{
								cycleLease.Dispose();
								cycleLease = DivisorCycleCache.Shared.Acquire(divisor);
								cycleLeaseStart = cycleLease.Start;
								cycleLeaseEnd = cycleLease.End;
								cycleValues = cycleLease.Values;
							}
							else if (cycleValues is null)
							{
								cycleValues = cycleLease.Values;
							}

							// TODO: This should never happen in production code. We should only have the acquired cycles. I.e. cycleValues should always be != null and always for the given divisor.
							if (cycleValues is not null && divisor >= cycleLeaseStart)
							{
								ulong offset = divisor - cycleLeaseStart;
								if (offset < (ulong)cycleValues.Length)
								{
									divisorCycle = cycleValues[(int)offset];
								}
								else
								{
									divisorCycle = cycleLease.GetCycle(divisor);
								}
							}
							else
							{
								divisorCycle = cycleLease.GetCycle(divisor);
							}
						}
						else
						{
							divisorCycle = 0UL;
						}

						// TODO: Can we prepare as many buffers as many threads were requested upfront and have them prepared in separate task, so that they're always ready when needed? Do we really need to always call BuildPrimeBuffer in the loop? The list of primes should be constant between divisors, because we always process the list of primes from the file. If the buffer is big enough, we can create one common list of primes and reuse it for all divisors.
						activeCount = BuildPrimeBuffer(divisor, primeValues, allowedMaxValues, stateFlags, primeBuffer, indexBuffer, completionsBuffer, ref completionsCount, ref remainingStates, activeStateMask, ref activeStartIndex);

						if (completionsCount > 0)
						{
							FlushPendingResults(completionsBuffer, ref completionsCount);
						}

						if (activeCount == 0)
						{
							continue;
						}

						hitsSpan = hitsBuffer.AsSpan(0, activeCount);
						hitsSpan.Clear();
						// TODO: The goal of the CPU implementation was to avoid all the hassle with playing with hitSpan buffer as it can directly compute things.
						session.CheckDivisor(divisor, divisorCycle, primeBuffer.AsSpan(0, activeCount), hitsSpan);

						// TODO: This implementation is highly over-complicated for CPU. We shouldn't probably even need the indexBuffer and hitsSpan.
						for (hitIndex = 0; hitIndex < activeCount; hitIndex++)
						{
							if (hitsSpan[hitIndex] == 0)
							{
								continue;
							}

							index = indexBuffer[hitIndex];
							if (Interlocked.CompareExchange(ref stateFlags[index], ByDivisorStateComposite, ByDivisorStateActive) == ByDivisorStateActive)
							{
								ClearActiveMask(activeStateMask, index);
								Interlocked.Decrement(ref remainingStates);
								compositesBuffer[compositesCount++] = new PendingResult(primeValues[index], detailedCheck: true, passedAllTests: false);
							}

							if (compositesCount == compositesBuffer.Length)
							{
								FlushPendingResults(compositesBuffer, ref compositesCount);
							}
						}

						if (compositesCount > 0)
						{
							FlushPendingResults(compositesBuffer, ref compositesCount);
						}
					}

					if (completionsCount > 0)
					{
						FlushPendingResults(completionsBuffer, ref completionsCount);
					}

					if (compositesCount > 0)
					{
						FlushPendingResults(compositesBuffer, ref compositesCount);
					}
				}
				finally
				{
					cycleLease.Dispose();
					ArrayPool<PendingResult>.Shared.Return(compositesBuffer, clearArray: true);
					ArrayPool<PendingResult>.Shared.Return(completionsBuffer, clearArray: true);
					ArrayPool<int>.Shared.Return(indexBuffer, clearArray: true);
					ArrayPool<ulong>.Shared.Return(primeBuffer, clearArray: true);
					ArrayPool<byte>.Shared.Return(hitsBuffer, clearArray: true);
				}
			});
		}

		Task.WaitAll(workers);
	}

	private static ulong AcquireNextDivisor(ref ulong nextDivisor, ulong divisorLimit, ref int divisorsExhaustedFlag, ref long finalDivisorBits, out bool exhausted, ref ulong localDivisorCursor, ref int localDivisorsRemaining)
	{
		ref long nextDivisorBits = ref Unsafe.As<ulong, long>(ref nextDivisor);
		// TODO: You should avoid unnecessary assignment of variables, if the initial value is unused
		ulong currentValue = 0UL;
		long currentBits = 0L;
		ulong maximumNext = 0UL;
		ulong blockStride = unchecked((ulong)(DivisorAllocationBlockSize * 2));
		ulong requestedNext = 0UL;
		long nextBits = 0L;
		ulong available = 0UL;
		int count = 0;

		if (localDivisorsRemaining > 0)
		{
			currentValue = localDivisorCursor;
			localDivisorCursor += 2UL;
			localDivisorsRemaining--;
			exhausted = false;
			return currentValue;
		}

		while (true)
		{
			if (Volatile.Read(ref divisorsExhaustedFlag) != 0)
			{
				exhausted = true;
				return 0UL;
			}

			currentBits = Volatile.Read(ref nextDivisorBits);
			currentValue = unchecked((ulong)currentBits);
			if (currentValue > divisorLimit)
			{
				if (Interlocked.CompareExchange(ref divisorsExhaustedFlag, 1, 0) == 0)
				{
					Volatile.Write(ref finalDivisorBits, currentBits);
				}

				exhausted = true;
				return 0UL;
			}

			maximumNext = divisorLimit >= ulong.MaxValue - 1UL ? ulong.MaxValue : divisorLimit + 2UL;
			requestedNext = currentValue > ulong.MaxValue - blockStride ? ulong.MaxValue : currentValue + blockStride;
			if (requestedNext > maximumNext)
			{
				requestedNext = maximumNext;
			}

			nextBits = unchecked((long)requestedNext);
			if (Interlocked.CompareExchange(ref nextDivisorBits, nextBits, currentBits) != currentBits)
			{
				continue;
			}

			available = requestedNext - currentValue;
			if (available == 0UL)
			{
				continue;
			}

			count = (int)(available >> 1);
			if (count <= 0)
			{
				continue;
			}

			localDivisorCursor = currentValue + 2UL;
			localDivisorsRemaining = count - 1;
			exhausted = false;
			return currentValue;
		}
	}

	private static int BuildPrimeBuffer(ulong divisor, ulong[] primeValues, ulong[] allowedMaxValues, int[] stateFlags, ulong[] primeBuffer, int[] indexBuffer, PendingResult[] completionsBuffer, ref int completionsCount, ref int remainingStates, long[] activeStateMask, ref int activeStartIndex)
	{
		int length = primeValues.Length;
		int startIndex = Volatile.Read(ref activeStartIndex);
		int index = startIndex;
		// TODO: You should avoid unnecessary assignment of variables, if the initial value is unused
		int wordIndex = 0;
		ulong word = 0UL;
		int bitOffset = 0;
		int candidateIndex = 0;

		while (index < length && allowedMaxValues[index] < divisor)
		{
			if (Interlocked.CompareExchange(ref stateFlags[index], ByDivisorStateCompletedDetailed, ByDivisorStateActive) == ByDivisorStateActive)
			{
				ClearActiveMask(activeStateMask, index);
				Interlocked.Decrement(ref remainingStates);
				if (completionsCount == completionsBuffer.Length)
				{
					FlushPendingResults(completionsBuffer, ref completionsCount);
				}
				completionsBuffer[completionsCount++] = new PendingResult(primeValues[index], detailedCheck: true, passedAllTests: true);
			}
			else
			{
				ClearActiveMask(activeStateMask, index);
			}

			index++;
		}

		if (index != startIndex)
		{
			Volatile.Write(ref activeStartIndex, index);
		}

		int activeCount = 0;
		while (index < length)
		{
			wordIndex = index >> 6;
			word = unchecked((ulong)Volatile.Read(ref activeStateMask[wordIndex]));
			if (word == 0UL)
			{
				index = (wordIndex + 1) << 6;
				continue;
			}

			bitOffset = index & 63;
			if (bitOffset != 0)
			{
				word &= ulong.MaxValue << bitOffset;
				if (word == 0UL)
				{
					index = (wordIndex + 1) << 6;
					continue;
				}
			}

			while (word != 0UL)
			{
				candidateIndex = (wordIndex << 6) + BitOperations.TrailingZeroCount(word);
				if (candidateIndex >= length)
				{
					return activeCount;
				}

				if (Volatile.Read(ref stateFlags[candidateIndex]) == ByDivisorStateActive)
				{
					primeBuffer[activeCount] = primeValues[candidateIndex];
					indexBuffer[activeCount] = candidateIndex;
					activeCount++;
				}
				else
				{
					ClearActiveMask(activeStateMask, candidateIndex);
				}

				word &= word - 1UL;
			}

			index = (wordIndex + 1) << 6;
		}

		return activeCount;
	}

	private static void FinalizeRemainingStates(ulong[] primeValues, ulong[] allowedMaxValues, int[] stateFlags, ref int remainingStates, ref long finalDivisorBits, PendingResult[] completionsBuffer, ref int completionsCount, long[] activeStateMask)
	{
		int finalizeIndex = 0;
		bool detailed = false;
		for (; finalizeIndex < primeValues.Length; finalizeIndex++)
		{
			if (Volatile.Read(ref stateFlags[finalizeIndex]) != ByDivisorStateActive)
			{
				continue;
			}

			detailed = unchecked((ulong)Volatile.Read(ref finalDivisorBits)) > allowedMaxValues[finalizeIndex];

			if (Interlocked.CompareExchange(ref stateFlags[finalizeIndex], detailed ? ByDivisorStateCompletedDetailed : ByDivisorStateCompleted, ByDivisorStateActive) != ByDivisorStateActive)
			{
				continue;
			}

			ClearActiveMask(activeStateMask, finalizeIndex);
			Interlocked.Decrement(ref remainingStates);
			if (completionsCount == completionsBuffer.Length)
			{
				FlushPendingResults(completionsBuffer, ref completionsCount);
			}
			completionsBuffer[completionsCount++] = new PendingResult(primeValues[finalizeIndex], detailedCheck: detailed, passedAllTests: true);
		}
	}

	private static void FlushPendingResults(PendingResult[] buffer, ref int count)
	{
		int flushIndex = 0;
		for (; flushIndex < count; flushIndex++)
		{
			PendingResult result = buffer[flushIndex];
			PrintResult(result.Prime, searchedMersenne: true, detailedCheck: result.DetailedCheck, passedAllTests: result.PassedAllTests);
		}

		count = 0;
	}

	private static void ClearActiveMask(long[] activeStateMask, int index)
	{
		int wordIndex = index >> 6;
		long bit = 1L << (index & 63);

		while (true)
		{
			long current = Volatile.Read(ref activeStateMask[wordIndex]);
			if ((current & bit) == 0)
			{
				return;
			}

			if (Interlocked.CompareExchange(ref activeStateMask[wordIndex], current & ~bit, current) == current)
			{
				return;
			}
		}
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

			if (bool.TryParse(detailedSpan, out detailed) && bool.TryParse(passedAllTestsSpan, out passedAllTests))
			{
				lineProcessorAction(parsedP, detailed, passedAllTests);
			}
		}
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
		Console.WriteLine("  --divisor=<value>     optional divisor for --mersenne=divisor mode");
		Console.WriteLine("  --residue-max-k=<value>  max k for residue Mersenne test (q = 2*p*k + 1)");
		Console.WriteLine("  --mersenne-device=cpu|gpu  Device for Mersenne method (default gpu)");
		Console.WriteLine("  --primes-device=cpu|gpu    Device for prime-scan kernels (default gpu)");
		Console.WriteLine("  --gpu-prime-batch=<n>      Batch size for GPU primality sieve (default 262144)");
		Console.WriteLine("  --order-device=cpu|gpu     Device for order computations (default gpu)");
		Console.WriteLine("  --ntt=reference|staged GPU NTT backend (default staged)");
		Console.WriteLine("  --mod-reduction=auto|uint128|mont64|barrett128  staged NTT reduction (default auto)");
		Console.WriteLine("  --gpu-prime-threads=<value>  max concurrent GPU prime checks (default 1)");
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
		Console.WriteLine("  --divisor-cycles-batch-size=<value> batch size for cycles generation (default 512)");
		Console.WriteLine("  --divisor-cycles-continue  continue divisor cycles generation");
		Console.WriteLine("  --divisor-cycles-limit=<value> cycle search iterations when --mersenne=divisor");
		Console.WriteLine("  --use-divisor-cycles=true|false  enable divisor cycle filtering in --mersenne=bydivisor (default false)");

		Console.WriteLine("  --use-order            test primality via q order");
		Console.WriteLine("  --workaround-mod       avoid '%' operator on the GPU");
		// mod-automaton removed
		Console.WriteLine("  --filter-p=<path>      process only p from previous run results (required for --mersenne=bydivisor)");
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
	internal static bool IsEvenPerfectCandidate(ulong p, ulong divisorCyclesSearchLimit) => IsEvenPerfectCandidate(p, divisorCyclesSearchLimit, out _, out _);

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
			if (_zeroFracHard >= 0)
			{
				zf = (double)zeroCountValue / bitLength;
				if (zf > _zeroFracHard)
				{
					return false;
				}
			}

			if (_zeroFracConj >= 0 && _maxZeroConj >= 0)
			{
				zf = (double)zeroCountValue / bitLength;
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

		// If primes-device=gpu, route p primality through GPU-assisted sieve (optional MR later)
		// TODO: Add deterministic Miller–Rabin rounds in GPU path for 64-bit range.
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
			if (_byDivisorPrecheckOnly)
			{
				searchedMersenne = false;
				detailedCheck = false;
				return true;
			}

			return _byDivisorTester!.IsPrime(p, out detailedCheck);
		}

		if (_useDivisor)
		{
			return _divisorTester!.IsPrime(p, _divisor, divisorCyclesSearchLimit, out detailedCheck);
		}

		detailedCheck = MersenneTesters.Value!.IsMersennePrime(p);
		return detailedCheck;
	}

	// Use ModResidueTracker with a small set of primes to pre-filter composite p.
	private static bool IsCompositeByResidues(ulong p)
	{
		var tracker = PResidue.Value!;
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
