using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using PerfectNumbers.Core;
using PerfectNumbers.Core.Gpu;

namespace EvenPerfectBitScanner;

internal static class Program
{
	private static ThreadLocal<PrimeTester> PrimeTesters = null!;
	private static ThreadLocal<MersenneNumberTester> MersenneTesters = null!;
	private static ThreadLocal<ModResidueTracker> PResidue = null!;      // p mod d tracker (per-thread)
	private const ulong InitialP = PerfectNumberConstants.BiggestKnownEvenPerfectP;

	private const int ConsoleInterval = 100_000;
	private const int WriteBatchSize = 100;
	private const string DefaultCyclesPath = "divisor_cycles.bin";
	private static string ResultsFileName = "even_perfect_bit_scan_results.csv";
	private const string PrimeFoundSuffix = " (FOUND VALID CANDIDATES)";
	private static StringBuilder? _outputBuilder;
	private static readonly object Sync = new();
	private static int _consoleCounter;
	private static int _writeIndex;
	private static int _primeCount;
	private static bool _primeFoundAfterInit;
	private static bool _useGcdFilter;
        private static bool _useDivisor;
        private static UInt128 _divisor;
	private static MersenneNumberDivisorGpuTester? _divisorTester;
	private static ulong? _orderWarmupLimitOverride;
	private static unsafe delegate*<ulong, ref ulong, ulong> _transformP;
	private static long _state;
	private static bool _limitReached;
	private static string? _resultsDir;
	private static string? _resultsPrefix;

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

		// NTT backend selection (GPU): reference vs staged
		for (int i = 0; i < args.Length; i++)
		{
			string arg = args[i];
			if (arg.Equals("--?", StringComparison.OrdinalIgnoreCase) || arg.Equals("-?", StringComparison.OrdinalIgnoreCase) || arg.Equals("--help", StringComparison.OrdinalIgnoreCase) || arg.Equals("-help", StringComparison.OrdinalIgnoreCase) || arg.Equals("/?", StringComparison.OrdinalIgnoreCase))
			{
				showHelp = true;
				break;
			}
			else if (arg.StartsWith("--prime=", StringComparison.OrdinalIgnoreCase))
			{
				int eq = arg.IndexOf('=');
				currentP = ulong.Parse(arg.AsSpan(eq + 1));
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
				int eq = arg.IndexOf('=');
				threadCount = Math.Max(1, int.Parse(arg.AsSpan(eq + 1)));
			}
			else if (arg.StartsWith("--mersenne=", StringComparison.OrdinalIgnoreCase))
			{
				int eq = arg.IndexOf('=');
				ReadOnlySpan<char> value = arg.AsSpan(eq + 1);
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
				else
				{
					kernelType = GpuKernelType.Incremental;
				}
			}
                        else if (arg.StartsWith("--divisor=", StringComparison.OrdinalIgnoreCase))
                        {
                                int eq = arg.IndexOf('=');
                                divisor = UInt128.Parse(arg.AsSpan(eq + 1));
                        }
                        else if (arg.StartsWith("--divisor-cycles-limit=", StringComparison.OrdinalIgnoreCase))
                        {
                                int eq = arg.IndexOf('=');
                                divisorCyclesSearchLimit = ulong.Parse(arg.AsSpan(eq + 1));
                        }
			else if (arg.StartsWith("--residue-max-k=", StringComparison.OrdinalIgnoreCase))
			{
				int eq = arg.IndexOf('=');
				if (ulong.TryParse(arg.AsSpan(eq + 1), out var kmax))
				{
					residueKMax = kmax;
				}
			}
			// Replaces --lucas=cpu|gpu; controls device for Lucas and for
			// incremental/pow2mod scanning
			else if (arg.StartsWith("--mersenne-device=", StringComparison.OrdinalIgnoreCase))
			{
				int eq = arg.IndexOf('=');
				ReadOnlySpan<char> value = arg.AsSpan(eq + 1);
				mersenneOnGpu = !value.Equals("cpu", StringComparison.OrdinalIgnoreCase);
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
				int eq = arg.IndexOf('=');
				// stored and used below when initializing testers
				if (ulong.TryParse(arg.AsSpan(eq + 1), out var limit))
				{
					_orderWarmupLimitOverride = limit;
				}
			}
			else if (arg.Equals("--gcd-filter", StringComparison.OrdinalIgnoreCase))
			{
				_useGcdFilter = true;
			}
			else if (arg.StartsWith("--ntt=", StringComparison.OrdinalIgnoreCase))
			{
				int eq = arg.IndexOf('=');
				ReadOnlySpan<char> value = arg.AsSpan(eq + 1);
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
				int eq = arg.IndexOf('=');
				ReadOnlySpan<char> value = arg.AsSpan(eq + 1);
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
				int eq = arg.IndexOf('=');
				ReadOnlySpan<char> value = arg.AsSpan(eq + 1);
				// Controls default device for library prime-related GPU kernels
				// that are not explicitly parameterized (backward compatibility).
				GpuContextPool.ForceCpu = value.Equals("cpu", StringComparison.OrdinalIgnoreCase);
			}
			// New: choose device for order computations (warm-ups and order scans)
			else if (arg.StartsWith("--order-device=", StringComparison.OrdinalIgnoreCase))
			{
				int eq = arg.IndexOf('=');
				ReadOnlySpan<char> value = arg.AsSpan(eq + 1);
				orderOnGpu = !value.Equals("cpu", StringComparison.OrdinalIgnoreCase);
			}
			// Optional: RLE blacklist for p
			else if (arg.StartsWith("--rle-blacklist=", StringComparison.OrdinalIgnoreCase))
			{
				int eq = arg.IndexOf('=');
				_rleBlacklistPath = arg.Substring(eq + 1);
			}
			else if (arg.StartsWith("--rle-hard-max=", StringComparison.OrdinalIgnoreCase))
			{
				int eq = arg.IndexOf('=');
				if (ulong.TryParse(arg.AsSpan(eq + 1), out var rleMaxP))
				{
					_rleHardMaxP = rleMaxP;
				}
			}
			else if (arg.StartsWith("--rle-only-last7=", StringComparison.OrdinalIgnoreCase))
			{
				int eq = arg.IndexOf('=');
				ReadOnlySpan<char> value = arg.AsSpan(eq + 1);
				_rleOnlyLast7 = !value.Equals("false", StringComparison.OrdinalIgnoreCase) && !value.Equals("0", StringComparison.OrdinalIgnoreCase);
			}
			else if (arg.StartsWith("--zero-hard=", StringComparison.OrdinalIgnoreCase))
			{
				int eq = arg.IndexOf('=');
				if (double.TryParse(arg.AsSpan(eq + 1), out var z))
				{
					_zeroFracHard = z;
				}
			}
			else if (arg.StartsWith("--zero-conj=", StringComparison.OrdinalIgnoreCase))
			{
				// format: <zeroFrac>:<maxZeroBlock>
				int eq = arg.IndexOf('=');
				ReadOnlySpan<char> value = arg.AsSpan(eq + 1);
				int colon = value.IndexOf(':');
				if (colon > 0)
				{
					if (double.TryParse(value[..colon], out var zf) && int.TryParse(value[(colon + 1)..], out var mz))
					{
						_zeroFracConj = zf;
						_maxZeroConj = mz;
					}
				}
			}
			// Backward-compat: accept deprecated device flags and map to new ones
			else if (arg.StartsWith("--lucas=", StringComparison.OrdinalIgnoreCase))
			{
				int eq = arg.IndexOf('=');
				ReadOnlySpan<char> value = arg.AsSpan(eq + 1);
				mersenneOnGpu = !value.Equals("cpu", StringComparison.OrdinalIgnoreCase);
			}
			else if (arg.StartsWith("--primes=", StringComparison.OrdinalIgnoreCase))
			{
				int eq = arg.IndexOf('=');
				ReadOnlySpan<char> value = arg.AsSpan(eq + 1);
				GpuContextPool.ForceCpu = value.Equals("cpu", StringComparison.OrdinalIgnoreCase);
			}
			else if (arg.StartsWith("--gpu-kernels=", StringComparison.OrdinalIgnoreCase) || arg.StartsWith("--accelerator=", StringComparison.OrdinalIgnoreCase))
			{
				int eq = arg.IndexOf('=');
				ReadOnlySpan<char> value = arg.AsSpan(eq + 1);
				// Legacy alias: treat as primes-device for backward compat
				GpuContextPool.ForceCpu = value.Equals("cpu", StringComparison.OrdinalIgnoreCase);
			}
			else if (arg.StartsWith("--results-dir=", StringComparison.OrdinalIgnoreCase))
			{
				int eq = arg.IndexOf('=');
				_resultsDir = arg.Substring(eq + 1);
			}
			else if (arg.StartsWith("--results-prefix=", StringComparison.OrdinalIgnoreCase))
			{
				int eq = arg.IndexOf('=');
				_resultsPrefix = arg.Substring(eq + 1);
			}

			else if (arg.StartsWith("--gpu-prime-threads=", StringComparison.OrdinalIgnoreCase))
			{
				int eq = arg.IndexOf('=');
				gpuPrimeThreads = Math.Max(1, int.Parse(arg.AsSpan(eq + 1)));
			}
			else if (arg.StartsWith("--gpu-prime-batch=", StringComparison.OrdinalIgnoreCase))
			{
				int eq = arg.IndexOf('=');
				gpuPrimeBatch = Math.Max(1, int.Parse(arg.AsSpan(eq + 1)));
			}
			else if (arg.StartsWith("--ll-slice=", StringComparison.OrdinalIgnoreCase))
			{
				int eq = arg.IndexOf('=');
				sliceSize = Math.Max(1, int.Parse(arg.AsSpan(eq + 1)));
			}
			else if (arg.StartsWith("--gpu-scan-batch=", StringComparison.OrdinalIgnoreCase))
			{
				int eq = arg.IndexOf('=');
				scanBatchSize = Math.Max(1, int.Parse(arg.AsSpan(eq + 1)));
			}
			else if (arg.StartsWith("--block-size=", StringComparison.OrdinalIgnoreCase))
			{
				int eq = arg.IndexOf('=');
				blockSize = Math.Max(1, int.Parse(arg.AsSpan(eq + 1)));
			}
			else if (arg.StartsWith("--filter-p=", StringComparison.OrdinalIgnoreCase))
			{
				int eq = arg.IndexOf('=');
				filterFile = arg.AsSpan(eq + 1).ToString();
			}
			else if (arg.StartsWith("--write-batch-size=", StringComparison.OrdinalIgnoreCase))
			{
				int eq = arg.IndexOf('=');
				_writeBatchSize = Math.Max(1, int.Parse(arg.AsSpan(eq + 1)));
			}
			if (arg.StartsWith("--divisor-cycles="))
			{
				int eq = arg.IndexOf('=');
				cyclesPath = arg.AsSpan(eq + 1).ToString();
			}
			else if (arg.StartsWith("--divisor-cycles-device=", StringComparison.OrdinalIgnoreCase))
			{
				int eq = arg.IndexOf('=');
				ReadOnlySpan<char> value = arg.AsSpan(eq + 1);
				useGpuCycles = !value.Equals("cpu", StringComparison.OrdinalIgnoreCase);
			}
			else if (arg.StartsWith("--divisor-cycles-batch-size=", StringComparison.OrdinalIgnoreCase))
			{
				int eq = arg.IndexOf('=');
				cyclesBatchSize = Math.Max(1, int.Parse(arg.AsSpan(eq + 1)));
			}
			else if (arg.Equals("--divisor-cycles-continue", StringComparison.OrdinalIgnoreCase))
			{
				continueCyclesGeneration = true;
			}
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

		Console.WriteLine("Divisor cycles are ready");

		_useDivisor = useDivisor;
		_divisor = divisor;
		_transformP = useBitTransform ? &TransformPBit : &TransformPAdd;
		PResidue = new ThreadLocal<ModResidueTracker>(() => new ModResidueTracker(ResidueModel.Identity, initialNumber: currentP, initialized: true), trackAllValues: true);
		PrimeTesters = new ThreadLocal<PrimeTester>(() => new PrimeTester(), trackAllValues: true);
		// Note: --primes-device controls default device for library kernels; p primality remains CPU here.
		// Initialize per-thread p residue tracker (Identity model) at currentP
		if (!useDivisor)
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
                else
                {
                        if (divisor == UInt128.Zero)
                        {
                                MersenneNumberDivisorGpuTester.BuildDivisorCandidates();
                        }

                        _divisorTester = new MersenneNumberDivisorGpuTester();
                }

		// Load RLE blacklist (optional)
		if (!string.IsNullOrEmpty(_rleBlacklistPath))
		{
			RleBlacklist.Load(_rleBlacklistPath!);
		}

		ulong remainder = currentP % 6UL;
		if (currentP == InitialP && string.IsNullOrEmpty(filterFile))
		{
			// bool isPerfect = IsEvenPerfectCandidate(InitialP, out bool searchedMersenne, out bool detailedCheck);
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
			LoadResultsFile(ResultsFileName, (p, detailedCheck, isPerfect) =>
			{
				if (detailedCheck && isPerfect)
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
		ulong maxP = 0UL;
		if (useFilter)
		{
			Console.WriteLine("Loading filter...");
			LoadResultsFile(filterFile, (p, detailedCheck, isPerfect) =>
			{
				if (isPerfect)
				{
					Console.WriteLine($"Adding {p}");
					filter.Add(p);
					if (p > maxP)
					{
						maxP = p;
					}
				}
			});

			// Restore this if you want to use List<ulong> instead of HashSet<ulong>
			// filter.Sort();
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

		if (!useDivisor)
		{
			Console.WriteLine("Warming up orders...");

			threadCount = Math.Max(1, threadCount);
			_ = MersenneTesters.Value;
			if (!useLucas)
			{
				MersenneTesters.Value!.WarmUpOrders(currentP, _orderWarmupLimitOverride ?? 5_000_000UL);
			}
		}

		Console.WriteLine("Starting scan...");
		_state = ((long)currentP << 3) | (long)remainder;
		Task[] tasks = new Task[threadCount];

		for (int i = 0; i < threadCount; i++)
		{
			tasks[i] = Task.Run(() =>
			{
				int count, j;
				ulong p;
				bool isPerfect, searchedMersenne, detailedCheck;
				ulong[] buffer = new ulong[blockSize];
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
							isPerfect = IsEvenPerfectCandidate(p, divisorCyclesSearchLimit, out searchedMersenne, out detailedCheck);
							PrintResult(p, searchedMersenne, detailedCheck, isPerfect);
						}
					}
					else
					{
						for (j = 0; j < count && !Volatile.Read(ref _limitReached); j++)
						{
							p = buffer[j];
							if (useFilter && !filter.Contains(p))
							{
								continue;
							}
							else
							{
								Console.WriteLine($"Testing {p}");
							}

							isPerfect = IsEvenPerfectCandidate(p, divisorCyclesSearchLimit, out searchedMersenne, out detailedCheck);
							PrintResult(p, searchedMersenne, detailedCheck, isPerfect);
							if (p == maxP)
							{
								Volatile.Write(ref _limitReached, true);
								break;
							}
						}
					}
				}
			});
		}

		Task.WaitAll(tasks);
		FlushBuffer();
		StringBuilderPool.Return(_outputBuilder!);
	}

	private static unsafe void LoadResultsFile(string resultsFileName, Action<ulong, bool, bool> lineProcessorAction)
	{
		using FileStream readStream = new(resultsFileName, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
		using StreamReader reader = new(readStream);
		string? line;
		bool headerSkipped = false;
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

			ReadOnlySpan<char> span = line.AsSpan();
			int first = span.IndexOf(',');
			if (first < 0)
			{
				continue;
			}

			ulong p = ulong.Parse(span[..first]);
			span = span[(first + 1)..];
			int second = span.IndexOf(',');
			if (second < 0)
			{
				continue;
			}

			span = span[(second + 1)..];
			int third = span.IndexOf(',');
			if (third < 0)
			{
				continue;
			}

			ReadOnlySpan<char> detailedSpan = span[..third];
			ReadOnlySpan<char> perfectSpan = span[(third + 1)..];

			if (bool.TryParse(detailedSpan, out bool detailed) && bool.TryParse(perfectSpan, out bool isPerfect))
			{
				lineProcessorAction(p, detailed, isPerfect);
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
		Console.WriteLine("  --mersenne=pow2mod|incremental|lucas|residue|divisor  Mersenne test method");
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

		Console.WriteLine("  --use-order            test primality via q order");
		Console.WriteLine("  --workaround-mod       avoid '%' operator on the GPU");
		// mod-automaton removed
		Console.WriteLine("  --filter-p=<path>      process only p from previous run results");
		Console.WriteLine("  --write-batch-size=<value> overwrite frequency of disk writes (default 100 lines)");
		Console.WriteLine("  --gcd-filter           enable early sieve based on GCD");
		Console.WriteLine("  --help, -help, --?, -?, /?   show this help message");
	}

	private static string BuildResultsFileName(bool bitInc, int threads, int block, GpuKernelType kernelType, bool useLucasFlag, bool useDivisorFlag, bool mersenneOnGpu, bool useOrder, bool useModWorkaround, bool useGcd, NttBackend nttBackend, int gpuPrimeThreads, int llSlice, int gpuScanBatch, ulong warmupLimit, ModReductionMode reduction, string mersenneDevice, string primesDevice, string orderDevice)
	{
		string inc = bitInc ? "bit" : "add";
		string mers = useDivisorFlag ? "divisor" : (useLucasFlag ? "lucas" : (kernelType == GpuKernelType.Pow2Mod ? "pow2mod" : "incremental"));
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
	private static void PrintResult(ulong currentP, bool searchedMersenne, bool detailedCheck, bool isPerfect)
	{
		if (isPerfect)
		{
			int newCount = Interlocked.Increment(ref _primeCount);
			if (newCount >= 2)
			{
				Volatile.Write(ref _primeFoundAfterInit, true);
			}
		}

		bool primeFlag = false, printToConsole = false;

		StringBuilder localBuilder = StringBuilderPool.Rent();
		_ = localBuilder
			.Append(currentP).Append(',')
			.Append(searchedMersenne).Append(',')
			.Append(detailedCheck).Append(',')
			.Append(isPerfect).Append('\n');

		lock (Sync)
		{
			_ = _outputBuilder!.Append(localBuilder);

			if (_consoleCounter >= ConsoleInterval)
			{
				printToConsole = true;
				primeFlag = _primeFoundAfterInit;
				_consoleCounter = 0;
			}
			else
			{
				_consoleCounter++;
			}

			_writeIndex++;
			if (_writeIndex >= _writeBatchSize)
			{
				FlushBuffer();
			}
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

		StringBuilderPool.Return(localBuilder);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void FlushBuffer()
	{
		var stream = new FileStream(ResultsFileName, FileMode.Append, FileAccess.Write, FileShare.Read);
		var writer = new StreamWriter(stream) { AutoFlush = false };

		if (_writeIndex > 0)
		{
			writer.Write(_outputBuilder);
			writer.Flush();
			_outputBuilder!.Clear();
			_writeIndex = 0;
		}

		writer.Dispose();
		stream.Dispose();
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	internal static ulong CountOnes(ulong value)
	{
		return (ulong)BitOperations.PopCount(value);
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
		remainder -= (remainder >= 6UL) ? 6UL : 0UL;
		remainder -= (remainder >= 6UL) ? 6UL : 0UL;

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
		remainder -= (remainder >= 6UL) ? 6UL : 0UL;

		return next + value;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	internal static ulong TransformPAdd(ulong value, ref ulong remainder)
	{
		ulong next = value;
		value = remainder switch
		{
			0UL => 1UL,
			1UL => 4UL,
			2UL => 3UL,
			3UL => 2UL,
			4UL => 1UL,
			_ => 2UL,
		}; // 'value' now holds diff

		if (next > ulong.MaxValue - value)
		{
			Volatile.Write(ref _limitReached, true);
			return next;
		}

		remainder += value;
		remainder -= (remainder >= 6UL) ? 6UL : 0UL;

		return next + value;
	}

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsEvenPerfectCandidate(ulong p, ulong divisorCyclesSearchLimit) => IsEvenPerfectCandidate(p, divisorCyclesSearchLimit, out _, out _);

        internal static bool IsEvenPerfectCandidate(ulong p, ulong divisorCyclesSearchLimit, out bool searchedMersenne, out bool detailedCheck)
	{
		searchedMersenne = false;
		detailedCheck = false;

		if (_useGcdFilter && IsCompositeByGcd(p))
		{
			return false;
		}

		// Optional: RLE blacklist and binary-threshold filters on p (safe only when configured)
		if (p <= _rleHardMaxP)
		{
			if (!_rleOnlyLast7 || (p % 10UL) == 7UL)
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
			ComputeBitStats(p, out int bitLen, out int zeroCount, out int maxZeroBlock);
			if (_zeroFracHard >= 0)
			{
				zf = (double)zeroCount / bitLen;
				if (zf > _zeroFracHard)
				{
					return false;
				}
			}

			if (_zeroFracConj >= 0 && _maxZeroConj >= 0)
			{
				zf = (double)zeroCount / bitLen;
				if (zf > _zeroFracConj && maxZeroBlock >= _maxZeroConj)
				{
					return false;
				}
			}
		}

		// Fast residue-based composite check for p using small primes
		if (!_useDivisor && IsCompositeByResidues(p))
		{
			searchedMersenne = false;
			detailedCheck = false;
			return false;
		}

		// If primes-device=gpu, route p primality through GPU-assisted sieve (optional MR later)
		// TODO: Add deterministic Miller–Rabin rounds in GPU path for 64-bit range.
		bool isPrimeP = GpuContextPool.ForceCpu
			? PrimeTesters.Value!.IsPrime(p, CancellationToken.None)
			: PrimeTesters.Value!.IsPrimeGpu(p, CancellationToken.None);
		if (!isPrimeP)
		{
			return false;
		}

                searchedMersenne = true;
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
		int i, len = primes.Length;
		bool divisible;
		for (i = 0; i < len; i++)
		{
			if (primesPow2[i] > p) { break; }

			if (tracker.MergeOrAppend(p, primes[i], out divisible) && divisible)
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

		for (int bi = msbIndex; bi >= 0; bi--)
		{
			byte b = (byte)(value >> (bi * 8));
			if (bi == msbIndex && bitsInTopByte < 8)
			{
				// Mask off leading unused bits by setting them to 1 (ignore)
				b |= (byte)(0xFF << bitsInTopByte);
			}

			int zc = ByteZeroCount[b];
			zeroCount += zc;

			if (zc == 8)
			{
				currentRun += 8;
				if (currentRun > maxZeroBlock)
				{
					maxZeroBlock = currentRun;
				}

				continue;
			}

			int pref = BytePrefixZero[b];
			int suff = ByteSuffixZero[b];
			int maxIn = ByteMaxZeroRun[b];

			int candidate = currentRun + pref;
			if (candidate > maxZeroBlock)
			{
				maxZeroBlock = candidate;
			}

			if (maxIn > maxZeroBlock)
			{
				maxZeroBlock = maxIn;
			}

			currentRun = suff;
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
		for (int v = 0; v < 256; v++)
		{
			int zeros = 0;
			int pref = 0;
			int suff = 0;
			int maxIn = 0;
			int run = 0;

			// MSB-first within byte: bit 7 .. bit 0
			for (int bit = 7; bit >= 0; bit--)
			{
				bool isZero = ((v >> bit) & 1) == 0;
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
					int t = 0;
					for (int b2 = 7; b2 >= 0; b2--)
					{
						if (((v >> b2) & 1) == 0) { t++; } else { break; }
					}
					pref = t;
				}

				if (bit == 0)
				{
					// trailing zeros (suffix)
					int t = 0;
					for (int b2 = 0; b2 < 8; b2++)
					{
						if (((v >> b2) & 1) == 0) { t++; } else { break; }
					}
					suff = t;
				}
			}

			ByteZeroCount[v] = (byte)zeros;
			BytePrefixZero[v] = (byte)pref;
			ByteSuffixZero[v] = (byte)suff;
			ByteMaxZeroRun[v] = (byte)maxIn;
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
