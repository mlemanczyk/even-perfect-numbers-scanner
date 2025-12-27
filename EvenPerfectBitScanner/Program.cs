using System.Buffers;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using EvenPerfectBitScanner.Candidates;
using EvenPerfectBitScanner.IO;
using Open.Collections;
using PerfectNumbers.Core;
using PerfectNumbers.Core.Cpu;
using PerfectNumbers.Core.Gpu;
using PerfectNumbers.Core.Gpu.Accelerators;
using MersenneNumberDivisorByDivisorCpuTesterWithCpuOrder = PerfectNumbers.Core.Cpu.MersenneNumberDivisorByDivisorCpuTesterWithForOneByOneDivisorSetForCpuOrder;
using MersenneNumberDivisorByDivisorCpuTesterWithHybridOrder = PerfectNumbers.Core.Cpu.MersenneNumberDivisorByDivisorCpuTesterWithForOneByOneDivisorSetForHybridOrder;
using MersenneNumberDivisorByDivisorCpuTesterWithGpuOrder = PerfectNumbers.Core.Cpu.MersenneNumberDivisorByDivisorCpuTesterWithForOneByOneDivisorSetForGpuOrder;
using ByDivisorCpuOrderPow2 = PerfectNumbers.Core.Cpu.MersenneNumberDivisorByDivisorCpuTesterWithForPow2GroupsDivisorSetForCpuOrder;
using ByDivisorHybridOrderPow2 = PerfectNumbers.Core.Cpu.MersenneNumberDivisorByDivisorCpuTesterWithForPow2GroupsDivisorSetForHybridOrder;
using ByDivisorGpuOrderPow2 = PerfectNumbers.Core.Cpu.MersenneNumberDivisorByDivisorCpuTesterWithForPow2GroupsDivisorSetForGpuOrder;
using System.Globalization;

namespace EvenPerfectBitScanner;

internal static class Program
{
	// private static ThreadLocal<PrimeTester> PrimeTesters = null!;
private static ThreadLocal<object>? MersenneTesters;
	private const ulong InitialP = PerfectNumberConstants.BiggestKnownEvenPerfectP;
	private static MersenneNumberDivisorGpuTester? _divisorTester;
	private static UInt128 _divisor;
	private static MersenneNumberDivisorByDivisorCpuTesterWithCpuOrder _byDivisorCpuOrderTester;
	private static MersenneNumberDivisorByDivisorCpuTesterWithHybridOrder _byDivisorHybridOrderTester;
	private static MersenneNumberDivisorByDivisorCpuTesterWithGpuOrder _byDivisorGpuOrderTester;
	private static ByDivisorCpuOrderPow2 _byDivisorCpuOrderTesterPow2;
	private static ByDivisorHybridOrderPow2 _byDivisorHybridOrderTesterPow2;
	private static ByDivisorGpuOrderPow2 _byDivisorGpuOrderTesterPow2;
	private static Dictionary<ulong, (bool DetailedCheck, bool PassedAllTests)>? _byDivisorPreviousResults;
	private static bool _limitReached;
	private static ulong _byDivisorStartPrime;
	private static CliArguments _cliArguments;
	private static int _testTargetPrimeCount;
	private static int _testProcessedPrimeCount;
	private static ComputationDevice _mersenneDevice;
	private static OrderDevice _orderCalculationDevice;
	private static CacheStatus _cacheStatus;
	private static CalculationMethod _calculationMethod;

	[ThreadStatic]
	private static bool _lastCompositeP;

	private static bool _runPrimesOnCpu;

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
			_divisor = UInt128.Zero;

			if (_cliArguments.ShowHelp)
			{
				CliArguments.PrintHelp();
				return;
			}

		Console.WriteLine("Initializing");
		int gpuPrimeThreads = Math.Max(1, _cliArguments.GpuPrimeThreads);
		int threadCount = Math.Max(1, _cliArguments.ThreadCount);

		EnvironmentConfiguration.MinK = _cliArguments.MinK;
		EnvironmentConfiguration.GpuBatchSize = Math.Max(1, _cliArguments.ScanBatchSize);
		EnvironmentConfiguration.GpuRatio = _cliArguments.GpuRatio;
		EnvironmentConfiguration.OrderDevice = _cliArguments.OrderDevice;
		EnvironmentConfiguration.UsePow2GroupDivisors = _cliArguments.UseDivisorPow2Increment;
		EnvironmentConfiguration.ByDivisorSpecialRange = _cliArguments.ByDivisorSpecialRange;
		EnvironmentConfiguration.RollingAccelerators = Math.Min(Math.Min(PerfectNumberConstants.RollingAccelerators, gpuPrimeThreads), threadCount);
		EnvironmentConfiguration.Initialize();

			// NttGpuMath.GpuTransformBackend = _cliArguments.NttBackend;
			// NttGpuMath.ReductionMode = _cliArguments.ModReductionMode;
			_runPrimesOnCpu = _cliArguments.ForcePrimeKernelsOnCpu;

			ulong currentP = _cliArguments.StartPrime;
			ulong remainder = currentP % 6UL;
			bool startPrimeProvided = _cliArguments.StartPrimeProvided;
			int gpuPrimeBatch = Math.Max(1, _cliArguments.GpuPrimeBatch);
			// int gpuPrimeThreads = SharedGpuContext.Device.MaxNumThreads;
			Console.WriteLine("Setting up parameters");
			UnboundedTaskScheduler.ConfigureThreadCount(threadCount);
			PrimeTester.GpuBatchSize = gpuPrimeBatch;
			GpuPrimeWorkLimiter.SetLimit(gpuPrimeThreads);
			Console.WriteLine("Warming up GPU kernels");
			PrimeTester.WarmUpGpuKernels(EnvironmentConfiguration.RollingAccelerators);
			// PrimeTester.WarmUpGpuKernels(gpuPrimeThreads >> 4);
			Console.WriteLine("Starting up threads...");
			_ = UnboundedTaskScheduler.Instance;
			int blockSize = Math.Max(1, _cliArguments.BlockSize);
			GpuKernelType kernelType = _cliArguments.KernelType;
			bool useBitTransform = _cliArguments.UseBitTransform;
			bool useOrder = _cliArguments.UseOrder;
			bool useResidue = _cliArguments.UseResidue;
			bool useDivisor = _cliArguments.UseDivisor;
			bool useByDivisor = _cliArguments.UseByDivisor;
			bool useLucas = _cliArguments.UseLucas;
			bool testMode = _cliArguments.TestMode;
			bool useGpuCycles = _cliArguments.UseGpuCycles;
			PrimeTransformMode transformMode = _cliArguments.UseBitTransform
					? PrimeTransformMode.Bit
					: (useResidue ? PrimeTransformMode.Add : PrimeTransformMode.AddPrimes);
			CandidatesCalculator.Configure(transformMode);
			ComputationDevice mersenneDevice = _cliArguments.MersenneDevice;
			_mersenneDevice = mersenneDevice;
			_calculationMethod = useResidue
				? CalculationMethod.Residue
			: (useLucas
				? CalculationMethod.LucasLehmer
				: (useOrder ? CalculationMethod.Divisor : CalculationMethod.Incremental));
			bool mersenneOnGpu = mersenneDevice == ComputationDevice.Gpu;
			OrderDevice orderCalculationDevice = EnvironmentConfiguration.UseGpuOrder
				? OrderDevice.Gpu
				: (EnvironmentConfiguration.UseHybridOrder ? OrderDevice.Hybrid : OrderDevice.Cpu);
			_orderCalculationDevice = orderCalculationDevice;
			_cacheStatus = CacheStatus.Disabled;
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

			_byDivisorStartPrime = useByDivisor && startPrimeProvided ? currentP : 0UL;

			if (useByDivisor && !testMode && string.IsNullOrEmpty(filterFile))
			{
				Console.WriteLine("--mersenne=bydivisor requires --filter-p=<path>.");
				return;
			}

			Directory.CreateDirectory(PerfectNumberConstants.ByDivisorStateDirectory);


			if (!File.Exists(cyclesPath) || continueCyclesGeneration)
			{
				Console.WriteLine($"Generating divisor cycles '{cyclesPath}' for all p <= {PerfectNumberConstants.MaxQForDivisorCycles}...");

				long nextPosition = 0L, completeCount = 0L;
				if (continueCyclesGeneration && File.Exists(cyclesPath))
				{
					Console.WriteLine("Finding last divisor position...");
					(nextPosition, completeCount) = useGpuCycles
						? MersenneDivisorCyclesGpu.FindLast(cyclesPath)
						: MersenneDivisorCyclesCpu.FindLast(cyclesPath);
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
					MersenneDivisorCyclesGpu.GenerateGpu(cyclesPath, PerfectNumberConstants.MaxQForDivisorCycles, cyclesBatchSize, skipCount: completeCount, nextPosition: nextPosition);
				}
				else
				{
					MersenneDivisorCyclesCpu.Generate(cyclesPath, PerfectNumberConstants.MaxQForDivisorCycles);
				}
			}

			Console.WriteLine($"Loading divisor cycles into memory...");
			if (string.Equals(Path.GetExtension(cyclesPath), ".csv", StringComparison.OrdinalIgnoreCase))
			{
				if (useGpuCycles)
				{
					MersenneDivisorCyclesGpu.Shared.LoadFrom(cyclesPath);
				}
				else
				{
					MersenneDivisorCyclesCpu.Shared.LoadFrom(cyclesPath);
				}
			}
			else
			{
				if (useGpuCycles)
				{
					MersenneDivisorCyclesGpu.Shared.LoadFrom(cyclesPath);
				}
				else
				{
					MersenneDivisorCyclesCpu.Shared.LoadFrom(cyclesPath);
				}
			}

			DivisorCycleCache.SetDivisorCyclesBatchSize(cyclesBatchSize);
			// TODO: Keep a single cached block loaded from disk and honor the configured device when
			// computing ad-hoc cycles for divisors that fall outside that snapshot instead of queuing
			// generation of additional blocks.
			DivisorCycleCache.Shared.RefreshSnapshot();
			// TODO: Stop reloading the full snapshot once the ad-hoc path streams results straight from
			// the configured device without persisting them, so memory stays bounded while preserving
			// the single cached block strategy.

			Console.WriteLine("Divisor cycles are ready");

			blockSize = useByDivisor ? 1 : blockSize;

			CandidatesCalculator.Initialize(currentP);
			// PrimeTesters = new ThreadLocal<PrimeTester>(() => new PrimeTester(), trackAllValues: true);
			// Note: --primes-device controls default device for library kernels; p primality remains CPU here.
			// Initialize per-thread p residue tracker (Identity model) at currentP
			if (!useDivisor && !_cliArguments.UseByDivisor)
			{
				MersenneTesters = new ThreadLocal<object>(() =>
				{
					object tester = CreateMersenneTester(
						_calculationMethod,
						_cacheStatus,
						_orderCalculationDevice,
						mersenneDevice,
						kernelType,
						residueKMax);
					if (!useLucas)
					{
						Console.WriteLine("Warming up orders");
						WarmUpOrders(tester, _orderCalculationDevice, _mersenneDevice, currentP, orderWarmupLimitOverride ?? 5_000_000UL);
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
				if (mersenneOnGpu)
				{
					throw new NotSupportedException("--mersenne-device=gpu is not supported for --mersenne=bydivisor on this execution path.");
				}

				bool usePow2Divisors = _cliArguments.UseDivisorPow2Increment;
				if (EnvironmentConfiguration.UseCpuOrder)
				{
					if (usePow2Divisors)
					{
						_byDivisorCpuOrderTesterPow2 = new ByDivisorCpuOrderPow2();
					}
					else
					{
						_byDivisorCpuOrderTester = new MersenneNumberDivisorByDivisorCpuTesterWithCpuOrder();
					}
				}
				else if (EnvironmentConfiguration.UseHybridOrder)
				{
					if (usePow2Divisors)
					{
						_byDivisorHybridOrderTesterPow2 = new ByDivisorHybridOrderPow2();
					}
					else
					{
						_byDivisorHybridOrderTester = new MersenneNumberDivisorByDivisorCpuTesterWithHybridOrder();
					}
				}
				else
				{
					if (usePow2Divisors)
					{
						_byDivisorGpuOrderTesterPow2 = new ByDivisorGpuOrderPow2();
					}
					else
					{
						_byDivisorGpuOrderTester = new MersenneNumberDivisorByDivisorCpuTesterWithGpuOrder();
					}
				}
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
					currentP = CandidatesCalculator.AdvancePrime(currentP, ref remainder, ref _limitReached);
				}

				if (Volatile.Read(ref _limitReached))
				{
					return;
				}
			}

			Console.WriteLine("Initialization...");
			// Compose a results file name that encodes configuration (before opening file)
			string mersenneDeviceLabel = mersenneDevice switch
			{
				ComputationDevice.Hybrid => "hybrid",
				ComputationDevice.Gpu => "gpu",
				_ => "cpu",
			};

			string orderDeviceLabel = _orderCalculationDevice switch
			{
				OrderDevice.Hybrid => "hybrid",
				OrderDevice.Gpu => "gpu",
				_ => "cpu",
			};

			var builtName = CalculationResultsFile.BuildFileName(
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
							EnvironmentConfiguration.GpuBatchSize,
							orderWarmupLimitOverride ?? 5_000_000UL,
							NttGpuMath.ReductionMode,
							mersenneDeviceLabel,
							_runPrimesOnCpu ? "cpu" : "gpu",
							orderDeviceLabel);

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
				CalculationResultsFile.EnumerateCandidates(resultsFileName, (p, detailedCheck, passedAllTests) =>
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
					byDivisorCandidates = CandidatesCalculator.BuildTestPrimeCandidates(testPrimeCandidateLimit);
				}
				else
				{
					Console.WriteLine("Loading filter...");
					int skippedByLimit = 0;
					byDivisorCandidates = CalculationResultsFile.LoadCandidatesWithinRange(filterFile, maxPrimeLimit, maxPrimeConfigured, out skippedByLimit);
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
				CalculationResultsFile.EnumerateCandidates(filterFile, (p, detailedCheck, passedAllTests) =>
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

			_ = AcceleratorPool.Shared;

			Stopwatch? stopwatch = null;
			if (testMode)
			{
				stopwatch = Stopwatch.StartNew();
			}

			Console.WriteLine("Starting scan...");

			if (useByDivisor)
			{
				int byDivisorPrimeRange = Math.Max(1, _cliArguments.BlockSize);
				bool usePow2Divisors = _cliArguments.UseDivisorPow2Increment;
				if (EnvironmentConfiguration.UseCpuOrder)
				{
					if (usePow2Divisors)
					{
						Console.WriteLine("Using by-divisor pow2groups increment (CPU order).");
						MersenneNumberDivisorByDivisorTester.Run(
							byDivisorCandidates,
							ref _byDivisorCpuOrderTesterPow2,
							_byDivisorPreviousResults,
							_byDivisorStartPrime,
							static () => _lastCompositeP = true,
							static () => _lastCompositeP = false,
							static (candidate, searchedMersenne, detailedCheck, passedAllTests, divisor) => CalculationResultHandler.HandleResult(candidate, divisor, searchedMersenne, detailedCheck, passedAllTests, _lastCompositeP),
							threadCount,
							byDivisorPrimeRange);
					}
					else
					{
						MersenneNumberDivisorByDivisorTester.Run(
							byDivisorCandidates,
							ref _byDivisorCpuOrderTester,
							_byDivisorPreviousResults,
							_byDivisorStartPrime,
							static () => _lastCompositeP = true,
							static () => _lastCompositeP = false,
							static (candidate, searchedMersenne, detailedCheck, passedAllTests, divisor) => CalculationResultHandler.HandleResult(candidate, divisor, searchedMersenne, detailedCheck, passedAllTests, _lastCompositeP),
							threadCount,
							byDivisorPrimeRange);
					}
				}
				else if (EnvironmentConfiguration.UseHybridOrder)
				{
					if (usePow2Divisors)
					{
						Console.WriteLine("Using by-divisor pow2groups increment (Hybrid order).");
						MersenneNumberDivisorByDivisorTester.Run(
							byDivisorCandidates,
							ref _byDivisorHybridOrderTesterPow2,
							_byDivisorPreviousResults,
							_byDivisorStartPrime,
							static () => _lastCompositeP = true,
							static () => _lastCompositeP = false,
							static (candidate, searchedMersenne, detailedCheck, passedAllTests, divisor) => CalculationResultHandler.HandleResult(candidate, divisor, searchedMersenne, detailedCheck, passedAllTests, _lastCompositeP),
							threadCount,
							byDivisorPrimeRange);
					}
					else
					{
						MersenneNumberDivisorByDivisorTester.Run(
							byDivisorCandidates,
							ref _byDivisorHybridOrderTester,
							_byDivisorPreviousResults,
							_byDivisorStartPrime,
							static () => _lastCompositeP = true,
							static () => _lastCompositeP = false,
							static (candidate, searchedMersenne, detailedCheck, passedAllTests, divisor) => CalculationResultHandler.HandleResult(candidate, divisor, searchedMersenne, detailedCheck, passedAllTests, _lastCompositeP),
							threadCount,
							byDivisorPrimeRange);
					}
				}
				else
				{
					if (usePow2Divisors)
					{
						Console.WriteLine("Using by-divisor pow2groups increment (GPU order).");
						MersenneNumberDivisorByDivisorTester.Run(
							byDivisorCandidates,
							ref _byDivisorGpuOrderTesterPow2,
							_byDivisorPreviousResults,
							_byDivisorStartPrime,
							static () => _lastCompositeP = true,
							static () => _lastCompositeP = false,
							static (candidate, searchedMersenne, detailedCheck, passedAllTests, divisor) => CalculationResultHandler.HandleResult(candidate, divisor, searchedMersenne, detailedCheck, passedAllTests, _lastCompositeP),
							threadCount,
							byDivisorPrimeRange);
					}
					else
					{
						MersenneNumberDivisorByDivisorTester.Run(
							byDivisorCandidates,
							ref _byDivisorGpuOrderTester,
							_byDivisorPreviousResults,
							_byDivisorStartPrime,
							static () => _lastCompositeP = true,
							static () => _lastCompositeP = false,
							static (candidate, searchedMersenne, detailedCheck, passedAllTests, divisor) => CalculationResultHandler.HandleResult(candidate, divisor, searchedMersenne, detailedCheck, passedAllTests, _lastCompositeP),
							threadCount,
							byDivisorPrimeRange);
					}
				}

				CalculationResultHandler.FlushBuffer();
				CalculationResultHandler.ReleaseOutputBuffer();
				if (stopwatch is not null)
				{
					stopwatch.Stop();
					CalculationTestTime.Report(stopwatch.Elapsed, _cliArguments.ResultsDirectory);
				}

				return;
			}

			if (!useDivisor && !_cliArguments.UseByDivisor)
			{
				threadCount = Math.Max(1, threadCount);
				// Don't remove this. It's to initialize the testers
				_ = MersenneTesters.Value;
				if (!useLucas)
				{
					Console.WriteLine("Warming up orders...");
					WarmUpOrders(MersenneTesters.Value!, _orderCalculationDevice, _mersenneDevice, currentP, orderWarmupLimitOverride ?? 5_000_000UL);
				}
			}

			if (testMode)
			{
				testPrimeCandidateLimit = Math.Max(1, threadCount) * 3;
				_testTargetPrimeCount = testPrimeCandidateLimit;
			}

			CandidatesCalculator.InitializeState(currentP, remainder);

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
					PrimeOrderCalculatorAccelerator gpu = PrimeOrderCalculatorAccelerator.Rent(1);
					while (!Volatile.Read(ref _limitReached))
					{
						count = CandidatesCalculator.ReserveBlock(buffer, blockSize, ref _limitReached);
						if (count == 0)
						{
							break;
						}

						if (!useFilter)
						{
							for (j = 0; j < count && !Volatile.Read(ref _limitReached); j++)
							{
								p = buffer[j];
								passedAllTests = IsEvenPerfectCandidate(gpu, p, divisorCyclesSearchLimit, out searchedMersenne, out detailedCheck);
								CalculationResultHandler.HandleResult(p, 0UL, searchedMersenne, detailedCheck, passedAllTests, _lastCompositeP);
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

							passedAllTests = IsEvenPerfectCandidate(gpu, p, divisorCyclesSearchLimit, out searchedMersenne, out detailedCheck);
							CalculationResultHandler.HandleResult(p, 0UL, searchedMersenne, detailedCheck, passedAllTests, _lastCompositeP);

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
					PrimeOrderCalculatorAccelerator.Return(gpu);
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
				CalculationTestTime.Report(testElapsed, _cliArguments.ResultsDirectory);
			}
		}
		finally
		{
			CalculationResultHandler.Dispose();
		}
	}

	private static bool IsMersennePrime(PrimeOrderCalculatorAccelerator gpu, ulong exponent)
	{
		object tester = MersenneTesters!.Value!;
		return (_calculationMethod, _orderCalculationDevice, _mersenneDevice) switch
		{
			(CalculationMethod.Residue, OrderDevice.Gpu, ComputationDevice.Gpu) => ((MersenneNumberTesterForResidueCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForGpu)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.Residue, OrderDevice.Gpu, ComputationDevice.Hybrid) => ((MersenneNumberTesterForResidueCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForHybrid)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.Residue, OrderDevice.Gpu, _) => ((MersenneNumberTesterForResidueCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForCpu)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.Residue, OrderDevice.Hybrid, ComputationDevice.Gpu) => ((MersenneNumberTesterForResidueCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForGpu)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.Residue, OrderDevice.Hybrid, ComputationDevice.Hybrid) => ((MersenneNumberTesterForResidueCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForHybrid)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.Residue, OrderDevice.Hybrid, _) => ((MersenneNumberTesterForResidueCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForCpu)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.Residue, _, ComputationDevice.Gpu) => ((MersenneNumberTesterForResidueCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForGpu)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.Residue, _, ComputationDevice.Hybrid) => ((MersenneNumberTesterForResidueCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForHybrid)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.Residue, _, _) => ((MersenneNumberTesterForResidueCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForCpu)tester).IsMersennePrime(gpu, exponent),

			(CalculationMethod.LucasLehmer, OrderDevice.Gpu, ComputationDevice.Gpu) => ((MersenneNumberTesterForLucasLehmerCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForGpu)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.LucasLehmer, OrderDevice.Gpu, ComputationDevice.Hybrid) => ((MersenneNumberTesterForLucasLehmerCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForHybrid)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.LucasLehmer, OrderDevice.Gpu, _) => ((MersenneNumberTesterForLucasLehmerCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForCpu)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.LucasLehmer, OrderDevice.Hybrid, ComputationDevice.Gpu) => ((MersenneNumberTesterForLucasLehmerCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForGpu)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.LucasLehmer, OrderDevice.Hybrid, ComputationDevice.Hybrid) => ((MersenneNumberTesterForLucasLehmerCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForHybrid)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.LucasLehmer, OrderDevice.Hybrid, _) => ((MersenneNumberTesterForLucasLehmerCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForCpu)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.LucasLehmer, _, ComputationDevice.Gpu) => ((MersenneNumberTesterForLucasLehmerCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForGpu)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.LucasLehmer, _, ComputationDevice.Hybrid) => ((MersenneNumberTesterForLucasLehmerCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForHybrid)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.LucasLehmer, _, _) => ((MersenneNumberTesterForLucasLehmerCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForCpu)tester).IsMersennePrime(gpu, exponent),

			(CalculationMethod.Divisor, OrderDevice.Gpu, ComputationDevice.Gpu) => ((MersenneNumberTesterForDivisorCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForGpu)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.Divisor, OrderDevice.Gpu, ComputationDevice.Hybrid) => ((MersenneNumberTesterForDivisorCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForHybrid)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.Divisor, OrderDevice.Gpu, _) => ((MersenneNumberTesterForDivisorCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForCpu)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.Divisor, OrderDevice.Hybrid, ComputationDevice.Gpu) => ((MersenneNumberTesterForDivisorCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForGpu)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.Divisor, OrderDevice.Hybrid, ComputationDevice.Hybrid) => ((MersenneNumberTesterForDivisorCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForHybrid)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.Divisor, OrderDevice.Hybrid, _) => ((MersenneNumberTesterForDivisorCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForCpu)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.Divisor, _, ComputationDevice.Gpu) => ((MersenneNumberTesterForDivisorCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForGpu)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.Divisor, _, ComputationDevice.Hybrid) => ((MersenneNumberTesterForDivisorCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForHybrid)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.Divisor, _, _) => ((MersenneNumberTesterForDivisorCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForCpu)tester).IsMersennePrime(gpu, exponent),

			(CalculationMethod.ByDivisor, OrderDevice.Gpu, ComputationDevice.Gpu) => ((MersenneNumberTesterForByDivisorCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForGpu)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.ByDivisor, OrderDevice.Gpu, ComputationDevice.Hybrid) => ((MersenneNumberTesterForByDivisorCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForHybrid)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.ByDivisor, OrderDevice.Gpu, _) => ((MersenneNumberTesterForByDivisorCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForCpu)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.ByDivisor, OrderDevice.Hybrid, ComputationDevice.Gpu) => ((MersenneNumberTesterForByDivisorCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForGpu)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.ByDivisor, OrderDevice.Hybrid, ComputationDevice.Hybrid) => ((MersenneNumberTesterForByDivisorCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForHybrid)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.ByDivisor, OrderDevice.Hybrid, _) => ((MersenneNumberTesterForByDivisorCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForCpu)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.ByDivisor, _, ComputationDevice.Gpu) => ((MersenneNumberTesterForByDivisorCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForGpu)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.ByDivisor, _, ComputationDevice.Hybrid) => ((MersenneNumberTesterForByDivisorCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForHybrid)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.ByDivisor, _, _) => ((MersenneNumberTesterForByDivisorCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForCpu)tester).IsMersennePrime(gpu, exponent),

			(CalculationMethod.Pow2Mod, OrderDevice.Gpu, ComputationDevice.Gpu) => ((MersenneNumberTesterForPow2ModCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForGpu)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.Pow2Mod, OrderDevice.Gpu, ComputationDevice.Hybrid) => ((MersenneNumberTesterForPow2ModCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForHybrid)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.Pow2Mod, OrderDevice.Gpu, _) => ((MersenneNumberTesterForPow2ModCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForCpu)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.Pow2Mod, OrderDevice.Hybrid, ComputationDevice.Gpu) => ((MersenneNumberTesterForPow2ModCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForGpu)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.Pow2Mod, OrderDevice.Hybrid, ComputationDevice.Hybrid) => ((MersenneNumberTesterForPow2ModCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForHybrid)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.Pow2Mod, OrderDevice.Hybrid, _) => ((MersenneNumberTesterForPow2ModCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForCpu)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.Pow2Mod, _, ComputationDevice.Gpu) => ((MersenneNumberTesterForPow2ModCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForGpu)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.Pow2Mod, _, ComputationDevice.Hybrid) => ((MersenneNumberTesterForPow2ModCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForHybrid)tester).IsMersennePrime(gpu, exponent),
			(CalculationMethod.Pow2Mod, _, _) => ((MersenneNumberTesterForPow2ModCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForCpu)tester).IsMersennePrime(gpu, exponent),

			(_, OrderDevice.Gpu, ComputationDevice.Gpu) => ((MersenneNumberTesterForIncrementalCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForGpu)tester).IsMersennePrime(gpu, exponent),
			(_, OrderDevice.Gpu, ComputationDevice.Hybrid) => ((MersenneNumberTesterForIncrementalCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForHybrid)tester).IsMersennePrime(gpu, exponent),
			(_, OrderDevice.Gpu, _) => ((MersenneNumberTesterForIncrementalCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForCpu)tester).IsMersennePrime(gpu, exponent),
			(_, OrderDevice.Hybrid, ComputationDevice.Gpu) => ((MersenneNumberTesterForIncrementalCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForGpu)tester).IsMersennePrime(gpu, exponent),
			(_, OrderDevice.Hybrid, ComputationDevice.Hybrid) => ((MersenneNumberTesterForIncrementalCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForHybrid)tester).IsMersennePrime(gpu, exponent),
			(_, OrderDevice.Hybrid, _) => ((MersenneNumberTesterForIncrementalCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForCpu)tester).IsMersennePrime(gpu, exponent),
			(_, _, ComputationDevice.Gpu) => ((MersenneNumberTesterForIncrementalCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForGpu)tester).IsMersennePrime(gpu, exponent),
			(_, _, ComputationDevice.Hybrid) => ((MersenneNumberTesterForIncrementalCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForHybrid)tester).IsMersennePrime(gpu, exponent),
			_ => ((MersenneNumberTesterForIncrementalCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForCpu)tester).IsMersennePrime(gpu, exponent),
		};
	}

	private static object CreateMersenneTester(
		CalculationMethod method,
		CacheStatus cacheStatus,
		OrderDevice orderDevice,
		ComputationDevice mersenneDevice,
		GpuKernelType kernelType,
		ulong residueKMax)
	{
		if (cacheStatus != CacheStatus.Disabled)
		{
			throw new NotSupportedException("CacheStatus.Enabled is not wired into Program.cs yet.");
		}

		return (method, orderDevice, mersenneDevice) switch
		{
			(CalculationMethod.Residue, OrderDevice.Gpu, ComputationDevice.Gpu) => new MersenneNumberTesterForResidueCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForGpu(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.Residue, OrderDevice.Gpu, ComputationDevice.Hybrid) => new MersenneNumberTesterForResidueCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForHybrid(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.Residue, OrderDevice.Gpu, _) => new MersenneNumberTesterForResidueCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForCpu(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.Residue, OrderDevice.Hybrid, ComputationDevice.Gpu) => new MersenneNumberTesterForResidueCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForGpu(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.Residue, OrderDevice.Hybrid, ComputationDevice.Hybrid) => new MersenneNumberTesterForResidueCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForHybrid(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.Residue, OrderDevice.Hybrid, _) => new MersenneNumberTesterForResidueCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForCpu(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.Residue, _, ComputationDevice.Gpu) => new MersenneNumberTesterForResidueCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForGpu(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.Residue, _, ComputationDevice.Hybrid) => new MersenneNumberTesterForResidueCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForHybrid(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.Residue, _, _) => new MersenneNumberTesterForResidueCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForCpu(kernelType: kernelType, maxK: residueKMax),

			(CalculationMethod.LucasLehmer, OrderDevice.Gpu, ComputationDevice.Gpu) => new MersenneNumberTesterForLucasLehmerCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForGpu(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.LucasLehmer, OrderDevice.Gpu, ComputationDevice.Hybrid) => new MersenneNumberTesterForLucasLehmerCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForHybrid(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.LucasLehmer, OrderDevice.Gpu, _) => new MersenneNumberTesterForLucasLehmerCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForCpu(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.LucasLehmer, OrderDevice.Hybrid, ComputationDevice.Gpu) => new MersenneNumberTesterForLucasLehmerCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForGpu(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.LucasLehmer, OrderDevice.Hybrid, ComputationDevice.Hybrid) => new MersenneNumberTesterForLucasLehmerCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForHybrid(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.LucasLehmer, OrderDevice.Hybrid, _) => new MersenneNumberTesterForLucasLehmerCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForCpu(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.LucasLehmer, _, ComputationDevice.Gpu) => new MersenneNumberTesterForLucasLehmerCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForGpu(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.LucasLehmer, _, ComputationDevice.Hybrid) => new MersenneNumberTesterForLucasLehmerCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForHybrid(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.LucasLehmer, _, _) => new MersenneNumberTesterForLucasLehmerCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForCpu(kernelType: kernelType, maxK: residueKMax),

			(CalculationMethod.Divisor, OrderDevice.Gpu, ComputationDevice.Gpu) => new MersenneNumberTesterForDivisorCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForGpu(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.Divisor, OrderDevice.Gpu, ComputationDevice.Hybrid) => new MersenneNumberTesterForDivisorCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForHybrid(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.Divisor, OrderDevice.Gpu, _) => new MersenneNumberTesterForDivisorCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForCpu(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.Divisor, OrderDevice.Hybrid, ComputationDevice.Gpu) => new MersenneNumberTesterForDivisorCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForGpu(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.Divisor, OrderDevice.Hybrid, ComputationDevice.Hybrid) => new MersenneNumberTesterForDivisorCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForHybrid(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.Divisor, OrderDevice.Hybrid, _) => new MersenneNumberTesterForDivisorCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForCpu(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.Divisor, _, ComputationDevice.Gpu) => new MersenneNumberTesterForDivisorCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForGpu(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.Divisor, _, ComputationDevice.Hybrid) => new MersenneNumberTesterForDivisorCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForHybrid(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.Divisor, _, _) => new MersenneNumberTesterForDivisorCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForCpu(kernelType: kernelType, maxK: residueKMax),

			(CalculationMethod.ByDivisor, OrderDevice.Gpu, ComputationDevice.Gpu) => new MersenneNumberTesterForByDivisorCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForGpu(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.ByDivisor, OrderDevice.Gpu, ComputationDevice.Hybrid) => new MersenneNumberTesterForByDivisorCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForHybrid(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.ByDivisor, OrderDevice.Gpu, _) => new MersenneNumberTesterForByDivisorCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForCpu(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.ByDivisor, OrderDevice.Hybrid, ComputationDevice.Gpu) => new MersenneNumberTesterForByDivisorCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForGpu(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.ByDivisor, OrderDevice.Hybrid, ComputationDevice.Hybrid) => new MersenneNumberTesterForByDivisorCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForHybrid(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.ByDivisor, OrderDevice.Hybrid, _) => new MersenneNumberTesterForByDivisorCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForCpu(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.ByDivisor, _, ComputationDevice.Gpu) => new MersenneNumberTesterForByDivisorCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForGpu(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.ByDivisor, _, ComputationDevice.Hybrid) => new MersenneNumberTesterForByDivisorCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForHybrid(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.ByDivisor, _, _) => new MersenneNumberTesterForByDivisorCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForCpu(kernelType: kernelType, maxK: residueKMax),

			(CalculationMethod.Pow2Mod, OrderDevice.Gpu, ComputationDevice.Gpu) => new MersenneNumberTesterForPow2ModCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForGpu(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.Pow2Mod, OrderDevice.Gpu, ComputationDevice.Hybrid) => new MersenneNumberTesterForPow2ModCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForHybrid(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.Pow2Mod, OrderDevice.Gpu, _) => new MersenneNumberTesterForPow2ModCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForCpu(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.Pow2Mod, OrderDevice.Hybrid, ComputationDevice.Gpu) => new MersenneNumberTesterForPow2ModCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForGpu(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.Pow2Mod, OrderDevice.Hybrid, ComputationDevice.Hybrid) => new MersenneNumberTesterForPow2ModCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForHybrid(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.Pow2Mod, OrderDevice.Hybrid, _) => new MersenneNumberTesterForPow2ModCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForCpu(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.Pow2Mod, _, ComputationDevice.Gpu) => new MersenneNumberTesterForPow2ModCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForGpu(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.Pow2Mod, _, ComputationDevice.Hybrid) => new MersenneNumberTesterForPow2ModCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForHybrid(kernelType: kernelType, maxK: residueKMax),
			(CalculationMethod.Pow2Mod, _, _) => new MersenneNumberTesterForPow2ModCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForCpu(kernelType: kernelType, maxK: residueKMax),

			(_, OrderDevice.Gpu, ComputationDevice.Gpu) => new MersenneNumberTesterForIncrementalCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForGpu(kernelType: kernelType, maxK: residueKMax),
			(_, OrderDevice.Gpu, ComputationDevice.Hybrid) => new MersenneNumberTesterForIncrementalCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForHybrid(kernelType: kernelType, maxK: residueKMax),
			(_, OrderDevice.Gpu, _) => new MersenneNumberTesterForIncrementalCalculationMethodForDisabledCacheStatusForGpuOrderDeviceForCpu(kernelType: kernelType, maxK: residueKMax),
			(_, OrderDevice.Hybrid, ComputationDevice.Gpu) => new MersenneNumberTesterForIncrementalCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForGpu(kernelType: kernelType, maxK: residueKMax),
			(_, OrderDevice.Hybrid, ComputationDevice.Hybrid) => new MersenneNumberTesterForIncrementalCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForHybrid(kernelType: kernelType, maxK: residueKMax),
			(_, OrderDevice.Hybrid, _) => new MersenneNumberTesterForIncrementalCalculationMethodForDisabledCacheStatusForHybridOrderDeviceForCpu(kernelType: kernelType, maxK: residueKMax),
			(_, _, ComputationDevice.Gpu) => new MersenneNumberTesterForIncrementalCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForGpu(kernelType: kernelType, maxK: residueKMax),
			(_, _, ComputationDevice.Hybrid) => new MersenneNumberTesterForIncrementalCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForHybrid(kernelType: kernelType, maxK: residueKMax),
			_ => new MersenneNumberTesterForIncrementalCalculationMethodForDisabledCacheStatusForCpuOrderDeviceForCpu(kernelType: kernelType, maxK: residueKMax),
		};
	}

	private static void WarmUpOrders(object tester, OrderDevice orderDevice, ComputationDevice device, ulong exponent, ulong limit)
	{
		if (_cacheStatus == CacheStatus.Disabled)
		{
			return;
		}

		// When cache warming is enabled, add calls for the generated classes keyed by orderDevice and device.
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

	internal static bool IsEvenPerfectCandidate(PrimeOrderCalculatorAccelerator gpu, ulong p, ulong divisorCyclesSearchLimit, out bool searchedMersenne, out bool detailedCheck)
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
		if (!_cliArguments.UseDivisor && !_cliArguments.UseByDivisor &&
			(
				(EnvironmentConfiguration.UseGpuOrder && CandidatesCalculator.IsCompositeByResiduesGpu(gpu, p)) ||
				(EnvironmentConfiguration.UseHybridOrder && CandidatesCalculator.IsCompositeByResiduesHybrid(gpu, p)) ||
				(EnvironmentConfiguration.UseCpuOrder && CandidatesCalculator.IsCompositeByResiduesCpu(p))
			))
		{
			searchedMersenne = false;
			detailedCheck = false;
			_lastCompositeP = true;
			return false;
		}

		// If primes-device=gpu, route p primality through GPU-assisted sieve with deterministic MR validation.
		if (!_cliArguments.UseByDivisor)
		{
			if (!(_runPrimesOnCpu
					? PrimeTester.IsPrimeCpu(p)
					: HeuristicCombinedPrimeTester.IsPrimeGpu(gpu, p)))
			{
				_lastCompositeP = true;
				return false;
			}

			OnPrimeCandidateConfirmed();
		}

		searchedMersenne = true;
		if (_cliArguments.UseDivisor)
		{
			bool divisorsExhausted;
			bool isPrime = _divisorTester!.IsPrime(gpu, p, _divisor, divisorCyclesSearchLimit, out divisorsExhausted);
			detailedCheck = divisorsExhausted;
			return isPrime;
		}

		if (_cliArguments.UseByDivisor)
		{
			// Windowed pow2mod kernel handles by-divisor scans.
			bool usePow2Divisors = _cliArguments.UseDivisorPow2Increment;
			if (EnvironmentConfiguration.UseCpuOrder)
			{
				return usePow2Divisors
					? _byDivisorCpuOrderTesterPow2.IsPrime(p, out detailedCheck, out _)
					: _byDivisorCpuOrderTester.IsPrime(p, out detailedCheck, out _);
			}

			if (EnvironmentConfiguration.UseHybridOrder)
			{
				return usePow2Divisors
					? _byDivisorHybridOrderTesterPow2.IsPrime(p, out detailedCheck, out _)
					: _byDivisorHybridOrderTester.IsPrime(p, out detailedCheck, out _);
			}

			return usePow2Divisors
				? _byDivisorGpuOrderTesterPow2.IsPrime(p, out detailedCheck, out _)
				: _byDivisorGpuOrderTester.IsPrime(p, out detailedCheck, out _);
		}

		detailedCheck = IsMersennePrime(gpu, p);
		return detailedCheck;
	}
}

