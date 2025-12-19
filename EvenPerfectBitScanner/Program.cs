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
	private static Dictionary<ulong, (bool DetailedCheck, bool PassedAllTests)>? _byDivisorPreviousResults;
	private static bool _limitReached;
	private static ulong _byDivisorStartPrime;
	private static CliArguments _cliArguments;
	private static int _testTargetPrimeCount;
	private static int _testProcessedPrimeCount;
	private static ComputationDevice _mersenneDevice;
	private static ComputationDevice _orderDevice;

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
			ComputationDevice orderDevice = EnvironmentConfiguration.OrderDevice;
			_mersenneDevice = mersenneDevice;
			_orderDevice = orderDevice;
			bool mersenneOnGpu = mersenneDevice == ComputationDevice.Gpu;
			bool orderOnGpu = EnvironmentConfiguration.UseGpuOrder;
			bool orderOnHybrid = EnvironmentConfiguration.UseHybridOrder;
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
					// ProcessEightBitWindows windowed pow2 ladder is the default kernel.
					object tester = orderDevice switch
					{
						ComputationDevice.Gpu => new MersenneNumberTesterGpu(
							useIncremental: !useLucas,
							useOrderCache: false,
							kernelType: kernelType,
							useOrder: useOrder,
							useGpuLucas: mersenneOnGpu,
							useGpuScan: mersenneOnGpu,
							useGpuOrder: orderOnGpu || orderOnHybrid,
							useResidue: useResidue,
							maxK: residueKMax,
							orderDevice: orderDevice),
						ComputationDevice.Hybrid => new MersenneNumberTesterHybrid(
							useIncremental: !useLucas,
							useOrderCache: false,
							kernelType: kernelType,
							useOrder: useOrder,
							useGpuLucas: mersenneOnGpu,
							useGpuScan: mersenneOnGpu,
							useGpuOrder: orderOnGpu || orderOnHybrid,
							useResidue: useResidue,
							maxK: residueKMax,
							orderDevice: orderDevice),
						_ => new MersenneNumberTesterCpu(
							useIncremental: !useLucas,
							useOrderCache: false,
							kernelType: kernelType,
							useOrder: useOrder,
							useGpuLucas: mersenneOnGpu,
							useGpuScan: mersenneOnGpu,
							useGpuOrder: orderOnGpu || orderOnHybrid,
							useResidue: useResidue,
							maxK: residueKMax,
							orderDevice: orderDevice),
					};
					if (!useLucas)
					{
						Console.WriteLine("Warming up orders");
						if (orderDevice == ComputationDevice.Hybrid)
						{
							((MersenneNumberTesterHybrid)tester).WarmUpOrders(currentP, orderWarmupLimitOverride ?? 5_000_000UL);
						}
						else if (orderDevice == ComputationDevice.Gpu)
						{
							((MersenneNumberTesterGpu)tester).WarmUpOrders(currentP, orderWarmupLimitOverride ?? 5_000_000UL);
						}
						else
						{
							((MersenneNumberTesterCpu)tester).WarmUpOrders(currentP, orderWarmupLimitOverride ?? 5_000_000UL);
						}
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

				if (EnvironmentConfiguration.UseCpuOrder)
				{
					_byDivisorCpuOrderTester = new MersenneNumberDivisorByDivisorCpuTesterWithCpuOrder();
				}
				else if (EnvironmentConfiguration.UseHybridOrder)
				{
					_byDivisorHybridOrderTester = new MersenneNumberDivisorByDivisorCpuTesterWithHybridOrder();
				}
				else
				{
					_byDivisorGpuOrderTester = new MersenneNumberDivisorByDivisorCpuTesterWithGpuOrder();
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

			string orderDeviceLabel = orderDevice switch
			{
				ComputationDevice.Hybrid => "hybrid",
				ComputationDevice.Gpu => "gpu",
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
				if (EnvironmentConfiguration.UseCpuOrder)
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
				else if (EnvironmentConfiguration.UseHybridOrder)
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
					if (orderDevice == ComputationDevice.Hybrid)
					{
						((MersenneNumberTesterHybrid)MersenneTesters.Value!).WarmUpOrders(currentP, orderWarmupLimitOverride ?? 5_000_000UL);
					}
					else if (orderDevice == ComputationDevice.Gpu)
					{
						((MersenneNumberTesterGpu)MersenneTesters.Value!).WarmUpOrders(currentP, orderWarmupLimitOverride ?? 5_000_000UL);
					}
					else
					{
						((MersenneNumberTesterCpu)MersenneTesters.Value!).WarmUpOrders(currentP, orderWarmupLimitOverride ?? 5_000_000UL);
					}
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
		return _orderDevice switch
		{
			ComputationDevice.Gpu => ((MersenneNumberTesterGpu)tester).IsMersennePrime(gpu, exponent),
			ComputationDevice.Hybrid => ((MersenneNumberTesterHybrid)tester).IsMersennePrime(gpu, exponent),
			_ => ((MersenneNumberTesterCpu)tester).IsMersennePrime(gpu, exponent),
		};
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
			if (EnvironmentConfiguration.UseCpuOrder)
			{
				return _byDivisorCpuOrderTester.IsPrime(p, out detailedCheck, out _);
			}

			if (EnvironmentConfiguration.UseHybridOrder)
			{
				return _byDivisorHybridOrderTester.IsPrime(p, out detailedCheck, out _);
			}

			return _byDivisorGpuOrderTester.IsPrime(p, out detailedCheck, out _);
		}

		detailedCheck = IsMersennePrime(gpu, p);
		return detailedCheck;
	}
}

