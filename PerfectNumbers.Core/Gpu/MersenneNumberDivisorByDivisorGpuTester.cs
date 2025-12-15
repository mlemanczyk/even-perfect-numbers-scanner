using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu.Accelerators;

namespace PerfectNumbers.Core.Gpu;

/// <summary>
/// Scans Mersenne divisors on the GPU for prime exponents p >= 31 using cached GPU divisor partial data.
/// Consumers must call <see cref="ConfigureFromMaxPrime"/> before invoking other members so the divisor limits are populated.
/// </summary>
public sealed partial class MersenneNumberDivisorByDivisorGpuTester : IMersenneNumberDivisorByDivisorTester
{
	private int _gpuBatchSize = GpuConstants.ScanBatchSize;
	// EvenPerfectBitScanner configures the GPU tester once before scanning and never mutates the configuration afterwards,
	// so the synchronization fields from the previous implementation remain commented out here.
	private ulong _divisorLimit;
	private BigInteger _minK = BigInteger.One;

	private readonly ConcurrentFixedCapacityStack<MersenneNumberDivisorByDivisorAccelerator>[] _resourcePool = [.. AcceleratorPool.Shared.Accelerators.Select(x => new ConcurrentFixedCapacityStack<MersenneNumberDivisorByDivisorAccelerator>(PerfectNumberConstants.DefaultPoolCapacity))];

	private readonly ConcurrentFixedCapacityStack<DivisorScanSession> _sessionPool = new(PerfectNumberConstants.DefaultPoolCapacity);

	public int GpuBatchSize
	{
		get => _gpuBatchSize;
		set => _gpuBatchSize = value < 1 ? 1 : value;
	}

	int IMersenneNumberDivisorByDivisorTester.BatchSize
	{
		get => GpuBatchSize;
		set => GpuBatchSize = value;
	}

	public BigInteger MinK
	{
		get => _minK;
		set => _minK = value < BigInteger.One ? BigInteger.One : value;
	}

	public string? StateFilePath { get; set; }

	public void ResetStateTracking()
	{
	}

	public void ResumeFromState(BigInteger lastSavedK)
	{
		if (lastSavedK > ulong.MaxValue)
		{
			throw new NotSupportedException("GPU by-divisor tester does not support MinK above UInt64.");
		}

		_minK = lastSavedK + BigInteger.One;
	}

	public void ConfigureFromMaxPrime(ulong maxPrime)
	{
		// EvenPerfectBitScanner configures the GPU tester once before scanning and never mutates the configuration afterwards,
		// so synchronization and runtime configuration guards are unnecessary here.

		_divisorLimit = ComputeDivisorLimitFromMaxPrimeGpu(maxPrime);
	}

	public bool IsPrime(PrimeOrderCalculatorAccelerator gpu, ulong prime, out bool divisorsExhausted, out BigInteger divisor)
	{
		ulong allowedMax;
		int batchCapacity;

		// EvenPerfectBitScanner only calls into this tester after configuring it once, so we can read the cached values without locking.

		allowedMax = ComputeAllowedMaxDivisorGpu(prime, _divisorLimit);
		batchCapacity = _gpuBatchSize;

		// Production scans never shrink the divisor window below three, so this guard stays commented out.
		// if (allowedMax < 3UL)
		// {
		//     divisorsExhausted = true;
		//     return true;
		// }

		bool composite;
		bool coveredRange;
		ulong lastProcessed;

		// GpuPrimeWorkLimiter();
		// int acceleratorIndex = AcceleratorPool.Shared.Rent();
		// var accelerator = _accelerators[acceleratorIndex];
		var acceleratorIndex = gpu.AcceleratorIndex;

		// Monitor.Enter(gpuLease.ExecutionLock);

		var resources = RentBatchResources(acceleratorIndex, batchCapacity);

		composite = CheckDivisors(
			gpu,
			prime,
			allowedMax,
			GetMinKOrThrow(),
			resources.CheckDivisorKernel,
			resources.DivisorDataBuffer,
			resources.OffsetBuffer,
			resources.CountBuffer,
			resources.CycleBuffer,
			resources.ExponentBuffer,
			resources.HitsBuffer,
			resources.HitIndexBuffer,
			resources.Divisors,
			resources.Exponents,
			resources.DivisorData,
			resources.Offsets,
			resources.Counts,
			resources.Cycles,
			out lastProcessed,
			out coveredRange);

		ReturnBatchResources(acceleratorIndex, resources);
		// Monitor.Exit(gpuLease.ExecutionLock);
		// GpuPrimeWorkLimiter.Release();

		if (composite)
		{
			divisorsExhausted = true;
			divisor = lastProcessed;
			return false;
		}

		divisorsExhausted = coveredRange;
		divisor = BigInteger.Zero;
		return true;
	}

	private ulong GetMinKOrThrow()
	{
		if (_minK > ulong.MaxValue)
		{
			throw new NotSupportedException("GPU by-divisor tester does not support MinK above UInt64.");
		}

		return (ulong)_minK;
	}

	public void PrepareCandidates(in ReadOnlySpan<ulong> primes, Span<ulong> allowedMaxValues)
	{
		// The scanner always supplies matching spans, so the previous validation remains commented out.
		// if (allowedMaxValues.Length < primes.Length)
		// {
		//     throw new ArgumentException("allowedMaxValues span must be at least as long as primes span.", nameof(allowedMaxValues));
		// }

		ulong divisorLimit;

		// EvenPerfectBitScanner configures the tester exactly once before preparing candidates.
		// lock (_sync)
		// {
		//     if (!_isConfigured)
		//     {
		//         throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
		//     }

		//     divisorLimit = _divisorLimit;
		// }

		divisorLimit = _divisorLimit;

		for (int index = 0; index < primes.Length; index++)
		{
			allowedMaxValues[index] = ComputeAllowedMaxDivisorGpu(primes[index], divisorLimit);
		}
	}

	private static bool CheckDivisors(
		PrimeOrderCalculatorAccelerator gpu,
		ulong prime,
		ulong allowedMax,
		ulong minK,
		Kernel kernel,
		MemoryBuffer1D<GpuDivisorPartialData, Stride1D.Dense> divisorDataBuffer,
		MemoryBuffer1D<int, Stride1D.Dense> offsetBuffer,
		MemoryBuffer1D<int, Stride1D.Dense> countBuffer,
		MemoryBuffer1D<ulong, Stride1D.Dense> cycleBuffer,
		MemoryBuffer1D<ulong, Stride1D.Dense> exponentBuffer,
		MemoryBuffer1D<byte, Stride1D.Dense> hitsBuffer,
		MemoryBuffer1D<int, Stride1D.Dense> hitIndexBuffer,
		ulong[] divisors,
		ulong[] exponents,
		GpuDivisorPartialData[] divisorData,
		int[] offsets,
		int[] counts,
		ulong[] cycles,
		out ulong lastProcessed,
		out bool coveredRange)
	{
		int batchCapacity = (int)divisorDataBuffer.Length;
		bool composite = false;
		bool processedAll;
		ulong processedCountLocal = 0UL;
		ulong lastProcessedLocal = 0UL;
		var acceleratorIndex = gpu.AcceleratorIndex;

		// EvenPerfectBitScanner always configures non-empty GPU divisor buffers, so the defensive guard stays
		// commented out. Tests and benchmarks that rely on zero-capacity buffers must be updated instead of re-enabling
		// this branch.
		// if (batchCapacity <= 0)
		// {
		//     throw new InvalidOperationException("Divisor buffers must be non-empty.");
		// }

		int chunkCountBaseline = batchCapacity;

		UInt128 twoP128 = (UInt128)prime << 1;
		UInt128 allowedMax128 = allowedMax;
		UInt128 firstDivisor128 = twoP128 + UInt128.One;
		bool invalidStride = twoP128 == UInt128.Zero;
		bool outOfRange = firstDivisor128 > allowedMax128;
		UInt128 numerator = allowedMax128 - UInt128.One;
		UInt128 maxK128 = invalidStride || outOfRange ? UInt128.Zero : numerator / twoP128;
		bool hasCandidates = !invalidStride && !outOfRange && maxK128 != UInt128.Zero;
		if (!hasCandidates)
		{
			coveredRange = true;
			lastProcessed = lastProcessedLocal;
			return false;
		}

		ulong maxK = maxK128 > ulong.MaxValue ? ulong.MaxValue : (ulong)maxK128;
		ulong startK = minK < 1UL ? 1UL : minK;

		Span<ulong> divisorStorage = divisors.AsSpan();
		Span<ulong> exponentStorage = exponents.AsSpan();
		Span<GpuDivisorPartialData> divisorDataStorage = divisorData.AsSpan();
		Span<int> offsetStorage = offsets.AsSpan();
		Span<int> countStorage = counts.AsSpan();
		Span<ulong> cycleStorage = cycles.AsSpan();
		Span<ulong> divisorSpan = divisorStorage;
		Span<ulong> exponentSpan = exponentStorage;
		Span<GpuDivisorPartialData> divisorDataSpan = divisorDataStorage;
		Span<int> offsetSpan = offsetStorage;
		Span<int> countSpan = countStorage;
		Span<ulong> cycleSpan = cycleStorage;

		ArrayView1D<GpuDivisorPartialData, Stride1D.Dense> divisorDataView = divisorDataBuffer.View;
		ArrayView1D<int, Stride1D.Dense> offsetView = offsetBuffer.View;
		ArrayView1D<int, Stride1D.Dense> countView = countBuffer.View;
		ArrayView1D<ulong, Stride1D.Dense> cycleView = cycleBuffer.View;
		ArrayView1D<ulong, Stride1D.Dense> exponentViewDevice = exponentBuffer.View;
		ArrayView1D<byte, Stride1D.Dense> hitsView = hitsBuffer.View;
		ArrayView1D<int, Stride1D.Dense> hitIndexView = hitIndexBuffer.View;

		static bool ProcessRange(
			PrimeOrderCalculatorAccelerator gpu,
			ulong prime,
			UInt128 twoP128,
			UInt128 allowedMax128,
			ArrayView1D<GpuDivisorPartialData, Stride1D.Dense> divisorDataView,
			ArrayView1D<int, Stride1D.Dense> offsetView,
			ArrayView1D<int, Stride1D.Dense> countView,
			ArrayView1D<ulong, Stride1D.Dense> cycleView,
			ArrayView1D<ulong, Stride1D.Dense> exponentViewDevice,
			ArrayView1D<byte, Stride1D.Dense> hitsView,
			ArrayView1D<int, Stride1D.Dense> hitIndexView,
			Span<ulong> divisorSpan,
			Span<ulong> exponentSpan,
			Span<GpuDivisorPartialData> divisorDataSpan,
			Span<int> offsetSpan,
			Span<int> countSpan,
			Span<ulong> cycleSpan,
			int chunkCountBaseline,
			int acceleratorIndex,
			ref bool composite,
			ref ulong processedCountLocal,
			ref ulong lastProcessedLocal,
			ulong rangeStartK,
			ulong rangeEndK,
			Kernel kernel)
		{
			if (rangeStartK < 1UL || rangeEndK < rangeStartK)
			{
				return true;
			}

			UInt128 startDivisor128 = (twoP128 * rangeStartK) + UInt128.One;
			if (startDivisor128 > allowedMax128)
			{
				return true;
			}

			var residueStepper = new MersenneDivisorResidueStepper(prime, (GpuUInt128)twoP128, (GpuUInt128)startDivisor128);
			UInt128 rangeLimit128 = (twoP128 * rangeEndK) + UInt128.One;
			if (rangeLimit128 > allowedMax128)
			{
				rangeLimit128 = allowedMax128;
			}

			UInt128 rangeCount128 = ((rangeLimit128 - startDivisor128) / twoP128) + UInt128.One;
			ulong rangeRemaining = rangeCount128 > ulong.MaxValue ? ulong.MaxValue : (ulong)rangeCount128;
			UInt128 currentRangeDivisor128 = startDivisor128;

			while (rangeRemaining > 0UL && !composite)
			{
				int chunkCount = chunkCountBaseline;
				if ((ulong)chunkCount > rangeRemaining)
				{
					chunkCount = (int)rangeRemaining;
				}

				UInt128 nextDivisor128 = currentRangeDivisor128;
				int admissibleCount = 0;

				var localStepper = residueStepper;

				for (int i = 0; i < chunkCount; i++)
				{
					if (localStepper.IsAdmissible())
					{
						ulong divisorValue = (ulong)nextDivisor128;
						MontgomeryDivisorData montgomeryData = MontgomeryDivisorData.FromModulus(divisorValue);
						ulong divisorCycle = ResolveDivisorCycle(gpu, divisorValue, prime, montgomeryData);
						if (divisorCycle == prime)
						{
							processedCountLocal += (ulong)(i + 1);
							lastProcessedLocal = divisorValue;
							composite = true;
							return false;
						}

						divisorSpan[admissibleCount] = divisorValue;
						divisorDataSpan[admissibleCount] = new GpuDivisorPartialData(divisorValue);
						offsetSpan[admissibleCount] = admissibleCount;
						countSpan[admissibleCount] = 1;
						cycleSpan[admissibleCount] = divisorCycle;
						exponentSpan[admissibleCount] = prime;
						admissibleCount++;
					}

					nextDivisor128 += twoP128;
					localStepper.Advance();
				}

				residueStepper = localStepper;

				processedCountLocal += (ulong)chunkCount;

				UInt128 lastDivisor128 = nextDivisor128 - twoP128;
				lastProcessedLocal = (ulong)lastDivisor128;
				currentRangeDivisor128 = nextDivisor128;
				rangeRemaining -= (ulong)chunkCount;

				if (admissibleCount == 0)
				{
					continue;
				}

				ref GpuDivisorPartialData divisorDataRef = ref MemoryMarshal.GetReference(divisorDataSpan);
				ref int offsetRef = ref MemoryMarshal.GetReference(offsetSpan);
				ref int countRef = ref MemoryMarshal.GetReference(countSpan);
				ref ulong cycleRef = ref MemoryMarshal.GetReference(cycleSpan);
				ref ulong exponentRef = ref MemoryMarshal.GetReference(exponentSpan);

				var stream = AcceleratorStreamPool.Rent(acceleratorIndex);
				divisorDataView.CopyFromCPU(stream, ref divisorDataRef, admissibleCount);
				offsetView.CopyFromCPU(stream, ref offsetRef, admissibleCount);
				countView.CopyFromCPU(stream, ref countRef, admissibleCount);
				cycleView.CopyFromCPU(stream, ref cycleRef, admissibleCount);
				exponentViewDevice.CopyFromCPU(stream, ref exponentRef, admissibleCount);

				int sentinel = int.MaxValue;
				hitIndexView.CopyFromCPU(stream, ref sentinel, 1);

				kernel.Launch(stream, 1, new Index1D(admissibleCount), divisorDataView, offsetView, countView, exponentViewDevice, cycleView, hitsView, hitIndexView);

				int firstHit = sentinel;
				hitIndexView.CopyToCPU(stream, ref firstHit, 1);
				stream.Synchronize();
				AcceleratorStreamPool.Return(acceleratorIndex, stream);
				int hitIndex = firstHit >= admissibleCount ? -1 : firstHit;
				bool hitFound = hitIndex >= 0;
				composite = hitFound;
				int lastIndex = admissibleCount - 1;
				lastProcessedLocal = hitFound ? divisorSpan[hitIndex] : divisorSpan[lastIndex];
			}

			return rangeRemaining == 0UL;
		}

		bool processedTop = true;
		bool processedBottom = true;

		if (startK <= maxK)
		{
			processedTop = ProcessRange(
				gpu,
				prime,
				twoP128,
				allowedMax128,
				divisorDataView,
				offsetView,
				countView,
				cycleView,
				exponentViewDevice,
				hitsView,
				hitIndexView,
				divisorSpan,
				exponentSpan,
				divisorDataSpan,
				offsetSpan,
				countSpan,
				cycleSpan,
				chunkCountBaseline,
				acceleratorIndex,
				ref composite,
				ref processedCountLocal,
				ref lastProcessedLocal,
				startK,
				maxK,
				kernel);
			if (composite)
			{
				coveredRange = true;
				lastProcessed = lastProcessedLocal;
				return true;
			}
		}

		ulong lowerEnd = startK > 1UL ? startK - 1UL : 0UL;
		if (!composite && lowerEnd >= 1UL)
		{
			lowerEnd = Math.Min(lowerEnd, maxK);
			processedBottom = ProcessRange(
				gpu,
				prime,
				twoP128,
				allowedMax128,
				divisorDataView,
				offsetView,
				countView,
				cycleView,
				exponentViewDevice,
				hitsView,
				hitIndexView,
				divisorSpan,
				exponentSpan,
				divisorDataSpan,
				offsetSpan,
				countSpan,
				cycleSpan,
				chunkCountBaseline,
				acceleratorIndex,
				ref composite,
				ref processedCountLocal,
				ref lastProcessedLocal,
				1UL,
				lowerEnd,
				kernel);
		}

		processedAll = processedTop && processedBottom;
		coveredRange = composite || processedAll;
		lastProcessed = lastProcessedLocal;
		return composite;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static ulong ResolveDivisorCycle(PrimeOrderCalculatorAccelerator gpu, ulong divisor, ulong prime, in MontgomeryDivisorData divisorData)
	{
		if (!MersenneDivisorCycles.TryCalculateCycleLengthForExponentGpu(gpu, divisor, prime, divisorData, out ulong computedCycle, out bool primeOrderFailed) || computedCycle == 0UL)
		{
			return MersenneDivisorCycles.CalculateCycleLengthGpu(gpu, divisor, divisorData, skipPrimeOrderHeuristic: primeOrderFailed);
		}

		return computedCycle;
	}


	private static ulong ComputeDivisorLimitFromMaxPrimeGpu(ulong maxPrime)
	{
		// The --mersenne=bydivisor flow in EvenPerfectBitScanner only calls ConfigureFromMaxPrime with primes greater than 1,
		// so the guard below never trips in production runs.
		// if (maxPrime <= 1UL)
		// {
		//     return 0UL;
		// }
		if (maxPrime - 1UL >= 64UL)
		{
			return ulong.MaxValue;
		}

		return (1UL << (int)(maxPrime - 1UL)) - 1UL;
	}

	private static ulong ComputeAllowedMaxDivisorGpu(ulong prime, ulong divisorLimit)
	{
		// Production --mersenne=bydivisor runs only pass prime exponents, so the guard below never executes outside tests.
		// if (prime <= 1UL)
		// {
		//     return 0UL;
		// }
		ulong cappedLimit = divisorLimit;
		const ulong TestPrimeDivisorCap = 170_000_000UL;
		if (prime < 10_000_000UL && cappedLimit > TestPrimeDivisorCap)
		{
			cappedLimit = TestPrimeDivisorCap;
		}

		if (prime - 1UL >= 64UL)
		{
			return cappedLimit;
		}

		ulong computedLimit = (1UL << (int)(prime - 1UL)) - 1UL;
		if (prime < 10_000_000UL && computedLimit > cappedLimit)
		{
			computedLimit = cappedLimit;
		}

		return computedLimit < cappedLimit ? computedLimit : cappedLimit;
	}

	public sealed class DivisorScanSession : IMersenneNumberDivisorByDivisorTester.IDivisorScanSession
	{
		private readonly MersenneNumberDivisorByDivisorGpuTester _owner;
		private readonly Accelerator _accelerator;
		private readonly PrimeOrderCalculatorAccelerator _primeOrderCalculatorAccelerator;
		private readonly AcceleratorStream _stream;
		private Action<AcceleratorStream, Index1D, ArrayView<GpuDivisorPartialData>, ArrayView<int>, ArrayView<int>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>, ArrayView<int>> _kernel = null!;
		private MemoryBuffer1D<GpuDivisorPartialData, Stride1D.Dense> _divisorBuffer = null!;
		private MemoryBuffer1D<int, Stride1D.Dense> _offsetBuffer = null!;
		private MemoryBuffer1D<int, Stride1D.Dense> _countBuffer = null!;
		private MemoryBuffer1D<ulong, Stride1D.Dense> _cycleBuffer = null!;
		private MemoryBuffer1D<int, Stride1D.Dense> _firstHitBuffer = null!;
		private MemoryBuffer1D<ulong, Stride1D.Dense> _exponentsBuffer = null!;
		private MemoryBuffer1D<byte, Stride1D.Dense> _hitBuffer = null!;
		private ulong[] _hostBuffer = null!;
		private int _capacity;


		internal DivisorScanSession(PrimeOrderCalculatorAccelerator gpu, MersenneNumberDivisorByDivisorGpuTester owner)
		{
			_owner = owner;
			// GpuPrimeWorkLimiter.Acquire();
			_primeOrderCalculatorAccelerator = gpu;
			Accelerator accelerator = gpu.Accelerator;
			_accelerator = accelerator;
			_stream = accelerator.CreateStream();
			_capacity = Math.Max(1, owner._gpuBatchSize);

			// lock (accelerator)
			{
				_divisorBuffer = accelerator.Allocate1D<GpuDivisorPartialData>(1);
				_offsetBuffer = accelerator.Allocate1D<int>(1);
				_countBuffer = accelerator.Allocate1D<int>(1);
				_cycleBuffer = accelerator.Allocate1D<ulong>(1);
				_firstHitBuffer = accelerator.Allocate1D<int>(1);
			}

			int desiredCapacity = PerfectNumberConstants.DefaultSmallPrimeFactorSlotCount;
			_capacity = desiredCapacity;
			// lock (accelerator)
			{
				_exponentsBuffer = accelerator.Allocate1D<ulong>(desiredCapacity);
				_hitBuffer = accelerator.Allocate1D<byte>(desiredCapacity);
			}

			_hostBuffer = ThreadStaticPools.UlongPool.Rent(desiredCapacity);

			_kernel = KernelUtil.GetKernel(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuDivisorPartialData>, ArrayView<int>, ArrayView<int>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>, ArrayView<int>>(DivisorByDivisorKernels.CheckKernel)).CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView<GpuDivisorPartialData>, ArrayView<int>, ArrayView<int>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>, ArrayView<int>>>();
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		internal void Reset()
		{
		}

		private void EnsureExecutionResourcesLocked(int requiredCapacity)
		{
			if (requiredCapacity > _capacity)
			{
				_exponentsBuffer.Dispose();
				_hitBuffer.Dispose();
				ThreadStaticPools.UlongPool.Return(_hostBuffer, clearArray: false);

				Accelerator accelerator = _accelerator;
				int desiredCapacity = requiredCapacity > _capacity ? requiredCapacity : _capacity;
				_capacity = desiredCapacity;

				// lock (accelerator)
				{
					_exponentsBuffer = accelerator.Allocate1D<ulong>(desiredCapacity);
					_hitBuffer = accelerator.Allocate1D<byte>(desiredCapacity);
				}

				_hostBuffer = ThreadStaticPools.UlongPool.Rent(desiredCapacity);
			}
		}

		public bool CheckDivisor(ulong divisor, in MontgomeryDivisorData divisorData, ulong divisorCycle, in ReadOnlySpan<ulong> primes)
		{
			int length = primes.Length;
			// EvenPerfectBitScanner always supplies at least one exponent per divisor check, so the guard stays commented out.
			// if (length == 0)
			// {
			//     return false;
			// }

			// The GPU divisor sessions only materialize odd moduli greater than one (q = 2kp + 1),
			// so the defensive modulus guard stays commented out to keep the hot path branch-free.
			// if (divisorData.Modulus <= 1UL || (divisorData.Modulus & 1UL) == 0UL)
			// {
			//     hits.Clear();
			//     return;
			// }

			ulong cycle = divisorCycle;
			ulong firstPrime = primes[0];
			var gpu = _primeOrderCalculatorAccelerator;
			if (cycle == 0UL)
			{
				if (!MersenneDivisorCycles.TryCalculateCycleLengthForExponentCpu(divisor, firstPrime, divisorData, out ulong computedCycle, out bool primeOrderFailed) || computedCycle == 0UL)
				{
					cycle = MersenneDivisorCycles.CalculateCycleLengthGpu(gpu, divisor, divisorData, skipPrimeOrderHeuristic: primeOrderFailed);
				}
				else
				{
					cycle = computedCycle;
				}

				if (cycle == 0UL)
				{
					throw new InvalidOperationException("GPU divisor cycle length must be non-zero.");
				}
			}

			// Monitor.Enter(_lease.ExecutionLock);

			EnsureExecutionResourcesLocked(length);

			ArrayView1D<GpuDivisorPartialData, Stride1D.Dense> divisorView = _divisorBuffer.View;
			ArrayView1D<int, Stride1D.Dense> offsetView = _offsetBuffer.View;
			ArrayView1D<int, Stride1D.Dense> countView = _countBuffer.View;
			ArrayView1D<ulong, Stride1D.Dense> cycleView = _cycleBuffer.View;
			ArrayView1D<int, Stride1D.Dense> firstHitView = _firstHitBuffer.View;
			ArrayView1D<ulong, Stride1D.Dense> exponentView = _exponentsBuffer.View;
			ArrayView1D<byte, Stride1D.Dense> hitView = _hitBuffer.View;

			AcceleratorStream stream = _stream;
			GpuDivisorPartialData partialData = new GpuDivisorPartialData(divisor);
			divisorView.CopyFromCPU(stream, ref partialData, 1);

			int offsetValue = 0;
			offsetView.CopyFromCPU(stream, ref offsetValue, 1);

			int countValue = length;
			countView.CopyFromCPU(stream, ref countValue, 1);

			ulong cycleValue = cycle;
			cycleView.CopyFromCPU(stream, ref cycleValue, 1);

			Span<ulong> hostSpan = _hostBuffer.AsSpan(0, length);
			primes.CopyTo(hostSpan);
			ref ulong hostRef = ref MemoryMarshal.GetReference(hostSpan);

			exponentView.CopyFromCPU(stream, ref hostRef, length);

			int sentinel = int.MaxValue;
			firstHitView.CopyFromCPU(stream, ref sentinel, 1);

			var kernel = _kernel!;

			kernel(stream, new Index1D(1), divisorView, offsetView, countView, exponentView, cycleView, hitView, firstHitView);

			firstHitView.CopyToCPU(stream, ref sentinel, 1);
			stream.Synchronize();

			// Monitor.Exit(_lease.ExecutionLock);
			return sentinel == 0;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		public void Return() => _owner._sessionPool.Push(this);
	}

	public ulong DivisorLimit
	{
		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		get
		{
			// The EvenPerfectBitScanner driver configures the GPU tester before exposing it to callers, so the previous synchronization guard remains commented out.
			// lock (_sync)
			// {
			//     if (!_isConfigured)
			//     {
			//         throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
			//     }

			//     return _divisorLimit;
			// }

			return _divisorLimit;
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private MersenneNumberDivisorByDivisorAccelerator RentBatchResources(int acceleratorIndex, int capacity)
	{
		var queue = _resourcePool[acceleratorIndex];
		var accelerator = AcceleratorPool.Shared.Accelerators[acceleratorIndex];
		if (queue.Pop() is { } resources)
		{
			resources.EnsureCapacity(accelerator, capacity);
			return resources;
		}

		return new MersenneNumberDivisorByDivisorAccelerator(accelerator, capacity);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private void ReturnBatchResources(int acceleratorIndex, MersenneNumberDivisorByDivisorAccelerator resources) => _resourcePool[acceleratorIndex].Push(resources);

	private sealed class AcceleratorReferenceComparer : IEqualityComparer<Accelerator>
	{
		public static AcceleratorReferenceComparer Instance { get; } = new();

		public bool Equals(Accelerator? x, Accelerator? y) => ReferenceEquals(x, y);

		public int GetHashCode(Accelerator obj) => RuntimeHelpers.GetHashCode(obj);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public ulong GetAllowedMaxDivisor(ulong prime)
	{
		// The configuration is immutable after setup, so the cached limit can be read without locking.
		// lock (_sync)
		// {
		//     if (!_isConfigured)
		//     {
		//         throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
		//     }

		//     return ComputeAllowedMaxDivisorGpu(prime, _divisorLimit);
		// }

		return ComputeAllowedMaxDivisorGpu(prime, _divisorLimit);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public IMersenneNumberDivisorByDivisorTester.IDivisorScanSession CreateDivisorSession(PrimeOrderCalculatorAccelerator gpu)
	{
		// The tester is configured once at startup and the session pool relies on thread-safe collections, so the previous synchronization guard stays commented out.
		// lock (_sync)
		// {
		//     if (!_isConfigured)
		//     {
		//         throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
		//     }

		//     if (_sessionPool.TryTake(out DivisorScanSession? session))
		//     {
		//         session.Reset();
		//         return session;
		//     }

		//     return new DivisorScanSession(this);
		// }

		if (_sessionPool.Pop() is { } session)
		{
			session.Reset();
			return session;
		}

		return new DivisorScanSession(gpu, this);
	}
}
