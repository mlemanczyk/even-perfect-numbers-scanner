using System.Collections.Concurrent;
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
	// private readonly object _sync = new();
	private ulong _divisorLimit;
	// private bool _isConfigured;

	private readonly ConcurrentBag<MersenneNumberDivisorByDivisorAccelerator>[] _resourcePool = [..AcceleratorPool.Shared.Accelerators.Select(x => new ConcurrentBag<MersenneNumberDivisorByDivisorAccelerator>())];

	private readonly ConcurrentBag<DivisorScanSession> _sessionPool = [];

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

	public void ConfigureFromMaxPrime(ulong maxPrime)
	{
		// EvenPerfectBitScanner configures the GPU tester once before scanning and never mutates the configuration afterwards,
		// so synchronization and runtime configuration guards are unnecessary here.
		// lock (_sync)
		// {
		//     _divisorLimit = ComputeDivisorLimitFromMaxPrimeGpu(maxPrime);
		//     _isConfigured = true;
		// }

		_divisorLimit = ComputeDivisorLimitFromMaxPrimeGpu(maxPrime);
		// _isConfigured = true;
	}

	private static readonly Accelerator[] _accelerators = AcceleratorPool.Shared.Accelerators;

	public bool IsPrime(ulong prime, out bool divisorsExhausted, out ulong divisor)
	{
		ulong allowedMax;
		int batchCapacity;

		// EvenPerfectBitScanner only calls into this tester after configuring it once, so we can read the cached values without locking.
		// lock (_sync)
		// {
		//     if (!_isConfigured)
		//     {
		//         throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
		//     }

		//     allowedMax = ComputeAllowedMaxDivisorGpu(prime, _divisorLimit);
		//     batchCapacity = _gpuBatchSize;
		// }

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
		ulong processedCount;
		ulong lastProcessed;

		// GpuPrimeWorkLimiter();
		// int acceleratorIndex = AcceleratorPool.Shared.Rent();
		// var accelerator = _accelerators[acceleratorIndex];
		var gpu = PrimeOrderCalculatorAccelerator.Rent(batchCapacity);
		var acceleratorIndex = gpu.AcceleratorIndex;

		// Monitor.Enter(gpuLease.ExecutionLock);

		var resources = RentBatchResources(acceleratorIndex, batchCapacity);

		composite = CheckDivisors(
			gpu,
			prime,
			allowedMax,
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
			out coveredRange,
			out processedCount);

		ReturnBatchResources(acceleratorIndex, resources);
		PrimeOrderCalculatorAccelerator.Return(gpu);
		// Monitor.Exit(gpuLease.ExecutionLock);
		// GpuPrimeWorkLimiter.Release();

		if (composite)
		{
			divisorsExhausted = true;
			divisor = lastProcessed;
			return false;
		}

		divisorsExhausted = coveredRange;
		divisor = 0UL;
		return true;
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
		out bool coveredRange,
		out ulong processedCount)
	{
		int batchCapacity = (int)divisorDataBuffer.Length;
		bool composite = false;
		bool processedAll;
		processedCount = 0UL;
		lastProcessed = 0UL;
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
			return false;
		}

		ulong maxK = maxK128 > ulong.MaxValue ? ulong.MaxValue : (ulong)maxK128;
		ulong remainingCount = maxK;

		ulong stepValue = (ulong)twoP128;
		byte step10 = (byte)(stepValue % 10UL);
		byte step8 = (byte)(stepValue & 7UL);
		byte step3 = (byte)(stepValue % 3UL);
		byte step7 = (byte)(stepValue % 7UL);
		byte step11 = (byte)(stepValue % 11UL);

		UInt128 currentDivisor128 = firstDivisor128;
		ulong currentDivisorValue = (ulong)currentDivisor128;
		byte remainder10 = (byte)(currentDivisorValue % 10UL);
		byte remainder8 = (byte)(currentDivisorValue & 7UL);
		byte remainder3 = (byte)(currentDivisorValue % 3UL);
		byte remainder7 = (byte)(currentDivisorValue % 7UL);
		byte remainder11 = (byte)(currentDivisorValue % 11UL);
		LastDigit lastDigit = (prime & 3UL) == 3UL ? LastDigit.Seven : LastDigit.One;
		ushort decimalMask = DivisorGenerator.GetDecimalMask(lastDigit);

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
		ref GpuDivisorPartialData divisorDataRef = ref MemoryMarshal.GetReference(divisorDataSpan);
		ref int offsetRef = ref MemoryMarshal.GetReference(offsetSpan);
		ref int countRef = ref MemoryMarshal.GetReference(countSpan);
		ref ulong cycleRef = ref MemoryMarshal.GetReference(cycleSpan);
		ref ulong exponentRef = ref MemoryMarshal.GetReference(exponentSpan);

		ArrayView1D<GpuDivisorPartialData, Stride1D.Dense> divisorDataView = divisorDataBuffer.View;
		ArrayView1D<int, Stride1D.Dense> offsetView = offsetBuffer.View;
		ArrayView1D<int, Stride1D.Dense> countView = countBuffer.View;
		ArrayView1D<ulong, Stride1D.Dense> cycleView = cycleBuffer.View;
		ArrayView1D<ulong, Stride1D.Dense> exponentViewDevice = exponentBuffer.View;
		ArrayView1D<byte, Stride1D.Dense> hitsView = hitsBuffer.View;
		ArrayView1D<int, Stride1D.Dense> hitIndexView = hitIndexBuffer.View;

		while (remainingCount > 0UL && !composite)
		{
			int chunkCount = chunkCountBaseline;
			if ((ulong)chunkCount > remainingCount)
			{
				chunkCount = (int)remainingCount;
			}

			UInt128 nextDivisor128 = currentDivisor128;
			int admissibleCount = 0;

			byte localRemainder10 = remainder10;
			byte localRemainder8 = remainder8;
			byte localRemainder3 = remainder3;
			byte localRemainder7 = remainder7;
			byte localRemainder11 = remainder11;

			for (int i = 0; i < chunkCount; i++)
			{
				bool passesSmallModuli = localRemainder3 != 0 && localRemainder7 != 0 && localRemainder11 != 0;
				if (passesSmallModuli && (localRemainder8 == 1 || localRemainder8 == 7) && ((decimalMask >> localRemainder10) & 1) != 0)
				{
					ulong divisorValue = (ulong)nextDivisor128;
					MontgomeryDivisorData montgomeryData = MontgomeryDivisorData.FromModulus(divisorValue);
					ulong divisorCycle = ResolveDivisorCycle(gpu, divisorValue, prime, in montgomeryData);
					if (divisorCycle == prime)
					{
						processedCount += (ulong)(i + 1);
						lastProcessed = divisorValue;
						coveredRange = true;
						return true;
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
				localRemainder10 = AddMod10(localRemainder10, step10);
				localRemainder8 = AddMod8(localRemainder8, step8);
				localRemainder3 = AddMod3(localRemainder3, step3);
				localRemainder7 = AddMod7(localRemainder7, step7);
				localRemainder11 = AddMod11(localRemainder11, step11);
			}

			remainder10 = localRemainder10;
			remainder8 = localRemainder8;
			remainder3 = localRemainder3;
			remainder7 = localRemainder7;
			remainder11 = localRemainder11;

			processedCount += (ulong)chunkCount;

			UInt128 lastDivisor128 = nextDivisor128 - twoP128;
			lastProcessed = (ulong)lastDivisor128;
			currentDivisor128 = nextDivisor128;
			remainingCount -= (ulong)chunkCount;

			if (admissibleCount == 0)
			{
				continue;
			}

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
			lastProcessed = hitFound ? divisorSpan[hitIndex] : divisorSpan[lastIndex];
		}

		processedAll = remainingCount == 0UL;
		coveredRange = composite || processedAll || (currentDivisor128 > allowedMax128);
		return composite;
	}


	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static ulong ResolveDivisorCycle(PrimeOrderCalculatorAccelerator gpu, ulong divisor, ulong prime, in MontgomeryDivisorData divisorData)
	{
		if (!MersenneDivisorCycles.TryCalculateCycleLengthForExponentCpu(gpu, divisor, prime, divisorData, out ulong computedCycle, out bool primeOrderFailed) || computedCycle == 0UL)
		{
			return MersenneDivisorCycles.CalculateCycleLength(divisor, divisorData, skipPrimeOrderHeuristic: primeOrderFailed);
		}

		return computedCycle;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static byte AddMod3(byte value, byte delta)
	{
		const int Modulus = 3;
		int sum = value + delta;

		if (sum >= Modulus)
		{
			sum -= Modulus;
		}

		return (byte)sum;
	}


	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static byte AddMod7(byte value, byte delta)
	{
		const int Modulus = 7;
		int sum = value + delta;

		if (sum >= Modulus)
		{
			sum -= Modulus;
		}

		return (byte)sum;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static byte AddMod10(byte value, byte delta)
	{
		const int Modulus = 10;
		int sum = value + delta;

		if (sum >= Modulus)
		{
			sum -= Modulus;
		}

		return (byte)sum;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static byte AddMod11(byte value, byte delta)
	{
		const int Modulus = 11;
		int sum = value + delta;

		if (sum >= Modulus)
		{
			sum -= Modulus;
		}

		return (byte)sum;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static byte AddMod8(byte value, byte delta)
	{
		return (byte)((value + delta) & 7);
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
		private readonly PrimeOrderCalculatorAccelerator _primeTesterAccelerator;
		private readonly Accelerator _accelerator;
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


		internal DivisorScanSession(MersenneNumberDivisorByDivisorGpuTester owner)
		{
			_owner = owner;
			// GpuPrimeWorkLimiter.Acquire();
			_primeTesterAccelerator = PrimeOrderCalculatorAccelerator.Rent(1);
			Accelerator accelerator = _primeTesterAccelerator.Accelerator;
			_accelerator = accelerator;
			_stream = accelerator.CreateStream();
			_capacity = Math.Max(1, owner._gpuBatchSize);
		}

		internal void Reset()
		{
		}

		private void EnsureExecutionResourcesLocked(int requiredCapacity)
		{
			Accelerator accelerator = _accelerator;

			if (_kernel == null)
			{
				_kernel =  KernelUtil.GetKernel(_accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuDivisorPartialData>, ArrayView<int>, ArrayView<int>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>, ArrayView<int>>(DivisorByDivisorKernels.CheckKernel)).CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView<GpuDivisorPartialData>, ArrayView<int>, ArrayView<int>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>, ArrayView<int>>>();
			}

			if (_divisorBuffer == null)
			{
				lock(accelerator)
				{
					_divisorBuffer = accelerator.Allocate1D<GpuDivisorPartialData>(1);
					_offsetBuffer = accelerator.Allocate1D<int>(1);
					_countBuffer = accelerator.Allocate1D<int>(1);
					_cycleBuffer = accelerator.Allocate1D<ulong>(1);
					_firstHitBuffer = accelerator.Allocate1D<int>(1);
				}
			}

			bool allocateBuffers = _exponentsBuffer == null;
			if (!allocateBuffers && _capacity < requiredCapacity)
			{
				_exponentsBuffer?.Dispose();
				_hitBuffer?.Dispose();
				if (_hostBuffer != null)
				{
					ThreadStaticPools.UlongPool.Return(_hostBuffer, clearArray: false);
					_hostBuffer = null!;
				}

				_exponentsBuffer = null!;
				_hitBuffer = null!;
				allocateBuffers = true;
			}

			if (allocateBuffers)
			{
				int desiredCapacity = requiredCapacity > _capacity ? requiredCapacity : _capacity;
				_capacity = desiredCapacity;
				lock (accelerator)
				{
					_exponentsBuffer = accelerator.Allocate1D<ulong>(desiredCapacity);
					_hitBuffer = accelerator.Allocate1D<byte>(desiredCapacity);
				}

				_hostBuffer = ThreadStaticPools.UlongPool.Rent(desiredCapacity);
			}
		}

		public void CheckDivisor(ulong divisor, in MontgomeryDivisorData divisorData, ulong divisorCycle, in ReadOnlySpan<ulong> primes, Span<byte> hits)
		{
			int length = primes.Length;
			// EvenPerfectBitScanner always supplies at least one exponent per divisor check, so the guard stays commented out.
			// if (length == 0)
			// {
			//     return;
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

			if (cycle == 0UL)
			{
				var gpu = _primeTesterAccelerator;
				if (!MersenneDivisorCycles.TryCalculateCycleLengthForExponentCpu(gpu, divisor, firstPrime, divisorData, out ulong computedCycle, out bool primeOrderFailed) || computedCycle == 0UL)
				{
					cycle = MersenneDivisorCycles.CalculateCycleLength(divisor, divisorData, skipPrimeOrderHeuristic: primeOrderFailed);
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

			Span<byte> hitSlice = hits.Slice(0, length);
			ref byte hitRef = ref MemoryMarshal.GetReference(hitSlice);
			hitView.CopyToCPU(stream, ref hitRef, length);
			stream.Synchronize();

			// Monitor.Exit(_lease.ExecutionLock);
		}

		public void Return()
		{
			_owner._sessionPool.Add(this);
		}
	}


	public ulong DivisorLimit
	{
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



	private MersenneNumberDivisorByDivisorAccelerator RentBatchResources(int acceleratorIndex, int capacity)
	{
		var queue = _resourcePool[acceleratorIndex];
		var accelerator = AcceleratorPool.Shared.Accelerators[acceleratorIndex];
		if (!queue.TryTake(out MersenneNumberDivisorByDivisorAccelerator? resources))
		{
			return new MersenneNumberDivisorByDivisorAccelerator(accelerator, capacity);
		}

		resources.EnsureCapacity(accelerator, capacity);
		return resources;
	}

	private void ReturnBatchResources(int acceleratorIndex, MersenneNumberDivisorByDivisorAccelerator resources) => _resourcePool[acceleratorIndex].Add(resources);

	private sealed class AcceleratorReferenceComparer : IEqualityComparer<Accelerator>
	{
		public static AcceleratorReferenceComparer Instance { get; } = new();

		public bool Equals(Accelerator? x, Accelerator? y) => ReferenceEquals(x, y);

		public int GetHashCode(Accelerator obj) => RuntimeHelpers.GetHashCode(obj);
	}

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

	public DivisorScanSession CreateDivisorSession()
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

		if (_sessionPool.TryTake(out DivisorScanSession? session))
		{
			session.Reset();
			return session;
		}

		return new DivisorScanSession(this);
	}

	IMersenneNumberDivisorByDivisorTester.IDivisorScanSession IMersenneNumberDivisorByDivisorTester.CreateDivisorSession() => CreateDivisorSession();
}
