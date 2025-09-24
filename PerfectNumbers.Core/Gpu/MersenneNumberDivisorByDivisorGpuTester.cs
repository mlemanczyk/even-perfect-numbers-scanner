using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core;

namespace PerfectNumbers.Core.Gpu;

public sealed class MersenneNumberDivisorByDivisorGpuTester
{
	private int _gpuBatchSize = GpuConstants.ScanBatchSize;
	private readonly object _sync = new();
	private ulong _divisorLimit;
	private bool _isConfigured;
	private ulong _lastStatusDivisor;
	private bool _useDivisorCycles;

	public readonly struct MontgomeryDivisorData(ulong modulus, ulong nPrime, ulong montgomeryOne, ulong montgomeryTwo)
	{
		public readonly ulong Modulus = modulus;
		public readonly ulong NPrime = nPrime;
		public readonly ulong MontgomeryOne = montgomeryOne;
		public readonly ulong MontgomeryTwo = montgomeryTwo;
	}

	private readonly ConcurrentDictionary<Accelerator, Action<Index1D, ulong, byte, ArrayView<MontgomeryDivisorData>, ArrayView<byte>>> _kernelCache = new();
	private readonly ConcurrentDictionary<Accelerator, Action<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<ulong>>> _kernelDeltaCache = new();
	private readonly ConcurrentDictionary<Accelerator, ConcurrentBag<BatchResources>> _resourcePools = new(AcceleratorReferenceComparer.Instance);
	private readonly ConcurrentBag<DivisorScanSession> _sessionPool = new();

	private Action<Index1D, ulong, byte, ArrayView<MontgomeryDivisorData>, ArrayView<byte>> GetKernel(Accelerator accelerator) =>
																	_kernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ulong, byte, ArrayView<MontgomeryDivisorData>, ArrayView<byte>>(CheckKernel));
	private Action<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<ulong>> GetKernelByPrimeDelta(Accelerator accelerator) =>
																	_kernelDeltaCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<ulong>>(ComputePrimeDeltaKernel));

	public int GpuBatchSize
	{
		get => _gpuBatchSize;
		set => _gpuBatchSize = Math.Max(1, value);
	}

	public bool UseDivisorCycles
	{
		get => _useDivisorCycles;
		set => _useDivisorCycles = value;
	}

	public void ConfigureFromMaxPrime(ulong maxPrime)
	{
		lock (_sync)
		{
			_divisorLimit = ComputeDivisorLimitFromMaxPrime(maxPrime);
			_lastStatusDivisor = 0UL;
			_isConfigured = true;
		}
	}

	public bool IsPrime(ulong prime, out bool divisorsExhausted)
	{
		ulong allowedMax;
		bool useCycles;
		int batchCapacity;

		lock (_sync)
		{
			if (!_isConfigured)
			{
				throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
			}

			allowedMax = ComputeAllowedMaxDivisor(prime, _divisorLimit);
			useCycles = _useDivisorCycles;
			batchCapacity = _gpuBatchSize;
		}

		if (allowedMax < 3UL)
		{
			divisorsExhausted = true;
			return true;
		}

		bool composite;
		bool coveredRange;
		ulong processedCount;
		ulong lastProcessed;

		var gpuLease = GpuContextPool.RentPreferred(preferCpu: false);
		var accelerator = gpuLease.Accelerator;
		var kernel = GetKernel(accelerator);
		BatchResources resources = RentBatchResources(accelerator, batchCapacity);

		try
		{
			composite = CheckDivisors(
							prime,
							allowedMax,
							useCycles,
							accelerator,
							kernel,
							resources.DivisorsBuffer,
							resources.HitsBuffer,
							resources.Divisors,
							resources.Hits,
							resources.DivisorData,
							out lastProcessed,
							out coveredRange,
							out processedCount
			);
		}
		finally
		{
			ReturnBatchResources(accelerator, resources);
			gpuLease.Dispose();
		}

		if (processedCount > 0UL)
		{
			lock (_sync)
			{
				UpdateStatusUnsafe(lastProcessed, processedCount);
			}
		}

		if (composite)
		{
			divisorsExhausted = true;
			return false;
		}

		divisorsExhausted = coveredRange;
		return true;
	}

	private static bool CheckDivisors(
			ulong prime,
			ulong allowedMax,
			bool useCycles,
			Accelerator accelerator,
						Action<Index1D, ulong, byte, ArrayView<MontgomeryDivisorData>, ArrayView<byte>> kernel,
			MemoryBuffer1D<MontgomeryDivisorData, Stride1D.Dense> divisorsBuffer,
			MemoryBuffer1D<byte, Stride1D.Dense> hitsBuffer,
			ulong[] divisors,
			byte[] hits,
			MontgomeryDivisorData[] divisorData,
			out ulong lastProcessed,
			out bool coveredRange,
			out ulong processedCount)
	{
		lastProcessed = 0UL;
		processedCount = 0UL;

		if (allowedMax < 3UL)
		{
			coveredRange = true;
			return false;
		}

		int batchCapacity = (int)divisorsBuffer.Length;

		bool composite = false;
		bool processedAll = false;
		ulong currentDivisor, divisor = 3UL, next, nextDivisor;
		int batchSize, i;
		bool reachedEndInBatch;

		Span<MontgomeryDivisorData> divisorDataSpan;
		Span<ulong> divisorSpan;
		Span<byte> hitsSpan;
		ArrayView1D<MontgomeryDivisorData, Stride1D.Dense> divisorView;
		ArrayView1D<byte, Stride1D.Dense> hitsView;

		while (divisor <= allowedMax)
		{
			batchSize = 0;
			reachedEndInBatch = false;

			while (batchSize < batchCapacity && divisor <= allowedMax)
			{
				currentDivisor = divisor;
				nextDivisor = currentDivisor + 2UL;
				processedCount++;

				divisors[batchSize++] = currentDivisor;
				lastProcessed = currentDivisor;
				next = nextDivisor;

				// This will be only true when we exceed ulong.MaxValue
				if (next <= divisor)
				{
					reachedEndInBatch = true;
					break;
				}

				if (next > allowedMax)
				{
					divisor = next;
					reachedEndInBatch = true;
					break;
				}

				divisor = next;
			}

			if (batchSize == 0)
			{
				if (reachedEndInBatch)
				{
					processedAll = true;
				}

				break;
			}

			divisorDataSpan = divisorData.AsSpan(0, batchSize);
			divisorView = divisorsBuffer.View.SubView(0, batchSize);
			hitsView = hitsBuffer.View.SubView(0, batchSize);

			divisorSpan = divisors.AsSpan(0, batchSize);
			hitsSpan = hits.AsSpan(0, batchSize);

			for (i = 0; i < batchSize; i++)
			{
				divisorDataSpan[i] = CreateMontgomeryDivisorData(divisorSpan[i]);
			}

			divisorView.CopyFromCPU(ref MemoryMarshal.GetReference(divisorDataSpan), batchSize);
			kernel(batchSize, prime, useCycles ? (byte)1 : (byte)0, divisorView, hitsView);
			accelerator.Synchronize();
			hitsView.CopyToCPU(ref MemoryMarshal.GetReference(hitsSpan), batchSize);

			for (i = 0; i < batchSize; i++)
			{
				if (hitsSpan[i] != 0)
				{
					composite = true;
					lastProcessed = divisorSpan[i];
					break;
				}
			}

			if (!composite)
			{
				lastProcessed = divisorSpan[batchSize - 1];
			}
			if (composite)
			{
				break;
			}

			if (reachedEndInBatch)
			{
				processedAll = true;
				break;
			}
		}

		coveredRange = composite || processedAll || divisor > allowedMax;

		return composite;
	}

	private static ulong ComputeDivisorLimitFromMaxPrime(ulong maxPrime)
	{
		if (maxPrime <= 1UL)
		{
			return 0UL;
		}

		if (maxPrime - 1UL >= 64UL)
		{
			return ulong.MaxValue;
		}

		return (1UL << (int)(maxPrime - 1UL)) - 1UL;
	}

	private static ulong ComputeAllowedMaxDivisor(ulong prime, ulong divisorLimit)
	{
		if (prime <= 1UL)
		{
			return 0UL;
		}

		if (prime - 1UL >= 64UL)
		{
			return divisorLimit;
		}

		return Math.Min((1UL << (int)(prime - 1UL)) - 1UL, divisorLimit);
	}

	private void UpdateStatusUnsafe(ulong lastProcessed, ulong processedCount)
	{
		if (processedCount == 0UL)
		{
			return;
		}

		ulong interval = PerfectNumberConstants.ConsoleInterval;
		if (interval == 0UL)
		{
			_lastStatusDivisor = 0UL;
			return;
		}

		ulong total = _lastStatusDivisor + processedCount;
		_lastStatusDivisor = total % interval;

		// Removed noisy console output; status is tracked internally only.
	}

	private static void CheckKernel(Index1D index, ulong prime, byte useCycleChecks, ArrayView<MontgomeryDivisorData> divisors, ArrayView<byte> hits)
	{
		MontgomeryDivisorData divisor = divisors[index];
		ulong modulus = divisor.Modulus;
		if (modulus <= 1UL || (modulus & 1UL) == 0UL)
		{
			hits[index] = 0;
			return;
		}

		if (useCycleChecks != 0)
		{
			ulong cycle = MersenneDivisorCycles.CalculateCycleLengthGpu(modulus);
			if (cycle == 0UL || prime % cycle != 0UL)
			{
				hits[index] = 0;
				return;
			}
		}

		hits[index] = prime.Pow2MontgomeryMod(divisor) == 1UL ? (byte)1 : (byte)0;
	}

	public sealed class DivisorScanSession : IDisposable
	{
		private readonly MersenneNumberDivisorByDivisorGpuTester _owner;
		private readonly GpuContextPool.GpuContextLease _lease;
		private readonly Accelerator _accelerator;
		private readonly Action<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<ulong>> _kernel;
		private MemoryBuffer1D<ulong, Stride1D.Dense> _exponentsBuffer;
		private MemoryBuffer1D<ulong, Stride1D.Dense> _resultsBuffer;
		private ulong[] _hostBuffer;
		private int[] _positionBuffer;
		private int _capacity;
		private bool _disposed;

		internal DivisorScanSession(MersenneNumberDivisorByDivisorGpuTester owner)
		{
			_owner = owner;
			_lease = GpuContextPool.RentPreferred(preferCpu: false);
			_accelerator = _lease.Accelerator;
			_kernel = owner.GetKernelByPrimeDelta(_accelerator);
			_capacity = Math.Max(1, owner._gpuBatchSize);
			_exponentsBuffer = _accelerator.Allocate1D<ulong>(_capacity);
			_resultsBuffer = _accelerator.Allocate1D<ulong>(_capacity);
			_hostBuffer = ArrayPool<ulong>.Shared.Rent(_capacity);
			_positionBuffer = ArrayPool<int>.Shared.Rent(_capacity);
		}

		internal void Reset()
		{
			_disposed = false;
		}

		private void EnsureCapacity(int requiredCapacity)
		{
			if (requiredCapacity <= _capacity)
			{
				return;
			}

			_exponentsBuffer.Dispose();
			_resultsBuffer.Dispose();
			ArrayPool<ulong>.Shared.Return(_hostBuffer, clearArray: false);
			ArrayPool<int>.Shared.Return(_positionBuffer, clearArray: false);

			_capacity = requiredCapacity;
			_exponentsBuffer = _accelerator.Allocate1D<ulong>(_capacity);
			_resultsBuffer = _accelerator.Allocate1D<ulong>(_capacity);
			_hostBuffer = ArrayPool<ulong>.Shared.Rent(_capacity);
			_positionBuffer = ArrayPool<int>.Shared.Rent(_capacity);
		}

		public void CheckDivisor(ulong divisor, ReadOnlySpan<ulong> primes, Span<byte> hits)
		{
			if (_disposed)
			{
				throw new ObjectDisposedException(nameof(DivisorScanSession));
			}

			int primesLength = primes.Length;
			if (primesLength == 0)
			{
				return;
			}

			MersenneNumberDivisorByDivisorGpuTester owner = _owner;
			int gpuBatchSize = owner._gpuBatchSize;
			EnsureCapacity(gpuBatchSize);

			MontgomeryDivisorData divisorData = CreateMontgomeryDivisorData(divisor);
			ulong modulus = divisorData.Modulus;
			if (modulus <= 1UL || (modulus & 1UL) == 0UL)
			{
				return;
			}

			bool useCycles = owner._useDivisorCycles;
			ulong cycle = 0UL;
			bool cycleEnabled = false;
			if (useCycles)
			{
				cycle = MersenneDivisorCycles.CalculateCycleLengthGpu(modulus);
				cycleEnabled = cycle != 0UL;
			}

			Accelerator accelerator = _accelerator;
			Action<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<ulong>> kernel = _kernel;
			ArrayView1D<ulong, Stride1D.Dense> exponentsView = _exponentsBuffer.View;
			ArrayView1D<ulong, Stride1D.Dense> resultsView = _resultsBuffer.View;

			int offset = 0;
			int batchSize;
			ReadOnlySpan<ulong> primesSlice;
			Span<byte> hitsSlice;
			Span<ulong> hostSpan = _hostBuffer.AsSpan();
			Span<int> positionSpan = _positionBuffer.AsSpan();
			bool hasResidue = false;
			ulong lastPrimeWithResidue = 0UL;
			ulong lastResidue = 0UL;

			while (offset < primesLength)
			{
				batchSize = Math.Min(gpuBatchSize, primesLength - offset);
				primesSlice = primes.Slice(offset, batchSize);
				hitsSlice = hits.Slice(offset, batchSize);

				int computeCount = 0;
				ulong deltaBasePrime = lastPrimeWithResidue;
				bool deltaHasResidue = hasResidue;

				for (int i = 0; i < batchSize; i++)
				{
					ulong prime = primesSlice[i];
					if (cycleEnabled && prime % cycle != 0UL)
					{
						hitsSlice[i] = 0;
						continue;
					}

					ulong delta;
					if (deltaHasResidue)
					{
						delta = prime >= deltaBasePrime ? prime - deltaBasePrime : prime;
					}
					else
					{
						delta = prime;
					}
					hostSpan[computeCount] = delta;
					positionSpan[computeCount] = i;
					computeCount++;
					deltaBasePrime = prime;
					deltaHasResidue = true;
				}

				if (computeCount > 0)
				{
					Span<ulong> exponentSlice = hostSpan.Slice(0, computeCount);
					ArrayView1D<ulong, Stride1D.Dense> exponentView = exponentsView.SubView(0, computeCount);
					exponentView.CopyFromCPU(ref MemoryMarshal.GetReference(exponentSlice), computeCount);
					ArrayView1D<ulong, Stride1D.Dense> resultView = resultsView.SubView(0, computeCount);
					kernel(computeCount, divisorData, exponentView, resultView);
					accelerator.Synchronize();
					resultView.CopyToCPU(ref MemoryMarshal.GetReference(exponentSlice), computeCount);

					for (int i = 0; i < computeCount; i++)
					{
						int position = positionSpan[i];
						ulong prime = primesSlice[position];
						ulong deltaResult = exponentSlice[i];
						ulong residue = hasResidue
												? lastResidue.MulMod64(deltaResult, modulus)
												: deltaResult % modulus;

						hitsSlice[position] = residue == 1UL ? (byte)1 : (byte)0;
						hasResidue = true;
						lastResidue = residue;
						lastPrimeWithResidue = prime;
					}
				}

				offset += batchSize;
			}
		}

		public void Dispose()
		{
			if (_disposed)
			{
				return;
			}

			_disposed = true;
			_owner.ReturnSession(this);
		}

	}

	internal void ReturnSession(DivisorScanSession session)
	{
		_sessionPool.Add(session);
	}

	private static void ComputePrimeDeltaKernel(Index1D index, MontgomeryDivisorData divisor, ArrayView<ulong> deltas, ArrayView<ulong> results)
	{
		ulong modulus = divisor.Modulus;
		if (modulus <= 1UL || (modulus & 1UL) == 0UL)
		{
			results[index] = 0UL;
			return;
		}

		ulong exponent = deltas[index];
		results[index] = exponent.Pow2MontgomeryMod(divisor);
	}

	private static MontgomeryDivisorData CreateMontgomeryDivisorData(ulong modulus)
	{
		if (modulus <= 1UL || (modulus & 1UL) == 0UL)
		{
			return new(modulus, 0UL, 0UL, 0UL);
		}

		return new(
			modulus,
			ComputeMontgomeryNPrime(modulus),
			ComputeMontgomeryResidue(1UL, modulus),
			ComputeMontgomeryResidue(2UL, modulus)
		);
	}

	private static ulong ComputeMontgomeryResidue(ulong value, ulong modulus) => (ulong)((UInt128)value * (UInt128.One << 64) % modulus);

	private static ulong ComputeMontgomeryNPrime(ulong modulus)
	{
		ulong inv = modulus;
		inv *= unchecked(2UL - modulus * inv);
		inv *= unchecked(2UL - modulus * inv);
		inv *= unchecked(2UL - modulus * inv);
		inv *= unchecked(2UL - modulus * inv);
		inv *= unchecked(2UL - modulus * inv);
		inv *= unchecked(2UL - modulus * inv);
		return unchecked(0UL - inv);
	}

	public ulong DivisorLimit
	{
		get
		{
			lock (_sync)
			{
				if (!_isConfigured)
				{
					throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
				}

				return _divisorLimit;
			}
		}
	}

	private BatchResources RentBatchResources(Accelerator accelerator, int capacity)
	{
		var bag = _resourcePools.GetOrAdd(accelerator, static _ => new ConcurrentBag<BatchResources>());
		if (bag.TryTake(out BatchResources? resources))
		{
			resources.EnsureCapacity(capacity);
			return resources;
		}

		return new BatchResources(accelerator, capacity);
	}

	private void ReturnBatchResources(Accelerator accelerator, BatchResources resources)
	{
		_resourcePools.GetOrAdd(accelerator, static _ => new ConcurrentBag<BatchResources>()).Add(resources);
	}

	private sealed class BatchResources : IDisposable
	{
		private readonly Accelerator _accelerator;

		internal BatchResources(Accelerator accelerator, int capacity)
		{
			_accelerator = accelerator;
			DivisorsBuffer = accelerator.Allocate1D<MontgomeryDivisorData>(capacity);
			HitsBuffer = accelerator.Allocate1D<byte>(capacity);
			Divisors = ArrayPool<ulong>.Shared.Rent(capacity);
			Hits = ArrayPool<byte>.Shared.Rent(capacity);
			DivisorData = ArrayPool<MontgomeryDivisorData>.Shared.Rent(capacity);
			Capacity = capacity;
		}

		internal MemoryBuffer1D<MontgomeryDivisorData, Stride1D.Dense> DivisorsBuffer { get; private set; }

		internal MemoryBuffer1D<byte, Stride1D.Dense> HitsBuffer { get; private set; }

		internal ulong[] Divisors { get; private set; }

		internal byte[] Hits { get; private set; }

		internal MontgomeryDivisorData[] DivisorData { get; private set; }

		internal int Capacity { get; private set; }

		public void EnsureCapacity(int requiredCapacity)
		{
			if (requiredCapacity <= Capacity)
			{
				return;
			}

			Resize(requiredCapacity);
		}

		private void Resize(int newCapacity)
		{
			DivisorsBuffer.Dispose();
			HitsBuffer.Dispose();
			ArrayPool<ulong>.Shared.Return(Divisors, clearArray: false);
			ArrayPool<byte>.Shared.Return(Hits, clearArray: false);
			ArrayPool<MontgomeryDivisorData>.Shared.Return(DivisorData, clearArray: false);

			DivisorsBuffer = _accelerator.Allocate1D<MontgomeryDivisorData>(newCapacity);
			HitsBuffer = _accelerator.Allocate1D<byte>(newCapacity);
			Divisors = ArrayPool<ulong>.Shared.Rent(newCapacity);
			Hits = ArrayPool<byte>.Shared.Rent(newCapacity);
			DivisorData = ArrayPool<MontgomeryDivisorData>.Shared.Rent(newCapacity);
			Capacity = newCapacity;
		}

		public void Dispose()
		{
			DivisorsBuffer.Dispose();
			HitsBuffer.Dispose();
			ArrayPool<ulong>.Shared.Return(Divisors, clearArray: false);
			ArrayPool<byte>.Shared.Return(Hits, clearArray: false);
			ArrayPool<MontgomeryDivisorData>.Shared.Return(DivisorData, clearArray: false);
		}
	}

	private sealed class AcceleratorReferenceComparer : IEqualityComparer<Accelerator>
	{
		public static AcceleratorReferenceComparer Instance { get; } = new();

		public bool Equals(Accelerator? x, Accelerator? y) => ReferenceEquals(x, y);

		public int GetHashCode(Accelerator obj) => RuntimeHelpers.GetHashCode(obj);
	}

	public ulong GetAllowedMaxDivisor(ulong prime)
	{
		lock (_sync)
		{
			if (!_isConfigured)
			{
				throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
			}

			return ComputeAllowedMaxDivisor(prime, _divisorLimit);
		}
	}

	public DivisorScanSession CreateDivisorSession()
	{
		lock (_sync)
		{
			if (!_isConfigured)
			{
				throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
			}

			if (_sessionPool.TryTake(out DivisorScanSession? session))
			{
				session.Reset();
				return session;
			}

			return new DivisorScanSession(this);
		}
	}
}

