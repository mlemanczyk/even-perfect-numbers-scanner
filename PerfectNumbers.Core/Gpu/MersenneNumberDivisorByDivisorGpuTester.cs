using System.Buffers;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

public sealed class MersenneNumberDivisorByDivisorGpuTester : IMersenneNumberDivisorByDivisorTester
{
	private int _gpuBatchSize = GpuConstants.ScanBatchSize;
	private readonly object _sync = new();
	private ulong _divisorLimit;
	private bool _isConfigured;
	private bool _useDivisorCycles;

	public readonly struct MontgomeryDivisorData(ulong modulus, ulong nPrime, ulong montgomeryOne, ulong montgomeryTwo)
	{
		public readonly ulong Modulus = modulus;
		public readonly ulong NPrime = nPrime;
		public readonly ulong MontgomeryOne = montgomeryOne;
		public readonly ulong MontgomeryTwo = montgomeryTwo;
	}

	private readonly ConcurrentDictionary<Accelerator, Action<Index1D, ulong, byte, ulong, ArrayView<MontgomeryDivisorData>, ArrayView<byte>>> _kernelCache = new();
	private readonly ConcurrentDictionary<Accelerator, Action<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<ulong>>> _kernelExponentCache = new();
	private readonly ConcurrentDictionary<Accelerator, ConcurrentBag<BatchResources>> _resourcePools = new(AcceleratorReferenceComparer.Instance);
	private readonly ConcurrentBag<DivisorScanSession> _sessionPool = new();

	private Action<Index1D, ulong, byte, ulong, ArrayView<MontgomeryDivisorData>, ArrayView<byte>> GetKernel(Accelerator accelerator) =>
																	_kernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ulong, byte, ulong, ArrayView<MontgomeryDivisorData>, ArrayView<byte>>(CheckKernel));
	private Action<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<ulong>> GetKernelByPrimeExponent(Accelerator accelerator) =>
																														  _kernelExponentCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<ulong>>(ComputePrimeExponentKernel));

	public int GpuBatchSize
	{
		get => _gpuBatchSize;
		set => _gpuBatchSize = Math.Max(1, value);
	}

	int IMersenneNumberDivisorByDivisorTester.BatchSize
	{
		get => GpuBatchSize;
		set => GpuBatchSize = value;
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
			Action<Index1D, ulong, byte, ulong, ArrayView<MontgomeryDivisorData>, ArrayView<byte>> kernel,
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
		ulong currentDivisor = 3UL, cycle = 0UL, next, nextDivisor;
		int batchSize, i;
		bool reachedEndInBatch;

		Span<MontgomeryDivisorData> divisorDataSpan;
		Span<ulong> divisorSpan;
		Span<byte> hitsSpan;
		ArrayView1D<MontgomeryDivisorData, Stride1D.Dense> divisorView;
		ArrayView1D<byte, Stride1D.Dense> hitsView;
		DivisorCycleCache divisorCyclesCache = DivisorCycleCache.Shared;
		DivisorCycleCache.CycleBlock? cycleLease = useCycles ? divisorCyclesCache.Acquire(currentDivisor) : null;

		while (currentDivisor <= allowedMax)
		{
			batchSize = 0;
			reachedEndInBatch = false;

			while (batchSize < batchCapacity && currentDivisor <= allowedMax)
			{
				nextDivisor = currentDivisor + 2UL;
				processedCount++;

				bool includeDivisor = true;
				if (useCycles)
				{
					if (currentDivisor > cycleLease!.End)
					{
						cycleLease = divisorCyclesCache.Acquire(currentDivisor);
					}

					cycle = prime % cycleLease.GetCycle(currentDivisor);
					if (cycle != 0UL)
					{
						includeDivisor = false;
					}
				}

				if (includeDivisor)
				{
					divisors[batchSize++] = currentDivisor;
				}

				lastProcessed = currentDivisor;
				next = nextDivisor;

				if (next <= currentDivisor)
				{
					reachedEndInBatch = true;
					break;
				}

				if (next > allowedMax)
				{
					currentDivisor = next;
					reachedEndInBatch = true;
					break;
				}

				currentDivisor = next;
			}

			if (batchSize == 0)
			{
				if (reachedEndInBatch)
				{
					processedAll = true;
					break;
				}

				continue;
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
			byte useCycleChecks = useCycles ? (byte)1 : (byte)0;
			kernel(batchSize, prime, useCycleChecks, cycle, divisorView, hitsView);
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

		coveredRange = composite || processedAll || currentDivisor > allowedMax;

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

	private static void CheckKernel(Index1D index, ulong prime, byte useCycleChecks, ulong cycle, ArrayView<MontgomeryDivisorData> divisors, ArrayView<byte> hits)
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
			prime = cycle;
		}

		hits[index] = prime.Pow2MontgomeryMod(divisor) == 1UL ? (byte)1 : (byte)0;
	}

	public sealed class DivisorScanSession : IMersenneNumberDivisorByDivisorTester.IDivisorScanSession
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
			_kernel = owner.GetKernelByPrimeExponent(_accelerator);
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

		public void CheckDivisor(ulong divisor, ulong divisorCycle, ReadOnlySpan<ulong> primes, Span<byte> hits)
		{
			if (_disposed)
			{
				throw new ObjectDisposedException(nameof(DivisorScanSession));
			}

			MersenneNumberDivisorByDivisorGpuTester owner = _owner;
			int gpuBatchSize = owner._gpuBatchSize,
				primesLength = primes.Length;

			MontgomeryDivisorData divisorData = CreateMontgomeryDivisorData(divisor);
			ulong modulus = divisorData.Modulus;
			if (modulus <= 1UL || (modulus & 1UL) == 0UL)
			{
				return;
			}

			Accelerator accelerator = _accelerator;
			Action<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<ulong>> kernel = _kernel;
			ArrayView1D<ulong, Stride1D.Dense> exponentView, exponentsView = _exponentsBuffer.View;
			ArrayView1D<ulong, Stride1D.Dense> resultView, resultsView = _resultsBuffer.View;

			Span<ulong> exponentSlice, hostSpan = _hostBuffer.AsSpan();
			Span<int> positionSpan = _positionBuffer.AsSpan();
			Span<byte> hitsSlice;
			ReadOnlySpan<ulong> primesSlice;

			int batchSize, computeCount, i, offset = 0;
			bool useDivisorCycles = owner._useDivisorCycles;
			ulong residue;
			while (offset < primesLength)
			{
				batchSize = Math.Min(gpuBatchSize, primesLength - offset);
				primesSlice = primes.Slice(offset, batchSize);
				hitsSlice = hits.Slice(offset, batchSize);

				computeCount = 0;

				if (useDivisorCycles)
				{
					for (i = 0; i < batchSize; i++)
					{
						residue = primesSlice[i] % divisorCycle;
						if (residue != 0UL)
						{
							// We don't need to clear hitsSlice[i] because the parent function calls .Clear() on the buffer
							continue;
						}

						hostSpan[computeCount] = residue;
						positionSpan[computeCount] = i;
						computeCount++;
					}
				}
				else
				{
					for (i = 0; i < batchSize; i++)
					{
						hostSpan[computeCount] = primesSlice[i];
						positionSpan[computeCount] = i;
						computeCount++;
					}
				}

				if (computeCount > 0)
				{
					exponentSlice = hostSpan[..computeCount];
					exponentView = exponentsView.SubView(0, computeCount);

					exponentView.CopyFromCPU(ref MemoryMarshal.GetReference(exponentSlice), computeCount);
					resultView = resultsView.SubView(0, computeCount);
					kernel(computeCount, divisorData, exponentView, resultView);
					accelerator.Synchronize();
					resultView.CopyToCPU(ref MemoryMarshal.GetReference(exponentSlice), computeCount);

					for (i = 0; i < computeCount; i++)
					{
						hitsSlice[positionSpan[i]] = exponentSlice[i] == 1UL ? (byte)1 : (byte)0;
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

	internal void ReturnSession(DivisorScanSession session) => _sessionPool.Add(session);

	private static void ComputePrimeExponentKernel(Index1D index, MontgomeryDivisorData divisor, ArrayView<ulong> exponents, ArrayView<ulong> results)
	{
		ulong modulus = divisor.Modulus;
		if (modulus <= 1UL || (modulus & 1UL) == 0UL)
		{
			results[index] = 0UL;
			return;
		}

		results[index] = exponents[index].Pow2MontgomeryMod(divisor);
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
		var bag = _resourcePools.GetOrAdd(accelerator, static _ => []);
		if (bag.TryTake(out BatchResources? resources))
		{
			return resources;
		}

		return new BatchResources(accelerator, capacity);
	}

	private void ReturnBatchResources(Accelerator accelerator, BatchResources resources) => _resourcePools.GetOrAdd(accelerator, static _ => []).Add(resources);

	private sealed class BatchResources : IDisposable
	{
		internal BatchResources(Accelerator accelerator, int capacity)
		{
			DivisorsBuffer = accelerator.Allocate1D<MontgomeryDivisorData>(capacity);
			HitsBuffer = accelerator.Allocate1D<byte>(capacity);
			Divisors = ArrayPool<ulong>.Shared.Rent(capacity);
			Hits = ArrayPool<byte>.Shared.Rent(capacity);
			DivisorData = ArrayPool<MontgomeryDivisorData>.Shared.Rent(capacity);
			Capacity = capacity;
		}

		internal readonly MemoryBuffer1D<MontgomeryDivisorData, Stride1D.Dense> DivisorsBuffer;

		internal readonly MemoryBuffer1D<byte, Stride1D.Dense> HitsBuffer;

		internal readonly ulong[] Divisors;

		internal readonly byte[] Hits;

		internal readonly MontgomeryDivisorData[] DivisorData;

		internal readonly int Capacity;

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

	IMersenneNumberDivisorByDivisorTester.IDivisorScanSession IMersenneNumberDivisorByDivisorTester.CreateDivisorSession() => CreateDivisorSession();
}

