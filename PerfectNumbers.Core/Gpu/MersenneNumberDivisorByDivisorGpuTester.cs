using System.Buffers;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Runtime;

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

	private readonly ConcurrentDictionary<Accelerator, Action<Index1D, ulong, ArrayView<MontgomeryDivisorData>, ArrayView<byte>>> _kernelCache = new();
	private readonly ConcurrentDictionary<Accelerator, Action<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<byte>>> _kernelByPrimeCache = new();
	private readonly ConcurrentBag<DivisorScanSession> _sessionPool = new();

	private Action<Index1D, ulong, ArrayView<MontgomeryDivisorData>, ArrayView<byte>> GetKernel(Accelerator accelerator) =>
									_kernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ulong, ArrayView<MontgomeryDivisorData>, ArrayView<byte>>(CheckKernel));
	private Action<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<byte>> GetKernelByPrime(Accelerator accelerator) =>
									_kernelByPrimeCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<byte>>(CheckKernelByPrime));

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

		// TODO: Can we move the creation of the buffers to one of the parent methods, outside of all loops and initialize them once for all divisors / candidates? Instead of creating buffers for each thread separately, let's pre-allocate them and re-use in threads.
		var divisorsBuffer = accelerator.Allocate1D<MontgomeryDivisorData>(batchCapacity);
		var hitsBuffer = accelerator.Allocate1D<byte>(batchCapacity);
		ulong[] divisors = ArrayPool<ulong>.Shared.Rent(batchCapacity);
		byte[] hits = ArrayPool<byte>.Shared.Rent(batchCapacity);
		MontgomeryDivisorData[] divisorData = ArrayPool<MontgomeryDivisorData>.Shared.Rent(batchCapacity);

		composite = CheckDivisors(
				prime,
				allowedMax,
				useCycles,
				accelerator,
				kernel,
				divisorsBuffer,
				hitsBuffer,
				divisors,
				hits,
				divisorData,
				out lastProcessed,
				out coveredRange,
				out processedCount
		);

		ArrayPool<byte>.Shared.Return(hits, clearArray: true);
		ArrayPool<ulong>.Shared.Return(divisors, clearArray: true);
		ArrayPool<MontgomeryDivisorData>.Shared.Return(divisorData, clearArray: true);
		hitsBuffer.Dispose();
		divisorsBuffer.Dispose();
		gpuLease.Dispose();

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
			Action<Index1D, ulong, ArrayView<MontgomeryDivisorData>, ArrayView<byte>> kernel,
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
		ulong cycle, currentDivisor, divisor = 3UL, next, nextDivisor;
		int batchSize, i;
		bool reachedEndInBatch;

		Span<MontgomeryDivisorData> divisorDataSpan;
		Span<ulong> divisorSpan;
		Span<byte> hitsSpan;
		MersenneDivisorCycles divisorCycles = MersenneDivisorCycles.Shared;
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

				// TODO: Let's move all the pre-checks to GPU.
				if (useCycles)
				{
					// TODO: Let's not use the divisor cache but instead calculate it on GPU. Let's move the entire check of cycle's divisibility to GPU. We should already be able to calculate in on GPU. There should be classes for that.
					cycle = divisorCycles!.GetCycle(currentDivisor);
					if (cycle == 0UL || prime % cycle != 0UL)
					{
						lastProcessed = currentDivisor;

						if (nextDivisor <= currentDivisor)
						{
							reachedEndInBatch = true;
							break;
						}

						if (nextDivisor > allowedMax)
						{
							divisor = nextDivisor;
							reachedEndInBatch = true;
							break;
						}

						divisor = nextDivisor;
						continue;
					}
				}

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
			kernel(batchSize, prime, divisorView, hitsView);
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
		ulong messages = total / interval;
		_lastStatusDivisor = total % interval;

		// TODO: Why do we output so many the same lines? Let's remove this printing to console.
		for (ulong i = 0; i < messages; i++)
		{
			Console.WriteLine($"...processed by-divisor candidate = {lastProcessed}");
		}
	}

	private static void CheckKernel(Index1D index, ulong prime, ArrayView<MontgomeryDivisorData> divisors, ArrayView<byte> hits)
	{
		MontgomeryDivisorData divisor = divisors[index];
		ulong modulus = divisor.Modulus;
		if (modulus <= 1UL || (modulus & 1UL) == 0UL)
		{
			hits[index] = 0;
			return;
		}

		hits[index] = Pow2Mod(prime, divisor) == 1UL ? (byte)1 : (byte)0;
	}

	public sealed class DivisorScanSession : IDisposable
	{
		private readonly MersenneNumberDivisorByDivisorGpuTester _owner;
		private readonly GpuContextPool.GpuContextLease _lease;
		private readonly Accelerator _accelerator;
		private readonly Action<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<byte>> _kernel;
		private readonly MemoryBuffer1D<ulong, Stride1D.Dense> _primesBuffer;
		private readonly MemoryBuffer1D<byte, Stride1D.Dense> _hitsBuffer;
		private readonly ulong[] _primesHost;
		private readonly byte[] _hitsHost;
		private readonly int[] _indexHost;
		private bool _disposed;

		internal DivisorScanSession(MersenneNumberDivisorByDivisorGpuTester owner)
		{
			_owner = owner;
			_lease = GpuContextPool.RentPreferred(preferCpu: false);
			_accelerator = _lease.Accelerator;
			_kernel = owner.GetKernelByPrime(_accelerator);
			_primesBuffer = _accelerator.Allocate1D<ulong>(owner._gpuBatchSize);
			_hitsBuffer = _accelerator.Allocate1D<byte>(owner._gpuBatchSize);
			_primesHost = ArrayPool<ulong>.Shared.Rent(owner._gpuBatchSize);
			_hitsHost = ArrayPool<byte>.Shared.Rent(owner._gpuBatchSize);
			_indexHost = ArrayPool<int>.Shared.Rent(owner._gpuBatchSize);
		}

		internal void Reset()
		{
			_disposed = false;
		}

		public void CheckDivisor(ulong divisor, ReadOnlySpan<ulong> primes, Span<byte> hits)
		{
			if (_disposed)
			{
				throw new ObjectDisposedException(nameof(DivisorScanSession));
			}

			int offset = 0;
			MersenneNumberDivisorByDivisorGpuTester owner = _owner;
			int gpuBatchSize = owner._gpuBatchSize;
			bool useCycles = owner._useDivisorCycles;
			byte[] hitsHost = _hitsHost;
			int[] indexHost = _indexHost;
			ulong[] primesHost = _primesHost;
			Accelerator accelerator = _accelerator;
			MemoryBuffer1D<byte, Stride1D.Dense> hitsBuffer = _hitsBuffer;
			MemoryBuffer1D<ulong, Stride1D.Dense> primesBuffer = _primesBuffer;
			ArrayView1D<ulong, Stride1D.Dense> primesBufferView = primesBuffer.View;
			ArrayView1D<byte, Stride1D.Dense> hitsBufferView = hitsBuffer.View;
			Action<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<byte>> kernel = _kernel;
			ArrayView1D<ulong, Stride1D.Dense> primesView;
			ArrayView1D<byte, Stride1D.Dense> hitsView;
			MersenneDivisorCycles? divisorCycles = MersenneDivisorCycles.Shared;
			ulong cycle = useCycles ? divisorCycles!.GetCycle(divisor) : 0UL, primeValue;
			MontgomeryDivisorData divisorData = CreateMontgomeryDivisorData(divisor);
			Span<ulong> primeSpan = [];
			ReadOnlySpan<ulong> primesSlice;
			Span<int> indexSpan = [];
			Span<byte> hitSpan = [];
			Span<byte> hitsSlice;
			int batchSize, currentSpanSize = -1, gpuCount, i, primesLength = primes.Length;

			while (offset < primesLength)
			{
				batchSize = Math.Min(gpuBatchSize, primesLength - offset);
				primesSlice = primes.Slice(offset, batchSize);
				hitsSlice = hits.Slice(offset, batchSize);

				if (useCycles)
				{
					if (batchSize != currentSpanSize)
					{
						primeSpan = primesHost.AsSpan(0, batchSize);
						indexSpan = indexHost.AsSpan(0, batchSize);
						hitSpan = hitsHost.AsSpan(0, batchSize);
						currentSpanSize = batchSize;
					}

					gpuCount = 0;

					for (i = 0; i < batchSize; i++)
					{
						primeValue = primesSlice[i];
						if (cycle == 0UL || primeValue % cycle != 0UL)
						{
							hitsSlice[i] = 0;
							continue;
						}

						indexSpan[gpuCount] = i;
						primeSpan[gpuCount] = primeValue;
						gpuCount++;
					}

					if (gpuCount > 0)
					{
						primesView = primesBufferView.SubView(0, gpuCount);
						hitsView = hitsBufferView.SubView(0, gpuCount);

						primesView.CopyFromCPU(ref MemoryMarshal.GetReference(primeSpan), gpuCount);
						kernel(gpuCount, divisorData, primesView, hitsView);
						accelerator.Synchronize();

						hitsView.CopyToCPU(ref MemoryMarshal.GetReference(hitSpan), gpuCount);
						for (i = 0; i < gpuCount; i++)
						{
							hitsSlice[indexSpan[i]] = hitSpan[i];
						}
					}
				}
				else
				{
					primesView = primesBufferView.SubView(0, batchSize);
					hitsView = hitsBufferView.SubView(0, batchSize);

					primesView.CopyFromCPU(ref MemoryMarshal.GetReference(primesSlice), batchSize);
					kernel(batchSize, divisorData, primesView, hitsView);
					accelerator.Synchronize();

					hitsView.CopyToCPU(ref MemoryMarshal.GetReference(hitsSlice), batchSize);
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

	// TODO: This method with all related should be moved to ULongExtensions and converted to extension method. We should already have similar methods. How this one is different?
	private static ulong Pow2Mod(ulong exponent, in MontgomeryDivisorData divisor)
	{
		// TODO: Do we use the dependency and the residue for the previous p in the calculations? Let's make use of the previous residue for the previous p when calculating the residue for the same divisor but different candidate. The candidates should be processed in increasing order.
		ulong modulus = divisor.Modulus;
		if (modulus <= 1UL || (modulus & 1UL) == 0UL)
		{
			return 0UL;
		}

		ulong result = divisor.MontgomeryOne;
		ulong baseVal = divisor.MontgomeryTwo;
		ulong nPrime = divisor.NPrime;
		while (exponent > 0UL)
		{
			if ((exponent & 1UL) != 0UL)
			{
				result = MontgomeryMultiply(result, baseVal, modulus, nPrime);
			}

			exponent >>= 1;
			if (exponent == 0UL)
			{
				break;
			}

			baseVal = MontgomeryMultiply(baseVal, baseVal, modulus, nPrime);
		}

		return MontgomeryMultiply(result, 1UL, modulus, nPrime);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static ulong MontgomeryMultiply(ulong a, ulong b, ulong modulus, ulong nPrime)
	{
		ulong tLow = unchecked(a * b);
		ulong m = unchecked(tLow * nPrime);

		// We're reusing tLow variable as the result to perform just a little bit better
		tLow = unchecked(MultiplyHigh(a, b) + MultiplyHigh(m, modulus) + (unchecked(tLow + m * modulus) < tLow ? 1UL : 0UL));
		if (tLow >= modulus)
		{
			tLow -= modulus;
		}

		return tLow;
	}

	private static void CheckKernelByPrime(Index1D index, MontgomeryDivisorData divisor, ArrayView<ulong> primes, ArrayView<byte> hits)
	{
		ulong modulus = divisor.Modulus;
		if (modulus <= 1UL || (modulus & 1UL) == 0UL)
		{
			hits[index] = 0;
			return;
		}

		hits[index] = Pow2Mod(primes[index], divisor) == 1UL ? (byte)1 : (byte)0;
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

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static ulong MultiplyHigh(ulong x, ulong y)
	{
		ulong xLow = (uint)x;
		ulong xHigh = x >> 32;
		ulong yLow = (uint)y;
		ulong yHigh = y >> 32;

		ulong mid = xHigh * yLow + ((xLow * yLow) >> 32);
		return xHigh * yHigh + (mid >> 32) + (((mid & 0xFFFFFFFFUL) + xLow * yHigh) >> 32);
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

