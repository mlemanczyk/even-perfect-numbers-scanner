using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

public sealed class MersenneNumberDivisorByDivisorGpuTester
{
	private int _gpuBatchSize = GpuConstants.ScanBatchSize;
	private readonly object _sync = new();
	private ulong _divisorLimit;
	private bool _isConfigured;
	private ulong _lastStatusDivisor;

	private readonly ConcurrentDictionary<Accelerator, Action<Index1D, ulong, ArrayView<ulong>, ArrayView<byte>>> _kernelCache = new();
	private readonly ConcurrentDictionary<Accelerator, Action<Index1D, ulong, ArrayView<ulong>, ArrayView<byte>>> _kernelByPrimeCache = new();

	private Action<Index1D, ulong, ArrayView<ulong>, ArrayView<byte>> GetKernel(Accelerator accelerator) =>
					_kernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ulong, ArrayView<ulong>, ArrayView<byte>>(CheckKernel));
	private Action<Index1D, ulong, ArrayView<ulong>, ArrayView<byte>> GetKernelByPrime(Accelerator accelerator) =>
					_kernelByPrimeCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ulong, ArrayView<ulong>, ArrayView<byte>>(CheckKernelByPrime));

	public int GpuBatchSize
	{
		get => _gpuBatchSize;
		set => _gpuBatchSize = Math.Max(1, value);
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
		lock (_sync)
		{
			if (!_isConfigured)
			{
				throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
			}

			ulong allowedMax = ComputeAllowedMaxDivisor(prime);
			if (allowedMax < 3UL)
			{
				divisorsExhausted = true;
				return true;
			}

			bool composite = CheckDivisors(prime, allowedMax, out ulong lastProcessed, out bool coveredRange);

			if (lastProcessed != 0UL)
			{
				ReportStatus(lastProcessed);
			}

			if (composite)
			{
				divisorsExhausted = true;
				return false;
			}

			divisorsExhausted = coveredRange;
			return true;
		}
	}

	private bool CheckDivisors(ulong prime, ulong allowedMax, out ulong lastProcessed, out bool coveredRange)
	{
		allowedMax = Math.Min(allowedMax, _divisorLimit);
		lastProcessed = 0UL;

		if (allowedMax < 3UL)
		{
			coveredRange = true;
			return false;
		}

		var gpuLease = GpuContextPool.RentPreferred(preferCpu: false);
		var accelerator = gpuLease.Accelerator;
		var kernel = GetKernel(accelerator);

		int batchCapacity = _gpuBatchSize;
		var divisorsBuffer = accelerator.Allocate1D<ulong>(batchCapacity);
		var hitsBuffer = accelerator.Allocate1D<byte>(batchCapacity);

		ulong[] divisors = ArrayPool<ulong>.Shared.Rent(batchCapacity);
		byte[] hits = ArrayPool<byte>.Shared.Rent(batchCapacity);

		bool composite = false;
		bool processedAll = false;
		ulong divisor = 3UL;

		try
		{
			while (divisor <= allowedMax)
			{
				int batchSize = 0;
				bool reachedEndInBatch = false;

				while (batchSize < batchCapacity && divisor <= allowedMax)
				{
					ulong current = divisor;
					divisors[batchSize++] = current;

					ulong next = current + 2UL;
					ReportStatus(lastProcessed);
					if (next <= current)
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
					break;
				}

				var divisorView = divisorsBuffer.View.SubView(0, batchSize);
				var hitsView = hitsBuffer.View.SubView(0, batchSize);

				Span<ulong> divisorSpan = divisors.AsSpan(0, batchSize);
				Span<byte> hitsSpan = hits.AsSpan(0, batchSize);

				divisorView.CopyFromCPU(ref MemoryMarshal.GetReference(divisorSpan), batchSize);
				kernel(batchSize, prime, divisorView, hitsView);
				accelerator.Synchronize();

				hitsView.CopyToCPU(ref MemoryMarshal.GetReference(hitsSpan), batchSize);

				for (int i = 0; i < batchSize; i++)
				{
					lastProcessed = divisorSpan[i];
					if (hitsSpan[i] != 0)
					{
						composite = true;
						break;
					}
				}

				ReportStatus(lastProcessed);
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
		}
		finally
		{
			ArrayPool<byte>.Shared.Return(hits, clearArray: true);
			ArrayPool<ulong>.Shared.Return(divisors, clearArray: true);
			hitsBuffer.Dispose();
			divisorsBuffer.Dispose();
			gpuLease.Dispose();
		}

		return composite;
	}

	private void ReportStatus(ulong divisor)
	{
		if (++_lastStatusDivisor == PerfectNumberConstants.ConsoleInterval)
		{
			Console.WriteLine($"...processed by-divisor candidate = {divisor}");
			_lastStatusDivisor = 0;
		}
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

	private ulong ComputeAllowedMaxDivisor(ulong prime)
	{
		if (prime <= 1UL)
		{
			return 0UL;
		}

		if (prime - 1UL >= 64UL)
		{
			return _divisorLimit;
		}

		ulong maxByPrime = (1UL << (int)(prime - 1UL)) - 1UL;
		return Math.Min(maxByPrime, _divisorLimit);
	}

	private static void CheckKernel(Index1D index, ulong prime, ArrayView<ulong> divisors, ArrayView<byte> hits)
	{
		ulong divisor = divisors[index];
		if (divisor <= 1UL || (divisor & 1UL) == 0UL)
		{
			hits[index] = 0;
			return;
		}

		ulong pow = Pow2Mod(prime, divisor);
		hits[index] = pow == 1UL ? (byte)1 : (byte)0;
	}

	public sealed class DivisorScanSession : IDisposable
	{
		private readonly MersenneNumberDivisorByDivisorGpuTester _owner;
		private readonly GpuContextPool.GpuContextLease _lease;
		private readonly Accelerator _accelerator;
		private readonly Action<Index1D, ulong, ArrayView<ulong>, ArrayView<byte>> _kernel;
		private readonly MemoryBuffer1D<ulong, Stride1D.Dense> _primesBuffer;
		private readonly MemoryBuffer1D<byte, Stride1D.Dense> _hitsBuffer;
		private readonly ulong[] _primesHost;
		private readonly byte[] _hitsHost;
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
		}

		public void CheckDivisor(ulong divisor, ReadOnlySpan<ulong> primes, Span<byte> hits)
		{
			if (_disposed)
			{
				throw new ObjectDisposedException(nameof(DivisorScanSession));
			}

			if (primes.Length == 0)
			{
				return;
			}

			int offset = 0;
			int consoleStatus = 0;
			while (offset < primes.Length)
			{
				int batchSize = Math.Min(_owner._gpuBatchSize, primes.Length - offset);
				Span<ulong> primeSpan = _primesHost.AsSpan(0, batchSize);
				primes.Slice(offset, batchSize).CopyTo(primeSpan);

				if (++consoleStatus == PerfectNumberConstants.ConsoleInterval)
				{
					Console.WriteLine($"...processed by-divisor candidate = {primes[0]}");
					consoleStatus = 0;
				}

				var primesView = _primesBuffer.View.SubView(0, batchSize);
				var hitsView = _hitsBuffer.View.SubView(0, batchSize);

				primesView.CopyFromCPU(ref MemoryMarshal.GetReference(primeSpan), batchSize);
				_kernel(batchSize, divisor, primesView, hitsView);
				_accelerator.Synchronize();

				Span<byte> hitSpan = _hitsHost.AsSpan(0, batchSize);
				hitsView.CopyToCPU(ref MemoryMarshal.GetReference(hitSpan), batchSize);
				hitSpan.CopyTo(hits.Slice(offset, batchSize));
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
			ArrayPool<ulong>.Shared.Return(_primesHost, clearArray: true);
			ArrayPool<byte>.Shared.Return(_hitsHost, clearArray: true);
			_primesBuffer.Dispose();
			_hitsBuffer.Dispose();
			_lease.Dispose();
		}
	}

	private static ulong Pow2Mod(ulong exponent, ulong modulus)
	{
		if (modulus <= 1UL)
		{
			return 0UL;
		}

		ulong result = 1UL % modulus;
		ulong baseVal = 2UL % modulus;
		ulong exp = exponent;
		while (exp > 0UL)
		{
			if ((exp & 1UL) != 0UL)
			{
				result = MulMod(result, baseVal, modulus);
			}

			exp >>= 1;
			if (exp == 0UL)
			{
				break;
			}

			baseVal = MulMod(baseVal, baseVal, modulus);
		}

		return result;
	}

	private static ulong MulMod(ulong a, ulong b, ulong modulus)
	{
		if (modulus == 0UL)
		{
			return 0UL;
		}

		ulong result = 0UL;
		ulong x = a % modulus;
		ulong y = b;
		while (y > 0UL)
		{
			if ((y & 1UL) != 0UL)
			{
				result += x;
				if (result >= modulus)
				{
					result -= modulus;
				}
			}

			x <<= 1;
			if (x >= modulus)
			{
				x -= modulus;
			}

			y >>= 1;
		}

		return result;
	}

	private static void CheckKernelByPrime(Index1D index, ulong divisor, ArrayView<ulong> primes, ArrayView<byte> hits)
	{
		ulong prime = primes[index];
		if (divisor <= 1UL || (divisor & 1UL) == 0UL)
		{
			hits[index] = 0;
			return;
		}

		ulong pow = Pow2Mod(prime, divisor);
		hits[index] = pow == 1UL ? (byte)1 : (byte)0;
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

			return ComputeAllowedMaxDivisor(prime);
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

			return new DivisorScanSession(this);
		}
	}
}

