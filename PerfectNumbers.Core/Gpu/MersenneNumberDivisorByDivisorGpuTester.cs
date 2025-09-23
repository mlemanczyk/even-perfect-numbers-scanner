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

	public readonly struct MontgomeryDivisorData
	{
		public readonly ulong Modulus;
		public readonly ulong NPrime;
		public readonly ulong MontgomeryOne;
		public readonly ulong MontgomeryTwo;

		public MontgomeryDivisorData(ulong modulus, ulong nPrime, ulong montgomeryOne, ulong montgomeryTwo)
		{
			Modulus = modulus;
			NPrime = nPrime;
			MontgomeryOne = montgomeryOne;
			MontgomeryTwo = montgomeryTwo;
		}
	}

	private readonly ConcurrentDictionary<Accelerator, Action<Index1D, ulong, ArrayView<MontgomeryDivisorData>, ArrayView<byte>>> _kernelCache = new();
	private readonly ConcurrentDictionary<Accelerator, Action<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<byte>>> _kernelByPrimeCache = new();

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
		var divisorsBuffer = accelerator.Allocate1D<MontgomeryDivisorData>(batchCapacity);
		var hitsBuffer = accelerator.Allocate1D<byte>(batchCapacity);

		ulong[] divisors = ArrayPool<ulong>.Shared.Rent(batchCapacity);
		byte[] hits = ArrayPool<byte>.Shared.Rent(batchCapacity);
		MontgomeryDivisorData[] divisorData = ArrayPool<MontgomeryDivisorData>.Shared.Rent(batchCapacity);

		bool composite = false;
		bool processedAll = false;
		ulong divisor = 3UL;

		try
		{
			bool useCycles = _useDivisorCycles;
			MersenneDivisorCycles? divisorCycles = useCycles ? MersenneDivisorCycles.Shared : null;
			while (divisor <= allowedMax)
			{
				int batchSize = 0;
				bool reachedEndInBatch = false;

				while (batchSize < batchCapacity && divisor <= allowedMax)
				{
					ulong current = divisor;
					ulong next = current + 2UL;
					bool include = true;
					if (useCycles)
					{
						ulong cycle = divisorCycles!.GetCycle(current);
						if (cycle == 0UL || prime % cycle != 0UL)
						{
							include = false;
							lastProcessed = current;
							ReportStatus(lastProcessed);
						}
					}

					if (include)
					{
						divisors[batchSize++] = current;
					}

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

				Span<MontgomeryDivisorData> divisorDataSpan = divisorData.AsSpan(0, batchSize);
				var divisorView = divisorsBuffer.View.SubView(0, batchSize);
				var hitsView = hitsBuffer.View.SubView(0, batchSize);

				Span<ulong> divisorSpan = divisors.AsSpan(0, batchSize);
				Span<byte> hitsSpan = hits.AsSpan(0, batchSize);

				for (int i = 0; i < batchSize; i++)
				{
					divisorDataSpan[i] = CreateMontgomeryDivisorData(divisorSpan[i]);
				}

				divisorView.CopyFromCPU(ref MemoryMarshal.GetReference(divisorDataSpan), batchSize);
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
			ArrayPool<MontgomeryDivisorData>.Shared.Return(divisorData, clearArray: true);
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

	private static void CheckKernel(Index1D index, ulong prime, ArrayView<MontgomeryDivisorData> divisors, ArrayView<byte> hits)
	{
		MontgomeryDivisorData divisor = divisors[index];
		if (divisor.Modulus <= 1UL || (divisor.Modulus & 1UL) == 0UL)
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

		public void CheckDivisor(ulong divisor, ReadOnlySpan<ulong> primes, Span<byte> hits)
		{
			if (_disposed)
			{
				throw new ObjectDisposedException(nameof(DivisorScanSession));
			}

			int offset = 0;
			int consoleStatus = 0;
			bool useCycles = _owner._useDivisorCycles;
			MersenneDivisorCycles? divisorCycles = useCycles ? MersenneDivisorCycles.Shared : null;
			ulong cycle = useCycles ? divisorCycles!.GetCycle(divisor) : 0UL;
			MontgomeryDivisorData divisorData = CreateMontgomeryDivisorData(divisor);
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

				Span<int> indexSpan = _indexHost.AsSpan(0, batchSize);
				Span<byte> hitSpan = _hitsHost.AsSpan(0, batchSize);
				Span<byte> hitsSlice = hits.Slice(offset, batchSize);
				int gpuCount = 0;

				if (useCycles)
				{
					for (int i = 0; i < batchSize; i++)
					{
						ulong primeValue = primeSpan[i];
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
						var primesView = _primesBuffer.View.SubView(0, gpuCount);
						var hitsView = _hitsBuffer.View.SubView(0, gpuCount);

						primesView.CopyFromCPU(ref MemoryMarshal.GetReference(primeSpan.Slice(0, gpuCount)), gpuCount);
						_kernel(gpuCount, divisorData, primesView, hitsView);
						_accelerator.Synchronize();

						hitsView.CopyToCPU(ref MemoryMarshal.GetReference(hitSpan.Slice(0, gpuCount)), gpuCount);
						for (int i = 0; i < gpuCount; i++)
						{
							hitsSlice[indexSpan[i]] = hitSpan[i];
						}
					}
				}
				else
				{
					var primesView = _primesBuffer.View.SubView(0, batchSize);
					var hitsView = _hitsBuffer.View.SubView(0, batchSize);

					primesView.CopyFromCPU(ref MemoryMarshal.GetReference(primeSpan), batchSize);
					_kernel(batchSize, divisorData, primesView, hitsView);
					_accelerator.Synchronize();

					hitsView.CopyToCPU(ref MemoryMarshal.GetReference(hitSpan), batchSize);
					hitSpan.CopyTo(hitsSlice);
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
			ArrayPool<ulong>.Shared.Return(_primesHost, clearArray: true);
			ArrayPool<byte>.Shared.Return(_hitsHost, clearArray: true);
			ArrayPool<int>.Shared.Return(_indexHost, clearArray: true);
			_primesBuffer.Dispose();
			_hitsBuffer.Dispose();
			_lease.Dispose();
		}
	}

	private static ulong Pow2Mod(ulong exponent, MontgomeryDivisorData divisor)
	{
		ulong modulus = divisor.Modulus;
		if (modulus <= 1UL || (modulus & 1UL) == 0UL)
		{
			return 0UL;
		}

		ulong result = divisor.MontgomeryOne;
		ulong baseVal = divisor.MontgomeryTwo;
		ulong exp = exponent;
		while (exp > 0UL)
		{
			if ((exp & 1UL) != 0UL)
			{
				result = MontgomeryMultiply(result, baseVal, modulus, divisor.NPrime);
			}

			exp >>= 1;
			if (exp == 0UL)
			{
				break;
			}

			baseVal = MontgomeryMultiply(baseVal, baseVal, modulus, divisor.NPrime);
		}

		return MontgomeryMultiply(result, 1UL, modulus, divisor.NPrime);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static ulong MontgomeryMultiply(ulong a, ulong b, ulong modulus, ulong nPrime)
	{
		ulong tLow = unchecked(a * b);
		ulong m = unchecked(tLow * nPrime);
		ulong sumLow = unchecked(tLow + m * modulus);
		ulong carry = sumLow < tLow ? 1UL : 0UL;
		ulong sumHigh = unchecked(MultiplyHigh(a, b) + MultiplyHigh(m, modulus) + carry);

		ulong result = sumHigh;
		if (result >= modulus)
		{
			result -= modulus;
		}

		return result;
	}

	private static void CheckKernelByPrime(Index1D index, MontgomeryDivisorData divisor, ArrayView<ulong> primes, ArrayView<byte> hits)
	{
		ulong prime = primes[index];
		if (divisor.Modulus <= 1UL || (divisor.Modulus & 1UL) == 0UL)
		{
			hits[index] = 0;
			return;
		}

		ulong pow = Pow2Mod(prime, divisor);
		hits[index] = pow == 1UL ? (byte)1 : (byte)0;
	}

	private static MontgomeryDivisorData CreateMontgomeryDivisorData(ulong modulus)
	{
		if (modulus <= 1UL || (modulus & 1UL) == 0UL)
		{
			return new MontgomeryDivisorData(modulus, 0UL, 0UL, 0UL);
		}

		ulong nPrime = ComputeMontgomeryNPrime(modulus);
		ulong montgomeryOne = ComputeMontgomeryResidue(1UL, modulus);
		ulong montgomeryTwo = ComputeMontgomeryResidue(2UL, modulus);
		return new MontgomeryDivisorData(modulus, nPrime, montgomeryOne, montgomeryTwo);
	}

	private static ulong ComputeMontgomeryResidue(ulong value, ulong modulus)
	{
		UInt128 r = UInt128.One << 64;
		UInt128 result = ((UInt128)value * r) % modulus;
		return (ulong)result;
	}

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

		ulong lowProduct = xLow * yLow;
		ulong mid = xHigh * yLow + (lowProduct >> 32);
		ulong midLow = mid & 0xFFFFFFFFUL;
		ulong midHigh = mid >> 32;
		midLow += xLow * yHigh;

		ulong resultHigh = xHigh * yHigh + midHigh + (midLow >> 32);
		return resultHigh;
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

