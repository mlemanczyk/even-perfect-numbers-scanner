using System.Numerics;
using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu.Accelerators;

namespace PerfectNumbers.Core;

public sealed class PrimeTesterByLastDigit
{
	public PrimeTesterByLastDigit()
	{
	}

	[ThreadStatic]
	private static PrimeTesterByLastDigit? _tester;

	public static PrimeTesterByLastDigit Exclusive => _tester ??= new();

	public static bool IsPrimeCpu(ulong n)
	{
		if (n <= 1UL)
		{
			return false;
		}

		if (n == 2UL)
		{
			throw new InvalidOperationException("PrimeTester.IsPrime encountered the sentinel input 2.");
		}

		bool isOdd = (n & 1UL) != 0UL;
		bool result = isOdd;

		bool requiresTrialDivision = result && n >= 7UL;

		if (requiresTrialDivision)
		{
			// EvenPerfectBitScanner streams exponents starting at 136,279,841, so the Mod10/GCD guard never fires on the
			// production path. Leave the logic commented out as instrumentation for diagnostic builds.
			// bool sharesMaxExponentFactor = n.Mod10() == 1UL && SharesFactorWithMaxExponent(n);
			// result &= !sharesMaxExponentFactor;

			if (result)
			{
				uint[] smallPrimeDivisors;
				ulong[] smallPrimeDivisorsMul;

				ulong nMod10 = n.Mod10();
				switch (nMod10)
				{
					case 1UL:
						smallPrimeDivisors = PrimesGenerator.SmallPrimesLastOne;
						smallPrimeDivisorsMul = PrimesGenerator.SmallPrimesPow2LastOne;
						break;
					case 3UL:
						smallPrimeDivisors = DivisorGenerator.SmallPrimesLastThree;
						smallPrimeDivisorsMul = DivisorGenerator.SmallPrimesPow2LastThree;
						break;
					case 7UL:
						smallPrimeDivisors = PrimesGenerator.SmallPrimesLastSeven;
						smallPrimeDivisorsMul = PrimesGenerator.SmallPrimesPow2LastSeven;
						break;
					default:
						smallPrimeDivisors = DivisorGenerator.SmallPrimesLastNine;
						smallPrimeDivisorsMul = DivisorGenerator.SmallPrimesPow2LastNine;
						break;
				}

				int smallPrimeDivisorsLength = smallPrimeDivisors.Length;
				for (int i = 0; i < smallPrimeDivisorsLength; i++)
				{
					ulong value = smallPrimeDivisorsMul[i];
					if (value > n)
					{
						break;
					}

					value = n % smallPrimeDivisors[i];
					if (value == 0)
					{
						result = false;
						break;
					}
				}
			}
		}

		return result;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsPrimeGpu(PrimeOrderCalculatorAccelerator gpu, ulong n)
	{
		byte flag = 0;
		var inputView = gpu.InputView;
		var outputView = gpu.OutputByteView;

		int acceleratorIndex = gpu.AcceleratorIndex;
		AcceleratorStream stream = AcceleratorStreamPool.Rent(acceleratorIndex);
		inputView.CopyFromCPU(stream, ref n, 1);

		var kernelLauncher = gpu.SmallPrimeSieveKernelLauncher;
		
		kernelLauncher(
						stream,
						1,
						inputView,
						gpu.DevicePrimesLastOne,
						gpu.DevicePrimesLastSeven,
						gpu.DevicePrimesLastThree,
						gpu.DevicePrimesLastNine,
						gpu.DevicePrimesPow2LastOne,
						gpu.DevicePrimesPow2LastSeven,
						gpu.DevicePrimesPow2LastThree,
						gpu.DevicePrimesPow2LastNine,
						outputView);

		outputView.CopyToCPU(stream, ref flag, 1);
		stream.Synchronize();
		AcceleratorStreamPool.Return(acceleratorIndex, stream);

		return flag != 0;
	}

	public static int GpuBatchSize { get; set; } = 262_144;

	private static readonly object GpuWarmUpLock = new();
	private static int WarmedGpuLeaseCount;

	public static void WarmUpGpuKernels(int threadCount)
	{
		int target = threadCount >> 2;
		if (target == 0)
		{
			target = threadCount;
		}

		lock (GpuWarmUpLock)
		{
			if (target <= WarmedGpuLeaseCount)
			{
				return;
			}

			PrimeOrderCalculatorAccelerator.WarmUp();
			WarmedGpuLeaseCount = target;
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static void IsPrimeBatchGpu(PrimeOrderCalculatorAccelerator gpu, ReadOnlySpan<ulong> values, Span<byte> results)
	{
		// GpuPrimeWorkLimiter.Acquire();
		int acceleratorIndex = gpu.AcceleratorIndex;
		int totalLength = values.Length;
		int batchSize = GpuBatchSize;

		var inputView = gpu.InputView;
		var outputView = gpu.OutputByteView;

		int pos = 0;
		AcceleratorStream stream = AcceleratorStreamPool.Rent(acceleratorIndex);
		var kernelLauncher = gpu.SmallPrimeSieveKernelLauncher;

		while (pos < totalLength)
		{
			int remaining = totalLength - pos;
			int count = remaining > batchSize ? batchSize : remaining;

			var valueSlice = values.Slice(pos, count);
			inputView.CopyFromCPU(stream, valueSlice);

			kernelLauncher(
					stream,
					count,
					inputView,
					gpu.DevicePrimesLastOne,
					gpu.DevicePrimesLastSeven,
					gpu.DevicePrimesLastThree,
					gpu.DevicePrimesLastNine,
					gpu.DevicePrimesPow2LastOne,
					gpu.DevicePrimesPow2LastSeven,
					gpu.DevicePrimesPow2LastThree,
					gpu.DevicePrimesPow2LastNine,
					outputView);

			var resultSlice = results.Slice(pos, count);
			outputView.CopyToCPU(stream, resultSlice);

			pos += count;
		}

		stream.Synchronize();
		AcceleratorStreamPool.Return(acceleratorIndex, stream);
		// GpuPrimeWorkLimiter.Release();
	}

	// Expose cache clearing for accelerator disposal coordination
	public static void ClearGpuCaches()
	{
		PrimeOrderCalculatorAccelerator.Clear();
	}

	internal static void DisposeGpuContexts()
	{
		PrimeOrderCalculatorAccelerator.DisposeAll();
		lock (GpuWarmUpLock)
		{
			WarmedGpuLeaseCount = 0;
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	internal static bool SharesFactorWithMaxExponent(ulong n)
	{
		// TODO: Replace this on-the-fly GCD probe with the cached factor table derived from
		// ResidueComputationBenchmarks so divisor-cycle metadata can short-circuit the test
		// instead of recomputing binary GCD for every candidate.
		ulong m = (ulong)BitOperations.Log2(n);
		return n.BinaryGcd(m) != 1UL;
	}

	internal static void SharesFactorWithMaxExponentBatch(PrimeOrderCalculatorAccelerator gpu, ReadOnlySpan<ulong> values, Span<byte> results)
	{
		// TODO: Route this batch helper through the shared GPU kernel pool from
		// GpuUInt128BinaryGcdBenchmarks so we reuse cached kernels, pinned host buffers,
		// and divisor-cycle staging instead of allocating new device buffers per call.
		// Check in benchmarks, which implementation was the fastest, is compatible with GPU,
		// and implement it.

		int length = values.Length;

		int acceleratorIndex = gpu.AcceleratorIndex;
		gpu.EnsureCapacity(0, length);
		var inputBufferView = gpu.InputView;
		var resultBufferView = gpu.OutputByteView;
		var kernelLauncher = gpu.SharesFactorKernelLauncher;

		AcceleratorStream stream = AcceleratorStreamPool.Rent(acceleratorIndex);
		inputBufferView.CopyFromCPU(stream, values);

		kernelLauncher(stream, length, inputBufferView, resultBufferView);

		resultBufferView.CopyToCPU(stream, in results);
		stream.Synchronize();

		AcceleratorStreamPool.Return(acceleratorIndex, stream);
	}
}

public sealed class PrimeTester
{
	public PrimeTester()
	{
	}

	[ThreadStatic]
	private static PrimeTester? _tester;

	public static PrimeTester Exclusive => _tester ??= new();

	public static bool IsPrime(ulong n)
	{
		if (n <= 1UL)
		{
			return false;
		}

		if (n == 2UL)
		{
			throw new InvalidOperationException("PrimeTester.IsPrime encountered the sentinel input 2.");
		}

		bool isOdd = (n & 1UL) != 0UL;
		bool result = isOdd;

		bool requiresTrialDivision = result && n >= 7UL;

		if (requiresTrialDivision)
		{
			// EvenPerfectBitScanner streams exponents starting at 136,279,841, so the Mod10/GCD guard never fires on the
			// production path. Leave the logic commented out as instrumentation for diagnostic builds.
			// bool sharesMaxExponentFactor = n.Mod10() == 1UL && SharesFactorWithMaxExponent(n);
			// result &= !sharesMaxExponentFactor;

			if (result)
			{
				uint[] smallPrimeDivisors = PrimesGenerator.SmallPrimes;
				ulong[] smallPrimeDivisorsMul = PrimesGenerator.SmallPrimesPow2;
				int smallPrimeDivisorsLength = smallPrimeDivisors.Length;
				for (int i = 0; i < smallPrimeDivisorsLength; i++)
				{
					if (smallPrimeDivisorsMul[i] > n)
					{
						break;
					}

					if (n % smallPrimeDivisors[i] == 0)
					{
						result = false;
						break;
					}
				}
			}
		}

		return result;
	}

	public static bool IsPrimeLastDigit(ulong n)
	{
		if (n <= 1UL)
		{
			return false;
		}

		if (n == 2UL)
		{
			throw new InvalidOperationException("PrimeTester.IsPrime encountered the sentinel input 2.");
		}

		bool isOdd = (n & 1UL) != 0UL;
		bool result = isOdd;

		bool requiresTrialDivision = result && n >= 7UL;

		if (requiresTrialDivision)
		{
			// EvenPerfectBitScanner streams exponents starting at 136,279,841, so the Mod10/GCD guard never fires on the
			// production path. Leave the logic commented out as instrumentation for diagnostic builds.
			// bool sharesMaxExponentFactor = n.Mod10() == 1UL && SharesFactorWithMaxExponent(n);
			// result &= !sharesMaxExponentFactor;

			if (result)
			{
				uint[] smallPrimeDivisors = PrimesGenerator.SmallPrimes;
				ulong[] smallPrimeDivisorsMul = PrimesGenerator.SmallPrimesPow2;

				ulong nMod10 = n.Mod10();
				switch (nMod10)
				{
					case 1UL:
						smallPrimeDivisors = PrimesGenerator.SmallPrimesLastOne;
						smallPrimeDivisorsMul = PrimesGenerator.SmallPrimesPow2LastOne;
						break;
					case 3UL:
						smallPrimeDivisors = DivisorGenerator.SmallPrimesLastThree;
						smallPrimeDivisorsMul = DivisorGenerator.SmallPrimesPow2LastThree;
						break;
					case 7UL:
						smallPrimeDivisors = PrimesGenerator.SmallPrimesLastSeven;
						smallPrimeDivisorsMul = PrimesGenerator.SmallPrimesPow2LastSeven;
						break;
					case 9UL:
						smallPrimeDivisors = DivisorGenerator.SmallPrimesLastNine;
						smallPrimeDivisorsMul = DivisorGenerator.SmallPrimesPow2LastNine;
						break;
				}

				int smallPrimeDivisorsLength = smallPrimeDivisors.Length;
				for (int i = 0; i < smallPrimeDivisorsLength; i++)
				{
					if (smallPrimeDivisorsMul[i] > n)
					{
						break;
					}

					if (n % smallPrimeDivisors[i] == 0)
					{
						result = false;
						break;
					}
				}
			}
		}

		return result;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsPrimeGpu(PrimeOrderCalculatorAccelerator gpu, ulong n)
	{
		byte flag = 0;
		var inputView = gpu.InputView;
		var outputView = gpu.OutputByteView;

		int acceleratorIndex = gpu.AcceleratorIndex;
		AcceleratorStream stream = AcceleratorStreamPool.Rent(acceleratorIndex);
		inputView.CopyFromCPU(stream, ref n, 1);

		var kernelLauncher = gpu.SmallPrimeSieveKernelLauncher;
		
		kernelLauncher(
						stream,
						1,
						inputView,
						gpu.DevicePrimesLastOne,
						gpu.DevicePrimesLastSeven,
						gpu.DevicePrimesLastThree,
						gpu.DevicePrimesLastNine,
						gpu.DevicePrimesPow2LastOne,
						gpu.DevicePrimesPow2LastSeven,
						gpu.DevicePrimesPow2LastThree,
						gpu.DevicePrimesPow2LastNine,
						outputView);

		outputView.CopyToCPU(stream, ref flag, 1);
		stream.Synchronize();
		AcceleratorStreamPool.Return(acceleratorIndex, stream);

		return flag != 0;
	}

	public static int GpuBatchSize { get; set; } = 262_144;

	private static readonly object GpuWarmUpLock = new();
	private static int WarmedGpuLeaseCount;

	public static void WarmUpGpuKernels(int threadCount)
	{
		int target = threadCount >> 2;
		if (target == 0)
		{
			target = threadCount;
		}

		lock (GpuWarmUpLock)
		{
			if (target <= WarmedGpuLeaseCount)
			{
				return;
			}

			PrimeOrderCalculatorAccelerator.WarmUp();
			WarmedGpuLeaseCount = target;
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static void IsPrimeBatchGpu(PrimeOrderCalculatorAccelerator gpu, ReadOnlySpan<ulong> values, Span<byte> results)
	{
		// GpuPrimeWorkLimiter.Acquire();
		int acceleratorIndex = gpu.AcceleratorIndex;
		int totalLength = values.Length;
		int batchSize = GpuBatchSize;

		var inputView = gpu.InputView;
		var outputView = gpu.OutputByteView;

		int pos = 0;
		AcceleratorStream stream = AcceleratorStreamPool.Rent(acceleratorIndex);
		var kernelLauncher = gpu.SmallPrimeSieveKernelLauncher;

		while (pos < totalLength)
		{
			int remaining = totalLength - pos;
			int count = remaining > batchSize ? batchSize : remaining;

			var valueSlice = values.Slice(pos, count);
			inputView.CopyFromCPU(stream, valueSlice);

			kernelLauncher(
					stream,
					count,
					inputView,
					gpu.DevicePrimesLastOne,
					gpu.DevicePrimesLastSeven,
					gpu.DevicePrimesLastThree,
					gpu.DevicePrimesLastNine,
					gpu.DevicePrimesPow2LastOne,
					gpu.DevicePrimesPow2LastSeven,
					gpu.DevicePrimesPow2LastThree,
					gpu.DevicePrimesPow2LastNine,
					outputView);

			var resultSlice = results.Slice(pos, count);
			outputView.CopyToCPU(stream, resultSlice);

			pos += count;
		}

		stream.Synchronize();
		AcceleratorStreamPool.Return(acceleratorIndex, stream);
		// GpuPrimeWorkLimiter.Release();
	}

	// Expose cache clearing for accelerator disposal coordination
	public static void ClearGpuCaches()
	{
		PrimeOrderCalculatorAccelerator.Clear();
	}

	internal static void DisposeGpuContexts()
	{
		PrimeOrderCalculatorAccelerator.DisposeAll();
		lock (GpuWarmUpLock)
		{
			WarmedGpuLeaseCount = 0;
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	internal static bool SharesFactorWithMaxExponent(ulong n)
	{
		// TODO: Replace this on-the-fly GCD probe with the cached factor table derived from
		// ResidueComputationBenchmarks so divisor-cycle metadata can short-circuit the test
		// instead of recomputing binary GCD for every candidate.
		ulong m = (ulong)BitOperations.Log2(n);
		return n.BinaryGcd(m) != 1UL;
	}

	internal static void SharesFactorWithMaxExponentBatch(PrimeOrderCalculatorAccelerator gpu, ReadOnlySpan<ulong> values, Span<byte> results)
	{
		// TODO: Route this batch helper through the shared GPU kernel pool from
		// GpuUInt128BinaryGcdBenchmarks so we reuse cached kernels, pinned host buffers,
		// and divisor-cycle staging instead of allocating new device buffers per call.
		// Check in benchmarks, which implementation was the fastest, is compatible with GPU,
		// and implement it.

		int length = values.Length;

		int acceleratorIndex = gpu.AcceleratorIndex;
		gpu.EnsureCapacity(0, length);
		var inputBufferView = gpu.InputView;
		var resultBufferView = gpu.OutputByteView;
		var kernelLauncher = gpu.SharesFactorKernelLauncher;

		AcceleratorStream stream = AcceleratorStreamPool.Rent(acceleratorIndex);
		inputBufferView.CopyFromCPU(stream, values);

		kernelLauncher(stream, length, inputBufferView, resultBufferView);

		resultBufferView.CopyToCPU(stream, in results);
		stream.Synchronize();

		AcceleratorStreamPool.Return(acceleratorIndex, stream);
	}
}
