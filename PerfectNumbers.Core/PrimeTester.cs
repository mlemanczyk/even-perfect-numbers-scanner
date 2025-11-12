using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu;
using PerfectNumbers.Core.Gpu.Accelerators;

namespace PerfectNumbers.Core;

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
	public static bool IsPrimeGpu(ulong n)
	{
		var gpu = Pow2MontgomeryAccelerator.Rent(1);
		var kernel = gpu.SmallPrimeSieveKernel!;

		// Span<byte> flag = stackalloc byte[1];

		var inputView = gpu.PrimeTestInput.View;

		GpuPrimeWorkLimiter.Acquire();
		AcceleratorStream stream = AcceleratorStreamPool.Rent(gpu.Accelerator);
		// var stream = gpu.Stream!;
		inputView.CopyFromCPU(stream, ref n, 1);

		var outputView = gpu.OutputByte.View;

		kernel.Launch(
						stream,
						1,
						inputView,
						gpu.DevicePrimesLastOne.View,
						gpu.DevicePrimesLastSeven.View,
						gpu.DevicePrimesLastThree.View,
						gpu.DevicePrimesLastNine.View,
						gpu.DevicePrimesPow2LastOne.View,
						gpu.DevicePrimesPow2LastSeven.View,
						gpu.DevicePrimesPow2LastThree.View,
						gpu.DevicePrimesPow2LastNine.View,
						outputView);

		byte flag = 0;
		outputView.CopyToCPU(stream, ref flag, 1);
		stream.Synchronize();

		AcceleratorStreamPool.Return(stream);
		// gpu.Dispose();
		Pow2MontgomeryAccelerator.Return(gpu);
		GpuPrimeWorkLimiter.Release();

		return flag != 0;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsPrimeGpu(Pow2MontgomeryAccelerator gpu, ulong n)
	{
		// var stream = gpu.Stream!;
		var input = gpu.PrimeTestInput;
		var inputView = input.View;

		AcceleratorStream stream = AcceleratorStreamPool.Rent(gpu.Accelerator);
		inputView.CopyFromCPU(stream, ref n, 1);

		// Span<byte> flag = stackalloc byte[1];
		var output = gpu.OutputByte;
		var outputView = output.View;

		Kernel kernel = gpu.SmallPrimeSieveKernel!;
		var kernelLauncher = kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>>>();
		
		kernelLauncher(
						stream,
						1,
						inputView,
						gpu.DevicePrimesLastOne.View,
						gpu.DevicePrimesLastSeven.View,
						gpu.DevicePrimesLastThree.View,
						gpu.DevicePrimesLastNine.View,
						gpu.DevicePrimesPow2LastOne.View,
						gpu.DevicePrimesPow2LastSeven.View,
						gpu.DevicePrimesPow2LastThree.View,
						gpu.DevicePrimesPow2LastNine.View,
						outputView);

		byte flag = 0;
		outputView.CopyToCPU(stream, ref flag, 1);
		stream.Synchronize();
		AcceleratorStreamPool.Return(stream);

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

			Pow2MontgomeryAccelerator.WarmUp();
			WarmedGpuLeaseCount = target;
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static void IsPrimeBatchGpu(ReadOnlySpan<ulong> values, Span<byte> results)
	{
		GpuPrimeWorkLimiter.Acquire();
		var gpu = Pow2MontgomeryAccelerator.Rent(GpuBatchSize);
		var accelerator = gpu.Accelerator;
		// var stream = gpu.Stream!;
		var kernel = gpu.SmallPrimeSieveKernel!;
		int totalLength = values.Length;
		int batchSize = GpuBatchSize;

		var input = gpu.PrimeTestInput;
		var output = gpu.OutputByte;
		var inputView = input.View;
		var outputView = output.View;

		int pos = 0;
		AcceleratorStream stream = AcceleratorStreamPool.Rent(accelerator);
		while (pos < totalLength)
		{
			int remaining = totalLength - pos;
			int count = remaining > batchSize ? batchSize : remaining;

			var valueSlice = values.Slice(pos, count);
			inputView.CopyFromCPU(stream, valueSlice);

			kernel.Launch(
					stream,
					count,
					inputView,
					gpu.DevicePrimesLastOne.View,
					gpu.DevicePrimesLastSeven.View,
					gpu.DevicePrimesLastThree.View,
					gpu.DevicePrimesLastNine.View,
					gpu.DevicePrimesPow2LastOne.View,
					gpu.DevicePrimesPow2LastSeven.View,
					gpu.DevicePrimesPow2LastThree.View,
					gpu.DevicePrimesPow2LastNine.View,
					outputView);

			var resultSlice = results.Slice(pos, count);
			ref byte resultRef = ref MemoryMarshal.GetReference(resultSlice);
			outputView.CopyToCPU(stream, ref resultRef, count);

			pos += count;
		}

		stream.Synchronize();
		AcceleratorStreamPool.Return(stream);
		Pow2MontgomeryAccelerator.Return(gpu);
		// gpu.Dispose();
		GpuPrimeWorkLimiter.Release();
	}

	// Expose cache clearing for accelerator disposal coordination
	public static void ClearGpuCaches()
	{
		Pow2MontgomeryAccelerator.Clear();
	}

	internal static void DisposeGpuContexts()
	{
		Pow2MontgomeryAccelerator.DisposeAll();
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
		return BinaryGcd(n, m) != 1UL;
	}

	[ThreadStatic]
	private static readonly Dictionary<Accelerator, Action<AcceleratorStream, Index1D, ArrayView<ulong>, ArrayView<byte>>>? _kernel;

	internal static void SharesFactorWithMaxExponentBatch(ReadOnlySpan<ulong> values, Span<byte> results)
	{
		// TODO: Route this batch helper through the shared GPU kernel pool from
		// GpuUInt128BinaryGcdBenchmarks so we reuse cached kernels, pinned host buffers,
		// and divisor-cycle staging instead of allocating new device buffers per call.
		var gpu = Pow2MontgomeryAccelerator.Rent(1);
		var accelerator = gpu.Accelerator;
		// var stream = gpu.Stream!;

		int length = values.Length;
		ArrayPool<ulong> pool = ThreadStaticPools.UlongPool;
		ulong[] temp = pool.Rent(length);
		values.CopyTo(temp);

		MemoryBuffer1D<ulong, Stride1D.Dense>? inputBuffer;
		MemoryBuffer1D<byte, Stride1D.Dense>? resultBuffer;

		// lock (accelerator)
		{
			inputBuffer = accelerator.Allocate1D<ulong>(length);
			resultBuffer = accelerator.Allocate1D<byte>(length);
		}

		AcceleratorStream stream = AcceleratorStreamPool.Rent(accelerator);
		inputBuffer.View.CopyFromCPU(stream, ref temp[0], length);
		var kernel = GetSharesFactorKernel(accelerator);
		kernel(stream, length, inputBuffer.View, resultBuffer.View);
		resultBuffer.View.CopyToCPU(stream, in results);
		stream.Synchronize();

		AcceleratorStreamPool.Return(stream);
		pool.Return(temp, clearArray: false);
		resultBuffer.Dispose();
		inputBuffer.Dispose();
		Pow2MontgomeryAccelerator.Return(gpu);
		// gpu.Dispose();
	}

	private static Action<AcceleratorStream, Index1D, ArrayView<ulong>, ArrayView<byte>> GetSharesFactorKernel(Accelerator accelerator)
	{
		var pool = _kernel ?? [];
		if (pool.TryGetValue(accelerator, out var cached))
		{
			return cached;
		}

		cached = KernelUtil.GetKernel(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ArrayView<byte>>(PrimeTesterKernels.SharesFactorKernel)).CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView<ulong>, ArrayView<byte>>>();
		pool[accelerator] = cached;
		return cached;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static ulong BinaryGcd(ulong u, ulong v)
	{
		// TODO: Swap this handwritten binary GCD for the optimized helper measured in
		// GpuUInt128BinaryGcdBenchmarks so CPU callers share the faster subtract-less
		// ladder once the common implementation is promoted into PerfectNumbers.Core.
		if (u == 0UL)
		{
			return v;
		}

		if (v == 0UL)
		{
			return u;
		}

		int shift = BitOperations.TrailingZeroCount(u | v);
		u >>= BitOperations.TrailingZeroCount(u);

		do
		{
			v >>= BitOperations.TrailingZeroCount(v);
			if (u > v)
			{
				(u, v) = (v, u);
			}

			v -= u;
		}
		while (v != 0UL);

		return u << shift;
	}
}
