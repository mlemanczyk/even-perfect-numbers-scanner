using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

public sealed class MersenneNumberDivisorGpuTester
{
	private readonly ConcurrentDictionary<Accelerator, Action<Index1D, ulong, ulong, ArrayView<byte>>> _kernelCache = new();

	private Action<Index1D, ulong, ulong, ArrayView<byte>> GetKernel(Accelerator accelerator) =>
		_kernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ulong, ulong, ArrayView<byte>>(Kernel));

	public bool IsDivisible(ulong exponent, ulong divisor)
	{
		var gpu = GpuContextPool.RentPreferred(preferCpu: false);
		var accelerator = gpu.Accelerator;
		var kernel = GetKernel(accelerator);
		var resultBuffer = accelerator.Allocate1D<byte>(1);
		kernel(1, exponent, divisor, resultBuffer.View);
		accelerator.Synchronize();
		bool divisible = resultBuffer.GetAsArray1D()[0] != 0;
		resultBuffer.Dispose();
		gpu.Dispose();
		return divisible;
	}

	private static void Kernel(Index1D _, ulong exponent, ulong divisor, ArrayView<byte> result)
	{
		if (divisor <= 1UL)
		{
			result[0] = 0;
			return;
		}

		GpuUInt128 baseVal, mod;
		int x = 63 - XMath.LeadingZeroCount(divisor);
		if (x == 0)
		{
			mod = new GpuUInt128(0UL, divisor);
			// baseVal is temp value here. We're reusing existing variable in a different meaning for the best performance.
			baseVal = GpuUInt128.Pow2Mod(exponent, mod);
			result[0] = baseVal.High == 0UL && baseVal.Low == 1UL ? (byte)1 : (byte)0;
			return;
		}

		ulong ux = (ulong)x;
		mod = new GpuUInt128(0UL, divisor);
		baseVal = new GpuUInt128(0UL, 1UL << x); // 2^x ≡ -y (mod divisor)
		baseVal.ModPow(exponent / ux, mod);
		var part2 = GpuUInt128.Pow2Mod(exponent % ux, mod);
		baseVal.MulMod(part2, mod);
		result[0] = baseVal.High == 0UL && baseVal.Low == 1UL ? (byte)1 : (byte)0;
	}
}
