using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu.Kernels;

internal static class Pow2MontgomeryKernels
{
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static GpuUInt128 AdvancePolynomialGpu(GpuUInt128 x, ulong c, ulong modulus)
	{
		x.Pow2();
		x.AddMod(c, modulus);
		return x;
	}

	public static void Pow2MontgomeryKernelKeepMontgomery(
		Index1D index,
		ArrayView1D<ulong, Stride1D.Dense> exponents,
		MontgomeryDivisorData divisor,
		ArrayView1D<ulong, Stride1D.Dense> results)
	{
		results[0] = ULongExtensions.Pow2MontgomeryModWindowedGpuKeepMontgomery(divisor, exponents[0]);
	}

	public static void Pow2MontgomeryKernelConvertToStandard(
		Index1D index,
		ArrayView1D<ulong, Stride1D.Dense> exponents,
		MontgomeryDivisorData divisor,
		ArrayView1D<ulong, Stride1D.Dense> results)
	{
		results[0] = ULongExtensions.Pow2MontgomeryModWindowedGpuConvertToStandard(divisor, exponents[0]);
	}


	public static void TryPollardRhoKernel(ulong n, int limit, ArrayView1D<ulong, Stride1D.Dense> randomState, ArrayView1D<byte, Stride1D.Dense> factored, ArrayView1D<ulong, Stride1D.Dense> factor)
	{
		factored[0] = 0;
		if ((n & 1UL) == 0UL)
		{
			factor[0] = 2UL;
			factored[0] = 1;
			return;
		}

		int attempt = 0;
		GpuUInt128 nGpuUInt128 = (GpuUInt128)n;
		DeterministicRandomGpu random = new(randomState[0]);
		while (attempt < limit)
		{
			ulong c = (random.NextUInt64() % (n - 1UL)) + 1UL;
			GpuUInt128 x = (random.NextUInt64() % (n - 2UL)) + 2UL;
			GpuUInt128 y = x;
			GpuUInt128 d = GpuUInt128.One;

			while (d.CompareTo(GpuUInt128.One) == 0)
			{
				x = AdvancePolynomialGpu(x, c, modulus: n);
				y = AdvancePolynomialGpu(y, c, n);
				y = AdvancePolynomialGpu(y, c, n);
				GpuUInt128 diff = x.CompareTo(y) > 0 ? x - y : y - x;
				d = GpuUInt128.BinaryGcd(diff, nGpuUInt128);
			}

			if (d != n)
			{
				if (d.High == 0)
				{
					factor[0] = d.Low;
					factored[0] = 1;
					return;
				}
			}

			attempt++;
		}

		randomState[0] = random.State;
	}
}
