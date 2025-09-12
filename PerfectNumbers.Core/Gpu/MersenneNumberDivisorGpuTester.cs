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

	public static void BuildDivisorCandidates()
	{
		uint[] snapshot = MersenneDivisorCycles.Shared.ExportSmallCyclesSnapshot();
		(ulong divisor, uint cycle)[] list = new (ulong divisor, uint cycle)[snapshot.Length / 2];
		uint cycle;
		int count = 0, i, snapshotLength = snapshot.Length;
		for (i = 3; i < snapshotLength; i += 2)
		{
			cycle = snapshot[i];
			if (cycle == 0U)
			{
				continue;
			}

			list[count++] = ((ulong)i, cycle);
		}

		_divisorCandidates = list[..count].ToArray();
	}

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

	private static (ulong divisor, uint cycle)[]? _divisorCandidates;

	public bool IsPrime(ulong p, ulong d, int divisorCyclesSearchLimit, out bool detailedCheck)
	{
		if (d != 0UL)
		{
			if (IsDivisible(p, d))
			{
				detailedCheck = true;
				return false;
			}

			detailedCheck = false;
			return true;
		}

		var candidates = _divisorCandidates!;
		int k, len = candidates.Length;
		uint cycle;

		// k is simple iterator here
		for (k = 0; k < len; k++)
		{
			(d, cycle) = candidates[k];
			if (p % cycle != 0UL)
			{
				continue;
			}

			if (IsDivisible(p, d))
			{
				detailedCheck = true;
				return false;
			}
		}

		ulong kMul2;
		var divisorCycles = MersenneDivisorCycles.Shared;
		for (k = 1; k <= divisorCyclesSearchLimit; k++)
		{
			kMul2 = (ulong)k << 1;
			if (p > ulong.MaxValue / kMul2)
			{
				break;
			}

			d = kMul2 * p + 1UL;
			// kkMul2 now becomes cycle. We're reusing existing variable for the best performance
			kMul2 = divisorCycles.GetCycle(d);
			if (p % kMul2 != 0UL)
			{
				continue;
			}

			if (IsDivisible(p, d))
			{
				detailedCheck = true;
				return false;
			}
		}

		detailedCheck = false;
		return true;
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
