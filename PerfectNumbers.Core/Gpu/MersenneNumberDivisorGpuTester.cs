using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

public sealed class MersenneNumberDivisorGpuTester
{
	private readonly ConcurrentDictionary<Accelerator, Action<Index1D, ulong, GpuUInt128, ArrayView<byte>>> _kernelCache = new();
	private readonly ConcurrentDictionary<Accelerator, MemoryBuffer1D<byte, Stride1D.Dense>> _resultBuffers = new();

	private Action<Index1D, ulong, GpuUInt128, ArrayView<byte>> GetKernel(Accelerator accelerator) =>
			_kernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ulong, GpuUInt128, ArrayView<byte>>(Kernel));

        public static void BuildDivisorCandidates()
        {
                ulong[] snapshot = MersenneDivisorCycles.Shared.ExportSmallCyclesSnapshot();
                (ulong divisor, uint cycle)[] list = new (ulong divisor, uint cycle)[snapshot.Length / 2];
                ulong cycle;
                int count = 0, i, snapshotLength = snapshot.Length;
                for (i = 3; i < snapshotLength; i += 2)
                {
                        cycle = snapshot[i];
                        if (cycle == 0U)
                        {
                                continue;
                        }

                        list[count++] = ((ulong)i, (uint)cycle);
                }

                _divisorCandidates = count == 0 ? [] : list[..count];
        }

	public bool IsDivisible(ulong exponent, UInt128 divisor)
	{
		var gpu = GpuContextPool.RentPreferred(preferCpu: false);
		var accelerator = gpu.Accelerator;
		var kernel = GetKernel(accelerator);
		var resultBuffer = _resultBuffers.GetOrAdd(accelerator, acc => acc.Allocate1D<byte>(1));
		resultBuffer.MemSetToZero();
		kernel(1, exponent, (GpuUInt128)divisor, resultBuffer.View);
		accelerator.Synchronize();
		Span<byte> result = stackalloc byte[1];
		resultBuffer.View.CopyToCPU(ref result[0], 1);
		bool divisible = result[0] != 0;
		result[0] = 0;
		resultBuffer.View.CopyFromCPU(ref result[0], 1);
		accelerator.Synchronize();
		gpu.Dispose();
		return divisible;
	}

        private static (ulong divisor, uint cycle)[]? _divisorCandidates = Array.Empty<(ulong divisor, uint cycle)>();

	public bool IsPrime(ulong p, UInt128 d, ulong divisorCyclesSearchLimit, out bool divisorsExhausted)
	{
		if (d != UInt128.Zero)
		{
			if (IsDivisible(p, d))
			{
				divisorsExhausted = true;
				return false;
			}

			divisorsExhausted = false;
			return true;
		}

                if (_divisorCandidates is { Length: > 0 } candidates)
                {
                        int len = candidates.Length;
                        uint cycle;
                        for (int k = 0; k < len; k++)
                        {
                                (ulong dSmall, cycle) = candidates[k];
                                // TODO: Replace this `%` check with the shared divisor-cycle divisibility helper so GPU scans
                                // reuse the cached cycle remainders instead of recomputing modulo for every candidate.
                                if (p % cycle != 0UL)
                                {
                                        continue;
                                }

				if (IsDivisible(p, dSmall))
				{
					divisorsExhausted = true;
					return false;
				}
			}
		}

		UInt128 kMul2;
		var divisorCycles = MersenneDivisorCycles.Shared;
		UInt128 maxK2 = UInt128.MaxValue / ((UInt128)p << 1);
		ulong limit = divisorCyclesSearchLimit;
		if ((UInt128)limit > maxK2)
		{
			limit = (ulong)maxK2;
		}

		ulong k2;
		for (k2 = 1UL; k2 <= limit; k2++)
		{
			kMul2 = (UInt128)k2 << 1;
			UInt128 candidate = checked(kMul2 * p);
			d = checked(candidate + UInt128.One);
			if (p < 64UL && d == ((UInt128)1 << (int)p) - UInt128.One)
			{
				continue;
			}

                        UInt128 cycle128 = MersenneDivisorCycles.GetCycle(d);
                        // TODO: When this lookup falls outside the preloaded snapshot, invoke the single-cycle GPU helper
                        // instead of allowing GetCycle to allocate an entire block; the divisor-cycle benchmarks showed the
                        // one-off computation keeping memory flat while still delivering the optimized stepping data.
                        // TODO: Feed this check through the ProcessEightBitWindows-based residue tracker so we avoid the slow
                        // UInt128 modulo on the CPU fallback highlighted by the Pow2Montgomery benchmarks.
                        if ((UInt128)p % cycle128 != UInt128.Zero)
                        {
                                continue;
                        }

			if (IsDivisible(p, d))
			{
				divisorsExhausted = true;
				return false;
			}
		}

		divisorsExhausted = (UInt128)divisorCyclesSearchLimit >= maxK2;
		return true;
	}

	private static void Kernel(Index1D _, ulong exponent, GpuUInt128 divisor, ArrayView<byte> result)
	{
		if (divisor.High == 0UL && divisor.Low <= 1UL)
		{
			result[0] = 0;
			return;
		}

		GpuUInt128 baseVal, mod;
		int x = divisor.High == 0UL
				? 63 - XMath.LeadingZeroCount(divisor.Low)
				: 127 - XMath.LeadingZeroCount(divisor.High);
		mod = divisor;
                if (x == 0)
                {
                        // TODO: Swap Pow2Mod for the ProcessEightBitWindows helper once Pow2Minus1Mod adopts it;
                        // GpuPow2ModBenchmarks showed the windowed kernel cutting large divisors from ~51 µs to ~21 µs.
                        baseVal = GpuUInt128.Pow2Mod(exponent, mod);
			result[0] = baseVal.High == 0UL && baseVal.Low == 1UL ? (byte)1 : (byte)0;
			return;
		}

		ulong ux = (ulong)x;
		if (x >= 64)
		{
			baseVal = new GpuUInt128(1UL << (x - 64), 0UL);
		}
		else
		{
			baseVal = new GpuUInt128(0UL, 1UL << x);
		}

                GpuUInt128 pow = baseVal;
                pow.ModPow(exponent / ux, mod);
                // TODO: Replace this trailing Pow2Mod call with the ProcessEightBitWindows variant once the shared
                // windowed helper lands so mixed-radix decompositions benefit from the same GPU speedup.
                var part2 = GpuUInt128.Pow2Mod(exponent % ux, mod);
		GpuUInt128 product = pow;
		product.MulMod(part2, mod);
		result[0] = product.High == 0UL && product.Low == 1UL ? (byte)1 : (byte)0;
	}
}

