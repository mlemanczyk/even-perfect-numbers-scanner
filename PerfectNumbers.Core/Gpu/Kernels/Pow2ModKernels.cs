using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

internal static class Pow2ModKernels
{
	/// This kernel always sets the result of the corresponding element. Callers don't need to clear the output buffers.
	public static void Pow2ModKernelScan(
		Index1D index,
		ulong exponent,
		GpuUInt128 twoP,
		GpuUInt128 kStart,
		byte lastIsSeven,
		ulong _,
		ResidueAutomatonArgs ra,
		ArrayView<ulong> orders,
		ArrayView1D<ulong, Stride1D.Dense> smallCycles,
		ArrayView1D<uint, Stride1D.Dense> smallPrimesLastOne,
		ArrayView1D<uint, Stride1D.Dense> smallPrimesLastSeven,
		ArrayView1D<ulong, Stride1D.Dense> smallPrimesPow2LastOne,
		ArrayView1D<ulong, Stride1D.Dense> smallPrimesPow2LastSeven)
	{
		ulong idx = (ulong)index.X;
		// TODO: Replace % operator in GPU and their outgoing paths with logical operations, where possible. 
		ulong idxMod3 = idx % 3UL;
		ulong idxMod5 = idx % 5UL;
		ulong r10 = ra.Q0M10 + (ra.Step10 * idx).Mod10(); r10 -= (r10 >= 10UL) ? 10UL : 0UL;
		bool shouldCheck = r10 != 5UL;
		// TODO: Replace these "if" statements with ?:
		if (shouldCheck)
		{
			ulong r8 = (ra.Q0M8 + ((ra.Step8 * idx) & 7UL)) & 7UL;
			if (r8 != 1UL && r8 != 7UL)
			{
				shouldCheck = false;
			}
			else
			{
				ulong r3 = ra.Q0M3 + (ra.Step3 * idxMod3);
				if (r3 >= 6UL) r3 -= 6UL;
				if (r3 >= 3UL) r3 -= 3UL;
				if (r3 == 0UL)
				{
					shouldCheck = false;
				}
				else
				{
					ulong r5 = ra.Q0M5 + (ra.Step5 * idxMod5);
					if (r5 >= 15UL) r5 -= 15UL;
					if (r5 >= 10UL) r5 -= 10UL;
					if (r5 >= 5UL) r5 -= 5UL;
					if (r5 == 0UL)
					{
						shouldCheck = false;
					}
				}
			}
		}
		if (!shouldCheck)
		{
			orders[index] = 0UL;
			return;
		}

		kStart.Add(idx);
		GpuUInt128 q = twoP;
		GpuUInt128.Mul64(ref q, kStart.High, kStart.Low);
		q.Add(GpuUInt128.One);
		ReadOnlyGpuUInt128 readOnlyQ = q.AsReadOnly();
		if (q.High == 0UL && q.Low < (ulong)smallCycles.Length)
		{
			ulong cycle = smallCycles[(int)q.Low];
			// cycle should always be initialized if we're within array limit in production code
			if (cycle != 0UL && cycle <= exponent && (exponent % cycle) != 0UL)
			{
				orders[index] = 0UL;
				return;
			}
		}
		if (GpuUInt128.Pow2Minus1Mod(exponent, in readOnlyQ) != GpuUInt128.Zero)
		{
			orders[index] = 0UL;
			return;
		}

		ArrayView1D<uint, Stride1D.Dense> primes = lastIsSeven != 0 ? smallPrimesLastSeven : smallPrimesLastOne;
		ArrayView1D<ulong, Stride1D.Dense> primesPow2 = lastIsSeven != 0 ? smallPrimesPow2LastSeven : smallPrimesPow2LastOne;
		int primesLen = (int)primes.Length;
		for (int i = 0; i < primesLen; i++)
		{
			ulong square = primesPow2[i];
			if (new GpuUInt128(0UL, square) > q)
			{
				break;
			}
			ulong prime = primes[i];
			if (KernelMathHelpers.Mod128By64(q, prime) == 0UL)
			{
				orders[index] = 0UL;
				return;
			}
		}

		orders[index] = exponent;
	}

	/// This kernel doesn't always set the result of the found array. It sets the value only when it needs to. The callers
	/// must clean the output buffer before the call to get a deterministic result.
	public static void Pow2ModOrderKernelScan(
		Index1D index,
		ulong exponent,
		GpuUInt128 twoP,
		GpuUInt128 kStart,
		byte _,
		ulong __,
		ResidueAutomatonArgs ra,
		ArrayView<int> found,
		ArrayView1D<ulong, Stride1D.Dense> smallCycles)
	{
		ulong idx = (ulong)index.X;
		ulong idxMod3 = idx % 3UL;
		ulong idxMod5 = idx % 5UL;
		ulong r10 = ra.Q0M10 + (ra.Step10 * idx).Mod10(); r10 -= (r10 >= 10UL) ? 10UL : 0UL;
		bool shouldCheck = r10 != 5UL;
		if (shouldCheck)
		{
			ulong r8 = (ra.Q0M8 + ((ra.Step8 * idx) & 7UL)) & 7UL;
			if (r8 != 1UL && r8 != 7UL)
			{
				shouldCheck = false;
			}
			else
			{
				ulong r3 = ra.Q0M3 + (ra.Step3 * idxMod3);
				if (r3 >= 6UL) r3 -= 6UL;
				if (r3 >= 3UL) r3 -= 3UL;
				if (r3 == 0UL)
				{
					shouldCheck = false;
				}
				else
				{
					ulong r5 = ra.Q0M5 + (ra.Step5 * idxMod5);
					if (r5 >= 15UL) r5 -= 15UL;
					if (r5 >= 10UL) r5 -= 10UL;
					if (r5 >= 5UL) r5 -= 5UL;
					if (r5 == 0UL)
					{
						shouldCheck = false;
					}
				}
			}
		}
		if (!shouldCheck)
		{
			return;
		}

		GpuUInt128 k = kStart + (GpuUInt128)idx;
		GpuUInt128 q = twoP;
		GpuUInt128.Mul64(ref q, k.High, k.Low);
		q.Add(GpuUInt128.One);
		ReadOnlyGpuUInt128 readOnlyQ = q.AsReadOnly();
		// Small-cycles in-kernel early rejection from device table
		if (q.High == 0UL && q.Low < (ulong)smallCycles.Length)
		{
			ulong cycle = smallCycles[(int)q.Low];
			if (cycle != 0UL && cycle <= exponent && (exponent % cycle) != 0UL)
			{
				return;
			}
		}
		if (GpuUInt128.Pow2Mod(exponent, in readOnlyQ) != GpuUInt128.One)
		{
			return;
		}

		Atomic.Or(ref found[0], 1);
	}
}
