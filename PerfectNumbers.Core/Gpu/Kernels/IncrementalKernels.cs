using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

internal static class IncrementalKernels
{
    // TODO: Plumb the small-cycles device buffer into all kernels that can benefit
    // (some already accept it). Consider a compact type (byte/ushort) for memory footprint.
    public static void IncrementalKernelScan(
        Index1D index,
        ulong exponent,
        GpuUInt128 twoP,
        GpuUInt128 kStart,
        byte lastIsSeven,
        ulong divMul,
        ulong q0m10,
        ulong q0m8,
        ulong q0m3,
        ulong q0m5,
        ArrayView<ulong> orders,
        ArrayView1D<ulong, Stride1D.Dense> smallCycles)
    {
        ulong idx = (ulong)index.X;
        // TODO: Replace these `%` computations with the precomputed Mod3/Mod5 tables so GPU kernels reuse cached residues
        // instead of performing modulo operations that the benchmarks showed slower on wide sweeps.
        ulong idxMod3 = idx % 3UL;
        ulong idxMod5 = idx % 5UL;
        // residue automaton
        ulong step10 = ((ulong)(exponent.Mod10() << 1)).Mod10();
        ulong r10 = q0m10 + (step10 * idx).Mod10();
        r10 -= (r10 >= 10UL) ? 10UL : 0UL;
        bool shouldCheck = r10 != 5UL; // skip q ≡ 5 (mod 10); other residues handled by r8/r3/r5
        if (shouldCheck)
        {
            ulong step8 = ((exponent & 7UL) << 1) & 7UL;
            ulong r8 = (q0m8 + ((step8 * idx) & 7UL)) & 7UL;
            if (r8 != 1UL && r8 != 7UL)
            {
                shouldCheck = false;
            }
            else
            {
                // TODO: Swap these `%` filters to the shared Mod helpers so the residue automaton matches the benchmarked
                // bitmask-based implementation for GPU workloads.
                ulong step3 = ((exponent % 3UL) << 1) % 3UL;
                ulong r3 = q0m3 + (step3 * idxMod3); r3 -= (r3 >= 3UL) ? 3UL : 0UL;
                if (r3 == 0UL)
                {
                    shouldCheck = false;
                }
                else
                {
                    ulong step5Loc = ((exponent % 5UL) << 1) % 5UL;
                    ulong r5 = q0m5 + (step5Loc * idxMod5); if (r5 >= 5UL) r5 -= 5UL;
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
                orders[index] = 0UL;
                return;
            }
        }

        // TODO: q will be lowered by GpuUInt128.One here. Is this expected? Maybe that's why we have issues with the incremental kernel? Review other places where we use GpuUInt128 operators.
        GpuUInt128 phi = q - GpuUInt128.One;
        // TODO: Do we have the check here because we don't support high values or this is expected result?
        if (phi.High != 0UL)
        {
            orders[index] = 0UL;
            return;
        }

        ulong phi64 = phi.Low;
        if (GpuUInt128.Pow2Mod(phi64, in readOnlyQ) != GpuUInt128.One)
        {
            orders[index] = 0UL;
            return;
        }

        GpuUInt128 halfPow = GpuUInt128.Pow2Mod(phi64 >> 1, in readOnlyQ) - GpuUInt128.One;
        if (GpuUInt128.BinaryGcd(halfPow, q) != GpuUInt128.One)
        {
            orders[index] = 0UL;
            return;
        }

        ulong div = KernelMathHelpers.FastDiv64Gpu(phi64, exponent, divMul);
        GpuUInt128 divPow = GpuUInt128.Pow2Mod(div, in readOnlyQ) - GpuUInt128.One;
        if (GpuUInt128.BinaryGcd(divPow, q) != GpuUInt128.One)
        {
            orders[index] = 0UL;
            return;
        }

        orders[index] = 1UL;
    }

    public static void IncrementalOrderKernelScan(
        Index1D index,
        ulong exponent,
        GpuUInt128 twoP,
        GpuUInt128 kStart,
        byte _,
        ulong divMul,
        ResidueAutomatonArgs ra,
        ArrayView<int> found,
        ArrayView1D<ulong, Stride1D.Dense> smallCycles)
	{
		// TODO: Modify this kernel to always assign the required found element to the result of the calculations and
		// remove any clearing of the output buffer in the caller methods.
        ulong idx = (ulong)index.X;
		// TODO: Replace % operator in GPU and their outgoing paths with logical operations, where possible. 
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
        GpuUInt128 phi = q - GpuUInt128.One;
        if (phi.High != 0UL)
        {
            return;
        }

        ulong phi64 = phi.Low;
        if (GpuUInt128.Pow2Mod(phi64, in readOnlyQ) != GpuUInt128.One)
        {
            return;
        }

        GpuUInt128 halfPow = GpuUInt128.Pow2Mod(phi64 >> 1, in readOnlyQ) - GpuUInt128.One;
        if (GpuUInt128.BinaryGcd(halfPow, q) != GpuUInt128.One)
        {
            return;
        }

        ulong div = KernelMathHelpers.FastDiv64Gpu(phi64, exponent, divMul);
        GpuUInt128 divPow = GpuUInt128.Pow2Mod(div, in readOnlyQ) - GpuUInt128.One;
        if (GpuUInt128.BinaryGcd(divPow, q) != GpuUInt128.One)
        {
            return;
        }

        Atomic.Or(ref found[0], 1);
    }
}
