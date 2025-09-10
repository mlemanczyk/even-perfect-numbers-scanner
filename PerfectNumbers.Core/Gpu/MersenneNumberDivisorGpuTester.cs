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
        using var resultBuffer = accelerator.Allocate1D<byte>(1);
        kernel(1, exponent, divisor, resultBuffer.View);
        accelerator.Synchronize();
        bool divisible = resultBuffer.GetAsArray1D()[0] != 0;
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

        int x = 63 - XMath.LeadingZeroCount(divisor);
        if (x == 0)
        {
            var mod = new GpuUInt128(0UL, divisor);
            var temp = GpuUInt128.Pow2Mod(exponent, mod);
            result[0] = temp.High == 0UL && temp.Low == 1UL ? (byte)1 : (byte)0;
            return;
        }

        ulong ux = (ulong)x;
        ulong q = exponent / ux;
        ulong r = exponent % ux;
        ulong pow2x = 1UL << x;
        ulong y = divisor - pow2x;
        var modVal = new GpuUInt128(0UL, divisor);
        var baseVal = new GpuUInt128(0UL, pow2x); // 2^x ≡ -y (mod divisor)
        var part1 = baseVal.ModPow(q, modVal);
        var part2 = GpuUInt128.Pow2Mod(r, modVal);
        var rem = part1.MulMod(part2, modVal);
        result[0] = rem.High == 0UL && rem.Low == 1UL ? (byte)1 : (byte)0;
    }
}
