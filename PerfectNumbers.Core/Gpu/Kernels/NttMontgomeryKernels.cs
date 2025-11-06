using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

internal static class NttMontgomeryKernels
{
    // TODO(MOD-OPT): Stage kernel using 64-bit Montgomery multiplication. Requires data and twiddles in Montgomery domain.
    public static void StageMontKernel(Index1D index, ArrayView<GpuUInt128> data, int len, int half, int stageOffset, ArrayView<GpuUInt128> twiddlesMont, ulong modulus, ulong nPrime)
    {
        int t = index.X;
        // TODO: Apply the bitmask remainder helper to this stage too so every butterfly path drops the slower `%` operation.
        int j = t % half;
        int block = t / half;
        int k = block * len;
        int i1 = k + j;
        int i2 = i1 + half;
        ulong u = data[i1].Low;
        ulong v = data[i2].Low;
        ulong wR = twiddlesMont[stageOffset + j].Low;
        v = MontMul64(v, wR, modulus, nPrime);
        ulong sum = u + v;
        if (sum >= modulus)
        {
            sum -= modulus;
        }
        ulong diff = u >= v ? u - v : unchecked(u + modulus - v);
        data[i1] = new GpuUInt128(0UL, sum);
        data[i2] = new GpuUInt128(0UL, diff);
    }

    public static void ToMont64Kernel(Index1D index, AcceleratorStream stream, ArrayView<GpuUInt128> data, ulong modulus, ulong nPrime, ulong r2)
    {
        ulong a = data[index].Low;
        ulong v = MontMul64(a, r2, modulus, nPrime);
        data[index] = new GpuUInt128(0UL, v);
    }

    public static void FromMont64Kernel(Index1D index, ArrayView<GpuUInt128> data, ulong modulus, ulong nPrime)
    {
        ulong aR = data[index].Low;
        ulong v = MontMul64(aR, 1UL, modulus, nPrime);
        data[index] = new GpuUInt128(0UL, v);
    }

    public static void SquareMont64Kernel(Index1D index, AcceleratorStream stream, ArrayView<GpuUInt128> data, ulong modulus, ulong nPrime)
    {
        ulong aR = data[index].Low;
        ulong v = MontMul64(aR, aR, modulus, nPrime);
        data[index] = new GpuUInt128(0UL, v);
    }

    public static void ScaleMont64Kernel(Index1D index, ArrayView<GpuUInt128> data, ulong modulus, ulong nPrime, ulong scaleMont)
    {
        ulong aR = data[index].Low;
        ulong v = MontMul64(aR, scaleMont, modulus, nPrime);
        data[index] = new GpuUInt128(0UL, v);
    }

    // Computes Montgomery multiplication for 64-bit operands in Montgomery domain.
    // Returns a*b*R^{-1} mod modulus, where R=2^64 and nPrime = -modulus^{-1} mod 2^64.
    private static ulong MontMul64(ulong aR, ulong bR, ulong modulus, ulong nPrime)
    {
        // t = aR * bR (128-bit)
        ulong tLow, tHigh;
        Mul64Parts(aR, bR, out tHigh, out tLow);
        // m = (tLow * nPrime) mod 2^64 -> just low 64 bits of the product
        ulong mLow, mHigh;
        Mul64Parts(tLow, nPrime, out mHigh, out mLow);
        // u = (t + m*modulus) >> 64
        ulong mmLow, mmHigh;
        Mul64Parts(mLow, modulus, out mmHigh, out mmLow);
        // add tLow + mmLow, keep carry
        ulong carry = 0UL;
        ulong low = tLow + mmLow;
        if (low < tLow)
        {
            carry = 1UL;
        }
        // high = tHigh + mmHigh + carry
        ulong high = tHigh + mmHigh + carry;
        ulong u = high; // this is (t + m*n) >> 64
        if (u >= modulus)
        {
            u -= modulus;
        }
        return u;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Mul64Parts(ulong a, ulong b, out ulong high, out ulong low)
    {
        // 64x64 -> 128 using 32-bit limbs
        ulong a0 = (uint)a;
        ulong a1 = a >> 32;
        ulong b0 = (uint)b;
        ulong b1 = b >> 32;

        ulong lo = a0 * b0;
        ulong mid1 = a1 * b0;
        ulong mid2 = a0 * b1;
        ulong hi = a1 * b1;

        ulong carry = (lo >> 32) + (uint)mid1 + (uint)mid2;
        low = (lo & 0xFFFFFFFFUL) | (carry << 32);
        hi += (mid1 >> 32) + (mid2 >> 32) + (carry >> 32);
        high = hi;
    }
}
