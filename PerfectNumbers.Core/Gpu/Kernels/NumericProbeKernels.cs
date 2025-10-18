using System.Numerics;
using ILGPU;
using ILGPU.Runtime;
using PeterO.Numbers;
using Open.Numeric.Primes;

namespace PerfectNumbers.Core.Gpu;

internal static class NumericProbeKernels
{
    public static void OpenNumericPrimesIsPrimeKernel(Index1D index, ArrayView1D<int, Stride1D.Dense> data)
    {
        ulong value = (ulong)(index + 2);
        bool isPrime = Prime.Numbers.IsPrime(value);
        data[index] = isPrime ? 1 : 0;
    }

    public static void BigIntegerKernel(Index1D index, ArrayView1D<int, Stride1D.Dense> data)
    {
        BigInteger value = new BigInteger(data[index]);
        value += BigInteger.One;
        value -= BigInteger.One;
        value *= new BigInteger(2);
        value /= new BigInteger(3);
        value %= new BigInteger(5);
        int comparison = value.CompareTo(BigInteger.Zero);
        data[index] = comparison;
    }

    public static void EIntegerKernel(Index1D index, ArrayView1D<int, Stride1D.Dense> data)
    {
        EInteger value = EInteger.FromInt32(data[index]);
        value = value + EInteger.One;
        value = value - EInteger.One;
        value = value * EInteger.FromInt32(2);
        value = value.Divide(EInteger.FromInt32(3));
        value = value.Remainder(EInteger.FromInt32(5));
        int comparison = value.CompareTo(EInteger.Zero);
        data[index] = comparison;
    }

    public static void ERationalKernel(Index1D index, ArrayView1D<int, Stride1D.Dense> data)
    {
        ERational value = ERational.FromInt32(data[index]);
        ERational adjusted = (value + ERational.FromInt32(1)) - ERational.FromInt32(1);
        ERational product = adjusted * ERational.FromInt32(2);
        ERational quotient = product / ERational.FromInt32(3);
        int comparison = quotient.CompareTo(ERational.FromInt32(0));
        data[index] = comparison;
    }
}
