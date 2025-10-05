using System;
using System.Buffers.Binary;
using System.Numerics;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core;
using PerfectNumbers.Core.Gpu;

namespace EvenPerfectBitScanner.Benchmarks;

[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net80, launchCount: 1, warmupCount: 1, iterationCount: 5)]
public class Pow2MontgomeryModUInt128Benchmarks
{
    private const int SampleCount = 256;
    private const ulong Pow2WindowFallbackThreshold = 32UL;

    private readonly UInt128[] _moduli = new UInt128[SampleCount];
    private readonly UInt128[] _exponents = new UInt128[SampleCount];

    private readonly Random _random = new(19);

    [GlobalSetup]
    public void Setup()
    {
        for (int i = 0; i < SampleCount; i++)
        {
            UInt128 modulus = NextWideOddModulus();
            _moduli[i] = modulus;

            UInt128 exponent = NextWideExponent();
            _exponents[i] = exponent;
        }
    }

    /// <summary>
    /// Current UInt128-native windowed Montgomery ladder that avoids GpuUInt128 on CPU paths.
    /// </summary>
    [Benchmark(Baseline = true)]
    public UInt128 UInt128Windowed()
    {
        UInt128 checksum = UInt128.Zero;

        for (int i = 0; i < SampleCount; i++)
        {
            checksum ^= _exponents[i].Pow2MontgomeryModWindowed(_moduli[i]);
        }

        return checksum;
    }

    /// <summary>
    /// Previous CPU shim that reused the GPU-oriented GpuUInt128 implementation.
    /// </summary>
    [Benchmark]
    public UInt128 GpuStructShim()
    {
        UInt128 checksum = UInt128.Zero;

        for (int i = 0; i < SampleCount; i++)
        {
            checksum ^= Pow2MontgomeryModWindowedGpuShim(_exponents[i], _moduli[i]);
        }

        return checksum;
    }

    private UInt128 NextWideOddModulus()
    {
        Span<byte> buffer = stackalloc byte[16];
        _random.NextBytes(buffer);

        ulong low = BinaryPrimitives.ReadUInt64LittleEndian(buffer);
        ulong high = BinaryPrimitives.ReadUInt64LittleEndian(buffer[8..]);
        if (high == 0UL)
        {
            high = 1UL;
        }

        UInt128 modulus = ((UInt128)high << 64) | low;
        modulus |= UInt128.One;

        if (modulus <= UInt128.One)
        {
            modulus += 2UL;
        }

        return modulus;
    }

    private UInt128 NextWideExponent()
    {
        Span<byte> buffer = stackalloc byte[16];
        _random.NextBytes(buffer);

        ulong low = BinaryPrimitives.ReadUInt64LittleEndian(buffer);
        ulong high = BinaryPrimitives.ReadUInt64LittleEndian(buffer[8..]);
        if (high == 0UL)
        {
            high = 1UL;
        }

        return ((UInt128)high << 64) | low;
    }

    private static UInt128 Pow2MontgomeryModWindowedGpuShim(UInt128 exponent, UInt128 modulus)
    {
        if (modulus == UInt128.One)
        {
            return UInt128.Zero;
        }

        if (exponent == UInt128.Zero)
        {
            return UInt128.One % modulus;
        }

        GpuUInt128 modulusGpu = new(modulus);
        GpuUInt128 exponentGpu = new(exponent);
        GpuUInt128 baseValue = new(2UL);
        if (baseValue.CompareTo(modulusGpu) >= 0)
        {
            baseValue.Sub(modulusGpu);
        }

        if (ShouldUseSingleBit(exponentGpu))
        {
            GpuUInt128 singleBitResult = Pow2MontgomeryModSingleBit(exponentGpu, modulusGpu, baseValue);
            return (UInt128)singleBitResult;
        }

        int bitLength = exponentGpu.GetBitLength();
        int windowSize = GetWindowSize(bitLength);
        int oddPowerCount = 1 << (windowSize - 1);

        Span<GpuUInt128> oddPowers = stackalloc GpuUInt128[oddPowerCount];
        InitializeOddPowers(baseValue, modulusGpu, oddPowers);

        GpuUInt128 result = GpuUInt128.One;
        int index = bitLength - 1;

        while (index >= 0)
        {
            if (!IsBitSet(exponentGpu, index))
            {
                result.MulMod(result, modulusGpu);
                index--;
                continue;
            }

            int windowStart = index - windowSize + 1;
            if (windowStart < 0)
            {
                windowStart = 0;
            }

            while (!IsBitSet(exponentGpu, windowStart))
            {
                windowStart++;
            }

            int windowBitCount = index - windowStart + 1;
            for (int square = 0; square < windowBitCount; square++)
            {
                result.MulMod(result, modulusGpu);
            }

            ulong windowValue = ExtractWindowValue(exponentGpu, windowStart, windowBitCount);
            int tableIndex = (int)((windowValue - 1UL) >> 1);
            GpuUInt128 factor = oddPowers[tableIndex];
            result.MulMod(factor, modulusGpu);

            index = windowStart - 1;
        }

        return (UInt128)result;
    }

    private static bool ShouldUseSingleBit(in GpuUInt128 exponent)
    {
        if (exponent.High == 0UL && exponent.Low <= Pow2WindowFallbackThreshold)
        {
            return true;
        }

        return false;
    }

    private static int GetWindowSize(int bitLength)
    {
        if (bitLength <= 8)
        {
            return Math.Max(bitLength, 1);
        }

        if (bitLength <= 23)
        {
            return 4;
        }

        if (bitLength <= 79)
        {
            return 5;
        }

        if (bitLength <= 239)
        {
            return 6;
        }

        if (bitLength <= 671)
        {
            return 7;
        }

        return 8;
    }

    private static void InitializeOddPowers(GpuUInt128 baseValue, GpuUInt128 modulus, Span<GpuUInt128> oddPowers)
    {
        oddPowers[0] = baseValue;
        if (oddPowers.Length == 1)
        {
            return;
        }

        // Reusing baseValue to hold base^2 for the shared odd-power ladder.
        baseValue.MulMod(baseValue, modulus);
        for (int i = 1; i < oddPowers.Length; i++)
        {
            oddPowers[i] = oddPowers[i - 1];
            oddPowers[i].MulMod(baseValue, modulus);
        }
    }

    private static GpuUInt128 Pow2MontgomeryModSingleBit(GpuUInt128 exponent, GpuUInt128 modulus, GpuUInt128 baseValue)
    {
        GpuUInt128 result = GpuUInt128.One;

        while (!exponent.IsZero)
        {
            if ((exponent.Low & 1UL) != 0UL)
            {
                result.MulMod(baseValue, modulus);
            }

            exponent.ShiftRight(1);
            if (exponent.IsZero)
            {
                break;
            }

            // Reusing baseValue to store the squared base for the next iteration.
            baseValue.MulMod(baseValue, modulus);
        }

        return result;
    }

    private static bool IsBitSet(in GpuUInt128 value, int bitIndex)
    {
        if (bitIndex >= 64)
        {
            return ((value.High >> (bitIndex - 64)) & 1UL) != 0UL;
        }

        return ((value.Low >> bitIndex) & 1UL) != 0UL;
    }

    private static ulong ExtractWindowValue(in GpuUInt128 exponent, int windowStart, int windowBitCount)
    {
        if (windowStart != 0)
        {
            GpuUInt128 shifted = exponent;
            shifted.ShiftRight(windowStart);
            ulong mask = (1UL << windowBitCount) - 1UL;
            return shifted.Low & mask;
        }

        ulong directMask = (1UL << windowBitCount) - 1UL;
        return exponent.Low & directMask;
    }
}
