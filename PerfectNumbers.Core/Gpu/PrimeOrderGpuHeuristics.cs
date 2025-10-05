using System;
using System.Buffers;
using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;
using System.Numerics;
using PerfectNumbers.Core;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
namespace PerfectNumbers.Core.Gpu;

internal enum GpuPow2ModStatus
{
    Success,
    Overflow,
    Unavailable,
}

internal static class PrimeOrderGpuHeuristics
{
    private static readonly ConcurrentDictionary<ulong, byte> OverflowedPrimes = new();
    private static readonly ConcurrentDictionary<UInt128, byte> OverflowedPrimesWide = new();
    private static readonly ConcurrentDictionary<Accelerator, Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>> Pow2ModKernelCache = new();
    private static readonly ConcurrentDictionary<Accelerator, Action<AcceleratorStream, Index1D, ArrayView1D<GpuUInt128, Stride1D.Dense>, GpuUInt128, ArrayView1D<GpuUInt128, Stride1D.Dense>>> Pow2ModKernelWideCache = new();
    private const int WideStackThreshold = 8;
    private static PrimeOrderGpuCapability s_capability = PrimeOrderGpuCapability.Default;

    private const int Pow2WindowSizeBits = 8;
    private const int Pow2WindowOddPowerCount = 1 << (Pow2WindowSizeBits - 1);
    private const ulong Pow2WindowFallbackThreshold = 32UL;

    [InlineArray(Pow2WindowOddPowerCount)]
    private struct Pow2OddPowerTable
    {
        private GpuUInt128 _element0;
    }

    internal static ConcurrentDictionary<ulong, byte> OverflowRegistry => OverflowedPrimes;
    internal static ConcurrentDictionary<UInt128, byte> OverflowRegistryWide => OverflowedPrimesWide;

    internal static void OverrideCapabilitiesForTesting(PrimeOrderGpuCapability capability)
    {
        s_capability = capability;
    }

    internal static void ResetCapabilitiesForTesting()
    {
        s_capability = PrimeOrderGpuCapability.Default;
    }

    public static GpuPow2ModStatus TryPow2Mod(ulong exponent, ulong prime, out ulong remainder)
    {
        Span<ulong> exponents = stackalloc ulong[1];
        Span<ulong> remainders = stackalloc ulong[1];
        exponents[0] = exponent;

        GpuPow2ModStatus status = TryPow2ModBatch(exponents, prime, remainders);
        remainder = remainders[0];
        return status;
    }

    public static GpuPow2ModStatus TryPow2ModBatch(ReadOnlySpan<ulong> exponents, ulong prime, Span<ulong> remainders)
    {
        if (exponents.Length == 0)
        {
            return GpuPow2ModStatus.Success;
        }

        if (remainders.Length < exponents.Length)
        {
            throw new ArgumentException("Remainder span is shorter than the exponent span.", nameof(remainders));
        }

        Span<ulong> target = remainders.Slice(0, exponents.Length);
        target.Clear();

        if (prime <= 1UL)
        {
            return GpuPow2ModStatus.Unavailable;
        }

        ConcurrentDictionary<ulong, byte> overflowRegistry = OverflowedPrimes;

        if (overflowRegistry.ContainsKey(prime))
        {
            return GpuPow2ModStatus.Overflow;
        }

        PrimeOrderGpuCapability capability = s_capability;

        if (prime.GetBitLength() > capability.ModulusBits)
        {
            overflowRegistry[prime] = 0;
            return GpuPow2ModStatus.Overflow;
        }

        for (int i = 0; i < exponents.Length; i++)
        {
            if (exponents[i].GetBitLength() > capability.ExponentBits)
            {
                return GpuPow2ModStatus.Overflow;
            }
        }

        bool computed = TryComputeOnGpu(exponents, prime, target);
        return computed ? GpuPow2ModStatus.Success : GpuPow2ModStatus.Unavailable;
    }

    public static GpuPow2ModStatus TryPow2Mod(UInt128 exponent, UInt128 prime, out UInt128 remainder)
    {
        Span<UInt128> exponents = stackalloc UInt128[1];
        Span<UInt128> remainders = stackalloc UInt128[1];
        exponents[0] = exponent;

        GpuPow2ModStatus status = TryPow2ModBatch(exponents, prime, remainders);
        remainder = remainders[0];
        return status;
    }

    public static GpuPow2ModStatus TryPow2ModBatch(ReadOnlySpan<UInt128> exponents, UInt128 prime, Span<UInt128> remainders)
    {
        if (exponents.Length == 0)
        {
            return GpuPow2ModStatus.Success;
        }

        if (remainders.Length < exponents.Length)
        {
            throw new ArgumentException("Remainder span is shorter than the exponent span.", nameof(remainders));
        }

        Span<UInt128> target = remainders.Slice(0, exponents.Length);
        target.Clear();

        ConcurrentDictionary<UInt128, byte> overflowRegistryWide = OverflowedPrimesWide;

        if (overflowRegistryWide.ContainsKey(prime))
        {
            return GpuPow2ModStatus.Overflow;
        }

        PrimeOrderGpuCapability capability = s_capability;

        if (prime.GetBitLength() > capability.ModulusBits)
        {
            overflowRegistryWide[prime] = 0;
            return GpuPow2ModStatus.Overflow;
        }

        for (int i = 0; i < exponents.Length; i++)
        {
            if (exponents[i].GetBitLength() > capability.ExponentBits)
            {
                return GpuPow2ModStatus.Overflow;
            }
        }

        bool computed = TryComputeOnGpuWide(exponents, prime, target);
        return computed ? GpuPow2ModStatus.Success : GpuPow2ModStatus.Unavailable;
    }

    private static bool TryComputeOnGpu(ReadOnlySpan<ulong> exponents, ulong prime, Span<ulong> results)
    {
        try
        {
            var lease = GpuKernelPool.GetKernel(useGpuOrder: true);
            try
            {
                using var execution = lease.EnterExecutionScope();
                Accelerator accelerator = lease.Accelerator;
                AcceleratorStream stream = lease.Stream;
                var kernel = GetPow2ModKernel(accelerator);
                using var exponentBuffer = accelerator.Allocate1D<ulong>(exponents.Length);
                using var remainderBuffer = accelerator.Allocate1D<ulong>(exponents.Length);

                exponentBuffer.View.CopyFromCPU(ref MemoryMarshal.GetReference(exponents), exponents.Length);
                remainderBuffer.MemSetToZero();

                MontgomeryDivisorData divisor = MontgomeryDivisorDataCache.Get(prime);
                kernel(stream, exponents.Length, exponentBuffer.View, divisor, remainderBuffer.View);

                stream.Synchronize();

                remainderBuffer.View.CopyToCPU(ref MemoryMarshal.GetReference(results), results.Length);
                return true;
            }
            finally
            {
                lease.Dispose();
            }
        }
        catch (Exception ex) when (ex is AcceleratorException or InternalCompilerException or NotSupportedException or InvalidOperationException or AggregateException)
        {
            ComputePow2ModCpu(exponents, prime, results);
            return true;
        }
    }

    private static void ComputePow2ModCpu(ReadOnlySpan<ulong> exponents, ulong prime, Span<ulong> results)
    {
        int length = exponents.Length;
        for (int i = 0; i < length; i++)
        {
            results[i] = Pow2ModCpu(exponents[i], prime);
        }
    }

    private static bool TryComputeOnGpuWide(ReadOnlySpan<UInt128> exponents, UInt128 prime, Span<UInt128> results)
    {
        int length = exponents.Length;
        if (length == 0)
        {
            return true;
        }

        GpuUInt128[]? rentedExponents = null;
        GpuUInt128[]? rentedResults = null;
        var lease = GpuKernelPool.GetKernel(useGpuOrder: true);

        try
        {
            using var execution = lease.EnterExecutionScope();
            Accelerator accelerator = lease.Accelerator;
            AcceleratorStream stream = lease.Stream;
            var kernel = GetPow2ModWideKernel(accelerator);
            using var exponentBuffer = accelerator.Allocate1D<GpuUInt128>(length);
            using var remainderBuffer = accelerator.Allocate1D<GpuUInt128>(length);

            Span<GpuUInt128> exponentSpan = length <= WideStackThreshold
                ? stackalloc GpuUInt128[length]
                : new Span<GpuUInt128>(rentedExponents = ArrayPool<GpuUInt128>.Shared.Rent(length), 0, length);

            for (int i = 0; i < length; i++)
            {
                exponentSpan[i] = (GpuUInt128)exponents[i];
            }

            exponentBuffer.View.CopyFromCPU(ref MemoryMarshal.GetReference(exponentSpan), length);
            remainderBuffer.MemSetToZero();

            GpuUInt128 modulus = (GpuUInt128)prime;
            kernel(stream, length, exponentBuffer.View, modulus, remainderBuffer.View);

            stream.Synchronize();

            Span<GpuUInt128> resultSpan = length <= WideStackThreshold
                ? stackalloc GpuUInt128[length]
                : new Span<GpuUInt128>(rentedResults = ArrayPool<GpuUInt128>.Shared.Rent(length), 0, length);

            remainderBuffer.View.CopyToCPU(ref MemoryMarshal.GetReference(resultSpan), length);

            for (int i = 0; i < length; i++)
            {
                results[i] = (UInt128)resultSpan[i];
            }

            return true;
        }
        catch (Exception ex) when (ex is AcceleratorException or InternalCompilerException or NotSupportedException or InvalidOperationException or AggregateException)
        {
            ComputePow2ModCpuWide(exponents, prime, results);
            return true;
        }
        finally
        {
            if (rentedExponents is not null)
            {
                ArrayPool<GpuUInt128>.Shared.Return(rentedExponents, clearArray: false);
            }

            if (rentedResults is not null)
            {
                ArrayPool<GpuUInt128>.Shared.Return(rentedResults, clearArray: false);
            }

            lease.Dispose();
        }
    }

    private static void ComputePow2ModCpuWide(ReadOnlySpan<UInt128> exponents, UInt128 prime, Span<UInt128> results)
    {
        int length = exponents.Length;
        for (int i = 0; i < length; i++)
        {
            results[i] = exponents[i].Pow2MontgomeryModWindowed(prime);
        }
    }

    private static ulong Pow2ModCpu(ulong exponent, ulong modulus)
    {
        if (modulus <= 1UL)
        {
            return 0UL;
        }

        MontgomeryDivisorData divisor = MontgomeryDivisorDataCache.Get(modulus);
        return exponent.Pow2MontgomeryModWindowed(divisor, keepMontgomery: false);
    }

    private static Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>> GetPow2ModKernel(Accelerator accelerator)
    {
        return Pow2ModKernelCache.GetOrAdd(accelerator, static accel =>
        {
            var loaded = accel.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>(Pow2ModKernel);
            var kernel = KernelUtil.GetKernel(loaded);
            return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>>();
        });
    }

    private static void Pow2ModKernel(Index1D index, ArrayView1D<ulong, Stride1D.Dense> exponents, MontgomeryDivisorData divisor, ArrayView1D<ulong, Stride1D.Dense> remainders)
    {
        ulong exponent = exponents[index];
        remainders[index] = exponent.Pow2MontgomeryModWindowed(divisor, keepMontgomery: false);
    }

    private static Action<AcceleratorStream, Index1D, ArrayView1D<GpuUInt128, Stride1D.Dense>, GpuUInt128, ArrayView1D<GpuUInt128, Stride1D.Dense>> GetPow2ModWideKernel(Accelerator accelerator)
    {
        return Pow2ModKernelWideCache.GetOrAdd(accelerator, static accel =>
        {
            var loaded = accel.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<GpuUInt128, Stride1D.Dense>, GpuUInt128, ArrayView1D<GpuUInt128, Stride1D.Dense>>(Pow2ModKernelWide);
            var kernel = KernelUtil.GetKernel(loaded);
            return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView1D<GpuUInt128, Stride1D.Dense>, GpuUInt128, ArrayView1D<GpuUInt128, Stride1D.Dense>>>();
        });
    }

    private static void Pow2ModKernelWide(Index1D index, ArrayView1D<GpuUInt128, Stride1D.Dense> exponents, GpuUInt128 modulus, ArrayView1D<GpuUInt128, Stride1D.Dense> remainders)
    {
        GpuUInt128 exponent = exponents[index];
        remainders[index] = Pow2ModKernelCore(exponent, modulus);
    }

    private static GpuUInt128 Pow2ModKernelCore(GpuUInt128 exponent, GpuUInt128 modulus)
    {
        if (modulus == GpuUInt128.One)
        {
            return GpuUInt128.Zero;
        }

        if (exponent.IsZero)
        {
            return GpuUInt128.One;
        }

        GpuUInt128 baseValue = new GpuUInt128(2UL);
        if (baseValue.CompareTo(modulus) >= 0)
        {
            baseValue.Sub(modulus);
        }

        if (ShouldUseSingleBit(exponent))
        {
            return Pow2MontgomeryModSingleBit(exponent, modulus, baseValue);
        }

        int bitLength = exponent.GetBitLength();
        int windowSize = GetWindowSize(bitLength);
        int oddPowerCount = 1 << (windowSize - 1);

        Pow2OddPowerTable oddPowers = default;
        InitializeOddPowers(ref oddPowers, baseValue, modulus, oddPowerCount);

        GpuUInt128 result = GpuUInt128.One;
        int index = bitLength - 1;

        while (index >= 0)
        {
            if (!IsBitSet(exponent, index))
            {
                result.MulMod(result, modulus);
                index--;
                continue;
            }

            int windowStart = index - windowSize + 1;
            if (windowStart < 0)
            {
                windowStart = 0;
            }

            while (!IsBitSet(exponent, windowStart))
            {
                windowStart++;
            }

            int windowBitCount = index - windowStart + 1;
            for (int square = 0; square < windowBitCount; square++)
            {
                result.MulMod(result, modulus);
            }

            ulong windowValue = ExtractWindowValue(exponent, windowStart, windowBitCount);
            int tableIndex = (int)((windowValue - 1UL) >> 1);
            GpuUInt128 factor = oddPowers[tableIndex];
            result.MulMod(factor, modulus);

            index = windowStart - 1;
        }

        return result;
    }

    private static bool ShouldUseSingleBit(in GpuUInt128 exponent) => exponent.High == 0UL && exponent.Low <= Pow2WindowFallbackThreshold;

    private static int GetWindowSize(int bitLength)
    {
        if (bitLength <= Pow2WindowSizeBits)
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

        return Pow2WindowSizeBits;
    }

    private static void InitializeOddPowers(ref Pow2OddPowerTable oddPowers, GpuUInt128 baseValue, GpuUInt128 modulus, int oddPowerCount)
    {
        oddPowers[0] = baseValue;
        if (oddPowerCount == 1)
        {
            return;
        }

        // Reusing baseValue to hold base^2 for the shared odd-power ladder that follows.
        baseValue.MulMod(baseValue, modulus);
        for (int i = 1; i < oddPowerCount; i++)
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

    internal readonly record struct PrimeOrderGpuCapability(int ModulusBits, int ExponentBits)
    {
        public static PrimeOrderGpuCapability Default => new(128, 128);
    }
}
