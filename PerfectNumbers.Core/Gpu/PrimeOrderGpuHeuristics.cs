using System;
using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core;
using System.Runtime.InteropServices;
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
    private static readonly ConcurrentDictionary<Accelerator, Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>> Pow2ModKernelCache = new();
    private static PrimeOrderGpuCapability s_capability = PrimeOrderGpuCapability.Default;

    internal static ConcurrentDictionary<ulong, byte> OverflowRegistry => OverflowedPrimes;

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
        for (int i = 0; i < exponents.Length; i++)
        {
            results[i] = Pow2ModCpu(exponents[i], prime);
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

    internal readonly record struct PrimeOrderGpuCapability(int ModulusBits, int ExponentBits)
    {
        public static PrimeOrderGpuCapability Default => new(128, 64);
    }
}
