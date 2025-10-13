using System;
using System.Runtime.InteropServices;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

internal static class MontgomeryOddPowerGpu
{
    public static bool TryCompute(in MontgomeryDivisorData divisor, int oddPowerCount, Span<ulong> destination)
    {
        if (oddPowerCount <= 0)
        {
            return true;
        }

        Span<MontgomeryDivisorData> divisorSpan = stackalloc MontgomeryDivisorData[1];
        divisorSpan[0] = divisor;
        Span<int> countSpan = stackalloc int[1];
        countSpan[0] = oddPowerCount;
        Span<ulong> resultSpan = stackalloc ulong[PerfectNumberConstants.MaxOddPowersCount];

        if (!TryComputeBatch(divisorSpan, countSpan, resultSpan, PerfectNumberConstants.MaxOddPowersCount))
        {
            return false;
        }

        resultSpan[..oddPowerCount].CopyTo(destination);
        return true;
    }

    public static bool TryComputeBatch(
        ReadOnlySpan<MontgomeryDivisorData> divisors,
        ReadOnlySpan<int> oddPowerCounts,
        Span<ulong> results,
        int stride)
    {
        int divisorCount = divisors.Length;
        if (divisorCount == 0)
        {
            return true;
        }

        if (oddPowerCounts.Length < divisorCount)
        {
            throw new ArgumentException(
                "oddPowerCounts span must contain at least as many elements as divisors span.",
                nameof(oddPowerCounts));
        }

        if (results.Length < divisorCount * stride)
        {
            throw new ArgumentException(
                "The result span is shorter than the required batched odd power storage.",
                nameof(results));
        }

        if (GpuContextPool.ForceCpu)
        {
            return false;
        }

        var lease = GpuKernelPool.GetKernel(useGpuOrder: true);
        var execution = lease.EnterExecutionScope();
        var accelerator = lease.Accelerator;
        var stream = lease.Stream;

        var divisorBuffer = accelerator.Allocate1D<MontgomeryDivisorData>(divisorCount);
        var countBuffer = accelerator.Allocate1D<int>(divisorCount);
        var resultBuffer = accelerator.Allocate1D<ulong>(divisorCount * stride);

        divisorBuffer.View.CopyFromCPU(ref MemoryMarshal.GetReference(divisors), divisorCount);
        countBuffer.View.CopyFromCPU(ref MemoryMarshal.GetReference(oddPowerCounts), divisorCount);
        resultBuffer.MemSetToZero();

        var kernel = lease.MontgomeryOddPowerKernel;
        kernel(stream, divisorCount, divisorBuffer.View, countBuffer.View, resultBuffer.View, stride);
        stream.Synchronize();

        resultBuffer.View.CopyToCPU(ref MemoryMarshal.GetReference(results), divisorCount * stride);

        resultBuffer.Dispose();
        countBuffer.Dispose();
        divisorBuffer.Dispose();
        execution.Dispose();
        lease.Dispose();
        return true;
    }
}
