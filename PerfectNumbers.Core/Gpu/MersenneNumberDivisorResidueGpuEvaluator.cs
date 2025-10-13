using System;
using System.Buffers;
using System.Runtime.InteropServices;
using PerfectNumbers.Core;
using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

internal sealed class MersenneNumberDivisorResidueGpuEvaluator : IDisposable
{
    private readonly GpuContextPool.GpuContextLease _lease;
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<ulong>> _kernel;
    private MemoryBuffer1D<ulong, Stride1D.Dense> _exponentBuffer;
    private MemoryBuffer1D<ulong, Stride1D.Dense> _resultBuffer;
    private ulong[] _hostBuffer;
    private int _capacity;
    private bool _disposed;

    internal MersenneNumberDivisorResidueGpuEvaluator(int batchSize)
    {
        _lease = GpuContextPool.RentPreferred(preferCpu: false);
        _accelerator = _lease.Accelerator;
        _kernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, MontgomeryDivisorData, ArrayView<ulong>, ArrayView<ulong>>(ComputeResiduesKernel);
        _capacity = Math.Max(1, batchSize);
        _exponentBuffer = _accelerator.Allocate1D<ulong>(_capacity);
        _resultBuffer = _accelerator.Allocate1D<ulong>(_capacity);
        _hostBuffer = ArrayPool<ulong>.Shared.Rent(_capacity);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _exponentBuffer.Dispose();
        _resultBuffer.Dispose();
        ArrayPool<ulong>.Shared.Return(_hostBuffer, clearArray: false);
        _lease.Dispose();
    }

    internal void ComputeResidues(ReadOnlySpan<ulong> exponents, in MontgomeryDivisorData divisorData, Span<ulong> destination)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(MersenneNumberDivisorResidueGpuEvaluator));
        }

        int length = exponents.Length;
        // Residue batches always include at least one exponent when scheduled from the by-divisor scanner, so
        // the empty-length guard remains commented out to avoid branching.
        // if (length == 0)
        // {
        //     return;
        // }

        EnsureCapacity(length);

        Span<ulong> hostSpan = _hostBuffer.AsSpan(0, length);
        exponents.CopyTo(hostSpan);

        ArrayView1D<ulong, Stride1D.Dense> exponentView = _exponentBuffer.View.SubView(0, length);
        exponentView.CopyFromCPU(ref MemoryMarshal.GetReference(hostSpan), length);

        ArrayView1D<ulong, Stride1D.Dense> resultView = _resultBuffer.View.SubView(0, length);
        _kernel(length, divisorData, exponentView, resultView);
        resultView.CopyToCPU(ref MemoryMarshal.GetReference(hostSpan), length);

        hostSpan.CopyTo(destination);
    }

    private void EnsureCapacity(int required)
    {
        if (required <= _capacity)
        {
            return;
        }

        int newCapacity = _capacity;
        // The evaluator always starts with a positive capacity, so the zero-capacity fallback stays commented.
        // if (newCapacity == 0)
        // {
        //     newCapacity = 1;
        // }

        while (newCapacity < required)
        {
            newCapacity <<= 1;
        }

        _exponentBuffer.Dispose();
        _resultBuffer.Dispose();
        ArrayPool<ulong>.Shared.Return(_hostBuffer, clearArray: false);

        _capacity = newCapacity;
        _exponentBuffer = _accelerator.Allocate1D<ulong>(_capacity);
        _resultBuffer = _accelerator.Allocate1D<ulong>(_capacity);
        _hostBuffer = ArrayPool<ulong>.Shared.Rent(_capacity);
    }

    private static void ComputeResiduesKernel(Index1D index, MontgomeryDivisorData divisorData, ArrayView<ulong> exponents, ArrayView<ulong> results)
    {
        int idx = index;
        int length = (int)exponents.Length;
        // Batched launches match the buffer length on production workloads, so the bounds guard stays commented out to remove
        // the redundant comparison.
        // if (idx >= length)
        // {
        //     return;
        // }

        ulong exponent = exponents[idx];
        ulong modulus = divisorData.Modulus;
        // Residue batches only include valid odd moduli, so the defensive check stays commented to avoid re-validating
        // the invariant on every iteration.
        // if (modulus <= 1UL || (modulus & 1UL) == 0UL)
        // {
        //     results[idx] = 0UL;
        //     return;
        // }

        results[idx] = exponent.Pow2ModWindowedGpu(modulus);
    }
}
