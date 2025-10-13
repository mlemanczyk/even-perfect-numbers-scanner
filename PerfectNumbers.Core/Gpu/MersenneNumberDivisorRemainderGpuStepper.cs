using System;
using System.Buffers;
using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

internal sealed class MersenneNumberDivisorRemainderGpuStepper : IDisposable
{
    private const int RemainderCount = 6;

    private readonly GpuContextPool.GpuContextLease _lease;
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, int, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>> _kernel;
    private MemoryBuffer1D<byte, Stride1D.Dense> _remainderBuffer;
    private MemoryBuffer1D<byte, Stride1D.Dense> _stepBuffer;
    private MemoryBuffer1D<byte, Stride1D.Dense> _resultBuffer;
    private byte[] _hostRemainders;
    private byte[] _hostSteps;
    private readonly int _length;
    private int _capacity;
    private bool _disposed;

    internal MersenneNumberDivisorRemainderGpuStepper(ReadOnlySpan<byte> steps)
    {
        if (steps.Length != RemainderCount)
        {
            throw new ArgumentException("Step table must contain exactly six entries.");
        }

        _lease = GpuContextPool.RentPreferred(preferCpu: false);
        _accelerator = _lease.Accelerator;
        _kernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, int, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>>(AdvanceRemaindersKernel);

        _length = steps.Length;
        _capacity = Math.Max(1, _length);
        _remainderBuffer = _accelerator.Allocate1D<byte>(_capacity);
        _stepBuffer = _accelerator.Allocate1D<byte>(_capacity);
        _resultBuffer = _accelerator.Allocate1D<byte>(_capacity);

        _hostRemainders = ArrayPool<byte>.Shared.Rent(_capacity);
        _hostSteps = ArrayPool<byte>.Shared.Rent(_capacity);

        steps.CopyTo(_hostSteps.AsSpan(0, _length));

        ArrayView1D<byte, Stride1D.Dense> stepView = _stepBuffer.View.SubView(0, _length);
        stepView.CopyFromCPU(ref MemoryMarshal.GetReference(_hostSteps.AsSpan(0, _length)), _length);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _remainderBuffer.Dispose();
        _stepBuffer.Dispose();
        _resultBuffer.Dispose();
        ArrayPool<byte>.Shared.Return(_hostRemainders, clearArray: false);
        ArrayPool<byte>.Shared.Return(_hostSteps, clearArray: false);
        _lease.Dispose();
    }

    internal void Advance(int advanceCount, Span<byte> remainders)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(MersenneNumberDivisorRemainderGpuStepper));
        }

        // GPU remainder batches always advance by a positive chunk count and operate on fixed-size tables, so
        // the defensive early-return remains commented out.
        // if (advanceCount <= 0 || remainders.Length == 0)
        // {
        //     return;
        // }

        if (remainders.Length != _length)
        {
            throw new ArgumentException("Remainder buffer length must match the configured length.");
        }

        Span<byte> hostRemainderSpan = _hostRemainders.AsSpan(0, _length);
        remainders.CopyTo(hostRemainderSpan);

        ArrayView1D<byte, Stride1D.Dense> remainderView = _remainderBuffer.View.SubView(0, _length);
        remainderView.CopyFromCPU(ref MemoryMarshal.GetReference(hostRemainderSpan), _length);

        ArrayView1D<byte, Stride1D.Dense> stepView = _stepBuffer.View.SubView(0, _length);
        ArrayView1D<byte, Stride1D.Dense> resultView = _resultBuffer.View.SubView(0, _length);

        _kernel(_length, advanceCount, remainderView, stepView, resultView);

        resultView.CopyToCPU(ref MemoryMarshal.GetReference(hostRemainderSpan), _length);
        hostRemainderSpan.CopyTo(remainders);
    }

    private static void AdvanceRemaindersKernel(Index1D index, int advanceCount, ArrayView<byte> remainders, ArrayView<byte> steps, ArrayView<byte> results)
    {
        int idx = index;
        // Kernel launches match the remainder table length, so the bounds guard stays commented to avoid redundant branching.
        // if (idx >= remainders.Length)
        // {
        //     return;
        // }

        int modulus = GetModulus(idx);
        // The modulus selector only emits positive bases (10, 8, 5, 3, 7, 11) for the configured indices, so the zero-modulus
        // fallback remains commented out.
        // if (modulus == 0)
        // {
        //     results[idx] = remainders[idx];
        //     return;
        // }

        int remainder = remainders[idx] % modulus;
        int step = steps[idx] % modulus;

        long delta = (long)step * advanceCount;
        int adjusted = (int)((remainder + delta) % modulus);
        if (adjusted < 0)
        {
            adjusted += modulus;
        }

        results[idx] = (byte)adjusted;
    }

    private static int GetModulus(int index)
    {
        return index switch
        {
            0 => 10,
            1 => 8,
            2 => 5,
            3 => 3,
            4 => 7,
            5 => 11,
            _ => 0,
        };
    }
}
