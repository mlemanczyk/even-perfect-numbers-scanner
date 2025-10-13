using System;
using System.Buffers;
using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

internal sealed class MersenneNumberDivisorCandidateGpuEvaluator : IDisposable
{
    private const int RemainderCount = 6;

    private readonly GpuContextPool.GpuContextLease _lease;
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, CandidateFilterArgs, ArrayView<ulong>, ArrayView<byte>> _kernel;
    private MemoryBuffer1D<ulong, Stride1D.Dense> _candidateBuffer;
    private MemoryBuffer1D<byte, Stride1D.Dense> _maskBuffer;
    private ulong[] _hostCandidates;
    private byte[] _hostMask;
    private int _capacity;
    private bool _disposed;

    internal MersenneNumberDivisorCandidateGpuEvaluator(int batchSize)
    {
        _lease = GpuContextPool.RentPreferred(preferCpu: false);
        _accelerator = _lease.Accelerator;
        _kernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, CandidateFilterArgs, ArrayView<ulong>, ArrayView<byte>>(EvaluateCandidatesKernel);
        _capacity = Math.Max(1, batchSize);
        _candidateBuffer = _accelerator.Allocate1D<ulong>(_capacity);
        _maskBuffer = _accelerator.Allocate1D<byte>(_capacity);
        _hostCandidates = ArrayPool<ulong>.Shared.Rent(_capacity);
        _hostMask = ArrayPool<byte>.Shared.Rent(_capacity);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _candidateBuffer.Dispose();
        _maskBuffer.Dispose();
        ArrayPool<ulong>.Shared.Return(_hostCandidates, clearArray: false);
        ArrayPool<byte>.Shared.Return(_hostMask, clearArray: false);
        _lease.Dispose();
    }

    internal void EvaluateCandidates(
        ulong startDivisor,
        ulong step,
        ulong limit,
        ReadOnlySpan<byte> remainders,
        ReadOnlySpan<byte> steps,
        bool lastIsSeven,
        Span<ulong> candidates,
        Span<byte> mask)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(MersenneNumberDivisorCandidateGpuEvaluator));
        }

        if (remainders.Length != RemainderCount || steps.Length != RemainderCount)
        {
            throw new ArgumentException("Remainder and step tables must contain exactly six entries.");
        }

        int count = candidates.Length;
        // Candidate batches always carry at least one divisor on the by-divisor path, so the empty-count
        // guard remains commented out to eliminate the branch.
        // if (count == 0)
        // {
        //     return;
        // }

        EnsureCapacity(count);

        Span<ulong> hostCandidateSpan = _hostCandidates.AsSpan(0, count);
        Span<byte> hostMaskSpan = _hostMask.AsSpan(0, count);

        ArrayView1D<ulong, Stride1D.Dense> candidateView = _candidateBuffer.View.SubView(0, count);
        ArrayView1D<byte, Stride1D.Dense> maskView = _maskBuffer.View.SubView(0, count);

        var args = new CandidateFilterArgs(
            startDivisor,
            step,
            limit,
            remainders[0],
            remainders[1],
            remainders[2],
            remainders[3],
            remainders[4],
            remainders[5],
            steps[0],
            steps[1],
            steps[2],
            steps[3],
            steps[4],
            steps[5],
            lastIsSeven ? (byte)1 : (byte)0);

        _kernel(count, args, candidateView, maskView);

        candidateView.CopyToCPU(ref MemoryMarshal.GetReference(hostCandidateSpan), count);
        maskView.CopyToCPU(ref MemoryMarshal.GetReference(hostMaskSpan), count);

        hostCandidateSpan.CopyTo(candidates);
        hostMaskSpan.CopyTo(mask);
    }

    private void EnsureCapacity(int required)
    {
        if (required <= _capacity)
        {
            return;
        }

        int newCapacity = _capacity;
        // The evaluator always starts with a positive capacity, so the zero-capacity fallback remains commented.
        // if (newCapacity == 0)
        // {
        //     newCapacity = 1;
        // }
        while (newCapacity < required)
        {
            newCapacity <<= 1;
        }

        _candidateBuffer.Dispose();
        _maskBuffer.Dispose();
        ArrayPool<ulong>.Shared.Return(_hostCandidates, clearArray: false);
        ArrayPool<byte>.Shared.Return(_hostMask, clearArray: false);

        _capacity = newCapacity;
        _candidateBuffer = _accelerator.Allocate1D<ulong>(_capacity);
        _maskBuffer = _accelerator.Allocate1D<byte>(_capacity);
        _hostCandidates = ArrayPool<ulong>.Shared.Rent(_capacity);
        _hostMask = ArrayPool<byte>.Shared.Rent(_capacity);
    }

    private static void EvaluateCandidatesKernel(
        Index1D index,
        CandidateFilterArgs args,
        ArrayView<ulong> candidates,
        ArrayView<byte> mask)
    {
        int idx = index;
        if (idx >= mask.Length)
        {
            return;
        }

        ulong offset = (ulong)idx;
        ulong candidate = args.StartDivisor + args.Step * offset;
        candidates[idx] = candidate;

        if (candidate > args.Limit || (args.Step != 0UL && candidate < args.StartDivisor))
        {
            mask[idx] = 0;
            return;
        }

        int remainderValue10 = ComputeRemainder(args.Remainder10, args.Step10, idx, 10);
        int remainderValue8 = ComputeRemainder(args.Remainder8, args.Step8, idx, 8);
        int remainderValue5 = ComputeRemainder(args.Remainder5, args.Step5, idx, 5);
        int remainderValue3 = ComputeRemainder(args.Remainder3, args.Step3, idx, 3);
        int remainderValue7 = ComputeRemainder(args.Remainder7, args.Step7, idx, 7);
        int remainderValue11 = ComputeRemainder(args.Remainder11, args.Step11, idx, 11);

        bool lastIsSeven = args.LastIsSevenFlag != 0;
        bool admissible = lastIsSeven
            ? (remainderValue10 == 3 || remainderValue10 == 7 || remainderValue10 == 9)
            : (remainderValue10 == 1 || remainderValue10 == 3 || remainderValue10 == 9);

        if (!admissible)
        {
            mask[idx] = 0;
            return;
        }

        if ((remainderValue8 != 1 && remainderValue8 != 7)
            || remainderValue3 == 0
            || remainderValue5 == 0
            || remainderValue7 == 0
            || remainderValue11 == 0)
        {
            mask[idx] = 0;
            return;
        }

        mask[idx] = 1;
    }

    private static int ComputeRemainder(byte remainder, byte step, int index, int modulus)
    {
        // Modulus lookups only cover {10, 8, 5, 3, 7, 11}, so the zero-modulus guard stays commented out.
        // if (modulus == 0)
        // {
        //     return 0;
        // }

        long value = remainder;
        value += (long)step * index;
        int result = (int)(value % modulus);
        if (result < 0)
        {
            result += modulus;
        }

        return result;
    }

    private readonly struct CandidateFilterArgs
    {
        public CandidateFilterArgs(
            ulong startDivisor,
            ulong step,
            ulong limit,
            byte remainder10,
            byte remainder8,
            byte remainder5,
            byte remainder3,
            byte remainder7,
            byte remainder11,
            byte step10,
            byte step8,
            byte step5,
            byte step3,
            byte step7,
            byte step11,
            byte lastIsSevenFlag)
        {
            StartDivisor = startDivisor;
            Step = step;
            Limit = limit;
            Remainder10 = remainder10;
            Remainder8 = remainder8;
            Remainder5 = remainder5;
            Remainder3 = remainder3;
            Remainder7 = remainder7;
            Remainder11 = remainder11;
            Step10 = step10;
            Step8 = step8;
            Step5 = step5;
            Step3 = step3;
            Step7 = step7;
            Step11 = step11;
            LastIsSevenFlag = lastIsSevenFlag;
        }

        public ulong StartDivisor { get; }
        public ulong Step { get; }
        public ulong Limit { get; }
        public byte Remainder10 { get; }
        public byte Remainder8 { get; }
        public byte Remainder5 { get; }
        public byte Remainder3 { get; }
        public byte Remainder7 { get; }
        public byte Remainder11 { get; }
        public byte Step10 { get; }
        public byte Step8 { get; }
        public byte Step5 { get; }
        public byte Step3 { get; }
        public byte Step7 { get; }
        public byte Step11 { get; }
        public byte LastIsSevenFlag { get; }
    }
}
