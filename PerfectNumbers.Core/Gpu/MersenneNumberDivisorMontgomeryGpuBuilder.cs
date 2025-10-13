using System;
using System.Buffers;
using System.Runtime.InteropServices;
using PerfectNumbers.Core;
using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

internal sealed class MersenneNumberDivisorMontgomeryGpuBuilder : IDisposable
{
    private readonly GpuContextPool.GpuContextLease _lease;
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<ulong>, ArrayView<MontgomeryDivisorData>> _kernel;
    private MemoryBuffer1D<ulong, Stride1D.Dense> _modulusBuffer;
    private MemoryBuffer1D<MontgomeryDivisorData, Stride1D.Dense> _resultBuffer;
    private ulong[] _hostModulusBuffer;
    private MontgomeryDivisorData[] _hostResultBuffer;
    private int _capacity;
    private bool _disposed;

    internal MersenneNumberDivisorMontgomeryGpuBuilder(int batchSize)
    {
        _lease = GpuContextPool.RentPreferred(preferCpu: false);
        _accelerator = _lease.Accelerator;
        _kernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ArrayView<MontgomeryDivisorData>>(BuildKernel);
        _capacity = Math.Max(1, batchSize);
        _modulusBuffer = _accelerator.Allocate1D<ulong>(_capacity);
        _resultBuffer = _accelerator.Allocate1D<MontgomeryDivisorData>(_capacity);
        _hostModulusBuffer = ArrayPool<ulong>.Shared.Rent(_capacity);
        _hostResultBuffer = ArrayPool<MontgomeryDivisorData>.Shared.Rent(_capacity);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _modulusBuffer.Dispose();
        _resultBuffer.Dispose();
        ArrayPool<ulong>.Shared.Return(_hostModulusBuffer, clearArray: false);
        ArrayPool<MontgomeryDivisorData>.Shared.Return(_hostResultBuffer, clearArray: false);
        _lease.Dispose();
    }

    internal void Build(ReadOnlySpan<ulong> moduli, Span<MontgomeryDivisorData> destination)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(MersenneNumberDivisorMontgomeryGpuBuilder));
        }

        int length = moduli.Length;
        if (length == 0)
        {
            return;
        }

        EnsureCapacity(length);

        Span<ulong> hostModulusSpan = _hostModulusBuffer.AsSpan(0, length);
        moduli.CopyTo(hostModulusSpan);

        ArrayView1D<ulong, Stride1D.Dense> modulusView = _modulusBuffer.View.SubView(0, length);
        modulusView.CopyFromCPU(ref MemoryMarshal.GetReference(hostModulusSpan), length);

        ArrayView1D<MontgomeryDivisorData, Stride1D.Dense> resultView = _resultBuffer.View.SubView(0, length);
        _kernel(length, modulusView, resultView);

        Span<MontgomeryDivisorData> hostResultSpan = _hostResultBuffer.AsSpan(0, length);
        resultView.CopyToCPU(ref MemoryMarshal.GetReference(hostResultSpan), length);

        hostResultSpan.CopyTo(destination);
    }

    private void EnsureCapacity(int required)
    {
        if (required <= _capacity)
        {
            return;
        }

        int newCapacity = _capacity == 0 ? 1 : _capacity;
        while (newCapacity < required)
        {
            newCapacity <<= 1;
        }

        _modulusBuffer.Dispose();
        _resultBuffer.Dispose();
        ArrayPool<ulong>.Shared.Return(_hostModulusBuffer, clearArray: false);
        ArrayPool<MontgomeryDivisorData>.Shared.Return(_hostResultBuffer, clearArray: false);

        _capacity = newCapacity;
        _modulusBuffer = _accelerator.Allocate1D<ulong>(_capacity);
        _resultBuffer = _accelerator.Allocate1D<MontgomeryDivisorData>(_capacity);
        _hostModulusBuffer = ArrayPool<ulong>.Shared.Rent(_capacity);
        _hostResultBuffer = ArrayPool<MontgomeryDivisorData>.Shared.Rent(_capacity);
    }

    private static void BuildKernel(Index1D index, ArrayView<ulong> moduli, ArrayView<MontgomeryDivisorData> destination)
    {
        int idx = index;
        if (idx >= moduli.Length)
        {
            return;
        }

        ulong modulus = moduli[idx];
        if (modulus <= 1UL || (modulus & 1UL) == 0UL)
        {
            destination[idx] = default;
            return;
        }

        ulong nPrime = ComputeMontgomeryNPrime(modulus);
        ulong montgomeryOne = ComputeMontgomeryResidue64(modulus);
        ulong montgomeryTwo = montgomeryOne.MulMod64(2UL, modulus);
        ulong montgomeryTwoSquared = MontgomeryMultiplyGpu(montgomeryTwo, montgomeryTwo, modulus, nPrime);

        destination[idx] = new MontgomeryDivisorData(
            modulus,
            nPrime,
            montgomeryOne,
            montgomeryTwo,
            montgomeryTwoSquared);
    }

    private static ulong ComputeMontgomeryResidue64(ulong modulus)
    {
        ulong remainder = ulong.MaxValue % modulus;
        remainder++;
        if (remainder >= modulus)
        {
            remainder -= modulus;
        }

        return remainder;
    }

    private static ulong MontgomeryMultiplyGpu(ulong a, ulong b, ulong modulus, ulong nPrime)
    {
        GpuUInt128 product = new(a);
        product.Mul64(new GpuUInt128(b));
        ulong tLow = product.Low;
        ulong tHigh = product.High;

        ulong m = unchecked(tLow * nPrime);
        GpuUInt128 mProduct = new(m);
        mProduct.Mul64(new GpuUInt128(modulus));
        ulong mLow = mProduct.Low;
        ulong mHigh = mProduct.High;

        ulong carry = unchecked((tLow + mLow) < tLow ? 1UL : 0UL);
        ulong result = tHigh + mHigh + carry;
        if (result >= modulus)
        {
            result -= modulus;
        }

        return result;
    }

    private static ulong ComputeMontgomeryNPrime(ulong modulus)
    {
        ulong inv = modulus;
        inv *= unchecked(2UL - modulus * inv);
        inv *= unchecked(2UL - modulus * inv);
        inv *= unchecked(2UL - modulus * inv);
        inv *= unchecked(2UL - modulus * inv);
        inv *= unchecked(2UL - modulus * inv);
        inv *= unchecked(2UL - modulus * inv);
        inv *= unchecked(2UL - modulus * inv);
        return unchecked(0UL - inv);
    }
}
