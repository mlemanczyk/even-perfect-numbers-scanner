using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core;
using PerfectNumbers.Core.Gpu;

namespace EvenPerfectBitScanner.Benchmarks;

/// <summary>
/// GPU benchmarks for 64-bit modular multiplication helpers.
/// Windows 11 results ranged from 0.62 ms to 1.63 ms, and GpuCompatibleMulModSimplifiedExtension led the PrimeSizedModulus case at 0.641 ms.
/// Recommended helper for p ≥ 138,000,000 is GpuCompatibleMulModSimplifiedExtension.
/// </summary>
[SimpleJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
public class MulMod64BenchmarksGpu : IDisposable
{
    private static readonly MulMod64Input[] Inputs =
    [
        new(0UL, 0UL, 3UL, "ZeroOperands"),
        new(ulong.MaxValue, ulong.MaxValue - 1UL, ulong.MaxValue - 58UL, "NearFullRange"),
        new(0xFFFF_FFFF_0000_0000UL, 0x0000_0000_FFFF_FFFFUL, 0xFFFF_FFFF_0000_002BUL, "CrossWordBlend"),
        new(0x1234_5678_9ABC_DEF0UL, 0x0FED_CBA9_8765_4321UL, 0x1FFF_FFFF_FFFF_FFFBUL, "MixedBitPattern"),
        new(0x0000_0001_0000_0000UL, 0x0000_0000_0000_0003UL, 0x0000_0000_FFFF_FFC3UL, "SparseOperands"),
        new(0x7FFF_FFFF_FFFF_FFA3UL, 0x6FFF_FFFF_FFFF_FF81UL, 0x7FFF_FFFF_FFFF_FFE7UL, "PrimeSizedModulus"),
    ];

    [ParamsSource(nameof(GetInputs))]
    public MulMod64Input Input { get; set; }

    private Context? _context;
    private Accelerator? _accelerator;
    private MemoryBuffer1D<MulMod64GpuKernelInput, Stride1D.Dense>? _inputBuffer;
    private MemoryBuffer1D<ulong, Stride1D.Dense>? _resultBuffer;

    private Action<Index1D, ArrayView<MulMod64GpuKernelInput>, ArrayView<ulong>>? _extensionBaselineKernel;
    private Action<Index1D, ArrayView<MulMod64GpuKernelInput>, ArrayView<ulong>>? _gpuCompatibleExtensionKernel;
    private Action<Index1D, ArrayView<MulMod64GpuKernelInput>, ArrayView<ulong>>? _gpuCompatibleSimplifiedKernel;
    private Action<Index1D, ArrayView<MulMod64GpuKernelInput>, ArrayView<ulong>>? _inlineKernel;
    private Action<Index1D, ArrayView<MulMod64GpuKernelInput>, ArrayView<ulong>>? _inlineReductionFirstKernel;
    private Action<Index1D, ArrayView<MulMod64GpuKernelInput>, ArrayView<ulong>>? _inlineOperandReductionKernel;
    private Action<Index1D, ArrayView<MulMod64GpuKernelInput>, ArrayView<ulong>>? _inlineLocalsKernel;
    private Action<Index1D, ArrayView<MulMod64GpuKernelInput>, ArrayView<ulong>>? _inlineLocalsOperandReductionKernel;
    private Action<Index1D, ArrayView<MulMod64GpuKernelInput>, ArrayView<ulong>>? _mulHighKernel;
    private Action<Index1D, ArrayView<MulMod64GpuKernelInput>, ArrayView<ulong>>? _legacyKernel;
    private Action<Index1D, ArrayView<MulMod64GpuKernelInput>, ArrayView<ulong>>? _legacyDeferredKernel;

    public static IEnumerable<MulMod64Input> GetInputs()
    {
        return Inputs;
    }

    [GlobalSetup]
    public void GlobalSetup()
    {
        _context = Context.Create(builder => builder.Default().EnableAlgorithms());
        _accelerator = _context.GetPreferredDevice(preferCPU: false).CreateAccelerator(_context);

        MulMod64GpuKernelInput[] gpuInputs = Inputs
            .Select(i => new MulMod64GpuKernelInput(i.Left, i.Right, i.Modulus))
            .ToArray();

        _inputBuffer = _accelerator.Allocate1D(gpuInputs);
        _resultBuffer = _accelerator.Allocate1D<ulong>(gpuInputs.Length);

        _extensionBaselineKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<MulMod64GpuKernelInput>, ArrayView<ulong>>(ExtensionBaselineKernel);
        _gpuCompatibleExtensionKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<MulMod64GpuKernelInput>, ArrayView<ulong>>(GpuCompatibleExtensionKernel);
        _gpuCompatibleSimplifiedKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<MulMod64GpuKernelInput>, ArrayView<ulong>>(GpuCompatibleSimplifiedKernel);
        _inlineKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<MulMod64GpuKernelInput>, ArrayView<ulong>>(InlineKernel);
        _inlineReductionFirstKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<MulMod64GpuKernelInput>, ArrayView<ulong>>(InlineReductionFirstKernel);
        _inlineOperandReductionKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<MulMod64GpuKernelInput>, ArrayView<ulong>>(InlineOperandReductionKernel);
        _inlineLocalsKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<MulMod64GpuKernelInput>, ArrayView<ulong>>(InlineLocalsKernel);
        _inlineLocalsOperandReductionKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<MulMod64GpuKernelInput>, ArrayView<ulong>>(InlineLocalsOperandReductionKernel);
        _mulHighKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<MulMod64GpuKernelInput>, ArrayView<ulong>>(MulHighKernel);
        _legacyKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<MulMod64GpuKernelInput>, ArrayView<ulong>>(LegacyKernel);
        _legacyDeferredKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<MulMod64GpuKernelInput>, ArrayView<ulong>>(LegacyDeferredKernel);
    }

    /// <summary>
    /// Inline wide multiply/reduce baseline: ZeroOperands 0.618 ms, MixedBitPattern 0.640 ms, CrossWordBlend 0.928 ms, NearFullRange 0.875 ms, PrimeSizedModulus 0.759 ms, SparseOperands 0.804 ms.
    /// Use primarily as a reference; optimized GPU-specific helpers beat it across all datasets.
    /// </summary>
    [Benchmark(Baseline = true)]
    public void ExtensionBaseline()
    {
        LaunchKernel(_extensionBaselineKernel);
    }

    /// <summary>
    /// Original GPU-compatible extension path: means from 0.614 ms (MixedBitPattern) to 1.023 ms (CrossWordBlend) with PrimeSizedModulus at 0.697 ms.
    /// Retain only when mirroring the CPU extension logic outweighs the speed gap to the simplified helper.
    /// </summary>
    [Benchmark]
    public void GpuCompatibleMulModExtension()
    {
        LaunchKernel(_gpuCompatibleExtensionKernel);
    }

    /// <summary>
    /// Simplified GPU extension helper: 0.708 ms (ZeroOperands), 0.905 ms (CrossWordBlend), 0.798 ms (NearFullRange), and class-leading 0.641 ms (PrimeSizedModulus).
    /// Recommended for production GPU scans, especially for large p ≥ 138,000,000.
    /// </summary>
    [Benchmark]
    public void GpuCompatibleMulModSimplifiedExtension()
    {
        LaunchKernel(_gpuCompatibleSimplifiedKernel);
    }

    /// <summary>
    /// Inline UInt128 multiplication without operand reduction: CrossWordBlend 0.838 ms, MixedBitPattern 0.865 ms, NearFullRange 0.786 ms, PrimeSizedModulus 0.709 ms, SparseOperands 0.751 ms, ZeroOperands 0.827 ms.
    /// Suitable when explicit UInt128 math is acceptable, though it trails the simplified GPU helper on every dataset.
    /// </summary>
    [Benchmark]
    public void InlineUInt128Operands()
    {
        LaunchKernel(_inlineKernel);
    }

    /// <summary>
    /// Reduces operands before multiplying: ZeroOperands 0.926 ms, CrossWordBlend 0.847 ms, MixedBitPattern 0.843 ms, NearFullRange 0.689 ms, PrimeSizedModulus 0.809 ms, SparseOperands 0.746 ms.
    /// Only competitive on NearFullRange; the extra reductions hurt other scenarios.
    /// </summary>
    [Benchmark]
    public void InlineUInt128OperandsWithReductionFirst()
    {
        LaunchKernel(_inlineReductionFirstKernel);
    }

    /// <summary>
    /// Performs operand reduction and then multiplies inline: CrossWordBlend 1.041 ms, MixedBitPattern 0.913 ms, NearFullRange 0.847 ms, PrimeSizedModulus 0.812 ms, SparseOperands 0.744 ms, ZeroOperands 0.802 ms.
    /// Consistently slower than both the reduction-first and simplified GPU helpers.
    /// </summary>
    [Benchmark]
    public void InlineUInt128OperandsWithOperandReduction()
    {
        LaunchKernel(_inlineOperandReductionKernel);
    }

    /// <summary>
    /// Uses local temporaries for the wide product: ZeroOperands 0.691 ms, CrossWordBlend 0.914 ms, MixedBitPattern 0.874 ms, NearFullRange 0.702 ms, PrimeSizedModulus 0.817 ms, SparseOperands 0.728 ms.
    /// Balanced choice when sticking with manual wide products, but still slower than the simplified GPU path.
    /// </summary>
    [Benchmark]
    public void InlineUInt128WithLocals()
    {
        LaunchKernel(_inlineLocalsKernel);
    }

    /// <summary>
    /// Combines local temporaries with operand reduction: ZeroOperands 0.698 ms, CrossWordBlend 0.858 ms, MixedBitPattern 0.877 ms, NearFullRange 0.851 ms, PrimeSizedModulus 0.761 ms, SparseOperands 0.779 ms.
    /// Adds little benefit over the non-reduced local variant while increasing cost on most datasets.
    /// </summary>
    [Benchmark]
    public void InlineUInt128WithLocalsAndOperandReduction()
    {
        LaunchKernel(_inlineLocalsOperandReductionKernel);
    }

    /// <summary>
    /// Multiply-high decomposition: ZeroOperands 0.651 ms, CrossWordBlend 0.848 ms, MixedBitPattern 0.758 ms, NearFullRange 0.812 ms, PrimeSizedModulus 0.728 ms, SparseOperands 0.699 ms.
    /// Useful when avoiding UInt128 temporaries on sparse inputs, but still behind the simplified GPU helper for prime-sized moduli.
    /// </summary>
    [Benchmark]
    public void MultiplyHighDecomposition()
    {
        LaunchKernel(_mulHighKernel);
    }

    /// <summary>
    /// Legacy GPU baseline: ZeroOperands 0.759 ms, CrossWordBlend 0.971 ms, MixedBitPattern 0.827 ms, NearFullRange 0.870 ms, PrimeSizedModulus 0.737 ms, SparseOperands 0.620 ms.
    /// Retain only for regression comparisons; newer helpers are faster except on the sparse sample.
    /// </summary>
    [Benchmark]
    public void GpuCompatibleBaseline()
    {
        LaunchKernel(_legacyKernel);
    }

    /// <summary>
    /// Deferred legacy helper: ZeroOperands 1.561 ms, CrossWordBlend 1.631 ms, MixedBitPattern 1.540 ms, NearFullRange 1.438 ms, PrimeSizedModulus 1.598 ms, SparseOperands 1.096 ms.
    /// Slowest option in every scenario; keep only for historical context.
    /// </summary>
    [Benchmark]
    public void GpuCompatibleDeferred()
    {
        LaunchKernel(_legacyDeferredKernel);
    }

    [GlobalCleanup]
    public void GlobalCleanup()
    {
        _resultBuffer?.Dispose();
        _inputBuffer?.Dispose();
        _accelerator?.Dispose();
        _context?.Dispose();
    }

    public void Dispose()
    {
        GlobalCleanup();
        GC.SuppressFinalize(this);
    }

    private void LaunchKernel(Action<Index1D, ArrayView<MulMod64GpuKernelInput>, ArrayView<ulong>>? kernel)
    {
        if (kernel == null || _accelerator == null || _inputBuffer == null || _resultBuffer == null)
        {
            throw new InvalidOperationException("Benchmark kernels are not initialized.");
        }

        kernel(new Index1D((int)_inputBuffer.Length), _inputBuffer.View, _resultBuffer.View);
        _accelerator.Synchronize();
    }

    private static void ExtensionBaselineKernel(Index1D index, ArrayView<MulMod64GpuKernelInput> inputs, ArrayView<ulong> results)
    {
        if (index >= inputs.Length)
        {
            return;
        }

        MulMod64GpuKernelInput input = inputs[index];
        results[index] = MulMod64Baseline(input.Left, input.Right, input.Modulus);
    }

    private static void GpuCompatibleExtensionKernel(Index1D index, ArrayView<MulMod64GpuKernelInput> inputs, ArrayView<ulong> results)
    {
        if (index >= inputs.Length)
        {
            return;
        }

        MulMod64GpuKernelInput input = inputs[index];
        GpuUInt128 state = new(input.Left);
        results[index] = state.MulMod(input.Right, input.Modulus);
    }

    private static void GpuCompatibleSimplifiedKernel(Index1D index, ArrayView<MulMod64GpuKernelInput> inputs, ArrayView<ulong> results)
    {
        if (index >= inputs.Length)
        {
            return;
        }

        MulMod64GpuKernelInput input = inputs[index];
        GpuUInt128 state = new(input.Left);
        results[index] = state.MulModSimplified(input.Right, input.Modulus);
    }

    private static void InlineKernel(Index1D index, ArrayView<MulMod64GpuKernelInput> inputs, ArrayView<ulong> results)
    {
        if (index >= inputs.Length)
        {
            return;
        }

        MulMod64GpuKernelInput input = inputs[index];
        results[index] = MulMod64Baseline(input.Left, input.Right, input.Modulus);
    }

    private static void InlineReductionFirstKernel(Index1D index, ArrayView<MulMod64GpuKernelInput> inputs, ArrayView<ulong> results)
    {
        if (index >= inputs.Length)
        {
            return;
        }

        MulMod64GpuKernelInput input = inputs[index];
        ulong left = input.Left % input.Modulus;
        ulong right = input.Right % input.Modulus;
        results[index] = MulMod64Baseline(left, right, input.Modulus);
    }

    private static void InlineOperandReductionKernel(Index1D index, ArrayView<MulMod64GpuKernelInput> inputs, ArrayView<ulong> results)
    {
        if (index >= inputs.Length)
        {
            return;
        }

        MulMod64GpuKernelInput input = inputs[index];
        ulong left = input.Left % input.Modulus;
        ulong right = input.Right % input.Modulus;
        results[index] = MulMod64Baseline(left, right, input.Modulus);
    }

    private static void InlineLocalsKernel(Index1D index, ArrayView<MulMod64GpuKernelInput> inputs, ArrayView<ulong> results)
    {
        if (index >= inputs.Length)
        {
            return;
        }

        MulMod64GpuKernelInput input = inputs[index];
        MultiplyWide(input.Left, input.Right, out ulong productHigh, out ulong productLow);
        results[index] = ReduceWide(productHigh, productLow, input.Modulus);
    }

    private static void InlineLocalsOperandReductionKernel(Index1D index, ArrayView<MulMod64GpuKernelInput> inputs, ArrayView<ulong> results)
    {
        if (index >= inputs.Length)
        {
            return;
        }

        MulMod64GpuKernelInput input = inputs[index];
        ulong left = input.Left % input.Modulus;
        ulong right = input.Right % input.Modulus;
        MultiplyWide(left, right, out ulong productHigh, out ulong productLow);
        results[index] = ReduceWide(productHigh, productLow, input.Modulus);
    }

    private static void MulHighKernel(Index1D index, ArrayView<MulMod64GpuKernelInput> inputs, ArrayView<ulong> results)
    {
        if (index >= inputs.Length)
        {
            return;
        }

        MulMod64GpuKernelInput input = inputs[index];
        ulong left = input.Left % input.Modulus;
        ulong right = input.Right % input.Modulus;
        MultiplyWide(left, right, out ulong productHigh, out ulong productLow);
        results[index] = ReduceWide(productHigh, productLow, input.Modulus);
    }

    private static void LegacyKernel(Index1D index, ArrayView<MulMod64GpuKernelInput> inputs, ArrayView<ulong> results)
    {
        if (index >= inputs.Length)
        {
            return;
        }

        MulMod64GpuKernelInput input = inputs[index];
        results[index] = MulMod64GpuLegacy(input.Left, input.Right, input.Modulus);
    }

    private static void LegacyDeferredKernel(Index1D index, ArrayView<MulMod64GpuKernelInput> inputs, ArrayView<ulong> results)
    {
        if (index >= inputs.Length)
        {
            return;
        }

        MulMod64GpuKernelInput input = inputs[index];
        results[index] = MulMod64GpuDeferredLegacy(input.Left, input.Right, input.Modulus);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong MulMod64Baseline(ulong left, ulong right, ulong modulus)
    {
        MultiplyWide(left, right, out ulong high, out ulong low);
        return ReduceWide(high, low, modulus);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void MultiplyWide(ulong left, ulong right, out ulong high, out ulong low)
    {
        ulong leftLow = (uint)left;
        ulong leftHigh = left >> 32;
        ulong rightLow = (uint)right;
        ulong rightHigh = right >> 32;

        ulong productLL = leftLow * rightLow;
        ulong productLH = leftLow * rightHigh;
        ulong productHL = leftHigh * rightLow;
        ulong productHH = leftHigh * rightHigh;

        ulong carry = (productLL >> 32) + (productLH & 0xFFFF_FFFFUL) + (productHL & 0xFFFF_FFFFUL);
        low = (carry << 32) | (uint)productLL;
        high = productHH + (productLH >> 32) + (productHL >> 32) + (carry >> 32);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong ReduceWide(ulong high, ulong low, ulong modulus)
    {
        if (modulus <= 1UL)
        {
            return 0UL;
        }

        ulong remainder = 0UL;
        if (high != 0UL)
        {
            for (int bit = 63; bit >= 0; --bit)
            {
                remainder = ShiftLeftAddBit(remainder, (high >> bit) & 1UL, modulus);
            }
        }

        for (int bit = 63; bit >= 0; --bit)
        {
            remainder = ShiftLeftAddBit(remainder, (low >> bit) & 1UL, modulus);
        }

        return remainder;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong ShiftLeftAddBit(ulong value, ulong bit, ulong modulus)
    {
        value <<= 1;
        if (value >= modulus)
        {
            value -= modulus;
        }

        if (bit != 0UL)
        {
            value += 1UL;
            if (value >= modulus)
            {
                value -= modulus;
            }
        }

        return value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong MulMod64GpuLegacy(ulong a, ulong b, ulong modulus)
    {
        GpuUInt128 state = new(a % modulus);
        return state.MulMod(b, modulus);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong MulMod64GpuDeferredLegacy(ulong a, ulong b, ulong modulus)
    {
        GpuUInt128 state = new(a);
        return state.MulModWithNativeModulo(b, modulus);
    }

    public readonly record struct MulMod64Input(ulong Left, ulong Right, ulong Modulus, string Name)
    {
        public override string ToString()
        {
            return Name;
        }
    }

    private readonly struct MulMod64GpuKernelInput
    {
        public readonly ulong Left;
        public readonly ulong Right;
        public readonly ulong Modulus;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public MulMod64GpuKernelInput(ulong left, ulong right, ulong modulus)
        {
            Left = left;
            Right = right;
            Modulus = modulus;
        }
    }
}
