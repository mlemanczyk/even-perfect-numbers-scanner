using System.Reflection;
using FluentAssertions;
using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu;
using Xunit;

namespace PerfectNumbers.Core.Tests.Gpu;

[Collection("GpuNtt")]
public class GpuKernelCompilationTests
{
    private static readonly MethodInfo DivisorCycleCacheLoadKernelMethod =
        typeof(DivisorCycleCache).GetMethod("LoadKernel", BindingFlags.NonPublic | BindingFlags.Static)
        ?? throw new InvalidOperationException("DivisorCycleCache.LoadKernel not found.");

    private static readonly MethodInfo NttGetBitReverseKernelMethod =
        typeof(NttGpuMath).GetMethod("GetBitReverseKernel", BindingFlags.NonPublic | BindingFlags.Static)
        ?? throw new InvalidOperationException("NttGpuMath.GetBitReverseKernel not found.");

    private static readonly MethodInfo NttGetMulKernelMethod =
        typeof(NttGpuMath).GetMethod("GetMulKernel", BindingFlags.NonPublic | BindingFlags.Static)
        ?? throw new InvalidOperationException("NttGpuMath.GetMulKernel not found.");

    private static readonly MethodInfo NttGetStageKernelMethod =
        typeof(NttGpuMath).GetMethod("GetStageKernel", BindingFlags.NonPublic | BindingFlags.Static)
        ?? throw new InvalidOperationException("NttGpuMath.GetStageKernel not found.");

    private static readonly MethodInfo NttGetScaleKernelMethod =
        typeof(NttGpuMath).GetMethod("GetScaleKernel", BindingFlags.NonPublic | BindingFlags.Static)
        ?? throw new InvalidOperationException("NttGpuMath.GetScaleKernel not found.");

    private static readonly MethodInfo NttGetStageMontKernelMethod =
        typeof(NttGpuMath).GetMethod("GetStageMontKernel", BindingFlags.NonPublic | BindingFlags.Static)
        ?? throw new InvalidOperationException("NttGpuMath.GetStageMontKernel not found.");

    private static readonly MethodInfo NttGetStageBarrettKernelMethod =
        typeof(NttGpuMath).GetMethod("GetStageBarrett128Kernel", BindingFlags.NonPublic | BindingFlags.Static)
        ?? throw new InvalidOperationException("NttGpuMath.GetStageBarrett128Kernel not found.");

    private static readonly MethodInfo NttGetScaleBarrettKernelMethod =
        typeof(NttGpuMath).GetMethod("GetScaleBarrett128Kernel", BindingFlags.NonPublic | BindingFlags.Static)
        ?? throw new InvalidOperationException("NttGpuMath.GetScaleBarrett128Kernel not found.");

    private static readonly MethodInfo NttGetSquareBarrettKernelMethod =
        typeof(NttGpuMath).GetMethod("GetSquareBarrett128Kernel", BindingFlags.NonPublic | BindingFlags.Static)
        ?? throw new InvalidOperationException("NttGpuMath.GetSquareBarrett128Kernel not found.");

    private static readonly MethodInfo NttGetToMont64KernelMethod =
        typeof(NttGpuMath).GetMethod("GetToMont64Kernel", BindingFlags.NonPublic | BindingFlags.Static)
        ?? throw new InvalidOperationException("NttGpuMath.GetToMont64Kernel not found.");

    private static readonly MethodInfo NttGetFromMont64KernelMethod =
        typeof(NttGpuMath).GetMethod("GetFromMont64Kernel", BindingFlags.NonPublic | BindingFlags.Static)
        ?? throw new InvalidOperationException("NttGpuMath.GetFromMont64Kernel not found.");

    private static readonly MethodInfo NttGetSquareMont64KernelMethod =
        typeof(NttGpuMath).GetMethod("GetSquareMont64Kernel", BindingFlags.NonPublic | BindingFlags.Static)
        ?? throw new InvalidOperationException("NttGpuMath.GetSquareMont64Kernel not found.");

    private static readonly MethodInfo NttGetScaleMont64KernelMethod =
        typeof(NttGpuMath).GetMethod("GetScaleMont64Kernel", BindingFlags.NonPublic | BindingFlags.Static)
        ?? throw new InvalidOperationException("NttGpuMath.GetScaleMont64Kernel not found.");

    private static readonly MethodInfo NttGetForwardKernelMethod =
        typeof(NttGpuMath).GetMethod("GetForwardKernel", BindingFlags.NonPublic | BindingFlags.Static)
        ?? throw new InvalidOperationException("NttGpuMath.GetForwardKernel not found.");

    private static readonly MethodInfo NttGetInverseKernelMethod =
        typeof(NttGpuMath).GetMethod("GetInverseKernel", BindingFlags.NonPublic | BindingFlags.Static)
        ?? throw new InvalidOperationException("NttGpuMath.GetInverseKernel not found.");

    private static readonly MethodInfo DivisorByDivisorGetKernelMethod =
        typeof(MersenneNumberDivisorByDivisorGpuTester).GetMethod("GetKernel", BindingFlags.NonPublic | BindingFlags.Instance)
        ?? throw new InvalidOperationException("MersenneNumberDivisorByDivisorGpuTester.GetKernel not found.");

    private static readonly MethodInfo DivisorTesterGetKernelMethod =
        typeof(MersenneNumberDivisorGpuTester).GetMethod("GetKernel", BindingFlags.NonPublic | BindingFlags.Instance)
        ?? throw new InvalidOperationException("MersenneNumberDivisorGpuTester.GetKernel not found.");

    private static readonly MethodInfo LucasLehmerGetKernelMethod =
        typeof(MersenneNumberLucasLehmerGpuTester).GetMethod("GetKernel", BindingFlags.NonPublic | BindingFlags.Instance)
        ?? throw new InvalidOperationException("MersenneNumberLucasLehmerGpuTester.GetKernel not found.");

    private static readonly MethodInfo LucasLehmerGetAddSmallKernelMethod =
        typeof(MersenneNumberLucasLehmerGpuTester).GetMethod("GetAddSmallKernel", BindingFlags.NonPublic | BindingFlags.Instance)
        ?? throw new InvalidOperationException("MersenneNumberLucasLehmerGpuTester.GetAddSmallKernel not found.");

    private static readonly MethodInfo LucasLehmerGetSubSmallKernelMethod =
        typeof(MersenneNumberLucasLehmerGpuTester).GetMethod("GetSubSmallKernel", BindingFlags.NonPublic | BindingFlags.Instance)
        ?? throw new InvalidOperationException("MersenneNumberLucasLehmerGpuTester.GetSubSmallKernel not found.");

    private static readonly MethodInfo LucasLehmerGetReduceKernelMethod =
        typeof(MersenneNumberLucasLehmerGpuTester).GetMethod("GetReduceKernel", BindingFlags.NonPublic | BindingFlags.Instance)
        ?? throw new InvalidOperationException("MersenneNumberLucasLehmerGpuTester.GetReduceKernel not found.");

    private static readonly MethodInfo LucasLehmerGetIsZeroKernelMethod =
        typeof(MersenneNumberLucasLehmerGpuTester).GetMethod("GetIsZeroKernel", BindingFlags.NonPublic | BindingFlags.Instance)
        ?? throw new InvalidOperationException("MersenneNumberLucasLehmerGpuTester.GetIsZeroKernel not found.");

    private static readonly MethodInfo LucasLehmerGetBatchKernelMethod =
        typeof(MersenneNumberLucasLehmerGpuTester).GetMethod("GetBatchKernel", BindingFlags.NonPublic | BindingFlags.Instance)
        ?? throw new InvalidOperationException("MersenneNumberLucasLehmerGpuTester.GetBatchKernel not found.");

    private static readonly MethodInfo PrimeOrderGetPartialFactorKernelMethod =
        typeof(PrimeOrderGpuHeuristics).GetMethod("GetPartialFactorKernel", BindingFlags.NonPublic | BindingFlags.Static)
        ?? throw new InvalidOperationException("PrimeOrderGpuHeuristics.GetPartialFactorKernel not found.");

    private static readonly MethodInfo PrimeOrderGetOrderKernelMethod =
        typeof(PrimeOrderGpuHeuristics).GetMethod("GetOrderKernel", BindingFlags.NonPublic | BindingFlags.Static)
        ?? throw new InvalidOperationException("PrimeOrderGpuHeuristics.GetOrderKernel not found.");

    private static readonly MethodInfo PrimeOrderGetPow2ModKernelMethod =
        typeof(PrimeOrderGpuHeuristics).GetMethod("GetPow2ModKernel", BindingFlags.NonPublic | BindingFlags.Static)
        ?? throw new InvalidOperationException("PrimeOrderGpuHeuristics.GetPow2ModKernel not found.");

    private static readonly MethodInfo PrimeOrderGetPow2ModWideKernelMethod =
        typeof(PrimeOrderGpuHeuristics).GetMethod("GetPow2ModWideKernel", BindingFlags.NonPublic | BindingFlags.Static)
        ?? throw new InvalidOperationException("PrimeOrderGpuHeuristics.GetPow2ModWideKernel not found.");

    private static readonly Action<Index1D, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>> MersenneDivisorCyclesKernel =
        DivisorCycleKernels.GpuDivisorCycleKernel;

    private static readonly Action<Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>> PrimeTesterSmallPrimeKernel =
        PrimeTesterKernels.SmallPrimeSieveKernel;

    private static readonly Action<Index1D, ArrayView<ulong>, ArrayView<byte>> PrimeTesterSharesFactorKernel =
        PrimeTesterKernels.SharesFactorKernel;

    public static IEnumerable<object[]> KernelLoaders()
    {
        yield return Loader("DivisorCycleCache.GpuAdvanceDivisorCyclesKernel", CompileDivisorCycleCacheKernel);
        yield return Loader("NttGpuMath.BitReverseKernel", CompileNttBitReverseKernel);
        yield return Loader("NttGpuMath.MulKernel", CompileNttMulKernel);
        yield return Loader("NttGpuMath.StageKernel", CompileNttStageKernel);
        yield return Loader("NttGpuMath.ScaleKernel", CompileNttScaleKernel);
        yield return Loader("NttGpuMath.StageMontKernel", CompileNttStageMontKernel);
        yield return Loader("NttGpuMath.StageBarrett128Kernel", CompileNttStageBarrettKernel);
        yield return Loader("NttGpuMath.ScaleBarrett128Kernel", CompileNttScaleBarrettKernel);
        yield return Loader("NttGpuMath.SquareBarrett128Kernel", CompileNttSquareBarrettKernel);
        yield return Loader("NttGpuMath.ToMont64Kernel", CompileNttToMont64Kernel);
        yield return Loader("NttGpuMath.FromMont64Kernel", CompileNttFromMont64Kernel);
        yield return Loader("NttGpuMath.SquareMont64Kernel", CompileNttSquareMont64Kernel);
        yield return Loader("NttGpuMath.ScaleMont64Kernel", CompileNttScaleMont64Kernel);
        yield return Loader("NttGpuMath.ForwardKernel", CompileNttForwardKernel);
        yield return Loader("NttGpuMath.InverseKernel", CompileNttInverseKernel);
        yield return Loader("MersenneNumberDivisorByDivisorGpuTester.CheckKernel", CompileDivisorByDivisorCheckKernel);
        yield return Loader("MersenneNumberDivisorGpuTester.Kernel", CompileDivisorTesterKernel);
        yield return Loader("MersenneNumberLucasLehmerGpuTester.Kernel", CompileLucasLehmerKernel);
        yield return Loader("MersenneNumberLucasLehmerGpuTester.AddSmallKernel", CompileLucasLehmerAddSmallKernel);
        yield return Loader("MersenneNumberLucasLehmerGpuTester.SubtractSmallKernel", CompileLucasLehmerSubSmallKernel);
        yield return Loader("MersenneNumberLucasLehmerGpuTester.ReduceModMersenneKernel", CompileLucasLehmerReduceKernel);
        yield return Loader("MersenneNumberLucasLehmerGpuTester.IsZeroKernel", CompileLucasLehmerIsZeroKernel);
        yield return Loader("MersenneNumberLucasLehmerGpuTester.KernelBatch", CompileLucasLehmerBatchKernel);
        yield return Loader("PrimeOrderGpuHeuristics.PartialFactorKernel", CompilePrimeOrderPartialFactorKernel);
        yield return Loader("PrimeOrderGpuHeuristics.CalculateOrderKernel", CompilePrimeOrderKernel);
        yield return Loader("PrimeOrderGpuHeuristics.Pow2ModKernel", CompilePrimeOrderPow2ModKernel);
        yield return Loader("PrimeOrderGpuHeuristics.Pow2ModKernelWide", CompilePrimeOrderPow2ModWideKernel);
        yield return Loader("DivisorCycleKernels.GpuDivisorCycleKernel", CompileMersenneDivisorCyclesKernel);
        yield return Loader("PrimeTesterKernels.SmallPrimeSieveKernel", CompilePrimeTesterSmallPrimeKernel);
        yield return Loader("PrimeTesterKernels.SharesFactorKernel", CompilePrimeTesterSharesFactorKernel);
        yield return Loader("GpuKernelPool.KernelLeaseKernels", CompileGpuKernelPoolKernels);
    }

    private static object[] Loader(string name, Action<Accelerator> loader)
    {
        return new object[] { name, loader };
    }

    // [Theory]
    // [Trait("Category", "Fast")]
    // [MemberData(nameof(KernelLoaders))]
    // public void Kernel_compiles_without_internal_compiler_errors(string name, Action<Accelerator> compile)
    // {
    //     GpuContextPool.GpuContextLease lease = GpuContextPool.RentPreferred(preferCpu: true);
    //     try
    //     {
    //         var accelerator = lease.Accelerator;
    //         Action action = () => compile(accelerator);
    //         action.Should().NotThrow($"kernel {name} should compile");
    //     }
    //     finally
    //     {
    //         lease.Dispose();
    //     }
    // }

    private static void CompileDivisorCycleCacheKernel(Accelerator accelerator)
    {
        _ = DivisorCycleCacheLoadKernelMethod.Invoke(null, new object[] { accelerator });
    }

    private static void CompileNttBitReverseKernel(Accelerator accelerator)
    {
        _ = NttGetBitReverseKernelMethod.Invoke(null, new object[] { accelerator });
    }

    private static void CompileNttMulKernel(Accelerator accelerator)
    {
        _ = NttGetMulKernelMethod.Invoke(null, new object[] { accelerator });
    }

    private static void CompileNttStageKernel(Accelerator accelerator)
    {
        _ = NttGetStageKernelMethod.Invoke(null, new object[] { accelerator });
    }

    private static void CompileNttScaleKernel(Accelerator accelerator)
    {
        _ = NttGetScaleKernelMethod.Invoke(null, new object[] { accelerator });
    }

    private static void CompileNttStageMontKernel(Accelerator accelerator)
    {
        _ = NttGetStageMontKernelMethod.Invoke(null, new object[] { accelerator });
    }

    private static void CompileNttStageBarrettKernel(Accelerator accelerator)
    {
        _ = NttGetStageBarrettKernelMethod.Invoke(null, new object[] { accelerator });
    }

    private static void CompileNttScaleBarrettKernel(Accelerator accelerator)
    {
        _ = NttGetScaleBarrettKernelMethod.Invoke(null, new object[] { accelerator });
    }

    private static void CompileNttSquareBarrettKernel(Accelerator accelerator)
    {
        _ = NttGetSquareBarrettKernelMethod.Invoke(null, new object[] { accelerator });
    }

    private static void CompileNttToMont64Kernel(Accelerator accelerator)
    {
        _ = NttGetToMont64KernelMethod.Invoke(null, new object[] { accelerator });
    }

    private static void CompileNttFromMont64Kernel(Accelerator accelerator)
    {
        _ = NttGetFromMont64KernelMethod.Invoke(null, new object[] { accelerator });
    }

    private static void CompileNttSquareMont64Kernel(Accelerator accelerator)
    {
        _ = NttGetSquareMont64KernelMethod.Invoke(null, new object[] { accelerator });
    }

    private static void CompileNttScaleMont64Kernel(Accelerator accelerator)
    {
        _ = NttGetScaleMont64KernelMethod.Invoke(null, new object[] { accelerator });
    }

    private static void CompileNttForwardKernel(Accelerator accelerator)
    {
        _ = NttGetForwardKernelMethod.Invoke(null, new object[] { accelerator });
    }

    private static void CompileNttInverseKernel(Accelerator accelerator)
    {
        _ = NttGetInverseKernelMethod.Invoke(null, new object[] { accelerator });
    }

    private static void CompileDivisorByDivisorCheckKernel(Accelerator accelerator)
    {
        var tester = new MersenneNumberDivisorByDivisorGpuTester();
        _ = DivisorByDivisorGetKernelMethod.Invoke(tester, new object[] { accelerator });
    }

    private static void CompileDivisorTesterKernel(Accelerator accelerator)
    {
        var tester = new MersenneNumberDivisorGpuTester();
        _ = DivisorTesterGetKernelMethod.Invoke(tester, new object[] { accelerator });
    }

    private static void CompileLucasLehmerKernel(Accelerator accelerator)
    {
        var tester = new MersenneNumberLucasLehmerGpuTester();
        _ = LucasLehmerGetKernelMethod.Invoke(tester, new object[] { accelerator });
    }

    private static void CompileLucasLehmerAddSmallKernel(Accelerator accelerator)
    {
        var tester = new MersenneNumberLucasLehmerGpuTester();
        _ = LucasLehmerGetAddSmallKernelMethod.Invoke(tester, new object[] { accelerator });
    }

    private static void CompileLucasLehmerSubSmallKernel(Accelerator accelerator)
    {
        var tester = new MersenneNumberLucasLehmerGpuTester();
        _ = LucasLehmerGetSubSmallKernelMethod.Invoke(tester, new object[] { accelerator });
    }

    private static void CompileLucasLehmerReduceKernel(Accelerator accelerator)
    {
        var tester = new MersenneNumberLucasLehmerGpuTester();
        _ = LucasLehmerGetReduceKernelMethod.Invoke(tester, new object[] { accelerator });
    }

    private static void CompileLucasLehmerIsZeroKernel(Accelerator accelerator)
    {
        var tester = new MersenneNumberLucasLehmerGpuTester();
        _ = LucasLehmerGetIsZeroKernelMethod.Invoke(tester, new object[] { accelerator });
    }

    private static void CompileLucasLehmerBatchKernel(Accelerator accelerator)
    {
        var tester = new MersenneNumberLucasLehmerGpuTester();
        _ = LucasLehmerGetBatchKernelMethod.Invoke(tester, new object[] { accelerator });
    }

    private static void CompilePrimeOrderPartialFactorKernel(Accelerator accelerator)
    {
        _ = PrimeOrderGetPartialFactorKernelMethod.Invoke(null, new object[] { accelerator });
    }

    private static void CompilePrimeOrderKernel(Accelerator accelerator)
    {
        _ = PrimeOrderGetOrderKernelMethod.Invoke(null, new object[] { accelerator });
    }

    private static void CompilePrimeOrderPow2ModKernel(Accelerator accelerator)
    {
        _ = PrimeOrderGetPow2ModKernelMethod.Invoke(null, new object[] { accelerator });
    }

    private static void CompilePrimeOrderPow2ModWideKernel(Accelerator accelerator)
    {
        _ = PrimeOrderGetPow2ModWideKernelMethod.Invoke(null, new object[] { accelerator });
    }

    private static void CompileMersenneDivisorCyclesKernel(Accelerator accelerator)
    {
        var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>(MersenneDivisorCyclesKernel);
        _ = kernel;
    }

    private static void CompilePrimeTesterSmallPrimeKernel(Accelerator accelerator)
    {
        var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>>(PrimeTesterSmallPrimeKernel);
        _ = kernel;
    }

    private static void CompilePrimeTesterSharesFactorKernel(Accelerator accelerator)
    {
        var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ArrayView<byte>>(PrimeTesterSharesFactorKernel);
        _ = kernel;
    }

    private static void CompileGpuKernelPoolKernels(Accelerator accelerator)
    {
        _ = accelerator;
        var lease = GpuKernelPool.GetKernel(useGpuOrder: false);
        try
        {
            _ = lease.OrderKernel;
            _ = lease.IncrementalKernel;
            _ = lease.Pow2ModKernel;
            _ = lease.IncrementalOrderKernel;
            _ = lease.Pow2ModOrderKernel;
        }
        finally
        {
            lease.Dispose();
        }

        var orderLease = GpuKernelPool.GetKernel(useGpuOrder: true);
        try
        {
            _ = orderLease.OrderKernel;
            _ = orderLease.IncrementalKernel;
            _ = orderLease.Pow2ModKernel;
            _ = orderLease.IncrementalOrderKernel;
            _ = orderLease.Pow2ModOrderKernel;
        }
        finally
        {
            orderLease.Dispose();
        }
    }
}
