using System.Reflection;
using FluentAssertions;
using PerfectNumbers.Core.Gpu;
using Xunit;

namespace EvenPerfectBitScanner.Tests;

[Trait("Category", "Fast")]
public class ResultsFileNameTests
{
    [Fact]
    public void BuildResultsFileName_contains_reduction_and_devices()
    {
        var programType = typeof(global::EvenPerfectBitScanner.Program);
        var method = programType.GetMethod(
            "BuildResultsFileName",
            BindingFlags.NonPublic | BindingFlags.Static,
            binder: null,
            types: new[]
            {
                typeof(bool), typeof(int), typeof(int), typeof(global::PerfectNumbers.Core.GpuKernelType),
                typeof(bool), typeof(bool), typeof(bool), typeof(bool), typeof(bool), typeof(bool), typeof(bool), typeof(NttBackend), typeof(int), typeof(int), typeof(int), typeof(ulong), typeof(ModReductionMode), typeof(string), typeof(string), typeof(string)
            },
            modifiers: null)!;

        string name = (string)method.Invoke(null, new object[]
        {
            true,            // bitInc
            8,               // threads
            1,               // block
            global::PerfectNumbers.Core.GpuKernelType.Incremental,
            false,           // useLucasFlag
            false,           // useDivisorFlag
            false,           // useByDivisorFlag
            true,            // mersenneOnGpu (unused inside name content, devices passed explicitly below)
            false,           // useOrder
            false,           // useModWorkaround
            false,           // useGcd
            NttBackend.Staged,
            1,               // gpuPrimeThreads
            32,              // llSlice
            1024,            // gpuScanBatch
            1000UL,          // warmupLimit
            ModReductionMode.Barrett128,
            "gpu",          // mersenneDevice
            "gpu",          // primesDevice
            "gpu"           // orderDevice
        })!;

        name.Should().Contain("ntt-staged");
        name.Should().Contain("red-barrett128");
        name.Should().Contain("mersdev-gpu");
        name.Should().Contain("primesdev-gpu");
        name.Should().Contain("orderdev-gpu");
    }
}
