using FluentAssertions;
using PerfectNumbers.Core;
using PerfectNumbers.Core.Gpu;
using Xunit;

using EvenPerfectBitScanner.IO;

namespace EvenPerfectBitScanner.Tests;

[Trait("Category", "Fast")]
public class ResultsFileNameTests
{
    [Fact]
    public void BuildResultsFileName_contains_reduction_and_devices()
    {
        string name = CalculationResultsFile.BuildFileName(
            bitInc: true,
            threads: 8,
            block: 1,
            kernelType: GpuKernelType.Incremental,
            useLucasFlag: false,
            useDivisorFlag: false,
            useByDivisorFlag: false,
            mersenneOnGpu: true,
            useOrder: false,
            useGcd: false,
            nttBackend: NttBackend.Staged,
            gpuPrimeThreads: 1,
            llSlice: 32,
            gpuScanBatch: 1024,
            warmupLimit: 1000UL,
            reduction: ModReductionMode.Barrett128,
            mersenneDevice: "gpu",
            primesDevice: "gpu",
            orderDevice: "gpu");

        name.Should().Contain("ntt-staged");
        name.Should().Contain("red-barrett128");
        name.Should().Contain("mersdev-gpu");
        name.Should().Contain("primesdev-gpu");
        name.Should().Contain("orderdev-gpu");
    }

    [Fact]
    public void LoadCandidatesWithinRange_skips_comments_and_applies_limit()
    {
        string path = Path.GetTempFileName();
        try
        {
            File.WriteAllLines(path, new[]
            {
                "# header",
                "5 7",
                string.Empty,
                "11",
                "17",
                "19"
            });

            List<ulong> candidates = CalculationResultsFile.LoadCandidatesWithinRange(path, (UInt128)13, maxPrimeConfigured: true, out int skipped);

            candidates.Should().Equal(5UL, 7UL, 11UL);
            skipped.Should().Be(2);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void EnumerateCandidates_reads_valid_rows_and_skips_invalid_entries()
    {
        string path = Path.GetTempFileName();
        try
        {
            File.WriteAllLines(path, new[]
            {
                "p,searched,detailed,passed",
                "5,true,true,true",
                string.Empty,
                "11,false,false,true",
                "invalid",
                "13,true,true,false"
            });

            List<(ulong P, bool Detailed, bool Passed)> rows = new();
            CalculationResultsFile.EnumerateCandidates(path, (p, detailed, passed) => rows.Add((p, detailed, passed)));

            rows.Should().Equal(new (ulong, bool, bool)[]
            {
                (5UL, true, true),
                (11UL, false, true),
                (13UL, true, false)
            });
        }
        finally
        {
            File.Delete(path);
        }
    }
}
