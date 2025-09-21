using System.IO;
using System.Linq;
using System.Reflection;
using FluentAssertions;
using PerfectNumbers.Core;
using PerfectNumbers.Core.Cpu;
using PerfectNumbers.Core.Gpu;
using Xunit;
using UInt128 = System.UInt128;

namespace PerfectNumbers.Core.Tests;

public class MersenneNumberResidueGpuTesterTests
{
    private static bool LastDigitIsSeven(ulong exponent) => (exponent & 3UL) == 3UL;

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    [Trait("Category", "Fast")]
    public void Scan_handles_various_prime_exponents(bool useGpuOrder)
    {
        var tester = new MersenneNumberResidueGpuTester(useGpuOrder);

        RunCase(tester, 23UL, 2UL, expectedPrime: false);
        RunCase(tester, 29UL, 37UL, expectedPrime: false);
        RunCase(tester, 31UL, 1_000UL, expectedPrime: true);
        RunCase(tester, 89UL, 1_001UL, expectedPrime: true);
        RunCase(tester, 107UL, 1_001UL, expectedPrime: true);
        RunCase(tester, 127UL, 1_001UL, expectedPrime: true);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    [Trait("Category", "Fast")]
    public void Gpu_residue_scan_matches_cpu_for_small_configurations(bool useGpuOrder)
    {
        var gpuTester = new MersenneNumberResidueGpuTester(useGpuOrder);
        var cpuTester = new MersenneNumberResidueCpuTester();

        var configurations = new (ulong Exponent, UInt128 PerSetLimit, UInt128 SetCount, UInt128 OverallLimit)[]
        {
            (31UL, 8UL, 6UL, 48UL),
            (61UL, 9UL, 9UL, 81UL),
            (89UL, 12UL, 8UL, 96UL),
            (107UL, 16UL, 8UL, 128UL),
        };

        foreach (var (exponent, perSetLimit, setCount, overallLimit) in configurations)
        {
            bool gpuPrime = true;
            bool gpuExhausted = false;
            gpuTester.Scan(
                exponent,
                (UInt128)exponent << 1,
                LastDigitIsSeven(exponent),
                perSetLimit,
                setCount,
                overallLimit,
                ref gpuPrime,
                ref gpuExhausted);

            bool cpuPrime = true;
            bool cpuExhausted = false;
            cpuTester.Scan(
                exponent,
                (UInt128)exponent << 1,
                LastDigitIsSeven(exponent),
                perSetLimit,
                setCount,
                overallLimit,
                ref cpuPrime,
                ref cpuExhausted);

            gpuPrime.Should().Be(cpuPrime, $"GPU residue scan should match CPU for exponent {exponent}");
            gpuExhausted.Should().Be(cpuExhausted, $"GPU residue scan should report identical exhaustion for exponent {exponent}");
        }
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void Scan_residue_gpu_handles_inaccurate_small_cycle_entries()
    {
        const ulong exponent = 23UL;
        const ulong divisor = 47UL; // 47 | M_23

        string tempPath = Path.GetTempFileName();
        try
        {
            using (var writer = new BinaryWriter(File.Open(tempPath, FileMode.Create, FileAccess.Write, FileShare.Read)))
            {
                writer.Write(divisor);
                writer.Write((ulong)uint.MaxValue);
            }

            var cycles = MersenneDivisorCycles.Shared;
            var tableField = typeof(MersenneDivisorCycles).GetField("_table", BindingFlags.NonPublic | BindingFlags.Instance)!;
            var smallCyclesField = typeof(MersenneDivisorCycles).GetField("_smallCycles", BindingFlags.NonPublic | BindingFlags.Instance)!;

            var originalTable = ((List<(ulong Divisor, ulong Cycle)>)tableField.GetValue(cycles)!).Select(x => x).ToList();
            var originalSmall = (uint[]?)smallCyclesField.GetValue(cycles);
            uint[]? backupSmall = originalSmall is null ? null : (uint[])originalSmall.Clone();

            cycles.LoadFrom(tempPath);
            GpuContextPool.DisposeAll();

            try
            {
                var tester = new MersenneNumberResidueGpuTester(useGpuOrder: false);

                bool isPrime = true;
                bool exhausted = false;

                tester.Scan(
                    exponent,
                    (UInt128)exponent << 1,
                    lastIsSeven: true,
                    perSetLimit: (UInt128)10UL,
                    setCount: UInt128.One,
                    overallLimit: (UInt128)10UL,
                    ref isPrime,
                    ref exhausted);

                isPrime.Should().BeFalse();
                exhausted.Should().BeTrue();
            }
            finally
            {
                tableField.SetValue(cycles, originalTable);
                smallCyclesField.SetValue(cycles, backupSmall);
                GpuContextPool.DisposeAll();
            }
        }
        finally
        {
            File.Delete(tempPath);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    [Trait("Category", "Fast")]
    public void Mersenne_tester_residue_gpu_handles_known_primes_with_many_sets(bool useGpuOrder)
    {
        var tester = new MersenneNumberTester(
            useIncremental: true,
            useOrderCache: false,
            kernelType: GpuKernelType.Pow2Mod,
            useModuloWorkaround: false,
            useOrder: false,
            useGpuLucas: false,
            useGpuScan: true,
            useGpuOrder: useGpuOrder,
            useResidue: true,
            maxK: (UInt128)2_400_000UL,
            residueDivisorSets: (UInt128)512UL);

        try
        {
            bool isPrime = tester.IsMersennePrime(127UL, out bool divisorsExhausted);

            isPrime.Should().BeTrue();
            divisorsExhausted.Should().BeTrue();
        }
        finally
        {
            GpuContextPool.DisposeAll();
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    [Trait("Category", "Fast")]
    public void Scan_recognizes_all_known_small_mersenne_primes(bool useGpuOrder)
    {
        var tester = new MersenneNumberResidueGpuTester(useGpuOrder);

        foreach (ulong exponent in MersennePrimeTestData.Exponents)
        {
            RunCase(tester, exponent, 1UL, expectedPrime: true);
        }
    }

    [Theory]
    [InlineData(31UL)]
    [Trait("Category", "Fast")]
    public void Scan_matches_cli_residue_configuration_for_primes(ulong exponent)
    {
        RunCliCase(useGpuOrder: false, exponent);
        RunCliCase(useGpuOrder: true, exponent);
    }

    [Theory]
    [InlineData(31UL, 1_024UL, 1UL)]
    [InlineData(31UL, 256UL, 4UL)]
    [InlineData(107UL, 512UL, 1UL)]
    [InlineData(107UL, 128UL, 4UL)]
    [InlineData(107UL, 15_625UL, 64UL)]
    [InlineData(127UL, 512UL, 1UL)]
    [InlineData(127UL, 128UL, 4UL)]
    [Trait("Category", "Fast")]
    public void Scan_handles_multiple_residue_sets_for_known_primes(ulong exponent, ulong perSetLimit, ulong setCount)
    {
        ulong overallLimit = perSetLimit * setCount;

        RunCase(new MersenneNumberResidueGpuTester(useGpuOrder: false), exponent, perSetLimit, setCount, overallLimit, expectedPrime: true);
        RunCase(new MersenneNumberResidueGpuTester(useGpuOrder: true), exponent, perSetLimit, setCount, overallLimit, expectedPrime: true);
    }

    [Theory]
    [InlineData(107UL, 1UL, 1_024UL)]
    [Trait("Category", "Fast")]
    public void Scan_handles_single_lane_sets_for_known_primes(ulong exponent, ulong perSetLimit, ulong overallLimit)
    {
        ulong setCount = overallLimit;

        RunCase(new MersenneNumberResidueGpuTester(useGpuOrder: false), exponent, perSetLimit, setCount, overallLimit, expectedPrime: true);
        RunCase(new MersenneNumberResidueGpuTester(useGpuOrder: true), exponent, perSetLimit, setCount, overallLimit, expectedPrime: true);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    [Trait("Category", "Fast")]
    public void Scan_handles_multiple_sets_without_false_positives(bool useGpuOrder)
    {
        var tester = new MersenneNumberResidueGpuTester(useGpuOrder);

        bool isPrime = true;
        bool exhausted = false;
        UInt128 perSetLimit = 1_666_667UL;
        UInt128 setCount = 3UL;
        UInt128 overallLimit = 5_000_000UL;

        tester.Scan(
            127UL,
            (UInt128)127UL << 1,
            LastDigitIsSeven(127UL),
            perSetLimit,
            setCount,
            overallLimit,
            ref isPrime,
            ref exhausted);

        isPrime.Should().BeTrue();
        exhausted.Should().BeTrue();

        isPrime = true;
        exhausted = false;
        perSetLimit = 20_000_000UL;
        setCount = 5UL;
        overallLimit = 100_000_000UL;

        tester.Scan(
            107UL,
            (UInt128)107UL << 1,
            LastDigitIsSeven(107UL),
            perSetLimit,
            setCount,
            overallLimit,
            ref isPrime,
            ref exhausted);

        isPrime.Should().BeTrue();
        exhausted.Should().BeTrue();
    }

    private static void RunCliCase(bool useGpuOrder, ulong exponent)
    {
        var tester = new MersenneNumberResidueGpuTester(useGpuOrder);
        RunCase(tester, exponent, 8UL, expectedPrime: true);
    }

    private static void RunCase(MersenneNumberResidueGpuTester tester, ulong exponent, ulong maxK, bool expectedPrime)
    {
        RunCase(tester, exponent, maxK, 1UL, maxK, expectedPrime);
    }

    private static void RunCase(
        MersenneNumberResidueGpuTester tester,
        ulong exponent,
        ulong perSetLimit,
        ulong setCount,
        ulong overallLimit,
        bool expectedPrime)
    {
        bool isPrime = true;
        tester.Scan(exponent, (UInt128)exponent << 1, LastDigitIsSeven(exponent), (UInt128)maxK, ref isPrime);
        isPrime.Should().Be(expectedPrime);
    }
}

