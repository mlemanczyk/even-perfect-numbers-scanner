using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using FluentAssertions;
using PerfectNumbers.Core;
using PerfectNumbers.Core.Gpu;
using Xunit;
using UInt128 = System.UInt128;

namespace PerfectNumbers.Core.Tests;

public class MersenneNumberResidueIntegrationTests
{
    public static IEnumerable<object[]> ResidueThreadCounts()
    {
        yield return new object[] { 1 };
        yield return new object[] { 2 };
        yield return new object[] { 4 };
    }

    [Theory]
    [InlineData(107UL)]
    [InlineData(127UL)]
    [Trait("Category", "Fast")]
    public void Residue_gpu_mode_handles_large_residue_set_counts_for_known_primes(ulong exponent)
    {
        var tester = new MersenneNumberTester(
            useIncremental: false,
            useOrderCache: false,
            kernelType: GpuKernelType.Pow2Mod,
            useModuloWorkaround: false,
            useOrder: false,
            useGpuLucas: false,
            useGpuScan: true,
            useGpuOrder: true,
            useResidue: true,
            maxK: (UInt128)1_024UL,
            residueDivisorSets: (UInt128)1_024UL);

        try
        {
            bool isPrime = tester.IsMersennePrime(exponent, out bool divisorsExhausted);

            isPrime.Should().BeTrue();
            divisorsExhausted.Should().BeTrue();
        }
        finally
        {
            GpuContextPool.DisposeAll();
        }
    }

    [Theory]
    [MemberData(nameof(ResidueThreadCounts))]
    [Trait("Category", "Fast")]
    public void Residue_gpu_mode_matches_cli_configuration_for_known_primes(int threadCount)
    {
        ulong[] exponents = MersennePrimeTestData.Exponents;
        ulong[][] partitions = PartitionExponents(exponents, threadCount);
        var results = new ConcurrentDictionary<ulong, (bool IsPrime, bool Exhausted)>();
        UInt128 residueSetCount = (UInt128)PerfectNumberConstants.ExtraDivisorCycleSearchLimit;

        try
        {
            Task[] tasks = partitions.Select(partition => Task.Run(() =>
            {
                if (partition.Length == 0)
                {
                    return;
                }

                var tester = new MersenneNumberTester(
                    useIncremental: false,
                    useOrderCache: false,
                    kernelType: GpuKernelType.Pow2Mod,
                    useModuloWorkaround: false,
                    useOrder: false,
                    useGpuLucas: false,
                    useGpuScan: true,
                    useGpuOrder: true,
                    useResidue: true,
                    maxK: (UInt128)1_024UL,
                    residueDivisorSets: residueSetCount);

                foreach (ulong exponent in partition)
                {
                    bool isPrime = tester.IsMersennePrime(exponent, out bool divisorsExhausted);
                    results[exponent] = (isPrime, divisorsExhausted);
                }
            })).ToArray();

            Task.WaitAll(tasks);
        }
        finally
        {
            GpuContextPool.DisposeAll();
        }

        results.Keys.Should().BeEquivalentTo(exponents);
        foreach (ulong exponent in exponents)
        {
            results[exponent].Should().Be((true, true));
        }
    }

    [Theory]
    [MemberData(nameof(ResidueThreadCounts))]
    [Trait("Category", "Fast")]
    public void Residue_cpu_mode_matches_cli_configuration_for_known_primes(int threadCount)
    {
        ulong[] exponents = MersennePrimeTestData.Exponents;
        ulong[][] partitions = PartitionExponents(exponents, threadCount);
        var results = new ConcurrentDictionary<ulong, (bool IsPrime, bool Exhausted)>();
        UInt128 residueSetCount = (UInt128)PerfectNumberConstants.ExtraDivisorCycleSearchLimit;

        Task[] tasks = partitions.Select(partition => Task.Run(() =>
        {
            if (partition.Length == 0)
            {
                return;
            }

            var tester = new MersenneNumberTester(
                useIncremental: false,
                useOrderCache: false,
                kernelType: GpuKernelType.Pow2Mod,
                useModuloWorkaround: false,
                useOrder: false,
                useGpuLucas: false,
                useGpuScan: false,
                useGpuOrder: false,
                useResidue: true,
                maxK: (UInt128)1_024UL,
                residueDivisorSets: residueSetCount);

            foreach (ulong exponent in partition)
            {
                bool isPrime = tester.IsMersennePrime(exponent, out bool divisorsExhausted);
                results[exponent] = (isPrime, divisorsExhausted);
            }
        })).ToArray();

        Task.WaitAll(tasks);

        results.Keys.Should().BeEquivalentTo(exponents);
        foreach (ulong exponent in exponents)
        {
            results[exponent].Should().Be((true, true));
        }
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void Residue_cpu_mode_respects_filter_configuration()
    {
        var filter = new HashSet<ulong>(MersennePrimeTestData.Exponents);
        ulong[] candidates = [31UL, 37UL, 73UL];

        var tester = new MersenneNumberTester(
            useIncremental: false,
            useOrderCache: false,
            kernelType: GpuKernelType.Pow2Mod,
            useModuloWorkaround: false,
            useOrder: false,
            useGpuLucas: false,
            useGpuScan: false,
            useGpuOrder: false,
            useResidue: true,
            maxK: (UInt128)32UL,
            residueDivisorSets: (UInt128)4UL);

        List<ulong> processed = new();
        List<ulong> skipped = new();

        filter.Count.Should().Be(MersennePrimeTestData.Exponents.Length);

        foreach (ulong exponent in candidates)
        {
            if (!filter.Contains(exponent))
            {
                skipped.Add(exponent);
                continue;
            }

            bool isPrime = tester.IsMersennePrime(exponent, out bool divisorsExhausted);
            isPrime.Should().BeTrue();
            divisorsExhausted.Should().BeTrue();
            processed.Add(exponent);
        }

        processed.Should().ContainSingle().Which.Should().Be(31UL);
        skipped.Should().BeEquivalentTo(new[] { 37UL, 73UL });
    }

    private static ulong[][] PartitionExponents(ulong[] exponents, int threadCount)
    {
        return Enumerable
            .Range(0, threadCount)
            .Select(partition => exponents.Where((_, index) => index % threadCount == partition).ToArray())
            .ToArray();
    }
}

