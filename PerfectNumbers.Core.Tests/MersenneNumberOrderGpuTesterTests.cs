using System.Reflection;
using FluentAssertions;
using PerfectNumbers.Core;
using PerfectNumbers.Core.Gpu;
using Xunit;
using UInt128 = System.UInt128;

namespace PerfectNumbers.Core.Tests;

public class MersenneNumberOrderGpuTesterTests
{
    private static LastDigit GetLastDigit(ulong exponent) => (exponent & 3UL) == 3UL ? LastDigit.Seven : LastDigit.One;

    [Theory]
    [InlineData(GpuKernelType.Pow2Mod, false)]
    [InlineData(GpuKernelType.Incremental, false)]
    [InlineData(GpuKernelType.Pow2Mod, true)]
    [Trait("Category", "Fast")]
    public void Scan_handles_various_prime_exponents(GpuKernelType type, bool useGpuOrder)
    {
        var tester = new MersenneNumberOrderGpuTester(type, useGpuOrder);

        if (type == GpuKernelType.Pow2Mod)
        {
            RunCase(tester, 23UL, 1UL, expectedPrime: false);
            RunCase(tester, 29UL, 36UL, expectedPrime: false);
        }

        RunCase(tester, 89UL, 1_000UL, expectedPrime: true);
        RunCase(tester, 127UL, 1_000UL, expectedPrime: true);
    }

    [Fact]
    [Trait("Category", "Fast")]
    public void Scan_order_gpu_handles_inaccurate_small_cycle_entries()
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
            var originalSmall = (ulong[]?)smallCyclesField.GetValue(cycles);
            ulong[]? backupSmall = originalSmall is null ? null : (ulong[])originalSmall.Clone();

            cycles.LoadFrom(tempPath);
            GpuContextPool.DisposeAll();

            try
            {
                var tester = new MersenneNumberOrderGpuTester(GpuKernelType.Pow2Mod, useGpuOrder: false);

                bool isPrime = true;
                tester.Scan(
                    exponent,
                    (UInt128)exponent << 1,
                    GetLastDigit(exponent),
                    (UInt128)10UL,
                    ref isPrime);

                isPrime.Should().BeFalse();
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

    private static void RunCase(MersenneNumberOrderGpuTester tester, ulong exponent, ulong maxK, bool expectedPrime)
    {
        bool isPrime = true;
        tester.Scan(exponent, (UInt128)exponent << 1, GetLastDigit(exponent), (UInt128)maxK, ref isPrime);
        isPrime.Should().Be(expectedPrime);
    }
}

