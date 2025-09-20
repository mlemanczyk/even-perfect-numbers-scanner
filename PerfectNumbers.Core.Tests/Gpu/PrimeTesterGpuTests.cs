using System.Threading;
using FluentAssertions;
using PerfectNumbers.Core;
using PerfectNumbers.Core.Gpu;
using Xunit;

namespace PerfectNumbers.Core.Tests.Gpu;

[Trait("Category", "Fast")]
public class PrimeTesterGpuTests
{
    [Fact]
    public void IsPrimeGpu_accepts_known_primes()
    {
        GpuContextPool.ForceCpu = false;
        try
        {
            var tester = new PrimeTester();
            ulong[] primes = [5UL, 31UL, 61UL, 89UL, 107UL, 127UL];
            foreach (ulong prime in primes)
            {
                tester.IsPrimeGpu(prime, CancellationToken.None).Should().BeTrue();
            }
        }
        finally
        {
            GpuContextPool.ForceCpu = false;
            GpuContextPool.DisposeAll();
        }
    }

    [Fact]
    public void IsPrimeGpu_rejects_composites()
    {
        GpuContextPool.ForceCpu = false;
        try
        {
            var tester = new PrimeTester();
            ulong[] composites = [21UL, 25UL, 27UL, 35UL, 55UL];
            foreach (ulong composite in composites)
            {
                tester.IsPrimeGpu(composite, CancellationToken.None).Should().BeFalse();
            }
        }
        finally
        {
            GpuContextPool.ForceCpu = false;
            GpuContextPool.DisposeAll();
        }
    }

}
