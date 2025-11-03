using FluentAssertions;
using PerfectNumbers.Core.Gpu;
using Xunit;

namespace PerfectNumbers.Core.Tests.Gpu;

[Trait("Category", "Fast")]
public class PrimeTesterGpuTests
{
    [Fact]
    public void IsPrimeGpu_accepts_known_primes()
    {
        try
        {
            var tester = new PrimeTester();
            ulong[] primes = [31UL, 61UL, 89UL, 107UL, 127UL, 521UL];
            foreach (ulong prime in primes)
            {
				PrimeTester.IsPrimeGpu(prime).Should().BeTrue();
            }
        }
        finally
        {
            GpuContextPool.DisposeAll();
        }
    }

    [Fact]
    public void IsPrimeGpu_rejects_composites()
    {
        try
        {
            var tester = new PrimeTester();
            ulong[] composites = [33UL, 39UL, 51UL, 77UL, 91UL];
            foreach (ulong composite in composites)
            {
				PrimeTester.IsPrimeGpu(composite).Should().BeFalse();
            }
        }
        finally
        {
            GpuContextPool.DisposeAll();
        }
    }

}
