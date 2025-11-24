using FluentAssertions;
using PerfectNumbers.Core.Gpu.Accelerators;
using Xunit;

namespace PerfectNumbers.Core.Tests;

[Trait("Category", "Fast")]
public class PrimeTesterGcdTests
{
	[Theory]
	[InlineData(81UL, true)]
	[InlineData(101UL, false)]
	public void SharesFactorWithMaxExponent_detects_gcd(ulong n, bool expected)
	{
		PrimeTester.SharesFactorWithMaxExponent(n).Should().Be(expected);
	}

	[Fact]
	public void SharesFactorWithMaxExponentBatch_filters_values()
	{
		ulong[] values = [81UL, 101UL];
		Span<byte> results = stackalloc byte[values.Length];
		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		try
		{
			PrimeTester.SharesFactorWithMaxExponentBatch(gpu, values, results);
			results.ToArray().Should().Equal(new byte[] { 1, 0 });
		}
		finally
		{
			PrimeOrderCalculatorAccelerator.Return(gpu);
		}
	}
}
