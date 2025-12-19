using FluentAssertions;
using PerfectNumbers.Core.Gpu.Accelerators;
using Xunit;

namespace PerfectNumbers.Core.Tests;

public class MersenneDivisorCyclesGenerateGpuTests
{
	[Fact]
	[Trait("Category", "Fast")]
	public void GenerateGpu_produces_expected_cycles_for_small_range()
	{
		string path = Path.GetTempFileName();
		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
		try
		{
			MersenneDivisorCyclesGpu.GenerateGpu(path, maxDivisor: 50UL, batchSize: 16, skipCount: 0, nextPosition: 0);
			var pairs = new List<(ulong divisor, ulong cycle)>();
			using var stream = new FileStream(path, FileMode.Open, FileAccess.Read);
			using var reader = new BinaryReader(stream);
			while (stream.Position < stream.Length)
			{
				ulong d = reader.ReadUInt64();
				ulong c = reader.ReadUInt64();
				pairs.Add((d, c));
			}

			var expected = new List<ulong>();
			for (ulong d = 3; d <= 50; d++)
			{
				if ((d & 1UL) == 0UL || d % 3UL == 0UL || d % 5UL == 0UL || d % 7UL == 0UL || d % 11UL == 0UL)
				{
					continue;
				}

				expected.Add(d);
			}

			pairs.Should().HaveCount(expected.Count);
			foreach (var pair in pairs)
			{
				expected.Should().Contain(pair.divisor);
				MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(pair.divisor);
				pair.cycle.Should().Be(MersenneDivisorCyclesCpu.CalculateCycleLength(pair.divisor, divisorData));
			}
		}
		finally
		{
			PrimeOrderCalculatorAccelerator.Return(gpu);
			if (File.Exists(path))
			{
				File.Delete(path);
			}
		}
	}

	[Fact]
	[Trait("Category", "Fast")]
	public void Generate_produces_unique_cycles_for_expected_divisors()
	{
		string path = Path.GetTempFileName();
		try
		{
			MersenneDivisorCyclesCpu.Generate(path, maxDivisor: 50UL, threads: 4);

			var pairs = new List<(ulong divisor, ulong cycle)>();
			using var stream = new FileStream(path, FileMode.Open, FileAccess.Read);
			using var reader = new BinaryReader(stream);
			while (stream.Position < stream.Length)
			{
				ulong divisor = reader.ReadUInt64();
				ulong cycle = reader.ReadUInt64();
				pairs.Add((divisor, cycle));
			}

			var expected = new List<ulong>();
			for (ulong divisor = 3; divisor <= 50; divisor++)
			{
				if ((divisor & 1UL) == 0UL || divisor % 3UL == 0UL || divisor % 5UL == 0UL || divisor % 7UL == 0UL || divisor % 11UL == 0UL)
				{
					continue;
				}

				expected.Add(divisor);
			}

			pairs.Should().HaveCount(expected.Count);

			HashSet<ulong> uniqueDivisors = new();
			foreach (var pair in pairs)
			{
				uniqueDivisors.Add(pair.divisor).Should().BeTrue();
				MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(pair.divisor);
				pair.cycle.Should().Be(MersenneDivisorCyclesCpu.CalculateCycleLength(pair.divisor, divisorData));
			}

			uniqueDivisors.Should().BeEquivalentTo(expected);

			var (_, completeCount) = MersenneDivisorCyclesCpu.FindLast(path);
			completeCount.Should().Be(expected.Count);
		}
		finally
		{
			if (File.Exists(path))
			{
				File.Delete(path);
			}
		}
	}
}
