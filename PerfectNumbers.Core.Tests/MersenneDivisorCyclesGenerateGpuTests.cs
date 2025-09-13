using System.Collections.Generic;
using System.IO;
using FluentAssertions;
using PerfectNumbers.Core;
using Xunit;

namespace PerfectNumbers.Core.Tests;

public class MersenneDivisorCyclesGenerateGpuTests
{
    [Fact]
    [Trait("Category", "Fast")]
    public void GenerateGpu_produces_expected_cycles_for_small_range()
    {
        string path = Path.GetTempFileName();
        try
        {
            MersenneDivisorCycles.GenerateGpu(path, maxDivisor: 50UL, batchSize: 16, skipCount: 0, nextPosition: 0);
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
                pair.cycle.Should().Be(MersenneDivisorCycles.CalculateCycleLength(pair.divisor));
            }
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
