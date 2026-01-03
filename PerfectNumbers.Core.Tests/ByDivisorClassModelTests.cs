using System;
using System.IO;
using System.Numerics;
using FluentAssertions;
using PerfectNumbers.Core.ByDivisor;
using Xunit;
using System.Text.Json;

namespace PerfectNumbers.Core.Tests;

public sealed class ByDivisorClassModelTests
{
    [Fact]
    public void LoadOrBuild_builds_model_from_results_file()
    {
        string tempPath = Path.GetTempFileName();
        try
        {
            File.WriteAllLines(
                tempPath,
                new[]
                {
                    "p,divisor,searchedMersenne,detailedCheck,passedAllTests",
                    "5,31,true,true,false",
                    "5,0,true,true,true",
                    "13,8191,true,true,false",
                });

            string modelPath = tempPath + ".model";
            ByDivisorScanTuning tuning = new()
            {
                CheapLimit = 32,
            };

            ByDivisorClassModel model = ByDivisorClassModelLoader.LoadOrBuild(modelPath, tempPath, tuning, tuning.CheapLimit);

            ByDivisorClassEntry classFive = model.Entries[5];
            classFive.K50.Should().Be(3d);
            classFive.K75.Should().Be(3d);
            classFive.GapProbability.Should().BeApproximately(0.5d, 1e-12);
            classFive.SampleCount.Should().Be(2);
            classFive.GapCount.Should().Be(1);

            ByDivisorClassEntry classThirteen = model.Entries[13];
            classThirteen.K50.Should().Be(315d);
            classThirteen.K75.Should().Be(315d);
            classThirteen.GapProbability.Should().Be(0d);
        }
        finally
        {
            TryDelete(tempPath);
            TryDelete(tempPath + ".model");
        }
    }

    [Fact]
    public void BuildPlan_uses_aggressive_tuning_for_hard_class()
    {
        var model = new ByDivisorClassModel
        {
            CheapLimit = 50,
        };
        model.Entries[0] = new ByDivisorClassEntry
        {
            K50 = 1000d,
            K75 = 2000d,
            GapProbability = 0.8d,
            Stability = "escaping",
        };

        ByDivisorScanTuning tuning = new()
        {
            CheapLimit = 50,
            Gamma50 = 0.8d,
            Gamma75 = 1.2d,
            AggressiveGamma50 = 1.0d,
            AggressiveGamma75 = 1.4d,
            Rho1 = 1.35d,
            Rho2 = 1.6d,
            AggressiveRho2 = 1.8d,
            HardGapThreshold = 0.75d,
        };

        ByDivisorScanPlan plan = model.BuildPlan(2UL, BigInteger.Zero, tuning);

        plan.CheapLimit.Should().Be(50d);
        plan.Start.Should().Be(1000d);
        plan.Target.Should().Be(2800d);
        plan.Rho1.Should().Be(1.35d);
        plan.Rho2.Should().Be(1.8d);
    }

    [Fact]
    public void LoadOrBuild_rebuilds_when_results_newer_than_model()
    {
        string tempResults = Path.GetTempFileName();
        string tempModel = tempResults + ".model";
        try
        {
            File.WriteAllLines(
                tempResults,
                new[]
                {
                    "p,divisor,searchedMersenne,detailedCheck,passedAllTests",
                    "5,31,true,true,false",
                });

            // Write a stale model
            var staleModel = new ByDivisorClassModel
            {
                CheapLimit = 10,
            };
            File.WriteAllText(tempModel, JsonSerializer.Serialize(staleModel));
            File.SetLastWriteTimeUtc(tempModel, DateTime.UtcNow.AddMinutes(-10));
            File.SetLastWriteTimeUtc(tempResults, DateTime.UtcNow);

            ByDivisorScanTuning tuning = new()
            {
                CheapLimit = 32,
            };

            ByDivisorClassModel model = ByDivisorClassModelLoader.LoadOrBuild(tempModel, tempResults, tuning, tuning.CheapLimit);

            model.Entries[5].K50.Should().Be(3d);
            File.GetLastWriteTimeUtc(tempModel).Should().BeAfter(File.GetLastWriteTimeUtc(tempResults).AddSeconds(-2));
        }
        finally
        {
            TryDelete(tempResults);
            TryDelete(tempModel);
        }
    }

    private static void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch
        {
            // ignore cleanup failure in tests
        }
    }
}
