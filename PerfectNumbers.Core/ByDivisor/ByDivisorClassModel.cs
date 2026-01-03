using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace PerfectNumbers.Core.ByDivisor;

public sealed class ByDivisorClassEntry
{
    public double K50 { get; set; }
    public double K75 { get; set; }
    public double GapProbability { get; set; }
    public int SampleCount { get; set; }
    public int GapCount { get; set; }
    public string Stability { get; set; } = "unknown";
}

public sealed class ByDivisorScanPlan
{
    public double CheapLimit { get; init; }
    public double Start { get; init; }
    public double Target { get; init; }
    public double Rho1 { get; init; }
    public double Rho2 { get; init; }
}

public sealed class ByDivisorScanTuning
{
    public double CheapLimit { get; init; } = 10_000d;
    public double Gamma50 { get; init; } = 0.8d;
    public double Gamma75 { get; init; } = 1.2d;
    public double AggressiveGamma50 { get; init; } = 1.0d;
    public double AggressiveGamma75 { get; init; } = 1.4d;
    public double Rho1 { get; init; } = 1.35d;
    public double Rho2 { get; init; } = 1.6d;
    public double AggressiveRho2 { get; init; } = 1.8d;
    public double HardGapThreshold { get; init; } = 0.75d;

    [JsonIgnore]
    public static ByDivisorScanTuning Default => new();
}

public sealed class ByDivisorClassModel
{
    private const int ClassCount = 1024;
    public int Version { get; set; } = 1;
    public double CheapLimit { get; set; } = 10_000d;
    public ByDivisorClassEntry[] Entries { get; set; } = Enumerable.Range(0, ClassCount)
        .Select(_ => new ByDivisorClassEntry { K50 = 1d, K75 = 1d })
        .ToArray();

    public ByDivisorScanPlan BuildPlan(ulong prime, BigInteger allowedMax, in ByDivisorScanTuning tuning)
    {
        int suffix = (int)(prime & 1023UL);
        ByDivisorClassEntry entry = Entries[suffix];

        double k50 = entry.K50 > 1d ? entry.K50 : 1d;
        double k75 = entry.K75 > 1d ? entry.K75 : Math.Max(k50, 1d);
        bool isHard = entry.GapProbability >= tuning.HardGapThreshold || string.Equals(entry.Stability, "escaping", StringComparison.OrdinalIgnoreCase);

        double gamma50 = isHard ? tuning.AggressiveGamma50 : tuning.Gamma50;
        double gamma75 = isHard ? tuning.AggressiveGamma75 : tuning.Gamma75;
        double start = Math.Ceiling(Math.Max(1d, gamma50 * k50));
        double target = Math.Ceiling(Math.Max(start + 1d, gamma75 * k75));

        double cheap = tuning.CheapLimit > 1d ? tuning.CheapLimit : CheapLimit;
        double rho2 = isHard ? tuning.AggressiveRho2 : tuning.Rho2;

        double maxKAllowed = ComputeMaxKAllowed(prime, allowedMax);
        if (maxKAllowed > 0d)
        {
            cheap = Math.Min(cheap, maxKAllowed);
            start = Math.Min(start, maxKAllowed);
            target = Math.Min(target, maxKAllowed);
        }

        return new ByDivisorScanPlan
        {
            CheapLimit = cheap,
            Start = start,
            Target = target,
            Rho1 = tuning.Rho1,
            Rho2 = rho2,
        };
    }

    private static double ComputeMaxKAllowed(ulong prime, in BigInteger allowedMax)
    {
        if (prime == 0UL || allowedMax.IsZero)
        {
            return 0d;
        }

        BigInteger step = ((BigInteger)prime) << 1;
        if (step.IsZero)
        {
            return 0d;
        }

        BigInteger maxK = (allowedMax - BigInteger.One) / step;
        if (maxK <= BigInteger.Zero)
        {
            return 0d;
        }

        double value = (double)maxK;
        return double.IsInfinity(value) || value <= 0d ? 0d : value;
    }
}

public static class ByDivisorClassModelLoader
{
    public static ByDivisorClassModel LoadOrBuild(string modelPath, string resultsPath, in ByDivisorScanTuning tuning, double cheapLimit, bool rebuildWhenResultsNewer = true)
    {
        DateTime modelTimestamp = File.Exists(modelPath) ? File.GetLastWriteTimeUtc(modelPath) : DateTime.MinValue;
        DateTime resultsTimestamp = File.Exists(resultsPath) ? File.GetLastWriteTimeUtc(resultsPath) : DateTime.MinValue;

        if (File.Exists(modelPath) && (!rebuildWhenResultsNewer || resultsTimestamp <= modelTimestamp))
        {
            try
            {
                ByDivisorClassModel? loaded = JsonSerializer.Deserialize<ByDivisorClassModel>(File.ReadAllText(modelPath));
                if (loaded is not null)
                {
                    loaded.CheapLimit = cheapLimit;
                    return loaded;
                }
            }
            catch
            {
                // fall back to rebuild
            }
        }

        ByDivisorClassModel model = BuildFromResults(resultsPath, cheapLimit);
        Save(modelPath, model);
        return model;
    }

    public static void RebuildAndSave(string modelPath, string resultsPath, double cheapLimit)
    {
        ByDivisorClassModel model = BuildFromResults(resultsPath, cheapLimit);
        Save(modelPath, model);
    }

    private static void Save(string modelPath, ByDivisorClassModel model)
    {
        string? directory = Path.GetDirectoryName(modelPath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
        };
        string json = JsonSerializer.Serialize(model, options);
        string tempPath = modelPath + ".tmp";
        File.WriteAllText(tempPath, json);
        File.Copy(tempPath, modelPath, overwrite: true);
        File.Delete(tempPath);
    }

    private static ByDivisorClassModel BuildFromResults(string resultsPath, double cheapLimit)
    {
        var model = new ByDivisorClassModel
        {
            CheapLimit = cheapLimit,
        };

        if (!File.Exists(resultsPath))
        {
            return model;
        }

        var samples = new List<double>[1024];
        Span<int> sampleCounts = stackalloc int[1024];
        Span<int> gapCounts = stackalloc int[1024];

        using FileStream stream = new(resultsPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
        using StreamReader reader = new(stream);
        string? line;
        bool headerSkipped = false;

        while ((line = reader.ReadLine()) is not null)
        {
            if (!headerSkipped)
            {
                headerSkipped = true;
                continue;
            }

            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            string[] parts = line.Split(',');
            if (parts.Length < 5)
            {
                continue;
            }

            if (!ulong.TryParse(parts[0], NumberStyles.None, CultureInfo.InvariantCulture, out ulong p))
            {
                continue;
            }

            if (!BigInteger.TryParse(parts[1], NumberStyles.None, CultureInfo.InvariantCulture, out BigInteger divisor))
            {
                continue;
            }

            if (!bool.TryParse(parts[3], out bool detailedCheck))
            {
                detailedCheck = false;
            }

            if (!bool.TryParse(parts[4], out bool passedAllTests))
            {
                passedAllTests = false;
            }

            int suffix = (int)(p & 1023UL);
            bool validDivisor = divisor > BigInteger.One && IsValidDivisor(p, divisor);
            if (validDivisor)
            {
                double kValue = ComputeK(divisor, p);
                if (!double.IsNaN(kValue) && !double.IsInfinity(kValue))
                {
                    samples[suffix] ??= new List<double>();
                    samples[suffix]!.Add(kValue);
                }
            }

            if (detailedCheck)
            {
                sampleCounts[suffix]++;
                if (!validDivisor || passedAllTests)
                {
                    gapCounts[suffix]++;
                }
            }
        }

        for (int i = 0; i < 1024; i++)
        {
            List<double>? classSamples = samples[i];
            double k50 = 1d;
            double k75 = 1d;
            if (classSamples is not null && classSamples.Count > 0)
            {
                classSamples.Sort();
                k50 = ComputePercentile(classSamples, 0.5d);
                k75 = ComputePercentile(classSamples, 0.75d);
            }

            int total = sampleCounts[i];
            int gaps = gapCounts[i];
            double gapProb = total > 0 ? (double)gaps / total : 0d;

            model.Entries[i] = new ByDivisorClassEntry
            {
                K50 = k50,
                K75 = k75,
                GapProbability = gapProb,
                SampleCount = total,
                GapCount = gaps,
                Stability = model.Entries[i].Stability,
            };
        }

        return model;
    }

    private static bool IsValidDivisor(ulong prime, in BigInteger divisor)
    {
        if (divisor <= BigInteger.One)
        {
            return false;
        }

        BigInteger modulus = ((BigInteger)prime) << 1;
        if (modulus.IsZero)
        {
            return false;
        }

        if ((divisor - BigInteger.One) % modulus != BigInteger.Zero)
        {
            return false;
        }

        return BigInteger.ModPow(2, prime, divisor) == BigInteger.One;
    }

    private static double ComputeK(in BigInteger divisor, ulong prime)
    {
        BigInteger numerator = divisor - BigInteger.One;
        BigInteger denominator = ((BigInteger)prime) << 1;
        if (denominator.IsZero)
        {
            return 0d;
        }

        BigInteger k = numerator / denominator;
        if (k <= BigInteger.Zero)
        {
            return 0d;
        }

        double value = (double)k;
        if (double.IsInfinity(value) || double.IsNaN(value) || value <= 0d)
        {
            return double.MaxValue;
        }

        return value;
    }

    private static double ComputePercentile(List<double> values, double percentile)
    {
        if (values.Count == 0)
        {
            return 1d;
        }

        double index = (values.Count - 1) * percentile;
        int lower = (int)Math.Floor(index);
        int upper = (int)Math.Ceiling(index);
        if (lower == upper)
        {
            return values[lower];
        }

        double fraction = index - lower;
        return values[lower] + ((values[upper] - values[lower]) * fraction);
    }
}
