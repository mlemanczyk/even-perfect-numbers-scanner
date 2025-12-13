using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using ILGPU;
using Open.Numeric.Primes;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[ShortRunJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
public class BTreeBenchmarks
{
    private ulong[] _candidates = [];
	private int[] _candidatesInt = [];

    [ParamsSource(nameof(GetBenchmarkCases))]
    public PrimeBenchmarkCase BenchmarkCase { get; set; }

    [GlobalSetup]
    public void GlobalSetup()
    {
        _candidates = BenchmarkCase.Candidates;
		_candidatesInt = [.. _candidates.Select(x => (int)x)];
    }

    /// <summary>
    /// Adds all candidate ints into the keyed B-tree (key = value), using tiny/inline and mid-size fast paths; count comes from the benchmark case or defaults to 4,096 when unspecified.
    /// </summary>
    [Benchmark]
    public BTree<int, int> BTreeAddInt()
	{
        int[] values = _candidatesInt;
        int prime;
		BTree<int, int> result = new();
        for (int i = 0; i < values.Length; i++)
        {
			prime = values[i];
			result.Add(prime, prime);
        }

        return result;
	}

    /// <summary>
    /// Adds all candidate ints into a dictionary keyed by their hash code (baseline); count comes from the benchmark case or defaults to 4,096 when unspecified.
    /// </summary>
    [Benchmark(Baseline = true)]
    public Dictionary<int, int> DictionaryAddInt()
    {
        int[] values = _candidatesInt;
        int prime;
		Dictionary<int, int> result = [];
        for (int i = 0; i < values.Length; i++)
        {
			prime = values[i];
			result.Add(prime.GetHashCode(), prime);
        }

        return result;
    }

    /// <summary>
    /// Adds all candidate ulongs into the keyed B-tree (key = value), using tiny/inline and mid-size fast paths; count comes from the benchmark case or defaults to 4,096 when unspecified.
    /// </summary>
    [Benchmark]
    public BTree<ulong, ulong> BTreeAddUlong()
	{
        ulong[] values = _candidates;
        ulong prime;
		BTree<ulong, ulong> result = new();
        for (int i = 0; i < values.Length; i++)
        {
			prime = values[i];
			result.Add(prime, prime);
        }

        return result;
	}

    /// <summary>
    /// Adds all candidate ulongs into a dictionary keyed by their hash code; count comes from the benchmark case or defaults to 4,096 when unspecified.
    /// </summary>
    [Benchmark]
    public Dictionary<int, ulong> DictionaryAddUlong()
    {
        ulong[] values = _candidates;
        ulong prime;
		Dictionary<int, ulong> result = [];
        for (int i = 0; i < values.Length; i++)
        {
			prime = values[i];
			result.Add(prime.GetHashCode(), prime);
        }

        return result;
    }

    public static IEnumerable<PrimeBenchmarkCase> GetBenchmarkCases()
    {
        yield return PrimeBenchmarkCase.FromRange(
            "≤100",
            "Admissible odd values ≤ 100 (k ≤ 5, not divisible by 5)",
            3UL,
            100UL,
            int.MaxValue);

        yield return PrimeBenchmarkCase.FromRange(
            "4x ≤4_000_000",
            "4x sample admissible odd values ≤ 4,000,000 (k ≤ 5, not divisible by 5)",
            3UL,
            4_000_000UL,
            4);

        yield return PrimeBenchmarkCase.FromRange(
            "12x ≤4_000_000",
            "12x sample admissible odd values ≤ 4,000,000 (k ≤ 5, not divisible by 5)",
            3UL,
            4_000_000UL,
            12);

        yield return PrimeBenchmarkCase.FromRange(
            "16x ≤4_000_000",
            "16x sample admissible odd values ≤ 4,000,000 (k ≤ 5, not divisible by 5)",
            3UL,
            4_000_000UL,
            16);

        yield return PrimeBenchmarkCase.FromRange(
            "32x ≤4_000_000",
            "16x sample admissible odd values ≤ 4,000,000 (k ≤ 5, not divisible by 5)",
            3UL,
            4_000_000UL,
            32);

        yield return PrimeBenchmarkCase.FromRange(
            "64x ≤4_000_000",
            "16x sample admissible odd values ≤ 4,000,000 (k ≤ 5, not divisible by 5)",
            3UL,
            4_000_000UL,
            64);

        yield return PrimeBenchmarkCase.FromRange(
            "128x ≤4_000_000",
            "128x sample admissible odd values ≤ 4,000,000 (k ≤ 5, not divisible by 5)",
            3UL,
            4_000_000UL,
            128);

        yield return PrimeBenchmarkCase.FromRange(
            "256x ≤4_000_000",
            "256x sample admissible odd values ≤ 4,000,000 (k ≤ 5, not divisible by 5)",
            3UL,
            4_000_000UL,
            256);

        yield return PrimeBenchmarkCase.FromRange(
            "512x ≤4_000_000",
            "512x sample admissible odd values ≤ 4,000,000 (k ≤ 5, not divisible by 5)",
            3UL,
            4_000_000UL,
            512);

        yield return PrimeBenchmarkCase.FromRange(
            "1024x ≤4_000_000",
            "1024x sample admissible odd values ≤ 4,000,000 (k ≤ 5, not divisible by 5)",
            3UL,
            4_000_000UL,
            1024);

        yield return PrimeBenchmarkCase.FromRange(
            "2048x ≤4_000_000",
            "2048x sample admissible odd values ≤ 4,000,000 (k ≤ 5, not divisible by 5)",
            3UL,
            4_000_000UL,
            2048);

        yield return PrimeBenchmarkCase.FromRange(
            "≤4_000_000",
            "Sampled admissible odd values ≤ 4,000,000 (k ≤ 5, not divisible by 5)",
            3UL,
            4_000_000UL,
            4_096);

        yield return PrimeBenchmarkCase.FromRange(
            "≥16_000_000",
            "Sampled admissible odd values ≥ 16,000,000 (k ≤ 5, not divisible by 5)",
            16_000_001UL,
            16_500_000UL,
            4_096);

        yield return PrimeBenchmarkCase.FromRange(
            "≥32_000_000",
            "Sampled admissible odd values ≥ 32,000,000 (k ≤ 5, not divisible by 5)",
            32_000_001UL,
            32_500_000UL,
            4_096);

        yield return PrimeBenchmarkCase.FromRange(
            "≥64_000_000",
            "Sampled admissible odd values ≥ 64,000,000 (k ≤ 5, not divisible by 5)",
            64_000_001UL,
            64_500_000UL,
            4_096);

        yield return PrimeBenchmarkCase.FromRange(
            "≥100_000_000",
            "Sampled admissible odd values ≥ 100,000,000 (k ≤ 5, not divisible by 5)",
            100_000_001UL,
            100_500_000UL,
            4_096);

        yield return PrimeBenchmarkCase.FromRange(
            "≥138_000_000",
            "Sampled admissible odd values ≥ 138,000,000 (k ≤ 5, not divisible by 5)",
            138_000_001UL,
            138_500_000UL,
            4_096);

        yield return PrimeBenchmarkCase.FromRange(
            "≥276_000_000",
            "Sampled admissible odd values ≥ 276,000,000 (k ≤ 5, not divisible by 5)",
            276_000_001UL,
            276_500_000UL,
            4_096);

        yield return PrimeBenchmarkCase.FromRange(
            "≥552_000_000",
            "Sampled admissible odd values ≥ 552,000,000 (k ≤ 5, not divisible by 5)",
            552_000_001UL,
            552_500_000UL,
            4_096);
    }

    public readonly record struct PrimeBenchmarkCase(string Name, string Description, ulong[] Candidates)
    {
        public override string ToString()
        {
            return Name;
        }

        public static PrimeBenchmarkCase FromRange(
            string name,
            string description,
            ulong startInclusive,
            ulong endInclusive,
            int maxCount)
        {
            ulong[] values = CollectRange(startInclusive, endInclusive, maxCount);
            return new PrimeBenchmarkCase(name, description, values);
        }
    }

    private static ulong[] CollectRange(ulong startInclusive, ulong endInclusive, int maxCount)
    {
        maxCount = maxCount < 0 ? 0 : maxCount;
        if (maxCount == 0)
        {
            return [];
        }

        int estimatedLength = (int)Math.Min((ulong)maxCount, endInclusive - startInclusive + 1UL);
        ulong[] buffer = new ulong[estimatedLength];
        int count = 0;

        for (ulong value = startInclusive; value <= endInclusive && count < maxCount; value++)
        {
            if (!IsAdmissibleMersenneDivisorCandidate(value))
            {
                continue;
            }

            if (count == buffer.Length)
            {
                Array.Resize(ref buffer, buffer.Length * 2);
            }

            buffer[count++] = value;
        }

        if (count == buffer.Length)
        {
            return buffer;
        }

        ulong[] result = new ulong[count];
        Array.Copy(buffer, result, count);
        return result;
    }

    private static bool IsAdmissibleMersenneDivisorCandidate(ulong value)
    {
        if ((value & 1UL) == 0UL || value % 5UL == 0UL || value <= 1UL)
        {
            return false;
        }

        ulong qMinusOne = value - 1UL;

        for (ulong k = 1; k <= 5; k++)
        {
            ulong denominator = k << 1;
            if (qMinusOne % denominator != 0UL)
            {
                continue;
            }

            ulong p = qMinusOne / denominator;
            if (p < 2UL)
            {
                continue;
            }

            if (Prime.Numbers.IsPrime(p))
            {
                return true;
            }
        }

        return false;
    }
}
