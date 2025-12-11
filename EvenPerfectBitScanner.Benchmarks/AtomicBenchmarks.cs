using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using ILGPU;
using Open.Numeric.Primes;

namespace EvenPerfectBitScanner.Benchmarks;

[ShortRunJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
public class AtomicBenchmarks
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

    [Benchmark]
    public int AtomicAddInt()
	{
        int[] values = _candidatesInt;
        int primeCount;
		int oldValue = 0;
        for (int i = 0; i < values.Length; i++)
        {
			primeCount = values[i];
			oldValue ^= Atomic.Add(ref primeCount, 1);
        }

        return oldValue;
	}

    [Benchmark(Baseline = true)]
    public int InterlockedIncInt()
    {
        int[] values = _candidatesInt;
        int primeCount;
		int oldValue = 0;
        for (int i = 0; i < values.Length; i++)
        {
			primeCount = values[i];
			oldValue ^= Interlocked.Increment(ref primeCount);
        }

        return oldValue;
    }

    [Benchmark]
    public ulong AtomicAddUlong()
	{
        ulong[] values = _candidates;
        ulong primeCount;
		ulong oldValue = 0UL;
        for (int i = 0; i < values.Length; i++)
        {
			primeCount = values[i];
			oldValue ^= Atomic.Add(ref primeCount, 1);
        }

        return oldValue;
	}

    [Benchmark]
    public ulong InterlockedIncUlong()
    {
        ulong[] values = _candidates;
        ulong primeCount;
		ulong oldValue = 0UL;
        for (int i = 0; i < values.Length; i++)
        {
			primeCount = values[i];
			oldValue ^= Interlocked.Increment(ref primeCount);
        }

        return oldValue;
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
