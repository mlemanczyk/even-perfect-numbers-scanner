using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using Open.Numeric.Primes;
using PerfectNumbers.Core;
using PerfectNumbers.Core.Gpu.Accelerators;

namespace EvenPerfectBitScanner.Benchmarks;

[ShortRunJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
public class PrimeTesterBenchmarks
{
    private static readonly HeuristicPrimeTester _heuristicTester = new();
    private ulong[] _candidates = [];

    [ParamsSource(nameof(GetBenchmarkCases))]
    public PrimeBenchmarkCase BenchmarkCase { get; set; }

    [GlobalSetup]
    public void GlobalSetup()
    {
        _candidates = BenchmarkCase.Candidates;

        if (_candidates.Length == 0)
        {
            return;
        }

		PerfectNumberConstants.RollingAccelerators = 1;

		PrimeOrderCalculatorAccelerator.WarmUp();
		// HeuristicGroupABPrimeTesterAccelerator.WarmUp();
		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);

        _ = HeuristicPrimeTester.IsPrimeCpu(_candidates[0]);
        _ = HeuristicPrimeTester.IsPrimeGpu(gpu, _candidates[0]);
        _ = Prime.Numbers.IsPrime(_candidates[0]);
		PrimeOrderCalculatorAccelerator.Return(gpu);
    }

    // [Benchmark]
    public int HeuristicCpu()
    {
        ulong[] values = _candidates;
        int primeCount = 0;

        for (int i = 0; i < values.Length; i++)
        {
            if (HeuristicPrimeTester.IsPrimeCpu(values[i]))
            {
                primeCount++;
            }
        }

        return primeCount;
    }

    // [Benchmark]
    public int HeuristicGpu()
    {
        ulong[] values = _candidates;
        int primeCount = 0;
		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);

        for (int i = 0; i < values.Length; i++)
        {
            if (HeuristicPrimeTester.IsPrimeGpu(gpu, values[i]))
            {
                primeCount++;
            }
        }

		PrimeOrderCalculatorAccelerator.Return(gpu);
        return primeCount;
    }

    // [Benchmark]
    public int HeuristicCombinedGpu()
    {
        ulong[] values = _candidates;
        int primeCount = 0;

		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
        for (int i = 0; i < values.Length; i++)
        {
            if (HeuristicCombinedPrimeTester.IsPrimeGpu(gpu, values[i]))
            {
                primeCount++;
            }
        }

        return primeCount;
    }

    // [Benchmark]
    public int HeuristicCombinedCpu()
    {
        ulong[] values = _candidates;
        int primeCount = 0;

        for (int i = 0; i < values.Length; i++)
        {
            if (HeuristicCombinedPrimeTester.IsPrimeCpu(values[i]))
            {
                primeCount++;
            }
        }

        return primeCount;
    }

    // [Benchmark]
    public int NonHeuristicGpu()
	{
        ulong[] values = _candidates;
        int primeCount = 0;

		var gpu = PrimeOrderCalculatorAccelerator.Rent(1);
        for (int i = 0; i < values.Length; i++)
        {
            if (PrimeTester.IsPrimeGpu(gpu, values[i]))
            {
                primeCount++;
            }
        }

		PrimeOrderCalculatorAccelerator.Return(gpu);
        return primeCount;
    }

    [Benchmark(Baseline = true)]
    public int NonHeuristicCpu()
    {
        ulong[] values = _candidates;
        int primeCount = 0;

        for (int i = 0; i < values.Length; i++)
        {
            if (PrimeTester.IsPrimeCpu(values[i]))
            {
                primeCount++;
            }
        }

        return primeCount;
    }

    [Benchmark]
    public int ByLastDigitCpu()
    {
        ulong[] values = _candidates;
        int primeCount = 0;

        for (int i = 0; i < values.Length; i++)
        {
            if (PrimeTesterByLastDigit.IsPrimeCpu(values[i]))
            {
                primeCount++;
            }
        }

        return primeCount;
    }

    [Benchmark]
    public int SevenOrOthersCpu()
    {
        ulong[] values = _candidates;
        int primeCount = 0;

        for (int i = 0; i < values.Length; i++)
        {
            if (PrimeTesterSeverOrOthers.IsPrimeCpu(values[i]))
            {
                primeCount++;
            }
        }

        return primeCount;
    }

    // [Benchmark]
    public int OpenNumericPrimes()
    {
        ulong[] values = _candidates;
        int primeCount = 0;

        for (int i = 0; i < values.Length; i++)
        {
            if (Prime.Numbers.IsPrime(values[i]))
            {
                primeCount++;
            }
        }

        return primeCount;
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
