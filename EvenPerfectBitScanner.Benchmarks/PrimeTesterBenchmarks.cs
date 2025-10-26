using System;
using System.Collections.Generic;
using System.Threading;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using Open.Numeric.Primes;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[ShortRunJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
public class PrimeTesterBenchmarks
{
    private HeuristicPrimeTester _cpuTester = null!;
    private HeuristicPrimeTester _gpuTester = null!;
    private ulong[] _candidates = [];

    [ParamsSource(nameof(GetBenchmarkCases))]
    public PrimeBenchmarkCase BenchmarkCase { get; set; }

    [GlobalSetup]
    public void GlobalSetup()
    {
        _cpuTester = new HeuristicPrimeTester();
        _gpuTester = new HeuristicPrimeTester();
        _candidates = BenchmarkCase.Candidates;

        if (_candidates.Length == 0)
        {
            return;
        }

        CancellationToken cancellationToken = CancellationToken.None;
        _cpuTester.IsPrimeCpu(_candidates[0], cancellationToken);
        _gpuTester.IsPrimeGpu(_candidates[0]);
        _ = Prime.Numbers.IsPrime(_candidates[0]);
    }

    [Benchmark(Baseline = true)]
    public int HeuristicCpu()
    {
        HeuristicPrimeTester tester = _cpuTester;
        ulong[] values = _candidates;
        CancellationToken cancellationToken = CancellationToken.None;
        int primeCount = 0;

        for (int i = 0; i < values.Length; i++)
        {
            if (tester.IsPrimeCpu(values[i], cancellationToken))
            {
                primeCount++;
            }
        }

        return primeCount;
    }

    [Benchmark]
    public int HeuristicGpu()
    {
        HeuristicPrimeTester tester = _gpuTester;
        ulong[] values = _candidates;
        CancellationToken cancellationToken = CancellationToken.None;
        int primeCount = 0;

        for (int i = 0; i < values.Length; i++)
        {
            if (tester.IsPrimeGpu(values[i]))
            {
                primeCount++;
            }
        }

        return primeCount;
    }

    // This benchmark is very slow. No point in comparing it.
    // [Benchmark]
    public int NonHeuristicGpu()
    {
        // PrimeTester tester = _gpuTester;
        ulong[] values = _candidates;
        CancellationToken cancellationToken = CancellationToken.None;
        int primeCount = 0;

        for (int i = 0; i < values.Length; i++)
        {
            if (PrimeTester.Exclusive.IsPrimeGpu(values[i], cancellationToken))
            {
                primeCount++;
            }
        }

        return primeCount;
    }

    [Benchmark]
    public int NonHeuristicCpu()
    {
        // PrimeTester tester = _gpuTester;
        ulong[] values = _candidates;
        CancellationToken cancellationToken = CancellationToken.None;
        int primeCount = 0;

        for (int i = 0; i < values.Length; i++)
        {
            if (PrimeTester.IsPrimeCp(values[i], cancellationToken))
            {
                primeCount++;
            }
        }

        return primeCount;
    }

    [Benchmark]
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
            "≥138_000_000",
            "Sampled admissible odd values ≥ 138,000,000 (k ≤ 5, not divisible by 5)",
            138_000_001UL,
            138_500_000UL,
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
