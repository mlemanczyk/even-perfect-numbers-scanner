using System;
using System.Buffers;
using System.Collections.Generic;
using System.Globalization;
using Open.Numeric.Primes;

namespace PerfectNumbers.Core;

public static class MersenneNumberDivisorByDivisorTester
{
    public static void Run(
            List<ulong> candidates,
            IMersenneNumberDivisorByDivisorTester tester,
            Dictionary<ulong, (bool DetailedCheck, bool PassedAllTests)>? previousResults,
            ulong startPrime,
            Action markComposite,
            Action clearComposite,
            Action<ulong, bool, bool, bool> printResult,
            int threadCount)
    {
        if (candidates.Count == 0)
        {
            Console.WriteLine("No candidates were provided for --mersenne=bydivisor.");
            return;
        }

        _ = threadCount;

        bool applyStartPrime = startPrime > 0UL;
        int skippedByPreviousResults = 0;
        List<ulong> primesToTest = new(candidates.Count);
        ulong maxPrime = 0UL;

        using IEnumerator<ulong> primeEnumerator = Prime.Numbers.GetEnumerator();
        bool hasPrime = primeEnumerator.MoveNext();
        ulong currentPrime = hasPrime ? primeEnumerator.Current : 0UL;

        for (int index = 0; index < candidates.Count; index++)
        {
            ulong candidate = candidates[index];

            if (applyStartPrime && candidate < startPrime)
            {
                continue;
            }

            if (previousResults is not null && previousResults.ContainsKey(candidate))
            {
                skippedByPreviousResults++;
                continue;
            }

            if (candidate <= 1UL)
            {
                markComposite();
                printResult(candidate, false, false, false);
                continue;
            }

            while (hasPrime && currentPrime < candidate)
            {
                hasPrime = primeEnumerator.MoveNext();
                if (!hasPrime)
                {
                    break;
                }

                currentPrime = primeEnumerator.Current;
            }

            if (!hasPrime || currentPrime != candidate)
            {
                markComposite();
                printResult(candidate, false, false, false);
                continue;
            }

            if (candidate > maxPrime)
            {
                maxPrime = candidate;
            }

            primesToTest.Add(candidate);
        }

        if (skippedByPreviousResults > 0)
        {
            Console.WriteLine($"Skipped {skippedByPreviousResults.ToString(CultureInfo.InvariantCulture)} candidates excluded by previous results.");
        }

        if (primesToTest.Count == 0)
        {
            if (applyStartPrime)
            {
                Console.WriteLine($"No primes greater than or equal to {startPrime.ToString(CultureInfo.InvariantCulture)} were found for --mersenne=bydivisor.");
            }
            else
            {
                Console.WriteLine("No prime candidates remain for --mersenne=bydivisor after filtering.");
            }

            return;
        }

        if (maxPrime <= 1UL)
        {
            Console.WriteLine("The filter specified by --filter-p must contain at least one prime exponent greater than 1 for --mersenne=bydivisor.");
            return;
        }

        tester.ConfigureFromMaxPrime(maxPrime);

        int primeCount = primesToTest.Count;
        ulong[] primeBatch = ArrayPool<ulong>.Shared.Rent(primeCount);
        ulong[] allowedMaxBatch = ArrayPool<ulong>.Shared.Rent(primeCount);
        List<ulong> filteredPrimes = new(primeCount);

        try
        {
            Span<ulong> primeSpan = primeBatch.AsSpan(0, primeCount);
            Span<ulong> allowedMaxSpan = allowedMaxBatch.AsSpan(0, primeCount);

            for (int i = 0; i < primeCount; i++)
            {
                primeSpan[i] = primesToTest[i];
            }

            tester.PrepareCandidates(primeSpan, allowedMaxSpan);

            for (int i = 0; i < primeCount; i++)
            {
                ulong prime = primeSpan[i];
                ulong allowedMax = allowedMaxSpan[i];

                if (allowedMax < 3UL)
                {
                    clearComposite();
                    printResult(prime, true, true, true);
                    continue;
                }

                filteredPrimes.Add(prime);
            }
        }
        finally
        {
            ArrayPool<ulong>.Shared.Return(allowedMaxBatch, clearArray: true);
            ArrayPool<ulong>.Shared.Return(primeBatch, clearArray: true);
        }

        if (filteredPrimes.Count == 0)
        {
            if (applyStartPrime)
            {
                Console.WriteLine($"No primes greater than or equal to {startPrime.ToString(CultureInfo.InvariantCulture)} were found for --mersenne=bydivisor.");
            }

            return;
        }

        foreach (ulong prime in filteredPrimes)
        {
            bool isPrime = tester.IsPrime(prime, out bool divisorsExhausted);

            if (!isPrime)
            {
                markComposite();
                printResult(prime, true, true, false);
                continue;
            }

            clearComposite();
            printResult(prime, true, divisorsExhausted, true);
        }
    }
}
