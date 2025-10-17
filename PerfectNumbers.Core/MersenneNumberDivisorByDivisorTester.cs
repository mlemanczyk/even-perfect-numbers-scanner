using System.Buffers;
using System.Globalization;
using System.Runtime.InteropServices;

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

		candidates.Sort();
		Span<ulong> candidateSpan = CollectionsMarshal.AsSpan(candidates);
		int candidateCount = candidateSpan.Length;

		bool applyStartPrime = startPrime > 0UL;
		int skippedByPreviousResults = 0;

		ArrayPool<ulong> pool = ThreadStaticPools.UlongPool;
		if (previousResults is not null && previousResults.Count > 0)
		{
			int recordedCount = previousResults.Count;
			ulong[] recordedCandidates = pool.Rent(recordedCount);

			Span<ulong> recordedSpan = recordedCandidates.AsSpan(0, recordedCount);
			int recordedIndex = 0;
			foreach (ulong value in previousResults.Keys)
			{
				recordedSpan[recordedIndex++] = value;
			}

			recordedSpan.Sort();

			int writeIndex = 0;
			int skipIndex = 0;

			for (int readIndex = 0; readIndex < candidateCount; readIndex++)
			{
				ulong candidate = candidateSpan[readIndex];

				while (skipIndex < recordedCount && recordedSpan[skipIndex] < candidate)
				{
					skipIndex++;
				}

				if (skipIndex < recordedCount && recordedSpan[skipIndex] == candidate)
				{
					skippedByPreviousResults++;
					skipIndex++;
					continue;
				}

				candidateSpan[writeIndex++] = candidate;
			}

			if (writeIndex < candidateCount)
			{
				candidates.RemoveRange(writeIndex, candidateCount - writeIndex);
				candidateSpan = CollectionsMarshal.AsSpan(candidates);
				candidateCount = candidateSpan.Length;
			}

			pool.Return(recordedCandidates, clearArray: true);
		}

		if (skippedByPreviousResults > 0)
		{
			Console.WriteLine($"Skipped {skippedByPreviousResults.ToString(CultureInfo.InvariantCulture)} candidates excluded by previous results.");
		}

		int startIndex = 0;
		if (applyStartPrime)
		{
			startIndex = candidates.BinarySearch(startPrime);
			if (startIndex < 0)
			{
				startIndex = ~startIndex;
			}
			else
			{
				while (startIndex > 0 && candidates[startIndex - 1] >= startPrime)
				{
					startIndex--;
				}
			}
		}

		List<ulong> primesToTest = candidates;
		Span<ulong> primesSpan = CollectionsMarshal.AsSpan(primesToTest);
		ulong maxPrime = 0UL;
		int primeWriteIndex = 0;

		// This implementation is terribly slow, while this method expect prime p given as --filter-p input already.
		// We don't need to additionally check it.

		// using IEnumerator<ulong> primeEnumerator = Prime.Numbers.GetEnumerator();
		// bool hasPrime = primeEnumerator.MoveNext();
		// ulong currentPrime = hasPrime ? primeEnumerator.Current : 0UL;

		for (int index = startIndex; index < candidateCount; index++)
		{
			ulong candidate = primesSpan[index];

			if (candidate <= 1UL)
			{
				markComposite();
				printResult(candidate, false, false, false);
				continue;
			}

			// This implementation is terribly slow, while this method expect prime p given as --filter-p input already.
			// We don't need to additionally check it.

			// while (hasPrime && currentPrime < candidate)
			// {
			//     hasPrime = primeEnumerator.MoveNext();
			//     if (!hasPrime)
			//     {
			//         break;
			//     }

			//     currentPrime = primeEnumerator.Current;
			// }

			// if (!hasPrime || currentPrime != candidate)
			// if (!Number.IsPrime(candidate))
			// {
			//     markComposite();
			//     printResult(candidate, false, false, false);
			//     continue;
			// }

			primesSpan[primeWriteIndex++] = candidate;

			if (candidate > maxPrime)
			{
				maxPrime = candidate;
			}
		}

		if (primeWriteIndex < primesToTest.Count)
		{
			primesToTest.RemoveRange(primeWriteIndex, primesToTest.Count - primeWriteIndex);
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
		ulong[] primeBatch = pool.Rent(primeCount);
		ulong[] allowedMaxBatch = pool.Rent(primeCount);
		List<ulong> filteredPrimes = new(primeCount);

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

		pool.Return(allowedMaxBatch, clearArray: true);
		pool.Return(primeBatch, clearArray: true);

		if (filteredPrimes.Count == 0)
		{
			if (applyStartPrime)
			{
				Console.WriteLine($"No primes greater than or equal to {startPrime.ToString(CultureInfo.InvariantCulture)} were found for --mersenne=bydivisor.");
			}

			return;
		}

		int workerCount = threadCount <= 0 ? Environment.ProcessorCount : threadCount;
		if (workerCount < 1)
		{
			workerCount = 1;
		}

		void ProcessPrime(ulong prime)
		{
			bool isPrime = tester.IsPrime(prime, out bool divisorsExhausted);

			if (!isPrime)
			{
				markComposite();
				printResult(prime, true, true, false);
				return;
			}

			clearComposite();
			printResult(prime, true, divisorsExhausted, true);
		}

		if (workerCount == 1)
		{
			foreach (ulong prime in filteredPrimes)
			{
				ProcessPrime(prime);
			}
		}
		else
		{
			ParallelOptions options = new()
			{
				MaxDegreeOfParallelism = workerCount,
				TaskScheduler = UnboundedTaskScheduler.Instance
			};

			Parallel.ForEach(filteredPrimes, options, ProcessPrime);
		}
	}
}
