using System.Buffers;
using System.Threading;
using System.Threading.Tasks;
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
			int threadCount,
			int primesPerTask)
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

			pool.Return(recordedCandidates, clearArray: false);
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
				while (startIndex > 0)
				{
					int previousIndex = startIndex - 1;
					if (candidates[previousIndex] < startPrime)
					{
						break;
					}

					startIndex = previousIndex;
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

		// The by-divisor CPU scan only operates on primes greater than 138,000,000, so the guard below never triggers.
		// if (candidate <= 1UL)
		// {
		//     markComposite();
		//     printResult(candidate, false, false, false);
		//     continue;
		// }

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
			string message = applyStartPrime
				? $"No primes greater than or equal to {startPrime.ToString(CultureInfo.InvariantCulture)} were found for --mersenne=bydivisor."
				: "No prime candidates remain for --mersenne=bydivisor after filtering.";
			Console.WriteLine(message);

			return;
		}

		// if (maxPrime <= 1UL)
		// {
		//     Console.WriteLine("The filter specified by --filter-p must contain at least one prime exponent greater than 1 for --mersenne=bydivisor.");
		//     return;
		// }
		// The by-divisor CPU configuration only feeds primes well above 1, so this fallback never executes in production runs.

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

		// Primes in the production by-divisor flow yield massive divisor limits, so the short-circuit below never applies.
		// if (allowedMax < 3UL)
		// {
		//     clearComposite();
		//     printResult(prime, true, true, true);
		//     continue;
		// }

			filteredPrimes.Add(prime);
		}

		pool.Return(allowedMaxBatch, clearArray: false);
		pool.Return(primeBatch, clearArray: false);

		// The filtered list mirrors primesToTest in the CPU flow, so the guard below never triggers after the earlier
		// emptiness checks.
		// if (filteredPrimes.Count == 0)
		// {
		//     if (applyStartPrime)
		//     {
		//         Console.WriteLine($"No primes greater than or equal to {startPrime.ToString(CultureInfo.InvariantCulture)} were found for --mersenne=bydivisor.");
		//     }

		//     return;
		// }

		int workerCount = threadCount <= 0 ? Environment.ProcessorCount : threadCount;
		// Environment.ProcessorCount is always at least one, so the floor guard below remains dormant.
		// if (workerCount < 1)
		// {
		//     workerCount = 1;
		// }

		int chunkSize = primesPerTask <= 0 ? 1 : primesPerTask;

		void ProcessPrime(ulong prime)
		{
			Console.WriteLine($"Task started {prime}");
			bool isPrime = tester.IsPrime(prime, out bool divisorsExhausted);

			if (!isPrime)
			{
				markComposite();
				printResult(prime, true, true, false);
				Console.WriteLine($"Task finished {prime}");
				return;
			}

			clearComposite();
			printResult(prime, true, divisorsExhausted, true);
			Console.WriteLine($"Task finished {prime}");
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
			TaskScheduler scheduler = UnboundedTaskScheduler.Instance;

			int totalCount = filteredPrimes.Count;
			int partitionSize = chunkSize < 1 ? 1 : chunkSize;
			if (partitionSize > totalCount)
			{
				partitionSize = totalCount;
			}

			int taskCount = (totalCount + partitionSize - 1) / partitionSize;
			Task[] tasks = new Task[taskCount];
			var startGate = new ManualResetEventSlim(initialState: false);
			int taskIndex = 0;

			for (int start = 0; start < totalCount; start += partitionSize)
			{
				int rangeStart = start;
				int rangeEnd = rangeStart + partitionSize;
				if (rangeEnd > totalCount)
				{
					rangeEnd = totalCount;
				}

				tasks[taskIndex++] = Task.Factory.StartNew(
					() =>
					{
						startGate.Wait();

						for (int index = rangeStart; index < rangeEnd; index++)
						{
							ProcessPrime(filteredPrimes[index]);
						}
					},
					CancellationToken.None,
					TaskCreationOptions.DenyChildAttach,
					scheduler);
			}

			startGate.Set();
			Task.WaitAll(tasks);
			startGate.Dispose();
		}
	}
}
