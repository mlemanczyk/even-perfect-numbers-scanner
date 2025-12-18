using System.Buffers;
using System.Globalization;
using System.Numerics;
using System.Runtime.InteropServices;
using PerfectNumbers.Core.Gpu;
using PerfectNumbers.Core.Gpu.Accelerators;
using PerfectNumbers.Core.Cpu;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public static class MersenneNumberDivisorByDivisorTester
{
	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static void Run(
		List<ulong> candidates,
		ref MersenneNumberDivisorByDivisorCpuTesterWithCpuOrder tester,
		Dictionary<ulong, (bool DetailedCheck, bool PassedAllTests)>? previousResults,
		ulong startPrime,
		Action markComposite,
		Action clearComposite,
		Action<ulong, bool, bool, bool, BigInteger> printResult,
		int threadCount,
		int primesPerTask)
		=> RunStructTester(
			candidates,
			ref tester,
			previousResults,
			startPrime,
			markComposite,
			clearComposite,
			printResult,
			threadCount,
			primesPerTask,
			static (
				List<ulong> filteredPrimes,
				in MersenneNumberDivisorByDivisorCpuTesterWithCpuOrder prototype,
				int workerCount,
				int partitionSize,
				Action mc,
				Action cc,
				Action<ulong, bool, bool, bool, BigInteger> pr) =>
				RunCpuOrderFiltered(filteredPrimes, prototype, workerCount, partitionSize, mc, cc, pr));

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static void Run(
		List<ulong> candidates,
		ref MersenneNumberDivisorByDivisorCpuTesterWithHybridOrder tester,
		Dictionary<ulong, (bool DetailedCheck, bool PassedAllTests)>? previousResults,
		ulong startPrime,
		Action markComposite,
		Action clearComposite,
		Action<ulong, bool, bool, bool, BigInteger> printResult,
		int threadCount,
		int primesPerTask)
		=> RunStructTester(
			candidates,
			ref tester,
			previousResults,
			startPrime,
			markComposite,
			clearComposite,
			printResult,
			threadCount,
			primesPerTask,
			static (
				List<ulong> filteredPrimes,
				in MersenneNumberDivisorByDivisorCpuTesterWithHybridOrder prototype,
				int workerCount,
				int partitionSize,
				Action mc,
				Action cc,
				Action<ulong, bool, bool, bool, BigInteger> pr) =>
				RunHybridOrderFiltered(filteredPrimes, prototype, workerCount, partitionSize, mc, cc, pr));

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static void Run(
		List<ulong> candidates,
		ref MersenneNumberDivisorByDivisorCpuTesterWithGpuOrder tester,
		Dictionary<ulong, (bool DetailedCheck, bool PassedAllTests)>? previousResults,
		ulong startPrime,
		Action markComposite,
		Action clearComposite,
		Action<ulong, bool, bool, bool, BigInteger> printResult,
		int threadCount,
		int primesPerTask)
		=> RunStructTester(
			candidates,
			ref tester,
			previousResults,
			startPrime,
			markComposite,
			clearComposite,
			printResult,
			threadCount,
			primesPerTask,
			static (
				List<ulong> filteredPrimes,
				in MersenneNumberDivisorByDivisorCpuTesterWithGpuOrder prototype,
				int workerCount,
				int partitionSize,
				Action mc,
				Action cc,
				Action<ulong, bool, bool, bool, BigInteger> pr) =>
				RunGpuOrderFiltered(filteredPrimes, prototype, workerCount, partitionSize, mc, cc, pr));

	internal static bool TryReadLastSavedK(string stateFile, out BigInteger lastSavedK)
	{
		lastSavedK = BigInteger.Zero;
		try
		{
			bool any = false;
			foreach (string line in File.ReadLines(stateFile))
			{
				if (string.IsNullOrWhiteSpace(line))
				{
					continue;
				}

				if (BigInteger.TryParse(line, NumberStyles.None, CultureInfo.InvariantCulture, out BigInteger parsed))
				{
					if (!any || parsed > lastSavedK)
					{
						lastSavedK = parsed;
					}

					any = true;
				}
			}

			return any;
		}
		catch (IOException)
		{
			return false;
		}
		catch (UnauthorizedAccessException)
		{
			return false;
		}
	}

	public static byte CheckDivisor(ulong prime, ulong divisorCycle, in MontgomeryDivisorData divisorData)
	{
		ulong residue = prime.Pow2MontgomeryModWithCycleConvertToStandardCpu(divisorCycle, divisorData);
		return residue == 1UL ? (byte)1 : (byte)0;
	}

	public static bool IsProbablePrimeBigInteger(BigInteger value)
	{
		if (value <= 3)
		{
			return value >= 2;
		}

		if (value.IsEven)
		{
			return false;
		}

		BigInteger d = value - 1;
		int s = 0;
		while ((d & 1) == 0)
		{
			d >>= 1;
			s++;
		}

		ReadOnlySpan<int> bases = stackalloc int[] { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 };
		for (int i = 0; i < bases.Length; i++)
		{
			int baseValue = bases[i];
			if ((BigInteger)baseValue >= value)
			{
				continue;
			}

			var expStepper = new BigIntegerExponentStepper(value, baseValue);
			BigInteger x = expStepper.Initialize(d);
			if (x == 1 || x == value - 1)
			{
				continue;
			}

			bool witnessFound = true;
			for (int r = 1; r < s; r++)
			{
				BigInteger targetExponent = d << r;
				x = expStepper.ComputeNext(targetExponent);
				if (x == value - 1)
				{
					witnessFound = false;
					break;
				}
			}

			if (witnessFound)
			{
				return false;
			}
		}

		return true;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static ulong GetAllowedDivisorLimitForSpan(BigInteger divisorLimit)
	{
		if (divisorLimit.IsZero)
		{
			return ulong.MaxValue;
		}

		return divisorLimit > ulong.MaxValue ? ulong.MaxValue : (ulong)divisorLimit;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static BigInteger ComputeDivisorLimitFromMaxPrimeBig(ulong maxPrime)
	{
		if (maxPrime <= 1UL)
		{
			return BigInteger.Zero;
		}

		if (maxPrime - 1UL >= 64UL)
		{
			return (BigInteger.One << 256) - BigInteger.One;
		}

		return (BigInteger.One << (int)(maxPrime - 1UL)) - BigInteger.One;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static BigInteger ComputeAllowedMaxDivisorBig(ulong prime, BigInteger divisorLimit)
	{
		if (prime <= 1UL)
		{
			return BigInteger.Zero;
		}

		if (divisorLimit.IsZero || prime - 1UL >= 64UL)
		{
			return divisorLimit;
		}

		BigInteger candidateLimit = (BigInteger.One << (int)(prime - 1UL)) - BigInteger.One;
		if (divisorLimit.IsZero)
		{
			return candidateLimit;
		}

		return BigInteger.Min(candidateLimit, divisorLimit);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static ulong ComputeAllowedMaxDivisor(ulong prime, BigInteger divisorLimit)
	{
		BigInteger allowed = ComputeAllowedMaxDivisorBig(prime, divisorLimit);
		if (allowed.IsZero)
		{
			return ulong.MaxValue;
		}

		return allowed > ulong.MaxValue ? ulong.MaxValue : (ulong)allowed;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static bool TryCalculateCycleLengthForExponentCpu(
		ulong divisor,
		ulong exponent,
		in MontgomeryDivisorData divisorData,
		out ulong cycleLength,
		out bool primeOrderFailed) => MersenneDivisorCycles.TryCalculateCycleLengthForExponentCpu(
			divisor,
			exponent,
			divisorData,
			out cycleLength,
			out primeOrderFailed);

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static bool TryCalculateCycleLengthForExponentHybrid(
		ulong divisor,
		PrimeOrderCalculatorAccelerator gpu,
		ulong exponent,
		in MontgomeryDivisorData divisorData,
		out ulong cycleLength,
		out bool primeOrderFailed) => MersenneDivisorCycles.TryCalculateCycleLengthForExponentHybrid(
			gpu,
			divisor,
			exponent,
			divisorData,
			out cycleLength,
			out primeOrderFailed);

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static bool TryCalculateCycleLengthForExponentGpu(
		ulong divisor,
		PrimeOrderCalculatorAccelerator gpu,
		ulong exponent,
		in MontgomeryDivisorData divisorData,
		out ulong cycleLength,
		out bool primeOrderFailed) => MersenneDivisorCycles.TryCalculateCycleLengthForExponentGpu(
			gpu,
			divisor,
			exponent,
			divisorData,
			out cycleLength,
			out primeOrderFailed);

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static ulong CalculateCycleLengthCpu(
		ulong divisor,
		in MontgomeryDivisorData divisorData,
		bool skipPrimeOrderHeuristic) => MersenneDivisorCycles.CalculateCycleLengthCpu(
			divisor,
			divisorData,
			skipPrimeOrderHeuristic);

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static ulong CalculateCycleLengthHybrid(
		ulong divisor,
		PrimeOrderCalculatorAccelerator gpu,
		in MontgomeryDivisorData divisorData,
		bool skipPrimeOrderHeuristic) => MersenneDivisorCycles.CalculateCycleLengthHybrid(
			gpu,
			divisor,
			divisorData,
			skipPrimeOrderHeuristic);

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static ulong CalculateCycleLengthGpu(
		ulong divisor,
		PrimeOrderCalculatorAccelerator gpu,
		in MontgomeryDivisorData divisorData,
		bool skipPrimeOrderHeuristic) => MersenneDivisorCycles.CalculateCycleLengthGpu(
			gpu,
			divisor,
			divisorData,
			skipPrimeOrderHeuristic);

	public static void Run(
		List<ulong> candidates,
		IMersenneNumberDivisorByDivisorTester tester,
		Dictionary<ulong, (bool DetailedCheck, bool PassedAllTests)>? previousResults,
		ulong startPrime,
		Action markComposite,
		Action clearComposite,
		Action<ulong, bool, bool, bool, BigInteger> printResult,
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

		int primeCount = primesToTest.Count;
		ulong[] primeBatch = pool.Rent(primeCount);
		ulong[] allowedMaxBatch = pool.Rent(primeCount);
		List<ulong> filteredPrimes = new(primeCount);

		Span<ulong> primeSpan = primeBatch.AsSpan(0, primeCount);
		primesToTest.CopyTo(primeSpan);
		Span<ulong> allowedMaxSpan = allowedMaxBatch.AsSpan(0, primeCount);

		// for (int i = 0; i < primeCount; i++)
		// {
		// 	primeSpan[i] = primesToTest[i];
		// }

		switch (tester)
		{
			case MersenneNumberDivisorByDivisorCpuTesterWithCpuOrder typedCpu:
				typedCpu.PrepareCandidates(maxPrime, primeSpan, allowedMaxSpan);
				tester = typedCpu;
				break;

			case MersenneNumberDivisorByDivisorCpuTesterWithHybridOrder typedHybrid:
				typedHybrid.PrepareCandidates(maxPrime, primeSpan, allowedMaxSpan);
				tester = typedHybrid;
				break;

			case MersenneNumberDivisorByDivisorCpuTesterWithGpuOrder typedGpuOrder:
				typedGpuOrder.PrepareCandidates(maxPrime, primeSpan, allowedMaxSpan);
				tester = typedGpuOrder;
				break;

			default:
				tester.PrepareCandidates(maxPrime, primeSpan, allowedMaxSpan);
				break;
		}

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
		int partitionSize = chunkSize < 1 ? 1 : chunkSize;

		if (tester is MersenneNumberDivisorByDivisorCpuTesterWithCpuOrder cpuOrderPrototype)
		{
			RunCpuOrderFiltered(filteredPrimes, cpuOrderPrototype, workerCount, partitionSize, markComposite, clearComposite, printResult);
			return;
		}

		if (tester is MersenneNumberDivisorByDivisorCpuTesterWithHybridOrder hybridOrderPrototype)
		{
			RunHybridOrderFiltered(filteredPrimes, hybridOrderPrototype, workerCount, partitionSize, markComposite, clearComposite, printResult);
			return;
		}

		if (tester is MersenneNumberDivisorByDivisorCpuTesterWithGpuOrder gpuOrderPrototype)
		{
			RunGpuOrderFiltered(filteredPrimes, gpuOrderPrototype, workerCount, partitionSize, markComposite, clearComposite, printResult);
			return;
		}

		if (tester is MersenneNumberDivisorByDivisorGpuTester gpuPrototype)
		{
			RunGpuTesterFiltered(filteredPrimes, gpuPrototype, workerCount, partitionSize, markComposite, clearComposite, printResult);
			return;
		}

		if (workerCount > 1)
		{
			throw new InvalidOperationException($"Custom by-divisor tester {tester.GetType().Name} must be thread-safe or run with --threads=1.");
		}

		void ProcessCustomPrime(ulong prime)
		{
			try
			{
				Console.WriteLine($"Processing {prime}");
				string stateFile = Path.Combine(
					PerfectNumberConstants.ByDivisorStateDirectory,
					prime.ToString(CultureInfo.InvariantCulture) + ".bin");

				BigInteger resumeK = EnvironmentConfiguration.MinK;
				if (File.Exists(stateFile))
				{
					if (TryReadLastSavedK(stateFile, out BigInteger lastK))
					{
						resumeK = lastK + BigInteger.One;
						tester.ResumeFromState(stateFile, lastK, resumeK);
					}
					else
					{
						tester.ResumeFromState(stateFile, BigInteger.Zero, resumeK);
					}
				}
				else
				{
					tester.ResumeFromState(stateFile, BigInteger.Zero, resumeK);
				}

				tester.ResetStateTracking();

				bool isPrime = tester.IsPrime(prime, out bool divisorsExhausted, out BigInteger divisor);

				if (!isPrime)
				{
					if (!string.IsNullOrEmpty(stateFile) && File.Exists(stateFile))
					{
						File.Delete(stateFile);
					}

					markComposite();
					printResult(prime, true, true, false, divisor);
					Console.WriteLine($"Finished processing {prime}");
					return;
				}

				clearComposite();
				printResult(prime, true, divisorsExhausted, true, divisor);
				Console.WriteLine($"Finished processing {prime}");
			}
			catch (Exception ex)
			{
				Console.WriteLine($"Error processing {ex.Message} {ex.StackTrace}");
				Environment.Exit(1);
			}
		}

		DeterministicRandomCpu.Initialize();
		foreach (ulong prime in filteredPrimes)
		{
			ProcessCustomPrime(prime);
		}
    }

	private delegate void RunPreparedCandidates<TTester>(
		List<ulong> filteredPrimes,
		in TTester prototype,
		int workerCount,
		int partitionSize,
		Action markComposite,
		Action clearComposite,
		Action<ulong, bool, bool, bool, BigInteger> printResult)
		where TTester : struct;

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private static void RunStructTester<TTester>(
		List<ulong> candidates,
		ref TTester tester,
		Dictionary<ulong, (bool DetailedCheck, bool PassedAllTests)>? previousResults,
		ulong startPrime,
		Action markComposite,
		Action clearComposite,
		Action<ulong, bool, bool, bool, BigInteger> printResult,
		int threadCount,
		int primesPerTask,
		RunPreparedCandidates<TTester> runPrepared)
		where TTester : struct, IMersenneNumberDivisorByDivisorTester
	{
		List<ulong> filteredPrimes = PrepareCandidatesAndGetFilteredPrimes(
			candidates,
			ref tester,
			previousResults,
			startPrime,
			threadCount,
			primesPerTask,
			out int workerCount,
			out int partitionSize);

		if (filteredPrimes.Count == 0)
		{
			return;
		}

		TTester prototype = tester;
		runPrepared(filteredPrimes, prototype, workerCount, partitionSize, markComposite, clearComposite, printResult);
	}

	private static List<ulong> PrepareCandidatesAndGetFilteredPrimes<TTester>(
		List<ulong> candidates,
		ref TTester tester,
		Dictionary<ulong, (bool DetailedCheck, bool PassedAllTests)>? previousResults,
		ulong startPrime,
		int threadCount,
		int primesPerTask,
		out int workerCount,
		out int partitionSize)
		where TTester : struct, IMersenneNumberDivisorByDivisorTester
	{
		if (candidates.Count == 0)
		{
			workerCount = 0;
			partitionSize = 0;
			return [];
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

		for (int index = startIndex; index < candidateCount; index++)
		{
			ulong candidate = primesSpan[index];
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

			workerCount = 0;
			partitionSize = 0;
			return [];
		}

		int primeCount = primesToTest.Count;
		ulong[] primeBatch = pool.Rent(primeCount);
		ulong[] allowedMaxBatch = pool.Rent(primeCount);
		List<ulong> filteredPrimes = new(primeCount);

		Span<ulong> primeSpan = primeBatch.AsSpan(0, primeCount);
		primesToTest.CopyTo(primeSpan);
		Span<ulong> allowedMaxSpan = allowedMaxBatch.AsSpan(0, primeCount);

		tester.PrepareCandidates(maxPrime, primeSpan, allowedMaxSpan);

		for (int i = 0; i < primeCount; i++)
		{
			ulong prime = primeSpan[i];
			filteredPrimes.Add(prime);
		}

		pool.Return(allowedMaxBatch, clearArray: false);
		pool.Return(primeBatch, clearArray: false);

		workerCount = threadCount <= 0 ? Environment.ProcessorCount : threadCount;
		partitionSize = primesPerTask <= 0 ? 1 : primesPerTask;
		return filteredPrimes;
	}

	private static void RunCpuOrderFiltered(
		List<ulong> filteredPrimes,
		in MersenneNumberDivisorByDivisorCpuTesterWithCpuOrder prototype,
		int workerCount,
		int partitionSize,
		Action markComposite,
		Action clearComposite,
		Action<ulong, bool, bool, bool, BigInteger> printResult)
	{
		MersenneNumberDivisorByDivisorCpuTesterWithCpuOrder prototypeCopy = prototype;
		int totalCount = filteredPrimes.Count;
		if (partitionSize > totalCount)
		{
			partitionSize = totalCount;
		}

		if (workerCount == 1)
		{
			DeterministicRandomCpu.Initialize();
			var session = new MersenneNumberDivisorByDivisorPrimeScanSessionWithCpuOrder(prototypeCopy, markComposite, clearComposite, printResult);
			foreach (ulong prime in filteredPrimes)
			{
				session.ProcessPrime(prime);
			}

			session.Dispose();
			return;
		}

		TaskScheduler scheduler = UnboundedTaskScheduler.Instance;
		int nextIndex = 0;
		Task[] tasks = new Task[workerCount];
		var startGate = new ManualResetEventSlim(initialState: false);

		for (int taskIndex = 0; taskIndex < workerCount; taskIndex++)
		{
			int workerIndex = taskIndex;
			tasks[taskIndex] = Task.Factory.StartNew(
				() =>
				{
					DeterministicRandomCpu.Initialize();
					var session = new MersenneNumberDivisorByDivisorPrimeScanSessionWithCpuOrder(prototypeCopy, markComposite, clearComposite, printResult);
					startGate.Wait();
					Console.WriteLine($"Task started for worker {workerIndex.ToString(CultureInfo.InvariantCulture)}");

					while (true)
					{
						int start = Interlocked.Add(ref nextIndex, partitionSize) - partitionSize;
						if (start >= totalCount)
						{
							break;
						}

						int end = start + partitionSize;
						if (end > totalCount)
						{
							end = totalCount;
						}

						for (int index = start; index < end; index++)
						{
							session.ProcessPrime(filteredPrimes[index]);
						}
					}

					session.Dispose();
					Console.WriteLine($"Task finished for worker {workerIndex.ToString(CultureInfo.InvariantCulture)}");
				},
				CancellationToken.None,
				TaskCreationOptions.DenyChildAttach,
				scheduler);
		}

		startGate.Set();
		Task.WaitAll(tasks);
		startGate.Dispose();
	}

	private static void RunHybridOrderFiltered(
		List<ulong> filteredPrimes,
		in MersenneNumberDivisorByDivisorCpuTesterWithHybridOrder prototype,
		int workerCount,
		int partitionSize,
		Action markComposite,
		Action clearComposite,
		Action<ulong, bool, bool, bool, BigInteger> printResult)
	{
		MersenneNumberDivisorByDivisorCpuTesterWithHybridOrder prototypeCopy = prototype;
		int totalCount = filteredPrimes.Count;
		if (partitionSize > totalCount)
		{
			partitionSize = totalCount;
		}

		int effectiveWorkerCount = workerCount;
		int gpuLimit = GpuPrimeWorkLimiter.CurrentLimit;
		if (effectiveWorkerCount > gpuLimit)
		{
			effectiveWorkerCount = gpuLimit;
		}

		if (workerCount == 1)
		{
			DeterministicRandomCpu.Initialize();
			var session = new MersenneNumberDivisorByDivisorPrimeScanSessionWithHybridOrder(prototypeCopy, markComposite, clearComposite, printResult);

			GpuPrimeWorkLimiter.Acquire();
			foreach (ulong prime in filteredPrimes)
			{
				session.ProcessPrime(prime);
			}

			GpuPrimeWorkLimiter.Release();
			session.Dispose();

			return;
		}

		TaskScheduler scheduler = UnboundedTaskScheduler.Instance;
		int nextIndex = 0;
		Task[] tasks = new Task[effectiveWorkerCount];
		var startGate = new ManualResetEventSlim(initialState: false);

		for (int taskIndex = 0; taskIndex < effectiveWorkerCount; taskIndex++)
		{
			int workerIndex = taskIndex;
			tasks[taskIndex] = Task.Factory.StartNew(
				() =>
				{
					DeterministicRandomCpu.Initialize();
					var session = new MersenneNumberDivisorByDivisorPrimeScanSessionWithHybridOrder(prototypeCopy, markComposite, clearComposite, printResult);
					startGate.Wait();
					Console.WriteLine($"Task started for worker {workerIndex.ToString(CultureInfo.InvariantCulture)}");

					GpuPrimeWorkLimiter.Acquire();
					while (true)
					{
						int start = Interlocked.Add(ref nextIndex, partitionSize) - partitionSize;
						if (start >= totalCount)
						{
							break;
						}

						int end = start + partitionSize;
						if (end > totalCount)
						{
							end = totalCount;
						}

						for (int index = start; index < end; index++)
						{
							session.ProcessPrime(filteredPrimes[index]);
						}
					}

					GpuPrimeWorkLimiter.Release();
					session.Dispose();
					Console.WriteLine($"Task finished for worker {workerIndex.ToString(CultureInfo.InvariantCulture)}");
				},
				CancellationToken.None,
				TaskCreationOptions.DenyChildAttach,
				scheduler);
		}

		startGate.Set();
		Task.WaitAll(tasks);
		startGate.Dispose();
	}

	private static void RunGpuOrderFiltered(
		List<ulong> filteredPrimes,
		in MersenneNumberDivisorByDivisorCpuTesterWithGpuOrder prototype,
		int workerCount,
		int partitionSize,
		Action markComposite,
		Action clearComposite,
		Action<ulong, bool, bool, bool, BigInteger> printResult)
	{
		MersenneNumberDivisorByDivisorCpuTesterWithGpuOrder prototypeCopy = prototype;
		int totalCount = filteredPrimes.Count;
		if (partitionSize > totalCount)
		{
			partitionSize = totalCount;
		}

		int effectiveWorkerCount = workerCount;
		int gpuLimit = GpuPrimeWorkLimiter.CurrentLimit;
		if (effectiveWorkerCount > gpuLimit)
		{
			effectiveWorkerCount = gpuLimit;
		}

		if (workerCount == 1)
		{
			DeterministicRandomCpu.Initialize();
			var session = new MersenneNumberDivisorByDivisorPrimeScanSessionWithGpuOrder(prototypeCopy, markComposite, clearComposite, printResult);

			GpuPrimeWorkLimiter.Acquire();
			foreach (ulong prime in filteredPrimes)
			{
				session.ProcessPrime(prime);
			}

			GpuPrimeWorkLimiter.Release();
			session.Dispose();

			return;
		}

		TaskScheduler scheduler = UnboundedTaskScheduler.Instance;
		int nextIndex = 0;
		Task[] tasks = new Task[effectiveWorkerCount];
		var startGate = new ManualResetEventSlim(initialState: false);

		for (int taskIndex = 0; taskIndex < effectiveWorkerCount; taskIndex++)
		{
			int workerIndex = taskIndex;
			tasks[taskIndex] = Task.Factory.StartNew(
				() =>
				{
					DeterministicRandomCpu.Initialize();
					var session = new MersenneNumberDivisorByDivisorPrimeScanSessionWithGpuOrder(prototypeCopy, markComposite, clearComposite, printResult);
					startGate.Wait();
					Console.WriteLine($"Task started for worker {workerIndex.ToString(CultureInfo.InvariantCulture)}");

					GpuPrimeWorkLimiter.Acquire();
					while (true)
					{
						int start = Interlocked.Add(ref nextIndex, partitionSize) - partitionSize;
						if (start >= totalCount)
						{
							break;
						}

						int end = start + partitionSize;
						if (end > totalCount)
						{
							end = totalCount;
						}

						for (int index = start; index < end; index++)
						{
							session.ProcessPrime(filteredPrimes[index]);
						}
					}

					GpuPrimeWorkLimiter.Release();
					session.Dispose();
					Console.WriteLine($"Task finished for worker {workerIndex.ToString(CultureInfo.InvariantCulture)}");
				},
				CancellationToken.None,
				TaskCreationOptions.DenyChildAttach,
				scheduler);
		}

		startGate.Set();
		Task.WaitAll(tasks);
		startGate.Dispose();
	}

	private static void RunGpuTesterFiltered(
		List<ulong> filteredPrimes,
		MersenneNumberDivisorByDivisorGpuTester prototype,
		int workerCount,
		int partitionSize,
		Action markComposite,
		Action clearComposite,
		Action<ulong, bool, bool, bool, BigInteger> printResult)
	{
		int totalCount = filteredPrimes.Count;
		if (partitionSize > totalCount)
		{
			partitionSize = totalCount;
		}

		int effectiveWorkerCount = workerCount;
		int gpuLimit = GpuPrimeWorkLimiter.CurrentLimit;
		if (effectiveWorkerCount > gpuLimit)
		{
			effectiveWorkerCount = gpuLimit;
		}

		if (workerCount == 1)
		{
			DeterministicRandomCpu.Initialize();
			var session = new MersenneNumberDivisorByDivisorGpuPrimeScanSession(prototype, markComposite, clearComposite, printResult);

			GpuPrimeWorkLimiter.Acquire();
			foreach (ulong prime in filteredPrimes)
			{
				session.ProcessPrime(prime);
			}

			GpuPrimeWorkLimiter.Release();
			session.Dispose();

			return;
		}

		TaskScheduler scheduler = UnboundedTaskScheduler.Instance;
		int nextIndex = 0;
		Task[] tasks = new Task[effectiveWorkerCount];
		var startGate = new ManualResetEventSlim(initialState: false);

		for (int taskIndex = 0; taskIndex < effectiveWorkerCount; taskIndex++)
		{
			int workerIndex = taskIndex;
			tasks[taskIndex] = Task.Factory.StartNew(
				() =>
				{
					DeterministicRandomCpu.Initialize();
					var session = new MersenneNumberDivisorByDivisorGpuPrimeScanSession(prototype, markComposite, clearComposite, printResult);
					startGate.Wait();
					Console.WriteLine($"Task started for worker {workerIndex.ToString(CultureInfo.InvariantCulture)}");

					GpuPrimeWorkLimiter.Acquire();
					while (true)
					{
						int start = Interlocked.Add(ref nextIndex, partitionSize) - partitionSize;
						if (start >= totalCount)
						{
							break;
						}

						int end = start + partitionSize;
						if (end > totalCount)
						{
							end = totalCount;
						}

						for (int index = start; index < end; index++)
						{
							session.ProcessPrime(filteredPrimes[index]);
						}
					}

					GpuPrimeWorkLimiter.Release();
					session.Dispose();
					Console.WriteLine($"Task finished for worker {workerIndex.ToString(CultureInfo.InvariantCulture)}");
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
