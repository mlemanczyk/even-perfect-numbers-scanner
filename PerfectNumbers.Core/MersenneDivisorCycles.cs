using System.Buffers;
using System.Buffers.Binary;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Runtime.InteropServices;
using PerfectNumbers.Core.Gpu;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace PerfectNumbers.Core;

public class MersenneDivisorCycles
{
	private List<(ulong divisor, ulong cycleLength)> _table = [];
	// Lightweight read-mostly cache for small divisors (<= 4,000,000). 0 => unknown
	private ulong[]? _smallCycles;
	public static MersenneDivisorCycles Shared { get; } = new MersenneDivisorCycles();

	public MersenneDivisorCycles()
	{
	}

	public void LoadFrom(string path)
	{
		EnsureSmallBuffer();
		List<(ulong divisor, ulong cycle)> cycles = [];
		using Stream outputStream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, BufferSize10M, useAsync: false);
		foreach (var (d, c) in EnumerateStream(outputStream))
		{
			cycles.Add((d, c));
			if (d <= PerfectNumberConstants.MaxQForDivisorCycles)
			{
				_smallCycles![(int)d] = c == 0UL ? 1UL : c;
			}
		}

		cycles.Sort((a, b) => a.divisor < b.divisor ? -1 : (a.divisor > b.divisor ? 1 : 0));
		_table = cycles;
	}

	public const int SmallDivisorsMax = PerfectNumberConstants.MaxQForDivisorCycles;

	// Provides a snapshot array of small cycles [0..PerfectNumberConstants.MaxQForDivisorCycles]. Index is the divisor.
	// The returned array is a copy safe for use across threads and device uploads.
	public ulong[] ExportSmallCyclesSnapshot()
	{
		EnsureSmallBuffer();
		ulong[] snapshot = new ulong[PerfectNumberConstants.MaxQForDivisorCycles + 1];
		Array.Copy(_smallCycles!, snapshot, snapshot.Length);
		return snapshot;
	}

	public static bool CycleEqualsExponent(ulong divisor, in MontgomeryDivisorData divisorData, ulong exponent)
	{
		if (CycleEqualsExponentForMersenneCandidate(divisor, divisorData, exponent))
		{
			return true;
		}

		if (exponent <= 1UL || divisor <= 1UL || (divisor & 1UL) == 0UL)
		{
			return false;
		}

		if (divisor <= PerfectNumberConstants.MaxQForDivisorCycles)
		{
			var small = Shared._smallCycles;
			if (small is not null)
			{
				ulong cached = small[(int)divisor];
				if (cached != 0UL)
				{
					return cached == exponent;
				}
			}
		}

		if (TryCalculateCycleLengthHeuristic(divisor, divisorData, out ulong heuristicCycle) && heuristicCycle != 0UL)
		{
			return heuristicCycle == exponent;
		}

		if (TryCalculateCycleLengthForExponentCpu(divisor, exponent, divisorData, out ulong cycleLength, out bool _) && cycleLength != 0UL)
		{
			return cycleLength == exponent;
		}

		return exponent.Pow2MontgomeryModWindowedCpu(divisorData, keepMontgomery: false) == 1UL;
	}

	public static bool CycleEqualsExponentForMersenneCandidate(ulong divisor, in MontgomeryDivisorData divisorData, ulong exponent)
	{
		// EvenPerfectBitScanner only calls this helper with divisors generated from 2 * k * p + 1 and primes greater than one,
		// so the validation guard would never trigger in production. Keep it commented out to leave the hot path branch-free.
		// if (!IsValidMersenneDivisorCandidate(divisor, exponent))
		// {
		//     return false;
		// }

		if (divisor <= PerfectNumberConstants.MaxQForDivisorCycles)
		{
			MersenneDivisorCycles shared = Shared;
			ulong[]? small = shared._smallCycles;
			if (small is not null)
			{
				ulong cached = small[(int)divisor];
				if (cached != 0UL)
				{
					return cached == exponent;
				}
			}
		}

		return exponent.Pow2MontgomeryModWindowedCpu(divisorData, keepMontgomery: false) == 1UL;
	}

	private static bool IsValidMersenneDivisorCandidate(ulong divisor, ulong exponent)
	{
		if (exponent <= 1UL || divisor <= 1UL || (divisor & 1UL) == 0UL)
		{
			return false;
		}

		UInt128 step = (UInt128)exponent << 1;
		if (step == UInt128.Zero)
		{
			return false;
		}

		UInt128 adjusted = (UInt128)divisor - UInt128.One;
		return adjusted % step == UInt128.Zero;
	}

	public static IEnumerable<(ulong divisor, ulong cycleLength)> EnumerateStream(Stream compressor)
	{
		// Binary pairs: (ulong divisor, ulong cycle)
		using var reader = new BinaryReader(compressor, Encoding.UTF8, leaveOpen: true);
		Stream stream = reader.BaseStream;
		byte[] buffer = new byte[16];
		while (true)
		{
			int read = 0;
			Span<byte> span = buffer;
			while (read < span.Length)
			{
				int chunk = stream.Read(span[read..]);
				if (chunk == 0)
				{
					if (read == 0)
					{
						yield break;
					}

					throw new EndOfStreamException("Incomplete divisor-cycle entry.");
				}

				read += chunk;
			}

			ulong divisor = BinaryPrimitives.ReadUInt64LittleEndian(span[..8]);
			ulong cycle = BinaryPrimitives.ReadUInt64LittleEndian(span[8..]);
			yield return (divisor, cycle);
		}
	}

	public ulong GetCycle(ulong divisor, in MontgomeryDivisorData divisorData)
	{
		// Fast-path: in-memory array for small divisors
		if (divisor <= PerfectNumberConstants.MaxQForDivisorCycles)
		{
			var arr = _smallCycles;
			if (arr is not null)
			{
				ulong cached = arr[(int)divisor];
				if (cached != 0UL)
				{
					return cached;
				}
			}
		}

		// Binary search for divisor in sorted _table
		int mid, right, left = 0;
		ulong d;
		List<(ulong divisor, ulong cycleLength)> cycles = _table;

		right = cycles.Count - 1;
		while (left <= right)
		{
			mid = left + ((right - left) >> 1);
			d = cycles[mid].divisor;

			if (d == divisor)
			{
				return cycles[mid].cycleLength;
			}

			if (d < divisor)
			{
				left = mid + 1;
			}
			else
			{
				right = mid - 1;
			}
		}

		// TODO: Route this miss to an ephemeral single-cycle computation on the device selected by the current
		// configuration (and skip persisting the result) when the divisor falls outside the in-memory snapshot
		// so we respect the large-p memory limits and avoid extra cache locks.
		return CalculateCycleLength(divisor, divisorData);
	}

	public static UInt128 GetCycle(UInt128 divisor)
	{
		// For divisor = 2^k, cycle is 1
		if ((divisor & (divisor - UInt128.One)) == UInt128.Zero)
		{
			return UInt128.One;
		}

		if (divisor <= ulong.MaxValue)
		{
			ulong divisor64 = (ulong)divisor;
			MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(divisor64);
			ulong cycle = CalculateCycleLength(divisor64, divisorData);
			return (UInt128)cycle;
		}

		UInt128 wideOrder = PrimeOrderCalculator.Calculate(
			divisor,
			previousOrder: null,
			PrimeOrderCalculator.PrimeOrderSearchConfig.HeuristicDefault,
			PrimeOrderCalculator.PrimeOrderHeuristicDevice.Cpu);
		if (wideOrder != UInt128.Zero)
		{
			return wideOrder;
		}

		// Otherwise, find order of 2 mod divisor
		// TODO: Port this UInt128 path to the unrolled-hex cycle calculator so wide
		// divisors stop relying on the slow shift-and-subtract loop measured in the
		// GPU benchmarks.
		UInt128 order = UInt128.One;
		UInt128 pow = UInt128Numbers.Two;
		while (pow != UInt128.One)
		{
			pow <<= 1;
			if (pow >= divisor)
			{
				pow -= divisor;
			}

			order++;
		}

		return order;
	}

	public static bool TryCalculateCycleLengthForExponentCpu(
		ulong divisor,
		ulong exponent,
		in MontgomeryDivisorData divisorData,
		out ulong cycleLength,
		out bool primeOrderFailed)
	{
		cycleLength = 0UL;
		primeOrderFailed = false;

		// EvenPerfectBitScanner never feeds powers of two into this helper, so leave the branch commented out to avoid redundant work.
		// if ((divisor & (divisor - 1UL)) == 0UL)
		// {
		//     cycleLength = 1UL;
		//     return true;
		// }

		// The scanner only calls this path with prime exponents greater than one and odd divisors above three, keeping the small-parameter guard unreachable.
		// if (divisor <= 3UL || exponent <= 1UL)
		// {
		//     cycleLength = CalculateCycleLength(divisor, divisorData);
		//     return true;
		// }

		ulong computedOrder = PrimeOrderCalculator.Calculate(
			divisor,
			previousOrder: null,
			divisorData,
			PrimeOrderCalculator.PrimeOrderSearchConfig.HeuristicDefault,
			PrimeOrderCalculator.PrimeOrderHeuristicDevice.Cpu);
		if (computedOrder != 0UL)
		{
			cycleLength = computedOrder;
			return true;
		}

		primeOrderFailed = true;

		ulong phi = divisor - 1UL;
		// A zero totient would imply divisor == 0, which never occurs on the tested path. Keep the guard documented without
		// branching in the hot loop.
		// if (phi == 0UL)
		// {
		//     return false;
		// }

		int twoCount = BitOperations.TrailingZeroCount(phi);
		ulong reducedPhi = phi >> twoCount;
		if (reducedPhi == 0UL || (reducedPhi % exponent) != 0UL)
		{
			return false;
		}

		ulong k = reducedPhi / exponent;
		// k == 0 would indicate reducedPhi < exponent, which the EvenPerfectBitScanner call flow prevents. Document the
		// assumption instead of branching.
		// if (k == 0UL)
		// {
		//     return false;
		// }

		Dictionary<ulong, int> factorCounts = ThreadStaticPools.RentFactorCountDictionary();
		bool success = false;

		if (twoCount > 0)
		{
			factorCounts[2UL] = twoCount;
		}

		bool exponentAccumulated = AccumulateFactors(exponent, factorCounts);
		bool factorsAccumulated = exponentAccumulated && AccumulateFactors(k, factorCounts);

		if (factorsAccumulated)
		{
			cycleLength = ReduceOrder(divisorData, phi, factorCounts);
			success = true;
			primeOrderFailed = false;
		}

		factorCounts.Clear();
		ThreadStaticPools.ReturnFactorCountDictionary(factorCounts);
		return success;
	}

	private static bool AccumulateFactors(
		ulong value,
		Dictionary<ulong, int> counts)
	{
		if (value <= 1UL)
		{
			return true;
		}

		Dictionary<ulong, int> scratch = ThreadStaticPools.RentFactorScratchDictionary();
		bool success = TryFactorIntoCountsInternal(value, scratch);
		if (!success)
		{
			scratch.Clear();
			ThreadStaticPools.ReturnFactorScratchDictionary(scratch);
			return false;
		}

		for (Dictionary<ulong, int>.Enumerator enumerator = scratch.GetEnumerator(); enumerator.MoveNext();)
		{
			KeyValuePair<ulong, int> pair = enumerator.Current;
			AddFactor(counts, pair.Key, pair.Value);
		}

		scratch.Clear();
		ThreadStaticPools.ReturnFactorScratchDictionary(scratch);
		return true;
	}

	// private static ulong _tryFactorIntoCountsInternalHits;
	// private static ulong _tryCalculateCycleLengthHeuristicHits;

	private static bool TryFactorIntoCountsInternal(ulong value, Dictionary<ulong, int> counts)
	{
		// The EvenPerfectBitScanner only requests factors for values >= 2 here. Leave the guard documented without executing it.
		// if (value <= 1UL)
		// {
		//     return true;
		// }

		ulong remaining = value;
		uint[] smallPrimes = PrimesGenerator.SmallPrimes;
		ulong[] smallPrimesSquared = PrimesGenerator.SmallPrimesPow2;
		int smallLength = smallPrimes.Length;

		for (int i = 0; i < smallLength; i++)
		{
			ulong prime = smallPrimes[i];
			ulong primeSquared = smallPrimesSquared[i];
			if (primeSquared > remaining)
			{
				break;
			}

			// TODO: Reuse the remainder-tracking update from the cycle heuristic so repeated divisions disappear from this loop.
			while ((remaining % prime) == 0UL)
			{
				AddFactor(counts, prime);
				remaining /= prime;
			}
		}

		// TODO: Benchmark PrimeTester.IsPrimeInternal against Open.Numeric.Primes on the EvenPerfectBitScanner workloads to confirm the faster option for wide composites.
		// if (HeuristicPrimeTester.Exclusive.IsPrime(remaining, CancellationToken.None))

		// Atomic.Add(ref _tryFactorIntoCountsInternalHits, 1UL);
		// Console.WriteLine($"MersenneDivisorCycles.TryFactorIntoCountsInternal hits {Volatile.Read(ref _tryFactorIntoCountsInternalHits)}");

		// HeuristicPrimeTester primeTester = _primeTester ??= new();
		if (PrimeTester.IsPrimeGpu(remaining))
		// if (PrimeTester.IsPrimeCpu(remaining, CancellationToken.None))
		{
			AddFactor(counts, remaining);
			return true;
		}

		// TODO: Evaluate folding the remainder-tracking mechanism into PollardRho64 so the walk leverages previously computed residues.
		ulong factor = PollardRho64(remaining);
		if (factor == 0UL || factor == remaining)
		{
			// Pollard-Rho can still stall on rare stubborn composites during test and benchmark runs, so keep the failure guard in place
			// and let the caller fall back to the cached cycle calculator.
			return false;
		}

		// TODO: Investigate computing this quotient via remainder tracking or reciprocal multiplication to avoid the division.
		ulong quotient = remaining / factor;
		// Pollard-Rho never returns factors <= 1 on the EvenPerfectBitScanner path; keep the old guard documented
		// for completeness without branching in the hot path.
		// if (factor > 1UL && !TryFactorIntoCountsInternal(factor, counts))
		// {
		//     return false;
		// }

		if (!TryFactorIntoCountsInternal(factor, counts))
		{
			return false;
		}

		// Guard the recursive calls to terminate once Pollard-Rho reduces either branch to 1.
		if (quotient > 1UL && !TryFactorIntoCountsInternal(quotient, counts))
		{
			return false;
		}

		return true;
	}

	private static void AddFactor(Dictionary<ulong, int> counts, ulong factor)
	{
		ref int entry = ref CollectionsMarshal.GetValueRefOrAddDefault(counts, factor, out bool exists);
		entry = exists ? entry + 1 : 1;
	}

	private static void AddFactor(Dictionary<ulong, int> counts, ulong factor, int multiplicity)
	{
		ref int entry = ref CollectionsMarshal.GetValueRefOrAddDefault(counts, factor, out bool exists);
		entry = exists ? entry + multiplicity : multiplicity;
	}

	private static ulong ReduceOrder(in MontgomeryDivisorData divisorData, ulong initialOrder, Dictionary<ulong, int> factorCounts)
	{
		if (factorCounts.Count == 0)
		{
			return initialOrder;
		}

		int count = factorCounts.Count;
		ulong[] primes = ThreadStaticPools.UlongPool.Rent(count);

		int index = 0;
		foreach (KeyValuePair<ulong, int> entry in factorCounts)
		{
			primes[index] = entry.Key;
			index++;
		}

		Array.Sort(primes, 0, count);

		const int StackMultiplicityCapacity = 16;
		const int HeapMultiplicityCapacity = 256;

		// Scanner workloads keep almost all multiplicities within the stack-backed span, so the stack path handles the hot loop.
		Span<ulong> stackCandidates = stackalloc ulong[StackMultiplicityCapacity];
		Span<bool> stackEvaluations = stackalloc bool[StackMultiplicityCapacity];

		ulong[]? heapCandidateArray = null;
		bool[]? heapEvaluationArray = null;

		ExponentRemainderStepperCpu stepper = ThreadStaticPools.RentExponentStepperCpu(divisorData);

		ulong order = initialOrder;
		for (int i = 0; i < count; i++)
		{
			ulong prime = primes[i];
			int multiplicity = factorCounts[prime];
			// Multiplicities collected by TryFactorIntoCountsInternal are strictly positive along the scanner path.
			// Keeping the guard commented documents the assumption so it does not resurface in future reviews.
			// if (multiplicity <= 0)
			// {
			//     continue;
			// }

			if (multiplicity <= StackMultiplicityCapacity)
			{
				ProcessReduceOrderPrime(stackCandidates.Slice(0, multiplicity), stackEvaluations.Slice(0, multiplicity), ref order, prime, multiplicity, ref stepper);
				continue;
			}

			if (multiplicity > HeapMultiplicityCapacity)
			{
				throw new InvalidOperationException($"Factor multiplicity {multiplicity} exceeds the supported limit of {HeapMultiplicityCapacity}.");
			}

			// Larger multiplicities appear during stress tests; reuse a shared heap buffer in those cases.
			heapCandidateArray ??= ThreadStaticPools.UlongPool.Rent(HeapMultiplicityCapacity);
			heapEvaluationArray ??= ThreadStaticPools.BoolPool.Rent(HeapMultiplicityCapacity);
			ProcessReduceOrderPrime(heapCandidateArray.AsSpan(0, multiplicity), heapEvaluationArray.AsSpan(0, multiplicity), ref order, prime, multiplicity, ref stepper);
		}

		ThreadStaticPools.ReturnExponentStepperCpu(stepper);
		ThreadStaticPools.UlongPool.Return(primes);

		if (heapCandidateArray is not null)
		{
			ThreadStaticPools.UlongPool.Return(heapCandidateArray);
			ThreadStaticPools.BoolPool.Return(heapEvaluationArray!);
		}

		return order;
	}

	private static void ProcessReduceOrderPrime(Span<ulong> candidateBuffer, Span<bool> evaluationBuffer, ref ulong order, ulong prime, int multiplicity, ref ExponentRemainderStepperCpu stepper)
	{
		ulong working = order;
		int writeIndex = multiplicity;
		while (writeIndex > 0)
		{
			if (working < prime)
			{
				break;
			}

			ulong reduced = working / prime;
			if (reduced == 0UL)
			{
				break;
			}

			writeIndex--;
			candidateBuffer[writeIndex] = reduced;
			working = reduced;
		}

		int actual = multiplicity - writeIndex;
		if (actual == 0)
		{
			return;
		}

		Span<ulong> candidates = candidateBuffer.Slice(writeIndex, actual);
		Span<bool> evaluations = evaluationBuffer.Slice(0, actual);

		stepper.Reset();

		evaluations[0] = stepper.InitializeCpuIsUnity(candidates[0]);
		for (int j = 1; j < actual; j++)
		{
			evaluations[j] = stepper.ComputeNextIsUnity(candidates[j]);
		}

		for (int j = actual - 1; j >= 0; j--)
		{
			if (!evaluations[j])
			{
				break;
			}

			order = candidates[j];
		}
	}

	private static ulong PollardRho64(ulong n)
	{
		// The EvenPerfectBitScanner path removes all factors of two before invoking Pollard-Rho, so this branch never executes
		// outside synthetic tests and benchmarks. Keep the guard documented to prevent future regressions without paying the hot-
		// path cost.
		// if ((n & 1UL) == 0UL)
		// {
		//     return 2UL;
		// }

		ulong modulus = n;
		ulong nMinus1 = modulus - 1UL;
		ulong nMinus2 = modulus - 2UL;

		while (true)
		{
			ulong c = (NextRandomUInt64() % nMinus1) + 1UL;
			ulong x = (NextRandomUInt64() % nMinus2) + 2UL;
			ulong y = x;
			ulong d = 1UL;

			while (d == 1UL)
			{
				x = AdvancePolynomial(x, c, modulus);
				y = AdvancePolynomial(y, c, modulus);
				y = AdvancePolynomial(y, c, modulus);

				ulong diff = x > y ? x - y : y - x;
				d = BinaryGcd(diff, modulus);
			}

			if (d != modulus)
			{
				return d;
			}
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static ulong AdvancePolynomial(ulong x, ulong c, ulong modulus)
	{
		UInt128 value = (UInt128)x * x + c;
		return (ulong)(value % modulus);
	}

	private static ulong BinaryGcd(ulong a, ulong b)
	{
		if (a == 0UL)
		{
			return b;
		}

		if (b == 0UL)
		{
			return a;
		}

		int shift = BitOperations.TrailingZeroCount(a | b);
		a >>= BitOperations.TrailingZeroCount(a);

		while (true)
		{
			b >>= BitOperations.TrailingZeroCount(b);
			if (a > b)
			{
				(a, b) = (b, a);
			}

			b -= a;
			if (b == 0UL)
			{
				return a << shift;
			}
		}
	}

	private static ulong NextRandomUInt64()
	{
		Span<byte> buffer = stackalloc byte[8];
		RandomNumberGenerator.Fill(buffer);
		return BinaryPrimitives.ReadUInt64LittleEndian(buffer);
	}

	private sealed class FactorPairValueComparer : IComparer<KeyValuePair<ulong, int>>
	{
		public static readonly FactorPairValueComparer Instance = new FactorPairValueComparer();

		private FactorPairValueComparer()
		{
		}

		public int Compare(KeyValuePair<ulong, int> x, KeyValuePair<ulong, int> y)
		{
			if (x.Key < y.Key)
			{
				return -1;
			}

			if (x.Key > y.Key)
			{
				return 1;
			}

			return 0;
		}
	}

	public static (long nextPosition, long completeCount) FindLast(string path)
	{
		using Stream outputStream = new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.Read, BufferSize10M, useAsync: false);
		long count = 0;
		foreach (var _ in EnumerateStream(outputStream))
		{
			count++;
		}
		return (outputStream.Position, count);
	}

	private const int BufferSize10M = 10 * 1024 * 1024;

	public static void Generate(string path, ulong maxDivisor, int threads = 16)
	{
		// Binary-only generation of (divisor, cycle) pairs. CSV scaffolding removed.
		ulong start = 2;
		ulong blockSize = maxDivisor / (ulong)threads + 1UL;
		Task[] tasks = new Task[threads];
		using Stream outputStream = new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.Read, BufferSize10M, useAsync: false);
		// Start fresh to avoid mixing formats
		outputStream.SetLength(0L);
		outputStream.Position = 0L;
		using var writer = new BinaryWriter(outputStream, Encoding.UTF8, leaveOpen: false);
		for (var taskIndex = 0; taskIndex < threads; taskIndex++)
		{
			ulong threadStart = start + (ulong)taskIndex * blockSize;
			if (threadStart > maxDivisor)
			{
				tasks[taskIndex] = Task.CompletedTask;
				continue;
			}

			ulong threadEnd = threadStart + blockSize - 1UL;
			if (threadEnd > maxDivisor)
			{
				threadEnd = maxDivisor;
			}

			ulong localStart = threadStart;
			ulong localEnd = threadEnd;
			tasks[taskIndex] = Task.Run(() =>
			{
				ulong rangeLength = localEnd - localStart + 1UL;
				(ulong divisor, ulong cycle)[] localCycles = ThreadStaticPools.RentUlongUlongArray(checked((int)rangeLength));
				// (ulong divisor, ulong cycle)[] localCycles = ArrayPool<(ulong, ulong)>.Shared.Rent(checked((int)rangeLength));
				int localCycleIndex = 0;

				for (ulong divisor = localStart; divisor <= localEnd; divisor++)
				{
					if ((divisor & 1UL) == 0UL)
					{
						continue;
					}

					// TODO: Replace this modulo-based filter with the cached Mod3/Mod5/Mod7/Mod11
					// helpers so the CPU generator stops paying for `%` in the small-prime sieve
					// and lines up with the divisor-cycle pipeline.
					if ((divisor % 3UL) == 0UL || (divisor % 5UL) == 0UL || (divisor % 7UL) == 0UL || (divisor % 11UL) == 0UL)
					{
						continue;
					}

					// TODO: Migrate this generation path to the shared
					// unrolled-hex calculator so the CPU generator matches
					// the GPU benchmark leader for large divisors.
					MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(divisor);
					ulong cycle = CalculateCycleLength(divisor, divisorData);
					localCycles[localCycleIndex++] = (divisor, cycle);
				}

				if (localCycleIndex == 0)
				{
					ThreadStaticPools.Return(localCycles);
					return;
				}

				lock (writer)
				{
					writer.BaseStream.Position = writer.BaseStream.Length;
					for (int i = 0; i < localCycleIndex; i++)
					{
						(ulong divisor, ulong cycle) = localCycles[i];
						writer.Write(divisor);
						writer.Write(cycle);
					}

					writer.Flush();
				}

				ThreadStaticPools.Return(localCycles);
			});
		}

		Task.WaitAll(tasks);
	}

	public static void GenerateGpu(string path, ulong maxDivisor, int batchSize = 1_000_000, long skipCount = 0L, long nextPosition = 0L)
	{
		// Prepare output file with header

		ulong start = 3UL;
		var locker = new object();

		var pool = ThreadStaticPools.UlongPool;
		ulong batchSizeUL = (ulong)batchSize, d, end;
		int count = (int)Math.Min(batchSizeUL, maxDivisor), i, idx;
		ulong[] divisors, outCycles, validDivisors;

		divisors = pool.Rent(count);
		outCycles = pool.Rent(count);
		var lease = GpuKernelPool.GetKernel();
		var accelerator = lease.Accelerator;
		var stream = lease.Stream;
		var loaded = accelerator.LoadAutoGroupedStreamKernel<
				Index1D,
				ArrayView1D<ulong, Stride1D.Dense>,
				ArrayView1D<ulong, Stride1D.Dense>>(DivisorCycleKernels.GpuDivisorCycleKernel);
		var kernel = KernelUtil.GetKernel(loaded);
		var launcher = kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>>();


		using Stream outputStream = new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.Read, BufferSize10M, useAsync: true);
		if (nextPosition > 0L)
		{
			outputStream.Position = nextPosition;
		}
		else
		{
			outputStream.SetLength(0L);
			outputStream.Position = 0L;
		}

		using var writer = new BinaryWriter(outputStream, Encoding.UTF8, leaveOpen: false);
		while (skipCount > 0L && start <= maxDivisor)
		{
			// TODO: Swap these modulo checks for the cached Mod3/Mod5/Mod7/Mod11 helpers so
			// range initialization avoids repeated `%` calls before the unrolled-hex pipeline
			// takes over.
			if ((start & 1UL) == 0UL || (start % 3UL) == 0UL || (start % 5UL) == 0UL || (start % 7UL) == 0UL || (start % 11UL) == 0UL)
			{
				start++;
				continue;
			}

			start++;
			skipCount--;
		}

		while (start <= maxDivisor)
		{
			end = Math.Min(start + batchSizeUL - 1UL, maxDivisor);
			count = checked((int)(end - start + 1UL));

			idx = 0;
			for (d = start; d <= end; d++)
			{
				if ((d & 1UL) == 0UL)
					continue;

				divisors[idx++] = d;
			}

			validDivisors = divisors[..idx];

			// Use GpuKernelPool to get a kernel and context
			// GpuKernelPool.Run((accelerator, stream) =>
			var bufferDiv = accelerator.Allocate1D(validDivisors);
			var bufferCycle = accelerator.Allocate1D<ulong>(idx);

			launcher(
					stream,
					idx,
					bufferDiv.View,
					bufferCycle.View);

			bufferCycle.View.CopyToCPU(stream, ref outCycles[0], idx);
			stream.Synchronize();

			bufferCycle.Dispose();
			bufferDiv.Dispose();

			// Collect results in order
			for (i = 0; i < idx; i++)
			{
				d = divisors[i];

				// TODO: Replace this modulo sieve with the cached Mod helpers once the divisor-cycle cache
				// is mandatory so cycle enumeration can drop the slower `%` operations in this hot loop.
				if ((d % 3UL) == 0UL || (d % 5UL) == 0UL || (d % 7UL) == 0UL || (d % 11UL) == 0UL)
					continue;

				writer.Write(d);
				writer.Write(outCycles[i]);
			}

			writer.Flush();
			Console.WriteLine($"...processed divisor = {d}");
			start = end + 1UL;
		}

		pool.Return(divisors, clearArray: false);
		pool.Return(outCycles, clearArray: false);

		lease.Dispose();
	}


	public static ulong CalculateCycleLengthGpu(ulong divisor) => DivisorCycleKernels.CalculateCycleLengthGpu(divisor);

	public static ulong CalculateCycleLength(ulong divisor, in MontgomeryDivisorData divisorData, bool skipPrimeOrderHeuristic = false)
	{
		if (TryCalculateCycleLengthHeuristic(divisor, divisorData, out ulong cycleLength, skipPrimeOrderHeuristic))
			return cycleLength;

		return CalculateCycleLengthFallback(divisor);
	}

	// [ThreadStatic]
	// private static HeuristicPrimeTester? _primeTester;

	internal static bool TryCalculateCycleLengthHeuristic(ulong divisor, in MontgomeryDivisorData divisorData, out ulong cycleLength, bool skipPrimeOrderHeuristic = false)
	{
		if ((divisor & (divisor - 1UL)) == 0UL)
		{
			cycleLength = 1UL;
			return true;
		}

		if (divisor <= 3UL)
		{
			cycleLength = CalculateCycleLengthFallback(divisor);
			return true;
		}

		// if (!skipPrimeOrderHeuristic)
		// {
		// 	Atomic.Add(ref _tryCalculateCycleLengthHeuristicHits, 1UL);
		// 	Console.WriteLine($"MersenneDivisorCycles.CalculateCycleLengthFallback hits {Volatile.Read(ref _tryCalculateCycleLengthHeuristicHits)}");			
		// }

		// HeuristicPrimeTester primeTester = _primeTester ??= new();
		// if (!skipPrimeOrderHeuristic && HeuristicPrimeTester.Exclusive.IsPrimeCpu(divisor, CancellationToken.None))
		// if (!skipPrimeOrderHeuristic && primeTester.IsPrimeGpu(divisor))
		if (!skipPrimeOrderHeuristic && PrimeTester.IsPrimeGpu(divisor))
		{
			ulong computedOrder = PrimeOrderCalculator.Calculate(
					divisor,
					previousOrder: null,
					divisorData,
					PrimeOrderCalculator.PrimeOrderSearchConfig.HeuristicDefault,
					PrimeOrderCalculator.PrimeOrderHeuristicDevice.Cpu);
			if (computedOrder != 0UL)
			{
				cycleLength = computedOrder;
				return true;
			}
		}

		cycleLength = 0UL;
		return false;
	}

	private static ulong CalculateCycleLengthFallback(ulong divisor)
	{
		// Otherwise, find order of 2 mod divisor using the shared unrolled-hex stepping sequence.
		return divisor.CalculateMersenneDivisorCycleLengthUnrolledHexCpu();
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private void EnsureSmallBuffer()
	{
		if (_smallCycles is null)
		{
			_smallCycles = new ulong[PerfectNumberConstants.MaxQForDivisorCycles + 1];
		}
	}
}
