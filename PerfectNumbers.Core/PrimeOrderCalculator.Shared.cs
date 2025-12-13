using System.Buffers;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

internal static partial class PrimeOrderCalculator
{
	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static void SortCandidatesWide(UInt128 prime, in UInt128? previousOrder, List<UInt128> candidates)
    {
        UInt128 previous = previousOrder ?? UInt128.Zero;
        bool hasPrevious = previousOrder.HasValue;
        int previousGroup = hasPrevious ? GetGroupWide(previous, prime) : 1;

        candidates.Sort((left, right) => CompareCandidatesWide(prime, left, right, previous, hasPrevious, previousGroup));
    }

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private static int CompareCandidatesWide(UInt128 prime, UInt128 left, UInt128 right, UInt128 previous, bool hasPrevious, int previousGroup)
	{
		int leftGroup = GetGroupWide(left, prime);
		int rightGroup = GetGroupWide(right, prime);
		if (leftGroup == 0)
		{
			return rightGroup == 0 ? 0 : 1;
		}

		if (rightGroup == 0)
		{
			return -1;
		}

		bool leftIsGe = !hasPrevious || left >= previous;
		bool rightIsGe = !hasPrevious || right >= previous;
		int leftPrimary = ComputePrimary(leftGroup, leftIsGe, previousGroup);
		int rightPrimary = ComputePrimary(rightGroup, rightIsGe, previousGroup);

		int compare = leftPrimary.CompareTo(rightPrimary);
		if (compare != 0)
		{
			return compare;
		}

		bool leftDescending = leftGroup == 3;
		bool rightDescending = rightGroup == 3;

		UInt128 leftSecondary;
		UInt128 leftTertiary;
		if (leftDescending)
		{
			leftSecondary = left;
			leftTertiary = left;
		}
		else
		{
			UInt128 leftReference = hasPrevious ? (left > previous ? left - previous : previous - left) : left;
			leftSecondary = leftReference;
			leftTertiary = left;
		}

		UInt128 rightSecondary;
		UInt128 rightTertiary;
		if (rightDescending)
		{
			rightSecondary = right;
			rightTertiary = right;
		}
		else
		{
			UInt128 rightReference = hasPrevious ? (right > previous ? right - previous : previous - right) : right;
			rightSecondary = rightReference;
			rightTertiary = right;
		}

		compare = CompareComponents(leftDescending, leftSecondary, rightDescending, rightSecondary);
		if (compare != 0)
		{
			return compare;
		}

		return CompareComponents(leftDescending, leftTertiary, rightDescending, rightTertiary);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private static int CompareComponents(bool descendingX, in UInt128 valueX, bool descendingY, in UInt128 valueY)
    {
        if (descendingX == descendingY)
        {
            if (valueX == valueY)
            {
                return 0;
            }

            if (descendingX)
            {
                return valueX > valueY ? -1 : 1;
            }

            return valueX > valueY ? 1 : -1;
        }

        if (valueX == valueY)
        {
            return 0;
        }

        return valueX > valueY ? 1 : -1;
    }

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static int GetGroupWide(in UInt128 value, in UInt128 prime)
    {
        UInt128 threshold = prime >> 3;
        if (value <= threshold)
        {
            return 1;
        }

        threshold = prime >> 2;
        if (value <= threshold)
        {
            return 2;
        }

        threshold = (UInt128)(((BigInteger)prime * 3) >> 3);
        if (value <= threshold)
        {
            return 3;
        }

        return 0;
    }

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static void BuildCandidatesWide(in UInt128 order, FactorEntry128[] factors, int count, List<UInt128> candidates, int limit)
    {
		// TODO: Is this condition ever met on EvenPerfectBitScanner execution paths for this method? Can we remove it?
        if (count == 0)
        {
            return;
        }

        Span<FactorEntry128> buffer = factors.AsSpan(0, count);
        buffer.Sort(static (a, b) => a.Value.CompareTo(b.Value));
        BuildCandidatesRecursiveWide(order, buffer, 0, UInt128.One, candidates, limit);
    }

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void BuildCandidatesRecursiveWide(
        in UInt128 order,
        in ReadOnlySpan<FactorEntry128> factors,
        int index,
        in UInt128 divisorProduct,
        List<UInt128> candidates,
        int limit)
    {
        if (candidates.Count >= limit)
        {
            return;
        }

        if (index >= factors.Length)
        {
            if (divisorProduct == UInt128.One || divisorProduct == order)
            {
                return;
            }

            UInt128 candidate = order / divisorProduct;
            if (candidate > UInt128.One && candidate < order)
            {
                candidates.Add(candidate);
            }

            return;
        }

        FactorEntry128 factor = factors[index];
        UInt128 primeFactor = factor.Value;
        UInt128 contribution = UInt128.One;
        for (int exponent = 0; exponent <= factor.Exponent; exponent++)
        {
            UInt128 nextDivisor = divisorProduct * contribution;
            if (nextDivisor > order)
            {
                break;
            }

            BuildCandidatesRecursiveWide(order, factors, index + 1, nextDivisor, candidates, limit);
            if (candidates.Count >= limit)
            {
                return;
            }

            if (exponent == factor.Exponent)
            {
                break;
            }

            if (contribution > order / primeFactor)
            {
                break;
            }

            contribution *= primeFactor;
        }
    }

    private static PartialFactorResult128 PartialFactorWide(in UInt128 value, in PrimeOrderCalculatorConfig config)
    {
        if (value <= UInt128.One)
        {
            return PartialFactorResult128.Empty;
        }

        PartialFactorResult128 result = PartialFactorResult128.Rent();
        Dictionary<UInt128, int> counts = result.FactorCounts;
        counts.Clear();
        uint limit = config.SmallFactorLimit == 0 ? uint.MaxValue : config.SmallFactorLimit;
        UInt128 remaining = PopulateSmallPrimeFactorsCpuWide(value, limit, counts);

        List<UInt128> pending = result.Pending;
        pending.Clear();
        if (remaining > UInt128.One)
        {
            pending.Add(remaining);
        }

        if (config.PollardRhoMilliseconds > 0 && pending.Count > 0)
        {
            long deadlineTimestamp = CreateDeadlineTimestamp(config.PollardRhoMilliseconds);
            FixedCapacityStack<UInt128> stack = result.CompositeStack;
            stack.Clear();
            stack.Push(remaining);
            pending.Clear();

            long timestamp; // reused for deadline checks.
            while (stack.Count > 0)
            {
                UInt128 composite = stack.Pop();
                if (composite <= UInt128.One)
                {
                    continue;
                }

                if (IsPrimeWide(composite))
                {
                    AddFactor(counts, composite, 1);
                    continue;
                }

                timestamp = Stopwatch.GetTimestamp();
                if (timestamp > deadlineTimestamp)
                {
                    pending.Add(composite);
                    continue;
                }

                if (!TryPollardRhoWide(composite, deadlineTimestamp, out UInt128 factor))
                {
                    pending.Add(composite);
                    continue;
                }

                UInt128 quotient = composite / factor;
                stack.Push(factor);
                stack.Push(quotient);
            }
        }

        UInt128 cofactor = UInt128.One;
        int pendingCount = pending.Count;
        for (int i = 0; i < pendingCount; i++)
        {
            UInt128 composite = pending[i];
            if (IsPrimeWide(composite))
            {
                AddFactor(counts, composite, 1);
            }
            else
            {
                BigInteger product = (BigInteger)cofactor * (BigInteger)composite;
                cofactor = (UInt128)product;
            }
        }

        if (counts.Count == 0 && cofactor == value)
        {
            result.Cofactor = value;
			// Everything else stays at default values initiated in .Rent.
            return result;
        }

        result.InitializeFromCounts(counts, cofactor, cofactor == UInt128.One);
        return result;
    }

    private static UInt128 PopulateSmallPrimeFactorsCpuWide(in UInt128 value, uint limit, Dictionary<UInt128, int> counts)
    {
        UInt128 remaining = value;
        uint[] primes = PrimesGenerator.SmallPrimes;
        ulong[] squares = PrimesGenerator.SmallPrimesPow2;
        int primeCount = primes.Length;

        for (int i = 0; i < primeCount; i++)
        {
            uint primeCandidate = primes[i];
            if (primeCandidate > limit)
            {
                break;
            }

            UInt128 primeSquare = (UInt128)squares[i];
            if (primeSquare > remaining)
            {
                break;
            }

            UInt128 primeValue = primeCandidate;
            if ((remaining % primeValue) != UInt128.Zero)
            {
                continue;
            }

            int exponent = 0;
            do
            {
                remaining /= primeValue;
                exponent++;
            }
            while ((remaining % primeValue) == UInt128.Zero);

            counts[primeValue] = exponent;
        }

        return remaining;
    }
    private static bool TryPollardRhoWide(in UInt128 n, long deadlineTimestamp, out UInt128 factor)
    {
        factor = UInt128.Zero;
        if ((n & UInt128.One) == UInt128.Zero)
        {
            factor = (UInt128)2UL;
            return true;
        }

        UInt128 c = UInt128.One;
        UInt128 x = (UInt128)2UL;
        UInt128 y = x;
        long timestamp; // reused for deadline checks.

        while (true)
        {
            timestamp = Stopwatch.GetTimestamp();
            if (timestamp > deadlineTimestamp)
            {
                return false;
            }

            x = AdvancePolynomialWide(x, c, n);
            y = AdvancePolynomialTwiceWide(y, c, n);

            UInt128 diff = x > y ? x - y : y - x;
            UInt128 d = BinaryGcdWide(diff, n);
            if (d == n)
            {
                c += UInt128.One;
                x = (UInt128)2UL;
                y = x;
                continue;
            }

            if (d > UInt128.One)
            {
                factor = d;
                return true;
            }
        }
    }

    private static UInt128 AdvancePolynomialWide(in UInt128 x, in UInt128 c, in UInt128 modulus)
    {
        BigInteger value = (BigInteger)x;
        value = (value * value + (BigInteger)c) % (BigInteger)modulus;
        return (UInt128)value;
    }

    private static UInt128 AdvancePolynomialTwiceWide(in UInt128 x, in UInt128 c, in UInt128 modulus)
    {
        BigInteger value = (BigInteger)x;
        value = (value * value + (BigInteger)c) % (BigInteger)modulus;
        value = (value * value + (BigInteger)c) % (BigInteger)modulus;
        return (UInt128)value;
    }

    private static UInt128 BinaryGcdWide(UInt128 a, UInt128 b)
    {
        if (a == UInt128.Zero)
        {
            return b;
        }

        if (b == UInt128.Zero)
        {
            return a;
        }

        int shift = TrailingZeroCountWide(a | b);
        a >>= TrailingZeroCountWide(a);

        while (true)
        {
            b >>= TrailingZeroCountWide(b);
            if (a > b)
            {
                (a, b) = (b, a);
            }

            b -= a;
            if (b == UInt128.Zero)
            {
                return a << shift;
            }
        }
    }

    private static int TrailingZeroCountWide(in UInt128 value)
    {
        if (value == UInt128.Zero)
        {
            return 128;
        }

        ulong low = (ulong)value;
        if (low != 0UL)
        {
            return BitOperations.TrailingZeroCount(low);
        }

        ulong high = (ulong)(value >> 64);
        return 64 + BitOperations.TrailingZeroCount(high);
    }

    private static void FactorCompletelyWide(in UInt128 value, Dictionary<UInt128, int> counts)
    {
        if (value <= UInt128.One)
        {
            return;
        }

        if (IsPrimeWide(value))
        {
            AddFactor(counts, value, 1);
            return;
        }

        UInt128 factor = PollardRhoWide(value);
        FactorCompletelyWide(factor, counts);
        FactorCompletelyWide(value / factor, counts);
    }

    private static UInt128 PollardRhoWide(in UInt128 n)
    {
        if ((n & UInt128.One) == UInt128.Zero)
        {
            return (UInt128)2UL;
        }

        UInt128 c = UInt128.One;
        while (true)
        {
            UInt128 x = (UInt128)2UL;
            UInt128 y = x;
            UInt128 d = UInt128.One;

            while (d == UInt128.One)
            {
                x = AdvancePolynomialWide(x, c, n);
                y = AdvancePolynomialTwiceWide(y, c, n);

                UInt128 diff = x > y ? x - y : y - x;
                d = BinaryGcdWide(diff, n);
            }

            if (d != n)
            {
                return d;
            }

            c += UInt128.One;
        }
    }

    private static bool IsPrimeWide(in UInt128 value)
    {
        if (value <= ulong.MaxValue)
        {
            // return HeuristicPrimeTester.Exclusive.IsPrimeCpu((ulong)value, CancellationToken.None);
            return PrimeTester.IsPrimeCpu((ulong)value);
        }

        if ((value & UInt128.One) == UInt128.Zero)
        {
            return false;
        }

        uint[] smallPrimes = PrimesGenerator.SmallPrimes;
        ulong[] smallPrimesPow2 = PrimesGenerator.SmallPrimesPow2;
        int length = smallPrimes.Length;
        for (int i = 0; i < length; i++)
        {
            uint prime = smallPrimes[i];
            UInt128 primeValue = prime;
            UInt128 primeSquare = (UInt128)smallPrimesPow2[i];
            if (primeSquare > value)
            {
                break;
            }

            if ((value % primeValue) == UInt128.Zero)
            {
                return false;
            }
        }

        BigInteger n = (BigInteger)value;
        BigInteger d = n - BigInteger.One;
        int s = 0;
        while ((d & 1) == 0)
        {
            d >>= 1;
            s++;
        }

        int[] bases = new[] { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 };
        for (int i = 0; i < bases.Length; i++)
        {
            int baseValue = bases[i];
            if (baseValue <= 1)
            {
                continue;
            }

            BigInteger a = baseValue;
            if (a >= n)
            {
                continue;
            }

            BigInteger x = BigInteger.ModPow(a, d, n);
            if (x == BigInteger.One || x == n - BigInteger.One)
            {
                continue;
            }

            bool witnessFound = true;
            for (int r = 1; r < s; r++)
            {
                x = BigInteger.ModPow(x, 2, n);
                if (x == n - BigInteger.One)
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

    private static void AddFactor(Dictionary<UInt128, int> counts, in UInt128 prime, int exponent)
    {
        if (counts.TryGetValue(prime, out int existing))
        {
            counts[prime] = existing + exponent;
        }
        else
        {
            counts[prime] = exponent;
        }
    }

	private sealed class PartialFactorResult128
	{
		[ThreadStatic]
		private static PartialFactorResult128? s_poolHead;

		public static readonly PartialFactorResult128 Empty = new()
		{
			Cofactor = UInt128.One,
			FullyFactored = true,
			Count = 0
		};

		private const int ExponentHardLimit = 256;

		internal readonly FactorEntry128[] TempFactors = new FactorEntry128[ExponentHardLimit];
		internal readonly List<UInt128> FactorCandidatesList = new(ExponentHardLimit);
		internal readonly Dictionary<UInt128, int> FactorCounts = new(capacity: ExponentHardLimit);
		internal readonly List<UInt128> Pending = new(capacity: ExponentHardLimit);
		internal readonly FixedCapacityStack<UInt128> CompositeStack = new(ExponentHardLimit);

		private PartialFactorResult128? _next;

		public UInt128 Cofactor;
		public int Count;
		public readonly FactorEntry128[] Factors = new FactorEntry128[ExponentHardLimit];
		public bool FullyFactored;
		public bool HasFactors;

		private PartialFactorResult128()
		{
		}

		public static PartialFactorResult128 Rent()
		{
			if (s_poolHead is { } instance)
			{
				s_poolHead = instance._next;
				instance._next = null;

				instance.Cofactor = UInt128.One;
				instance.Count = 0;
				instance.FullyFactored = false;
				instance.HasFactors = false;

				return instance;
			}

			return new PartialFactorResult128();
		}

		public void WithAdditionalPrime(UInt128 prime)
		{
			int index = Count;

			FactorEntry128[] buffer = Factors;
			buffer[index] = new FactorEntry128(prime, 1);
			index++;
			Count = index;

			buffer.AsSpan(0, index).Sort(static (a, b) => a.Value.CompareTo(b.Value));
			Cofactor = UInt128.One;
			FullyFactored = true;
			HasFactors = true;
		}

		public void Dispose()
		{
			if (this == Empty)
			{
				return;
			}

			_next = s_poolHead;
			s_poolHead = this;
		}

		internal void InitializeFromCounts(Dictionary<UInt128, int> counts, UInt128 cofactor, bool fullyFactored)
		{
			int count = counts.Count;
			Count = count;
			Cofactor = cofactor;
			FullyFactored = fullyFactored;
			bool hasFactors = count != 0;
			HasFactors = hasFactors;

			if (!hasFactors)
			{
				return;
			}

			int index = 0;
			foreach (KeyValuePair<UInt128, int> entry in counts)
			{
				Factors[index++] = new FactorEntry128(entry.Key, entry.Value);
			}

			// We benefit from static (non-virtual) call to Sort. When JIT finds a method call after a constructor, it optimizes it to a true static call.
			new Span<FactorEntry128>(Factors, 0, count).Sort(static (a, b) => a.Value.CompareTo(b.Value));
		}
	}
}
