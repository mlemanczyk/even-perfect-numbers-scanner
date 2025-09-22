using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public static class PrimesGenerator
{
	public static readonly uint[] SmallPrimes;
	public static readonly ulong[] SmallPrimesPow2;
	public static readonly uint[] SmallPrimesLastOne;
	public static readonly ulong[] SmallPrimesPow2LastOne;
	public static readonly uint[] SmallPrimesLastSeven;
	public static readonly ulong[] SmallPrimesPow2LastSeven;

	static PrimesGenerator()
	{
		BuildSmallPrimes(
		out SmallPrimes,
		out SmallPrimesPow2,
		out SmallPrimesLastOne,
		out SmallPrimesPow2LastOne,
		out SmallPrimesLastSeven,
		out SmallPrimesPow2LastSeven);
	}

	private static void BuildSmallPrimes(
	out uint[] all,
	out ulong[] allPow2,
	out uint[] lastOne,
	out ulong[] lastOnePow2,
	out uint[] lastSeven,
	out ulong[] lastSevenPow2)
	{
		uint target = PerfectNumberConstants.PrimesLimit;
		int targetInt = checked((int)target);
		all = new uint[targetInt];
		allPow2 = new ulong[targetInt];
		lastOne = new uint[targetInt];
		lastOnePow2 = new ulong[targetInt];
		lastSeven = new uint[targetInt];
		lastSevenPow2 = new ulong[targetInt];

		List<uint> primes = new(targetInt * 4 / 3 + 16);
		uint candidate = 2U;
		int allCount = 0;
		int lastOneCount = 0;
		int lastSevenCount = 0;

		while (allCount < targetInt || lastOneCount < targetInt || lastSevenCount < targetInt)
		{
			bool isPrime = true;
			int primesCount = primes.Count;
			for (int i = 0; i < primesCount; i++)
			{
				uint p = primes[i];
				if (p * p > candidate)
				{
					break;
				}

				if (candidate % p == 0U)
				{
					isPrime = false;
					break;
				}
			}

			if (isPrime)
			{
				primes.Add(candidate);
				if (allCount < targetInt)
				{
					all[allCount] = candidate;
					allPow2[allCount] = checked((ulong)candidate * candidate);
					allCount++;
				}

				if (candidate != 2U)
				{
					if (lastOneCount < targetInt && IsAllowedForLastOne(candidate))
					{
						lastOne[lastOneCount] = candidate;
						lastOnePow2[lastOneCount] = checked((ulong)candidate * candidate);
						lastOneCount++;
					}

					if (lastSevenCount < targetInt && IsAllowedForLastSeven(candidate))
					{
						lastSeven[lastSevenCount] = candidate;
						lastSevenPow2[lastSevenCount] = checked((ulong)candidate * candidate);
						lastSevenCount++;
					}
				}
			}

			candidate = candidate == 2U ? 3U : candidate + 2U;
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static bool IsAllowedForLastOne(uint prime)
	{
                uint mod10 = prime % 10U;
		return mod10 == 1U || mod10 == 3U || mod10 == 9U || prime == 7U || prime == 11U;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static bool IsAllowedForLastSeven(uint prime)
	{
                uint mod10 = prime % 10U;
		return mod10 == 3U || mod10 == 7U || mod10 == 9U || prime == 11U;
	}
}
