namespace PerfectNumbers.Core;

public static class PrimesGenerator
{
	private static uint[]? _primes;
	private static ulong[]? _primesPow2;

	public static readonly uint[] SmallPrimes = BuildSmallPrimesInternal();
	public static readonly ulong[] SmallPrimesPow2 = BuildSmallPrimesPow2Internal();


	private static uint[] BuildSmallPrimesInternal()
	{
		if (_primes == null)
		{
			BuildSmallPrimes();
		}

		return _primes!;
	}

	private static ulong[] BuildSmallPrimesPow2Internal()
	{
		if (_primesPow2 == null)
		{
			BuildSmallPrimes();
		}

		return _primesPow2!;
	}

	private static void BuildSmallPrimes()
	{
		uint limit = PerfectNumberConstants.PrimesLimit;
		var primes = new uint[limit];
		primes[0] = 2;
		var primesPow2 = new ulong[limit];
		primesPow2[0] = 4;

		int j;
		uint i = 3, p, sqrt;
		bool isPrime;
		int primeIndex = 1;

		for (; i <= limit; i += 2)
		{
			isPrime = true;
			sqrt = (uint)Math.Sqrt(i);
			for (j = 0; j < primeIndex; j++)
			{
				p = primes[j];
				if (p > sqrt)
				{
					break;
				}

				if (i % p == 0UL)
				{
					isPrime = false;
					break;
				}
			}

			if (isPrime)
			{
				primes[primeIndex] = i;
				primesPow2[primeIndex] = checked(i * i);
				primeIndex++;
			}
		}

		Array.Resize(ref primes, primeIndex);
		Array.Resize(ref primesPow2, primeIndex);

		_primes = primes;
		_primesPow2 = primesPow2;

		return;
	}
}