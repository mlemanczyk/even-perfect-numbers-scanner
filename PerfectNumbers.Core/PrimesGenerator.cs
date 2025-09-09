namespace PerfectNumbers.Core;

public static class PrimesGenerator
{
	private static ulong[]? _primes;
	private static ulong[]? _primesPow2;	
	// private static UInt128[]? _primes128;
	// private static UInt128[]? _primesPow2_128;	

	public static readonly ulong[] SmallPrimes = BuildSmallPrimesInternal();
	public static readonly ulong[] SmallPrimesPow2 = BuildSmallPrimesPow2Internal();
	// public static readonly UInt128[] SmallPrimes128 = BuildSmallPrimes128Internal();
	// public static readonly UInt128[] SmallPrimesPow2_128 = BuildSmallPrimesPow2_128Internal();
	

	private static ulong[] BuildSmallPrimesInternal()
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

	// private static UInt128[] BuildSmallPrimes128Internal()
	// {
	// 	if (_primes == null)
	// 	{
	// 		BuildSmallPrimes();
	// 	}

	// 	return _primes128!;
	// }

	// private static UInt128[] BuildSmallPrimesPow2_128Internal()
	// {
	// 	if (_primesPow2 == null)
	// 	{
	// 		BuildSmallPrimes();
	// 	}

	// 	return _primesPow2_128!;
	// }

	private static void BuildSmallPrimes()
	{
               ulong limit = PerfectNumberConstants.PrimesLimit;
               var primes = new ulong[limit];
               primes[0] = 2UL;
               var primesPow2 = new ulong[limit];
               primesPow2[0] = 4UL;

               int j;
               ulong i = 3UL, p, sqrt;
               bool isPrime;
               int primeIndex = 1;

               for (; i <= limit; i += 2UL)
               {
                       isPrime = true;
                       sqrt = (ulong)Math.Sqrt(i);
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
                               primesPow2[primeIndex] = i * i;
                               primeIndex++;
                       }
               }

               Array.Resize(ref primes, primeIndex);
               Array.Resize(ref primesPow2, primeIndex);

               _primes = primes;
               _primesPow2 = primesPow2;
               // _primes128 = Array.ConvertAll(_primes, p => (UInt128)p);
               // _primesPow2_128 = Array.ConvertAll(_primesPow2, p => (UInt128)p);

               return;
       }
}