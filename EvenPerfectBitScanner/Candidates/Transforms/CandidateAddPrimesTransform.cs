using System.Runtime.CompilerServices;
using System.Threading;
using EvenPerfectBitScanner.Candidates;
using Open.Numeric.Primes;

namespace EvenPerfectBitScanner.Candidates.Transforms;

internal static class CandidateAddPrimesTransform
{
    private static readonly Optimized PrimeIterator = new();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static ulong Transform(ulong value, ref ulong remainder, ref bool limitReached)
    {
        ulong originalRemainder = remainder;
        ulong addRemainder = remainder;
        ulong candidate = value;
        ulong primeCandidate = value;
        bool advanceAdd = true;
        bool advancePrime = true;
        ulong diff;
        ulong nextPrime;

        while (true)
        {
            if (advanceAdd)
            {
                diff = addRemainder.GetNextAddDiff();
                if (candidate > ulong.MaxValue - diff)
                {
                    Volatile.Write(ref limitReached, true);
                    remainder = originalRemainder;
                    return candidate;
                }

                candidate += diff;
                addRemainder += diff;
                // Skip the Mod6 lookup: benchmarked `%` stays ahead for these prime increments.
                // benchmarked fastest remainder updates instead of looping subtraction.
                while (addRemainder >= 6UL)
                {
                    addRemainder -= 6UL;
                }
            }

            if (advancePrime)
            {
                try
                {
                    nextPrime = PrimeIterator.Next(in primeCandidate);
                }
                catch (InvalidOperationException)
                {
                    Volatile.Write(ref limitReached, true);
                    remainder = originalRemainder;
                    return value;
                }

                if (nextPrime <= primeCandidate)
                {
                    Volatile.Write(ref limitReached, true);
                    remainder = originalRemainder;
                    return value;
                }

                primeCandidate = nextPrime;
            }

            if (candidate == primeCandidate)
            {
                remainder = addRemainder;
                return candidate;
            }

            advanceAdd = candidate < primeCandidate;
            advancePrime = candidate > primeCandidate;
        }
    }
}
