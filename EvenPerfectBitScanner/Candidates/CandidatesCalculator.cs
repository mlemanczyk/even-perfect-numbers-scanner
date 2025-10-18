using System.Runtime.CompilerServices;
using System.Threading;
using Open.Numeric.Primes;
using EvenPerfectBitScanner.Candidates.Transforms;

namespace EvenPerfectBitScanner.Candidates;

internal static class CandidatesCalculator
{
    private static readonly Optimized PrimeIterator = new();
    private static long _state;
    private static PrimeTransformMode _transformMode;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void Configure(PrimeTransformMode transformMode)
    {
        _transformMode = transformMode;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void InitializeState(ulong startPrime, ulong remainder)
    {
        Volatile.Write(ref _state, ((long)startPrime << 3) | (long)remainder);
    }

    internal static List<ulong> BuildTestPrimeCandidates(int targetCount)
    {
        if (targetCount <= 0)
        {
            return [];
        }

        List<ulong> candidates = new(targetCount);
        ulong previous = 29UL;
        int remaining = targetCount;
        ulong nextPrime;

        while (remaining > 0)
        {
            try
            {
                nextPrime = PrimeIterator.Next(in previous);
            }
            catch (InvalidOperationException)
            {
                break;
            }

            if (nextPrime < 31UL)
            {
                previous = nextPrime;
                continue;
            }

            candidates.Add(nextPrime);
            remaining--;
            previous = nextPrime;
        }

        if (remaining > 0)
        {
            Console.WriteLine("Unable to populate the requested number of test primes before reaching the 64-bit limit.");
        }

        return candidates;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static ulong AdvancePrime(ulong value, ref ulong remainder, ref bool limitReached)
    {
        return _transformMode switch
        {
            PrimeTransformMode.Bit => CandidateBitTransform.Transform(value, ref remainder, ref limitReached),
            PrimeTransformMode.Add => CandidateAddTransform.Transform(value, ref remainder, ref limitReached),
            _ => CandidateAddPrimesTransform.Transform(value, ref remainder, ref limitReached),
        };
    }

    internal static int ReserveBlock(ulong[] buffer, int blockSize, ref bool limitReached)
    {
        Span<ulong> bufferSpan = new(buffer);

        while (true)
        {
            long stateSnapshot = Volatile.Read(ref _state);
            ulong p = (ulong)stateSnapshot >> 3;
            ulong remainder = ((ulong)stateSnapshot) & 7UL;

            int count = 0;
            switch (_transformMode)
            {
                case PrimeTransformMode.Bit:
                    while (count < blockSize && !Volatile.Read(ref limitReached))
                    {
                        bufferSpan[count++] = p;
                        p = CandidateBitTransform.Transform(p, ref remainder, ref limitReached);
                    }

                    break;

                case PrimeTransformMode.Add:
                    while (count < blockSize && !Volatile.Read(ref limitReached))
                    {
                        bufferSpan[count++] = p;
                        p = CandidateAddTransform.Transform(p, ref remainder, ref limitReached);
                    }

                    break;

                default:
                    while (count < blockSize && !Volatile.Read(ref limitReached))
                    {
                        bufferSpan[count++] = p;
                        p = CandidateAddPrimesTransform.Transform(p, ref remainder, ref limitReached);
                    }

                    break;
            }

            if (count == 0)
            {
                return 0;
            }

            long newState = ((long)p << 3) | (long)remainder;
            long original = Interlocked.CompareExchange(ref _state, newState, stateSnapshot);
            if (original == stateSnapshot)
            {
                return count;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static ulong GetNextAddDiff(this ulong remainder)
    {
        return remainder switch
        {
            0UL => 1UL,
            1UL => 4UL,
            2UL => 3UL,
            3UL => 2UL,
            4UL => 1UL,
            _ => 2UL,
        };
    }
}
