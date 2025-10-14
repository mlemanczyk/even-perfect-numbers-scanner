using System;

namespace PerfectNumbers.Core;

internal static class MersenneNumberDivisorCandidateCpuEvaluator
{
    private const int RemainderCount = 6;

    public static void EvaluateCandidates(
        ulong startDivisor,
        ulong step,
        ulong limit,
        Span<byte> remainders,
        ReadOnlySpan<byte> stepDeltas,
        bool lastIsSeven,
        Span<ulong> candidates,
        Span<byte> mask)
    {
        if (remainders.Length != RemainderCount)
        {
            throw new ArgumentException("Remainder buffer must contain exactly six entries.", nameof(remainders));
        }

        if (stepDeltas.Length != RemainderCount)
        {
            throw new ArgumentException("Step buffer must contain exactly six entries.", nameof(stepDeltas));
        }

        if (mask.Length < candidates.Length)
        {
            throw new ArgumentException("Mask span must be at least as long as the candidate span.", nameof(mask));
        }

        byte remainder10 = remainders[0];
        byte remainder8 = remainders[1];
        byte remainder5 = remainders[2];
        byte remainder3 = remainders[3];
        byte remainder7 = remainders[4];
        byte remainder11 = remainders[5];

        byte step10 = stepDeltas[0];
        byte step8 = stepDeltas[1];
        byte step5 = stepDeltas[2];
        byte step3 = stepDeltas[3];
        byte step7 = stepDeltas[4];
        byte step11 = stepDeltas[5];

        for (int index = 0; index < candidates.Length; index++)
        {
            ulong offset = (ulong)index;
            ulong candidate = startDivisor + step * offset;
            candidates[index] = candidate;

            if (candidate > limit || (step != 0UL && candidate < startDivisor))
            {
                mask[index] = 0;
            }
            else
            {
                bool accept10 = lastIsSeven
                    ? (remainder10 == 3 || remainder10 == 7 || remainder10 == 9)
                    : (remainder10 == 1 || remainder10 == 3 || remainder10 == 9);

                if (accept10
                    && (remainder8 == 1 || remainder8 == 7)
                    && remainder3 != 0
                    && remainder5 != 0
                    && remainder7 != 0
                    && remainder11 != 0)
                {
                    mask[index] = 1;
                }
                else
                {
                    mask[index] = 0;
                }
            }

            remainder10 = AddMod(remainder10, step10, 10);
            remainder8 = AddMod(remainder8, step8, 8);
            remainder5 = AddMod(remainder5, step5, 5);
            remainder3 = AddMod(remainder3, step3, 3);
            remainder7 = AddMod(remainder7, step7, 7);
            remainder11 = AddMod(remainder11, step11, 11);
        }

        remainders[0] = remainder10;
        remainders[1] = remainder8;
        remainders[2] = remainder5;
        remainders[3] = remainder3;
        remainders[4] = remainder7;
        remainders[5] = remainder11;
    }

    private static byte AddMod(byte value, byte delta, byte modulus)
    {
        int result = value + delta;
        if (result >= modulus)
        {
            result -= modulus;
        }

        return (byte)result;
    }
}
