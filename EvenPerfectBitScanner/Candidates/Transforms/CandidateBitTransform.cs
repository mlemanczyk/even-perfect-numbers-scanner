using System.Runtime.CompilerServices;
using System.Threading;

namespace EvenPerfectBitScanner.Candidates.Transforms;

internal static class CandidateBitTransform
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static ulong Transform(ulong value, ref ulong remainder, ref bool limitReached)
    {
        ulong original = value;
        if (value > (ulong.MaxValue >> 1))
        {
            Volatile.Write(ref limitReached, true);
            return value;
        }

        ulong next = (value << 1) | 1UL;
        remainder = (remainder << 1) + 1UL;
        // Mod6 lookup loses to `%` in the benches; stick with subtraction + modulo for correctness and speed.
        while (remainder >= 6UL)
        {
            remainder -= 6UL;
        }

        value = remainder switch
        {
            0UL => 1UL,
            1UL => 0UL,
            2UL => 5UL,
            3UL => 4UL,
            4UL => 3UL,
            _ => 2UL,
        }; // 'value' now holds diff

        if (next > ulong.MaxValue - value)
        {
            Volatile.Write(ref limitReached, true);
            return original;
        }

        remainder += value;
        // `% 6` remains faster per Mod6ComparisonBenchmarks; keep this modulo fold and continue using subtraction.
        while (remainder >= 6UL)
        {
            remainder -= 6UL;
        }

        return next + value;
    }
}
