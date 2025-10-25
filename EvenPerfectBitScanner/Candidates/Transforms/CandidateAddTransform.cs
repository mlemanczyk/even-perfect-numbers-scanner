using System.Runtime.CompilerServices;
using System.Threading;
using EvenPerfectBitScanner.Candidates;

namespace EvenPerfectBitScanner.Candidates.Transforms;

internal static class CandidateAddTransform
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static ulong Transform(ulong value, ref ulong remainder, ref bool limitReached)
    {
        ulong next = value;
        ulong diff = remainder.GetNextAddDiff();

        if (next > ulong.MaxValue - diff)
        {
            Volatile.Write(ref limitReached, true);
            return next;
        }

        remainder += diff;
        // Retain direct modulo because the Mod6 helper underperforms on 64-bit operands.
        while (remainder >= 6UL)
        {
            remainder -= 6UL;
        }

        return next + diff;
    }
}
