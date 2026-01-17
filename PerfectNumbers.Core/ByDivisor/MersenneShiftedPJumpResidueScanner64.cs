using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core.Cpu;

/// <summary>
/// Jump-capable (power-of-two composition) residue automaton for M_p = 2^p - 1.
///
/// For a fixed candidate divisor q (bit-length <= maxQBits), this is an exact decision procedure:
/// it checks whether there exists an a-bitstream such that a*q equals 2^p-1, i.e. the output bits are
///   - 1 for columns j = 0..p-1
///   - 0 for columns j >= p (and the multiplication "drains" to zero).
///
/// No powmod2 is used.
///
/// NOTE: This algorithm is intended for your "sparse shifted-p" family:
///   q = 1 + Σ_{u in U} (p << u)
/// with small |U|, where q itself is not supplied as ulong but can be materialized as BigInteger at the end.
///
/// IMPORTANT: This implementation supports q of arbitrary size (bounded by maxQBits for the run), but
/// the state space can still be large for big maxQBits.
/// </summary>
public static class MersenneShiftedPJumpResidueScanner64
{
    public enum ScanResult
    {
        RuledOut = 0,
        FoundDivisor = 1,
    }

    // --------------------------- Bit window (immutable) ---------------------------

    private sealed class BitWindow : IEquatable<BitWindow>
    {
        private readonly ulong[] _words; // little endian words; bit i in word i>>6
        private readonly ulong _lastMask;
        private readonly int _hash;

        public int Bits { get; }
        public int WordCount => _words.Length;

        public BitWindow(int bits, bool setBit0)
        {
            if (bits <= 0) throw new ArgumentOutOfRangeException(nameof(bits));
            Bits = bits;
            _words = new ulong[(bits + 63) >> 6];
            int lastBits = bits - ((_words.Length - 1) << 6);
            _lastMask = lastBits == 64 ? ulong.MaxValue : ((1UL << lastBits) - 1UL);
            if (setBit0) _words[0] = 1UL;
            _words[^1] &= _lastMask;
            _hash = ComputeHash(_words);
        }

        private BitWindow(int bits, ulong[] words, ulong lastMask, int hash)
        {
            Bits = bits;
            _words = words;
            _lastMask = lastMask;
            _hash = hash;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetBit(int idx)
        {
            if ((uint)idx >= (uint)Bits) return 0;
            return (int)((_words[idx >> 6] >> (idx & 63)) & 1UL);
        }

        /// <summary>
        /// Shift left by 1 within the fixed width and OR-in nextBit at position 0 (i.e. new a_j at LSB).
        /// This matches the representation: bit0 = a_j, bit i = a_{j-i}.
        /// </summary>
        public BitWindow ShiftAddBit(int nextBit)
        {
            ulong carry = (uint)nextBit & 1U;
            var nw = new ulong[_words.Length];

            for (int i = 0; i < _words.Length; i++)
            {
                ulong w = _words[i];
                ulong nextCarry = w >> 63;
                ulong shifted = (w << 1) | carry;
                nw[i] = shifted;
                carry = nextCarry;
            }

            nw[^1] &= _lastMask;
            int h = ComputeHash(nw);
            return new BitWindow(Bits, nw, _lastMask, h);
        }

        public bool IsZero()
        {
            for (int i = 0; i < _words.Length; i++)
                if (_words[i] != 0) return false;
            return true;
        }

        public bool Equals(BitWindow? other)
        {
            if (ReferenceEquals(this, other)) return true;
            if (other is null) return false;
            if (Bits != other.Bits) return false;
            if (_hash != other._hash) return false;
            if (_words.Length != other._words.Length) return false;
            for (int i = 0; i < _words.Length; i++)
                if (_words[i] != other._words[i]) return false;
            return true;
        }

        public override bool Equals(object? obj) => obj is BitWindow bw && Equals(bw);
        public override int GetHashCode() => _hash;

        private static int ComputeHash(ulong[] words)
        {
            // FNV-1a-ish
            const ulong FnvOffset = 1469598103934665603UL;
            const ulong FnvPrime = 1099511628211UL;
            ulong h = FnvOffset;
            for (int i = 0; i < words.Length; i++)
            {
                h ^= words[i];
                h *= FnvPrime;
            }
            return unchecked((int)(h ^ (h >> 32)));
        }
    }

    // ------------------------------ DP state ------------------------------

    private readonly struct State : IEquatable<State>
    {
        public readonly int Carry;
        public readonly BitWindow AWin;

        public State(int carry, BitWindow aWin)
        {
            Carry = carry;
            AWin = aWin;
        }

        public bool Equals(State other) => Carry == other.Carry && AWin.Equals(other.AWin);
        public override bool Equals(object? obj) => obj is State s && Equals(s);
        public override int GetHashCode() => HashCode.Combine(Carry, AWin);
    }

    // ------------------------------ q representation ------------------------------

    private sealed class QBits
    {
        private readonly ulong[] _words;
        public int Bits { get; }
        public int PopCount { get; }

        public QBits(BigInteger q, int maxQBits)
        {
            if (maxQBits <= 0) throw new ArgumentOutOfRangeException(nameof(maxQBits));
            Bits = maxQBits;
            int wc = (maxQBits + 63) >> 6;
            _words = new ulong[wc];

            BigInteger tmp = q;
            for (int i = 0; i < wc; i++)
            {
                _words[i] = (ulong)(tmp & ulong.MaxValue);
                tmp >>= 64;
            }

            // mask out bits above maxQBits
            int lastBits = maxQBits - ((wc - 1) << 6);
            ulong lastMask = lastBits == 64 ? ulong.MaxValue : ((1UL << lastBits) - 1UL);
            _words[^1] &= lastMask;

            int pc = 0;
            for (int i = 0; i < wc; i++) pc += BitOperations.PopCount(_words[i]);
            PopCount = pc;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetBit(int idx)
        {
            if ((uint)idx >= (uint)Bits) return 0;
            return (int)((_words[idx >> 6] >> (idx & 63)) & 1UL);
        }

        /// <summary>Enumerate indices of set bits (0..Bits-1).</summary>
        public IEnumerable<int> EnumerateOnes()
        {
            for (int wi = 0; wi < _words.Length; wi++)
            {
                ulong w = _words[wi];
                while (w != 0)
                {
                    int tz = BitOperations.TrailingZeroCount(w);
                    int bit = (wi << 6) + tz;
                    if (bit >= Bits) yield break;
                    yield return bit;
                    w &= w - 1;
                }
            }
        }
    }

    // -------------------------- One-step transition --------------------------

    private static void NextStatesOneStep(
        QBits qBits,
        int maxOnes,
        int targetBit,
        in State s,
        List<State> dst)
    {
        int sum = s.Carry;

        // Σ_{i where q_i=1} a_{j-i}  -- note: AWin bit i is a_{j-i}
        foreach (int i in qBits.EnumerateOnes())
        {
            sum += s.AWin.GetBit(i);
        }

        if ((sum & 1) != targetBit) return;

        int newCarry = sum >> 1;

        // Very conservative ceiling to avoid runaway state blow-up on junk candidates.
        // For your sparse q, maxOnes is typically small.
        if (newCarry > (maxOnes + 256)) return;

        // branch on next a-bit (a_{j+1} ∈ {0,1})
        dst.Add(new State(newCarry, s.AWin.ShiftAddBit(0)));
        dst.Add(new State(newCarry, s.AWin.ShiftAddBit(1)));
    }

    // -------------------------- Jump table (targetBit=1) --------------------------

    private sealed class JumpTable
    {
        private readonly QBits _qBits;
        private readonly int _maxOnes;
        private readonly List<Dictionary<State, HashSet<State>>> _pow = new();

        public JumpTable(QBits qBits)
        {
            _qBits = qBits;
            _maxOnes = qBits.PopCount;
            _pow.Add(new Dictionary<State, HashSet<State>>(capacity: 1024)); // k=0
        }

        private HashSet<State> GetPow0(in State s)
        {
            var d0 = _pow[0];
            if (d0.TryGetValue(s, out var set)) return set;

            var tmp = new List<State>(2);
            NextStatesOneStep(_qBits, _maxOnes, targetBit: 1, s, tmp);

            set = new HashSet<State>(tmp);
            d0[s] = set;
            return set;
        }

        private HashSet<State> GetPowK(int k, in State s)
        {
            EnsurePow(k);
            var dk = _pow[k];
            if (dk.TryGetValue(s, out var set)) return set;

            if (k == 0) return GetPow0(s);

            var mid = GetPowK(k - 1, s);
            var outSet = new HashSet<State>();
            foreach (var m in mid)
            {
                foreach (var e in GetPowK(k - 1, m))
                    outSet.Add(e);
            }
            dk[s] = outSet;
            return outSet;
        }

        private void EnsurePow(int k)
        {
            while (_pow.Count <= k)
                _pow.Add(new Dictionary<State, HashSet<State>>(capacity: 1024));
        }

        public HashSet<State> Advance(HashSet<State> start, ulong steps)
        {
            if (steps == 0) return start;

            int hi = 63 - BitOperations.LeadingZeroCount(steps);

            var cur = start;
            for (int k = 0; k <= hi; k++)
            {
                if (((steps >> k) & 1UL) == 0) continue;

                var next = new HashSet<State>();
                foreach (var s in cur)
                {
                    foreach (var e in GetPowK(k, s))
                        next.Add(e);
                }

                cur = next;
                if (cur.Count == 0) break;
            }

            return cur;
        }

        public int MaxOnes => _maxOnes;
        public QBits QBits => _qBits;
    }

    // ------------------------------ Tail drain (targetBit=0) ------------------------------

    private static bool CanDrainToZero(QBits qBits, int maxOnes, HashSet<State> afterMain)
    {
        var frontier = new HashSet<State>(afterMain);
        var seen = new HashSet<State>(afterMain);

        int maxSteps = qBits.Bits + 256; // conservative

        for (int step = 0; step < maxSteps; step++)
        {
            if (frontier.Count == 0) break;

            var nextFrontier = new HashSet<State>();

            foreach (var s in frontier)
            {
                if (s.Carry == 0 && s.AWin.IsZero())
                    return true;

                var tmp = new List<State>(2);
                NextStatesOneStep(qBits, maxOnes, targetBit: 0, s, tmp);
                foreach (var ns in tmp)
                    if (seen.Add(ns))
                        nextFrontier.Add(ns);
            }

            frontier = nextFrontier;
        }

        return false;
    }

    // ------------------------------ Public API ------------------------------

    /// <summary>
    /// Exact decision: does there exist an a-bitstream such that a*q == 2^p-1 and no higher bits,
    /// using only the automaton (no powmod2)?
    /// </summary>
    public static bool IsDivisorByJumpAutomaton(ulong primeP, BigInteger q, int maxQBits)
    {
        if (maxQBits <= 0) throw new ArgumentOutOfRangeException(nameof(maxQBits));
        if (q.Sign <= 0) return false;
        if ((q & 1) == 0) return false;
        if (q == BigInteger.One) return false;

        // Require q to fit within maxQBits.
        if (BitLength(q) > maxQBits) return false;

        // quick low-bit filter: q mod 8 in {1,7} (necessary in your current search family)
        int mod8 = (int)(q & 7);
        if (mod8 != 1 && mod8 != 7) return false;

        var qb = new QBits(q, maxQBits);

        // Initial state at j=0: a0=1 => AWin bit0=1; carry=0
        var initA = new BitWindow(maxQBits, setBit0: true);
        var initSet = new HashSet<State> { new State(0, initA) };

        var jt = new JumpTable(qb);
        var afterMain = jt.Advance(initSet, primeP);
        if (afterMain.Count == 0) return false;

        return CanDrainToZero(qb, jt.MaxOnes, afterMain);
    }

    /// <summary>
    /// Enumerate sparse U combinations and test q via the jump automaton.
    /// Returns the found q as BigInteger (materialized) with no 64-bit limit.
    /// </summary>
    public static ScanResult TryFindDivisorBySparseShiftU(
        ulong primeP,
        int maxQBits,
        int maxShiftTerms,
        out BigInteger foundQ)
    {
        foundQ = BigInteger.Zero;

        if (maxQBits <= 0) throw new ArgumentOutOfRangeException(nameof(maxQBits));
        if (maxShiftTerms <= 0) throw new ArgumentOutOfRangeException(nameof(maxShiftTerms));

        // Candidate shifts u are in [1, maxQBits-1]
        int maxU = maxQBits - 1;
        if (maxU <= 0) return ScanResult.RuledOut;

        BigInteger pBI = new BigInteger(primeP);

        for (int terms = 1; terms <= maxShiftTerms; terms++)
        {
            if (terms > maxU) break;

            var comb = new int[terms];
            for (int i = 0; i < terms; i++) comb[i] = i + 1;

            while (true)
            {
                BigInteger q = BigInteger.One;
                for (int i = 0; i < comb.Length; i++)
                {
                    int u = comb[i];
                    q += (pBI << u);

                    // Early cut: if already exceeds bit budget, skip building further.
                    if (BitLength(q) > maxQBits)
                        break;
                }

                if (BitLength(q) <= maxQBits)
                {
                    if (IsDivisorByJumpAutomaton(primeP, q, maxQBits))
                    {
                        foundQ = q;
                        return ScanResult.FoundDivisor;
                    }
                }

                // next combination
                int pos = terms - 1;
                while (pos >= 0 && comb[pos] == maxU - (terms - 1 - pos)) pos--;
                if (pos < 0) break;
                comb[pos]++;
                for (int j = pos + 1; j < terms; j++)
                    comb[j] = comb[j - 1] + 1;
            }
        }

        return ScanResult.RuledOut;
    }

    // ------------------------------ Helpers ------------------------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int BitLength(in BigInteger x)
    {
        if (x.Sign == 0) return 0;
        // BigInteger uses two's complement; for positive numbers this is OK.
        byte[] bytes = x.ToByteArray(isUnsigned: true, isBigEndian: true);
        if (bytes.Length == 0) return 0;
        byte msb = bytes[0];
        int lz = 0;
        for (int b = 7; b >= 0; b--)
        {
            if (((msb >> b) & 1) != 0) break;
            lz++;
        }
        return bytes.Length * 8 - lz;
    }
}
