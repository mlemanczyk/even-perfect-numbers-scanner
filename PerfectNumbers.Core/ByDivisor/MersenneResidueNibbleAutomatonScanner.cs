
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core.Cpu;

/// <summary>
/// Block-ordered streaming divisor scan for M_p = 2^p - 1 in the family q = (2p)k + 1,
/// using a 4-bit (base-16) residue automaton.
///
/// Key design change vs column-wise branching:
/// - We always assign an a-nibble (4 planned a-bits) before stepping the 4 columns.
/// - With planned bits present, each of the 4 columns becomes deterministic (no 0/1 branching),
///   so we can test each a-nibble in O(1) per column and cache allowed masks by signatures.
///
/// This makes the "residue per block" perspective operational while staying exact for the explored tree.
/// </summary>
public static class MersenneResidueNibbleAutomatonScanner
{
    public enum ScanResult
    {
        RuledOut = 0,
        Candidate = 1,
        FoundDivisor = 2
    }

    public const int MaxSupportedQBits = 8192;

    // Cache allowed a-nibble masks only for moderate carry values.
    // For larger carry we compute directly to avoid any risk of false pruning.
    private const int CarryCacheMax = 511;

    // Reuse the same block-0 filter as the original scanner: q odd and q mod 8 in {1,7}.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IsAllowedQ0Nibble(byte q0)
    {
        if ((q0 & 0x1) == 0) return false;
        int mod8 = q0 & 0x7;
        return mod8 == 0x1 || mod8 == 0x7;
    }

    private readonly struct AllowedMaskKey(
        int pBits,
        int maxQBits,
        int block,
        int carry,
        byte qNibble,
        ulong qSig,
        ulong aSig) : IEquatable<AllowedMaskKey>
    {
        public readonly int PBits = pBits;
        public readonly int MaxQBits = maxQBits;
        public readonly int Block = block;
        public readonly int Carry = carry;
        public readonly byte QNibble = qNibble;
        public readonly ulong QSig = qSig;
        public readonly ulong ASig = aSig;

        public bool Equals(AllowedMaskKey other)
            => PBits == other.PBits
               && MaxQBits == other.MaxQBits
               && Block == other.Block
               && Carry == other.Carry
               && QNibble == other.QNibble
               && QSig == other.QSig
               && ASig == other.ASig;

        public override bool Equals(object? obj) => obj is AllowedMaskKey k && Equals(k);
        public override int GetHashCode() => HashCode.Combine(PBits, MaxQBits, Block, Carry, QNibble, QSig, ASig);
    }

    private struct State
    {
        public int Block;
        public int BitIndex;
        public int Carry;

        public bool KTerminated;
        public byte KFlushRemaining;
        public byte PostTermZeroQNibbles;

        public MersenneCombinedDivisorScannerFullStreamExtended.DynamicBitset QBits;
        public MersenneCombinedDivisorScannerFullStreamExtended.ABitsWindow AWindow;
        public MersenneCombinedDivisorScannerFullStreamExtended.QGenState16 QState;

        public State CloneShallow() => new()
        {
            Block = Block,
            BitIndex = BitIndex,
            Carry = Carry,
            KTerminated = KTerminated,
            KFlushRemaining = KFlushRemaining,
            PostTermZeroQNibbles = PostTermZeroQNibbles,
            QBits = QBits.CloneShared(),
            AWindow = AWindow.Clone(),
            QState = QState
        };

        public static State CreateInitial(MersenneCombinedDivisorScannerFullStreamExtended.QGenState16 initQ, int maxQBits)
        {
            var s = new State
            {
                Block = initQ.NibbleIndex,
                BitIndex = 0,
                Carry = 0,
                QBits = new MersenneCombinedDivisorScannerFullStreamExtended.DynamicBitset(maxQBits),
                AWindow = new MersenneCombinedDivisorScannerFullStreamExtended.ABitsWindow(maxQBits),
                QState = initQ,
                KTerminated = false,
                KFlushRemaining = 0,
                PostTermZeroQNibbles = 0
            };

            s.QBits.Set(0);      // q0 = 1
            s.AWindow.SetPlanned(0, 1); // a0 = 1
            return s;
        }
    }

    /// <summary>
    /// Try to find a divisor q (within maxQBits) for M_p by streaming q via QNibbleGenerator16 and
    /// validating with the deterministic planned-nibble automaton.
    /// </summary>
    public static ScanResult TryFindDivisorByResidueAutomaton(
        ulong prime,
        int maxQBits,
        bool verifyWithModPow,
        out BigInteger foundQ)
    {
        foundQ = BigInteger.Zero;

        if (prime == 0) return ScanResult.RuledOut;
        if (maxQBits < 2) throw new ArgumentOutOfRangeException(nameof(maxQBits));
        if (maxQBits > MaxSupportedQBits) return ScanResult.Candidate;

        int pBits = checked((int)prime);
        int maxBlocks = (maxQBits + 3) / 4;

        var qGen = new MersenneCombinedDivisorScannerFullStreamExtended.QNibbleGenerator16(prime);

        // Block-ordered exploration (BFS by block index).
        var openByBlock = new Dictionary<int, Queue<State>>(capacity: 4096);
        var termByBlock = new Dictionary<int, Queue<State>>(capacity: 4096);
        var activeBlocks = new SortedSet<int>();

        void Enqueue(State s)
        {
            var dict = s.KTerminated ? termByBlock : openByBlock;
            if (!dict.TryGetValue(s.Block, out var q))
            {
                q = new Queue<State>();
                dict[s.Block] = q;
            }
            q.Enqueue(s);
            activeBlocks.Add(s.Block);
        }

        bool TryDequeue(out State s)
        {
            s = default;
            if (activeBlocks.Count == 0) return false;
            int b = activeBlocks.Min;

            if (termByBlock.TryGetValue(b, out var tq) && tq.Count > 0) s = tq.Dequeue();
            else if (openByBlock.TryGetValue(b, out var oq) && oq.Count > 0) s = oq.Dequeue();
            else
            {
                activeBlocks.Remove(b);
                return false;
            }

            bool termEmpty = !termByBlock.TryGetValue(b, out var tq2) || tq2.Count == 0;
            bool openEmpty = !openByBlock.TryGetValue(b, out var oq2) || oq2.Count == 0;
            if (termEmpty && openEmpty) activeBlocks.Remove(b);

            return true;
        }

        // Cache of allowed a-nibble masks for (state signature, qNibble, carry).
        var allowedMaskCache = new Dictionary<AllowedMaskKey, ushort>(capacity: 1 << 16);

        // Reusable buffers.
        Span<byte> qNibByKn = stackalloc byte[16];
        Span<MersenneCombinedDivisorScannerFullStreamExtended.QGenState16> nextByKn = stackalloc MersenneCombinedDivisorScannerFullStreamExtended.QGenState16[16];
        Span<int> countsByQn = stackalloc int[16];

        var init = State.CreateInitial(MersenneCombinedDivisorScannerFullStreamExtended.QNibbleGenerator16.InitialState, maxQBits);
        Enqueue(init);

        while (TryDequeue(out var st))
        {
            if (st.Block >= maxBlocks)
            {
                if (FinishScanToEndDeterministic(pBits, maxQBits, ref st) && st.Carry == 0)
                {
                    BigInteger qVal = st.QBits.ToBigInteger();
                    if (qVal > BigInteger.One)
                    {
                        foundQ = qVal;
                        if (verifyWithModPow)
                        {
                            bool ok = BigInteger.ModPow(2, prime, foundQ) == BigInteger.One;
                            if (!ok) return ScanResult.Candidate;
                        }
                        return ScanResult.FoundDivisor;
                    }
                }
                continue;
            }

            int startBit = st.Block << 2;

            // Reset buffers
            for (int i = 0; i < 16; i++) { qNibByKn[i] = 0xFF; nextByKn[i] = default; countsByQn[i] = 0; }

            if (st.KTerminated)
            {
                byte kn = 0;
                qGen.ComputeNext(st.QState, kn, out byte qn, out var ns);

                // Advance generator state and post-termination flushing
                st.QState = ns;
                if (st.KFlushRemaining > 0) st.KFlushRemaining--;

                ApplyQNibble(ref st, startBit, qn, maxQBits);

                if (qn == 0) st.PostTermZeroQNibbles++;
                else st.PostTermZeroQNibbles = 0;

                int requiredZeroRun = qGen.KWinLen + 1;
                if (st.KFlushRemaining == 0 && st.QState.Carry16 == 0 && st.PostTermZeroQNibbles >= requiredZeroRun)
                {
                    // Freeze q and finish
                    var done = st.CloneShallow();
                    done.Block = maxBlocks;
                    Enqueue(done);
                    continue;
                }

                qNibByKn[kn] = qn;
                nextByKn[kn] = ns;
                countsByQn[qn] = 1;
            }
            else
            {
                qGen.ComputeAllNext(st.QState, qNibByKn, nextByKn, countsByQn);
            }

            for (int qnVal = 0; qnVal < 16; qnVal++)
            {
                if (countsByQn[qnVal] == 0) continue;
                byte qn = (byte)qnVal;

                if (st.Block == 0 && !IsAllowedQ0Nibble(qn))
                    continue;

                // Compute allowed a-nibbles deterministically for this (state, qn).
                int sigEndBit = Math.Min(startBit + 3, pBits + maxQBits);
                ulong qSig = st.QBits.Signature64WithNibbleUpTo(sigEndBit, startBit, qn, maxQBits);
                ulong aSig = st.AWindow.Signature64();

                ushort allowedMask;
                int carryForCache = st.Carry;
                if ((uint)carryForCache <= (uint)CarryCacheMax)
                {
                    var key = new AllowedMaskKey(pBits, maxQBits, st.Block, carryForCache, qn, qSig, aSig);
                    if (!allowedMaskCache.TryGetValue(key, out allowedMask))
                    {
                        allowedMask = ComputeAllowedANibbleMaskDeterministic(pBits, maxQBits, st, startBit, qn);
                        allowedMaskCache[key] = allowedMask;
                    }
                }
                else
                {
                    // Large carry: compute without caching.
                    allowedMask = ComputeAllowedANibbleMaskDeterministic(pBits, maxQBits, st, startBit, qn);
                }

                if (allowedMask == 0)
                    continue;

                // Expand k-nibbles mapping to this q-nibble.
                for (int kn = 0; kn < 16; kn++)
                {
                    if (qNibByKn[kn] != qn) continue;
                    if (st.Block == 0 && kn == 0) continue; // keep k>=1 (so q>1)

                    var child = st.CloneShallow();
                    child.QState = nextByKn[kn];

                    bool canTerminateHere = !st.KTerminated && kn == 0 && st.Block > 0;

                    ApplyQNibble(ref child, startBit, qn, maxQBits);

                    // normal branch
                    ExpandANibbles(pBits, maxQBits, maxQBits, ref child, startBit, allowedMask, Enqueue);

                    // terminated branch
                    if (canTerminateHere)
                    {
                        var termChild = st.CloneShallow();
                        termChild.QState = nextByKn[kn];
                        termChild.KTerminated = true;
                        termChild.KFlushRemaining = (byte)qGen.KWinLen;
                        termChild.PostTermZeroQNibbles = 0;

                        ApplyQNibble(ref termChild, startBit, qn, maxQBits);

                        ExpandANibbles(pBits, maxQBits, maxQBits, ref termChild, startBit, allowedMask, Enqueue);
                    }
                }
            }
        }

        return ScanResult.RuledOut;
    }

    private static void ExpandANibbles(
        int pBits,
        int maxQBits,
        int windowBits,
        ref State baseState,
        int startBit,
        ushort allowedMask,
        Action<State> push)
    {
        // Iterate allowed a-nibbles.
        for (int aNib = 0; aNib < 16; aNib++)
        {
            if ((allowedMask & (1 << aNib)) == 0) continue;
            if (startBit == 0 && (aNib & 1) == 0) continue; // enforce a0=1

            var st = baseState.CloneShallow();
            ApplyANibblePlanned(ref st, startBit, (byte)aNib);

            // Deterministically step the 4 columns (planned bits remove branching).
            if (!StepColumnsForBlockDeterministic(pBits, maxQBits, ref st, startBit))
                continue;

            st.Block = st.QState.NibbleIndex;
            push(st);
        }
    }

    private static ushort ComputeAllowedANibbleMaskDeterministic(int pBits, int maxQBits, State st, int startBit, byte qNibble)
    {
        // Ensure qNibble is applied to the state for reading q bits within this block.
        var baseState = st.CloneShallow();
        ApplyQNibble(ref baseState, startBit, qNibble, maxQBits);

        ushort mask = 0;
        for (int aNib = 0; aNib < 16; aNib++)
        {
            if (startBit == 0 && (aNib & 1) == 0) continue;

            var tmp = baseState.CloneShallow();
            ApplyANibblePlanned(ref tmp, startBit, (byte)aNib);

            // Only test the 4 columns deterministically.
            if (StepColumnsForBlockDeterministic(pBits, maxQBits, ref tmp, startBit))
                mask |= (ushort)(1 << aNib);
        }

        return mask;
    }

    private static bool StepColumnsForBlockDeterministic(int pBits, int maxQBits, ref State st, int startBit)
    {
        int endBit = Math.Min(startBit + 3, pBits + maxQBits);
        while (st.BitIndex <= endBit)
        {
            if (!StepSingleColumnDeterministic(pBits, maxQBits, ref st))
                return false;
            st.BitIndex++;
        }
        return true;
    }

    private static bool FinishScanToEndDeterministic(int pBits, int maxQBits, ref State st)
    {
        int end = pBits + 1;
        while (st.BitIndex <= end)
        {
            int startBit = (st.BitIndex >> 2) << 2;
            if (!StepColumnsForBlockDeterministic(pBits, maxQBits, ref st, startBit))
                return false;
        }
        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool StepSingleColumnDeterministic(int pBits, int maxQBits, ref State st)
    {
        int j = st.BitIndex;
        int requiredBit = (j < pBits) ? 1 : 0;

        int sumOther = st.Carry;

        int maxT = Math.Min(j, maxQBits - 1);

        // Contributions from q_t=1 for t>0: add a_{j-t}
        foreach (int t in st.QBits.EnumerateSetBitsUpTo(maxT))
        {
            if (t == 0) continue;
            int aIndex = j - t;
            if (aIndex < 0) continue;
            if (st.AWindow.GetBitRelative(aIndex) != 0)
                sumOther++;
        }

        // Deterministic a_j: must be planned or already committed; if missing, treat as unknown => not deterministic
        if (!st.AWindow.TryGetKnownBit(j, out int aJ))
            return false;

        int sum = sumOther + aJ;
        if ((sum & 1) != requiredBit)
            return false;

        int newCarry = (requiredBit == 1) ? ((sum - 1) >> 1) : (sum >> 1);
        st.Carry = newCarry;

        // Commit bit into window (PushBit also clears planned for j).
        st.AWindow.PushBit(j, aJ);
        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplyQNibble(ref State st, int startBit, byte qNibble, int maxQBits)
    {
        for (int b = 0; b < 4; b++)
        {
            int bitPos = startBit + b;
            if (bitPos >= maxQBits) continue;
            int bit = (qNibble >> b) & 1;
            if (bit != 0) st.QBits.Set(bitPos);
        }
        st.QBits.Set(0); // enforce q0=1
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplyANibblePlanned(ref State st, int startBit, byte aNibble)
    {
        for (int b = 0; b < 4; b++)
        {
            int bitPos = startBit + b;
            int bit = (aNibble >> b) & 1;
            st.AWindow.SetPlanned(bitPos, bit);
        }
    }
}
