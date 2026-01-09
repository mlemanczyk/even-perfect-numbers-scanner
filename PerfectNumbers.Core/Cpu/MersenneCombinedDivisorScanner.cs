using System;
using System.Buffers;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core.Cpu;

/// <summary>
/// Deterministic streaming (column-wise) reconstruction scan for M_p = 2^p - 1,
/// exploring divisors in the family q = (2p)*k + 1 without materializing full q during search.
/// 
/// IMPORTANT:
/// - This scanner is exact for the explored (q,k) search tree up to maxQBits.
/// - However, exploring "all possible" k for very large maxQBits is combinatorially explosive.
///   The intent is to use strong pruning to rule out branches early.
/// - For practical use, keep maxQBits in a regime you can actually explore (hundreds or a few thousands),
///   and rely on pruning/caches.
/// </summary>
public static class MersenneCombinedDivisorScannerFullStreamExtended
{
	// ===== DEBUG SETTINGS =====
	private static readonly bool DebugEnabled = true;
	private static readonly BigInteger DebugTargetQ =
		BigInteger.Parse("1209708008767");
	private const int DebugMaxBlocks = 20; // wystarczy (q ma ~11 bloków)

	private static bool DebugPrefixMatches(BigInteger qNow, int producedNibbles)
	{
		int producedBits = producedNibbles * 4;
		// target ma ~41 bitów; powyżej nie ma sensu maskować dalej
		if (producedBits >= 128) producedBits = 128; // bezpiecznie, debug tylko

		BigInteger mask = (BigInteger.One << producedBits) - BigInteger.One;
		return (qNow & mask) == (DebugTargetQ & mask);
	}

	private const int MaxColumnChoices = 2;


	public enum ScanResult
	{
		RuledOut = 0,
		Candidate = 1,
		FoundDivisor = MaxColumnChoices
	}

	// Hard safety ceiling for in-memory bitsets/windows in this implementation.
	// You can raise it, but exploration cost grows extremely fast with maxQBits.
	public const int MaxSupportedQBits = 8192;

	// Block-0 filter: q is odd and q ≡ 1 or 7 (mod 8).
	// NOTE: Do NOT apply any “not divisible by 5” filter based on low bits; it is not valid in base-2 local form.
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static bool IsAllowedQ0Nibble(byte q0)
	{
		// odd
		if ((q0 & 0x1) == 0) return false;
		// q mod 8 in {1,7}
		int mod8 = q0 & 0x7;
		return mod8 == 0x1 || mod8 == 0x7;
	}

	/// <summary>
	/// Deterministic streaming reconstruction scan for M_p = 2^p - 1.
	///
	/// The scan explores q in the family q = (2p)*k + 1, where k is an unbounded nonnegative integer,
	/// but truncates q to maxQBits. The scan maintains a rolling window of a-bits of size maxQBits,
	/// sufficient to evaluate the convolution exactly for all columns j up to p+maxQBits.
	///
	/// Return values:
	/// - FoundDivisor: found a q within maxQBits such that the reconstructed a satisfies a*q = 2^p - 1 (in all columns).
	/// - RuledOut: no such q exists within the explored tree and limits.
	/// - Candidate: scan did not find a divisor within the explored q-bit budget (maxQBits),
	///              or maxQBits exceeds MaxSupportedQBits.
	///
	/// Notes:
	/// - This is an exact column-wise decision procedure for each explored branch, but exploration is exponential in maxQBits.
	/// - Optional ModPow verification is performed only when FoundDivisor is reached.
	/// </summary>
	public static ScanResult TryFindDivisorByStreamingScan(
		ulong prime,
		int maxQBits,
		bool verifyWithModPow,
		out BigInteger foundQ)
	{
		foundQ = BigInteger.Zero;

		if (prime == 0) return ScanResult.RuledOut;
		ArgumentOutOfRangeException.ThrowIfLessThan(maxQBits, MaxColumnChoices);
		if (maxQBits > MaxSupportedQBits) return ScanResult.Candidate;

		int pBits = checked((int)prime);
		int maxBlocks = (maxQBits + 3) / 4;

		var qGen = new QNibbleGenerator16(prime);

		// Layered (block-ordered) exploration: always exhaust smaller block indices first.
		// This is important for correctness of conservative block-level pruning and for predictable progress.
		var openByBlock = new Dictionary<int, Queue<State>>(capacity: 4096);
		var termByBlock = new Dictionary<int, Queue<State>>(capacity: 4096);
		var activeBlocks = new SortedSet<int>();

		// Per-block conservative global pruning stats (computed over the set of *actually reachable* reduced states at that block).
		var blockPrune = new Dictionary<int, BlockPruneStats>(capacity: 1024);

		// Per-state allowed-a cache (must remain state-sensitive; see AllowedMaskKey).
		var allowedMaskCache = new Dictionary<AllowedMaskKey, ushort>(capacity: 1 << 16);

		void EnqueueState(State s)
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

		bool TryDequeueState(out State s)
		{
			s = default;
			if (activeBlocks.Count == 0) return false;
			int b = activeBlocks.Min;

			if (termByBlock.TryGetValue(b, out var tq) && tq.Count > 0)
			{
				s = tq.Dequeue();
			}
			else if (openByBlock.TryGetValue(b, out var oq) && oq.Count > 0)
			{
				s = oq.Dequeue();
			}
			else
			{
				// Should not happen, but keep structure consistent.
				activeBlocks.Remove(b);
				return false;
			}

			// Cleanup active block if both queues are empty.
			bool termEmpty = !termByBlock.TryGetValue(b, out var tq2) || tq2.Count == 0;
			bool openEmpty = !openByBlock.TryGetValue(b, out var oq2) || oq2.Count == 0;
			if (termEmpty && openEmpty) activeBlocks.Remove(b);

			return true;
		}

		var init = State.CreateInitial(QNibbleGenerator16.InitialState, maxQBits);
		EnqueueState(init);

		Span<int> counts = stackalloc int[16];
		Span<byte> qNib = stackalloc byte[16];
		Span<QGenState16> nextState = stackalloc QGenState16[16];
		Span<byte> aOrder = stackalloc byte[16]; // reusable buffer for nibble ordering (avoid stackalloc in loops)
		int lastLoggedBlock = -1;

		State st;
		while (TryDequeueState(out st))
		{

			// Intentionally no internal max-nodes limit.

			if (DebugEnabled && st.Block != lastLoggedBlock)
			{
				lastLoggedBlock = st.Block;
				Console.WriteLine($"[PROGRESS] p={prime}, reached block={st.Block}");
			}

			// If we have finished producing all q nibbles (to maxQBits), finish scanning remaining columns.
			if (st.Block >= maxBlocks)
			{
				if (FinishScanToEnd(pBits, maxQBits, ref st) && st.Carry == 0)
				{
					// Found a fully consistent reconstruction for this branch.
					BigInteger qVal = st.QBits.ToBigInteger();
					if (qVal > BigInteger.One)
					{
						foundQ = qVal;
						Console.WriteLine($"[stream-scan] Found divisor candidate q(bits<={maxQBits}) for p={prime}");
						if (verifyWithModPow)
						{
							bool ok = BigInteger.ModPow(MaxColumnChoices, prime, foundQ) == BigInteger.One;
							Console.WriteLine($"[stream-scan] ModPow verification: {(ok ? "OK" : "FAILED")}");
						}

						return ScanResult.FoundDivisor;
					}
				}

				continue;
			}

			int startBit = st.Block << 2;

			// Enumerate possible k-nibbles for this q-nibble; group by qNibble to avoid recomputation
			// of allowed a choices (which depend on qNibble and carry).

			// Reset tables for this state.
			qNib.Clear();
			nextState.Clear();
			counts.Clear();

			// byte forcedKn;
			// int ni = st.QState.NibbleIndex;

			// if (ni == 0) forcedKn = 0xF;
			// else if (ni == 1) forcedKn = 0x1;
			// else if (ni == 2) forcedKn = 0x1;
			// else if (ni == 3) forcedKn = 0x1;
			// else forcedKn = 0x0;

			// for (byte kn = 0; kn <= 15; kn++)
			// {
			// 	if (kn != forcedKn) continue;

			// 	for (int i = 0; i < 16; i++) qNib[i] = 0xFF;
			// 	for (int i = 0; i < 16; i++) nextState[i] = default; // opcjonalnie
			// 	qGen.ComputeNext(st.QState, forcedKn, out byte qn, out QGenState16 ns);
			// 	qNib[forcedKn] = qn;
			// 	nextState[forcedKn] = ns;
			// 	counts[qn] = 1;
			// }

			// Sentinel init: mark unused kn entries as invalid (0xFF) to avoid accidental matches when qn==0.
			for (int i = 0; i < 16; i++) qNib[i] = 0xFF;
			// nextState default is fine; we will only read it when qNib[kn] == qn (so it was written).
			for (int i = 0; i < 16; i++) nextState[i] = default;

			// Jeśli k już zakończone, jedyny dozwolony kn=0
			if (st.KTerminated)
			{
				byte kn = 0;
				qGen.ComputeNext(st.QState, kn, out byte qn, out QGenState16 ns);

				// Aktualizuj stan generatora
				st.QState = ns;

				// Flush okna k: po terminacji trzeba wykonać dokładnie KWinLen kroków kn=0,
				// żeby wszystkie historyczne nibble k przestały wpływać na q.
				if (st.KFlushRemaining > 0)
					st.KFlushRemaining--;

				// Dopisz q nibble do bitsetu q
				ApplyQNibble(ref st, startBit, qn, maxQBits);

				// After k termination we keep streaming q with kn=0.
				// We can only declare "end of q" once the k-window is flushed AND we observe a stable run of zero q-nibbles.
				// This avoids premature freezing when a later carry/window interaction could still produce a non-zero nibble.
				if (qn == 0) st.PostTermZeroQNibbles++;
				else st.PostTermZeroQNibbles = 0;

				int requiredZeroRun = qGen.KWinLen + 1;
				if (st.KFlushRemaining == 0 && st.QState.Carry16 == 0 && st.PostTermZeroQNibbles >= requiredZeroRun)
				{
					Console.WriteLine($"[QFROZEN] p={prime} at block={st.Block} nib={st.QState.NibbleIndex} carry16={st.QState.Carry16} zeroRun={st.PostTermZeroQNibbles}");

					// Jump to final phase (FinishScanToEnd) without materializing full q.
					var done = st.CloneShallow();
					done.Block = maxBlocks;
					EnqueueState(done);
					continue;
				}

				qNib[kn] = qn;
				nextState[kn] = ns;
				counts[qn] = 1;
			}
			else
			{
				// Enumerate all possible k-nibbles at this position (optimized: compute base once per state).
				qGen.ComputeAllNext(st.QState, qNib, nextState, counts);

				// Dodatkowa gałąź: “zakończ k TERAZ” (czyli od tego miejsca kn=0 zawsze).
				// Realizujemy to poprzez dopuszczenie kn=0, ale z flagą KTerminated w dziecku.
				// Nie musisz robić dodatkowego liczenia counts; obsłużymy to przy tworzeniu child.
			}


			// Conservative per-block pruning:
			// Track which q-nibbles are impossible for *all reachable reduced states* at this block.
			// This is safe because we explore blocks in increasing order; once we move past a block, no new states for it appear.
			ulong qPrefixSig = st.QBits.Signature64UpTo(startBit - 1);
			ulong aSigNow = st.AWindow.Signature64();
			var reducedKey = new ReducedStateKey(pBits, maxQBits, st.Block, st.Carry, qPrefixSig, aSigNow);

			if (!blockPrune.TryGetValue(st.Block, out var pruneStats))
			{
				pruneStats = new BlockPruneStats();
				blockPrune[st.Block] = pruneStats;
			}
			pruneStats.RegisterState(reducedKey);
			for (int qnVal = 0; qnVal < 16; qnVal++)
			{
				if (counts[qnVal] == 0) continue;
				byte qn = (byte)qnVal;

				if (st.Block == 0 && !IsAllowedQ0Nibble(qn))
					continue;

				// Global (block-level) forbidden q-nibble mask, conservatively derived from all reachable reduced states.
				if ((pruneStats.GlobalForbiddenMask & (1 << qn)) != 0)
					continue;

				// Early contradiction pruning (interval-based carry propagation), inspired by BitContradictionSolver.
				// Important: bits of 'a' outside the window are treated as Unknown (never as 0), so this pruning cannot
				// introduce false negatives.
				if (QuickRejectQNibbleByBounds(pBits, maxQBits, in st, startBit, qn))
				{
					pruneStats.MarkRejected(reducedKey, qn);
					continue;
				}

				// Additional early pruning: derive a conservative candidate mask for the 4-bit a-nibble at this block
				// by checking (using the same bounds logic) whether each a_j in this block can be 0/1.
				// If a bit is forced (only one value feasible), we restrict the candidate a-nibble mask.
				// We also derive a conservative *preference* mask/value for branch ordering (no pruning).
				// Bits outside the a-window are treated as Unknown, so this cannot introduce false negatives.
				var boundsInfo = ComputeCandidateANibbleMaskAndPreferencesByBounds(pBits, maxQBits, in st, startBit, qn);
				ushort preMask = boundsInfo.CandidateMask;
				if (preMask == 0)
				{
					pruneStats.MarkRejected(reducedKey, qn);
					continue;
				}

				// Precompute which a-nibbles are compatible with this q-nibble and current carry,
				// using LUT for delta contribution plus exact base via the rolling a-window and current q-bitset.
				int sigEndBit = Math.Min(startBit + 3, pBits + maxQBits);
				ulong qSig = st.QBits.Signature64WithNibbleUpTo(sigEndBit, startBit, qn, maxQBits);
				ulong aSig = st.AWindow.Signature64();
				var key = new AllowedMaskKey(pBits, maxQBits, st.Block, st.Carry, qn, qSig, aSig, preMask);

				if (!allowedMaskCache.TryGetValue(key, out ushort allowedAMask))
				{
					allowedAMask = ComputeAllowedANibbleMaskFiltered(pBits, maxQBits, ref st, startBit, qn, preMask);
					allowedMaskCache[key] = allowedAMask;
				}

				if (allowedAMask == 0 && DebugEnabled && st.QState.NibbleIndex <= DebugMaxBlocks)
				{
					var tmp = st.CloneShallow();
					ApplyQNibble(ref tmp, startBit, qn, maxQBits);
					BigInteger qTmp = tmp.QBits.ToBigInteger();

					int produced = st.QState.NibbleIndex + 1;
					if (DebugPrefixMatches(qTmp, produced))
						Console.WriteLine($"[DEBUG CUT-A-PREFIX] block={produced} qLow={qTmp} qNib=0x{qn:X}");
				}

				if (allowedAMask == 0)
				{
					pruneStats.MarkRejected(reducedKey, qn);
					continue;
				}

				// Expand all k-nibbles mapping to this q-nibble.
				// We will apply branch ordering to a-nibbles using bounds-derived preferences (no pruning).
				byte prefMask = boundsInfo.PrefMask;
				byte prefValue = boundsInfo.PrefValue;
				for (int kn = 0; kn < 16; kn++)
				{
					if (qNib[kn] != qn) continue;
					if (st.Block == 0 && kn == 0) continue;

					// if (DebugEnabled && kn == 0 && qNib[kn] == qn)
					// {
					// 	Console.WriteLine($"[KN0] p={prime} block={st.Block} qn=0x{qn:X} canTerminateHere={!st.KTerminated}");
					// }

					var child = st.CloneShallow(); // share arrays via copy-on-write where safe (implemented as cloning bitsets)
					child.QState = nextState[kn];

					// Jeśli nie jesteśmy w trybie terminated i wybraliśmy kn==0,
					// to rozgałęź na dwa warianty:
					// - zwykły: k nadal może mieć wyższe nible,
					// - terminated: uznajemy, że k kończy się tutaj.
					bool canTerminateHere = !st.KTerminated && kn == 0 && st.Block > 0;
					child.KTerminated = st.KTerminated; // default

					// wariant 1: zwykły (nie terminujemy)
					ApplyQNibble(ref child, startBit, qn, maxQBits);

					// if (DebugEnabled && child.QState.NibbleIndex <= DebugMaxBlocks)
					// {
					// 	BigInteger qNow = child.QBits.ToBigInteger();
					// 	if (DebugPrefixMatches(qNow, child.QState.NibbleIndex))
					// 	{
					// 		Console.WriteLine($"[DEBUG HIT-Q-ONLY-PREFIX] block={child.QState.NibbleIndex} qLow={qNow}");
					// 	}
					// }

					// ===== wariant 1: normalnie kontynuujemy (KTerminated=false) =====
					int aCount = BuildOrderedNibbleList(allowedAMask, prefMask, prefValue, aOrder);
					for (int ai = 0; ai < aCount; ai++)
					{
						int aNib = aOrder[ai];

						var grand = child.CloneShallow();
						ApplyANibble(ref grand, startBit, (byte)aNib);

						foreach (var branched in StepColumnsForBlockMulti(pBits, maxQBits, grand, startBit))
						{
							var pushState = branched;
							pushState.Block = pushState.QState.NibbleIndex;
							EnqueueState(pushState);
						}
					}

					// wariant 2: terminujemy k (tylko jeśli kn==0 i jeszcze nie terminated)
					if (canTerminateHere)
					{
						var termChild = st.CloneShallow();
						termChild.QState = nextState[kn];
						termChild.KFlushRemaining = (byte)qGen.KWinLen;
						termChild.PostTermZeroQNibbles = 0;
						termChild.KTerminated = true;

						// Console.WriteLine($"[KTERM] p={prime} creating terminated branch at block={st.Block}, nib={st.QState.NibbleIndex}");

						ApplyQNibble(ref termChild, startBit, qn, maxQBits);

						// kontynuujemy stan z terminacją k
						int aCount2 = BuildOrderedNibbleList(allowedAMask, prefMask, prefValue, aOrder);
						for (int ai2 = 0; ai2 < aCount2; ai2++)
						{
							int aNib = aOrder[ai2];

							var grand = termChild.CloneShallow();
							ApplyANibble(ref grand, startBit, (byte)aNib);

							foreach (var branched in StepColumnsForBlockMulti(pBits, maxQBits, grand, startBit))
							{
								var pushState = branched;
								pushState.Block = pushState.QState.NibbleIndex;
								EnqueueState(pushState);
							}
						}
					}
				}
			}
		}

		return ScanResult.RuledOut;
	}


	private static ushort ComputeAllowedANibbleMask(int pBits, int maxQBits, ref State st, int startBit, byte qNibble)
	{
		return ComputeAllowedANibbleMaskFiltered(pBits, maxQBits, ref st, startBit, qNibble, 0xFFFF);
	}

	private static ushort ComputeAllowedANibbleMaskFiltered(int pBits, int maxQBits, ref State st, int startBit, byte qNibble, ushort candidateMask)
	{
		// Correctness-first implementation:
		// - never mutate the caller state
		// - pre-apply qNibble once to improve performance (it is the same for all aNib candidates)

		ushort mask = 0;

		var baseState = st.CloneShallow();
		ApplyQNibble(ref baseState, startBit, qNibble, maxQBits);

		for (int aNib = 0; aNib < 16; aNib++)
		{
			if ((candidateMask & (1 << aNib)) == 0) continue;
			// In block0, enforce a0 bit0 = 1 (since a0=1).
			if (startBit == 0 && (aNib & 1) == 0) continue;

			var tmp = baseState.CloneShallow();
			ApplyANibble(ref tmp, startBit, (byte)aNib);

			bool ok = false;
			foreach (var _ in StepColumnsForBlockMulti(pBits, maxQBits, tmp, startBit))
			{
				ok = true;
				break;
			}
			if (ok) mask |= (ushort)(1 << aNib);
		}

		return mask;
	}

	// ------------------------------------------------------------------------
	// Early a-nibble candidate mask derivation (forced bits) via bounds
	// ------------------------------------------------------------------------
	// This uses the same conservative parity/bounds logic as QuickRejectQNibbleByBounds, but instead of only
	// rejecting qNibble, it attempts to determine whether each a_j in this 4-column block can be 0/1.
	// If exactly one of {0,1} is feasible for a_j (under the conservative bounds model), that bit is treated as forced
	// and used to restrict the candidate a-nibble space.
	//
	// IMPORTANT: This MUST be conservative. It may fail to detect some forced bits (false negatives),
	// but it must never rule out a real solution (no false positives on forcing). Our feasibility checks use
	// an over-approximate interval [minSum,maxSum] and therefore only declare "impossible" when parity cannot
	// be achieved anywhere in that interval.
	private readonly struct BoundsANibbleInfo
	{
		public readonly ushort CandidateMask;
		public readonly byte PrefMask;  // bits within the 4-bit nibble for which we have a preference
		public readonly byte PrefValue; // preferred bit values for those positions

		public BoundsANibbleInfo(ushort candidateMask, byte prefMask, byte prefValue)
		{
			CandidateMask = candidateMask;
			PrefMask = prefMask;
			PrefValue = prefValue;
		}
	}

	private static BoundsANibbleInfo ComputeCandidateANibbleMaskAndPreferencesByBounds(int pBits, int maxQBits, in State st, int startBit, byte qNibble)
	{
		// Start with all 16 nibbles allowed.
		ushort mask = 0xFFFF;
		byte prefMask = 0;
		byte prefValue = 0;

		long carryMin = st.Carry;
		long carryMax = st.Carry;

		for (int j = startBit; j <= startBit + 3; j++)
		{
			if (j > pBits + maxQBits) break;
			int requiredBit = (j < pBits) ? 1 : 0;

			// Compute forced/possible contributions from offsets t>0 (excluding a_j itself).
			long forcedOther = 0;
			long possibleOther = 0;

			int maxT = Math.Min(j, maxQBits - 1);
			for (int t = 1; t <= maxT; t++)
			{
				if (GetQBitWithNibble(in st, startBit, qNibble, t, maxQBits) == 0) continue;
				int aIndex = j - t;
				if (aIndex < 0) continue;

				if (st.AWindow.TryGetKnownBit(aIndex, out int abit))
				{
					if (abit != 0)
					{
						forcedOther++;
						possibleOther++;
					}
				}
				else
				{
					possibleOther++;
				}
			}

			bool ajKnown = st.AWindow.TryGetKnownBit(j, out int ajBit);
			bool can0, can1;
			if (ajKnown)
			{
				can0 = ajBit == 0 && IsParityReachableForCarryRange(carryMin, carryMax, forcedOther, possibleOther, 0, requiredBit);
				can1 = ajBit == 1 && IsParityReachableForCarryRange(carryMin, carryMax, forcedOther, possibleOther, 1, requiredBit);
			}
			else
			{
				can0 = IsParityReachableForCarryRange(carryMin, carryMax, forcedOther, possibleOther, 0, requiredBit);
				can1 = IsParityReachableForCarryRange(carryMin, carryMax, forcedOther, possibleOther, 1, requiredBit);
			}

			if (!can0 && !can1)
				return new BoundsANibbleInfo(0, 0, 0);

			// If only one value is feasible (under our conservative model), restrict the nibble mask.
			int bitInNibble = j - startBit;
			if (bitInNibble >= 0 && bitInNibble < 4)
			{
				if (!can0 && can1)
				{
					mask &= (ushort)ForceNibbleBit(bitInNibble, 1);
					prefMask |= (byte)(1 << bitInNibble);
					prefValue |= (byte)(1 << bitInNibble);
				}
				else if (can0 && !can1)
				{
					mask &= (ushort)ForceNibbleBit(bitInNibble, 0);
					prefMask |= (byte)(1 << bitInNibble);
					prefValue &= (byte)~(1 << bitInNibble);
				}
				else
				{
					// Both possible: derive a *preference* for branch ordering.
					// Prefer the value that yields a narrower next carry range under the same conservative model.
					long n0Min, n0Max, n1Min, n1Max;
					bool ok0 = TryPropagateCarryBounds(carryMin, carryMax, forcedOther + 0, possibleOther + 0, requiredBit, out n0Min, out n0Max);
					bool ok1 = TryPropagateCarryBounds(carryMin, carryMax, forcedOther + 1, possibleOther + 1, requiredBit, out n1Min, out n1Max);
					// ok0/ok1 should match can0/can1, but keep defensive.
					if (ok0 && ok1)
					{
						long w0 = n0Max - n0Min;
						long w1 = n1Max - n1Min;
						int prefer = (w1 < w0) ? 1 : (w0 < w1 ? 0 : 0);
						prefMask |= (byte)(1 << bitInNibble);
						if (prefer == 1) prefValue |= (byte)(1 << bitInNibble);
						else prefValue &= (byte)~(1 << bitInNibble);
					}
				}
			}

			// Update carry range for next column using *unknown* a_j if not known and not forced.
			// For safety, we propagate using the full [min,max] interval that covers all feasible choices of a_j.
			long forced = forcedOther;
			long possible = possibleOther;
			if (ajKnown)
			{
				if (ajBit != 0) { forced++; possible++; }
			}
			else
			{
				// if both 0 and 1 are feasible, treat as unknown -> possible++
				// if forced, we can reflect it in forced/possible.
				if (!can0 && can1) { forced++; possible++; }
				else if (can0 && !can1) { /* forced 0 -> nothing */ }
				else { possible++; }
			}

			if (!TryPropagateCarryBounds(carryMin, carryMax, forced, possible, requiredBit, out long nextMin, out long nextMax))
				return new BoundsANibbleInfo(0, 0, 0);
			carryMin = nextMin;
			carryMax = nextMax;
		}

		return new BoundsANibbleInfo(mask, prefMask, prefValue);
	}

	private static ushort ComputeCandidateANibbleMaskByBounds(int pBits, int maxQBits, in State st, int startBit, byte qNibble)
		=> ComputeCandidateANibbleMaskAndPreferencesByBounds(pBits, maxQBits, in st, startBit, qNibble).CandidateMask;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static bool IsParityReachableForCarryRange(long carryMin, long carryMax, long forcedOther, long possibleOther, int aJ, int requiredBit)
	{
		int parity = requiredBit & 1;
		long minSum = carryMin + forcedOther + aJ;
		long maxSum = carryMax + possibleOther + aJ;
		long minAligned = AlignUpToParity(minSum, parity);
		long maxAligned = AlignDownToParity(maxSum, parity);
		return minAligned <= maxAligned;
	}

	// Returns a 16-bit mask allowing only those a-nibbles whose bit 'b' equals requiredValue.
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static int ForceNibbleBit(int b, int requiredValue)
	{
		int mask = 0;
		for (int n = 0; n < 16; n++)
		{
			if (((n >> b) & 1) == requiredValue)
				mask |= 1 << n;
		}
		return mask;
	}

	// Build an ordered list of a-nibbles from a bitmask, using bounds-derived preferences for branch ordering.
	// This never prunes: it only reorders exploration to reach contradictions / valid branches earlier.
	//
	// Scoring: prefer nibbles matching preferred bit-values on positions indicated by prefMask.
	private static int BuildOrderedNibbleList(ushort allowedMask, byte prefMask, byte prefValue, Span<byte> outList)
	{
		int count = 0;
		for (int n = 0; n < 16; n++)
		{
			if ((allowedMask & (1 << n)) == 0) continue;
			outList[count++] = (byte)n;
		}

		if (count <= 1 || prefMask == 0)
			return count;

		// Small selection sort (count <= 16).
		for (int i = 0; i < count - 1; i++)
		{
			int best = i;
			int bestScore = ScoreNibble(outList[i], prefMask, prefValue);
			for (int j = i + 1; j < count; j++)
			{
				int s = ScoreNibble(outList[j], prefMask, prefValue);
				if (s > bestScore)
				{
					best = j;
					bestScore = s;
				}
			}
			if (best != i)
			{
				(byte a, byte b) = (outList[i], outList[best]);
				outList[i] = b;
				outList[best] = a;
			}
		}

		return count;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static int ScoreNibble(byte nib, byte prefMask, byte prefValue)
	{
		uint matches = (uint)(~(nib ^ prefValue) & prefMask) & 0xFu;
		return (int)BitOperations.PopCount(matches);
	}

	// ------------------------------------------------------------------------
	// Early contradiction pruning (interval-based carry propagation)
	// ------------------------------------------------------------------------
	// This is a conservative pre-filter inspired by BitContradictionSolver.TryPropagateCarry.
	// We do NOT commit any new a-bits here. We only prove that a given (state, qNibble) cannot
	// satisfy the required parity constraints in any of the next 4 columns, even if all currently
	// unknown a-bits were chosen adversarially.
	//
	// Key correctness constraint (per user request): a-bits outside the window are treated as UNKNOWN,
	// never as 0. Therefore this pruning cannot eliminate a valid solution branch.
	private static bool QuickRejectQNibbleByBounds(int pBits, int maxQBits, in State st, int startBit, byte qNibble)
	{
		long carryMin = st.Carry;
		long carryMax = st.Carry;

		for (int j = startBit; j <= startBit + 3; j++)
		{
			if (j > pBits + maxQBits) break;
			int requiredBit = (j < pBits) ? 1 : 0;

			// Compute forced/possible ones contributed to column j excluding carry.
			// We model: sum = carry + a_j + Σ_{t>0, q_t=1} a_{j-t}
			long forced = 0;
			long possible = 0;

			int maxT = Math.Min(j, maxQBits - 1);
			for (int t = 1; t <= maxT; t++)
			{
				if (GetQBitWithNibble(in st, startBit, qNibble, t, maxQBits) == 0) continue;
				int aIndex = j - t;
				if (aIndex < 0) continue;

				if (st.AWindow.TryGetKnownBit(aIndex, out int abit))
				{
					if (abit != 0)
					{
						forced++;
						possible++;
					}
				}
				else
				{
					// Unknown a-bit outside window or not yet assigned.
					possible++;
				}
			}

			// Include a_j itself.
			if (st.AWindow.TryGetKnownBit(j, out int aJKnown))
			{
				if (aJKnown != 0)
				{
					forced++;
					possible++;
				}
			}
			else
			{
				possible++;
			}

			if (!TryPropagateCarryBounds(carryMin, carryMax, forced, possible, requiredBit, out long nextMin, out long nextMax))
				return true;

			carryMin = nextMin;
			carryMax = nextMax;
		}

		return false;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static int GetQBitWithNibble(in State st, int startBit, byte qNibble, int bitPos, int maxQBits)
	{
		if (bitPos == 0) return 1; // enforce q0=1 always
		if ((uint)bitPos >= (uint)maxQBits) return 0;

		int baseBit = st.QBits.Get(bitPos);
		if (bitPos >= startBit && bitPos < startBit + 4)
		{
			int b = bitPos - startBit;
			int nibBit = (qNibble >> b) & 1;
			return baseBit | nibBit;
		}
		return baseBit;
	}

	private static bool TryPropagateCarryBounds(long carryMin, long carryMax, long forcedOnes, long possibleOnes, int requiredBit,
		out long nextCarryMin, out long nextCarryMax)
	{
		// Equivalent in spirit to BitContradictionSolver.TryPropagateCarry, but we track a carry *range*
		// and return the union of feasible next carry values.
		int parity = requiredBit & 1;

		nextCarryMin = long.MaxValue;
		nextCarryMax = long.MinValue;

		for (long carry = carryMin; carry <= carryMax; carry++)
		{
			long minSum = carry + forcedOnes;
			long maxSum = carry + possibleOnes;

			long minAligned = AlignUpToParity(minSum, parity);
			long maxAligned = AlignDownToParity(maxSum, parity);
			if (minAligned > maxAligned) continue;

			long cMin = (minAligned - requiredBit) >> 1;
			long cMax = (maxAligned - requiredBit) >> 1;

			if (cMin < nextCarryMin) nextCarryMin = cMin;
			if (cMax > nextCarryMax) nextCarryMax = cMax;
		}

		if (nextCarryMin == long.MaxValue)
		{
			nextCarryMin = 0;
			nextCarryMax = 0;
			return false;
		}

		return true;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static long AlignUpToParity(long value, int parity)
		=> ((value & 1) == parity) ? value : value + 1;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static long AlignDownToParity(long value, int parity)
		=> ((value & 1) == parity) ? value : value - 1;

	private static bool StepColumnsForBlock(int pBits, int maxQBits, ref State st, int startBit)
	{
		int endBit = Math.Min(startBit + 3, pBits + maxQBits);
		while (st.BitIndex <= endBit)
		{
			if (!StepSingleColumn(pBits, maxQBits, ref st))
				return false;
			st.BitIndex++;
		}
		return true;
	}

	private static IEnumerable<State> StepColumnsForBlockMulti(int pBits, int maxQBits, State seed, int startBit)
	{
		int endBit = Math.Min(startBit + 3, pBits + maxQBits);

		// Lokalny work-stack: eksplorujemy tylko 4 kolumny.
		// Maksymalnie 2^4 = 16 stanów na seed (zwykle znacznie mniej).
		var work = new FixedCapacityStack<State>(capacity: 32);
		work.Push(seed);

		int[] choices = new int[MaxColumnChoices];
		while (work.Count > 0)
		{
			var st = work.Pop();

			// Jeśli przeszliśmy już cały blok, zwracamy stan jako wynik.
			if (st.BitIndex > endBit)
			{
				yield return st;
				continue;
			}

			int j = st.BitIndex;
			int requiredBit = (j < pBits) ? 1 : 0;

			int sumOther = st.Carry;

			int maxT = Math.Min(j, maxQBits - 1);

			// policz wkład z q_t=1 (t>0): add a_{j-t}
			// (bez capture ref – używamy lokalnych)
			var qBits = st.QBits;
			var aWin = st.AWindow;

			foreach (int t in qBits.EnumerateSetBitsUpTo(maxT))
			{
				if (t == 0) continue;
				int aIndex = j - t;
				if (aIndex < 0) continue;
				if (aWin.GetBitRelative(aIndex) != 0)
					sumOther++;
			}

			// planned?
			bool hasPlanned = aWin.TryGetPlanned(j, out int plannedAJ);

			// Policz, które aJ są dopuszczalne dla tej kolumny
			int choiceCount = 0;

			for (int aJ = 0; aJ <= 1; aJ++)
			{
				if (hasPlanned && aJ != plannedAJ) continue;

				int sum = sumOther + aJ;
				if ((sum & 1) != requiredBit)
					continue;

				choices[choiceCount++] = aJ;
			}

			if (choiceCount == 0)
			{
				// dead end for this local state
				continue;
			}

			// Dla każdej dopuszczalnej wartości aJ twórz stan następny.
			// Pierwszy wariant modyfikujemy "in place", kolejne klonujemy.
			ScheduleState(work, choices[0], j, requiredBit, sumOther, st);
			if (choiceCount == 2)
			{
				ScheduleState(work, choices[1], j, requiredBit, sumOther, st.CloneShallow());
			}
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	private static void ScheduleState(FixedCapacityStack<State> work, int aJ, int j, int requiredBit, int sumOther, State next)
	{
		int sum = sumOther + aJ;
		int nextCarry = (requiredBit == 1) ? ((sum - 1) >> 1) : (sum >> 1);

		next.Carry = nextCarry;
		next.AWindow.PushBit(j, aJ);
		next.BitIndex = j + 1;

		work.Push(next);
	}


	private static bool FinishScanToEnd(int pBits, int maxQBits, ref State st)
	{
		int end = pBits + 1;

		while (st.BitIndex <= end)
		{
			int startBit = (st.BitIndex >> MaxColumnChoices) << MaxColumnChoices;
			bool advanced = false;

			foreach (var next in StepColumnsForBlockMulti(pBits, maxQBits, st, startBit))
			{
				st = next;
				advanced = true;
				break; // deterministycznie wybieramy jedną spójną ścieżkę
			}

			if (!advanced)
				return false;
		}

		return true;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static bool StepSingleColumn(int pBits, int maxQBits, ref State st)
	{
		int j = st.BitIndex;
		int requiredBit = (j < pBits) ? 1 : 0;

		int sumOther = st.Carry;

		// contributions from q_t=1 for t>0: add a_{j-t}
		// iterate over set bits of q (excluding t=0)
		int maxT = Math.Min(j, maxQBits - 1);

		// NOTE: 'st' is a ref parameter; do not capture it in a lambda (CS1628).
		// Copy required members to locals.
		ABitsWindow aWindow = st.AWindow;
		int jj = j;
		int baseJ = aWindow.CurrentJ; // last committed column (typically j-1 while processing column j)

		foreach (int t in st.QBits.EnumerateSetBitsUpTo(maxT))
		{
			if (t == 0) continue;
			int aIndex = jj - t;
			if (aIndex < 0) continue;
			if (aWindow.GetBitRelative(aIndex) != 0)
				sumOther++;
		}

		// decide a_j by parity, because q0=1 is always set.
		bool anyOk = false;
		int newCarry = 0;

		bool hasPlanned = st.AWindow.TryGetPlanned(j, out int plannedAJ);
		for (int aJ = 0; aJ <= 1; aJ++)
		{
			if (hasPlanned && aJ != plannedAJ) continue;

			int sum = sumOther + aJ;
			if ((sum & 1) != requiredBit)
				continue;

			newCarry = (requiredBit == 1) ? ((sum - 1) >> 1) : (sum >> 1);
			st.AWindow.PushBit(j, aJ);
			anyOk = true;
			break;
		}

		if (!anyOk)
		{
			if (DebugEnabled && st.Block <= DebugMaxBlocks)
			{
				BigInteger qPrefix = st.QBits.ToBigInteger();
				if (qPrefix == DebugTargetQ)
				{
					Console.WriteLine(
						$"[DEBUG CUT-BIT] block={st.Block}, col={j}, sumOther={sumOther}, carryIn={st.Carry}, hasPlanned={hasPlanned}, plannedAJ={plannedAJ}"
					);
				}
			}
			return false;
		}

		st.Carry = newCarry;
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
			if (bit != 0)
				st.QBits.Set(bitPos);
		}
		st.QBits.Set(0); // enforce q0=1 always
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void ApplyANibble(ref State st, int startBit, byte aNibble)
	{
		// Apply 4 bits a_{startBit..startBit+3} into the rolling window.
		// The window is addressed by absolute j; we push bits in order as columns advance, not at nibble application time.
		// Here we store "scheduled" bits into a sparse map (only within window), so StepSingleColumn can read them.
		for (int b = 0; b < 4; b++)
		{
			int bitPos = startBit + b;
			int bit = (aNibble >> b) & 1;
			st.AWindow.SetPlanned(bitPos, bit);
		}
	}


	private readonly struct AllowedMaskKey(
		int pBits,
		int maxQBits,
		int block,
		int carry,
		byte qNibble,
		ulong qSig,
		ulong aSig,
		ushort candidateMask) : IEquatable<AllowedMaskKey>
	{
		public readonly int PBits = pBits;
		public readonly int MaxQBits = maxQBits;
		public readonly int Block = block;
		public readonly int Carry = carry;
		public readonly byte QNibble = qNibble;
		public readonly ulong QSig = qSig;
		public readonly ulong ASig = aSig;
		public readonly ushort CandidateMask = candidateMask;

		public bool Equals(AllowedMaskKey other)
			=> PBits == other.PBits
			&& MaxQBits == other.MaxQBits
			&& Block == other.Block
			&& Carry == other.Carry
			&& QNibble == other.QNibble
			&& QSig == other.QSig
			&& ASig == other.ASig
			&& CandidateMask == other.CandidateMask;

		public override bool Equals(object? obj) => obj is AllowedMaskKey k && Equals(k);

		public override int GetHashCode()
			=> HashCode.Combine(PBits, MaxQBits, Block, Carry, QNibble, QSig, ASig, CandidateMask);
	}



	// Reduced state key for conservative per-block pruning.
	// This key must be strong enough to avoid conflating states that can yield different allowed masks.
	private readonly struct ReducedStateKey(
		int pBits,
		int maxQBits,
		int block,
		int carry,
		ulong qPrefixSig,
		ulong aSig) : IEquatable<ReducedStateKey>
	{
		public readonly int PBits = pBits;
		public readonly int MaxQBits = maxQBits;
		public readonly int Block = block;
		public readonly int Carry = carry;
		public readonly ulong QPrefixSig = qPrefixSig;
		public readonly ulong ASig = aSig;

		public bool Equals(ReducedStateKey other)
			=> PBits == other.PBits
			   && MaxQBits == other.MaxQBits
			   && Block == other.Block
			   && Carry == other.Carry
			   && QPrefixSig == other.QPrefixSig
			   && ASig == other.ASig;

		public override bool Equals(object? obj) => obj is ReducedStateKey k && Equals(k);

		public override int GetHashCode() => HashCode.Combine(PBits, MaxQBits, Block, Carry, QPrefixSig, ASig);
	}

	// Conservative per-block pruning statistics:
	// For a given block, if a q-nibble is rejected (allowed mask == 0) for every reachable reduced state at that block,
	// we can safely skip it for the remainder of the run.
	private sealed class BlockPruneStats
	{
		private readonly Dictionary<ReducedStateKey, ushort> _stateRejectedMask = new(capacity: 1024);

		public int SeenStates { get; private set; }
		public ushort GlobalForbiddenMask { get; private set; }

		// For each qNibble d, count how many reduced states have already rejected it.
		private readonly int[] _rejectCounts = new int[16];

		public void RegisterState(in ReducedStateKey key)
		{
			if (_stateRejectedMask.ContainsKey(key)) return;

			_stateRejectedMask[key] = 0;
			SeenStates++;

			// New state potentially invalidates previously "global" forbids; recompute conservatively.
			RecomputeGlobalMask();
		}

		public void MarkRejected(in ReducedStateKey key, byte qNibble)
		{
			if (!_stateRejectedMask.TryGetValue(key, out ushort mask))
			{
				// Should not happen if RegisterState() is called, but keep it safe.
				_stateRejectedMask[key] = 0;
				SeenStates++;
				mask = 0;
			}

			ushort bit = (ushort)(1 << qNibble);
			if ((mask & bit) != 0) return; // already counted for this reduced state

			mask |= bit;
			_stateRejectedMask[key] = mask;

			_rejectCounts[qNibble]++;

			// If all seen states reject this qNibble, we can forbid it globally at this block.
			if (_rejectCounts[qNibble] == SeenStates)
				GlobalForbiddenMask |= bit;
		}

		private void RecomputeGlobalMask()
		{
			ushort m = 0;
			for (int d = 0; d < 16; d++)
				if (_rejectCounts[d] == SeenStates && SeenStates > 0)
					m |= (ushort)(1 << d);
			GlobalForbiddenMask = m;
		}
	}

	private struct State
	{
		public ABitsWindow AWindow;   // rolling a window + planned bits map
		public int BitIndex;
		public int Block;
		public int Carry;
		public byte KFlushRemaining;
		public bool KTerminated;
		public byte PostTermZeroQNibbles;
		public DynamicBitset QBits;   // q bits [0..maxQBits-1]
		public QGenState16 QState;

		public static State CreateInitial(QGenState16 initQ, int maxQBits)
		{
			var s = new State
			{
				Block = initQ.NibbleIndex,
				BitIndex = 0,
				Carry = 0,
				QState = initQ,
				QBits = new DynamicBitset(maxQBits),
				AWindow = new ABitsWindow(maxQBits),
				KFlushRemaining = 0,
				KTerminated = false,
				PostTermZeroQNibbles = 0,
			};

			s.QBits.Set(0);
			// a0=1
			s.AWindow.SetPlanned(0, 1);
			return s;
		}

		// Clone bitsets/windows to isolate branches.
		// For performance, replace with copy-on-write/pooling.

		public State CloneShallow() => new()

		{
			Block = Block,
			BitIndex = BitIndex,
			Carry = Carry,
			QState = QState,
			QBits = QBits.CloneShared(),
			AWindow = AWindow.Clone(),
			KFlushRemaining = KFlushRemaining,
			KTerminated = KTerminated,
			PostTermZeroQNibbles = PostTermZeroQNibbles,
		};
	}

	// ----------------------------- Bit primitives -----------------------------

	/// <summary>Dynamic bitset with fast enumeration of set bits up to a limit.</summary>
	
	public sealed class DynamicBitset
	{
		private sealed class BitsetData
		{
			public readonly int Bits;
			public ulong[] Words;

			// Indices of words that are non-zero (monotone increasing in this scanner).
			public int[] NzWords;
			public int NzCount;

			public BitsetData(int bits)
			{
				Bits = bits;
				Words = new ulong[(bits + 63) >> 6];
				NzWords = new int[16];
				NzCount = 0;
			}

			public BitsetData(int bits, ulong[] words, int[] nzWords, int nzCount)
			{
				Bits = bits;
				Words = words;
				NzWords = nzWords;
				NzCount = nzCount;
			}
		}

		private readonly BitsetData _data;
		private bool _shared; // copy-on-write guard

		public DynamicBitset(int bits)
		{
			_data = new BitsetData(bits);
			_shared = false;
		}

		private DynamicBitset(BitsetData data, bool shared)
		{
			_data = data;
			_shared = shared;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private void EnsureWritable()
		{
			if (!_shared) return;

			// Deep copy underlying arrays.
			var w = new ulong[_data.Words.Length];
			Array.Copy(_data.Words, w, w.Length);

			var nz = new int[_data.NzWords.Length];
			Array.Copy(_data.NzWords, nz, nz.Length);

			var copied = new BitsetData(_data.Bits, w, nz, _data.NzCount);

			// Switch this instance to its private copy.
			// Note: _data is readonly, so we rely on the fact that CloneShared creates a *new* DynamicBitset
			// instance for children. This instance becomes non-shared by clearing _shared; its _data remains unique
			// because no other instance references 'copied'.
			//
			// To preserve that with readonly _data, we keep _data as reference to mutable arrays inside BitsetData.
			// Here we mutate the arrays in-place by replacing their references.
			_data.Words = copied.Words;
			_data.NzWords = copied.NzWords;
			_data.NzCount = copied.NzCount;

			_shared = false;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private void AddNzWord(int wIndex)
		{
			int n = _data.NzCount;
			if ((uint)n >= (uint)_data.NzWords.Length)
			{
				// grow
				int newLen = _data.NzWords.Length * 2;
				if (newLen < 16) newLen = 16;
				var tmp = new int[newLen];
				Array.Copy(_data.NzWords, tmp, _data.NzWords.Length);
				_data.NzWords = tmp;
			}
			_data.NzWords[n] = wIndex;
			_data.NzCount = n + 1;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public void Set(int bit)
		{
			int bits = _data.Bits;
			if ((uint)bit >= (uint)bits) return;

			int w = bit >> 6;
			int b = bit & 63;
			ulong mask = 1UL << b;

			ulong before = _data.Words[w];
			if ((before & mask) != 0) return; // already set; no mutation needed

			EnsureWritable();

			before = _data.Words[w];
			ulong after = before | mask;
			_data.Words[w] = after;

			if (before == 0 && after != 0)
				AddNzWord(w);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public int Get(int bit)
		{
			int bits = _data.Bits;
			if ((uint)bit >= (uint)bits) return 0;
			ulong v = _data.Words[bit >> 6];
			return (int)((v >> (bit & 63)) & 1UL);
		}

		/// <summary>Deep clone (copies backing arrays).</summary>
		public DynamicBitset Clone()
		{
			var w = new ulong[_data.Words.Length];
			Array.Copy(_data.Words, w, w.Length);

			var nz = new int[_data.NzWords.Length];
			Array.Copy(_data.NzWords, nz, nz.Length);

			return new DynamicBitset(new BitsetData(_data.Bits, w, nz, _data.NzCount), shared: false);
		}

		/// <summary>
		/// Shared clone (copy-on-write). Children share the same arrays until one of them mutates.
		/// </summary>
		public DynamicBitset CloneShared()
		{
			// Mark both this and the clone as shared so either mutation triggers EnsureWritable.
			_shared = true;
			return new DynamicBitset(_data, shared: true);
		}

		public ulong Signature64UpTo(int maxBitInclusive)
		{
			if (maxBitInclusive < 0) return 0UL;
			int bits = _data.Bits;
			int maxBit = Math.Min(maxBitInclusive, bits - 1);
			int maxWord = maxBit >> 6;
			int lastBitInWord = maxBit & 63;

			const ulong FnvOffset = 1469598103934665603UL;
			const ulong FnvPrime = 1099511628211UL;

			ulong h = FnvOffset;
			for (int i = 0; i <= maxWord; i++)
			{
				ulong v = _data.Words[i];
				if (i == maxWord)
				{
					ulong mask = lastBitInWord == 63 ? ulong.MaxValue : ((1UL << (lastBitInWord + 1)) - 1UL);
					v &= mask;
				}
				h ^= v;
				h *= FnvPrime;
			}
			return h;
		}

		public ulong Signature64WithNibble(int startBit, byte qNibble, int maxQBits)
			=> Signature64WithNibbleUpTo(_data.Bits - 1, startBit, qNibble, maxQBits);

		public ulong Signature64WithNibbleUpTo(int maxBitInclusive, int startBit, byte qNibble, int maxQBits)
		{
			const ulong FnvOffset = 1469598103934665603UL;
			const ulong FnvPrime = 1099511628211UL;

			int bits = _data.Bits;
			int maxBit = Math.Min(Math.Max(maxBitInclusive, 0), bits - 1);
			int maxWord = maxBit >> 6;

			int w0 = startBit >> 6;
			int w1 = (startBit + 3) >> 6;

			ulong add0 = 0;
			ulong add1 = 0;

			for (int b = 0; b < 4; b++)
			{
				int bitPos = startBit + b;
				if (bitPos >= maxQBits || bitPos > maxBit) break;

				if (((qNibble >> b) & 1) == 0) continue;

				int wi = bitPos >> 6;
				int bi = bitPos & 63;
				if (wi == w0) add0 |= 1UL << bi;
				else add1 |= 1UL << bi;
			}

			// enforce q0=1 always
			if (w0 == 0) add0 |= 1UL;
			else if (w1 == 0) add1 |= 1UL;

			ulong h = FnvOffset;
			for (int i = 0; i <= maxWord; i++)
			{
				ulong v = _data.Words[i];
				if (i == w0) v |= add0;
				if (i == w1 && w1 != w0) v |= add1;

				if (i == maxWord)
				{
					int lastBit = maxBit & 63;
					ulong mask = lastBit == 63 ? ulong.MaxValue : ((1UL << (lastBit + 1)) - 1UL);
					v &= mask;
				}

				h ^= v;
				h *= FnvPrime;
			}
			return h;
		}

		public BigInteger ToBigInteger()
		{
			// BigInteger expects little-endian byte array.
			byte[] bytes = new byte[_data.Words.Length * 8 + 1];
			int idx = 0;
			for (int i = 0; i < _data.Words.Length; i++)
			{
				ulong v = _data.Words[i];
				for (int b = 0; b < 8; b++)
				{
					bytes[idx++] = (byte)(v & 0xFF);
					v >>= 8;
				}
			}
			return new BigInteger(bytes);
		}

		/// <summary>
		/// Allocation-free enumeration of set bits up to a limit (inclusive), iterating only over non-zero words.
		/// </summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public SetBitsUpToEnumerable EnumerateSetBitsUpTo(int maxBitInclusive)
			=> new(_data.Words, _data.NzWords, _data.NzCount, _data.Bits, maxBitInclusive);

		public readonly struct SetBitsUpToEnumerable
		{
			private readonly ulong[] _words;
			private readonly int[] _nzWords;
			private readonly int _nzCount;
			private readonly int _bits;
			private readonly int _maxBitInclusive;

			[MethodImpl(MethodImplOptions.AggressiveInlining)]
			public SetBitsUpToEnumerable(ulong[] words, int[] nzWords, int nzCount, int bits, int maxBitInclusive)
			{
				_words = words;
				_nzWords = nzWords;
				_nzCount = nzCount;
				_bits = bits;
				_maxBitInclusive = maxBitInclusive;
			}

			[MethodImpl(MethodImplOptions.AggressiveInlining)]
			public Enumerator GetEnumerator() => new(_words, _nzWords, _nzCount, _bits, _maxBitInclusive);

			public struct Enumerator
			{
				private readonly ulong[] _words;
				private readonly int[] _nzWords;
				private readonly int _nzCount;
				private readonly int _maxWord;
				private readonly int _lastBitInWord;

				private int _i;     // index in nzWords list
				private int _w;     // current word index
				private ulong _word;
				private bool _started;

				public int Current { get; private set; }

				[MethodImpl(MethodImplOptions.AggressiveInlining)]
				public Enumerator(ulong[] words, int[] nzWords, int nzCount, int bits, int maxBitInclusive)
				{
					_words = words;
					_nzWords = nzWords;
					_nzCount = nzCount;

					if (maxBitInclusive < 0 || bits <= 0)
					{
						_maxWord = -1;
						_lastBitInWord = 0;
					}
					else
					{
						int maxBit = Math.Min(maxBitInclusive, bits - 1);
						_maxWord = maxBit >> 6;
						_lastBitInWord = maxBit & 63;
					}

					_i = 0;
					_w = 0;
					_word = 0;
					_started = false;
					Current = 0;
				}

				[MethodImpl(MethodImplOptions.AggressiveInlining)]
				public bool MoveNext()
				{
					if (_maxWord < 0) return false;

					if (!_started)
					{
						_started = true;
						_i = 0;
						if (!AdvanceToNextNonZeroWord())
							return false;
					}

					while (true)
					{
						while (_word != 0)
						{
							int tz = BitOperations.TrailingZeroCount(_word);
							Current = (_w << 6) + tz;
							_word &= _word - 1;
							return true;
						}

						_i++;
						if (!AdvanceToNextNonZeroWord())
							return false;
					}
				}

				[MethodImpl(MethodImplOptions.AggressiveInlining)]
				private bool AdvanceToNextNonZeroWord()
				{
					while (_i < _nzCount)
					{
						int w = _nzWords[_i];
						if (w > _maxWord) return false;
						_w = w;
						_word = LoadWord(_w);
						if (_word != 0) return true;
						_i++;
					}
					return false;
				}

				[MethodImpl(MethodImplOptions.AggressiveInlining)]
				private ulong LoadWord(int w)
				{
					ulong word = _words[w];
					if (w == _maxWord)
					{
						ulong mask = _lastBitInWord == 63 ? ulong.MaxValue : ((1UL << (_lastBitInWord + 1)) - 1UL);
						word &= mask;
					}
					return word;
				}
			}
		}

		public void ForEachSetBitUpTo(int maxBitInclusive, Action<int> action)
		{
			if (maxBitInclusive < 0) return;

			int bits = _data.Bits;
			int maxBit = Math.Min(maxBitInclusive, bits - 1);
			int maxWord = maxBit >> 6;
			int lastBitInWord = maxBit & 63;

			int nzCount = _data.NzCount;
			int[] nzWords = _data.NzWords;
			ulong[] words = _data.Words;

			for (int i = 0; i < nzCount; i++)
			{
				int w = nzWords[i];
				if (w > maxWord) break;

				ulong word = words[w];
				if (w == maxWord)
				{
					ulong mask = lastBitInWord == 63 ? ulong.MaxValue : ((1UL << (lastBitInWord + 1)) - 1UL);
					word &= mask;
				}

				while (word != 0)
				{
					int tz = BitOperations.TrailingZeroCount(word);
					int bit = (w << 6) + tz;
					action(bit);
					word &= word - 1;
				}
			}
		}
	}


	/// <summary>
	/// Rolling window of a-bits of length maxQBits with support for "planned" bits.
	/// We separate:
	/// - planned bits: explicit assignments for specific a_i within the window
	/// - pushed bits: bits that have been committed as BitIndex advances
	///
	/// This allows nibble-level assignment while still streaming column-by-column.
	/// </summary>

	public sealed class ABitsWindow
	{
		private readonly int _windowBits;
		private readonly int _wordCount;

		// Committed bits are stored in a circular (bit-level) ring buffer.
		// headBit points to the position holding the most recently committed bit (distance 0).
		private readonly ulong[] _ring;
		private int _headBit;
		private int _currentJ; // last committed index j (monotone increasing)

		// Planned a_i values (sparse) for not-yet-committed indices.
		private readonly Dictionary<int, int> _planned;

		public int CurrentJ => _currentJ;

		public ABitsWindow(int windowBits)
		{
			_windowBits = windowBits;
			_wordCount = (windowBits + 63) >> 6;
			_ring = new ulong[_wordCount];
			_headBit = -1;  // "empty"
			_currentJ = -1;
			_planned = new Dictionary<int, int>(capacity: 1024);
		}

		private ABitsWindow(int windowBits, ulong[] ring, int headBit, int currentJ, Dictionary<int, int> planned)
		{
			_windowBits = windowBits;
			_wordCount = (windowBits + 63) >> 6;
			_ring = ring;
			_headBit = headBit;
			_currentJ = currentJ;
			_planned = planned;
		}

		public ABitsWindow Clone()
		{
			var ring2 = new ulong[_ring.Length];
			Array.Copy(_ring, ring2, ring2.Length);
			var planned2 = new Dictionary<int, int>(_planned);
			return new ABitsWindow(_windowBits, ring2, _headBit, _currentJ, planned2);
		}

		/// <summary>
		/// A compact signature of the current a-window state (committed ring + planned bits).
		/// Used to keep AllowedMaskCache semantically correct: allowed-a depends on previously committed a-bits.
		/// </summary>
		public ulong Signature64()
		{
			const ulong FnvOffset = 1469598103934665603UL;
			const ulong FnvPrime = 1099511628211UL;

			ulong h = FnvOffset;

			h ^= (ulong)(uint)_windowBits; h *= FnvPrime;
			h ^= (ulong)(uint)_headBit; h *= FnvPrime;
			h ^= (ulong)(uint)_currentJ; h *= FnvPrime;

			for (int i = 0; i < _ring.Length; i++)
			{
				h ^= _ring[i];
				h *= FnvPrime;
			}

			// Planned bits: order-independent combine
			ulong ph = 0;
			foreach (var kv in _planned)
			{
				// mix index and bit (bit is 0/1)
				ulong v = ((ulong)(uint)kv.Key << 1) ^ (uint)kv.Value;
				ph ^= v + 0x9e3779b97f4a7c15UL + (ph << 6) + (ph >> 2);
			}
			h ^= ph;
			h *= FnvPrime;

			return h;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public void SetPlanned(int aIndex, int bit)
		{
			if (bit != 0) _planned[aIndex] = 1;
			else _planned.Remove(aIndex);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public void PushBit(int j, int bit)
		{
			// Commit bit for column j (a_j).
			_planned.Remove(j);
			_currentJ = j;

			// Advance ring head and write the new bit.
			_headBit = (_headBit + 1) % _windowBits;

			int word = _headBit >> 6;
			int b = _headBit & 63;

			_ring[word] &= ~(1UL << b);
			if (bit != 0) _ring[word] |= 1UL << b;

			// planned map should not grow without bound; remove entries older than window.
			int minKeep = j - _windowBits;
			if (_planned.Count > 0)
			{
				var toRemove = new List<int>();
				foreach (var kv in _planned)
					if (kv.Key <= minKeep) toRemove.Add(kv.Key);
				foreach (int k in toRemove) _planned.Remove(k);
			}
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public int GetBitRelative(int aIndex)
		{
			// planned must have priority even if aIndex > _currentJ
			if (_planned.TryGetValue(aIndex, out int plannedBit))
				return plannedBit;

			int d = _currentJ - aIndex;
			if (d < 0 || d >= _windowBits) return 0;
			if (_headBit < 0) return 0; // empty

			int pos = _headBit - d;
			pos %= _windowBits;
			if (pos < 0) pos += _windowBits;

			int word = pos >> 6;
			int bit = pos & 63;
			return (int)((_ring[word] >> bit) & 1UL);
		}

		/// <summary>
		/// Try to read an a-bit if it is currently KNOWN (planned or already committed within the window).
		/// Returns false when the bit is outside the committed window and not planned yet.
		///
		/// IMPORTANT: Callers that implement contradiction pruning must treat "unknown" as Unknown (may be 0 or 1),
		/// never as 0, to avoid false negatives.
		/// </summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public bool TryGetKnownBit(int aIndex, out int bit)
		{
			// Planned has priority and is always a known assignment.
			if (_planned.TryGetValue(aIndex, out int plannedBit))
			{
				bit = plannedBit;
				return true;
			}

			// Not yet committed.
			if (aIndex > _currentJ)
			{
				bit = 0;
				return false;
			}

			int d = _currentJ - aIndex;
			if (d < 0 || d >= _windowBits || _headBit < 0)
			{
				bit = 0;
				return false;
			}

			int pos = _headBit - d;
			pos %= _windowBits;
			if (pos < 0) pos += _windowBits;

			int word = pos >> 6;
			int b = pos & 63;
			bit = (int)((_ring[word] >> b) & 1UL);
			return true;
		}

		public bool TryGetPlanned(int aIndex, out int bit)
			=> _planned.TryGetValue(aIndex, out bit);
	}

	// ----------------------------------------------------------------------------
	// q nibble generator for q = (2p)*k + 1 in base 16, without materializing k/q.
	// ----------------------------------------------------------------------------

	public readonly struct QGenState16
	{
		public readonly int NibbleIndex;   // <-- DODAJ
		public readonly uint Carry16;
		public readonly ulong KWindowPacked;

		public QGenState16(int nibbleIndex, uint carry16, ulong kWindowPacked)
		{
			NibbleIndex = nibbleIndex;
			Carry16 = carry16;
			KWindowPacked = kWindowPacked;
		}
	}

	public sealed class QNibbleGenerator16
	{
		private readonly byte[] _mDigits; // nibbles of m=2p in base 16, LSB-first
		private readonly int _mLen;
		private readonly int _kWinLen;    // mLen-1
		private readonly ulong _kWinMask;

		public QNibbleGenerator16(ulong p)
		{
			ulong m = checked(2UL * p);
			_mDigits = ToBase16DigitsLSB(m);
			_mLen = _mDigits.Length;
			_kWinLen = Math.Max(0, _mLen - 1);
			_kWinMask = _kWinLen == 0 ? 0UL : ((1UL << (4 * _kWinLen)) - 1UL);
		}

		public static QGenState16 InitialState => new(0, 0, 0);
		public int KWinLen => _kWinLen;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public void ComputeNext(QGenState16 state, byte kNibble, out byte qNibble, out QGenState16 nextState)
		{
			ulong sum = state.Carry16;
			int nibbleIndex = state.NibbleIndex;

			// +1 only contributes to the very first nibble, once
			if (nibbleIndex == 0)
				sum += 1UL;

			// j=0 term
			sum += (ulong)_mDigits[0] * kNibble;

			int maxJ = Math.Min(nibbleIndex, _mLen - 1);
			for (int j = 1; j <= maxJ; j++)
			{
				byte mj = _mDigits[j];
				byte kPrev = GetKFromWindow(state.KWindowPacked, j - 1, _kWinLen);
				sum += (ulong)mj * kPrev;
			}

			qNibble = (byte)(sum & 0xFUL);
			uint nextCarry = (uint)(sum >> 4);

			ulong nextWin = state.KWindowPacked;
			if (_kWinLen > 0)
				nextWin = ((nextWin << 4) | kNibble) & _kWinMask;

			nextState = new QGenState16(
				nibbleIndex + 1,
				nextCarry,
				nextWin
			);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public void ComputeAllNext(
			QGenState16 state,
			Span<byte> qNibByKn,          // len=16
			Span<QGenState16> nextByKn,   // len=16
			Span<int> countsByQn          // len=16
		)
		{
			ulong baseSum = state.Carry16;
			int nibbleIndex = state.NibbleIndex;

			// +1 only contributes to the very first nibble, once
			if (nibbleIndex == 0)
				baseSum += 1UL;

			// Σ_{j=1..maxJ} mj * kPrev (part independent of current kn)
			int maxJ = Math.Min(nibbleIndex, _mLen - 1);
			for (int j = 1; j <= maxJ; j++)
			{
				byte mj = _mDigits[j];
				byte kPrev = GetKFromWindow(state.KWindowPacked, j - 1, _kWinLen);
				baseSum += (ulong)mj * kPrev;
			}

			ulong m0 = _mDigits[0];

			for (byte kn = 0; kn <= 15; kn++)
			{
				ulong sum = baseSum + m0 * kn;

				byte qn = (byte)(sum & 0xFUL);
				uint nextCarry = (uint)(sum >> 4);

				ulong nextWin = state.KWindowPacked;
				if (_kWinLen > 0)
					nextWin = ((nextWin << 4) | kn) & _kWinMask;

				qNibByKn[kn] = qn;
				nextByKn[kn] = new QGenState16(nibbleIndex + 1, nextCarry, nextWin);
				countsByQn[qn]++;
			}
		}


		private static byte[] ToBase16DigitsLSB(ulong x)
		{
			if (x == 0) return [0];
			Span<byte> tmp = stackalloc byte[16];
			int len = 0;
			while (x != 0)
			{
				tmp[len++] = (byte)(x & 0xF);
				x >>= 4;
			}
			var r = new byte[len];
			for (int i = 0; i < len; i++) r[i] = tmp[i];
			return r;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private static byte GetKFromWindow(ulong packed, int idxFromNewest, int winLen)
		{
			// newest nibble is stored at LSB (shift 0)
			// idxFromNewest=0 => most recently appended nibble
			if ((uint)idxFromNewest >= (uint)winLen) return 0;
			int shift = idxFromNewest * 4;
			return (byte)((packed >> shift) & 0xF);
		}

		public static void SelfTest(ulong p)
		{
			var gen = new QNibbleGenerator16(p);
			var st = InitialState;

			// k = 4383 = 0x111F => nibble LSB-first: F,1,1,1,0,0,...
			byte[] kNibs = { 0xF, 0x1, 0x1, 0x1, 0x0, 0x0, 0x0, 0x0 };

			BigInteger q = BigInteger.Zero;
			for (int i = 0; i < 12; i++)
			{
				byte kn = i < kNibs.Length ? kNibs[i] : (byte)0;
				gen.ComputeNext(st, kn, out byte qn, out var ns);
				q |= (new BigInteger(qn) << (4 * i));
				st = ns;
			}

			Console.WriteLine($"[SELFTEST] p={p} q(hex)={q.ToString("X")}");
			Console.WriteLine($"[SELFTEST] p={p} q(dec)={q}");
		}
	}
}
