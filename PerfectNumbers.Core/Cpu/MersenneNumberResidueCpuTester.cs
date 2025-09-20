namespace PerfectNumbers.Core.Cpu;

public class MersenneNumberResidueCpuTester
{
	private ModResidueTracker? _mersenneResidueTracker;

	// CPU residue variant using tracker + unrolled residue updates (no method calls in the loop).
    public void Scan(
            ulong exponent,
            UInt128 twoP,
            bool lastIsSeven,
            UInt128 perSetLimit,
            UInt128 setCount,
            UInt128 overallLimit,
            ref bool isPrime,
            ref bool divisorsExhausted)
    {
            if (setCount == UInt128.Zero || perSetLimit == UInt128.Zero || overallLimit == UInt128.Zero)
            {
                    return;
            }

                // Initialize/update Mersenne residue tracker and start a merge walk for ascending q divisors

                _mersenneResidueTracker ??= new(ResidueModel.Mersenne);
                // For small divisors q <= 4,000,000 use cached cycle length to fast-path checks
                // and avoid per-q powmods when possible when scanning lanes.
                _mersenneResidueTracker.BeginMerge(exponent);

                ulong step10 = (ulong)(twoP % 10UL) % 10UL;
                ulong step8 = (ulong)(twoP % 8UL);
                ulong step3 = (ulong)(twoP % 3UL);
                ulong step5 = (ulong)(twoP % 5UL);

		// Allowed last-digit sets for q depending on last digit of M_p
		// lastIsSeven == true  => allow {7,9}
		// lastIsSeven == false => allow {1,3,7,9}
		int allowMask = lastIsSeven ? CpuConstants.LastSevenMask10 : CpuConstants.LastOneMask10;

		// Predeclare temps to avoid redeclaration overhead in the loop
                UInt128 remaining;
                int i, lanes;
                // per-lane residues
                ulong r10_0, r10_1, r10_2, r10_3, r8_0, r8_1, r8_2, r8_3, r3_0, r3_1, r3_2, r3_3, r5_0, r5_1, r5_2, r5_3;
                // q lanes
                UInt128 twoP2 = twoP + twoP,
                                twoP3 = twoP2 + twoP,
                                twoP4 = twoP3 + twoP,
                                qIncremental;

                bool localIsPrime, isDivider;

                ModResidueTracker tracker = _mersenneResidueTracker!;
                UInt128 limitInclusive = overallLimit + UInt128.One;
                for (UInt128 setIndex = UInt128.Zero; setIndex < setCount && Volatile.Read(ref isPrime); setIndex++)
                {
                        UInt128 setOffset = perSetLimit * setIndex;
                        UInt128 k = setOffset + UInt128.One;
                        if (k >= limitInclusive)
                        {
                                break;
                        }

                        UInt128 setLimitExclusive = setOffset + perSetLimit + UInt128.One;
                        if (setLimitExclusive > limitInclusive)
                        {
                                setLimitExclusive = limitInclusive;
                        }

                        UInt128 q = twoP * k + UInt128.One;
                        q.Mod10_8_5_3(out ulong r10, out ulong r8, out ulong r5, out ulong r3);

                        while (k < setLimitExclusive && (localIsPrime = Volatile.Read(ref isPrime)))
                        {
                                remaining = setLimitExclusive - k;
                                if (remaining == UInt128.Zero)
                                {
                                        break;
                                }

                                lanes = remaining > 4UL ? 4 : (int)remaining;
                                if (lanes <= 0)
                                {
                                        break;
                                }

                                // Compute lane residues incrementally (avoid multiplications, minimize reductions)
                                // r10
                                r10_0 = r10;
                                r10_1 = r10_0 + step10; if (r10_1 >= 20UL) r10_1 -= 20UL; if (r10_1 >= 10UL) r10_1 -= 10UL;
                                r10_2 = r10_1 + step10; if (r10_2 >= 20UL) r10_2 -= 20UL; if (r10_2 >= 10UL) r10_2 -= 10UL;
                                r10_3 = r10_2 + step10; if (r10_3 >= 20UL) r10_3 -= 20UL; if (r10_3 >= 10UL) r10_3 -= 10UL;

                                // r8
                                r8_0 = r8 & 7UL;
                                r8_1 = (r8_0 + step8) & 7UL;
                                r8_2 = (r8_1 + step8) & 7UL;
                                r8_3 = (r8_2 + step8) & 7UL;

                                // r3 (pattern 0,1,2,0)
                                r3_0 = r3; if (r3_0 >= 3UL) r3_0 -= 3UL;
                                r3_1 = r3_0 + step3; if (r3_1 >= 3UL) r3_1 -= 3UL;
                                r3_2 = r3_1 + step3; if (r3_2 >= 3UL) r3_2 -= 3UL;
                                r3_3 = r3; if (r3_3 >= 3UL) r3_3 -= 3UL;

                                // r5 (pattern 0,1,2,3)
                                r5_0 = r5; if (r5_0 >= 5UL) r5_0 -= 5UL;
                                r5_1 = r5_0 + step5; if (r5_1 >= 10UL) r5_1 -= 10UL; if (r5_1 >= 5UL) r5_1 -= 5UL;
                                r5_2 = r5_1 + step5; if (r5_2 >= 10UL) r5_2 -= 10UL; if (r5_2 >= 5UL) r5_2 -= 5UL;
                                r5_3 = r5_2 + step5; if (r5_3 >= 10UL) r5_3 -= 10UL; if (r5_3 >= 5UL) r5_3 -= 5UL;

                                // lane 0
                                if (lanes >= 1 &&
                                        (((allowMask >> (int)r10_0) & 1) != 0 && (r8_0 == 1UL || r8_0 == 7UL) && r3_0 != 0UL && r5_0 != 0UL))
                                {
                                        if (localIsPrime && tracker.MergeOrAppend(exponent, q, out isDivider) && isDivider &&
                                                (q <= ulong.MaxValue ? ((ulong)q).IsPrimeCandidate() : q.IsPrimeCandidate()))
                                        {
                                                localIsPrime = false;
                                        }
                                }

                                // lane 1
                                qIncremental = q + twoP;
                                if (lanes >= 2 &&
                                        ((allowMask >> (int)r10_1) & 1) != 0 && (r8_1 == 1UL || r8_1 == 7UL) && r3_1 != 0UL && r5_1 != 0UL &&
                                        localIsPrime)
                                {
                                        if (localIsPrime && tracker.MergeOrAppend(exponent, qIncremental, out isDivider) && isDivider &&
                                                (qIncremental <= ulong.MaxValue ? ((ulong)qIncremental).IsPrimeCandidate() : qIncremental.IsPrimeCandidate()))
                                        {
                                                localIsPrime = false;
                                        }
                                }

                                // lane 2
                                qIncremental += twoP;
                                if (lanes >= 3 &&
                                        ((allowMask >> (int)r10_2) & 1) != 0 && (r8_2 == 1UL || r8_2 == 7UL) && r3_2 != 0UL && r5_2 != 0UL &&
                                        localIsPrime)
                                {
                                        if (localIsPrime && tracker.MergeOrAppend(exponent, qIncremental, out isDivider) && isDivider &&
                                                (qIncremental <= ulong.MaxValue ? ((ulong)qIncremental).IsPrimeCandidate() : qIncremental.IsPrimeCandidate()))
                                        {
                                                localIsPrime = false;
                                        }
                                }

                                // lane 3
                                qIncremental += twoP;
                                if (lanes >= 4 &&
                                        ((allowMask >> (int)r10_3) & 1) != 0 && (r8_3 == 1UL || r8_3 == 7UL) && r3_3 != 0UL && r5_3 != 0UL &&
                                        localIsPrime)
                                {
                                        if (localIsPrime && tracker.MergeOrAppend(exponent, qIncremental, out isDivider) && isDivider &&
                                                (qIncremental <= ulong.MaxValue ? ((ulong)qIncremental).IsPrimeCandidate() : qIncremental.IsPrimeCandidate()))
                                        {
                                                localIsPrime = false;
                                        }
                                }

                                if (!localIsPrime)
                                {
                                        Volatile.Write(ref isPrime, false);
                                        divisorsExhausted = true;
                                        return;
                                }
                                else if (!Volatile.Read(ref isPrime))
                                {
                                        divisorsExhausted = true;
                                        return;
                                }

                                // advance base by processed lanes
                                k += (UInt128)lanes;
                                q += lanes switch
                                {
                                        0 => 0UL,
                                        1 => twoP,
                                        2 => twoP2,
                                        3 => twoP3,
                                        4 => twoP4,
                                        _ => throw new ArgumentException("Unsupproted value", nameof(twoP))
                                };

                                // advance residues by 'lanes' steps (consistent with base advance)
                                for (i = 0; i < lanes; i++)
                                {
                                        r10 += step10; if (r10 >= 20UL) r10 -= 20UL; if (r10 >= 10UL) r10 -= 10UL;
                                        r8 = (r8 + step8) & 7UL;
                                        r3 += step3; if (r3 >= 3UL) r3 -= 3UL;
                                        r5 += step5; if (r5 >= 10UL) r5 -= 10UL; if (r5 >= 5UL) r5 -= 5UL;
                }

                divisorsExhausted = true;
    }
}
        }
}
