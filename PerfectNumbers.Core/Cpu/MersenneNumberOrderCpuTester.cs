namespace PerfectNumbers.Core.Cpu;

public class MersenneNumberOrderCpuTester(GpuKernelType kernelType)
{
	private readonly GpuKernelType _kernelType = kernelType;

	public void Scan(ulong exponent, UInt128 twoP, bool lastIsSeven, UInt128 maxK, ref bool isPrime)
	{
		UInt128 k = 1UL;
		var auto = new MersenneResidueAutomaton(exponent);
		UInt128 qCycle,
				q = auto.CurrentQ();

		while (k <= maxK && Volatile.Read(ref isPrime))
		{
            qCycle = MersenneDivisorCycles.GetCycle(q);
            // TODO: When this lookup misses the snapshot, invoke the configured device to compute the
            // single required cycle on demand without persisting it so the CPU order path retains the
            // cycle-stepping speedups while honoring the no-extra-cache constraint for large divisors.
			int allowMask2 = lastIsSeven ? CpuConstants.LastSevenMask10 : CpuConstants.LastOneMask10;
			bool shouldCheck = ((allowMask2 >> (int)auto.Mod10R) & 1) != 0;
			if (shouldCheck)
			{
				ulong mod8 = auto.Mod8R;
				if (mod8 != 1UL && mod8 != 7UL)
				{
					shouldCheck = false;
				}
				else if (auto.Mod3R == 0UL || auto.Mod5R == 0UL)
				{
					shouldCheck = false;
				}
			}
			if (shouldCheck)
			{
                                if (_kernelType == GpuKernelType.Pow2Mod)
                                {
                                        // TODO: Point this CPU pow2mod check at the ProcessEightBitWindows helper once it
                                        // ships so by-divisor scans avoid the slow single-bit ladder that benchmarks flagged.
                                        if (exponent.PowModWithCycle(q, qCycle) == 1UL)
                                        {
                                                Volatile.Write(ref isPrime, false);
                                                break;
                                        }
                                }
                                else
                                {
                                        UInt128 phi = q - 1UL;
                                        if (phi <= ulong.MaxValue)
                                        {
                                                ulong phi64 = (ulong)phi;
                                                // TODO: Switch these phi-based powmods to the shared windowed helper so CPU
                                                // fallback paths keep pace with the optimized GPU kernels on large divisors.
                                                if (phi64.PowModWithCycle(q, qCycle) == 1UL)
                                                {
                                                        // TODO: Reuse the windowed pow2 helper for halfPow as soon as it is
                                                        // available instead of recalculating via square-and-multiply.
                                                        UInt128 halfPow = (phi64 >> 1).PowModWithCycle(q, qCycle) - 1UL;
                                                        if (halfPow.BinaryGcd(q) == 1UL)
                                                        {
                                                                ulong divMul = (ulong)((((UInt128)1 << 64) - 1UL) / exponent) + 1UL;
                                                                ulong div = phi64.FastDiv64(exponent, divMul);
                                                                // TODO: Replace this divisor powmod with the ProcessEightBitWindows
                                                                // implementation so order scans stop paying the slower bit-serial
                                                                // loop measured in the MulMod benchmarks.
                                                                UInt128 divPow = div.PowModWithCycle(q, qCycle) - 1UL;
                                                                if (divPow.BinaryGcd(q) == 1UL)
                                                                {
                                                                        Volatile.Write(ref isPrime, false);
                                                                        break;
                                                                }
							}
						}
					}
				}
			}

			k += 1UL;
			auto.Next();
			q = auto.CurrentQ();
		}
		
		return;
	}

}