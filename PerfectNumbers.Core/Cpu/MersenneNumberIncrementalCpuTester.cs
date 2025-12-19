using PerfectNumbers.Core.Gpu.Accelerators;

namespace PerfectNumbers.Core.Cpu;

public class MersenneNumberIncrementalCpuTester(GpuKernelType kernelType)
{
	private static readonly MersenneDivisorCyclesCpu _cycles = MersenneDivisorCyclesCpu.Shared;
	private readonly GpuKernelType _kernelType = kernelType;

	public void Scan(PrimeOrderCalculatorAccelerator gpu, ulong exponent, UInt128 twoP, LastDigit lastDigit, UInt128 maxK, ref bool isPrime)
	{
		UInt128 k = UInt128.One;
		UInt128 one = k;
		ulong divMul = (ulong)((((UInt128)1 << 64) - UInt128.One) / exponent) + 1UL;
		// Use residue automaton always on CPU
		var auto = new MersenneResidueAutomaton(exponent);

		bool reject;
		bool shouldCheck;
		UInt128 q = auto.CurrentQ();
		UInt128 divPow;
		UInt128 halfPow;
		UInt128 phi;
		UInt128 qCycle;

		ulong div;
		ulong mod8;
		ulong phi64;

		while (k <= maxK && Volatile.Read(ref isPrime))
		{
			// TODO: Replace this direct GetCycle call with DivisorCycleCache.Lookup so we reuse the single snapshot block
			// and compute missing cycles on the configured device without queuing additional block generation.
			qCycle = MersenneDivisorCyclesCpu.GetCycle(q);
			shouldCheck = auto.Mod10R != 5UL;
			if (shouldCheck)
			{
				mod8 = auto.Mod8R;
				if (mod8 != 1UL && mod8 != 7UL)
				{
					shouldCheck = false;
				}
				else if (auto.Mod3R == 0UL || auto.Mod5R == 0UL)
				{
					// TODO: Swap these Mod3/Mod5 zero checks to the cached residue tables once the automaton exposes
					// the benchmarked lookup-based helpers so CPU scans avoid runtime modulo instructions.
					shouldCheck = false;
				}
			}

			if (shouldCheck)
			{
				if (_kernelType == GpuKernelType.Pow2Mod)
				{
					reject = exponent.PowModWithCycle(q, qCycle) == UInt128.One;
				}
				else
				{
					phi = q - one;
					reject = false;
					if (phi <= ulong.MaxValue)
					{
						phi64 = (ulong)phi;
						if (phi64.PowModWithCycle(q, qCycle) == one)
						{
							halfPow = (phi64 >> 1).PowModWithCycle(q, qCycle) - one;
							if (halfPow.BinaryGcd(q) == one)
							{
								div = phi64.FastDiv64(exponent, divMul);
								divPow = div.PowModWithCycle(q, qCycle) - one;
								if (divPow.BinaryGcd(q) == one)
								{
									reject = true;
								}
							}
						}
					}
				}

				if (reject && q.IsPrimeCandidate())
				{
					Volatile.Write(ref isPrime, false);
					break;
				}
			}

			k += 1UL;
			auto.Next();
			q = auto.CurrentQ();
		}
	}
}
