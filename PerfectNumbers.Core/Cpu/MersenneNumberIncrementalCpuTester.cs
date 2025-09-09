namespace PerfectNumbers.Core.Cpu;

public class MersenneNumberIncrementalCpuTester(GpuKernelType kernelType)
{
	private static readonly MersenneDivisorCycles _cycles = MersenneDivisorCycles.Shared;
	private readonly GpuKernelType _kernelType = kernelType;

	public void Scan(ulong exponent, UInt128 twoP, bool lastIsSeven, UInt128 maxK, ref bool isPrime)
	{
		UInt128 k, one = k = UInt128.One;
		ulong divMul = (ulong)((((UInt128)1 << 64) - UInt128.One) / exponent) + 1UL;
		// Use residue automaton always on CPU
		var auto = new MersenneResidueAutomaton(exponent);

		bool reject, shouldCheck;
		UInt128 q = auto.CurrentQ(),
				divPow, halfPow, phi, qCycle;

		ulong div, mod8, phi64;

		while (k <= maxK && Volatile.Read(ref isPrime))
		{
			qCycle = MersenneDivisorCycles.GetCycle(q);
			shouldCheck = lastIsSeven
				? (auto.Mod10R != 5UL)
				: (auto.Mod10R != 5UL);
			if (shouldCheck)
			{
				mod8 = auto.Mod8R;
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

		return;
	}
}