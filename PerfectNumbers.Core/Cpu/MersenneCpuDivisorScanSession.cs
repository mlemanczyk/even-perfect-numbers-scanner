using System.Runtime.CompilerServices;
using PerfectNumbers.Core.Gpu.Accelerators;

namespace PerfectNumbers.Core.Cpu;

internal sealed class MersenneCpuDivisorScanSession(PrimeOrderCalculatorAccelerator gpu, ComputationDevice orderDevice) : IMersenneNumberDivisorByDivisorTester.IDivisorScanSession
{
	private Func<ulong, MontgomeryDivisorData, ulong> _getCycleLength = orderDevice switch
	{
		ComputationDevice.Gpu => (divisor, divisorData) => DivisorCycleCache.Shared.GetCycleLengthGpu(gpu, divisor, divisorData),
		ComputationDevice.Hybrid => (divisor, divisorData) => DivisorCycleCache.Shared.GetCycleLengthHybrid(gpu, divisor, divisorData),
		_ => static (divisor, divisorData) => DivisorCycleCache.Shared.GetCycleLengthCpu(divisor, divisorData),
	};

	public void Configure(PrimeOrderCalculatorAccelerator gpu, ComputationDevice orderDevice)
	{
		_getCycleLength = orderDevice switch
		{
			ComputationDevice.Gpu => (divisor, divisorData) => DivisorCycleCache.Shared.GetCycleLengthGpu(gpu, divisor, divisorData),
			ComputationDevice.Hybrid => (divisor, divisorData) => DivisorCycleCache.Shared.GetCycleLengthHybrid(gpu, divisor, divisorData),
			_ => static (divisor, divisorData) => DivisorCycleCache.Shared.GetCycleLengthCpu(divisor, divisorData),
		};
	}

    public bool CheckDivisor(
        ulong divisor,
        in MontgomeryDivisorData divisorData,
        ulong divisorCycle,
        in ReadOnlySpan<ulong> primes)
    {
        int length = primes.Length;
        // EvenPerfectBitScanner always provides at least one exponent on this path.
        // Keep the guard commented out for tests that might reuse the session differently.
        // if (length == 0)
        // {
        //     return false;
        // }

		// This never kicks off in production code
        // if (divisorData.Modulus != divisor)
        // {
		// 	throw new InvalidOperationException("Divisor data is for a different modulus");
        //     // cachedData = MontgomeryDivisorData.FromModulus(divisor);
        // }

		if (divisorCycle == 0UL)
		{
			divisorCycle = _getCycleLength(divisor, divisorData);
			// This never kicks off in production code
			// if (divisorCycle == 0UL)
			// {
			// 	// DivisorCycleCache guarantees a positive cycle for divisors greater than one.
			// 	throw new InvalidOperationException($"Divisor cycle solver returned zero for divisor {divisor}.");
			// }
		}

        // Keep these remainder steppers in place so future updates continue reusing the previously computed residues.
        // They are critical for avoiding repeated full Montgomery exponentiation work when scanning divisors.
        var cycleStepper = new CycleRemainderStepper(divisorCycle);
        if (cycleStepper.Initialize(primes[0]) == 0UL)
        {
            return true;
        }

        var exponentStepper = new ExponentRemainderStepperCpu(divisorData, divisorCycle);
        if (exponentStepper.InitializeCpuIsUnity(primes[0]))
        {
            return true;
        }

        for (int i = 1; i < length; i++)
        {
            if (cycleStepper.ComputeNext(primes[i]) != 0UL)
            {
                continue;
            }

            if (exponentStepper.ComputeNextIsUnity(primes[i]))
            {
                return true;
            }
        }

        return false;
    }

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public void Return() => ThreadStaticPools.ReturnMersenneCpuDivisorSession(this);
}
