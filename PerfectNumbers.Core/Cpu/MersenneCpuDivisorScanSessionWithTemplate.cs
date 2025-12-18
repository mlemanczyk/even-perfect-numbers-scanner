using System.Runtime.CompilerServices;
using PerfectNumbers.Core.Gpu.Accelerators;

namespace PerfectNumbers.Core.Cpu;

[DeviceDependentTemplate(typeof(ComputationDevice), suffix: "Order")]
internal struct MersenneCpuDivisorScanSessionWithTemplate() : IMersenneNumberDivisorByDivisorTester.IDivisorScanSession
{
#if DEVICE_GPU || DEVICE_HYBRID
	private PrimeOrderCalculatorAccelerator? _gpu;
#endif
	private readonly CycleRemainderStepper _cycleStepper = new();
	private ExponentRemainderStepperCpu? _exponentStepper;

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#if DEVICE_GPU || DEVICE_HYBRID
	public void Configure(PrimeOrderCalculatorAccelerator gpu)
	{
		_gpu = gpu;
	}
#else
	public readonly void Configure()
	{
	}
#endif

	private static ulong _hits;
	private static ulong _misses;

	private static void PrintStats(bool wasHit)
	{
		if (wasHit)
		{
			Interlocked.Increment(ref _hits);
		}
		else
		{
			Interlocked.Increment(ref _misses);
		}

		Console.WriteLine($"MersenneCpuDivisorScanSession, ExponentStepper hits: $_hits, misses: $_misses");
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
#if DEVICE_GPU || DEVICE_HYBRID
		var gpu = _gpu;
#endif

		if (divisorCycle == 0UL)
		{
#if DEVICE_CPU
			divisorCycle = DivisorCycleCache.Shared.GetCycleLengthCpu(divisor, divisorData);
#elif DEVICE_HYBRID
			divisorCycle = DivisorCycleCache.Shared.GetCycleLengthHybrid(gpu, divisor, divisorData);
#elif DEVICE_GPU
			divisorCycle = DivisorCycleCache.Shared.GetCycleLengthGpu(gpu, divisor, divisorData);
#endif
			// This never kicks off in production code
			// if (divisorCycle == 0UL)
			// {	
			// 	// DivisorCycleCache guarantees a positive cycle for divisors greater than one.
			// 	throw new InvalidOperationException($"Divisor cycle solver returned zero for divisor {divisor}.");
			// }
		}

		// Keep these remainder steppers in place so future updates continue reusing the previously computed residues.
		// They are critical for avoiding repeated full Montgomery exponentiation work when scanning divisors.
		CycleRemainderStepper cycleStepper = _cycleStepper;
		cycleStepper.CycleLength = divisorCycle;
		if (cycleStepper.Initialize(primes[0]) == 0UL)
		{
			return true;
		}

		if (_exponentStepper is not { } exponentStepper)
		{
			_exponentStepper = exponentStepper = new(divisorData, divisorCycle);
		}
		else
		{
			if (!exponentStepper.MatchesDivisor(divisorData, divisorCycle))
			{
				_exponentStepper = exponentStepper = new(divisorData, divisorCycle);
			}
		}

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
}
