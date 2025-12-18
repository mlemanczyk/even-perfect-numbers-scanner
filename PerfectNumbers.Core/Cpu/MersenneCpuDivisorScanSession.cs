using System.Runtime.CompilerServices;
using PerfectNumbers.Core.Gpu.Accelerators;

namespace PerfectNumbers.Core.Cpu;

internal struct MersenneCpuDivisorScanSessionWithCpuOrder() : IMersenneNumberDivisorByDivisorTester.IDivisorScanSession
	{
	private readonly CycleRemainderStepper _cycleStepper = new(); 
	private ExponentRemainderStepperCpu? _exponentStepper;

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public readonly void Configure()
	{
	}

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

		if (divisorCycle == 0UL)
		{
			divisorCycle = DivisorCycleCache.Shared.GetCycleLengthCpu(divisor, divisorData);

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

	// [MethodImpl(MethodImplOptions.AggressiveInlining)]
	// public void Return() => ThreadStaticPools.ReturnMersenneCpuDivisorSession(this);
}

internal struct MersenneCpuDivisorScanSessionWithHybridOrder() : IMersenneNumberDivisorByDivisorTester.IDivisorScanSession
{
	private PrimeOrderCalculatorAccelerator? _gpu;

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public void Configure(PrimeOrderCalculatorAccelerator gpu)
	{
		_gpu = gpu;
	}

    public bool CheckDivisor(
        ulong divisor,
        in MontgomeryDivisorData divisorData,
        ulong divisorCycle,
        in ReadOnlySpan<ulong> primes)
    {
        int length = primes.Length;
		var gpu = _gpu ?? throw new InvalidOperationException("Session was not configured with an accelerator.");
		if (divisorCycle == 0UL)
		{
			divisorCycle = DivisorCycleCache.Shared.GetCycleLengthHybrid(gpu, divisor, divisorData);
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

	// [MethodImpl(MethodImplOptions.AggressiveInlining)]
	// public void Return() => ThreadStaticPools.ReturnMersenneCpuDivisorSession(this);
}

internal struct MersenneCpuDivisorScanSessionWithGpuOrder() : IMersenneNumberDivisorByDivisorTester.IDivisorScanSession
{
	private PrimeOrderCalculatorAccelerator? _gpu;

	public void Configure(PrimeOrderCalculatorAccelerator gpu)
	{
		_gpu = gpu;
	}

    public bool CheckDivisor(
        ulong divisor,
        in MontgomeryDivisorData divisorData,
        ulong divisorCycle,
        in ReadOnlySpan<ulong> primes)
    {
        int length = primes.Length;
		var gpu = _gpu ?? throw new InvalidOperationException("Session was not configured with an accelerator.");
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
			divisorCycle = DivisorCycleCache.Shared.GetCycleLengthGpu(gpu, divisor, divisorData);

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

	// [MethodImpl(MethodImplOptions.AggressiveInlining)]
	// public void Return() => ThreadStaticPools.ReturnMersenneCpuDivisorSession(this);
}
