namespace PerfectNumbers.Core.Cpu;

internal sealed class MersenneCpuDivisorScanSession : IMersenneNumberDivisorByDivisorTester.IDivisorScanSession
{
    private bool _disposed;

    public void Reset()
    {
        _disposed = false;
    }

    public void CheckDivisor(
        ulong divisor,
        in MontgomeryDivisorData divisorData,
        ulong divisorCycle,
        in ReadOnlySpan<ulong> primes,
        Span<byte> hits)
    {
        int length = primes.Length;
        // EvenPerfectBitScanner always provides at least one exponent on this path.
        // Keep the guard commented out for tests that might reuse the session differently.
        // if (length == 0)
        // {
        //     return;
        // }

        MontgomeryDivisorData cachedData = divisorData;
        if (cachedData.Modulus != divisor)
        {
            cachedData = MontgomeryDivisorData.FromModulus(divisor);
        }

        if (divisorCycle == 0UL)
        {
            divisorCycle = DivisorCycleCache.Shared.GetCycleLength(divisor);
            if (divisorCycle == 0UL)
            {
                hits.Clear();
                return;
            }
        }

        // Keep these remainder steppers in place so future updates continue reusing the previously computed residues.
        // They are critical for avoiding repeated full Montgomery exponentiation work when scanning divisors.
        var exponentStepper = new ExponentRemainderStepper(cachedData);
        if (!exponentStepper.IsValidModulus)
        {
            hits.Clear();
            return;
        }

        var cycleStepper = new CycleRemainderStepper(divisorCycle);

        ulong remainder = cycleStepper.Initialize(primes[0]);
        hits[0] = remainder == 0UL
            ? (exponentStepper.ComputeNextIsUnity(primes[0]) ? (byte)1 : (byte)0)
            : (byte)0;

        for (int i = 1; i < length; i++)
        {
            remainder = cycleStepper.ComputeNext(primes[i]);
            if (remainder != 0UL)
            {
                hits[i] = 0;
                continue;
            }

            hits[i] = exponentStepper.ComputeNextIsUnity(primes[i]) ? (byte)1 : (byte)0;
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(MersenneCpuDivisorScanSession), "Divisor scan session disposed twice.");
        }

        _disposed = true;
        ThreadStaticPools.ReturnMersenneCpuDivisorSession(this);
    }
}
