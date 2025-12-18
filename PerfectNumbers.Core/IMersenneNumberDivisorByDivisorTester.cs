using System.Numerics;
using PerfectNumbers.Core.Gpu.Accelerators;

namespace PerfectNumbers.Core;

public interface IMersenneNumberDivisorByDivisorTester
{
    bool IsPrime(ulong prime, out bool divisorsExhausted, out BigInteger divisor);

    IDivisorScanSession CreateDivisorSession(PrimeOrderCalculatorAccelerator gpu);

    void ResetStateTracking();

    void ResumeFromState(in string stateFile, in BigInteger lastSavedK, in BigInteger minK);

    void PrepareCandidates(ulong maxPrime, in ReadOnlySpan<ulong> primes, Span<ulong> allowedMaxValues);

    public interface IDivisorScanSession
    {
        bool CheckDivisor(ulong divisor, in MontgomeryDivisorData divisorData, ulong divisorCycle, in ReadOnlySpan<ulong> primes);

        // void Return();
    }
}
