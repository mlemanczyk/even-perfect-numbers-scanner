using System.Numerics;
using PerfectNumbers.Core.Gpu.Accelerators;

namespace PerfectNumbers.Core;

public interface IMersenneNumberDivisorByDivisorTester
{
    int BatchSize { get; set; }

    BigInteger MinK { get; set; }

    void ConfigureFromMaxPrime(ulong maxPrime);

    ulong DivisorLimit { get; }

    ulong GetAllowedMaxDivisor(ulong prime);

    bool IsPrime(PrimeOrderCalculatorAccelerator gpu, ulong prime, out bool divisorsExhausted, out BigInteger divisor);

    IDivisorScanSession CreateDivisorSession(PrimeOrderCalculatorAccelerator gpu);

    string? StateFilePath { get; set; }

    void ResetStateTracking();

    void ResumeFromState(BigInteger lastSavedK);

    void PrepareCandidates(in ReadOnlySpan<ulong> primes, Span<ulong> allowedMaxValues);

    public interface IDivisorScanSession
    {
        bool CheckDivisor(ulong divisor, in MontgomeryDivisorData divisorData, ulong divisorCycle, in ReadOnlySpan<ulong> primes);

        void Return();
    }
}
