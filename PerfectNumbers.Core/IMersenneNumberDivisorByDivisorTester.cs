using PerfectNumbers.Core.Gpu.Accelerators;

namespace PerfectNumbers.Core;

public interface IMersenneNumberDivisorByDivisorTester
{
    int BatchSize { get; set; }

    void ConfigureFromMaxPrime(ulong maxPrime);

    ulong DivisorLimit { get; }

    ulong GetAllowedMaxDivisor(ulong prime);

    bool IsPrime(PrimeOrderCalculatorAccelerator gpu, ulong prime, out bool divisorsExhausted, out ulong divisor);

    IDivisorScanSession CreateDivisorSession(PrimeOrderCalculatorAccelerator gpu);

    void PrepareCandidates(in ReadOnlySpan<ulong> primes, Span<ulong> allowedMaxValues);

    public interface IDivisorScanSession
    {
        void CheckDivisor(ulong divisor, in MontgomeryDivisorData divisorData, ulong divisorCycle, in ReadOnlySpan<ulong> primes, Span<byte> hits);

        void Return();
    }
}
