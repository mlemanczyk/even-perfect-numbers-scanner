using System;

namespace PerfectNumbers.Core;

public interface IMersenneNumberDivisorByDivisorTester
{
    int BatchSize { get; set; }

    void ConfigureFromMaxPrime(ulong maxPrime);

    ulong DivisorLimit { get; }

    ulong GetAllowedMaxDivisor(ulong prime);

    bool IsPrime(ulong prime, out bool divisorsExhausted, TimeSpan? timeLimit = null);

    IDivisorScanSession CreateDivisorSession();

    void PrepareCandidates(ReadOnlySpan<ulong> primes, Span<ulong> allowedMaxValues);

    public interface IDivisorScanSession
    {
        void CheckDivisor(ulong divisor, in MontgomeryDivisorData divisorData, ulong divisorCycle, ReadOnlySpan<ulong> primes, Span<byte> hits);

        void Dispose();
    }
}
