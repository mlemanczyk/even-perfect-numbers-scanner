using System;

namespace PerfectNumbers.Core;

public interface IMersenneNumberDivisorByDivisorTester
{
        bool UseDivisorCycles { get; set; } // TODO: Remove the setter once divisor cycle usage becomes mandatory so all implementations always leverage the cached cycles.

        int BatchSize { get; set; }

        void ConfigureFromMaxPrime(ulong maxPrime);

        ulong DivisorLimit { get; }

        ulong GetAllowedMaxDivisor(ulong prime);

        bool IsPrime(ulong prime, out bool divisorsExhausted);

        IDivisorScanSession CreateDivisorSession();

        void PrepareCandidates(ReadOnlySpan<ulong> primes, Span<ulong> allowedMaxValues);

        public interface IDivisorScanSession : IDisposable
        {
                void CheckDivisor(ulong divisor, ulong divisorCycle, ReadOnlySpan<ulong> primes, Span<byte> hits);
        }
}

