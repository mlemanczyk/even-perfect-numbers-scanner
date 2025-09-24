using System;

namespace PerfectNumbers.Core;

public interface IMersenneNumberDivisorByDivisorTester
{
        bool UseDivisorCycles { get; set; }

        int BatchSize { get; set; }

        void ConfigureFromMaxPrime(ulong maxPrime);

        ulong DivisorLimit { get; }

        ulong GetAllowedMaxDivisor(ulong prime);

        bool IsPrime(ulong prime, out bool divisorsExhausted);

        IDivisorScanSession CreateDivisorSession();

        public interface IDivisorScanSession : IDisposable
        {
                void CheckDivisor(ulong divisor, ulong divisorCycle, ReadOnlySpan<ulong> primes, Span<byte> hits);
        }
}

