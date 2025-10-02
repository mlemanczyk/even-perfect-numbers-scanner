namespace PerfectNumbers.Core;

public static class PerfectNumberConstants
{
    public const ulong BiggestKnownEvenPerfectP = 136279841UL;
    public const int ConsoleInterval = 100_000;
    public const ulong ExtraDivisorCycleSearchLimit = 64UL;
    public const int MaxQForDivisorCycles = 4_000_000;
    public static readonly uint PrimesLimit = 1_000_000; //(ulong)Array.MaxLength;// 1_000_000;
    // TODO: Load these limits from the benchmark-driven configuration so CPU and GPU scans stay aligned with the optimal
    // divisor-cycle datasets we generate offline.
    // TODO: Promote these magic numbers into a runtime profile derived from EvenPerfectBitScanner.Benchmarks so we can retune
    // them automatically when the optimal ranges for 2kp+1 >= 138M change.
}
