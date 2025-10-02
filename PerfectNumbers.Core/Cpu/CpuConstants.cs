namespace PerfectNumbers.Core.Cpu;

public static class CpuConstants
{
    // Allowed q mod 10 sets depending on last digit of M_p
    public const int LastSevenMask10 = (1 << 7) | (1 << 9);
    // For M_p ending with 1, restrict to {1,3,9}; 7 considered only for q == 7 (not possible for large p)
    public const int LastOneMask10 = (1 << 1) | (1 << 3) | (1 << 9);
    // TODO: Generate these masks at startup from the benchmarked Mod10 automaton tables so CPU filtering stays aligned with
    // the optimized divisor-cycle residue steps.
    // TODO: Fold in the Mod6ComparisonBenchmarks stride table here so callers can merge the mod 6 skips with these masks and
    // avoid redundant residue work in the CPU hot loop.
}
