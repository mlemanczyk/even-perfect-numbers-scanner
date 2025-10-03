namespace PerfectNumbers.Core
{
    public static class UInt128Numbers
    {
        public static readonly UInt128 Two = 2UL;
        public static readonly UInt128 Four = 4UL;
        public static readonly UInt128 SixtyFour = 64UL;
        public static readonly UInt128 OneHundredTwentySeven = 127UL;
        public static readonly UInt128 OneShiftedLeft64 = UInt128.One << 64;
        public static readonly UInt128 OneShiftedLeft64x2 = Two * OneShiftedLeft64;

        // TODO: Populate this table from the UInt128 residue benchmarks so GPU kernels can pull precomputed constants
        // without re-deriving them at runtime.
        // TODO: Extend the table with cached Mod3/Mod5 folding constants so both CPU and GPU residue helpers can reuse
        // the benchmarked multiply-high reductions without recomputing them per call.
    }
}

