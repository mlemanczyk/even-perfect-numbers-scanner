namespace PerfectNumbers.Core;

public static class PerfectNumberConstants
{
	public const ulong BiggestKnownEvenPerfectP = 136279841UL;
	public static readonly uint PrimesLimit = 1_000_000; //(ulong)Array.MaxLength;// 1_000_000;

	// Maximum K for residue-based Mersenne test (q = 2*p*k + 1 up to this k)
	public static readonly UInt128 ResidueMaxK = 5_000_000UL;
    public const int MaxQForDivisorCycles = 4_000_000;
}