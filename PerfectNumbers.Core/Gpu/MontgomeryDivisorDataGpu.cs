namespace PerfectNumbers.Core.Gpu;

public readonly struct MontgomeryDivisorDataGpu(ulong modulus, ulong nPrime, ulong montgomeryOne, ulong montgomeryTwo, ulong montgomeryTwoSquared)
{
    public readonly ulong Modulus = modulus;
	public readonly ulong NPrime = nPrime;
	public readonly ulong MontgomeryOne = montgomeryOne;
	public readonly ulong MontgomeryTwo = montgomeryTwo;
	public readonly ulong MontgomeryTwoSquared = montgomeryTwoSquared;
}
	