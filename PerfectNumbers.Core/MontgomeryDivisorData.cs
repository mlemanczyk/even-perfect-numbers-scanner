namespace PerfectNumbers.Core;

public sealed class MontgomeryDivisorData
{
	public static readonly MontgomeryDivisorData Empty = new();
    public ulong Modulus;
	public ulong NPrime;
	public ulong MontgomeryOne;
	public ulong MontgomeryTwo;
	public ulong MontgomeryTwoSquared;

	public MontgomeryDivisorData()
	{	
	}

	public MontgomeryDivisorData(ulong modulus, ulong nPrime, ulong montgomeryOne, ulong montgomeryTwo, ulong montgomeryTwoSquared)
	{
		Modulus = modulus;
		NPrime = nPrime;
		MontgomeryOne = montgomeryOne;
		MontgomeryTwo = montgomeryTwo;
		MontgomeryTwoSquared = montgomeryTwoSquared;
	}
}
