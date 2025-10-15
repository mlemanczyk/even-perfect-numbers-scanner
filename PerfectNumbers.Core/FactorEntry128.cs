namespace PerfectNumbers.Core;

public readonly struct FactorEntry128(in UInt128 value, int exponent)
{
	public readonly UInt128 Value = value;

	public readonly int Exponent = exponent;
}
