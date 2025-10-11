namespace PerfectNumbers.Core
{
    public readonly struct FactorEntry
	{
		public readonly ulong Value;
		public readonly int Exponent = 1;

		public FactorEntry(ulong value, int exponent)
		{
			Value = value;
			Exponent = exponent;
		}

		public FactorEntry(ulong value)
		{
			Value = value;
		}
	}
}