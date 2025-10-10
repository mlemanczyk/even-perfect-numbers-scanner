namespace PerfectNumbers.Core
{
    public readonly struct FactorEntry(ulong value, int exponent)
    {
        public readonly ulong Value = value;
        public readonly int Exponent = exponent;
    }
}