namespace PerfectNumbers.Core
{
    public readonly struct FactorEntry(ulong value, int exponent, bool isPrime)
    {
        public readonly ulong Value = value;
		public readonly int Exponent = exponent;
		public readonly bool CofactorIsPrime = isPrime;
    }
}