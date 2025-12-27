using System.Numerics;

namespace PerfectNumbers.Core
{
    public static class BigIntegerNumbers
	{
		public static readonly BigInteger OneShiftedLeft63 = BigInteger.One << 63;
		public static readonly BigInteger OneShiftedLeft256MinusOne = (BigInteger.One << 256) - BigInteger.One;
		public static readonly BigInteger Two = (BigInteger)2;
		public static readonly BigInteger UlongMaxValue = (BigInteger)ulong.MaxValue; 
		public static readonly BigInteger UlongMaxValuePlusOne = UlongMaxValue + BigInteger.One;
		public static readonly BigInteger UInt128MaxValue = (BigInteger)UInt128.MaxValue;
		public static readonly BigInteger UInt128MaxValuePlusOne = UInt128MaxValue + BigInteger.One;
	}
}