using System.Numerics;
using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks
{
	public class Pow2ProviderBigIntegerBenchmarks
	{
		[Benchmark(Baseline = true)]
		public BigInteger ByShifting()
		{
			BigInteger result = BigInteger.Zero;
			for (int i = 0; i <= 63; i++)
			{
				result = BigInteger.One << i;
			}

			return result;
		}

		[Benchmark]
		public BigInteger ByMapping()
		{
			BigInteger result = BigInteger.Zero;
			for (int i = 0; i <= 63; i++)
			{
				result = Pow2Provider.BigInteger(i);
			}

			return result;
		}
	}

	public class Pow2MinusOneeProviderBigIntegerBenchmarks
	{
		[Benchmark(Baseline = true)]
		public BigInteger ByShifting()
		{
			BigInteger result = BigInteger.Zero;
			for (int i = 0; i <= 63; i++)
			{
				result = (BigInteger.One << i) - BigInteger.One;
			}

			return result;
		}

		[Benchmark]
		public BigInteger ByMapping()
		{
			BigInteger result = BigInteger.Zero;
			for (int i = 0; i <= 63; i++)
			{
				result = Pow2Provider.BigIntegerMinusOne(i);
			}

			return result;
		}
	}

	public class Pow2ProviderULongBenchmarks
	{
		public static class ULongNumbers
		{
			public const ulong TwoMinusOne = 2L - 1L;
			public const ulong FourMinusOne = 4L - 1L;
			public const ulong EightMinusOne = 8L - 1L;
			public const ulong SixteenMinusOne = 16L - 1L;
			public const ulong ThirtyTwoMinusOne = 32L - 1L;
			public const ulong SixtyFourMinusOne = 64L - 1L;
			public const ulong OneHundredTwentyEightMinusOne = 128L - 1L;
			public const ulong TwoHundredFiftySixMinusOne = 256L - 1L;
			public const ulong FiveHundredTwelveMinusOne = 512L - 1L;
			public const ulong OneThousandTwentyFourMinusOne = 1_024L - 1L;
			public const ulong TwoThousandFortyEightMinusOne = 2_048L - 1L;
			public const ulong FourThousandNinetySixMinusOne = 4_096L - 1L;
			public const ulong EightThousandOneHundredNinetyTwoMinusOne = 8_192L - 1L;
			public const ulong SixteenThousandThreeHundredEightyFourMinusOne = 16_384L - 1L;
			public const ulong ThirtyTwoThousandSevenHundredSixtyEightMinusOne = 32_768L - 1L;
			public const ulong SixtyFiveThousandFiveHundredThirtySixMinusOne = 65_536L - 1L;
			public const ulong OneHundredThirtyOneThousandSeventyTwoMinusOne = 131_072L - 1L;
			public const ulong TwoHundredSixtyTwoThousandOneHundredFortyFourMinusOne = 262_144L - 1L;
			public const ulong FiveHundredTwentyFourThousandTwoHundredEightyEightMinusOne = 524_288L - 1L;
			public const ulong OneMillionFortyEightThousandFiveHundredSeventySixMinusOne = 1_048_576L - 1L;
			public const ulong TwoMillionNinetySevenThousandOneHundredFiftyTwoMinusOne = 2_097_152L - 1L;
			public const ulong FourMillionOneHundredNinetyFourThousandThreeHundredFourMinusOne = 4_194_304L - 1L;
			public const ulong EightMillionThreeHundredEightyEightThousandSixHundredEightMinusOne = 8_388_608L - 1L;
			public const ulong SixteenMillionSevenHundredSeventySevenThousandTwoHundredSixteenMinusOne = 16_777_216L - 1L;
			public const ulong ThirtyThreeMillionFiveHundredFiftyFourThousandFourHundredThirtyTwoMinusOne = 33_554_432L - 1L;
			public const ulong SixtySevenMillionOneHundredEightThousandEightHundredSixtyFourMinusOne = 67_108_864L - 1L;
			public const ulong OneHundredThirtyFourMillionTwoHundredSeventeenThousandSevenHundredTwentyEightMinusOne = 134_217_728L - 1L;
			public const ulong TwoHundredSixtyEightMillionFourHundredThirtyFiveThousandFourHundredFiftySixMinusOne = 268_435_456L - 1L;
			public const ulong FiveHundredThirtySixMillionEightHundredSeventyThousandNineHundredTwelveMinusOne = 536_870_912L - 1L;
			public const ulong OneBillionSeventyThreeMillionSevenHundredFortyOneThousandEightHundredTwentyFourMinusOne = 1_073_741_824L - 1L;
			public const ulong TwoBillionOneHundredFortySevenMillionFourHundredEightyThreeThousandSixHundredFortyEightMinusOne = 2_147_483_648L - 1L;
			public const ulong FourBillionTwoHundredNinetyFourMillionNineHundredSixtySevenThousandTwoHundredNinetySixMinusOne = 4_294_967_296L - 1L;
			public const ulong EightBillionFiveHundredEightyNineMillionNineHundredThirtyFourThousandFiveHundredNinetyTwoMinusOne = 8_589_934_592L - 1L;
			public const ulong SeventeenBillionOneHundredSeventyNineMillionEightHundredSixtyNineThousandOneHundredEightyFourMinusOne = 17_179_869_184L - 1L;
			public const ulong ThirtyFourBillionThreeHundredFiftyNineMillionSevenHundredThirtyEightThousandThreeHundredSixtyEightMinusOne = 34_359_738_368L - 1L;
			public const ulong SixtyEightBillionSevenHundredNineteenMillionFourHundredSeventySixThousandSevenHundredThirtySixMinusOne = 68_719_476_736L - 1L;
			public const ulong OneHundredThirtySevenBillionFourHundredThirtyEightMillionNineHundredFiftyThreeThousandFourHundredSeventyTwoMinusOne = 137_438_953_472L - 1L;
			public const ulong TwoHundredSeventyFourBillionEightHundredSeventySevenMillionNineHundredSixThousandNineHundredFortyFourMinusOne = 274_877_906_944L - 1L;
			public const ulong FiveHundredFortyNineBillionSevenHundredFiftyFiveMillionEightHundredThirteenThousandEightHundredEightyEightMinusOne = 549_755_813_888L - 1L;
			public const ulong OneTrillionNinetyNineBillionFiveHundredElevenMillionSixHundredTwentySevenThousandSevenHundredSeventySixMinusOne = 1_099_511_627_776L - 1L;
			public const ulong TwoTrillionOneHundredNinetyNineBillionTwentyThreeMillionTwoHundredFiftyFiveThousandFiveHundredFiftyTwoMinusOne = 2_199_023_255_552L - 1L;
			public const ulong FourTrillionThreeHundredNinetyEightBillionFortySixMillionFiveHundredElevenThousandOneHundredFourMinusOne = 4_398_046_511_104L - 1L;
			public const ulong EightTrillionSevenHundredNinetySixBillionNinetyThreeMillionTwentyTwoThousandTwoHundredEightMinusOne = 8_796_093_022_208L - 1L;
			public const ulong SeventeenTrillionFiveHundredNinetyTwoBillionOneHundredEightySixMillionFortyFourThousandFourHundredSixteenMinusOne = 17_592_186_044_416L - 1L;
			public const ulong ThirtyFiveTrillionOneHundredEightyFourBillionThreeHundredSeventyTwoMillionEightyEightThousandEightHundredThirtyTwoMinusOne = 35_184_372_088_832L - 1L;
			public const ulong SeventyTrillionThreeHundredSixtyEightBillionSevenHundredFortyFourMillionOneHundredSeventySevenThousandSixHundredSixtyFourMinusOne = 70_368_744_177_664L - 1L;
			public const ulong OneHundredFortyTrillionSevenHundredThirtySevenBillionFourHundredEightyEightMillionThreeHundredFiftyFiveThousandThreeHundredTwentyEightMinusOne = 140_737_488_355_328L - 1L;
			public const ulong TwoHundredEightyOneTrillionFourHundredSeventyFourBillionNineHundredSeventySixMillionSevenHundredTenThousandSixHundredFiftySixMinusOne = 281_474_976_710_656L - 1L;
			public const ulong FiveHundredSixtyTwoTrillionNineHundredFortyNineBillionNineHundredFiftyThreeMillionFourHundredTwentyOneThousandThreeHundredTwelveMinusOne = 562_949_953_421_312L - 1L;
			public const ulong OneQuadrillionOneHundredTwentyFiveTrillionEightHundredNinetyNineBillionNineHundredSixMillionEightHundredFortyTwoThousandSixHundredTwentyFourMinusOne = 1_125_899_906_842_624L - 1L;
			public const ulong TwoQuadrillionTwoHundredFiftyOneTrillionSevenHundredNinetyNineBillionEightHundredThirteenMillionSixHundredEightyFiveThousandTwoHundredFortyEightMinusOne = 2_251_799_813_685_248L - 1L;
			public const ulong FourQuadrillionFiveHundredThreeTrillionFiveHundredNinetyNineBillionSixHundredTwentySevenMillionThreeHundredSeventyThousandFourHundredNinetySixMinusOne = 4_503_599_627_370_496L - 1L;
			public const ulong NineQuadrillionSevenTrillionOneHundredNinetyNineBillionTwoHundredFiftyFourMillionSevenHundredFortyThousandNineHundredNinetyTwoMinusOne = 9_007_199_254_740_992L - 1L;
			public const ulong EighteenQuadrillionFourteenTrillionThreeHundredNinetyEightBillionFiveHundredEightMillionEightHundredEightyThousandOneHundredEightyFourMinusOne = 18_014_398_509_481_984L - 1L;
			public const ulong ThirtySixQuadrillionTwentyEightTrillionSevenHundredNinetySevenBillionSeventeenMillionSevenHundredSixtyThousandThreeHundredSixtyEightMinusOne = 36_028_797_018_963_968L - 1L;
			public const ulong SeventyTwoQuadrillionFiftySevenTrillionFiveHundredNinetyFourBillionThirtyFiveMillionFiveHundredTwentyThousandSevenHundredThirtySixMinusOne = 72_057_594_037_927_936L - 1L;
			public const ulong OneHundredFortyFourQuadrillionOneHundredFifteenTrillionOneHundredEightyEightBillionSeventyOneMillionFortyOneThousandFourHundredSeventyTwoMinusOne = 144_115_188_075_855_872L - 1L;
			public const ulong TwoHundredEightyEightQuadrillionTwoHundredThirtyTrillionThreeHundredSeventySixBillionOneHundredFortyTwoMillionEightyTwoThousandNineHundredFortyFourMinusOne = 288_230_376_151_711_744L - 1L;
			public const ulong FiveHundredSeventySixQuadrillionFourHundredSixtyTrillionSevenHundredFiftyTwoBillionTwoHundredEightyFourMillionOneHundredSixtyFiveThousandEightHundredEightyEightMinusOne = 576_460_752_303_423_488L - 1L;
			public const ulong OneQuintillionOneHundredFiftyTwoQuadrillionNineHundredTwentyOneTrillionFiveHundredFourBillionSixHundredSixtyEightMillionThreeHundredThirtyOneThousandSevenHundredSeventySixMinusOne = 1_152_921_504_606_846_976L - 1L;
			public const ulong TwoQuintillionThreeHundredFiveQuadrillionEightHundredFortyThreeTrillionNineBillionThreeHundredThirtySixMillionSixHundredSixtyThreeThousandFiveHundredFiftyTwoMinusOne = 2_305_843_009_213_693_952L - 1L;
			public const ulong FourQuintillionSixHundredElevenQuadrillionSixHundredEightySevenTrillionEightHundredSixteenBillionSixHundredSeventyThreeMillionTwoHundredTwentySevenThousandOneHundredFourMinusOne = 4_611_686_018_427_387_904L - 1L;
			public const ulong NineQuintillionTwoHundredTwentyThreeQuadrillionThreeHundredSeventyTwoTrillionThirtySixBillionEightHundredFiftyFourMillionSevenHundredSeventyFiveThousandEightHundredEightMinusOne = 9_223_372_036_854_775_808UL - 1UL;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		public static ulong ULongMinusOne(int i) => i switch
		{
			0 => 0UL,
			1 => ULongNumbers.TwoMinusOne,
			2 => ULongNumbers.FourMinusOne,
			3 => ULongNumbers.EightMinusOne,
			4 => ULongNumbers.SixteenMinusOne,
			5 => ULongNumbers.ThirtyTwoMinusOne,
			6 => ULongNumbers.SixtyFourMinusOne,
			7 => ULongNumbers.OneHundredTwentyEightMinusOne,
			8 => ULongNumbers.TwoHundredFiftySixMinusOne,
			9 => ULongNumbers.FiveHundredTwelveMinusOne,
			10 => ULongNumbers.OneThousandTwentyFourMinusOne,
			11 => ULongNumbers.TwoThousandFortyEightMinusOne,
			12 => ULongNumbers.FourThousandNinetySixMinusOne,
			13 => ULongNumbers.EightThousandOneHundredNinetyTwoMinusOne,
			14 => ULongNumbers.SixteenThousandThreeHundredEightyFourMinusOne,
			15 => ULongNumbers.ThirtyTwoThousandSevenHundredSixtyEightMinusOne,
			16 => ULongNumbers.SixtyFiveThousandFiveHundredThirtySixMinusOne,
			17 => ULongNumbers.OneHundredThirtyOneThousandSeventyTwoMinusOne,
			18 => ULongNumbers.TwoHundredSixtyTwoThousandOneHundredFortyFourMinusOne,
			19 => ULongNumbers.FiveHundredTwentyFourThousandTwoHundredEightyEightMinusOne,
			20 => ULongNumbers.OneMillionFortyEightThousandFiveHundredSeventySixMinusOne,
			21 => ULongNumbers.TwoMillionNinetySevenThousandOneHundredFiftyTwoMinusOne,
			22 => ULongNumbers.FourMillionOneHundredNinetyFourThousandThreeHundredFourMinusOne,
			23 => ULongNumbers.EightMillionThreeHundredEightyEightThousandSixHundredEightMinusOne,
			24 => ULongNumbers.SixteenMillionSevenHundredSeventySevenThousandTwoHundredSixteenMinusOne,
			25 => ULongNumbers.ThirtyThreeMillionFiveHundredFiftyFourThousandFourHundredThirtyTwoMinusOne,
			26 => ULongNumbers.SixtySevenMillionOneHundredEightThousandEightHundredSixtyFourMinusOne,
			27 => ULongNumbers.OneHundredThirtyFourMillionTwoHundredSeventeenThousandSevenHundredTwentyEightMinusOne,
			28 => ULongNumbers.TwoHundredSixtyEightMillionFourHundredThirtyFiveThousandFourHundredFiftySixMinusOne,
			29 => ULongNumbers.FiveHundredThirtySixMillionEightHundredSeventyThousandNineHundredTwelveMinusOne,
			30 => ULongNumbers.OneBillionSeventyThreeMillionSevenHundredFortyOneThousandEightHundredTwentyFourMinusOne,
			31 => ULongNumbers.TwoBillionOneHundredFortySevenMillionFourHundredEightyThreeThousandSixHundredFortyEightMinusOne,
			32 => ULongNumbers.FourBillionTwoHundredNinetyFourMillionNineHundredSixtySevenThousandTwoHundredNinetySixMinusOne,
			33 => ULongNumbers.EightBillionFiveHundredEightyNineMillionNineHundredThirtyFourThousandFiveHundredNinetyTwoMinusOne,
			34 => ULongNumbers.SeventeenBillionOneHundredSeventyNineMillionEightHundredSixtyNineThousandOneHundredEightyFourMinusOne,
			35 => ULongNumbers.ThirtyFourBillionThreeHundredFiftyNineMillionSevenHundredThirtyEightThousandThreeHundredSixtyEightMinusOne,
			36 => ULongNumbers.SixtyEightBillionSevenHundredNineteenMillionFourHundredSeventySixThousandSevenHundredThirtySixMinusOne,
			37 => ULongNumbers.OneHundredThirtySevenBillionFourHundredThirtyEightMillionNineHundredFiftyThreeThousandFourHundredSeventyTwoMinusOne,
			38 => ULongNumbers.TwoHundredSeventyFourBillionEightHundredSeventySevenMillionNineHundredSixThousandNineHundredFortyFourMinusOne,
			39 => ULongNumbers.FiveHundredFortyNineBillionSevenHundredFiftyFiveMillionEightHundredThirteenThousandEightHundredEightyEightMinusOne,
			40 => ULongNumbers.OneTrillionNinetyNineBillionFiveHundredElevenMillionSixHundredTwentySevenThousandSevenHundredSeventySixMinusOne,
			41 => ULongNumbers.TwoTrillionOneHundredNinetyNineBillionTwentyThreeMillionTwoHundredFiftyFiveThousandFiveHundredFiftyTwoMinusOne,
			42 => ULongNumbers.FourTrillionThreeHundredNinetyEightBillionFortySixMillionFiveHundredElevenThousandOneHundredFourMinusOne,
			43 => ULongNumbers.EightTrillionSevenHundredNinetySixBillionNinetyThreeMillionTwentyTwoThousandTwoHundredEightMinusOne,
			44 => ULongNumbers.SeventeenTrillionFiveHundredNinetyTwoBillionOneHundredEightySixMillionFortyFourThousandFourHundredSixteenMinusOne,
			45 => ULongNumbers.ThirtyFiveTrillionOneHundredEightyFourBillionThreeHundredSeventyTwoMillionEightyEightThousandEightHundredThirtyTwoMinusOne,
			46 => ULongNumbers.SeventyTrillionThreeHundredSixtyEightBillionSevenHundredFortyFourMillionOneHundredSeventySevenThousandSixHundredSixtyFourMinusOne,
			47 => ULongNumbers.OneHundredFortyTrillionSevenHundredThirtySevenBillionFourHundredEightyEightMillionThreeHundredFiftyFiveThousandThreeHundredTwentyEightMinusOne,
			48 => ULongNumbers.TwoHundredEightyOneTrillionFourHundredSeventyFourBillionNineHundredSeventySixMillionSevenHundredTenThousandSixHundredFiftySixMinusOne,
			49 => ULongNumbers.FiveHundredSixtyTwoTrillionNineHundredFortyNineBillionNineHundredFiftyThreeMillionFourHundredTwentyOneThousandThreeHundredTwelveMinusOne,
			50 => ULongNumbers.OneQuadrillionOneHundredTwentyFiveTrillionEightHundredNinetyNineBillionNineHundredSixMillionEightHundredFortyTwoThousandSixHundredTwentyFourMinusOne,
			51 => ULongNumbers.TwoQuadrillionTwoHundredFiftyOneTrillionSevenHundredNinetyNineBillionEightHundredThirteenMillionSixHundredEightyFiveThousandTwoHundredFortyEightMinusOne,
			52 => ULongNumbers.FourQuadrillionFiveHundredThreeTrillionFiveHundredNinetyNineBillionSixHundredTwentySevenMillionThreeHundredSeventyThousandFourHundredNinetySixMinusOne,
			53 => ULongNumbers.NineQuadrillionSevenTrillionOneHundredNinetyNineBillionTwoHundredFiftyFourMillionSevenHundredFortyThousandNineHundredNinetyTwoMinusOne,
			54 => ULongNumbers.EighteenQuadrillionFourteenTrillionThreeHundredNinetyEightBillionFiveHundredEightMillionEightHundredEightyThousandOneHundredEightyFourMinusOne,
			55 => ULongNumbers.ThirtySixQuadrillionTwentyEightTrillionSevenHundredNinetySevenBillionSeventeenMillionSevenHundredSixtyThousandThreeHundredSixtyEightMinusOne,
			56 => ULongNumbers.SeventyTwoQuadrillionFiftySevenTrillionFiveHundredNinetyFourBillionThirtyFiveMillionFiveHundredTwentyThousandSevenHundredThirtySixMinusOne,
			57 => ULongNumbers.OneHundredFortyFourQuadrillionOneHundredFifteenTrillionOneHundredEightyEightBillionSeventyOneMillionFortyOneThousandFourHundredSeventyTwoMinusOne,
			58 => ULongNumbers.TwoHundredEightyEightQuadrillionTwoHundredThirtyTrillionThreeHundredSeventySixBillionOneHundredFortyTwoMillionEightyTwoThousandNineHundredFortyFourMinusOne,
			59 => ULongNumbers.FiveHundredSeventySixQuadrillionFourHundredSixtyTrillionSevenHundredFiftyTwoBillionTwoHundredEightyFourMillionOneHundredSixtyFiveThousandEightHundredEightyEightMinusOne,
			60 => ULongNumbers.OneQuintillionOneHundredFiftyTwoQuadrillionNineHundredTwentyOneTrillionFiveHundredFourBillionSixHundredSixtyEightMillionThreeHundredThirtyOneThousandSevenHundredSeventySixMinusOne,
			61 => ULongNumbers.TwoQuintillionThreeHundredFiveQuadrillionEightHundredFortyThreeTrillionNineBillionThreeHundredThirtySixMillionSixHundredSixtyThreeThousandFiveHundredFiftyTwoMinusOne,
			62 => ULongNumbers.FourQuintillionSixHundredElevenQuadrillionSixHundredEightySevenTrillionEightHundredSixteenBillionSixHundredSeventyThreeMillionTwoHundredTwentySevenThousandOneHundredFourMinusOne,
			63 => ULongNumbers.NineQuintillionTwoHundredTwentyThreeQuadrillionThreeHundredSeventyTwoTrillionThirtySixBillionEightHundredFiftyFourMillionSevenHundredSeventyFiveThousandEightHundredEightMinusOne,
			_ => throw new ArgumentOutOfRangeException(nameof(i)),
		};

		[Benchmark(Baseline = true)]
		public ulong ByShifting()
		{
			ulong result = 0UL;
			for (int i = 0; i <= 63; i++)
			{
				result = (1UL << i) - 1UL;
			}

			return result;
		}

		[Benchmark]
		public ulong ByMapping()
		{
			ulong result = 0UL;
			for (int i = 0; i <= 63; i++)
			{
				result = ULongMinusOne(i);
			}

			return result;
		}
	}

	public class Pow2ProviderLongBenchmarks
	{
		public static class LongNumbers
		{
			public const long TwoMinusOne = 2L - 1L;
			public const long FourMinusOne = 4L - 1L;
			public const long EightMinusOne = 8L - 1L;
			public const long SixteenMinusOne = 16L - 1L;
			public const long ThirtyTwoMinusOne = 32L - 1L;
			public const long SixtyFourMinusOne = 64L - 1L;
			public const long OneHundredTwentyEightMinusOne = 128L - 1L;
			public const long TwoHundredFiftySixMinusOne = 256L - 1L;
			public const long FiveHundredTwelveMinusOne = 512L - 1L;
			public const long OneThousandTwentyFourMinusOne = 1_024L - 1L;
			public const long TwoThousandFortyEightMinusOne = 2_048L - 1L;
			public const long FourThousandNinetySixMinusOne = 4_096L - 1L;
			public const long EightThousandOneHundredNinetyTwoMinusOne = 8_192L - 1L;
			public const long SixteenThousandThreeHundredEightyFourMinusOne = 16_384L - 1L;
			public const long ThirtyTwoThousandSevenHundredSixtyEightMinusOne = 32_768L - 1L;
			public const long SixtyFiveThousandFiveHundredThirtySixMinusOne = 65_536L - 1L;
			public const long OneHundredThirtyOneThousandSeventyTwoMinusOne = 131_072L - 1L;
			public const long TwoHundredSixtyTwoThousandOneHundredFortyFourMinusOne = 262_144L - 1L;
			public const long FiveHundredTwentyFourThousandTwoHundredEightyEightMinusOne = 524_288L - 1L;
			public const long OneMillionFortyEightThousandFiveHundredSeventySixMinusOne = 1_048_576L - 1L;
			public const long TwoMillionNinetySevenThousandOneHundredFiftyTwoMinusOne = 2_097_152L - 1L;
			public const long FourMillionOneHundredNinetyFourThousandThreeHundredFourMinusOne = 4_194_304L - 1L;
			public const long EightMillionThreeHundredEightyEightThousandSixHundredEightMinusOne = 8_388_608L - 1L;
			public const long SixteenMillionSevenHundredSeventySevenThousandTwoHundredSixteenMinusOne = 16_777_216L - 1L;
			public const long ThirtyThreeMillionFiveHundredFiftyFourThousandFourHundredThirtyTwoMinusOne = 33_554_432L - 1L;
			public const long SixtySevenMillionOneHundredEightThousandEightHundredSixtyFourMinusOne = 67_108_864L - 1L;
			public const long OneHundredThirtyFourMillionTwoHundredSeventeenThousandSevenHundredTwentyEightMinusOne = 134_217_728L - 1L;
			public const long TwoHundredSixtyEightMillionFourHundredThirtyFiveThousandFourHundredFiftySixMinusOne = 268_435_456L - 1L;
			public const long FiveHundredThirtySixMillionEightHundredSeventyThousandNineHundredTwelveMinusOne = 536_870_912L - 1L;
			public const long OneBillionSeventyThreeMillionSevenHundredFortyOneThousandEightHundredTwentyFourMinusOne = 1_073_741_824L - 1L;
			public const long TwoBillionOneHundredFortySevenMillionFourHundredEightyThreeThousandSixHundredFortyEightMinusOne = 2_147_483_648L - 1L;
			public const long FourBillionTwoHundredNinetyFourMillionNineHundredSixtySevenThousandTwoHundredNinetySixMinusOne = 4_294_967_296L - 1L;
			public const long EightBillionFiveHundredEightyNineMillionNineHundredThirtyFourThousandFiveHundredNinetyTwoMinusOne = 8_589_934_592L - 1L;
			public const long SeventeenBillionOneHundredSeventyNineMillionEightHundredSixtyNineThousandOneHundredEightyFourMinusOne = 17_179_869_184L - 1L;
			public const long ThirtyFourBillionThreeHundredFiftyNineMillionSevenHundredThirtyEightThousandThreeHundredSixtyEightMinusOne = 34_359_738_368L - 1L;
			public const long SixtyEightBillionSevenHundredNineteenMillionFourHundredSeventySixThousandSevenHundredThirtySixMinusOne = 68_719_476_736L - 1L;
			public const long OneHundredThirtySevenBillionFourHundredThirtyEightMillionNineHundredFiftyThreeThousandFourHundredSeventyTwoMinusOne = 137_438_953_472L - 1L;
			public const long TwoHundredSeventyFourBillionEightHundredSeventySevenMillionNineHundredSixThousandNineHundredFortyFourMinusOne = 274_877_906_944L - 1L;
			public const long FiveHundredFortyNineBillionSevenHundredFiftyFiveMillionEightHundredThirteenThousandEightHundredEightyEightMinusOne = 549_755_813_888L - 1L;
			public const long OneTrillionNinetyNineBillionFiveHundredElevenMillionSixHundredTwentySevenThousandSevenHundredSeventySixMinusOne = 1_099_511_627_776L - 1L;
			public const long TwoTrillionOneHundredNinetyNineBillionTwentyThreeMillionTwoHundredFiftyFiveThousandFiveHundredFiftyTwoMinusOne = 2_199_023_255_552L - 1L;
			public const long FourTrillionThreeHundredNinetyEightBillionFortySixMillionFiveHundredElevenThousandOneHundredFourMinusOne = 4_398_046_511_104L - 1L;
			public const long EightTrillionSevenHundredNinetySixBillionNinetyThreeMillionTwentyTwoThousandTwoHundredEightMinusOne = 8_796_093_022_208L - 1L;
			public const long SeventeenTrillionFiveHundredNinetyTwoBillionOneHundredEightySixMillionFortyFourThousandFourHundredSixteenMinusOne = 17_592_186_044_416L - 1L;
			public const long ThirtyFiveTrillionOneHundredEightyFourBillionThreeHundredSeventyTwoMillionEightyEightThousandEightHundredThirtyTwoMinusOne = 35_184_372_088_832L - 1L;
			public const long SeventyTrillionThreeHundredSixtyEightBillionSevenHundredFortyFourMillionOneHundredSeventySevenThousandSixHundredSixtyFourMinusOne = 70_368_744_177_664L - 1L;
			public const long OneHundredFortyTrillionSevenHundredThirtySevenBillionFourHundredEightyEightMillionThreeHundredFiftyFiveThousandThreeHundredTwentyEightMinusOne = 140_737_488_355_328L - 1L;
			public const long TwoHundredEightyOneTrillionFourHundredSeventyFourBillionNineHundredSeventySixMillionSevenHundredTenThousandSixHundredFiftySixMinusOne = 281_474_976_710_656L - 1L;
			public const long FiveHundredSixtyTwoTrillionNineHundredFortyNineBillionNineHundredFiftyThreeMillionFourHundredTwentyOneThousandThreeHundredTwelveMinusOne = 562_949_953_421_312L - 1L;
			public const long OneQuadrillionOneHundredTwentyFiveTrillionEightHundredNinetyNineBillionNineHundredSixMillionEightHundredFortyTwoThousandSixHundredTwentyFourMinusOne = 1_125_899_906_842_624L - 1L;
			public const long TwoQuadrillionTwoHundredFiftyOneTrillionSevenHundredNinetyNineBillionEightHundredThirteenMillionSixHundredEightyFiveThousandTwoHundredFortyEightMinusOne = 2_251_799_813_685_248L - 1L;
			public const long FourQuadrillionFiveHundredThreeTrillionFiveHundredNinetyNineBillionSixHundredTwentySevenMillionThreeHundredSeventyThousandFourHundredNinetySixMinusOne = 4_503_599_627_370_496L - 1L;
			public const long NineQuadrillionSevenTrillionOneHundredNinetyNineBillionTwoHundredFiftyFourMillionSevenHundredFortyThousandNineHundredNinetyTwoMinusOne = 9_007_199_254_740_992L - 1L;
			public const long EighteenQuadrillionFourteenTrillionThreeHundredNinetyEightBillionFiveHundredEightMillionEightHundredEightyThousandOneHundredEightyFourMinusOne = 18_014_398_509_481_984L - 1L;
			public const long ThirtySixQuadrillionTwentyEightTrillionSevenHundredNinetySevenBillionSeventeenMillionSevenHundredSixtyThousandThreeHundredSixtyEightMinusOne = 36_028_797_018_963_968L - 1L;
			public const long SeventyTwoQuadrillionFiftySevenTrillionFiveHundredNinetyFourBillionThirtyFiveMillionFiveHundredTwentyThousandSevenHundredThirtySixMinusOne = 72_057_594_037_927_936L - 1L;
			public const long OneHundredFortyFourQuadrillionOneHundredFifteenTrillionOneHundredEightyEightBillionSeventyOneMillionFortyOneThousandFourHundredSeventyTwoMinusOne = 144_115_188_075_855_872L - 1L;
			public const long TwoHundredEightyEightQuadrillionTwoHundredThirtyTrillionThreeHundredSeventySixBillionOneHundredFortyTwoMillionEightyTwoThousandNineHundredFortyFourMinusOne = 288_230_376_151_711_744L - 1L;
			public const long FiveHundredSeventySixQuadrillionFourHundredSixtyTrillionSevenHundredFiftyTwoBillionTwoHundredEightyFourMillionOneHundredSixtyFiveThousandEightHundredEightyEightMinusOne = 576_460_752_303_423_488L - 1L;
			public const long OneQuintillionOneHundredFiftyTwoQuadrillionNineHundredTwentyOneTrillionFiveHundredFourBillionSixHundredSixtyEightMillionThreeHundredThirtyOneThousandSevenHundredSeventySixMinusOne = 1_152_921_504_606_846_976L - 1L;
			public const long TwoQuintillionThreeHundredFiveQuadrillionEightHundredFortyThreeTrillionNineBillionThreeHundredThirtySixMillionSixHundredSixtyThreeThousandFiveHundredFiftyTwoMinusOne = 2_305_843_009_213_693_952L - 1L;
			public const long FourQuintillionSixHundredElevenQuadrillionSixHundredEightySevenTrillionEightHundredSixteenBillionSixHundredSeventyThreeMillionTwoHundredTwentySevenThousandOneHundredFourMinusOne = 4_611_686_018_427_387_904L - 1L;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		public static long LongMinusOne(int i) => i switch
		{
			0 => 0L,
			1 => LongNumbers.TwoMinusOne,
			2 => LongNumbers.FourMinusOne,
			3 => LongNumbers.EightMinusOne,
			4 => LongNumbers.SixteenMinusOne,
			5 => LongNumbers.ThirtyTwoMinusOne,
			6 => LongNumbers.SixtyFourMinusOne,
			7 => LongNumbers.OneHundredTwentyEightMinusOne,
			8 => LongNumbers.TwoHundredFiftySixMinusOne,
			9 => LongNumbers.FiveHundredTwelveMinusOne,
			10 => LongNumbers.OneThousandTwentyFourMinusOne,
			11 => LongNumbers.TwoThousandFortyEightMinusOne,
			12 => LongNumbers.FourThousandNinetySixMinusOne,
			13 => LongNumbers.EightThousandOneHundredNinetyTwoMinusOne,
			14 => LongNumbers.SixteenThousandThreeHundredEightyFourMinusOne,
			15 => LongNumbers.ThirtyTwoThousandSevenHundredSixtyEightMinusOne,
			16 => LongNumbers.SixtyFiveThousandFiveHundredThirtySixMinusOne,
			17 => LongNumbers.OneHundredThirtyOneThousandSeventyTwoMinusOne,
			18 => LongNumbers.TwoHundredSixtyTwoThousandOneHundredFortyFourMinusOne,
			19 => LongNumbers.FiveHundredTwentyFourThousandTwoHundredEightyEightMinusOne,
			20 => LongNumbers.OneMillionFortyEightThousandFiveHundredSeventySixMinusOne,
			21 => LongNumbers.TwoMillionNinetySevenThousandOneHundredFiftyTwoMinusOne,
			22 => LongNumbers.FourMillionOneHundredNinetyFourThousandThreeHundredFourMinusOne,
			23 => LongNumbers.EightMillionThreeHundredEightyEightThousandSixHundredEightMinusOne,
			24 => LongNumbers.SixteenMillionSevenHundredSeventySevenThousandTwoHundredSixteenMinusOne,
			25 => LongNumbers.ThirtyThreeMillionFiveHundredFiftyFourThousandFourHundredThirtyTwoMinusOne,
			26 => LongNumbers.SixtySevenMillionOneHundredEightThousandEightHundredSixtyFourMinusOne,
			27 => LongNumbers.OneHundredThirtyFourMillionTwoHundredSeventeenThousandSevenHundredTwentyEightMinusOne,
			28 => LongNumbers.TwoHundredSixtyEightMillionFourHundredThirtyFiveThousandFourHundredFiftySixMinusOne,
			29 => LongNumbers.FiveHundredThirtySixMillionEightHundredSeventyThousandNineHundredTwelveMinusOne,
			30 => LongNumbers.OneBillionSeventyThreeMillionSevenHundredFortyOneThousandEightHundredTwentyFourMinusOne,
			31 => LongNumbers.TwoBillionOneHundredFortySevenMillionFourHundredEightyThreeThousandSixHundredFortyEightMinusOne,
			32 => LongNumbers.FourBillionTwoHundredNinetyFourMillionNineHundredSixtySevenThousandTwoHundredNinetySixMinusOne,
			33 => LongNumbers.EightBillionFiveHundredEightyNineMillionNineHundredThirtyFourThousandFiveHundredNinetyTwoMinusOne,
			34 => LongNumbers.SeventeenBillionOneHundredSeventyNineMillionEightHundredSixtyNineThousandOneHundredEightyFourMinusOne,
			35 => LongNumbers.ThirtyFourBillionThreeHundredFiftyNineMillionSevenHundredThirtyEightThousandThreeHundredSixtyEightMinusOne,
			36 => LongNumbers.SixtyEightBillionSevenHundredNineteenMillionFourHundredSeventySixThousandSevenHundredThirtySixMinusOne,
			37 => LongNumbers.OneHundredThirtySevenBillionFourHundredThirtyEightMillionNineHundredFiftyThreeThousandFourHundredSeventyTwoMinusOne,
			38 => LongNumbers.TwoHundredSeventyFourBillionEightHundredSeventySevenMillionNineHundredSixThousandNineHundredFortyFourMinusOne,
			39 => LongNumbers.FiveHundredFortyNineBillionSevenHundredFiftyFiveMillionEightHundredThirteenThousandEightHundredEightyEightMinusOne,
			40 => LongNumbers.OneTrillionNinetyNineBillionFiveHundredElevenMillionSixHundredTwentySevenThousandSevenHundredSeventySixMinusOne,
			41 => LongNumbers.TwoTrillionOneHundredNinetyNineBillionTwentyThreeMillionTwoHundredFiftyFiveThousandFiveHundredFiftyTwoMinusOne,
			42 => LongNumbers.FourTrillionThreeHundredNinetyEightBillionFortySixMillionFiveHundredElevenThousandOneHundredFourMinusOne,
			43 => LongNumbers.EightTrillionSevenHundredNinetySixBillionNinetyThreeMillionTwentyTwoThousandTwoHundredEightMinusOne,
			44 => LongNumbers.SeventeenTrillionFiveHundredNinetyTwoBillionOneHundredEightySixMillionFortyFourThousandFourHundredSixteenMinusOne,
			45 => LongNumbers.ThirtyFiveTrillionOneHundredEightyFourBillionThreeHundredSeventyTwoMillionEightyEightThousandEightHundredThirtyTwoMinusOne,
			46 => LongNumbers.SeventyTrillionThreeHundredSixtyEightBillionSevenHundredFortyFourMillionOneHundredSeventySevenThousandSixHundredSixtyFourMinusOne,
			47 => LongNumbers.OneHundredFortyTrillionSevenHundredThirtySevenBillionFourHundredEightyEightMillionThreeHundredFiftyFiveThousandThreeHundredTwentyEightMinusOne,
			48 => LongNumbers.TwoHundredEightyOneTrillionFourHundredSeventyFourBillionNineHundredSeventySixMillionSevenHundredTenThousandSixHundredFiftySixMinusOne,
			49 => LongNumbers.FiveHundredSixtyTwoTrillionNineHundredFortyNineBillionNineHundredFiftyThreeMillionFourHundredTwentyOneThousandThreeHundredTwelveMinusOne,
			50 => LongNumbers.OneQuadrillionOneHundredTwentyFiveTrillionEightHundredNinetyNineBillionNineHundredSixMillionEightHundredFortyTwoThousandSixHundredTwentyFourMinusOne,
			51 => LongNumbers.TwoQuadrillionTwoHundredFiftyOneTrillionSevenHundredNinetyNineBillionEightHundredThirteenMillionSixHundredEightyFiveThousandTwoHundredFortyEightMinusOne,
			52 => LongNumbers.FourQuadrillionFiveHundredThreeTrillionFiveHundredNinetyNineBillionSixHundredTwentySevenMillionThreeHundredSeventyThousandFourHundredNinetySixMinusOne,
			53 => LongNumbers.NineQuadrillionSevenTrillionOneHundredNinetyNineBillionTwoHundredFiftyFourMillionSevenHundredFortyThousandNineHundredNinetyTwoMinusOne,
			54 => LongNumbers.EighteenQuadrillionFourteenTrillionThreeHundredNinetyEightBillionFiveHundredEightMillionEightHundredEightyThousandOneHundredEightyFourMinusOne,
			55 => LongNumbers.ThirtySixQuadrillionTwentyEightTrillionSevenHundredNinetySevenBillionSeventeenMillionSevenHundredSixtyThousandThreeHundredSixtyEightMinusOne,
			56 => LongNumbers.SeventyTwoQuadrillionFiftySevenTrillionFiveHundredNinetyFourBillionThirtyFiveMillionFiveHundredTwentyThousandSevenHundredThirtySixMinusOne,
			57 => LongNumbers.OneHundredFortyFourQuadrillionOneHundredFifteenTrillionOneHundredEightyEightBillionSeventyOneMillionFortyOneThousandFourHundredSeventyTwoMinusOne,
			58 => LongNumbers.TwoHundredEightyEightQuadrillionTwoHundredThirtyTrillionThreeHundredSeventySixBillionOneHundredFortyTwoMillionEightyTwoThousandNineHundredFortyFourMinusOne,
			59 => LongNumbers.FiveHundredSeventySixQuadrillionFourHundredSixtyTrillionSevenHundredFiftyTwoBillionTwoHundredEightyFourMillionOneHundredSixtyFiveThousandEightHundredEightyEightMinusOne,
			60 => LongNumbers.OneQuintillionOneHundredFiftyTwoQuadrillionNineHundredTwentyOneTrillionFiveHundredFourBillionSixHundredSixtyEightMillionThreeHundredThirtyOneThousandSevenHundredSeventySixMinusOne,
			61 => LongNumbers.TwoQuintillionThreeHundredFiveQuadrillionEightHundredFortyThreeTrillionNineBillionThreeHundredThirtySixMillionSixHundredSixtyThreeThousandFiveHundredFiftyTwoMinusOne,
			62 => LongNumbers.FourQuintillionSixHundredElevenQuadrillionSixHundredEightySevenTrillionEightHundredSixteenBillionSixHundredSeventyThreeMillionTwoHundredTwentySevenThousandOneHundredFourMinusOne,
			_ => throw new ArgumentOutOfRangeException(nameof(i)),
		};

		[Benchmark(Baseline = true)]
		public long ByShifting()
		{
			long result = 0L;
			for (int i = 0; i <= 62; i++)
			{
				result = (1L << i) - 1L;
			}

			return result;
		}

		[Benchmark]
		public long ByMapping()
		{
			long result = 0L;
			for (int i = 0; i <= 62; i++)
			{
				result = LongMinusOne(i);
			}

			return result;
		}
	}
}