using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

internal static partial class PrimeOrderCalculator
{
	internal readonly struct PrimeOrderCalculatorConfig(uint smallFactorLimit, int pollardRhoMilliseconds, int maxPowChecks, bool strictMode)
	{
		private static PrimeOrderCalculatorConfig _heuristicDefault = CreateHeuristicDefault(UnboundedTaskScheduler.ConfiguredThreadCount);

		public static PrimeOrderCalculatorConfig HeuristicDefault
		{
			[MethodImpl(MethodImplOptions.AggressiveInlining)]
			get => _heuristicDefault;
		}

		public static readonly PrimeOrderCalculatorConfig StrictDefault = new(smallFactorLimit: 1_000_000, pollardRhoMilliseconds: 0, maxPowChecks: 0, true);

		public static void ConfigureHeuristicDefault(int threadCount)
		{
			_heuristicDefault = CreateHeuristicDefault(threadCount);
		}

		public static PrimeOrderCalculatorConfig CreateHeuristicDefault(int threadCount)
		{
			int normalizedThreadCount = threadCount < 1 ? 1 : threadCount;
			return new(smallFactorLimit: 100_000, pollardRhoMilliseconds: normalizedThreadCount * 24, maxPowChecks: 24, false);
		}

		public readonly uint SmallFactorLimit = smallFactorLimit;
		public readonly int PollardRhoMilliseconds = pollardRhoMilliseconds;
		public readonly int MaxPowChecks = maxPowChecks;
		public readonly bool StrictMode = strictMode;
	}

	// TODO: Remove branching on the CPU / GPU paths
	// TODO: Split big non-static functions into smaller / extract static code to limit JIT time
}
