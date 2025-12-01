using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

internal static partial class PrimeOrderCalculator
{
	internal enum PrimeOrderMode
	{
		Heuristic,
		Strict,
	}

	internal readonly struct PrimeOrderSearchConfig(uint smallFactorLimit, int pollardRhoMilliseconds, int maxPowChecks, PrimeOrderMode mode)
	{
		private static PrimeOrderSearchConfig _heuristicDefault = CreateHeuristicDefault(UnboundedTaskScheduler.ConfiguredThreadCount);

		public static PrimeOrderSearchConfig HeuristicDefault
		{
			[MethodImpl(MethodImplOptions.AggressiveInlining)]
			get => _heuristicDefault;
		}

		public static readonly PrimeOrderSearchConfig StrictDefault = new(smallFactorLimit: 1_000_000, pollardRhoMilliseconds: 0, maxPowChecks: 0, PrimeOrderMode.Strict);

		public static void ConfigureHeuristicDefault(int threadCount)
		{
			_heuristicDefault = CreateHeuristicDefault(threadCount);
		}

		public static PrimeOrderSearchConfig CreateHeuristicDefault(int threadCount)
		{
			int normalizedThreadCount = threadCount < 1 ? 1 : threadCount;
			return new(smallFactorLimit: 100_000, pollardRhoMilliseconds: normalizedThreadCount * 24, maxPowChecks: 24, PrimeOrderMode.Heuristic);
		}

		public readonly uint SmallFactorLimit = smallFactorLimit;
		public readonly int PollardRhoMilliseconds = pollardRhoMilliseconds;
		public readonly int MaxPowChecks = maxPowChecks;
		public readonly PrimeOrderMode Mode = mode;
	}

	// TODO: Remove branching on the CPU / GPU paths
	// TODO: Split big non-static functions into smaller / extract static code to limit JIT time
}
