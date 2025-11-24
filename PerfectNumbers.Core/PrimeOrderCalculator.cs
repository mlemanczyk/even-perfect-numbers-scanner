using System.Runtime.CompilerServices;
using PerfectNumbers.Core.Gpu.Accelerators;

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

	internal enum PrimeOrderHeuristicDevice
	{
		Cpu,
		Gpu,
	}

	// TODO: Remove branching on the CPU / GPU paths
	// TODO: Split big non-static functions into smaller / extract static code to limit JIT time
	public static ulong Calculate(
			PrimeOrderCalculatorAccelerator gpu,
			ulong prime,
			ulong? previousOrder,
			in MontgomeryDivisorData divisorData,
			in PrimeOrderSearchConfig config,
			PrimeOrderHeuristicDevice device)
	{
		var scope = UsePow2Mode(device);
		ulong order = CalculateInternal(gpu, prime, previousOrder, divisorData, config);
		scope.Dispose();
		return order;
	}

	public static UInt128 Calculate(
			PrimeOrderCalculatorAccelerator gpu,
			in UInt128 prime,
			in UInt128? previousOrder,
			in PrimeOrderSearchConfig config,
			in PrimeOrderHeuristicDevice device)
	{
		var scope = UsePow2Mode(device);
		MontgomeryDivisorData divisorData;
		UInt128 result;
		if (prime <= ulong.MaxValue)
		{
			ulong? previous = null;
			if (previousOrder.HasValue)
			{
				UInt128 previousValue = previousOrder.Value;
				if (previousValue <= ulong.MaxValue)
				{
					previous = (ulong)previousValue;
				}
				else
				{
					previous = ulong.MaxValue;
				}
			}

			ulong prime64 = (ulong)prime;
			Queue<MontgomeryDivisorData> divisorPool = MontgomeryDivisorDataPool.Shared;
			divisorData = divisorPool.FromModulus(prime64);
			ulong order64 = Calculate(gpu, prime64, previous, divisorData, config, device);
			divisorPool.Return(divisorData);
			result = order64 == 0UL ? UInt128.Zero : (UInt128)order64;
		}
		else
		{
			divisorData = MontgomeryDivisorData.Empty;
			result = CalculateWideInternal(gpu, prime, previousOrder, divisorData, config);
		}

		scope.Dispose();
		return result;
	}
}
