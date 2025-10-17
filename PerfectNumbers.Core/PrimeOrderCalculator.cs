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
                public static readonly PrimeOrderSearchConfig HeuristicDefault = new(smallFactorLimit: 4_000_000, pollardRhoMilliseconds: 24 * 10_240, maxPowChecks: 24, PrimeOrderMode.Heuristic);
                public static readonly PrimeOrderSearchConfig StrictDefault = new(smallFactorLimit: 1_000_000, pollardRhoMilliseconds: 0, maxPowChecks: 0, PrimeOrderMode.Strict);

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

        public static ulong Calculate(
                ulong prime,
                ulong? previousOrder,
                in MontgomeryDivisorData divisorData,
                in PrimeOrderSearchConfig config,
                PrimeOrderHeuristicDevice device)
        {
                var scope = UsePow2Mode(device);
                ulong order = CalculateInternal(prime, previousOrder, divisorData, config);
                scope.Dispose();
                return order;
        }

        public static UInt128 Calculate(
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
                        divisorData = MontgomeryDivisorData.FromModulus(prime64);
                        ulong order64 = Calculate(prime64, previous, divisorData, config, device);
                        result = order64 == 0UL ? UInt128.Zero : (UInt128)order64;
                }
                else
                {
                        divisorData = default;
                        result = CalculateWideInternal(prime, previousOrder, divisorData, config);
                }

                scope.Dispose();
                return result;
        }
}
