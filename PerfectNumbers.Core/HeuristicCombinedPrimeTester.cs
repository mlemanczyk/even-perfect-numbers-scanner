using Open.Numeric.Primes;
using System;
using ILGPU.Runtime;
using System.Runtime.CompilerServices;
using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core;

public sealed class HeuristicCombinedPrimeTester
{
	static HeuristicCombinedPrimeTester()
	{
		HeuristicPrimeSieves.EnsureInitialized();

		CombinedDivisorsEnding1TwoAOneB = BuildCombinedDivisors(1, CombinedDivisorPattern.TwoAOneB, out CombinedDivisorsEnding1TwoAOneBSquares);
		CombinedDivisorsEnding3TwoAOneB = BuildCombinedDivisors(3, CombinedDivisorPattern.TwoAOneB, out CombinedDivisorsEnding3TwoAOneBSquares);
		CombinedDivisorsEnding7TwoAOneB = BuildCombinedDivisors(7, CombinedDivisorPattern.TwoAOneB, out CombinedDivisorsEnding7TwoAOneBSquares);
		CombinedDivisorsEnding9TwoAOneB = BuildCombinedDivisors(9, CombinedDivisorPattern.TwoAOneB, out CombinedDivisorsEnding9TwoAOneBSquares);

		CombinedDivisorsEnding1OneAOneB = BuildCombinedDivisors(1, CombinedDivisorPattern.OneAOneB, out CombinedDivisorsEnding1OneAOneBSquares);
		CombinedDivisorsEnding3OneAOneB = BuildCombinedDivisors(3, CombinedDivisorPattern.OneAOneB, out CombinedDivisorsEnding3OneAOneBSquares);
		CombinedDivisorsEnding7OneAOneB = BuildCombinedDivisors(7, CombinedDivisorPattern.OneAOneB, out CombinedDivisorsEnding7OneAOneBSquares);
		CombinedDivisorsEnding9OneAOneB = BuildCombinedDivisors(9, CombinedDivisorPattern.OneAOneB, out CombinedDivisorsEnding9OneAOneBSquares);

		CombinedDivisorsEnding1ThreeAOneB = BuildCombinedDivisors(1, CombinedDivisorPattern.ThreeAOneB, out CombinedDivisorsEnding1ThreeAOneBSquares);
		CombinedDivisorsEnding3ThreeAOneB = BuildCombinedDivisors(3, CombinedDivisorPattern.ThreeAOneB, out CombinedDivisorsEnding3ThreeAOneBSquares);
		CombinedDivisorsEnding7ThreeAOneB = BuildCombinedDivisors(7, CombinedDivisorPattern.ThreeAOneB, out CombinedDivisorsEnding7ThreeAOneBSquares);
		CombinedDivisorsEnding9ThreeAOneB = BuildCombinedDivisors(9, CombinedDivisorPattern.ThreeAOneB, out CombinedDivisorsEnding9ThreeAOneBSquares);

		CombinedDivisorsEnding1ThreeATwoB = BuildCombinedDivisors(1, CombinedDivisorPattern.ThreeATwoB, out CombinedDivisorsEnding1ThreeATwoBSquares);
		CombinedDivisorsEnding3ThreeATwoB = BuildCombinedDivisors(3, CombinedDivisorPattern.ThreeATwoB, out CombinedDivisorsEnding3ThreeATwoBSquares);
		CombinedDivisorsEnding7ThreeATwoB = BuildCombinedDivisors(7, CombinedDivisorPattern.ThreeATwoB, out CombinedDivisorsEnding7ThreeATwoBSquares);
		CombinedDivisorsEnding9ThreeATwoB = BuildCombinedDivisors(9, CombinedDivisorPattern.ThreeATwoB, out CombinedDivisorsEnding9ThreeATwoBSquares);
		CombinedDivisorsEnding1 = CombinedDivisorsEnding1OneAOneB;
		CombinedDivisorsEnding3 = CombinedDivisorsEnding3OneAOneB;
		CombinedDivisorsEnding7 = CombinedDivisorsEnding7OneAOneB;
		CombinedDivisorsEnding9 = CombinedDivisorsEnding9OneAOneB;
		CombinedDivisorsEnding1Squares = CombinedDivisorsEnding1OneAOneBSquares;
		CombinedDivisorsEnding3Squares = CombinedDivisorsEnding3OneAOneBSquares;
		CombinedDivisorsEnding7Squares = CombinedDivisorsEnding7OneAOneBSquares;
		CombinedDivisorsEnding9Squares = CombinedDivisorsEnding9OneAOneBSquares;
	}


        public static void EnsureInitialized()
        {
        // Intentionally left blank. Accessing this method forces the static constructor to run.
        }
        
        internal static ReadOnlySpan<ulong> CombinedDivisorsEnding1Span => CombinedDivisorsEnding1;
        internal static ReadOnlySpan<ulong> CombinedDivisorsEnding3Span => CombinedDivisorsEnding3;
        internal static ReadOnlySpan<ulong> CombinedDivisorsEnding7Span => CombinedDivisorsEnding7;
        internal static ReadOnlySpan<ulong> CombinedDivisorsEnding9Span => CombinedDivisorsEnding9;
        
        internal static ReadOnlySpan<ulong> CombinedDivisorsEnding1SquaresSpan => CombinedDivisorsEnding1Squares;
        internal static ReadOnlySpan<ulong> CombinedDivisorsEnding3SquaresSpan => CombinedDivisorsEnding3Squares;
        internal static ReadOnlySpan<ulong> CombinedDivisorsEnding7SquaresSpan => CombinedDivisorsEnding7Squares;
        internal static ReadOnlySpan<ulong> CombinedDivisorsEnding9SquaresSpan => CombinedDivisorsEnding9Squares;

	private const ulong Wheel210 = 210UL;

	private static readonly ulong[] CombinedDivisorsEnding1TwoAOneB;
	private static readonly ulong[] CombinedDivisorsEnding3TwoAOneB;
	private static readonly ulong[] CombinedDivisorsEnding7TwoAOneB;
	private static readonly ulong[] CombinedDivisorsEnding9TwoAOneB;

	private static readonly ulong[] CombinedDivisorsEnding1OneAOneB;
	private static readonly ulong[] CombinedDivisorsEnding3OneAOneB;
	private static readonly ulong[] CombinedDivisorsEnding7OneAOneB;
	private static readonly ulong[] CombinedDivisorsEnding9OneAOneB;

	private static readonly ulong[] CombinedDivisorsEnding1ThreeAOneB;
	private static readonly ulong[] CombinedDivisorsEnding3ThreeAOneB;
	private static readonly ulong[] CombinedDivisorsEnding7ThreeAOneB;
	private static readonly ulong[] CombinedDivisorsEnding9ThreeAOneB;

	private static readonly ulong[] CombinedDivisorsEnding1ThreeATwoB;
	private static readonly ulong[] CombinedDivisorsEnding3ThreeATwoB;
	private static readonly ulong[] CombinedDivisorsEnding7ThreeATwoB;
	private static readonly ulong[] CombinedDivisorsEnding9ThreeATwoB;

	private static readonly ulong[] CombinedDivisorsEnding1;
	private static readonly ulong[] CombinedDivisorsEnding3;
	private static readonly ulong[] CombinedDivisorsEnding7;
	private static readonly ulong[] CombinedDivisorsEnding9;

	private static readonly ulong[] CombinedDivisorsEnding1TwoAOneBSquares;
	private static readonly ulong[] CombinedDivisorsEnding3TwoAOneBSquares;
	private static readonly ulong[] CombinedDivisorsEnding7TwoAOneBSquares;
	private static readonly ulong[] CombinedDivisorsEnding9TwoAOneBSquares;

	private static readonly ulong[] CombinedDivisorsEnding1OneAOneBSquares;
	private static readonly ulong[] CombinedDivisorsEnding3OneAOneBSquares;
	private static readonly ulong[] CombinedDivisorsEnding7OneAOneBSquares;
	private static readonly ulong[] CombinedDivisorsEnding9OneAOneBSquares;

	private static readonly ulong[] CombinedDivisorsEnding1ThreeAOneBSquares;
	private static readonly ulong[] CombinedDivisorsEnding3ThreeAOneBSquares;
	private static readonly ulong[] CombinedDivisorsEnding7ThreeAOneBSquares;
	private static readonly ulong[] CombinedDivisorsEnding9ThreeAOneBSquares;

	private static readonly ulong[] CombinedDivisorsEnding1ThreeATwoBSquares;
	private static readonly ulong[] CombinedDivisorsEnding3ThreeATwoBSquares;
	private static readonly ulong[] CombinedDivisorsEnding7ThreeATwoBSquares;
	private static readonly ulong[] CombinedDivisorsEnding9ThreeATwoBSquares;

	private static readonly ulong[] CombinedDivisorsEnding1Squares;
	private static readonly ulong[] CombinedDivisorsEnding3Squares;
	private static readonly ulong[] CombinedDivisorsEnding7Squares;
	private static readonly ulong[] CombinedDivisorsEnding9Squares;

	private static readonly ulong[] HeuristicSmallCycleSnapshot = MersenneDivisorCycles.Shared.ExportSmallCyclesSnapshot();

	internal enum CombinedDivisorPattern : byte
	{
		TwoAOneB = 0,
		OneAOneB = 1,
		ThreeAOneB = 2,
		ThreeATwoB = 3,
	}

	internal enum HeuristicDivisorGroup : byte
	{
		None = 0,
		GroupAConstant = 1,
		GroupAWheel = 2,
		GroupB = 3,
	}

	internal readonly struct HeuristicDivisorCandidate(ulong value, HeuristicDivisorGroup group, byte ending, byte priorityIndex, ushort wheelResidue)
	{
		public readonly ulong Value = value;
		public readonly HeuristicDivisorGroup Group = group;
		public readonly byte Ending = ending;
		public readonly byte PriorityIndex = priorityIndex;
		public readonly ushort WheelResidue = wheelResidue;
	}

	internal readonly struct HeuristicDivisorPreparation(
		in HeuristicDivisorCandidate candidate,
		in MontgomeryDivisorData divisorData,
		ulong cycleLengthHint,
		bool hasCycleLengthHint)
	{
		public readonly HeuristicDivisorCandidate Candidate = candidate;
		public readonly MontgomeryDivisorData DivisorData = divisorData;
		public readonly ulong CycleLengthHint = cycleLengthHint;
		public readonly bool HasCycleLengthHint = hasCycleLengthHint;
		public readonly bool RequiresCycleComputation = !hasCycleLengthHint;
	}



	[ThreadStatic]
	private static HeuristicCombinedPrimeTester? _tester;

	public static HeuristicCombinedPrimeTester Exclusive => _tester ??= new();

	public static bool IsPrimeCpu(ulong n)
	{
		byte nMod10 = (byte)n.Mod10();
		ulong maxDivisorSquare = ComputeHeuristicDivisorSquareLimit(n);

		if (maxDivisorSquare < 9UL)
		{
			return EvaluateWithOpenNumericFallback(n);
		}

		return HeuristicTrialDivisionCpu(n, maxDivisorSquare, nMod10);
	}

	public static bool IsPrimeGpu(ulong n)
	{
		byte nMod10 = (byte)n.Mod10();
		ulong maxDivisorSquare = ComputeHeuristicDivisorSquareLimit(n);
		return IsPrimeGpu(n, maxDivisorSquare, nMod10);
	}

	public static bool IsPrimeGpu(ulong n, ulong maxDivisorSquare, byte nMod10)
	{
		// TODO: Is this condition ever met on the execution path in EvenPerfectBitScanner?
		if (maxDivisorSquare < 9UL)
		{
			return EvaluateWithOpenNumericFallback(n);
		}

		bool compositeDetected = HeuristicTrialDivisionGpuDetectsDivisor(n, maxDivisorSquare, nMod10);
		return !compositeDetected;
	}

	private static bool HeuristicTrialDivisionCpu(ulong n, ulong maxDivisorSquare, byte nMod10)
	{
		ReadOnlySpan<ulong> combinedDivisors = GetCombinedDivisors(nMod10);
		ReadOnlySpan<ulong> combinedDivisorSquares = GetCombinedDivisorSquares(nMod10);

		int length = combinedDivisors.Length;

		for (int i = 0; i < length; i++)
		{
			ulong divisorSquare = combinedDivisorSquares[i];
			if (divisorSquare > maxDivisorSquare)
			{
				break;
			}

			if (n % combinedDivisors[i] == 0UL)
			{
				return false;
			}
		}

		return EvaluateWithOpenNumericFallback(n);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	internal static int GetBatchSize(ulong n) => n switch
	{
		< 1_000 => 8,
		< 10_000 => 16,
		< 100_000 => 32,
		< 1_000_000 => 128,
		_ => 256,
	};


        private static bool HeuristicTrialDivisionGpuDetectsDivisor(ulong n, ulong maxDivisorSquare, byte nMod10)
        {
                var limiter = GpuPrimeWorkLimiter.Acquire();
                var gpu = PrimeTester.PrimeTesterGpuContextPool.Rent(1);
                var accelerator = gpu.Accelerator;
                var kernel = gpu.HeuristicTrialDivisionKernel;
                var flagView1D = gpu.HeuristicFlag.View;
                var flagView = flagView1D.AsContiguous();

                var tables = new HeuristicGpuDivisorTables(
                        gpu.HeuristicCombinedDivisorsEnding1.View,
                        gpu.HeuristicCombinedDivisorSquaresEnding1.View,
                        gpu.HeuristicCombinedDivisorsEnding3.View,
                        gpu.HeuristicCombinedDivisorSquaresEnding3.View,
                        gpu.HeuristicCombinedDivisorsEnding7.View,
                        gpu.HeuristicCombinedDivisorSquaresEnding7.View,
                        gpu.HeuristicCombinedDivisorsEnding9.View,
                        gpu.HeuristicCombinedDivisorSquaresEnding9.View,
                        gpu.HeuristicGroupADivisors.View,
                        gpu.HeuristicGroupADivisorSquares.View,
                        gpu.HeuristicGroupBDivisorsEnding1.View,
                        gpu.HeuristicGroupBDivisorSquaresEnding1.View,
                        gpu.HeuristicGroupBDivisorsEnding7.View,
                        gpu.HeuristicGroupBDivisorSquaresEnding7.View,
                        gpu.HeuristicGroupBDivisorsEnding9.View,
                        gpu.HeuristicGroupBDivisorSquaresEnding9.View);

                bool compositeDetected = false;
                int divisorLength = GetCombinedDivisors(nMod10).Length;

                if (divisorLength > 0)
                {
                        int compositeFlag = 0;
                        flagView1D.CopyFromCPU(ref compositeFlag, 1);
                        kernel(
                                divisorLength,
                                tables,
                                flagView,
                                n,
                                maxDivisorSquare,
                                HeuristicGpuDivisorTableKind.Combined,
                                nMod10);
                        accelerator.Synchronize();
                        flagView1D.CopyToCPU(ref compositeFlag, 1);
                        compositeDetected = compositeFlag != 0;
                }

                gpu.Dispose();
                limiter.Dispose();
                return compositeDetected;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool EvaluateWithOpenNumericFallback(ulong n)
	{
		return Prime.Numbers.IsPrime(n);
	}
	internal static ulong ComputeHeuristicDivisorSquareLimit(ulong n)
	{
		return n;
	}

	private static ReadOnlySpan<ulong> GetCombinedDivisors(byte nMod10)
	{
		return nMod10 switch
		{
			1 => CombinedDivisorsEnding1,
			3 => CombinedDivisorsEnding3,
			7 => CombinedDivisorsEnding7,
			9 => CombinedDivisorsEnding9,
			_ => throw new InvalidOperationException($"Unsupported combined divisor selector for digit {nMod10}."),
		};
	}

	private static ReadOnlySpan<ulong> GetCombinedDivisorSquares(byte nMod10)
	{
		return nMod10 switch
		{
			1 => CombinedDivisorsEnding1Squares,
			3 => CombinedDivisorsEnding3Squares,
			7 => CombinedDivisorsEnding7Squares,
			9 => CombinedDivisorsEnding9Squares,
			_ => throw new InvalidOperationException($"Unsupported combined divisor square selector for digit {nMod10}."),
		};
	}

	internal static ReadOnlySpan<ulong> GetCombinedDivisors(byte nMod10, CombinedDivisorPattern pattern)
	{
		return pattern switch
		{
			CombinedDivisorPattern.TwoAOneB => nMod10 switch
			{
				1 => CombinedDivisorsEnding1TwoAOneB,
				3 => CombinedDivisorsEnding3TwoAOneB,
				7 => CombinedDivisorsEnding7TwoAOneB,
				9 => CombinedDivisorsEnding9TwoAOneB,
				_ => throw new InvalidOperationException($"Unsupported combined divisor selector for digit {nMod10}."),
			},
			CombinedDivisorPattern.OneAOneB => nMod10 switch
			{
				1 => CombinedDivisorsEnding1OneAOneB,
				3 => CombinedDivisorsEnding3OneAOneB,
				7 => CombinedDivisorsEnding7OneAOneB,
				9 => CombinedDivisorsEnding9OneAOneB,
				_ => throw new InvalidOperationException($"Unsupported combined divisor selector for digit {nMod10}."),
			},
			CombinedDivisorPattern.ThreeAOneB => nMod10 switch
			{
				1 => CombinedDivisorsEnding1ThreeAOneB,
				3 => CombinedDivisorsEnding3ThreeAOneB,
				7 => CombinedDivisorsEnding7ThreeAOneB,
				9 => CombinedDivisorsEnding9ThreeAOneB,
				_ => throw new InvalidOperationException($"Unsupported combined divisor selector for digit {nMod10}."),
			},
			CombinedDivisorPattern.ThreeATwoB => nMod10 switch
			{
				1 => CombinedDivisorsEnding1ThreeATwoB,
				3 => CombinedDivisorsEnding3ThreeATwoB,
				7 => CombinedDivisorsEnding7ThreeATwoB,
				9 => CombinedDivisorsEnding9ThreeATwoB,
				_ => throw new InvalidOperationException($"Unsupported combined divisor selector for digit {nMod10}."),
			},
			_ => throw new InvalidOperationException($"Unsupported combined divisor pattern: {pattern}."),
		};
	}

	internal static ReadOnlySpan<ulong> GetCombinedDivisorSquares(byte nMod10, CombinedDivisorPattern pattern)
	{
		return pattern switch
		{
			CombinedDivisorPattern.TwoAOneB => nMod10 switch
			{
				1 => CombinedDivisorsEnding1TwoAOneBSquares,
				3 => CombinedDivisorsEnding3TwoAOneBSquares,
				7 => CombinedDivisorsEnding7TwoAOneBSquares,
				9 => CombinedDivisorsEnding9TwoAOneBSquares,
				_ => throw new InvalidOperationException($"Unsupported combined divisor square selector for digit {nMod10}."),
			},
			CombinedDivisorPattern.OneAOneB => nMod10 switch
			{
				1 => CombinedDivisorsEnding1OneAOneBSquares,
				3 => CombinedDivisorsEnding3OneAOneBSquares,
				7 => CombinedDivisorsEnding7OneAOneBSquares,
				9 => CombinedDivisorsEnding9OneAOneBSquares,
				_ => throw new InvalidOperationException($"Unsupported combined divisor square selector for digit {nMod10}."),
			},
			CombinedDivisorPattern.ThreeAOneB => nMod10 switch
			{
				1 => CombinedDivisorsEnding1ThreeAOneBSquares,
				3 => CombinedDivisorsEnding3ThreeAOneBSquares,
				7 => CombinedDivisorsEnding7ThreeAOneBSquares,
				9 => CombinedDivisorsEnding9ThreeAOneBSquares,
				_ => throw new InvalidOperationException($"Unsupported combined divisor square selector for digit {nMod10}."),
			},
			CombinedDivisorPattern.ThreeATwoB => nMod10 switch
			{
				1 => CombinedDivisorsEnding1ThreeATwoBSquares,
				3 => CombinedDivisorsEnding3ThreeATwoBSquares,
				7 => CombinedDivisorsEnding7ThreeATwoBSquares,
				9 => CombinedDivisorsEnding9ThreeATwoBSquares,
				_ => throw new InvalidOperationException($"Unsupported combined divisor square selector for digit {nMod10}."),
			},
			_ => throw new InvalidOperationException($"Unsupported combined divisor pattern for squares: {pattern}."),
		};
	}

	private static ulong[] BuildCombinedDivisors(byte nMod10, CombinedDivisorPattern pattern, out ulong[] squares)
	{
		ReadOnlySpan<ulong> groupA = HeuristicPrimeSieves.GroupADivisors;
		ReadOnlySpan<uint> groupB = GetGroupBDivisors(nMod10);

		var combined = new List<ulong>(groupA.Length + groupB.Length);

		int groupAIndex = 0;
		while (groupAIndex < groupA.Length && groupA[groupAIndex] <= 19)
		{
			combined.Add(groupA[groupAIndex]);
			groupAIndex++;
		}

		int groupBIndex = GetGroupBStartIndex(groupB);
		int groupAInterleaveCount;
		int groupBInterleaveCount;
		switch (pattern)
		{
			case CombinedDivisorPattern.OneAOneB:
				groupAInterleaveCount = 1;
				groupBInterleaveCount = 1;
				break;
			case CombinedDivisorPattern.TwoAOneB:
				groupAInterleaveCount = 2;
				groupBInterleaveCount = 1;
				break;
			case CombinedDivisorPattern.ThreeAOneB:
				groupAInterleaveCount = 3;
				groupBInterleaveCount = 1;
				break;
			case CombinedDivisorPattern.ThreeATwoB:
				groupAInterleaveCount = 3;
				groupBInterleaveCount = 2;
				break;
			default:
				groupAInterleaveCount = 2;
				groupBInterleaveCount = 1;
				break;
		}

		while (groupAIndex < groupA.Length || groupBIndex < groupB.Length)
		{
			int addedA = 0;
			while (addedA < groupAInterleaveCount && groupAIndex < groupA.Length)
			{
				combined.Add(groupA[groupAIndex]);
				groupAIndex++;
				addedA++;
			}

			int addedB = 0;
			while (addedB < groupBInterleaveCount && groupBIndex < groupB.Length)
			{
				combined.Add(groupB[groupBIndex]);
				groupBIndex++;
				addedB++;
			}
		}

		ulong[] result = [.. combined];
		ulong[] resultSquares = new ulong[result.Length];
		for (int i = 0; i < result.Length; i++)
		{
			ulong divisor = result[i];
			resultSquares[i] = divisor * divisor;
		}

		squares = resultSquares;
		return result;
	}



	private static int GetGroupBStartIndex(ReadOnlySpan<uint> divisors)
	{
		int index = 0;
		while (index < divisors.Length && IsGroupAPrefixValue(divisors[index]))
		{
			index++;
		}

		return index;
	}

	private static bool IsGroupAPrefixValue(ulong value) => value is 3UL or 7UL or 11UL or 13UL;

	private static HeuristicDivisorGroup ResolveGroup(ulong value) => value switch
	{
		3UL or 7UL or 11UL or 13UL => HeuristicDivisorGroup.GroupAConstant,
		_ when value % 10UL == 3UL => HeuristicDivisorGroup.GroupAWheel,
		_ => HeuristicDivisorGroup.GroupB,
	};

	private static ReadOnlySpan<uint> GetGroupBDivisors(byte ending) => ending switch
	{
		1 => DivisorGenerator.SmallPrimesLastOneWithoutLastThree,
		3 => [],
		7 => DivisorGenerator.SmallPrimesLastSevenWithoutLastThree,
		9 => DivisorGenerator.SmallPrimesLastNineWithoutLastThree,
		_ => [],
	};

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	internal static HeuristicDivisorEnumerator CreateHeuristicDivisorEnumerator(ulong maxDivisorSquare, byte nMod10, Span<HeuristicGroupBSequenceState> groupBBuffer)
	{
		return new HeuristicDivisorEnumerator(maxDivisorSquare, nMod10, groupBBuffer);
	}

	internal struct HeuristicGroupBSequenceState
	{
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	internal static MersenneHeuristicDivisorEnumerator CreateMersenneDivisorEnumerator(ulong exponent, ulong maxDivisor)
	{
		return new MersenneHeuristicDivisorEnumerator(exponent, maxDivisor);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	internal static HeuristicDivisorPreparation PrepareHeuristicDivisor(in HeuristicDivisorCandidate candidate)
	{
		ulong divisor = candidate.Value;
		MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(divisor);
		bool hasCycleHint = TryGetCycleLengthHint(divisor, out ulong cycleLength);
		return new HeuristicDivisorPreparation(candidate, divisorData, cycleLength, hasCycleHint);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	internal static ulong ResolveHeuristicCycleLength(
		ulong exponent,
		in HeuristicDivisorPreparation preparation,
		out bool cycleFromHint,
		out bool cycleComputed,
		out bool primeOrderFailed)
	{
		if (preparation.HasCycleLengthHint && preparation.CycleLengthHint != 0UL)
		{
			cycleFromHint = true;
			cycleComputed = true;
			primeOrderFailed = false;
			return preparation.CycleLengthHint;
		}

		ulong divisor = preparation.Candidate.Value;
		MontgomeryDivisorData divisorData = preparation.DivisorData;

		bool trySuccess = MersenneDivisorCycles.TryCalculateCycleLengthForExponentCpu(
			divisor,
			exponent,
			divisorData,
			out ulong computedCycle,
			out bool primeOrderFailedLocal);

		if (trySuccess && computedCycle != 0UL)
		{
			cycleFromHint = false;
			cycleComputed = true;
			primeOrderFailed = primeOrderFailedLocal;
			return computedCycle;
		}

		primeOrderFailed = primeOrderFailedLocal || !trySuccess || computedCycle == 0UL;
		ulong resolvedCycle = MersenneDivisorCycles.CalculateCycleLength(divisor, divisorData, skipPrimeOrderHeuristic: primeOrderFailed);
		cycleFromHint = false;
		cycleComputed = resolvedCycle != 0UL;
		return resolvedCycle;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	internal static bool TryGetCycleLengthHint(ulong divisor, out ulong cycleLength)
	{
		ulong[] snapshot = HeuristicSmallCycleSnapshot;
		if (divisor < (ulong)snapshot.Length)
		{
			cycleLength = snapshot[(int)divisor];
			if (cycleLength != 0UL)
			{
				return true;
			}
		}

		cycleLength = 0UL;
		return false;
	}

	internal ref struct HeuristicDivisorEnumerator
	{
		private readonly ulong maxDivisorSquare;
		private readonly ReadOnlySpan<ulong> combinedDivisors;
		private readonly ReadOnlySpan<ulong> combinedDivisorSquares;
		private int index;

		public HeuristicDivisorEnumerator(ulong maxDivisorSquare, byte nMod10, Span<HeuristicGroupBSequenceState> groupBBuffer)
		{
			this.maxDivisorSquare = maxDivisorSquare;
			_ = groupBBuffer;
			combinedDivisors = GetCombinedDivisors(nMod10);
			combinedDivisorSquares = GetCombinedDivisorSquares(nMod10);
			index = 0;
		}

		public bool TryGetNext(out HeuristicDivisorCandidate candidate)
		{
			while (index < combinedDivisors.Length)
			{
				int currentIndex = index;
				ulong divisorSquare = combinedDivisorSquares[currentIndex];
				if (divisorSquare > maxDivisorSquare)
				{
					index = combinedDivisors.Length;
					break;
				}

				ulong entry = combinedDivisors[currentIndex];
				index = currentIndex + 1;

				HeuristicDivisorGroup group = ResolveGroup(entry);
				ulong value = entry;
				byte ending = (byte)(value % 10UL);
				ushort residue = (ushort)(value % Wheel210);
				candidate = new HeuristicDivisorCandidate(value, group, ending, 0, residue);
				return true;
			}

			candidate = default;
			return false;
		}
	}

	internal ref struct MersenneHeuristicDivisorEnumerator
	{
		private readonly GpuUInt128 step;
		private readonly GpuUInt128 limit;
		private GpuUInt128 current;
		private MersenneDivisorResidueStepper residueStepper;
		private bool active;
		private ulong processedCount;
		private ulong lastDivisor;

		public MersenneHeuristicDivisorEnumerator(ulong exponent, ulong maxDivisor)
		{
			var stepLocal = new GpuUInt128(exponent);
			stepLocal.ShiftLeft(1);
			step = stepLocal;

			var limitLocal = new GpuUInt128(maxDivisor);
			limit = limitLocal;

			var firstDivisor = stepLocal;
			firstDivisor.Add(1UL);

			bool hasCandidates = !stepLocal.IsZero && firstDivisor.CompareTo(limitLocal) <= 0;

			current = hasCandidates ? firstDivisor : GpuUInt128.Zero;
			residueStepper = hasCandidates ? new MersenneDivisorResidueStepper(exponent, stepLocal, firstDivisor) : default;
			active = hasCandidates;
			processedCount = 0UL;
			lastDivisor = 0UL;
		}

		public bool TryGetNext(out HeuristicDivisorCandidate candidate)
		{
			while (active)
			{
				ulong value = current.Low;
				processedCount++;
				lastDivisor = value;

				bool admissible = residueStepper.IsAdmissible();

				Advance();

				if (admissible)
				{
					candidate = CreateCandidate(value);
					return true;
				}
			}

			candidate = default;
			return false;
		}

		private void Advance()
		{
			GpuUInt128 next = current;
			next.Add(step);

			if (next.CompareTo(limit) > 0 || next.CompareTo(current) <= 0)
			{
				active = false;
				return;
			}

			current = next;
			residueStepper.Advance();
		}

		public readonly ulong ProcessedCount => processedCount;

		public readonly ulong LastDivisor => lastDivisor;

		public readonly bool Exhausted => !active;

		private static HeuristicDivisorCandidate CreateCandidate(ulong value)
		{
			byte ending = (byte)(value % 10UL);
			HeuristicDivisorGroup group;

			if (value == 3UL || value == 7UL || value == 11UL || value == 13UL)
			{
				group = HeuristicDivisorGroup.GroupAConstant;
			}
			else if (ending == 3)
			{
				group = HeuristicDivisorGroup.GroupAWheel;
			}
			else
			{
				group = HeuristicDivisorGroup.GroupB;
			}

			ushort residue = (ushort)(value % Wheel210);
			return new HeuristicDivisorCandidate(value, group, ending, 0, residue);
		}
	}
}
