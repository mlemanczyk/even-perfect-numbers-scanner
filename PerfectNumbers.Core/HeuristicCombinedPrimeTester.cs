using System;
using Open.Numeric.Primes;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
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

	private const ulong Wheel210 = 210UL;

	private static readonly bool UseHeuristicGroupBTrialDivision = false; // Temporary fallback gate for Group B.

	private static readonly uint[] CombinedDivisorsEnding1TwoAOneB;
	private static readonly uint[] CombinedDivisorsEnding3TwoAOneB;
	private static readonly uint[] CombinedDivisorsEnding7TwoAOneB;
	private static readonly uint[] CombinedDivisorsEnding9TwoAOneB;

	private static readonly uint[] CombinedDivisorsEnding1OneAOneB;
	private static readonly uint[] CombinedDivisorsEnding3OneAOneB;
	private static readonly uint[] CombinedDivisorsEnding7OneAOneB;
	private static readonly uint[] CombinedDivisorsEnding9OneAOneB;

	private static readonly uint[] CombinedDivisorsEnding1ThreeAOneB;
	private static readonly uint[] CombinedDivisorsEnding3ThreeAOneB;
	private static readonly uint[] CombinedDivisorsEnding7ThreeAOneB;
	private static readonly uint[] CombinedDivisorsEnding9ThreeAOneB;

	private static readonly uint[] CombinedDivisorsEnding1ThreeATwoB;
	private static readonly uint[] CombinedDivisorsEnding3ThreeATwoB;
	private static readonly uint[] CombinedDivisorsEnding7ThreeATwoB;
	private static readonly uint[] CombinedDivisorsEnding9ThreeATwoB;

	private static readonly uint[] CombinedDivisorsEnding1;
	private static readonly uint[] CombinedDivisorsEnding3;
	private static readonly uint[] CombinedDivisorsEnding7;
	private static readonly uint[] CombinedDivisorsEnding9;

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


	internal static int HeuristicGpuDivisorBatchSize = 4_096;
	internal static int HeuristicDivisorInterleaveBatchSize = 64;

	[ThreadStatic]
	private static HeuristicCombinedPrimeTester? _tester;

	public static HeuristicCombinedPrimeTester Exclusive => _tester ??= new();

	public static bool IsPrimeCpu(ulong n)
	{
		byte nMod10 = (byte)n.Mod10();
		ulong sqrtLimit = ComputeHeuristicSqrt(n);

		if (sqrtLimit < 3UL)
		{
			return EvaluateWithOpenNumericFallback(n);
		}

		return HeuristicTrialDivisionCpu(n, sqrtLimit, nMod10);
	}

	public static bool IsPrimeGpu(ulong n)
	{
		byte nMod10 = (byte)n.Mod10();
		ulong sqrtLimit = ComputeHeuristicSqrt(n);
		return IsPrimeGpu(n, sqrtLimit, nMod10);
	}

	public static bool IsPrimeGpu(ulong n, ulong sqrtLimit, byte nMod10)
	{
		// TODO: Is this condition ever met on the execution path in EvenPerfectBitScanner?
		if (sqrtLimit < 3UL)
		{
			return EvaluateWithOpenNumericFallback(n);
		}

		bool compositeDetected = HeuristicTrialDivisionGpuDetectsDivisor(n, sqrtLimit, nMod10);
		return !compositeDetected;
	}

	private static bool HeuristicTrialDivisionCpu(ulong n, ulong sqrtLimit, byte nMod10)
	{
		ReadOnlySpan<uint> combinedDivisors = GetCombinedDivisors(nMod10);
		ReadOnlySpan<ulong> combinedDivisorSquares = GetCombinedDivisorSquares(nMod10);
		if (combinedDivisors.IsEmpty || combinedDivisorSquares.IsEmpty)
		{
			return EvaluateWithOpenNumericFallback(n);
		}

		int length = combinedDivisors.Length;
		int squaresLength = combinedDivisorSquares.Length;
		if (squaresLength < length)
		{
			length = squaresLength;
		}

		for (int i = 0; i < length; i++)
		{
			ulong divisorSquare = combinedDivisorSquares[i];
			if (divisorSquare > n)
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

	private static bool HeuristicTrialDivisionGpuDetectsDivisor(ulong n, ulong sqrtLimit, byte nMod10)
	{
	        int batchCapacity = HeuristicGpuDivisorBatchSize;
	        var divisorPool = ThreadStaticPools.UlongPool;

	        ulong[] divisorArray = divisorPool.Rent(batchCapacity);

	        var limiter = GpuPrimeWorkLimiter.Acquire();
	        var gpu = PrimeTester.PrimeTesterGpuContextPool.Rent(batchCapacity, 1);
	        var accelerator = gpu.Accelerator;
	        var state = gpu.State;
	        var scratch = gpu.Scratch;

	        bool compositeDetected = false;
	        int count = 0;

	        var inputView = scratch.Input.View;
	        var outputView = scratch.Output.View;

	        bool ProcessBatch(int length)
	        {
	                inputView.CopyFromCPU(ref divisorArray[0], length);
	                byte compositeFlag = 0;
	                outputView.CopyFromCPU(ref compositeFlag, 1);
	                state.HeuristicTrialDivisionKernel(length, inputView, n, outputView);
	                accelerator.Synchronize();
	                outputView.CopyToCPU(ref compositeFlag, 1);

	                return compositeFlag != 0;
	        }

	        ReadOnlySpan<uint> combinedDivisors = GetCombinedDivisors(nMod10);
                ReadOnlySpan<ulong> combinedDivisorSquares = GetCombinedDivisorSquares(nMod10);

                if (combinedDivisors.IsEmpty || combinedDivisorSquares.IsEmpty)
                {
                        goto Cleanup;
                }

                int length = combinedDivisors.Length;
                int squaresLength = combinedDivisorSquares.Length;
                if (squaresLength < length)
                {
                        length = squaresLength;
                }

                for (int i = 0; i < length; i++)
                {
                        ulong divisorSquare = combinedDivisorSquares[i];
                        if (divisorSquare > n)
                        {
                                break;
                        }

                        divisorArray[count] = combinedDivisors[i];
                        count++;

                        if (count == batchCapacity)
                        {
                                if (ProcessBatch(count))
                                {
                                        compositeDetected = true;
                                        goto Cleanup;
                                }

                                count = 0;
                        }
                }

                if (count > 0 && ProcessBatch(count))
                {
                        compositeDetected = true;
                        goto Cleanup;
                }


Cleanup:
	        divisorPool.Return(divisorArray);
	        gpu.Dispose();
	        limiter.Dispose();
	        return compositeDetected;
	}


	private static bool EvaluateWithOpenNumericFallback(ulong n)
	{
		return Prime.Numbers.IsPrime(n);
	}

	internal static ulong ComputeHeuristicSqrt(ulong n)
	{
		ulong sqrt = (ulong)Math.Sqrt(n);
		UInt128 square = (UInt128)sqrt * sqrt;

		while (square > n)
		{
			sqrt--;
			square = (UInt128)sqrt * sqrt;
		}

		ulong next = sqrt + 1UL;
		UInt128 nextSquare = (UInt128)next * next;
		while (nextSquare <= n)
		{
			sqrt = next;
			next++;
			nextSquare = (UInt128)next * next;
		}

		return sqrt;
	}

	private static ReadOnlySpan<uint> GetCombinedDivisors(byte nMod10)
	{
		return nMod10 switch
		{
			1 => CombinedDivisorsEnding1,
			3 => CombinedDivisorsEnding3,
			7 => CombinedDivisorsEnding7,
			9 => CombinedDivisorsEnding9,
			_ => ReadOnlySpan<uint>.Empty,
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
			_ => ReadOnlySpan<ulong>.Empty,
		};
	}

	internal static ReadOnlySpan<uint> GetCombinedDivisors(byte nMod10, CombinedDivisorPattern pattern)
	{
		return pattern switch
		{
			CombinedDivisorPattern.TwoAOneB => nMod10 switch
			{
				1 => CombinedDivisorsEnding1TwoAOneB,
				3 => CombinedDivisorsEnding3TwoAOneB,
				7 => CombinedDivisorsEnding7TwoAOneB,
				9 => CombinedDivisorsEnding9TwoAOneB,
				_ => ReadOnlySpan<uint>.Empty,
			},
			CombinedDivisorPattern.OneAOneB => nMod10 switch
			{
				1 => CombinedDivisorsEnding1OneAOneB,
				3 => CombinedDivisorsEnding3OneAOneB,
				7 => CombinedDivisorsEnding7OneAOneB,
				9 => CombinedDivisorsEnding9OneAOneB,
				_ => ReadOnlySpan<uint>.Empty,
			},
			CombinedDivisorPattern.ThreeAOneB => nMod10 switch
			{
				1 => CombinedDivisorsEnding1ThreeAOneB,
				3 => CombinedDivisorsEnding3ThreeAOneB,
				7 => CombinedDivisorsEnding7ThreeAOneB,
				9 => CombinedDivisorsEnding9ThreeAOneB,
				_ => ReadOnlySpan<uint>.Empty,
			},
			CombinedDivisorPattern.ThreeATwoB => nMod10 switch
			{
				1 => CombinedDivisorsEnding1ThreeATwoB,
				3 => CombinedDivisorsEnding3ThreeATwoB,
				7 => CombinedDivisorsEnding7ThreeATwoB,
				9 => CombinedDivisorsEnding9ThreeATwoB,
				_ => ReadOnlySpan<uint>.Empty,
			},
			_ => ReadOnlySpan<uint>.Empty,
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
				_ => ReadOnlySpan<ulong>.Empty,
			},
			CombinedDivisorPattern.OneAOneB => nMod10 switch
			{
				1 => CombinedDivisorsEnding1OneAOneBSquares,
				3 => CombinedDivisorsEnding3OneAOneBSquares,
				7 => CombinedDivisorsEnding7OneAOneBSquares,
				9 => CombinedDivisorsEnding9OneAOneBSquares,
				_ => ReadOnlySpan<ulong>.Empty,
			},
			CombinedDivisorPattern.ThreeAOneB => nMod10 switch
			{
				1 => CombinedDivisorsEnding1ThreeAOneBSquares,
				3 => CombinedDivisorsEnding3ThreeAOneBSquares,
				7 => CombinedDivisorsEnding7ThreeAOneBSquares,
				9 => CombinedDivisorsEnding9ThreeAOneBSquares,
				_ => ReadOnlySpan<ulong>.Empty,
			},
			CombinedDivisorPattern.ThreeATwoB => nMod10 switch
			{
				1 => CombinedDivisorsEnding1ThreeATwoBSquares,
				3 => CombinedDivisorsEnding3ThreeATwoBSquares,
				7 => CombinedDivisorsEnding7ThreeATwoBSquares,
				9 => CombinedDivisorsEnding9ThreeATwoBSquares,
				_ => ReadOnlySpan<ulong>.Empty,
			},
			_ => ReadOnlySpan<ulong>.Empty,
		};
	}

	private static uint[] BuildCombinedDivisors(byte nMod10, CombinedDivisorPattern pattern, out ulong[] squares)
	{
		ReadOnlySpan<int> groupA = HeuristicPrimeSieves.GroupADivisors;
		ReadOnlySpan<uint> groupB = GetGroupBDivisors(nMod10);

		var combined = new List<uint>(groupA.Length + groupB.Length);

		int groupAIndex = 0;
		while (groupAIndex < groupA.Length && groupA[groupAIndex] <= 19)
		{
			combined.Add((uint)groupA[groupAIndex]);
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
				combined.Add((uint)groupA[groupAIndex]);
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

		uint[] result = combined.ToArray();
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

	private static bool IsGroupAPrefixValue(uint value) => value is 3U or 7U or 11U or 13U;

	private static HeuristicDivisorGroup ResolveGroup(uint value) => value switch
	{
		3U or 7U or 11U or 13U => HeuristicDivisorGroup.GroupAConstant,
		_ when value % 10U == 3U => HeuristicDivisorGroup.GroupAWheel,
		_ => HeuristicDivisorGroup.GroupB,
	};

	private static ReadOnlySpan<uint> GetGroupBDivisors(byte ending) => ending switch
	{
		1 => DivisorGenerator.SmallPrimesLastOneWithoutLastThree,
		3 => ReadOnlySpan<uint>.Empty,
		7 => DivisorGenerator.SmallPrimesLastSevenWithoutLastThree,
		9 => DivisorGenerator.SmallPrimesLastNineWithoutLastThree,
		_ => ReadOnlySpan<uint>.Empty,
	};

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	internal static HeuristicDivisorEnumerator CreateHeuristicDivisorEnumerator(ulong sqrtLimit, byte nMod10, Span<HeuristicGroupBSequenceState> groupBBuffer)
	{
		return new HeuristicDivisorEnumerator(sqrtLimit, nMod10, groupBBuffer);
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
		private readonly ulong sqrtLimit;
		private readonly ReadOnlySpan<uint> combinedDivisors;
		private int index;

		public HeuristicDivisorEnumerator(ulong sqrtLimit, byte nMod10, Span<HeuristicGroupBSequenceState> groupBBuffer)
		{
			this.sqrtLimit = sqrtLimit;
			_ = groupBBuffer;
			combinedDivisors = GetCombinedDivisors(nMod10);
			index = 0;
		}

		public bool TryGetNext(out HeuristicDivisorCandidate candidate)
		{
			while (index < combinedDivisors.Length)
			{
				uint entry = combinedDivisors[index++];
				if (entry > sqrtLimit)
				{
					index = combinedDivisors.Length;
					break;
				}

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

		public ulong ProcessedCount => processedCount;

		public ulong LastDivisor => lastDivisor;

		public bool Exhausted => !active;

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
