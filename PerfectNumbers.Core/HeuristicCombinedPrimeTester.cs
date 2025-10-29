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

		CombinedDivisorsEnding1TwoAOneB = BuildCombinedDivisors(1, CombinedDivisorPattern.TwoAOneB);
		CombinedDivisorsEnding3TwoAOneB = BuildCombinedDivisors(3, CombinedDivisorPattern.TwoAOneB);
		CombinedDivisorsEnding7TwoAOneB = BuildCombinedDivisors(7, CombinedDivisorPattern.TwoAOneB);
		CombinedDivisorsEnding9TwoAOneB = BuildCombinedDivisors(9, CombinedDivisorPattern.TwoAOneB);

		CombinedDivisorsEnding1OneAOneB = BuildCombinedDivisors(1, CombinedDivisorPattern.OneAOneB);
		CombinedDivisorsEnding3OneAOneB = BuildCombinedDivisors(3, CombinedDivisorPattern.OneAOneB);
		CombinedDivisorsEnding7OneAOneB = BuildCombinedDivisors(7, CombinedDivisorPattern.OneAOneB);
		CombinedDivisorsEnding9OneAOneB = BuildCombinedDivisors(9, CombinedDivisorPattern.OneAOneB);

		CombinedDivisorsEnding1ThreeAOneB = BuildCombinedDivisors(1, CombinedDivisorPattern.ThreeAOneB);
		CombinedDivisorsEnding3ThreeAOneB = BuildCombinedDivisors(3, CombinedDivisorPattern.ThreeAOneB);
		CombinedDivisorsEnding7ThreeAOneB = BuildCombinedDivisors(7, CombinedDivisorPattern.ThreeAOneB);
		CombinedDivisorsEnding9ThreeAOneB = BuildCombinedDivisors(9, CombinedDivisorPattern.ThreeAOneB);

		CombinedDivisorsEnding1ThreeATwoB = BuildCombinedDivisors(1, CombinedDivisorPattern.ThreeATwoB);
		CombinedDivisorsEnding3ThreeATwoB = BuildCombinedDivisors(3, CombinedDivisorPattern.ThreeATwoB);
		CombinedDivisorsEnding7ThreeATwoB = BuildCombinedDivisors(7, CombinedDivisorPattern.ThreeATwoB);
		CombinedDivisorsEnding9ThreeATwoB = BuildCombinedDivisors(9, CombinedDivisorPattern.ThreeATwoB);
		CombinedDivisorsEnding1 = CombinedDivisorsEnding1OneAOneB;
		CombinedDivisorsEnding3 = CombinedDivisorsEnding3OneAOneB;
		CombinedDivisorsEnding7 = CombinedDivisorsEnding7OneAOneB;
		CombinedDivisorsEnding9 = CombinedDivisorsEnding9OneAOneB;
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
		if (combinedDivisors.IsEmpty)
		{
			return EvaluateWithOpenNumericFallback(n);
		}

		int length = combinedDivisors.Length;
		for (int i = 0; i < length; i++)
		{
			uint entry = combinedDivisors[i];
			if (entry > sqrtLimit)
			{
				break;
			}

			if (n % entry == 0UL)
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
		var hitPool = ThreadStaticPools.BytePool;

		ulong[] divisorArray = divisorPool.Rent(batchCapacity);
		byte[] hitFlags = hitPool.Rent(batchCapacity);

		var limiter = GpuPrimeWorkLimiter.Acquire();
		var gpu = PrimeTester.PrimeTesterGpuContextPool.Rent();
		var accelerator = gpu.Accelerator;
		var state = PrimeTester.GpuKernelState.GetOrCreate(accelerator);
		var scratch = state.RentScratch(batchCapacity, accelerator);

		bool compositeDetected = false;
		int count = 0;

		var input = scratch.Input;
		var output = scratch.Output;

		bool ProcessBatch(int length)
		{
			input.View.CopyFromCPU(ref divisorArray[0], length);
			state.HeuristicTrialDivisionKernel(length, input.View, n, output.View);
			accelerator.Synchronize();
			output.View.CopyToCPU(ref hitFlags[0], length);

			for (int i = 0; i < length; i++)
			{
				if (hitFlags[i] == 0)
				{
					continue;
				}

				return true;
				// ulong divisor = divisorArray[i];
				// if (divisor > 1UL && n % divisor == 0UL)
				// {
				// 	return true;
				// }
			}

			return false;
		}

		ReadOnlySpan<uint> combinedDivisors = GetCombinedDivisors(nMod10);
		if (combinedDivisors.IsEmpty)
		{
			goto Cleanup;
		}

		int length = combinedDivisors.Length;
		for (int i = 0; i < length; i++)
		{
			uint entry = combinedDivisors[i];
			if (entry > sqrtLimit)
			{
				break;
			}

			divisorArray[count] = entry;
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

		if (!compositeDetected && count > 0)
		{
			if (ProcessBatch(count))
			{
				compositeDetected = true;
				goto Cleanup;
			}
		}

	Cleanup:
		state.ReturnScratch(scratch);
		divisorPool.Return(divisorArray);
		hitPool.Return(hitFlags);
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

	private static uint[] BuildCombinedDivisors(byte nMod10, CombinedDivisorPattern pattern)
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

		return combined.ToArray();
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
