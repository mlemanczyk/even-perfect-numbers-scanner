using Open.Numeric.Primes;
using System.Runtime.CompilerServices;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu;
using PerfectNumbers.Core.Gpu.Accelerators;

namespace PerfectNumbers.Core;

public sealed class HeuristicPrimeTester
{
	static HeuristicPrimeTester()
	{
		HeuristicPrimeSieves.EnsureInitialized();
	}

	// TODO: Don't use these tiny arrays. Hard-code the values / checks instead
	private static readonly byte[] GroupBEndingOrderMod1 = [9, 1];
	private static readonly byte[] GroupBEndingOrderMod3 = [9, 7];
	private static readonly byte[] GroupBEndingOrderMod7 = [7, 1];
	private static readonly byte[] GroupBEndingOrderMod9 = [9, 7, 1];

	// TODO: Would it speed up things if we implement something like ExponentRemainderStepper,
	// MersenneDivisorResidueStepper, or CycleRemainderStepper work with these?
	private const ulong Wheel210 = 210UL;
	private const int GroupAConstantCount = 4;
	internal const int MaxGroupBSequences = 36;

	private static readonly bool UseHeuristicGroupBTrialDivision = false; // Temporary fallback gate for Group B.


	private static readonly ulong[] HeuristicSmallCycleSnapshot = MersenneDivisorCycles.Shared.ExportSmallCyclesSnapshot();

	[ThreadStatic]
	private static BidirectionalCycleRemainderStepper _mod10Stepper;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static byte GetMod10Remainder(ulong value)
	{
		if (_mod10Stepper.Modulus != 10UL)
		{
			_mod10Stepper = new BidirectionalCycleRemainderStepper(10UL);
			_mod10Stepper.Initialize(value);
			return (byte)_mod10Stepper.Step(value);
		}

		// Caller must initialize once per thread before first use.
		return (byte)_mod10Stepper.Step(value);
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
	private static HeuristicPrimeTester? _tester;

	public static HeuristicPrimeTester Exclusive => _tester ??= new();

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsPrimeCpu(ulong n)
	{
		byte nMod10 = GetMod10Remainder(n);
		ulong maxDivisorSquare = ComputeHeuristicDivisorSquareLimit(n);

		if (maxDivisorSquare < 9UL)
		{
			return EvaluateWithOpenNumericFallback(n);
		}

		bool includeGroupB = UseHeuristicGroupBTrialDivision;
		return HeuristicTrialDivisionCpu(n, maxDivisorSquare, nMod10, includeGroupB);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsPrimeGpu(PrimeOrderCalculatorAccelerator gpu, ulong n)
	{
		byte nMod10 = GetMod10Remainder(n);
		ulong maxDivisorSquare = ComputeHeuristicDivisorSquareLimit(n);
		return IsPrimeGpu(gpu, n, maxDivisorSquare, nMod10);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsPrimeGpu(PrimeOrderCalculatorAccelerator gpu, ulong n, ulong maxDivisorSquare, byte nMod10) => HeuristicIsPrimeGpuCore(gpu, n, maxDivisorSquare, nMod10);

	private static bool HeuristicIsPrimeGpuCore(PrimeOrderCalculatorAccelerator gpu, ulong n, ulong maxDivisorSquare, byte nMod10)
	{
		// TODO: Is this condition ever met on the execution path in EvenPerfectBitScanner?
		if (maxDivisorSquare < 9UL)
		{
			return EvaluateWithOpenNumericFallback(n);
		}

		// if (!UseHeuristicGroupBTrialDivision)
		// {
		//     return HeuristicTrialDivisionCpu(n, maxDivisorSquare, nMod10, includeGroupB: false);
		// }

		bool compositeDetected = HeuristicTrialDivisionGpuDetectsDivisor(gpu, n, maxDivisorSquare, nMod10);
		return !compositeDetected;
	}

	private static bool HeuristicTrialDivisionCpu(ulong n, ulong maxDivisorSquare, byte nMod10, bool includeGroupB)
	{
		ReadOnlySpan<ulong> groupADivisors = HeuristicPrimeSieves.GroupADivisors;
		ReadOnlySpan<ulong> groupADivisorSquares = HeuristicPrimeSieves.GroupADivisorSquares;
		int interleaveBatchSize = HeuristicCombinedPrimeTester.GetBatchSize(n);
		int groupAIndex = 0;

		if (!includeGroupB)
		{
			int groupALength = groupADivisors.Length;
			while (groupAIndex < groupALength)
			{
				int processed = 0;
				while (processed < interleaveBatchSize && groupAIndex < groupALength)
				{
					ulong divisorSquare = groupADivisorSquares[groupAIndex];
					if (divisorSquare > maxDivisorSquare)
					{
						groupAIndex = groupALength;
						break;
					}

                    ulong divisor = groupADivisors[groupAIndex];
                    if (n.ReduceCycleRemainder(divisor) == 0UL)
                    {
                        return false;
                    }

					groupAIndex++;
					processed++;
				}

				if (groupAIndex >= groupALength)
				{
					break;
				}

				if (groupADivisorSquares[groupAIndex] > maxDivisorSquare)
				{
					groupAIndex = groupALength;
					break;
				}
			}

			return EvaluateWithOpenNumericFallback(n);
		}

		ReadOnlySpan<byte> endingOrder = GetGroupBEndingOrder(nMod10);
		if (endingOrder.IsEmpty)
		{
			return EvaluateWithOpenNumericFallback(n);
		}

		Span<int> indices = endingOrder.Length <= 8
			? stackalloc int[endingOrder.Length]
			: new int[endingOrder.Length];

		bool groupAHasMore = groupAIndex < groupADivisors.Length && groupADivisorSquares[groupAIndex] <= maxDivisorSquare;
		bool groupBHasMore = HasGroupBCandidates(endingOrder, indices, maxDivisorSquare);

		while (groupAHasMore || groupBHasMore)
		{
			if (groupAHasMore)
			{
				int processed = 0;
				while (processed < interleaveBatchSize && groupAIndex < groupADivisors.Length)
				{
					ulong divisorSquare = groupADivisorSquares[groupAIndex];
					if (divisorSquare > maxDivisorSquare)
					{
						groupAIndex = groupADivisors.Length;
						groupAHasMore = false;
						break;
					}

					ulong divisor = groupADivisors[groupAIndex];
                    if (n.ReduceCycleRemainder(divisor) == 0UL)
                    {
                        return false;
                    }

					groupAIndex++;
					processed++;
				}

				if (groupAIndex >= groupADivisors.Length)
				{
					groupAHasMore = false;
				}
				else
				{
					groupAHasMore = groupADivisorSquares[groupAIndex] <= maxDivisorSquare;
					if (!groupAHasMore)
					{
						groupAIndex = groupADivisors.Length;
					}
				}
			}

			if (groupBHasMore)
			{
				int processed = 0;
				while (processed < interleaveBatchSize)
				{
					if (!TrySelectNextGroupBDivisor(endingOrder, indices, maxDivisorSquare, out ulong divisor))
					{
						groupBHasMore = false;
						break;
					}

					if (n.ReduceCycleRemainder(divisor) == 0UL)
					{
						return false;
					}

					processed++;
				}

				if (groupBHasMore)
				{
					groupBHasMore = HasGroupBCandidates(endingOrder, indices, maxDivisorSquare);
				}
			}
		}

		return true;
	}


	private static bool HeuristicTrialDivisionGpuDetectsDivisor(PrimeOrderCalculatorAccelerator gpu, ulong n, ulong maxDivisorSquare, byte nMod10)
	{
		int acceleratorIndex = gpu.AcceleratorIndex;
		var kernelLauncher = gpu.HeuristicCombinedTrialDivisionKernelLauncher;
		bool compositeDetected;
		int compositeFlag = 0;
		var flagView1DView = gpu.OutputIntView;
		var (divisors, divisorSquares) = gpu.DivisorTables.SelectDivisorsAndSquares(nMod10);

		var stream = AcceleratorStreamPool.Rent(acceleratorIndex);
		flagView1DView.CopyFromCPU(stream, ref compositeFlag, 1);

		kernelLauncher(
				stream,
				1,
				flagView1DView,
				n,
				maxDivisorSquare,
				divisors,
				divisorSquares);

		flagView1DView.CopyToCPU(stream, ref compositeFlag, 1);
		stream.Synchronize();

		AcceleratorStreamPool.Return(acceleratorIndex, stream);
		compositeDetected = compositeFlag != 0;

		return compositeDetected;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static bool EvaluateWithOpenNumericFallback(ulong n) => Prime.Numbers.IsPrime(n);

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	internal static ulong ComputeHeuristicDivisorSquareLimit(ulong n) => n;

	private static ReadOnlySpan<byte> GetGroupBEndingOrder(byte nMod10) => nMod10 switch
	{
		1 => GroupBEndingOrderMod1,
		3 => GroupBEndingOrderMod3,
		7 => GroupBEndingOrderMod7,
		9 => GroupBEndingOrderMod9,
		_ => [],
	};

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static bool TrySelectNextGroupBDivisor(ReadOnlySpan<byte> endingOrder, Span<int> indices, ulong maxDivisorSquare, out ulong divisor)
	{
		ulong bestCandidate = ulong.MaxValue;
		int bestEndingIndex = -1;

		for (int i = 0; i < endingOrder.Length; i++)
		{
			ReadOnlySpan<uint> divisors = GetGroupBDivisors(endingOrder[i]);
			ReadOnlySpan<ulong> squares = GetGroupBDivisorSquares(endingOrder[i]);
			int index = indices[i];
			if ((uint)index >= (uint)divisors.Length)
			{
				continue;
			}

			ulong candidateSquare = squares[index];
			if (candidateSquare > maxDivisorSquare)
			{
				indices[i] = divisors.Length;
				continue;
			}

			ulong candidate = divisors[index];
			if (bestEndingIndex == -1 || candidate < bestCandidate)
			{
				bestCandidate = candidate;
				bestEndingIndex = i;
			}
		}

		if (bestEndingIndex == -1)
		{
			divisor = 0UL;
			return false;
		}

		ReadOnlySpan<uint> selectedDivisors = GetGroupBDivisors(endingOrder[bestEndingIndex]);
		divisor = selectedDivisors[indices[bestEndingIndex]];
		indices[bestEndingIndex]++;
		return true;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static bool HasGroupBCandidates(ReadOnlySpan<byte> endingOrder, Span<int> indices, ulong maxDivisorSquare)
	{
		for (int i = 0; i < endingOrder.Length; i++)
		{
			ReadOnlySpan<uint> divisors = GetGroupBDivisors(endingOrder[i]);
			ReadOnlySpan<ulong> squares = GetGroupBDivisorSquares(endingOrder[i]);
			int index = indices[i];
			if ((uint)index >= (uint)divisors.Length)
			{
				continue;
			}

			if (squares[index] > maxDivisorSquare)
			{
				indices[i] = divisors.Length;
				continue;
			}

			return true;
		}

		return false;
	}



	private static ReadOnlySpan<uint> GetGroupBDivisors(byte ending) => ending switch
	{
		1 => DivisorGenerator.SmallPrimesLastOneWithoutLastThree,
		3 => [],
		7 => DivisorGenerator.SmallPrimesLastSevenWithoutLastThree,
		9 => DivisorGenerator.SmallPrimesLastNineWithoutLastThree,
		_ => [],
	};

	private static ReadOnlySpan<ulong> GetGroupBDivisorSquares(byte ending) => ending switch
	{
		1 => DivisorGenerator.SmallPrimesPow2LastOneWithoutLastThree,
		3 => [],
		7 => DivisorGenerator.SmallPrimesPow2LastSevenWithoutLastThree,
		9 => DivisorGenerator.SmallPrimesPow2LastNineWithoutLastThree,
		_ => [],
	};

	private static int InitializeGroupBStates(byte nMod10, Span<HeuristicGroupBSequenceState> buffer)
	{
		ReadOnlySpan<byte> endings = GetGroupBEndingOrder(nMod10);
		int count = endings.Length;

		for (int i = 0; i < count; i++)
		{
			buffer[i] = new HeuristicGroupBSequenceState(endings[i], (byte)i);
		}

		return count;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	internal static HeuristicDivisorEnumerator CreateHeuristicDivisorEnumerator(ulong maxDivisorSquare, byte nMod10, Span<HeuristicGroupBSequenceState> groupBBuffer) => new(maxDivisorSquare, nMod10, groupBBuffer);

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	internal static MersenneHeuristicDivisorEnumerator CreateMersenneDivisorEnumerator(ulong exponent, ulong maxDivisor)
		=> new(exponent, maxDivisor);

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	internal static HeuristicDivisorPreparation PrepareHeuristicDivisor(in HeuristicDivisorCandidate candidate)
	{
		ulong divisor = candidate.Value;
		MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(divisor);
		bool hasCycleHint = TryGetCycleLengthHint(divisor, out ulong cycleLength);
		return new HeuristicDivisorPreparation(candidate, divisorData, cycleLength, hasCycleHint);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	internal static ulong ResolveHeuristicCycleLengthCpu(
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
		ulong resolvedCycle = MersenneDivisorCycles.CalculateCycleLengthCpu(divisor, divisorData, skipPrimeOrderHeuristic: primeOrderFailed);
		cycleFromHint = false;
		cycleComputed = resolvedCycle != 0UL;
		return resolvedCycle;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	internal static ulong ResolveHeuristicCycleLengthGpu(
		PrimeOrderCalculatorAccelerator gpu,
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
		ulong resolvedCycle = MersenneDivisorCycles.CalculateCycleLengthGpu(gpu, divisor, divisorData, skipPrimeOrderHeuristic: primeOrderFailed);
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
		private readonly ReadOnlySpan<ulong> groupADivisors;
		private readonly ReadOnlySpan<ulong> groupADivisorSquares;
		private int groupAIndex;
		private Span<HeuristicGroupBSequenceState> groupBStates;
		private CycleRemainderStepper endingStepper;
		private CycleRemainderStepper wheelStepper;
		private bool hasRemainders;
		private ulong lastDivisor;

		public HeuristicDivisorEnumerator(ulong maxDivisorSquare, byte nMod10, Span<HeuristicGroupBSequenceState> groupBBuffer)
		{
			this.maxDivisorSquare = maxDivisorSquare;
			groupADivisors = HeuristicPrimeSieves.GroupADivisors;
			groupADivisorSquares = HeuristicPrimeSieves.GroupADivisorSquares;
			groupAIndex = 0;

			int count = InitializeGroupBStates(nMod10, groupBBuffer);
			groupBStates = count == 0 ? Span<HeuristicGroupBSequenceState>.Empty : groupBBuffer[..count];
			endingStepper = new CycleRemainderStepper(10UL);
			wheelStepper = new CycleRemainderStepper(Wheel210);
			hasRemainders = false;
			lastDivisor = 0UL;
		}

		public bool TryGetNext(out HeuristicDivisorCandidate candidate)
		{
			while (groupAIndex < groupADivisors.Length)
			{
				ulong divisorSquare = groupADivisorSquares[groupAIndex];
				if (divisorSquare > maxDivisorSquare)
				{
					groupAIndex = groupADivisors.Length;
					break;
				}

				int currentIndex = groupAIndex;
				ulong divisor = groupADivisors[currentIndex];
				groupAIndex++;

				HeuristicDivisorGroup group = currentIndex < GroupAConstantCount
					? HeuristicDivisorGroup.GroupAConstant
					: HeuristicDivisorGroup.GroupAWheel;
				(byte ending, ushort wheelResidue) = UpdateRemainders(divisor);

				candidate = new HeuristicDivisorCandidate(
					divisor,
					group,
					ending,
					0,
					wheelResidue);
				return true;
			}

			ulong bestCandidate = ulong.MaxValue;
			int bestIndex = -1;

			for (int i = 0; i < groupBStates.Length; i++)
			{
				ref HeuristicGroupBSequenceState state = ref groupBStates[i];
				if (!state.TryPeek(maxDivisorSquare, out ulong candidateValue))
				{
					continue;
				}

				if (bestIndex == -1 || candidateValue < bestCandidate ||
					(candidateValue == bestCandidate && state.PriorityIndex < groupBStates[bestIndex].PriorityIndex))
				{
					bestCandidate = candidateValue;
					bestIndex = i;
				}
			}

			if (bestIndex == -1)
			{
				candidate = default;
				return false;
			}

			ref HeuristicGroupBSequenceState bestState = ref groupBStates[bestIndex];
			bestState.Advance();
			(byte candidateEnding, ushort wheelResidueBest) = UpdateRemainders(bestCandidate);
			candidate = new HeuristicDivisorCandidate(
				bestCandidate,
				HeuristicDivisorGroup.GroupB,
				candidateEnding,
				bestState.PriorityIndex,
				wheelResidueBest);
			return true;
		}

		private (byte ending, ushort wheelResidue) UpdateRemainders(ulong divisor)
		{
			byte ending;
			ushort wheelResidue;
			if (!hasRemainders)
			{
				ending = (byte)endingStepper.Initialize(divisor);
				wheelResidue = (ushort)wheelStepper.Initialize(divisor);
				hasRemainders = true;
			}
			else
			{
				ending = (byte)endingStepper.ComputeNext(divisor);
				wheelResidue = (ushort)wheelStepper.ComputeNext(divisor);
			}

			lastDivisor = divisor;
			return (ending, wheelResidue);
		}
	}



	internal struct HeuristicGroupBSequenceState(byte ending, byte priorityIndex)
	{
		public byte Ending = ending;
		public byte PriorityIndex = priorityIndex;
		public int Index = 0;

		public bool TryPeek(ulong maxDivisorSquare, out ulong divisor)
		{
			ReadOnlySpan<uint> divisors = GetGroupBDivisors(Ending);
			ReadOnlySpan<ulong> squares = GetGroupBDivisorSquares(Ending);
			int index = Index;
			if (index >= divisors.Length)
			{
				divisor = 0UL;
				return false;
			}

			if (squares[index] > maxDivisorSquare)
			{
				Index = divisors.Length;
				divisor = 0UL;
				return false;
			}

			divisor = divisors[index];
			return true;
		}

		public void Advance()
		{
			int next = Index + 1;
			ReadOnlySpan<uint> divisors = GetGroupBDivisors(Ending);
			if ((uint)next >= (uint)divisors.Length)
			{
				Index = divisors.Length;
			}
			else
			{
				Index = next;
			}
		}
	}


	internal ref struct MersenneHeuristicDivisorEnumerator
	{
		private readonly GpuUInt128 step;
		private readonly GpuUInt128 limit;
		private GpuUInt128 current;
		private CycleRemainderStepper endingStepper;
		private CycleRemainderStepper wheelStepper;
		private byte currentEnding;
		private ushort currentWheelResidue;
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
			if (hasCandidates)
			{
				endingStepper = new CycleRemainderStepper(10UL);
				currentEnding = (byte)endingStepper.Initialize(firstDivisor.Low);

				wheelStepper = new CycleRemainderStepper(Wheel210);
				currentWheelResidue = (ushort)wheelStepper.Initialize(firstDivisor.Low);
			}
			else
			{
				endingStepper = default;
				wheelStepper = default;
				currentEnding = 0;
				currentWheelResidue = 0;
			}
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
					candidate = CreateCandidate(value, currentEnding, currentWheelResidue);
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
			currentEnding = (byte)endingStepper.ComputeNext(next.Low);
			currentWheelResidue = (ushort)wheelStepper.ComputeNext(next.Low);
		}

		public readonly ulong ProcessedCount => processedCount;

		public readonly ulong LastDivisor => lastDivisor;

		public readonly bool Exhausted => !active;

		private static HeuristicDivisorCandidate CreateCandidate(ulong value, byte ending, ushort wheelResidue)
		{
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

			return new HeuristicDivisorCandidate(value, group, ending, 0, wheelResidue);
		}
	}
}
