using System.Buffers;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu.Accelerators;
using PerfectNumbers.Core.Gpu.Kernels;

namespace PerfectNumbers.Core.Gpu;

internal enum GpuPow2ModStatus
{
	Success,
	Overflow,
	Unavailable,
}

internal static partial class PrimeOrderGpuHeuristics
{
	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	internal static void PreloadStaticTables(AcceleratorStream stream)
	{
		// _ = GetSmallPrimeDeviceCache(stream);
	}

	private static readonly ConcurrentDictionary<ulong, byte> OverflowedPrimes = new();
	private static readonly ConcurrentDictionary<UInt128, byte> OverflowedPrimesWide = new();

	private static PrimeOrderGpuCapability s_capability = PrimeOrderGpuCapability.Default;



	internal static ConcurrentDictionary<ulong, byte> OverflowRegistry => OverflowedPrimes;
	internal static ConcurrentDictionary<UInt128, byte> OverflowRegistryWide => OverflowedPrimesWide;

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	internal static void OverrideCapabilitiesForTesting(PrimeOrderGpuCapability capability)
	{
		s_capability = capability;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	internal static void ResetCapabilitiesForTesting()
	{
		s_capability = PrimeOrderGpuCapability.Default;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	public static bool TryPartialFactor(
		PrimeOrderCalculatorAccelerator gpu,
		ulong value,
		uint limit,
		Span<ulong> primeTargets,
		Span<int> exponentTargets,
		out int factorCount,
		out ulong remaining)
	{
		factorCount = 0;
		remaining = value;

		if (primeTargets.Length == 0 || exponentTargets.Length == 0)
		{
			return false;
		}

		primeTargets.Clear();
		exponentTargets.Clear();

		if (!TryLaunchPartialFactorKernel(
				gpu,
				value,
				limit,
				primeTargets,
				exponentTargets,
				out int extracted,
				out ulong leftover))
		{
			return false;
		}

		int capacity = Math.Min(primeTargets.Length, exponentTargets.Length);
		if (extracted > capacity)
		{
			factorCount = 0;
			remaining = value;
			return false;
		}

		factorCount = extracted;
		remaining = leftover;
		return true;
	}

	private static readonly Accelerator[] _accelerators = AcceleratorPool.Shared.Accelerators;

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static bool TryLaunchPartialFactorKernel(
		PrimeOrderCalculatorAccelerator gpu,
		ulong value,
		uint limit,
		Span<ulong> primeTargets,
		Span<int> exponentTargets,
		out int factorCount,
		out ulong remaining)
	{
		factorCount = 0;
		remaining = value;

		// GpuPrimeWorkLimiter.Acquire();
		int acceleratorIndex = gpu.AcceleratorIndex;
		var smallPrimesView = gpu.SmallPrimeFactorPrimes;
		var smallSquaresView = gpu.SmallPrimeFactorSquares;

		var kernelLauncher = gpu.PartialFactorKernelLauncher;

		int primeTargetsLength = primeTargets.Length;
		gpu.EnsurePartialFactorCapacity(primeTargetsLength);

		var factorBufferView = gpu.OutputUlongView;
		var exponentBufferView = gpu.OutputIntView;
		var countBufferView = gpu.OutputIntView2;
		var remainingBufferView = gpu.OutputUlongView2;
		var fullyFactoredBufferView = gpu.OutputByteView;

		// There is no need to clear these buffers because the kernel will always assign values within the required bounds. Keep it commented out.
		// factorBufferView.MemSetToZero(stream);
		// exponentBufferView.MemSetToZero(stream);
		// countBufferView.MemSetToZero(stream);
		// remainingBufferView.MemSetToZero(stream);
		// fullyFactoredBufferView.MemSetToZero(stream);

		AcceleratorStream stream = AcceleratorStreamPool.Rent(acceleratorIndex);

		kernelLauncher(
			stream,
			1,
			smallPrimesView,
			smallSquaresView,
			(int)smallPrimesView.Length,
			primeTargetsLength,
			value,
			limit,
			factorBufferView,
			exponentBufferView,
			countBufferView,
			remainingBufferView,
			fullyFactoredBufferView);

		byte fullyFactoredFlag = 0;
		factorBufferView.CopyToCPU(stream, primeTargets);
		exponentBufferView.CopyToCPU(stream, exponentTargets);
		countBufferView.CopyToCPU(stream, ref factorCount, 1);
		remainingBufferView.CopyToCPU(stream, ref remaining, 1);
		fullyFactoredBufferView.CopyToCPU(stream, ref fullyFactoredFlag, 1);

		stream.Synchronize();
		AcceleratorStreamPool.Return(acceleratorIndex, stream);

		factorCount = Math.Min(factorCount, primeTargetsLength);

		// GpuPrimeWorkLimiter.Release();
		return true;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	public static GpuPow2ModStatus TryPow2Mod(PrimeOrderCalculatorAccelerator gpu, ulong exponent, ulong prime, out ulong remainder, in MontgomeryDivisorData divisorData)
	{
		Span<ulong> exponents = stackalloc ulong[1];
		Span<ulong> remainders = stackalloc ulong[1];
		exponents[0] = exponent;

		GpuPow2ModStatus status = TryPow2ModBatch(gpu, exponents, prime, remainders, divisorData);
		remainder = remainders[0];
		return status;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	public static GpuPow2ModStatus TryPow2ModBatch(PrimeOrderCalculatorAccelerator gpu, ReadOnlySpan<ulong> exponents, ulong prime, Span<ulong> remainders, in MontgomeryDivisorData divisorData)
	{
		// This will never occur in production code
		// if (exponents.Length == 0)
		// {
		//     return GpuPow2ModStatus.Success;
		// }

		// This will never occur in production code
		// if (remainders.Length < exponents.Length)
		// {
		//     throw new ArgumentException("Remainder span is shorter than the exponent span.", nameof(remainders));
		// }

		// This will never occur in production code
		// if (remainders.Length > exponents.Length)
		//     throw new ArgumentException("Remainder span is longer than the exponent span.", nameof(remainders));

		// We need to clear more, but we save on allocations    
		// Span<ulong> target = remainders[..exponents.Length];
		remainders.Clear();

		// This will never occur in production code
		// if (prime <= 1UL)
		// {
		//     return GpuPow2ModStatus.Unavailable;
		// }

		ConcurrentDictionary<ulong, byte> overflowRegistry = OverflowedPrimes;

		if (overflowRegistry.ContainsKey(prime))
		{
			return GpuPow2ModStatus.Overflow;
		}

		PrimeOrderGpuCapability capability = s_capability;

		if (prime.GetBitLength() > capability.ModulusBits)
		{
			overflowRegistry[prime] = 0;
			return GpuPow2ModStatus.Overflow;
		}

		for (int i = 0; i < exponents.Length; i++)
		{
			if (exponents[i].GetBitLength() > capability.ExponentBits)
			{
				return GpuPow2ModStatus.Overflow;
			}
		}

		bool computed = TryComputeOnGpu(gpu, exponents, prime, divisorData, remainders);
		return computed ? GpuPow2ModStatus.Success : GpuPow2ModStatus.Unavailable;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	public static GpuPow2ModStatus TryPow2Mod(PrimeOrderCalculatorAccelerator gpu, in UInt128 exponent, in UInt128 prime, out UInt128 remainder)
	{
		Span<UInt128> exponents = stackalloc UInt128[1];
		Span<UInt128> remainders = stackalloc UInt128[1];
		exponents[0] = exponent;

		GpuPow2ModStatus status = TryPow2ModBatch(gpu, exponents, prime, remainders);
		remainder = remainders[0];
		return status;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	public static GpuPow2ModStatus TryPow2ModBatch(PrimeOrderCalculatorAccelerator gpu, ReadOnlySpan<UInt128> exponents, UInt128 prime, Span<UInt128> remainders)
	{
		return TryPow2ModBatchInternal(gpu, exponents, prime, remainders);
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	internal static bool TryCalculateOrder(
		PrimeOrderCalculatorAccelerator gpu,
		ulong prime,
		ulong? previousOrder,
		PrimeOrderCalculator.PrimeOrderCalculatorConfig config,
		in MontgomeryDivisorData divisorData,
		out ulong order)
	{
		order = 0UL;

		int acceleratorIndex = gpu.AcceleratorIndex;
		var smallPrimesView = gpu.SmallPrimeFactorPrimes;
		var smallSquaresView = gpu.SmallPrimeFactorSquares;
		var resultBufferView = gpu.OutputUlongView2;
		var statusBufferView = gpu.OutputByteView;

		uint limit = config.SmallFactorLimit == 0 ? uint.MaxValue : config.SmallFactorLimit;
		byte hasPrevious = previousOrder.HasValue ? (byte)1 : (byte)0;
		ulong previousValue = previousOrder ?? 0UL;

		var kernelConfig = new CalculateOrderKernelConfig(previousValue, hasPrevious, limit, config.MaxPowChecks, config.StrictMode);
		ref var buffers = ref gpu.CalculateOrderKernelBuffers;

		var kernelLauncher = gpu.CalculateOrderKernelLauncher;

		AcceleratorStream stream = AcceleratorStreamPool.Rent(acceleratorIndex);
		// TODO: Remove the cleaning after the order kernel is modified to always set the result.
		resultBufferView.MemSetToZero(stream);
		statusBufferView.MemSetToZero(stream);

		kernelLauncher(
			stream,
			1,
			prime,
			kernelConfig,
			divisorData.Modulus,
			divisorData.NPrime,
			divisorData.MontgomeryOne,
			divisorData.MontgomeryTwo,
			divisorData.MontgomeryTwoSquared,
			smallPrimesView,
			smallSquaresView,
			(int)smallPrimesView.Length,
			buffers);


		byte status = 0;
		statusBufferView.CopyToCPU(stream, ref status, 1);
		resultBufferView.CopyToCPU(stream, ref order, 1);
		stream.Synchronize();

		AcceleratorStreamPool.Return(acceleratorIndex, stream);

		PrimeOrderKernelStatus kernelStatus = (PrimeOrderKernelStatus)status;
		if (kernelStatus == PrimeOrderKernelStatus.Fallback)
		{
			return false;
		}

		if (kernelStatus == PrimeOrderKernelStatus.PollardOverflow)
		{
			throw new InvalidOperationException("GPU Pollard Rho stack overflow; increase HeuristicStackCapacity.");
		}

		if (kernelStatus == PrimeOrderKernelStatus.FactoringFailure)
		{
			order = 0UL;
		}

		return order != 0UL;
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	private static GpuPow2ModStatus TryPow2ModBatchInternal(PrimeOrderCalculatorAccelerator gpu, ReadOnlySpan<UInt128> exponents, UInt128 prime, Span<UInt128> remainders)
	{
		if (exponents.Length == 0)
		{
			return GpuPow2ModStatus.Success;
		}

		if (remainders.Length < exponents.Length)
		{
			throw new ArgumentException("Remainder span is shorter than the exponent span.", nameof(remainders));
		}

		Span<UInt128> target = remainders[..exponents.Length];
		target.Clear();

		ConcurrentDictionary<UInt128, byte> overflowRegistryWide = OverflowedPrimesWide;

		if (overflowRegistryWide.ContainsKey(prime))
		{
			return GpuPow2ModStatus.Overflow;
		}

		PrimeOrderGpuCapability capability = s_capability;

		if (prime.GetBitLength() > capability.ModulusBits)
		{
			overflowRegistryWide[prime] = 0;
			return GpuPow2ModStatus.Overflow;
		}

		for (int i = 0; i < exponents.Length; i++)
		{
			if (exponents[i].GetBitLength() > capability.ExponentBits)
			{
				return GpuPow2ModStatus.Overflow;
			}
		}

		ComputeOnGpuWide(gpu, exponents, prime, target);
		return GpuPow2ModStatus.Success;
	}

	private static bool TryComputeOnGpu(PrimeOrderCalculatorAccelerator gpu, ReadOnlySpan<ulong> exponents, ulong prime, MontgomeryDivisorData divisorData, Span<ulong> results)
	{
		// GpuPrimeWorkLimiter.Acquire();
		int acceleratorIndex = gpu.AcceleratorIndex;
		Accelerator accelerator = gpu.Accelerator;
		var kernelLauncher = gpu.Pow2ModKernelLauncher;

		// Modify the callers to use their own pool of buffers per accelerator, so that other threads don't use the accelerators
		// with out preallocated buffers. Share the pool with Pow2ModWide kernel.

		gpu.EnsureUlongInputOutputCapacity(exponents.Length);

		var exponentBufferView = gpu.InputView;
		var remainderBufferView = gpu.OutputUlongView;

		// if (divisorData.Equals(MontgomeryDivisorData.Empty) is null)
		// if (divisorData is null)
		// {
		// 	divisorData = MontgomeryDivisorData.Empty;
		// }

		AcceleratorStream? stream = AcceleratorStreamPool.Rent(acceleratorIndex);
		exponentBufferView.CopyFromCPU(stream, exponents);

		// Pow2Mod kernel always assigns the required output elements so we don't need to worry about clearing these.
		// remainderBuffer.MemSetToZero(stream);

		kernelLauncher(stream, exponents.Length, exponentBufferView, divisorData.Modulus, divisorData.NPrime, divisorData.MontgomeryOne, divisorData.MontgomeryTwo, divisorData.MontgomeryTwoSquared, remainderBufferView);

		remainderBufferView.CopyToCPU(stream, results);
		stream.Synchronize();

		AcceleratorStreamPool.Return(acceleratorIndex, stream);
		// GpuPrimeWorkLimiter.Release();

		return true;
	}

	private static void ComputePow2ModCpu(ReadOnlySpan<ulong> exponents, ulong prime, in MontgomeryDivisorDataGpu divisorData, Span<ulong> results)
	{
		int length = exponents.Length;
		for (int i = 0; i < length; i++)
		{
			results[i] = Pow2ModCpu(exponents[i], prime, divisorData);
		}
	}

	private static void ComputeOnGpuWide(PrimeOrderCalculatorAccelerator gpu, ReadOnlySpan<UInt128> exponents, UInt128 prime, Span<UInt128> results)
	{
		int length = exponents.Length;

		GpuUInt128[]? rentedExponents = null;
		GpuUInt128[]? rentedResults = null;
		ArrayPool<GpuUInt128> gpuUInt128Pool = ThreadStaticPools.GpuUInt128Pool;
		bool poolingRequired = length > PrimeOrderConstants.WideStackThreshold;
		Span<GpuUInt128> exponentSpan = poolingRequired
			? stackalloc GpuUInt128[length]
			: new Span<GpuUInt128>(rentedExponents = gpuUInt128Pool.Rent(length), 0, length);
		Span<GpuUInt128> resultSpan = length <= PrimeOrderConstants.WideStackThreshold
			? stackalloc GpuUInt128[length]
			: new Span<GpuUInt128>(rentedResults = gpuUInt128Pool.Rent(length), 0, length);


		for (int i = 0; i < length; i++)
		{
			exponentSpan[i] = (GpuUInt128)exponents[i];
		}

		GpuUInt128 modulus = (GpuUInt128)prime;
		int acceleratorIndex = gpu.AcceleratorIndex;
		Accelerator accelerator = gpu.Accelerator;

		gpu.EnsureCalculateOrderWideCapacity(length);

		var exponentBufferView = gpu.CalculateOrderWideExponentBufferView;
		var remainderBufferView = gpu.CalculateOrderWideRemainderBufferView;

		AcceleratorStream stream = AcceleratorStreamPool.Rent(acceleratorIndex);
		exponentBufferView.CopyFromCPU(stream, exponentSpan);
		var kernelLauncher = gpu.Pow2ModWideKernelLauncher;

		kernelLauncher(stream, length, exponentBufferView, modulus, remainderBufferView);
		remainderBufferView.CopyToCPU(stream, ref MemoryMarshal.GetReference(resultSpan), length);
		stream.Synchronize();

		AcceleratorStreamPool.Return(acceleratorIndex, stream);

		for (int i = 0; i < length; i++)
		{
			results[i] = (UInt128)resultSpan[i];
		}

		if (rentedExponents is not null)
		{
			gpuUInt128Pool.Return(rentedExponents, clearArray: false);
			gpuUInt128Pool.Return(rentedResults!, clearArray: false);
		}
	}

	private static void ComputePow2ModCpuWide(ReadOnlySpan<UInt128> exponents, UInt128 prime, Span<UInt128> results)
	{
		int length = exponents.Length;
		for (int i = 0; i < length; i++)
		{
			results[i] = exponents[i].Pow2MontgomeryModWindowed(prime);
		}
	}

	private static ulong Pow2ModCpu(ulong exponent, ulong modulus, in MontgomeryDivisorDataGpu divisorData)
	{
		if (modulus <= 1UL)
		{
			return 0UL;
		}

		return divisorData.Pow2MontgomeryModWindowedGpuConvertToStandard(exponent);
	}

	internal readonly record struct PrimeOrderGpuCapability(int ModulusBits, int ExponentBits)
	{
		public static PrimeOrderGpuCapability Default => new(128, 128);
	}
}
