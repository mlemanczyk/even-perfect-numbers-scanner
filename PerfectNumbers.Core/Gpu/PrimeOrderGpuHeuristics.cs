using System.Buffers;
using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;
using System.Runtime.InteropServices;
using ILGPU.Runtime.OpenCL;
namespace PerfectNumbers.Core.Gpu;

internal enum GpuPow2ModStatus
{
	Success,
	Overflow,
	Unavailable,
}

internal static partial class PrimeOrderGpuHeuristics
{
	internal static void PreloadStaticTables(AcceleratorStream stream)
	{
		// _ = GetSmallPrimeDeviceCache(stream);
	}

	private static readonly ConcurrentDictionary<ulong, byte> OverflowedPrimes = new();
	private static readonly ConcurrentDictionary<UInt128, byte> OverflowedPrimesWide = new();

	[ThreadStatic]
	private static Dictionary<Accelerator, Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>>? _pow2ModKernel;

	[ThreadStatic]
	private static Dictionary<Accelerator, Action<AcceleratorStream, Index1D, ArrayView1D<GpuUInt128, Stride1D.Dense>, GpuUInt128, ArrayView1D<GpuUInt128, Stride1D.Dense>>>? _pow2ModWideKernel;

	[ThreadStatic]
	private static Dictionary<Accelerator, Action<AcceleratorStream, Index1D, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, int, ulong, uint, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<byte, Stride1D.Dense>>>? _partialFactorKernel;

	[ThreadStatic]
	private static Dictionary<Accelerator, OrderKernelLauncher>? _orderKernel;

	[ThreadStatic]
	private static Dictionary<Accelerator, SmallPrimeDeviceCache>? _smallPrimeDeviceCache;

	public readonly struct OrderKernelConfig(ulong previousOrder, byte hasPreviousOrder, uint smallFactorLimit, int maxPowChecks, int mode)
	{
		public readonly ulong PreviousOrder = previousOrder;
		public readonly byte HasPreviousOrder = hasPreviousOrder;
		public readonly uint SmallFactorLimit = smallFactorLimit;
		public readonly int MaxPowChecks = maxPowChecks;
		public readonly int Mode = mode;
	}

	public readonly struct OrderKernelBuffers(
		ArrayView1D<ulong, Stride1D.Dense> phiFactors,
		ArrayView1D<int, Stride1D.Dense> phiExponents,
		ArrayView1D<ulong, Stride1D.Dense> workFactors,
		ArrayView1D<int, Stride1D.Dense> workExponents,
		ArrayView1D<ulong, Stride1D.Dense> candidates,
		ArrayView1D<int, Stride1D.Dense> stackIndex,
		ArrayView1D<int, Stride1D.Dense> stackExponent,
		ArrayView1D<ulong, Stride1D.Dense> stackProduct,
		ArrayView1D<ulong, Stride1D.Dense> result,
		ArrayView1D<byte, Stride1D.Dense> status)
	{
		public readonly ArrayView1D<ulong, Stride1D.Dense> PhiFactors = phiFactors;
		public readonly ArrayView1D<int, Stride1D.Dense> PhiExponents = phiExponents;
		public readonly ArrayView1D<ulong, Stride1D.Dense> WorkFactors = workFactors;
		public readonly ArrayView1D<int, Stride1D.Dense> WorkExponents = workExponents;
		public readonly ArrayView1D<ulong, Stride1D.Dense> Candidates = candidates;
		public readonly ArrayView1D<int, Stride1D.Dense> StackIndex = stackIndex;
		public readonly ArrayView1D<int, Stride1D.Dense> StackExponent = stackExponent;
		public readonly ArrayView1D<ulong, Stride1D.Dense> StackProduct = stackProduct;
		public readonly ArrayView1D<ulong, Stride1D.Dense> Result = result;
		public readonly ArrayView1D<byte, Stride1D.Dense> Status = status;
	}

	private delegate void OrderKernelLauncher(
		AcceleratorStream stream,
		Index1D extent,
		ulong prime,
		OrderKernelConfig config,
		MontgomeryDivisorData divisor,
		ArrayView1D<uint, Stride1D.Dense> primes,
		ArrayView1D<ulong, Stride1D.Dense> squares,
		int primeCount,
		OrderKernelBuffers buffers);

	private sealed class SmallPrimeDeviceCache
	{
		public MemoryBuffer1D<uint, Stride1D.Dense>? Primes;
		public MemoryBuffer1D<ulong, Stride1D.Dense>? Squares;
		public int Count;
	}

	private const int WideStackThreshold = 8;
	private static PrimeOrderGpuCapability s_capability = PrimeOrderGpuCapability.Default;

	private const int Pow2WindowSizeBits = 8;
	private const int Pow2WindowOddPowerCount = 1 << (Pow2WindowSizeBits - 1);
	private const ulong Pow2WindowFallbackThreshold = 32UL;
	private const int HeuristicCandidateLimit = 512;
	private const int HeuristicStackCapacity = 256;

	private const int GpuSmallPrimeFactorSlots = 64;

	internal static ConcurrentDictionary<ulong, byte> OverflowRegistry => OverflowedPrimes;
	internal static ConcurrentDictionary<UInt128, byte> OverflowRegistryWide => OverflowedPrimesWide;

	internal static void OverrideCapabilitiesForTesting(PrimeOrderGpuCapability capability)
	{
		s_capability = capability;
	}

	internal static void ResetCapabilitiesForTesting()
	{
		s_capability = PrimeOrderGpuCapability.Default;
	}

	private static SmallPrimeDeviceCache GetSmallPrimeDeviceCache(Accelerator accelerator, AcceleratorStream stream)
	{
		var pool = _smallPrimeDeviceCache ??= [];
		if (pool.TryGetValue(accelerator, out var cached))
		{
			return cached;
		}

		uint[] primes = PrimesGenerator.SmallPrimes;
		ulong[] squares = PrimesGenerator.SmallPrimesPow2;
		MemoryBuffer1D<uint, Stride1D.Dense>? primeBuffer;
		MemoryBuffer1D<ulong, Stride1D.Dense>? squareBuffer;
		// lock (accelerator)
		{
			primeBuffer = accelerator.Allocate1D<uint>(primes.Length);
			squareBuffer = accelerator.Allocate1D<ulong>(squares.Length);
		}
		
		primeBuffer.View.CopyFromCPU(stream, primes);
		squareBuffer.View.CopyFromCPU(stream, squares);
		cached = new SmallPrimeDeviceCache
		{
			Primes = primeBuffer,
			Squares = squareBuffer,
			Count = primes.Length,
		};

		pool[accelerator] = cached;
		return cached;
	}

	private static Action<AcceleratorStream, Index1D, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, int, ulong, uint, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<byte, Stride1D.Dense>> GetPartialFactorKernel(Accelerator accelerator)
	{
		var pool = _partialFactorKernel ??= [];
		if (pool.TryGetValue(accelerator, out var cached))
		{
			return cached;
		}

		var loaded = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, int, ulong, uint, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<byte, Stride1D.Dense>>(PartialFactorKernel);

		var kernel = KernelUtil.GetKernel(loaded);

		cached = kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, int, ulong, uint, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<byte, Stride1D.Dense>>>();

		pool[accelerator] = cached;
		return cached;
	}

	public static bool TryPartialFactor(
		ulong value,
		uint limit,
		Span<ulong> primeTargets,
		Span<int> exponentTargets,
		out int factorCount,
		out ulong remaining,
		out bool fullyFactored)
	{
		factorCount = 0;
		remaining = value;
		fullyFactored = false;

		if (primeTargets.Length == 0 || exponentTargets.Length == 0)
		{
			return false;
		}

		primeTargets.Clear();
		exponentTargets.Clear();

		if (!TryLaunchPartialFactorKernel(
				value,
				limit,
				primeTargets,
				exponentTargets,
				out int extracted,
				out ulong leftover,
				out bool kernelFullyFactored))
		{
			return false;
		}

		int capacity = Math.Min(primeTargets.Length, exponentTargets.Length);
		if (extracted > capacity)
		{
			factorCount = 0;
			remaining = value;
			fullyFactored = false;
			return false;
		}

		factorCount = extracted;
		remaining = leftover;
		fullyFactored = kernelFullyFactored && leftover == 1UL;
		return true;
	}

	private static bool TryLaunchPartialFactorKernel(
		ulong value,
		uint limit,
		Span<ulong> primeTargets,
		Span<int> exponentTargets,
		out int factorCount,
		out ulong remaining,
		out bool fullyFactored)
	{
		factorCount = 0;
		remaining = value;
		fullyFactored = false;

		try
		{
			GpuPrimeWorkLimiter.Acquire();
			Accelerator accelerator = AcceleratorPool.Shared.Rent();
			AcceleratorStream stream = accelerator.CreateStream();

			SmallPrimeDeviceCache cache = GetSmallPrimeDeviceCache(accelerator, stream);

			// TODO: We should create / reallocate these buffer only once or if the new required length is bigger than capacity.
			MemoryBuffer1D<ulong, Stride1D.Dense>? factorBuffer;
			MemoryBuffer1D<int, Stride1D.Dense>? exponentBuffer;
			MemoryBuffer1D<int, Stride1D.Dense>? countBuffer;
			MemoryBuffer1D<ulong, Stride1D.Dense>? remainingBuffer;
			MemoryBuffer1D<byte, Stride1D.Dense>? fullyFactoredBuffer;
			// lock(accelerator)
			{
				factorBuffer = accelerator.Allocate1D<ulong>(primeTargets.Length);
				exponentBuffer = accelerator.Allocate1D<int>(exponentTargets.Length);
				countBuffer = accelerator.Allocate1D<int>(1);
				remainingBuffer = accelerator.Allocate1D<ulong>(1);
				fullyFactoredBuffer = accelerator.Allocate1D<byte>(1);				
			}

			// TODO: There is no need to clear these buffers because the kernel will always assign values within the required bounds.
			// factorBuffer.MemSetToZero(stream);
			// exponentBuffer.MemSetToZero(stream);
			// countBuffer.MemSetToZero(stream);
			// remainingBuffer.MemSetToZero(stream);
			// fullyFactoredBuffer.MemSetToZero(stream);

			var partialFactorKernel = GetPartialFactorKernel(accelerator);

			partialFactorKernel(
				stream,
				1,
				cache.Primes!.View,
				cache.Squares!.View,
				cache.Count,
				primeTargets.Length,
				value,
				limit,
				factorBuffer.View,
				exponentBuffer.View,
				countBuffer.View,
				remainingBuffer.View,
				fullyFactoredBuffer.View);


			countBuffer.View.CopyToCPU(stream, ref factorCount, 1);
			factorBuffer.View.CopyToCPU(stream, ref MemoryMarshal.GetReference(primeTargets), primeTargets.Length);
			exponentBuffer.View.CopyToCPU(stream, ref MemoryMarshal.GetReference(exponentTargets), exponentTargets.Length);
			remainingBuffer.View.CopyToCPU(stream, ref remaining, 1);

			byte fullyFactoredFlag = 0;
			fullyFactoredBuffer.View.CopyToCPU(stream, ref fullyFactoredFlag, 1);
			stream.Synchronize();
			stream.Dispose();

			fullyFactored = fullyFactoredFlag != 0;
			factorCount = Math.Min(factorCount, primeTargets.Length);

			// TODO: These buffers shouldn't be disposed but rather reused accross call with the assigned accelerator.
			factorBuffer.Dispose();
			exponentBuffer.Dispose();
			countBuffer.Dispose();
			remainingBuffer.Dispose();
			fullyFactoredBuffer.Dispose();
			GpuPrimeWorkLimiter.Release();
			return true;
		}
		catch (CLException ex)
		{
			Console.WriteLine($"GPU ERROR ({ex.Error}): {ex.Message}");
			factorCount = 0;
			remaining = value;
			fullyFactored = false;
			return false;
		}
		catch (Exception ex) when (ex is AcceleratorException or InternalCompilerException or NotSupportedException or InvalidOperationException or AggregateException)
		{
			Console.WriteLine($"GPU ERROR: {ex.Message}");
			factorCount = 0;
			remaining = value;
			fullyFactored = false;
			return false;
		}
	}

	public static GpuPow2ModStatus TryPow2Mod(ulong exponent, ulong prime, out ulong remainder, in MontgomeryDivisorData divisorData)
	{
		Span<ulong> exponents = stackalloc ulong[1];
		Span<ulong> remainders = stackalloc ulong[1];
		exponents[0] = exponent;

		GpuPow2ModStatus status = TryPow2ModBatch(exponents, prime, remainders, divisorData);
		remainder = remainders[0];
		return status;
	}

	public static GpuPow2ModStatus TryPow2ModBatch(ReadOnlySpan<ulong> exponents, ulong prime, Span<ulong> remainders, in MontgomeryDivisorData divisorData)
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

		bool computed = TryComputeOnGpu(exponents, prime, divisorData, remainders);
		return computed ? GpuPow2ModStatus.Success : GpuPow2ModStatus.Unavailable;
	}

	public static GpuPow2ModStatus TryPow2Mod(in UInt128 exponent, in UInt128 prime, out UInt128 remainder)
	{
		Span<UInt128> exponents = stackalloc UInt128[1];
		Span<UInt128> remainders = stackalloc UInt128[1];
		exponents[0] = exponent;

		GpuPow2ModStatus status = TryPow2ModBatch(exponents, prime, remainders);
		remainder = remainders[0];
		return status;
	}

	public static GpuPow2ModStatus TryPow2ModBatch(ReadOnlySpan<UInt128> exponents, UInt128 prime, Span<UInt128> remainders)
	{
		return TryPow2ModBatchInternal(exponents, prime, remainders);
	}

	internal static bool TryCalculateOrder(
		ulong prime,
		ulong? previousOrder,
		PrimeOrderCalculator.PrimeOrderSearchConfig config,
		in MontgomeryDivisorData divisorData,
		out ulong order)
	{
		order = 0UL;

		GpuPrimeWorkLimiter.Acquire();
		Accelerator accelerator = AcceleratorPool.Shared.Rent();
		AcceleratorStream stream = accelerator.CreateStream();

		SmallPrimeDeviceCache cache = GetSmallPrimeDeviceCache(accelerator, stream);

		// TODO: These buffers should be created reallocated once and assigned to an accelerator. Callers should use their own
		// thread static cache to prevent other threads from using taking accelerators with pre-allocated buffers if they don't
		// use them.
		MemoryBuffer1D<ulong, Stride1D.Dense>? phiFactorBuffer;
		MemoryBuffer1D<int, Stride1D.Dense>? phiExponentBuffer;
		MemoryBuffer1D<ulong, Stride1D.Dense>? workFactorBuffer;
		MemoryBuffer1D<int, Stride1D.Dense>? workExponentBuffer;
		MemoryBuffer1D<ulong, Stride1D.Dense>? candidateBuffer;
		MemoryBuffer1D<int, Stride1D.Dense>? stackIndexBuffer;
		MemoryBuffer1D<int, Stride1D.Dense>? stackExponentBuffer;
		MemoryBuffer1D<ulong, Stride1D.Dense>? stackProductBuffer;
		MemoryBuffer1D<ulong, Stride1D.Dense>? resultBuffer;
		MemoryBuffer1D<byte, Stride1D.Dense>? statusBuffer;

		// lock(accelerator)
		{
			phiFactorBuffer = accelerator.Allocate1D<ulong>(GpuSmallPrimeFactorSlots);
			phiExponentBuffer = accelerator.Allocate1D<int>(GpuSmallPrimeFactorSlots);
			workFactorBuffer = accelerator.Allocate1D<ulong>(GpuSmallPrimeFactorSlots);
			workExponentBuffer = accelerator.Allocate1D<int>(GpuSmallPrimeFactorSlots);
			candidateBuffer = accelerator.Allocate1D<ulong>(HeuristicCandidateLimit);
			stackIndexBuffer = accelerator.Allocate1D<int>(HeuristicStackCapacity);
			stackExponentBuffer = accelerator.Allocate1D<int>(HeuristicStackCapacity);
			stackProductBuffer = accelerator.Allocate1D<ulong>(HeuristicStackCapacity);
			resultBuffer = accelerator.Allocate1D<ulong>(1);
			statusBuffer = accelerator.Allocate1D<byte>(1);			
		}

		// TODO: Remove the cleaning after the order kernel is modified to always set the result.
		phiFactorBuffer.MemSetToZero(stream);
		phiExponentBuffer.MemSetToZero(stream);
		workFactorBuffer.MemSetToZero(stream);
		workExponentBuffer.MemSetToZero(stream);
		candidateBuffer.MemSetToZero(stream);
		stackIndexBuffer.MemSetToZero(stream);
		stackExponentBuffer.MemSetToZero(stream);
		stackProductBuffer.MemSetToZero(stream);
		resultBuffer.MemSetToZero(stream);
		statusBuffer.MemSetToZero(stream);

		uint limit = config.SmallFactorLimit == 0 ? uint.MaxValue : config.SmallFactorLimit;
		byte hasPrevious = previousOrder.HasValue ? (byte)1 : (byte)0;
		ulong previousValue = previousOrder ?? 0UL;

		var kernelConfig = new OrderKernelConfig(previousValue, hasPrevious, limit, config.MaxPowChecks, (int)config.Mode);
		var buffers = new OrderKernelBuffers(
			phiFactorBuffer.View,
			phiExponentBuffer.View,
			workFactorBuffer.View,
			workExponentBuffer.View,
			candidateBuffer.View,
			stackIndexBuffer.View,
			stackExponentBuffer.View,
			stackProductBuffer.View,
			resultBuffer.View,
			statusBuffer.View);

		var orderKernel = GetOrderKernel(accelerator);
		orderKernel(
			stream,
			1,
			prime,
			kernelConfig,
			divisorData,
			cache.Primes!.View,
			cache.Squares!.View,
			cache.Count,
			buffers);


		byte status = 0;
		statusBuffer.View.CopyToCPU(stream, ref status, 1);
		resultBuffer.View.CopyToCPU(stream, ref order, 1);
		stream.Synchronize();

		stream.Dispose();
		phiFactorBuffer.Dispose();
		phiExponentBuffer.Dispose();
		workFactorBuffer.Dispose();
		workExponentBuffer.Dispose();
		candidateBuffer.Dispose();
		stackIndexBuffer.Dispose();
		stackExponentBuffer.Dispose();
		stackProductBuffer.Dispose();
		resultBuffer.Dispose();
		statusBuffer.Dispose();
		GpuPrimeWorkLimiter.Release();

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

	private static OrderKernelLauncher GetOrderKernel(Accelerator accelerator)
	{
		var pool = _orderKernel ??= [];
		if (pool.TryGetValue(accelerator, out var cached))
		{
			return cached;
		}

		var loaded = accelerator.LoadAutoGroupedStreamKernel<Index1D, ulong, OrderKernelConfig, MontgomeryDivisorData, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, OrderKernelBuffers>(CalculateOrderKernel);
		var kernel = KernelUtil.GetKernel(loaded);
		cached = kernel.CreateLauncherDelegate<OrderKernelLauncher>();
		pool[accelerator] = cached;
		return cached;
	}

	private static GpuPow2ModStatus TryPow2ModBatchInternal(ReadOnlySpan<UInt128> exponents, UInt128 prime, Span<UInt128> remainders)
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

		bool computed = TryComputeOnGpuWide(exponents, prime, target);
		return computed ? GpuPow2ModStatus.Success : GpuPow2ModStatus.Unavailable;
	}

	private static bool TryComputeOnGpu(ReadOnlySpan<ulong> exponents, ulong prime, in MontgomeryDivisorData divisorData, Span<ulong> results)
	{
		GpuPrimeWorkLimiter.Acquire();
		Accelerator accelerator = AcceleratorPool.Shared.Rent();
		AcceleratorStream? stream = accelerator.CreateStream();
		// TODO: These buffers should be allocated once per accelerator and only reallocated when the new length exceeds capacity.
		// Modify the callers to use their own pool of buffers per accelerator, so that other threads don't use the accelerators
		// with out preallocated buffers. Share the pool with Pow2ModWide kernel.
		MemoryBuffer1D<ulong, Stride1D.Dense>? exponentBuffer;
		MemoryBuffer1D<ulong, Stride1D.Dense>? remainderBuffer;
		// lock(accelerator)
		{
			exponentBuffer = accelerator.Allocate1D<ulong>(exponents.Length);
			remainderBuffer = accelerator.Allocate1D<ulong>(exponents.Length);			
		}

		exponentBuffer.View.CopyFromCPU(stream, ref MemoryMarshal.GetReference(exponents), exponents.Length);

		// Pow2Mod kernel always assigns the required output elements so we don't need to worry about clearing these.
		// remainderBuffer.MemSetToZero(stream);

		var pow2ModKernel = GetPow2ModKernel(accelerator);
		try
		{
			pow2ModKernel(stream, exponents.Length, exponentBuffer.View, divisorData, remainderBuffer.View);
			remainderBuffer.View.CopyToCPU(stream, ref MemoryMarshal.GetReference(results), exponents.Length);
			stream.Synchronize();

			stream.Dispose();
			stream = null;
			exponentBuffer.Dispose();
			remainderBuffer.Dispose();
			GpuPrimeWorkLimiter.Release();
		}
		catch (Exception)
		{
			Console.WriteLine($"Exception for {prime} and exponents: {string.Join(",", exponents.ToArray())}.");
			stream?.Dispose();
			exponentBuffer.Dispose();
			remainderBuffer.Dispose();
			GpuPrimeWorkLimiter.Release();
			throw;
		}

		return true;
	}

	private static void ComputePow2ModCpu(ReadOnlySpan<ulong> exponents, ulong prime, in MontgomeryDivisorData divisorData, Span<ulong> results)
	{
		int length = exponents.Length;
		for (int i = 0; i < length; i++)
		{
			results[i] = Pow2ModCpu(exponents[i], prime, divisorData);
		}
	}

	private static bool TryComputeOnGpuWide(ReadOnlySpan<UInt128> exponents, UInt128 prime, Span<UInt128> results)
	{
		int length = exponents.Length;
		if (length == 0)
		{
			return true;
		}

		GpuUInt128[]? rentedExponents = null;
		GpuUInt128[]? rentedResults = null;
		GpuPrimeWorkLimiter.Acquire();
		ArrayPool<GpuUInt128> gpuUInt128Pool = ThreadStaticPools.GpuUInt128Pool;

		try
		{
			Accelerator accelerator = AcceleratorPool.Shared.Rent();
			AcceleratorStream stream = accelerator.CreateStream();
			// TODO: These buffers should be allocated once per accelerator and only reallocated when the new length exceeds capacity.
			// Modify the callers to use their own pool of buffers per accelerator, so that other threads don't use the accelerators
			// with out preallocated buffers. Share the pool with Pow2Mod kernel.
			MemoryBuffer1D<GpuUInt128, Stride1D.Dense>? exponentBuffer;
			MemoryBuffer1D<GpuUInt128, Stride1D.Dense>? remainderBuffer;
			// lock(accelerator)
			{
				exponentBuffer = accelerator.Allocate1D<GpuUInt128>(length);
				remainderBuffer = accelerator.Allocate1D<GpuUInt128>(length);				
			}

			Span<GpuUInt128> exponentSpan = length <= WideStackThreshold
				? stackalloc GpuUInt128[length]
				: new Span<GpuUInt128>(rentedExponents = gpuUInt128Pool.Rent(length), 0, length);

			for (int i = 0; i < length; i++)
			{
				exponentSpan[i] = (GpuUInt128)exponents[i];
			}

			exponentBuffer.View.CopyFromCPU(stream, ref MemoryMarshal.GetReference(exponentSpan), length);
			// Pow2ModWide kernel always assigns the required output elements so we don't need to worry about clearing these.
			// remainderBuffer.MemSetToZero(stream);

			GpuUInt128 modulus = (GpuUInt128)prime;
			var pow2ModWideKernel = GetPow2ModWideKernel(accelerator);
			pow2ModWideKernel(stream, length, exponentBuffer.View, modulus, remainderBuffer.View);


			Span<GpuUInt128> resultSpan = length <= WideStackThreshold
				? stackalloc GpuUInt128[length]
				: new Span<GpuUInt128>(rentedResults = gpuUInt128Pool.Rent(length), 0, length);

			remainderBuffer.View.CopyToCPU(stream, ref MemoryMarshal.GetReference(resultSpan), length);
			stream.Synchronize();
			stream.Dispose();

			for (int i = 0; i < length; i++)
			{
				results[i] = (UInt128)resultSpan[i];
			}

			exponentBuffer.Dispose();
			remainderBuffer.Dispose();
			return true;
		}
		catch (Exception)
		{
			Console.WriteLine($"Exception when computing {prime} with exponents: {string.Join(", ", exponents.ToArray())}");
			throw;
		}
		finally
		{
			if (rentedExponents is not null)
			{
				gpuUInt128Pool.Return(rentedExponents, clearArray: false);
				gpuUInt128Pool.Return(rentedResults!, clearArray: false);
			}

			GpuPrimeWorkLimiter.Release();
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

	private static ulong Pow2ModCpu(ulong exponent, ulong modulus, in MontgomeryDivisorData divisorData)
	{
		if (modulus <= 1UL)
		{
			return 0UL;
		}

		return ULongExtensions.Pow2MontgomeryModWindowedGpuConvertToStandard(divisorData, exponent);
	}

	private static Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>> GetPow2ModKernel(Accelerator accelerator)
	{
		var pool = _pow2ModKernel ??= [];
		if (pool.TryGetValue(accelerator, out var cached))
		{
			return cached;
		}

		var loaded = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>(Pow2ModKernel);

		var kernel = KernelUtil.GetKernel(loaded);

		cached = kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>>();
		pool[accelerator] = cached;
		return cached;
	}

	internal readonly record struct PrimeOrderGpuCapability(int ModulusBits, int ExponentBits)
	{
		public static PrimeOrderGpuCapability Default => new(128, 128);
	}
}
