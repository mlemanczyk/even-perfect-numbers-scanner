using System.Buffers;
using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core;

public sealed class PrimeTester
{
	public PrimeTester()
	{
	}

	[ThreadStatic]
	private static PrimeTester? _tester;

	public static PrimeTester Exclusive => _tester ??= new();

	public static bool IsPrime(ulong n, CancellationToken ct)
	{
		if (n <= 1UL)
		{
			return false;
		}

		if (n == 2UL)
		{
			throw new InvalidOperationException("PrimeTester.IsPrime encountered the sentinel input 2.");
		}

		bool isOdd = (n & 1UL) != 0UL;
		bool result = isOdd;

		bool requiresTrialDivision = result && n >= 7UL;

		if (requiresTrialDivision)
		{
			// EvenPerfectBitScanner streams exponents starting at 136,279,841, so the Mod10/GCD guard never fires on the
			// production path. Leave the logic commented out as instrumentation for diagnostic builds.
			// bool sharesMaxExponentFactor = n.Mod10() == 1UL && SharesFactorWithMaxExponent(n);
			// result &= !sharesMaxExponentFactor;

			if (result)
			{
				uint[] smallPrimeDivisors = PrimesGenerator.SmallPrimes;
				ulong[] smallPrimeDivisorsMul = PrimesGenerator.SmallPrimesPow2;

				ulong nMod10 = n.Mod10();
				switch (nMod10)
				{
					case 1UL:
						smallPrimeDivisors = PrimesGenerator.SmallPrimesLastOne;
						smallPrimeDivisorsMul = PrimesGenerator.SmallPrimesPow2LastOne;
						break;
					case 3UL:
						smallPrimeDivisors = DivisorGenerator.SmallPrimesLastThree;
						smallPrimeDivisorsMul = DivisorGenerator.SmallPrimesPow2LastThree;
						break;
					case 7UL:
						smallPrimeDivisors = PrimesGenerator.SmallPrimesLastSeven;
						smallPrimeDivisorsMul = PrimesGenerator.SmallPrimesPow2LastSeven;
						break;
					case 9UL:
						smallPrimeDivisors = DivisorGenerator.SmallPrimesLastNine;
						smallPrimeDivisorsMul = DivisorGenerator.SmallPrimesPow2LastNine;
						break;
				}

				int smallPrimeDivisorsLength = smallPrimeDivisors.Length;
				for (int i = 0; i < smallPrimeDivisorsLength; i++)
				{
					if (smallPrimeDivisorsMul[i] > n)
					{
						break;
					}

					if (n % smallPrimeDivisors[i] == 0)
					{
						result = false;
						break;
					}
				}
			}
		}

		return result;
	}

	public static bool IsPrimeGpu(ulong n)
	{
		return Exclusive.IsPrimeGpu(n, CancellationToken.None);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public bool IsPrimeGpu(ulong n, CancellationToken ct)
	{
		bool forceCpu = GpuContextPool.ForceCpu;
		bool belowGpuRange = n < 31UL;

		if (forceCpu || belowGpuRange)
		{
			return IsPrimeCpu(n, ct);
		}

		Span<ulong> one = stackalloc ulong[1];
		Span<byte> outFlags = stackalloc byte[1];
		one[0] = n;
		outFlags[0] = 0;

		IsPrimeBatchGpu(one, outFlags);

		return outFlags[0] != 0;
	}

	public static bool IsPrimeCpu(ulong n, CancellationToken ct)
	{
		// Preserve the legacy entry point for callers that bypass the thread-local caches.
		return IsPrime(n, ct);
	}


	public static int GpuBatchSize { get; set; } = 262_144;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static void IsPrimeBatchGpu(ReadOnlySpan<ulong> values, Span<byte> results)
	{
		var limiter = GpuPrimeWorkLimiter.Acquire();
		var gpu = PrimeTesterGpuContextPool.Rent(GpuBatchSize);
		var state = gpu.State;
		var accelerator = gpu.Accelerator;
		int totalLength = values.Length;
		int batchSize = GpuBatchSize;

		var input = gpu.Input;
		var output = gpu.Output;
		ArrayPool<ulong> pool = ThreadStaticPools.UlongPool;
		ulong[] temp = pool.Rent(batchSize);

		int pos = 0;
		while (pos < totalLength)
		{
			int remaining = totalLength - pos;
			int count = remaining > batchSize ? batchSize : remaining;

			values.Slice(pos, count).CopyTo(temp);
			input.View.CopyFromCPU(ref temp[0], count);

			state.Kernel(
				count,
				input.View,
				state.DevicePrimesDefault.View,
				state.DevicePrimesLastOne.View,
				state.DevicePrimesLastSeven.View,
				state.DevicePrimesLastThree.View,
				state.DevicePrimesLastNine.View,
				state.DevicePrimesPow2Default.View,
				state.DevicePrimesPow2LastOne.View,
				state.DevicePrimesPow2LastSeven.View,
				state.DevicePrimesPow2LastThree.View,
				state.DevicePrimesPow2LastNine.View,
				output.View);
			accelerator.Synchronize();
			output.View.CopyToCPU(ref results[pos], count);

			pos += count;
		}

		pool.Return(temp, clearArray: false);

		gpu.Dispose();
		limiter.Dispose();
	}

	internal static class PrimeTesterGpuContextPool
	{
		internal sealed class PooledContext
		{
			public Context Context { get; }

			public Accelerator Accelerator { get; }

			private KernelState? _kernelState;

			public PooledContext()
			{
				Context = Context.CreateDefault();
				Accelerator = Context.GetPreferredDevice(false).CreateAccelerator(Context);
				_kernelState = GpuKernelState.GetOrCreate(Accelerator);
			}

			public KernelState KernelState
			{
				get
				{
					var state = _kernelState;
					if (state is null || state.IsDisposed)
					{
						state = GpuKernelState.GetOrCreate(Accelerator);
						_kernelState = state;
					}

					return state;
				}
			}

			public void Dispose()
			{
				ClearGpuCaches(Accelerator);
				_kernelState = null;
				Accelerator.Dispose();
				Context.Dispose();
			}
		}

		private static readonly ConcurrentQueue<PooledContext> Pool = new();

		private static readonly ConcurrentQueue<(Accelerator Accelerator, MemoryBuffer1D<ulong, Stride1D.Dense> Input, MemoryBuffer1D<byte, Stride1D.Dense> Output, int Capacity)> BufferPool = new();

		internal static PrimeTesterGpuContextLease Rent(int minBufferCapacity = 0)
		{
			if (Pool.TryDequeue(out var ctx))
			{
				return new PrimeTesterGpuContextLease(ctx, minBufferCapacity);
			}

			return new PrimeTesterGpuContextLease(new PooledContext(), minBufferCapacity);
		}

		internal static void DisposeAll()
		{
			while (Pool.TryDequeue(out var ctx))
			{
				ctx.Dispose();
			}

			while (BufferPool.TryDequeue(out var entry))
			{
				entry.Output.Dispose();
				entry.Input.Dispose();
			}
		}

		private static void AcquireBuffers(Accelerator accelerator, int minCapacity, out MemoryBuffer1D<ulong, Stride1D.Dense> input, out MemoryBuffer1D<byte, Stride1D.Dense> output, out int capacity)
		{
			int required = Math.Max(1, minCapacity);

			while (BufferPool.TryDequeue(out var entry))
			{
				if (entry.Accelerator != accelerator)
				{
					entry.Output.Dispose();
					entry.Input.Dispose();
					continue;
				}

				if (entry.Capacity >= required)
				{
					input = entry.Input;
					output = entry.Output;
					capacity = entry.Capacity;
					return;
				}

				entry.Output.Dispose();
				entry.Input.Dispose();
			}

			capacity = required;
			input = accelerator.Allocate1D<ulong>(capacity);
			output = accelerator.Allocate1D<byte>(capacity);
		}

		private static void ReturnBuffers(Accelerator accelerator, MemoryBuffer1D<ulong, Stride1D.Dense> input, MemoryBuffer1D<byte, Stride1D.Dense> output, int capacity)
		{
			BufferPool.Enqueue((accelerator, input, output, capacity));
		}

		private static void Return(PooledContext ctx)
		{
			ctx.Accelerator.Synchronize();
			Pool.Enqueue(ctx);
		}

		internal readonly struct PrimeTesterGpuContextLease
		{
			private readonly PooledContext _ctx;

			public readonly Accelerator Accelerator;

			public readonly KernelState State;

			public readonly MemoryBuffer1D<ulong, Stride1D.Dense> Input;

			public readonly MemoryBuffer1D<byte, Stride1D.Dense> Output;

			public readonly int BufferCapacity;

			internal PrimeTesterGpuContextLease(PooledContext ctx, int minBufferCapacity)
			{
				_ctx = ctx;
				Accelerator = ctx.Accelerator;
				State = ctx.KernelState;
				AcquireBuffers(Accelerator, minBufferCapacity, out Input, out Output, out BufferCapacity);
			}

			public void Dispose()
			{
				ReturnBuffers(Accelerator, Input, Output, BufferCapacity);
				Return(_ctx);
			}
		}
	}
	// Per-accelerator GPU state for prime sieve (kernel + uploaded primes).
	internal sealed class KernelState
	{
		public Action<Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>> Kernel;
		public Action<Index1D, ArrayView<ulong>, ulong, ArrayView<byte>> HeuristicTrialDivisionKernel;
		public MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimesDefault;
		public MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimesLastOne;
		public MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimesLastSeven;
		public MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimesLastThree;
		public MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimesLastNine;
		public MemoryBuffer1D<ulong, Stride1D.Dense> DevicePrimesPow2Default;
		public MemoryBuffer1D<ulong, Stride1D.Dense> DevicePrimesPow2LastOne;
		public MemoryBuffer1D<ulong, Stride1D.Dense> DevicePrimesPow2LastSeven;
		public MemoryBuffer1D<ulong, Stride1D.Dense> DevicePrimesPow2LastThree;
		public MemoryBuffer1D<ulong, Stride1D.Dense> DevicePrimesPow2LastNine;
		public bool IsDisposed;

		public KernelState(Accelerator accelerator)
		{
			IsDisposed = false;
			// Compile once per accelerator and upload primes once.
			Kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>>(PrimeTesterKernels.SmallPrimeSieveKernel);
			HeuristicTrialDivisionKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ulong, ArrayView<byte>>(PrimeTesterKernels.HeuristicTrialDivisionKernel);

			var primesDefault = DivisorGenerator.SmallPrimes;
			DevicePrimesDefault = accelerator.Allocate1D<uint>(primesDefault.Length);
			DevicePrimesDefault.View.CopyFromCPU(primesDefault);

			var primesLastOne = DivisorGenerator.SmallPrimesLastOne;
			DevicePrimesLastOne = accelerator.Allocate1D<uint>(primesLastOne.Length);
			DevicePrimesLastOne.View.CopyFromCPU(primesLastOne);

			var primesLastSeven = DivisorGenerator.SmallPrimesLastSeven;
			DevicePrimesLastSeven = accelerator.Allocate1D<uint>(primesLastSeven.Length);
			DevicePrimesLastSeven.View.CopyFromCPU(primesLastSeven);

			var primesLastThree = DivisorGenerator.SmallPrimesLastThree;
			DevicePrimesLastThree = accelerator.Allocate1D<uint>(primesLastThree.Length);
			DevicePrimesLastThree.View.CopyFromCPU(primesLastThree);

			var primesLastNine = DivisorGenerator.SmallPrimesLastNine;
			DevicePrimesLastNine = accelerator.Allocate1D<uint>(primesLastNine.Length);
			DevicePrimesLastNine.View.CopyFromCPU(primesLastNine);

			var primesPow2Default = DivisorGenerator.SmallPrimesPow2;
			DevicePrimesPow2Default = accelerator.Allocate1D<ulong>(primesPow2Default.Length);
			DevicePrimesPow2Default.View.CopyFromCPU(primesPow2Default);

			var primesPow2LastOne = DivisorGenerator.SmallPrimesPow2LastOne;
			DevicePrimesPow2LastOne = accelerator.Allocate1D<ulong>(primesPow2LastOne.Length);
			DevicePrimesPow2LastOne.View.CopyFromCPU(primesPow2LastOne);

			var primesPow2LastSeven = DivisorGenerator.SmallPrimesPow2LastSeven;
			DevicePrimesPow2LastSeven = accelerator.Allocate1D<ulong>(primesPow2LastSeven.Length);
			DevicePrimesPow2LastSeven.View.CopyFromCPU(primesPow2LastSeven);

			var primesPow2LastThree = DivisorGenerator.SmallPrimesPow2LastThree;
			DevicePrimesPow2LastThree = accelerator.Allocate1D<ulong>(primesPow2LastThree.Length);
			DevicePrimesPow2LastThree.View.CopyFromCPU(primesPow2LastThree);

			var primesPow2LastNine = DivisorGenerator.SmallPrimesPow2LastNine;
			DevicePrimesPow2LastNine = accelerator.Allocate1D<ulong>(primesPow2LastNine.Length);
			DevicePrimesPow2LastNine.View.CopyFromCPU(primesPow2LastNine);
		}

		public void Clear()
		{
			if (IsDisposed)
			{
				return;
			}

			DevicePrimesDefault.Dispose();
			DevicePrimesLastOne.Dispose();
			DevicePrimesLastSeven.Dispose();
			DevicePrimesLastThree.Dispose();
			DevicePrimesLastNine.Dispose();
			DevicePrimesPow2Default.Dispose();
			DevicePrimesPow2LastOne.Dispose();
			DevicePrimesPow2LastSeven.Dispose();
			DevicePrimesPow2LastThree.Dispose();
			DevicePrimesPow2LastNine.Dispose();
			IsDisposed = true;
		}
	}

	internal static class GpuKernelState
	{
		// Map accelerator to cached state; use Lazy to serialize kernel creation
		private static readonly System.Collections.Concurrent.ConcurrentDictionary<Accelerator, Lazy<KernelState>> States = new();
		// TODO: Prewarm this per-accelerator cache during startup (and reuse a simple array keyed by accelerator index)
		// once the kernel pool exposes deterministic ordering; the Lazy wrappers showed measurable overhead in the
		// GpuModularArithmeticBenchmarks hot path.

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static KernelState GetOrCreate(Accelerator accelerator)
		{
			var lazy = States.GetOrAdd(
				accelerator,
				acc => new Lazy<KernelState>(() => new KernelState(acc), LazyThreadSafetyMode.ExecutionAndPublication));
			return lazy.Value;
		}

		public static void Clear(Accelerator accelerator)
		{
			if (States.TryRemove(accelerator, out var lazy))
			{
				if (lazy.IsValueCreated)
				{
					var state = lazy.Value;
					state.Clear();
				}
			}
		}
	}

	// Expose cache clearing for accelerator disposal coordination
	public static void ClearGpuCaches(Accelerator accelerator)
	{
		GpuKernelState.Clear(accelerator);
	}

	internal static void DisposeGpuContexts()
	{
		PrimeTesterGpuContextPool.DisposeAll();
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	internal static bool SharesFactorWithMaxExponent(ulong n)
	{
		// TODO: Replace this on-the-fly GCD probe with the cached factor table derived from
		// ResidueComputationBenchmarks so divisor-cycle metadata can short-circuit the test
		// instead of recomputing binary GCD for every candidate.
		ulong m = (ulong)BitOperations.Log2(n);
		return BinaryGcd(n, m) != 1UL;
	}

	internal static void SharesFactorWithMaxExponentBatch(ReadOnlySpan<ulong> values, Span<byte> results)
	{
		// TODO: Route this batch helper through the shared GPU kernel pool from
		// GpuUInt128BinaryGcdBenchmarks so we reuse cached kernels, pinned host buffers,
		// and divisor-cycle staging instead of allocating new device buffers per call.
		var gpu = PrimeTesterGpuContextPool.Rent();
		var accelerator = gpu.Accelerator;
		var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ArrayView<byte>>(PrimeTesterKernels.SharesFactorKernel);

		int length = values.Length;
		var inputBuffer = accelerator.Allocate1D<ulong>(length);
		var resultBuffer = accelerator.Allocate1D<byte>(length);

		ArrayPool<ulong> pool = ThreadStaticPools.UlongPool;
		ulong[] temp = pool.Rent(length);
		values.CopyTo(temp);
		inputBuffer.View.CopyFromCPU(ref temp[0], length);
		kernel(length, inputBuffer.View, resultBuffer.View);
		accelerator.Synchronize();
		resultBuffer.View.CopyToCPU(ref results[0], length);
		pool.Return(temp, clearArray: false);
		resultBuffer.Dispose();
		inputBuffer.Dispose();
		gpu.Dispose();
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static ulong BinaryGcd(ulong u, ulong v)
	{
		// TODO: Swap this handwritten binary GCD for the optimized helper measured in
		// GpuUInt128BinaryGcdBenchmarks so CPU callers share the faster subtract-less
		// ladder once the common implementation is promoted into PerfectNumbers.Core.
		if (u == 0UL)
		{
			return v;
		}

		if (v == 0UL)
		{
			return u;
		}

		int shift = BitOperations.TrailingZeroCount(u | v);
		u >>= BitOperations.TrailingZeroCount(u);

		do
		{
			v >>= BitOperations.TrailingZeroCount(v);
			if (u > v)
			{
				(u, v) = (v, u);
			}

			v -= u;
		}
		while (v != 0UL);

		return u << shift;
	}
}
