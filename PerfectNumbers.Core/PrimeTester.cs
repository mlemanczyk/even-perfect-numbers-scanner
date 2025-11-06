using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
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

	public static bool IsPrime(ulong n)
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

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static bool IsPrimeGpu(ulong n)
	{
		GpuPrimeWorkLimiter.Acquire();
		var gpu = PrimeTesterGpuContextPool.Rent(1);
		var accelerator = gpu.Accelerator;
		var stream = gpu.Stream;
		var kernel = PrimeTesterGpuContextPool.PrimeTesterGpuContextLease.Kernel;

		ulong value = n;
		byte flag = 0;

		var input = gpu.Input;
		var output = gpu.Output;
		var inputView = input.View;
		var outputView = output.View;

		inputView.CopyFromCPU(stream, ref value, 1);
		kernel(
						stream,
						1,
						inputView,
						gpu.DevicePrimesDefault.View,
						gpu.DevicePrimesLastOne.View,
						gpu.DevicePrimesLastSeven.View,
						gpu.DevicePrimesLastThree.View,
						gpu.DevicePrimesLastNine.View,
						gpu.DevicePrimesPow2Default.View,
						gpu.DevicePrimesPow2LastOne.View,
						gpu.DevicePrimesPow2LastSeven.View,
						gpu.DevicePrimesPow2LastThree.View,
						gpu.DevicePrimesPow2LastNine.View,
						outputView);
		outputView.CopyToCPU(stream, ref flag, 1);
		stream.Synchronize();

		gpu.Dispose();
		GpuPrimeWorkLimiter.Release();

		return flag != 0;
	}

	public static int GpuBatchSize { get; set; } = 262_144;

	private static readonly object GpuWarmUpLock = new();
	private static int WarmedGpuLeaseCount;

	public static void WarmUpGpuKernels(int threadCount)
	{
		int target = threadCount >> 2;
		if (target == 0)
		{
			target = threadCount;
		}

		lock (GpuWarmUpLock)
		{
			if (target <= WarmedGpuLeaseCount)
			{
				return;
			}

			PrimeTesterGpuContextPool.WarmUp(target, GpuBatchSize);
			WarmedGpuLeaseCount = target;
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static void IsPrimeBatchGpu(ReadOnlySpan<ulong> values, Span<byte> results)
	{
		GpuPrimeWorkLimiter.Acquire();
		var gpu = PrimeTesterGpuContextPool.Rent(GpuBatchSize);
		var accelerator = gpu.Accelerator;
		var stream = gpu.Stream;
		var kernel = PrimeTesterGpuContextPool.PrimeTesterGpuContextLease.Kernel;
		int totalLength = values.Length;
		int batchSize = GpuBatchSize;

		var input = gpu.Input;
		var output = gpu.Output;
		var inputView = input.View;
		var outputView = output.View;

		int pos = 0;
		while (pos < totalLength)
		{
			int remaining = totalLength - pos;
			int count = remaining > batchSize ? batchSize : remaining;

			var valueSlice = values.Slice(pos, count);
			ref ulong valueRef = ref MemoryMarshal.GetReference(valueSlice);
			inputView.CopyFromCPU(stream, ref valueRef, count);

			kernel(
					stream,
					count,
					inputView,
					gpu.DevicePrimesDefault.View,
					gpu.DevicePrimesLastOne.View,
					gpu.DevicePrimesLastSeven.View,
					gpu.DevicePrimesLastThree.View,
					gpu.DevicePrimesLastNine.View,
					gpu.DevicePrimesPow2Default.View,
					gpu.DevicePrimesPow2LastOne.View,
					gpu.DevicePrimesPow2LastSeven.View,
					gpu.DevicePrimesPow2LastThree.View,
					gpu.DevicePrimesPow2LastNine.View,
					outputView);

			var resultSlice = results.Slice(pos, count);
			ref byte resultRef = ref MemoryMarshal.GetReference(resultSlice);
			outputView.CopyToCPU(stream, ref resultRef, count);

			pos += count;
		}

		stream.Synchronize();
		gpu.Dispose();
		GpuPrimeWorkLimiter.Release();
	}

	internal static class PrimeTesterGpuContextPool
	{
		private static readonly ConcurrentQueue<PrimeTesterGpuContextLease> Pool = new();
		private static readonly object WarmUpLock = new();
		private static int WarmedLeaseCount;
		private static readonly object SharedTablesLock = new();
		private static readonly Dictionary<Accelerator, SharedHeuristicGpuTables> SharedTables = new(AcceleratorReferenceComparer.Instance);

		private static SharedHeuristicGpuTables RentSharedTables(Accelerator accelerator, AcceleratorStream stream)
		{
			lock (SharedTablesLock)
			{
				if (!SharedTables.TryGetValue(accelerator, out var tables))
				{
					tables = new SharedHeuristicGpuTables(accelerator, stream);
					SharedTables.Add(accelerator, tables);
				}

				tables.AddReference();
				return tables;
			}
		}

		private static void ReleaseSharedTables(SharedHeuristicGpuTables tables)
		{
			if (tables is null)
			{
				return;
			}

			lock (SharedTablesLock)
			{
				tables.Release();
			}
		}

		internal static void EnsureStaticTables(KernelContainer kernels, Accelerator accelerator, AcceleratorStream stream)
		{
			var tables = RentSharedTables(accelerator, stream);
			var heuristicTables = tables.CreateHeuristicDivisorTables();
			HeuristicGpuDivisorTables.InitializeShared(in heuristicTables);
			ReleaseSharedTables(tables);
		}

		internal static void WarmUp(int target, int minBufferCapacity)
		{
			if (target <= 0)
			{
				return;
			}

			lock (WarmUpLock)
			{
				if (target <= WarmedLeaseCount)
				{
					return;
				}

				Accelerator accelerator = SharedGpuContext.Accelerator;
				AcceleratorStream stream = accelerator.CreateStream();
				KernelContainer kernels = GpuKernelPool.GetKernels(accelerator, stream);

				GpuStaticTableInitializer.EnsureStaticTables(kernels, accelerator, stream);

				int toCreate = target - WarmedLeaseCount;
				for (int i = 0; i < toCreate; i++)
				{
					Pool.Enqueue(new PrimeTesterGpuContextLease(minBufferCapacity, stream));
				}

				stream.Synchronize();
				stream.Dispose();
				WarmedLeaseCount = target;
			}
		}

		internal static PrimeTesterGpuContextLease Rent(int minBufferCapacity = 1)
		{
			if (!Pool.TryDequeue(out var lease))
			{
				lease = new PrimeTesterGpuContextLease(minBufferCapacity);
			}
			else
			{
				lease.EnsureCapacity(minBufferCapacity);
			}

			return lease;
		}


		internal static void Return(PrimeTesterGpuContextLease lease)
		{
			Pool.Enqueue(lease);
		}

		internal static void DisposeAll()
		{
			while (Pool.TryDequeue(out var lease))
			{
				lease.DisposeResources();
			}

			lock (SharedTablesLock)
			{
				foreach (var entry in SharedTables.Values)
				{
					entry.Dispose();
				}

				SharedTables.Clear();
				var emptyTables = default(HeuristicGpuDivisorTables);
				HeuristicGpuDivisorTables.InitializeShared(in emptyTables);
			}

			WarmedLeaseCount = 0;
		}

		internal static void Clear(Accelerator accelerator)
		{
			if (accelerator is null)
			{
				return;
			}

			lock (WarmUpLock)
			{
				var retained = new List<PrimeTesterGpuContextLease>();
				while (Pool.TryDequeue(out var lease))
				{
					if (lease.Accelerator == accelerator)
					{
						lease.DisposeResources();
					}
					else
					{
						retained.Add(lease);
					}
				}

				int retainedCount = retained.Count;
				for (int i = 0; i < retainedCount; i++)
				{
					Pool.Enqueue(retained[i]);
				}

				WarmedLeaseCount = Pool.Count;
			}

			lock (SharedTablesLock)
			{
				if (SharedTables.Remove(accelerator, out var tables))
				{
					tables.Dispose();
				}

				if (SharedTables.Count == 0)
				{
					var emptyTables = default(HeuristicGpuDivisorTables);
					HeuristicGpuDivisorTables.InitializeShared(in emptyTables);
				}
			}
		}

		internal sealed class PrimeTesterGpuContextLease
		{
			private readonly SharedHeuristicGpuTables _sharedTables;
			private readonly HeuristicGpuDivisorTables _heuristicDivisorTables;
			internal readonly Context AcceleratorContext;
			public readonly Accelerator Accelerator;
			private AcceleratorStream? _stream;
			public AcceleratorStream Stream
			{
				[MethodImpl(MethodImplOptions.AggressiveInlining)]
				get => _stream ?? Accelerator.CreateStream();
			}

			public static readonly Action<AcceleratorStream, Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>> Kernel = KernelUtil.GetKernel(SharedGpuContext.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>>(PrimeTesterKernels.SmallPrimeSieveKernel)).CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>>>();

			public static readonly Action<AcceleratorStream, Index1D, ArrayView<int>, ulong, ulong, HeuristicGpuDivisorTableKind, HeuristicGpuDivisorTables> HeuristicTrialDivisionKernel = KernelUtil.GetKernel(SharedGpuContext.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ulong, ulong, HeuristicGpuDivisorTableKind, HeuristicGpuDivisorTables>(PrimeTesterKernels.HeuristicTrialDivisionKernel)).CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView<int>, ulong, ulong, HeuristicGpuDivisorTableKind, HeuristicGpuDivisorTables>>();

			public MemoryBuffer1D<ulong, Stride1D.Dense> Input = null!;
			public MemoryBuffer1D<byte, Stride1D.Dense> Output = null!;
			public MemoryBuffer1D<int, Stride1D.Dense> HeuristicFlag = null!;
			public int BufferCapacity;

			public MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimesDefault => _sharedTables.DevicePrimesDefault;
			public MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimesLastOne => _sharedTables.DevicePrimesLastOne;
			public MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimesLastSeven => _sharedTables.DevicePrimesLastSeven;
			public MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimesLastThree => _sharedTables.DevicePrimesLastThree;
			public MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimesLastNine => _sharedTables.DevicePrimesLastNine;
			public MemoryBuffer1D<ulong, Stride1D.Dense> DevicePrimesPow2Default => _sharedTables.DevicePrimesPow2Default;
			public MemoryBuffer1D<ulong, Stride1D.Dense> DevicePrimesPow2LastOne => _sharedTables.DevicePrimesPow2LastOne;
			public MemoryBuffer1D<ulong, Stride1D.Dense> DevicePrimesPow2LastSeven => _sharedTables.DevicePrimesPow2LastSeven;
			public MemoryBuffer1D<ulong, Stride1D.Dense> DevicePrimesPow2LastThree => _sharedTables.DevicePrimesPow2LastThree;
			public MemoryBuffer1D<ulong, Stride1D.Dense> DevicePrimesPow2LastNine => _sharedTables.DevicePrimesPow2LastNine;
			public MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupADivisors => _sharedTables.HeuristicGroupADivisors;
			public MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupADivisorSquares => _sharedTables.HeuristicGroupADivisorSquares;
			public MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupBDivisorsEnding1 => _sharedTables.HeuristicGroupBDivisorsEnding1;
			public MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupBDivisorSquaresEnding1 => _sharedTables.HeuristicGroupBDivisorSquaresEnding1;
			public MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupBDivisorsEnding7 => _sharedTables.HeuristicGroupBDivisorsEnding7;
			public MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupBDivisorSquaresEnding7 => _sharedTables.HeuristicGroupBDivisorSquaresEnding7;
			public MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupBDivisorsEnding9 => _sharedTables.HeuristicGroupBDivisorsEnding9;
			public MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupBDivisorSquaresEnding9 => _sharedTables.HeuristicGroupBDivisorSquaresEnding9;
			public MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorsEnding1 => _sharedTables.HeuristicCombinedDivisorsEnding1;
			public MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorSquaresEnding1 => _sharedTables.HeuristicCombinedDivisorSquaresEnding1;
			public MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorsEnding3 => _sharedTables.HeuristicCombinedDivisorsEnding3;
			public MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorSquaresEnding3 => _sharedTables.HeuristicCombinedDivisorSquaresEnding3;
			public MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorsEnding7 => _sharedTables.HeuristicCombinedDivisorsEnding7;
			public MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorSquaresEnding7 => _sharedTables.HeuristicCombinedDivisorSquaresEnding7;
			public MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorsEnding9 => _sharedTables.HeuristicCombinedDivisorsEnding9;
			public MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorSquaresEnding9 => _sharedTables.HeuristicCombinedDivisorSquaresEnding9;

			public HeuristicGpuDivisorTables HeuristicGpuTables => _heuristicDivisorTables;

			internal PrimeTesterGpuContextLease(int minBufferCapacity)
			{
				AcceleratorContext = SharedGpuContext.Context;
				Accelerator = SharedGpuContext.Accelerator;
				AcceleratorStream stream = SharedGpuContext.Accelerator.CreateStream();
				var kernels = GpuKernelPool.GetKernels(Accelerator, stream);
				GpuStaticTableInitializer.EnsureStaticTables(kernels, Accelerator, stream);

				_sharedTables = RentSharedTables(Accelerator, stream);

				_heuristicDivisorTables = _sharedTables.CreateHeuristicDivisorTables();
				HeuristicGpuDivisorTables.InitializeShared(in _heuristicDivisorTables);

				EnsureCapacity(minBufferCapacity);
				stream.Synchronize();
				stream.Dispose();
			}

			internal PrimeTesterGpuContextLease(int minBufferCapacity, AcceleratorStream stream)
			{
				AcceleratorContext = SharedGpuContext.Context;
				Accelerator = SharedGpuContext.Accelerator;
				// GpuStaticTableInitializer.EnsureStaticTables(Accelerator);
				// Kernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>>(PrimeTesterKernels.SmallPrimeSieveKernel);
				// HeuristicTrialDivisionKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ulong, ulong, HeuristicGpuDivisorTableKind, HeuristicGpuDivisorTables>(PrimeTesterKernels.HeuristicTrialDivisionKernel);

				_sharedTables = RentSharedTables(Accelerator, stream);

				_heuristicDivisorTables = _sharedTables.CreateHeuristicDivisorTables();
				HeuristicGpuDivisorTables.InitializeShared(in _heuristicDivisorTables);

				EnsureCapacity(minBufferCapacity);
			}

			public void EnsureCapacity(int minCapacity)
			{
				if (minCapacity <= BufferCapacity)
				{
					return;
				}

				Input?.Dispose();
				Output?.Dispose();
				HeuristicFlag?.Dispose();

				Input = Accelerator.Allocate1D<ulong>(minCapacity);
				Output = Accelerator.Allocate1D<byte>(minCapacity);
				HeuristicFlag = Accelerator.Allocate1D<int>(minCapacity);
				BufferCapacity = minCapacity;
			}

			public void Dispose()
			{
				Return(this);
			}

			internal void DisposeResources()
			{
				Input?.Dispose();
				Output?.Dispose();
				HeuristicFlag?.Dispose();
				_stream?.Dispose();
				_stream = null;
				ReleaseSharedTables(_sharedTables);
				// These resources are shared between GPU leases
				// Accelerator.Dispose();
				// AcceleratorContext.Dispose();
			}
		}

		private sealed class AcceleratorReferenceComparer : IEqualityComparer<Accelerator>
		{
			internal static AcceleratorReferenceComparer Instance { get; } = new();

			public bool Equals(Accelerator? x, Accelerator? y)
			{
				return ReferenceEquals(x, y);
			}

			public int GetHashCode(Accelerator obj)
			{
				return RuntimeHelpers.GetHashCode(obj);
			}
		}

		private sealed class SharedHeuristicGpuTables
		{
			private int _referenceCount;

			internal SharedHeuristicGpuTables(Accelerator accelerator, AcceleratorStream stream)
			{

				var primesDefault = DivisorGenerator.SmallPrimes;
				DevicePrimesDefault = accelerator.Allocate1D<uint>(primesDefault.Length);
				DevicePrimesDefault.View.CopyFromCPU(stream, primesDefault);

				var primesLastOne = DivisorGenerator.SmallPrimesLastOne;
				DevicePrimesLastOne = accelerator.Allocate1D<uint>(primesLastOne.Length);
				DevicePrimesLastOne.View.CopyFromCPU(stream, primesLastOne);

				var primesLastSeven = DivisorGenerator.SmallPrimesLastSeven;
				DevicePrimesLastSeven = accelerator.Allocate1D<uint>(primesLastSeven.Length);
				DevicePrimesLastSeven.View.CopyFromCPU(stream, primesLastSeven);

				var primesLastThree = DivisorGenerator.SmallPrimesLastThree;
				DevicePrimesLastThree = accelerator.Allocate1D<uint>(primesLastThree.Length);
				DevicePrimesLastThree.View.CopyFromCPU(stream, primesLastThree);

				var primesLastNine = DivisorGenerator.SmallPrimesLastNine;
				DevicePrimesLastNine = accelerator.Allocate1D<uint>(primesLastNine.Length);
				DevicePrimesLastNine.View.CopyFromCPU(stream, primesLastNine);

				var primesPow2Default = DivisorGenerator.SmallPrimesPow2;
				DevicePrimesPow2Default = accelerator.Allocate1D<ulong>(primesPow2Default.Length);
				DevicePrimesPow2Default.View.CopyFromCPU(stream, primesPow2Default);

				var primesPow2LastOne = DivisorGenerator.SmallPrimesPow2LastOne;
				DevicePrimesPow2LastOne = accelerator.Allocate1D<ulong>(primesPow2LastOne.Length);
				DevicePrimesPow2LastOne.View.CopyFromCPU(stream, primesPow2LastOne);

				var primesPow2LastSeven = DivisorGenerator.SmallPrimesPow2LastSeven;
				DevicePrimesPow2LastSeven = accelerator.Allocate1D<ulong>(primesPow2LastSeven.Length);
				DevicePrimesPow2LastSeven.View.CopyFromCPU(stream, primesPow2LastSeven);

				var primesPow2LastThree = DivisorGenerator.SmallPrimesPow2LastThree;
				DevicePrimesPow2LastThree = accelerator.Allocate1D<ulong>(primesPow2LastThree.Length);
				DevicePrimesPow2LastThree.View.CopyFromCPU(stream, primesPow2LastThree);

				var primesPow2LastNine = DivisorGenerator.SmallPrimesPow2LastNine;
				DevicePrimesPow2LastNine = accelerator.Allocate1D<ulong>(primesPow2LastNine.Length);
				DevicePrimesPow2LastNine.View.CopyFromCPU(stream, primesPow2LastNine);

				HeuristicPrimeSieves.EnsureInitialized();
				HeuristicCombinedPrimeTester.EnsureInitialized();

				var heuristicGroupA = HeuristicPrimeSieves.GroupADivisors;
				HeuristicGroupADivisors = CopySpanToDevice(accelerator, stream, heuristicGroupA);

				var heuristicGroupASquares = HeuristicPrimeSieves.GroupADivisorSquares;
				HeuristicGroupADivisorSquares = CopySpanToDevice(accelerator, stream, heuristicGroupASquares);

				HeuristicGroupBDivisorsEnding1 = CopyUintSpanToDevice(accelerator, stream, DivisorGenerator.SmallPrimesLastOneWithoutLastThree);
				HeuristicGroupBDivisorSquaresEnding1 = CopySpanToDevice(accelerator, stream, DivisorGenerator.SmallPrimesPow2LastOneWithoutLastThree);

				HeuristicGroupBDivisorsEnding7 = CopyUintSpanToDevice(accelerator, stream, DivisorGenerator.SmallPrimesLastSevenWithoutLastThree);
				HeuristicGroupBDivisorSquaresEnding7 = CopySpanToDevice(accelerator, stream, DivisorGenerator.SmallPrimesPow2LastSevenWithoutLastThree);

				HeuristicGroupBDivisorsEnding9 = CopyUintSpanToDevice(accelerator, stream, DivisorGenerator.SmallPrimesLastNineWithoutLastThree);
				HeuristicGroupBDivisorSquaresEnding9 = CopySpanToDevice(accelerator, stream, DivisorGenerator.SmallPrimesPow2LastNineWithoutLastThree);

				var combinedEnding1 = HeuristicCombinedPrimeTester.CombinedDivisorsEnding1Span;
				HeuristicCombinedDivisorsEnding1 = CopySpanToDevice(accelerator, stream, combinedEnding1);
				var combinedSquaresEnding1 = HeuristicCombinedPrimeTester.CombinedDivisorsEnding1SquaresSpan;
				HeuristicCombinedDivisorSquaresEnding1 = CopySpanToDevice(accelerator, stream, combinedSquaresEnding1);

				var combinedEnding3 = HeuristicCombinedPrimeTester.CombinedDivisorsEnding3Span;
				HeuristicCombinedDivisorsEnding3 = CopySpanToDevice(accelerator, stream, combinedEnding3);
				var combinedSquaresEnding3 = HeuristicCombinedPrimeTester.CombinedDivisorsEnding3SquaresSpan;
				HeuristicCombinedDivisorSquaresEnding3 = CopySpanToDevice(accelerator, stream, combinedSquaresEnding3);

				var combinedEnding7 = HeuristicCombinedPrimeTester.CombinedDivisorsEnding7Span;
				HeuristicCombinedDivisorsEnding7 = CopySpanToDevice(accelerator, stream, combinedEnding7);
				var combinedSquaresEnding7 = HeuristicCombinedPrimeTester.CombinedDivisorsEnding7SquaresSpan;
				HeuristicCombinedDivisorSquaresEnding7 = CopySpanToDevice(accelerator, stream, combinedSquaresEnding7);

				var combinedEnding9 = HeuristicCombinedPrimeTester.CombinedDivisorsEnding9Span;
				HeuristicCombinedDivisorsEnding9 = CopySpanToDevice(accelerator, stream, combinedEnding9);
				var combinedSquaresEnding9 = HeuristicCombinedPrimeTester.CombinedDivisorsEnding9SquaresSpan;
				HeuristicCombinedDivisorSquaresEnding9 = CopySpanToDevice(accelerator, stream, combinedSquaresEnding9);
			}

			internal MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimesDefault { get; }
			internal MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimesLastOne { get; }
			internal MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimesLastSeven { get; }
			internal MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimesLastThree { get; }
			internal MemoryBuffer1D<uint, Stride1D.Dense> DevicePrimesLastNine { get; }
			internal MemoryBuffer1D<ulong, Stride1D.Dense> DevicePrimesPow2Default { get; }
			internal MemoryBuffer1D<ulong, Stride1D.Dense> DevicePrimesPow2LastOne { get; }
			internal MemoryBuffer1D<ulong, Stride1D.Dense> DevicePrimesPow2LastSeven { get; }
			internal MemoryBuffer1D<ulong, Stride1D.Dense> DevicePrimesPow2LastThree { get; }
			internal MemoryBuffer1D<ulong, Stride1D.Dense> DevicePrimesPow2LastNine { get; }
			internal MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupADivisors { get; }
			internal MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupADivisorSquares { get; }
			internal MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupBDivisorsEnding1 { get; }
			internal MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupBDivisorSquaresEnding1 { get; }
			internal MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupBDivisorsEnding7 { get; }
			internal MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupBDivisorSquaresEnding7 { get; }
			internal MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupBDivisorsEnding9 { get; }
			internal MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicGroupBDivisorSquaresEnding9 { get; }
			internal MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorsEnding1 { get; }
			internal MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorSquaresEnding1 { get; }
			internal MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorsEnding3 { get; }
			internal MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorSquaresEnding3 { get; }
			internal MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorsEnding7 { get; }
			internal MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorSquaresEnding7 { get; }
			internal MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorsEnding9 { get; }
			internal MemoryBuffer1D<ulong, Stride1D.Dense> HeuristicCombinedDivisorSquaresEnding9 { get; }

			internal HeuristicGpuDivisorTables CreateHeuristicDivisorTables()
			{
				return new HeuristicGpuDivisorTables(
					HeuristicCombinedDivisorsEnding1.View,
					HeuristicCombinedDivisorsEnding3.View,
					HeuristicCombinedDivisorsEnding7.View,
					HeuristicCombinedDivisorsEnding9.View,
					HeuristicCombinedDivisorSquaresEnding1.View,
					HeuristicCombinedDivisorSquaresEnding3.View,
					HeuristicCombinedDivisorSquaresEnding7.View,
					HeuristicCombinedDivisorSquaresEnding9.View,
					HeuristicGroupADivisors.View,
					HeuristicGroupADivisorSquares.View,
					HeuristicGroupBDivisorsEnding1.View,
					HeuristicGroupBDivisorSquaresEnding1.View,
					HeuristicGroupBDivisorsEnding7.View,
					HeuristicGroupBDivisorSquaresEnding7.View,
					HeuristicGroupBDivisorsEnding9.View,
					HeuristicGroupBDivisorSquaresEnding9.View);
			}

			internal void AddReference()
			{
				_referenceCount++;
			}

			internal void Release()
			{
				if (_referenceCount > 0)
				{
					_referenceCount--;
				}
			}

			private static MemoryBuffer1D<ulong, Stride1D.Dense> CopySpanToDevice(Accelerator accelerator, AcceleratorStream stream, ReadOnlySpan<ulong> span)
			{
				var buffer = accelerator.Allocate1D<ulong>(span.Length);
				if (!span.IsEmpty)
				{
					ref ulong sourceRef = ref MemoryMarshal.GetReference(span);
					buffer.View.CopyFromCPU(stream, ref sourceRef, span.Length);
				}

				return buffer;
			}

			private static MemoryBuffer1D<ulong, Stride1D.Dense> CopyUintSpanToDevice(Accelerator accelerator, AcceleratorStream stream, ReadOnlySpan<uint> span)
			{
				var buffer = accelerator.Allocate1D<ulong>(span.Length);
				if (!span.IsEmpty)
				{
					var converted = new ulong[span.Length];
					for (int i = 0; i < span.Length; i++)
					{
						converted[i] = span[i];
					}

					buffer.View.CopyFromCPU(stream, converted);
				}

				return buffer;
			}

			internal void Dispose()
			{
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
				HeuristicGroupADivisors.Dispose();
				HeuristicGroupADivisorSquares.Dispose();
				HeuristicGroupBDivisorsEnding1.Dispose();
				HeuristicGroupBDivisorSquaresEnding1.Dispose();
				HeuristicGroupBDivisorsEnding7.Dispose();
				HeuristicGroupBDivisorSquaresEnding7.Dispose();
				HeuristicGroupBDivisorsEnding9.Dispose();
				HeuristicGroupBDivisorSquaresEnding9.Dispose();
				HeuristicCombinedDivisorsEnding1.Dispose();
				HeuristicCombinedDivisorSquaresEnding1.Dispose();
				HeuristicCombinedDivisorsEnding3.Dispose();
				HeuristicCombinedDivisorSquaresEnding3.Dispose();
				HeuristicCombinedDivisorsEnding7.Dispose();
				HeuristicCombinedDivisorSquaresEnding7.Dispose();
				HeuristicCombinedDivisorsEnding9.Dispose();
				HeuristicCombinedDivisorSquaresEnding9.Dispose();
			}
		}
	}

	// Expose cache clearing for accelerator disposal coordination
	public static void ClearGpuCaches(Accelerator accelerator)
	{
		PrimeTesterGpuContextPool.Clear(accelerator);
	}

	internal static void DisposeGpuContexts()
	{
		PrimeTesterGpuContextPool.DisposeAll();
		lock (GpuWarmUpLock)
		{
			WarmedGpuLeaseCount = 0;
		}
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
		var stream = gpu.Stream;

		var kernel = KernelUtil.GetKernel(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ArrayView<byte>>(PrimeTesterKernels.SharesFactorKernel)).CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView<ulong>, ArrayView<byte>>>();

		int length = values.Length;
		var inputBuffer = accelerator.Allocate1D<ulong>(length);
		var resultBuffer = accelerator.Allocate1D<byte>(length);

		ArrayPool<ulong> pool = ThreadStaticPools.UlongPool;
		ulong[] temp = pool.Rent(length);
		values.CopyTo(temp);
		inputBuffer.View.CopyFromCPU(stream, ref temp[0], length);
		kernel(stream, length, inputBuffer.View, resultBuffer.View);
		resultBuffer.View.CopyToCPU(stream, ref results[0], length);
		stream.Synchronize();
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
