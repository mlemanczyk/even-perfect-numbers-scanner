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
	public bool IsPrimeGpu(ulong n)
	{
		bool forceCpu = GpuContextPool.ForceCpu;
		bool belowGpuRange = n < 31UL;

		if (forceCpu || belowGpuRange)
		{
			return IsPrime(n);
		}

		var limiter = GpuPrimeWorkLimiter.Acquire();
		var gpu = PrimeTesterGpuContextPool.Rent(1);
		var accelerator = gpu.Accelerator;
		var kernel = gpu.Kernel;

		ulong value = n;
		byte flag = 0;

		var input = gpu.Input;
		var output = gpu.Output;
		var inputView = input.View;
		var outputView = output.View;

		inputView.CopyFromCPU(ref value, 1);
		kernel(
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
		accelerator.Synchronize();
		outputView.CopyToCPU(ref flag, 1);

		gpu.Dispose();
		limiter.Dispose();

		return flag != 0;
	}

	public static int GpuBatchSize { get; set; } = 262_144;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static void IsPrimeBatchGpu(ReadOnlySpan<ulong> values, Span<byte> results)
	{
		var limiter = GpuPrimeWorkLimiter.Acquire();
		var gpu = PrimeTesterGpuContextPool.Rent(GpuBatchSize);
		var accelerator = gpu.Accelerator;
		var kernel = gpu.Kernel;
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
			inputView.CopyFromCPU(ref valueRef, count);

			kernel(
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
			accelerator.Synchronize();
			var resultSlice = results.Slice(pos, count);
			ref byte resultRef = ref MemoryMarshal.GetReference(resultSlice);
			outputView.CopyToCPU(ref resultRef, count);

			pos += count;
		}

		gpu.Dispose();
		limiter.Dispose();
	}

	internal static class PrimeTesterGpuContextPool
	{
		private static readonly ConcurrentQueue<PrimeTesterGpuContextLease> Pool = new();
		private static readonly object SharedTablesLock = new();
		private static readonly Dictionary<string, SharedHeuristicGpuTables> SharedTables = new();

		private static string CreateAcceleratorKey(Accelerator accelerator)
		{
			return accelerator.AcceleratorType + ":" + accelerator.Name;
		}

		private static SharedHeuristicGpuTables RentSharedTables(Accelerator accelerator)
		{
			string key = CreateAcceleratorKey(accelerator);

			lock (SharedTablesLock)
			{
				if (!SharedTables.TryGetValue(key, out var tables))
				{
					tables = new SharedHeuristicGpuTables(accelerator, key);
					SharedTables.Add(key, tables);
				}

				tables.AddReference();
				return tables;
			}
		}

		private static void ReleaseSharedTables(SharedHeuristicGpuTables tables)
		{
			lock (SharedTablesLock)
			{
				if (tables.Release())
				{
					SharedTables.Remove(tables.Key);
				}
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

			lease.ResetReturnFlag();
			return lease;
		}

		internal static void Return(PrimeTesterGpuContextLease lease)
		{
			lease.Accelerator.Synchronize();
			Pool.Enqueue(lease);
		}

		internal static void DisposeAll()
		{
			while (Pool.TryDequeue(out var lease))
			{
				lease.DisposeResources();
			}
		}

		internal static void Clear(Accelerator accelerator)
		{
			if (accelerator is null)
			{
				return;
			}

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
		}

		internal sealed class PrimeTesterGpuContextLease : IDisposable
		{
			private bool _returned;
			private readonly SharedHeuristicGpuTables _sharedTables;

			internal readonly Context AcceleratorContext;
			public readonly Accelerator Accelerator;
			public readonly Action<Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>> Kernel;
			public readonly Action<Index1D, HeuristicGpuDivisorTables, ArrayView<int>, ulong, ulong, HeuristicGpuDivisorTableKind, byte> HeuristicTrialDivisionKernel;
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

			internal PrimeTesterGpuContextLease(int minBufferCapacity)
			{
				AcceleratorContext = Context.CreateDefault();
				Accelerator = AcceleratorContext.GetPreferredDevice(false).CreateAccelerator(AcceleratorContext);
				Kernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>>(PrimeTesterKernels.SmallPrimeSieveKernel);
				HeuristicTrialDivisionKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, HeuristicGpuDivisorTables, ArrayView<int>, ulong, ulong, HeuristicGpuDivisorTableKind, byte>(PrimeTesterKernels.HeuristicTrialDivisionKernel);

				_sharedTables = RentSharedTables(Accelerator);

				EnsureCapacity(minBufferCapacity);
				_returned = false;
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

			internal void ResetReturnFlag()
			{
				_returned = false;
			}

			public void Dispose()
			{
				if (_returned)
				{
					return;
				}

				_returned = true;
				PrimeTesterGpuContextPool.Return(this);
			}

			internal void DisposeResources()
			{
				Input?.Dispose();
				Output?.Dispose();
				HeuristicFlag?.Dispose();
				ReleaseSharedTables(_sharedTables);
				Accelerator.Dispose();
				AcceleratorContext.Dispose();
			}
		}

		private sealed class SharedHeuristicGpuTables
		{
			private int _referenceCount;

			internal SharedHeuristicGpuTables(Accelerator accelerator, string key)
			{
				Key = key;

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

				HeuristicPrimeSieves.EnsureInitialized();
				HeuristicCombinedPrimeTester.EnsureInitialized();

				var heuristicGroupA = HeuristicPrimeSieves.GroupADivisors;
				HeuristicGroupADivisors = CopySpanToDevice(accelerator, heuristicGroupA);

				var heuristicGroupASquares = HeuristicPrimeSieves.GroupADivisorSquares;
				HeuristicGroupADivisorSquares = CopySpanToDevice(accelerator, heuristicGroupASquares);

				HeuristicGroupBDivisorsEnding1 = CopyUintSpanToDevice(accelerator, DivisorGenerator.SmallPrimesLastOneWithoutLastThree);
				HeuristicGroupBDivisorSquaresEnding1 = CopySpanToDevice(accelerator, DivisorGenerator.SmallPrimesPow2LastOneWithoutLastThree);

				HeuristicGroupBDivisorsEnding7 = CopyUintSpanToDevice(accelerator, DivisorGenerator.SmallPrimesLastSevenWithoutLastThree);
				HeuristicGroupBDivisorSquaresEnding7 = CopySpanToDevice(accelerator, DivisorGenerator.SmallPrimesPow2LastSevenWithoutLastThree);

				HeuristicGroupBDivisorsEnding9 = CopyUintSpanToDevice(accelerator, DivisorGenerator.SmallPrimesLastNineWithoutLastThree);
				HeuristicGroupBDivisorSquaresEnding9 = CopySpanToDevice(accelerator, DivisorGenerator.SmallPrimesPow2LastNineWithoutLastThree);

				var combinedEnding1 = HeuristicCombinedPrimeTester.CombinedDivisorsEnding1Span;
				HeuristicCombinedDivisorsEnding1 = CopySpanToDevice(accelerator, combinedEnding1);
				var combinedSquaresEnding1 = HeuristicCombinedPrimeTester.CombinedDivisorsEnding1SquaresSpan;
				HeuristicCombinedDivisorSquaresEnding1 = CopySpanToDevice(accelerator, combinedSquaresEnding1);

				var combinedEnding3 = HeuristicCombinedPrimeTester.CombinedDivisorsEnding3Span;
				HeuristicCombinedDivisorsEnding3 = CopySpanToDevice(accelerator, combinedEnding3);
				var combinedSquaresEnding3 = HeuristicCombinedPrimeTester.CombinedDivisorsEnding3SquaresSpan;
				HeuristicCombinedDivisorSquaresEnding3 = CopySpanToDevice(accelerator, combinedSquaresEnding3);

				var combinedEnding7 = HeuristicCombinedPrimeTester.CombinedDivisorsEnding7Span;
				HeuristicCombinedDivisorsEnding7 = CopySpanToDevice(accelerator, combinedEnding7);
				var combinedSquaresEnding7 = HeuristicCombinedPrimeTester.CombinedDivisorsEnding7SquaresSpan;
				HeuristicCombinedDivisorSquaresEnding7 = CopySpanToDevice(accelerator, combinedSquaresEnding7);

				var combinedEnding9 = HeuristicCombinedPrimeTester.CombinedDivisorsEnding9Span;
				HeuristicCombinedDivisorsEnding9 = CopySpanToDevice(accelerator, combinedEnding9);
				var combinedSquaresEnding9 = HeuristicCombinedPrimeTester.CombinedDivisorsEnding9SquaresSpan;
				HeuristicCombinedDivisorSquaresEnding9 = CopySpanToDevice(accelerator, combinedSquaresEnding9);
			}

			internal string Key { get; }
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

			internal void AddReference()
			{
				_referenceCount++;
			}

			internal bool Release()
			{
				_referenceCount--;
				if (_referenceCount == 0)
				{
					Dispose();
					return true;
				}

				return false;
			}

			private static MemoryBuffer1D<ulong, Stride1D.Dense> CopySpanToDevice(Accelerator accelerator, ReadOnlySpan<ulong> span)
			{
				var buffer = accelerator.Allocate1D<ulong>(span.Length);
				if (!span.IsEmpty)
				{
					ref ulong sourceRef = ref MemoryMarshal.GetReference(span);
					buffer.View.CopyFromCPU(ref sourceRef, span.Length);
				}

				return buffer;
			}

			private static MemoryBuffer1D<ulong, Stride1D.Dense> CopyUintSpanToDevice(Accelerator accelerator, ReadOnlySpan<uint> span)
			{
				var buffer = accelerator.Allocate1D<ulong>(span.Length);
				if (!span.IsEmpty)
				{
					var converted = new ulong[span.Length];
					for (int i = 0; i < span.Length; i++)
					{
						converted[i] = span[i];
					}

					buffer.View.CopyFromCPU(converted);
				}

				return buffer;
			}

			private void Dispose()
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
