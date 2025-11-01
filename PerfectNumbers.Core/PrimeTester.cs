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
		// bool belowGpuRange = n < 31UL;

		// if (belowGpuRange)
		// {
			return IsPrime(n);
		// }

		// var limiter = GpuPrimeWorkLimiter.Acquire();
		// var gpu = PrimeTesterGpuContextPool.Rent(1);
		// var accelerator = gpu.Accelerator;
		// var kernel = gpu.Kernel;

		// ulong value = n;
		// byte flag = 0;

		// var input = gpu.Input;
		// var output = gpu.Output;
		// var inputView = input.View;
		// var outputView = output.View;

		// inputView.CopyFromCPU(ref value, 1);
		// kernel(
		// 				1,
		// 				inputView,
		// 				gpu.DevicePrimesDefault.View,
		// 				gpu.DevicePrimesLastOne.View,
		// 				gpu.DevicePrimesLastSeven.View,
		// 				gpu.DevicePrimesLastThree.View,
		// 				gpu.DevicePrimesLastNine.View,
		// 				gpu.DevicePrimesPow2Default.View,
		// 				gpu.DevicePrimesPow2LastOne.View,
		// 				gpu.DevicePrimesPow2LastSeven.View,
		// 				gpu.DevicePrimesPow2LastThree.View,
		// 				gpu.DevicePrimesPow2LastNine.View,
		// 				outputView);
		// accelerator.Synchronize();
		// outputView.CopyToCPU(ref flag, 1);

		// gpu.Dispose();
		// limiter.Dispose();

		// return flag != 0;
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

			int toWarm = target - WarmedGpuLeaseCount;
			var leases = new PrimeTesterGpuContextPool.PrimeTesterGpuContextLease[toWarm];
			ReadOnlySpan<byte> supportedDigits = PrimeTesterGpuContextPool.SupportedDigits;
			int digitIndex = 0;
			for (int i = 0; i < toWarm; i++)
			{
				byte trailingDigit = supportedDigits[digitIndex];
				leases[i] = PrimeTesterGpuContextPool.Rent(trailingDigit, GpuBatchSize);
				digitIndex++;
				if (digitIndex == supportedDigits.Length)
				{
					digitIndex = 0;
				}
			}

			for (int i = 0; i < toWarm; i++)
			{
				leases[i].Dispose();
			}

			WarmedGpuLeaseCount = target;
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static void IsPrimeBatchGpu(ReadOnlySpan<ulong> values, Span<byte> results)
	{
		var limiter = GpuPrimeWorkLimiter.Acquire();
		int totalLength = values.Length;
		int batchSize = GpuBatchSize;
		int pos = 0;

		while (pos < totalLength)
		{
			ulong currentValue = values[pos];
			byte trailingDigit = (byte)(currentValue % 10UL);
			var gpu = PrimeTesterGpuContextPool.Rent(trailingDigit, GpuBatchSize);
			var accelerator = gpu.Accelerator;
			var kernel = gpu.Kernel;
			var input = gpu.Input;
			var output = gpu.Output;
			var inputView = input.View;
			var outputView = output.View;

			int blockStart = pos;
			while (blockStart < totalLength)
			{
				byte blockDigit = (byte)(values[blockStart] % 10UL);
				if (blockDigit != trailingDigit)
				{
					break;
				}

				int limit = Math.Min(batchSize, totalLength - blockStart);
				int count = 0;
				while (count < limit && (byte)(values[blockStart + count] % 10UL) == trailingDigit)
				{
					count++;
				}

				if (count == 0)
				{
					break;
				}

				var valueSlice = values.Slice(blockStart, count);
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
				var resultSlice = results.Slice(blockStart, count);
				ref byte resultRef = ref MemoryMarshal.GetReference(resultSlice);
				outputView.CopyToCPU(ref resultRef, count);

				blockStart += count;
			}

			gpu.Dispose();
			pos = blockStart;
		}

		limiter.Dispose();
	}

	internal static class PrimeTesterGpuContextPool
	{
		internal enum HeuristicGpuTableScope : byte
		{
			GroupAB = 0,
			GroupABWithCombined = 1,
		}

		private static readonly byte[] SupportedTrailingDigits = [1, 3, 7, 9];
		private const int ScopeCount = 2;
		private static readonly ConcurrentQueue<PrimeTesterGpuContextLease>[,] Pools;
		private static readonly object SharedTablesLock = new();
		private static readonly Dictionary<string, SharedKernelTables> SharedTables = new();
		private static readonly Dictionary<(string Key, byte Digit, HeuristicGpuTableScope Scope), TrailingDigitTables> DigitTables = new();

		static PrimeTesterGpuContextPool()
		{
			Pools = new ConcurrentQueue<PrimeTesterGpuContextLease>[SupportedTrailingDigits.Length, ScopeCount];
			for (int digitIndex = 0; digitIndex < SupportedTrailingDigits.Length; digitIndex++)
			{
				for (int scopeIndex = 0; scopeIndex < ScopeCount; scopeIndex++)
				{
					Pools[digitIndex, scopeIndex] = new ConcurrentQueue<PrimeTesterGpuContextLease>();
				}
			}
		}

		internal static ReadOnlySpan<byte> SupportedDigits => SupportedTrailingDigits;

		private static int GetPoolIndex(byte trailingDigit)
		{
			return trailingDigit switch
			{
				1 => 0,
				3 => 1,
				7 => 2,
				9 => 3,
				_ => throw new ArgumentOutOfRangeException(nameof(trailingDigit), trailingDigit, "Only trailing digits 1, 3, 7, and 9 are supported."),
			};
		}

		private static int GetScopeIndex(HeuristicGpuTableScope scope)
		{
			return scope switch
			{
				HeuristicGpuTableScope.GroupAB => 0,
				HeuristicGpuTableScope.GroupABWithCombined => 1,
				_ => throw new ArgumentOutOfRangeException(nameof(scope), scope, "Unsupported heuristic table scope."),
			};
		}

		internal static PrimeTesterGpuContextLease Rent(byte trailingDigit, int minBufferCapacity = 1, HeuristicGpuTableScope tableScope = HeuristicGpuTableScope.GroupAB)
		{
			var pool = Pools[GetPoolIndex(trailingDigit), GetScopeIndex(tableScope)];
			if (!pool.TryDequeue(out var lease))
			{
				lease = new PrimeTesterGpuContextLease(trailingDigit, minBufferCapacity, tableScope);
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
			Pools[GetPoolIndex(lease.TrailingDigit), GetScopeIndex(lease.TableScope)].Enqueue(lease);
		}

		internal static void DisposeAll()
		{
			for (int digitIndex = 0; digitIndex < SupportedTrailingDigits.Length; digitIndex++)
			{
				for (int scopeIndex = 0; scopeIndex < ScopeCount; scopeIndex++)
				{
					var pool = Pools[digitIndex, scopeIndex];
					while (pool.TryDequeue(out var lease))
					{
						lease.DisposeResources();
					}
				}
			}

			lock (SharedTablesLock)
			{
				foreach (var entry in DigitTables.Values)
				{
					entry.Dispose();
				}

				DigitTables.Clear();

				foreach (var entry in SharedTables.Values)
				{
					entry.Dispose();
				}

				SharedTables.Clear();
			}
		}

		internal static void Clear(Accelerator accelerator)
		{
			if (accelerator is null)
			{
				return;
			}

			var retained = new List<PrimeTesterGpuContextLease>();
			for (int digitIndex = 0; digitIndex < SupportedTrailingDigits.Length; digitIndex++)
			{
				for (int scopeIndex = 0; scopeIndex < ScopeCount; scopeIndex++)
				{
					var pool = Pools[digitIndex, scopeIndex];
					while (pool.TryDequeue(out var lease))
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
						pool.Enqueue(retained[i]);
					}

					retained.Clear();
				}
			}

			lock (SharedTablesLock)
			{
				string key = CreateAcceleratorKey(accelerator);

				if (SharedTables.Remove(key, out var shared))
				{
					shared.Dispose();
				}

				if (DigitTables.Count > 0)
				{
					var toRemove = new List<(string Key, byte Digit, HeuristicGpuTableScope Scope)>();
					foreach (var entry in DigitTables)
					{
						if (entry.Key.Key == key)
						{
							entry.Value.Dispose();
							toRemove.Add(entry.Key);
						}
					}

					for (int i = 0; i < toRemove.Count; i++)
					{
						DigitTables.Remove(toRemove[i]);
					}
				}
			}
		}

		private static string CreateAcceleratorKey(Accelerator accelerator)
		{
			return accelerator.AcceleratorType + ":" + accelerator.Name;
		}

		private static SharedKernelTables RentSharedTables(Accelerator accelerator)
		{
			string key = CreateAcceleratorKey(accelerator);

			lock (SharedTablesLock)
			{
				if (!SharedTables.TryGetValue(key, out var tables))
				{
					tables = new SharedKernelTables(accelerator, key);
					SharedTables.Add(key, tables);
				}

				tables.AddReference();
				return tables;
			}
		}

		private static void ReleaseSharedTables(SharedKernelTables tables)
		{
			if (tables is null)
			{
				return;
			}

			lock (SharedTablesLock)
			{
				if (tables.Release() && SharedTables.Remove(tables.Key, out _))
				{
					tables.Dispose();
				}
			}
		}

		private static TrailingDigitTables RentTrailingDigitTables(Accelerator accelerator, byte trailingDigit, HeuristicGpuTableScope scope)
		{
			string key = CreateAcceleratorKey(accelerator);
			var compositeKey = (key, trailingDigit, scope);

			lock (SharedTablesLock)
			{
				if (!DigitTables.TryGetValue(compositeKey, out var tables))
				{
					tables = new TrailingDigitTables(accelerator, key, trailingDigit, scope);
					DigitTables.Add(compositeKey, tables);
				}

				tables.AddReference();
				return tables;
			}
		}

		private static void ReleaseTrailingDigitTables(TrailingDigitTables tables)
		{
			if (tables is null)
			{
				return;
			}

			lock (SharedTablesLock)
			{
				var compositeKey = (tables.Key, tables.TrailingDigit, tables.Scope);
				if (tables.Release() && DigitTables.Remove(compositeKey))
				{
					tables.Dispose();
				}
			}
		}

		internal sealed class PrimeTesterGpuContextLease : IDisposable
	{
		private bool _returned;
		private readonly SharedKernelTables _sharedTables;
		private readonly TrailingDigitTables _digitTables;
		private readonly HeuristicGpuDivisorTables _heuristicDivisorTables;
		private readonly HeuristicGpuCombinedDivisorTables _heuristicCombinedTables;

		internal readonly Context AcceleratorContext;
		public readonly Accelerator Accelerator;
		public readonly byte TrailingDigit;
		public readonly HeuristicGpuTableScope TableScope;
		public readonly Action<Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>> Kernel;
		public readonly Action<Index1D, ArrayView<int>, ulong, ulong, HeuristicGpuDivisorTableKind, HeuristicGpuDivisorTables, HeuristicGpuCombinedDivisorTables> HeuristicTrialDivisionKernel;
		public MemoryBuffer1D<ulong, Stride1D.Dense> Input = null!;
		public MemoryBuffer1D<byte, Stride1D.Dense> Output = null!;
		public MemoryBuffer1D<int, Stride1D.Dense> HeuristicFlag = null!;
		public int BufferCapacity;

		internal PrimeTesterGpuContextLease(byte trailingDigit, int minBufferCapacity, HeuristicGpuTableScope tableScope)
		{
			TrailingDigit = trailingDigit;
			TableScope = tableScope;

			AcceleratorContext = Context.CreateDefault();
			Accelerator = AcceleratorContext.GetPreferredDevice(false).CreateAccelerator(AcceleratorContext);
			Kernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>>(PrimeTesterKernels.SmallPrimeSieveKernel);
			HeuristicTrialDivisionKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ulong, ulong, HeuristicGpuDivisorTableKind, HeuristicGpuDivisorTables, HeuristicGpuCombinedDivisorTables>(PrimeTesterKernels.HeuristicTrialDivisionKernel);

			_sharedTables = RentSharedTables(Accelerator);
			_digitTables = RentTrailingDigitTables(Accelerator, trailingDigit, tableScope);

			_heuristicDivisorTables = new HeuristicGpuDivisorTables(
				_sharedTables.HeuristicGroupADivisors.View,
				_digitTables.GroupBDivisorsEnding1View,
				_digitTables.GroupBDivisorsEnding7View,
				_digitTables.GroupBDivisorsEnding9View);
			HeuristicGpuDivisorTables.InitializeShared(in _heuristicDivisorTables);

			if (_digitTables.HasCombinedTables)
			{
				_heuristicCombinedTables = new HeuristicGpuCombinedDivisorTables(
					_digitTables.CombinedDivisorsEnding1View,
					_digitTables.CombinedDivisorsEnding3View,
					_digitTables.CombinedDivisorsEnding7View,
					_digitTables.CombinedDivisorsEnding9View);
				HeuristicGpuCombinedDivisorTables.InitializeShared(in _heuristicCombinedTables);
			}
			else
			{
				_heuristicCombinedTables = HeuristicGpuCombinedDivisorTables.Shared;
			}

			EnsureCapacity(minBufferCapacity);
			_returned = false;
		}

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

		public HeuristicGpuDivisorTables HeuristicGpuTables => _heuristicDivisorTables;
		public HeuristicGpuCombinedDivisorTables HeuristicCombinedTables => _heuristicCombinedTables;

		internal void EnsureCapacity(int minCapacity)
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
			ReleaseTrailingDigitTables(_digitTables);
			ReleaseSharedTables(_sharedTables);
			Accelerator.Dispose();
			AcceleratorContext.Dispose();
		}
	}
	private sealed class SharedKernelTables
		{
			private int _referenceCount;

			internal SharedKernelTables(Accelerator accelerator, string key)
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
				HeuristicGroupADivisors = CopySpanToDevice(accelerator, HeuristicPrimeSieves.GroupADivisors);
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

			internal void AddReference()
			{
				_referenceCount++;
			}

			internal bool Release()
			{
				if (_referenceCount > 0)
				{
					_referenceCount--;
				}

				return _referenceCount == 0;
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
			}
		}

		private sealed class TrailingDigitTables
		{
			private int _referenceCount;

			internal TrailingDigitTables(Accelerator accelerator, string key, byte trailingDigit, HeuristicGpuTableScope scope)
			{
				Key = key;
				TrailingDigit = trailingDigit;
				Scope = scope;

				HeuristicPrimeSieves.EnsureInitialized();
				HeuristicCombinedPrimeTester.EnsureInitialized();

				if (trailingDigit == 1 || trailingDigit == 7 || trailingDigit == 9)
				{
					GroupBDivisorsEnding1 = CopyUintSpanToDevice(accelerator, DivisorGenerator.SmallPrimesLastOneWithoutLastThree);
				}

				if (trailingDigit == 3 || trailingDigit == 7 || trailingDigit == 9)
				{
					GroupBDivisorsEnding7 = CopyUintSpanToDevice(accelerator, DivisorGenerator.SmallPrimesLastSevenWithoutLastThree);
				}

				if (trailingDigit == 1 || trailingDigit == 3 || trailingDigit == 9)
				{
					GroupBDivisorsEnding9 = CopyUintSpanToDevice(accelerator, DivisorGenerator.SmallPrimesLastNineWithoutLastThree);
				}

				if (scope == HeuristicGpuTableScope.GroupABWithCombined)
				{
					CombinedDivisorsEnding1 = trailingDigit == 1 ? CopySpanToDevice(accelerator, HeuristicCombinedPrimeTester.CombinedDivisorsEnding1Span) : null;
					CombinedDivisorsEnding3 = trailingDigit == 3 ? CopySpanToDevice(accelerator, HeuristicCombinedPrimeTester.CombinedDivisorsEnding3Span) : null;
					CombinedDivisorsEnding7 = trailingDigit == 7 ? CopySpanToDevice(accelerator, HeuristicCombinedPrimeTester.CombinedDivisorsEnding7Span) : null;
					CombinedDivisorsEnding9 = trailingDigit == 9 ? CopySpanToDevice(accelerator, HeuristicCombinedPrimeTester.CombinedDivisorsEnding9Span) : null;
				}
			}

			internal string Key { get; }
			internal byte TrailingDigit { get; }
			internal HeuristicGpuTableScope Scope { get; }

			internal bool HasCombinedTables => Scope == HeuristicGpuTableScope.GroupABWithCombined;

			internal MemoryBuffer1D<ulong, Stride1D.Dense>? CombinedDivisorsEnding1 { get; }
			internal MemoryBuffer1D<ulong, Stride1D.Dense>? CombinedDivisorsEnding3 { get; }
			internal MemoryBuffer1D<ulong, Stride1D.Dense>? CombinedDivisorsEnding7 { get; }
			internal MemoryBuffer1D<ulong, Stride1D.Dense>? CombinedDivisorsEnding9 { get; }
			internal MemoryBuffer1D<ulong, Stride1D.Dense>? GroupBDivisorsEnding1 { get; }
			internal MemoryBuffer1D<ulong, Stride1D.Dense>? GroupBDivisorsEnding7 { get; }
			internal MemoryBuffer1D<ulong, Stride1D.Dense>? GroupBDivisorsEnding9 { get; }

			internal ArrayView1D<ulong, Stride1D.Dense> CombinedDivisorsEnding1View => CombinedDivisorsEnding1?.View ?? ArrayView1D<ulong, Stride1D.Dense>.Empty;
			internal ArrayView1D<ulong, Stride1D.Dense> CombinedDivisorsEnding3View => CombinedDivisorsEnding3?.View ?? ArrayView1D<ulong, Stride1D.Dense>.Empty;
			internal ArrayView1D<ulong, Stride1D.Dense> CombinedDivisorsEnding7View => CombinedDivisorsEnding7?.View ?? ArrayView1D<ulong, Stride1D.Dense>.Empty;
			internal ArrayView1D<ulong, Stride1D.Dense> CombinedDivisorsEnding9View => CombinedDivisorsEnding9?.View ?? ArrayView1D<ulong, Stride1D.Dense>.Empty;
			internal ArrayView1D<ulong, Stride1D.Dense> GroupBDivisorsEnding1View => GroupBDivisorsEnding1?.View ?? ArrayView1D<ulong, Stride1D.Dense>.Empty;
			internal ArrayView1D<ulong, Stride1D.Dense> GroupBDivisorsEnding7View => GroupBDivisorsEnding7?.View ?? ArrayView1D<ulong, Stride1D.Dense>.Empty;
			internal ArrayView1D<ulong, Stride1D.Dense> GroupBDivisorsEnding9View => GroupBDivisorsEnding9?.View ?? ArrayView1D<ulong, Stride1D.Dense>.Empty;

			internal void AddReference()
			{
				_referenceCount++;
			}

			internal bool Release()
			{
				if (_referenceCount > 0)
				{
					_referenceCount--;
				}

				return _referenceCount == 0;
			}

			internal void Dispose()
			{
				CombinedDivisorsEnding1?.Dispose();
				CombinedDivisorsEnding3?.Dispose();
				CombinedDivisorsEnding7?.Dispose();
				CombinedDivisorsEnding9?.Dispose();
				GroupBDivisorsEnding1?.Dispose();
				GroupBDivisorsEnding7?.Dispose();
				GroupBDivisorsEnding9?.Dispose();
			}
		}

		private static MemoryBuffer1D<ulong, Stride1D.Dense> CopySpanToDevice(Accelerator accelerator, ReadOnlySpan<ulong> span)
		{
			var buffer = accelerator.Allocate1D<ulong>(span.Length);
			if (!span.IsEmpty)
			{
				ref ulong source = ref MemoryMarshal.GetReference(span);
				buffer.View.CopyFromCPU(ref source, span.Length);
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
		var gpu = PrimeTesterGpuContextPool.Rent(1);
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
