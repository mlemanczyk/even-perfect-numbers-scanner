using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu.Kernels;

namespace PerfectNumbers.Core.Gpu.Accelerators;

public static class PrimeOrderCalculatorAcceleratorCapacityProviders
{
	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	public static void EnsureCalculateOrderWideCapacity(this PrimeOrderCalculatorAccelerator gpu, int newSize)
	{
		if (newSize > gpu.Input.Length)
		{
			gpu.CalculateOrderWideExponentBuffer.Dispose();
			gpu.CalculateOrderWideRemainderBuffer.Dispose();

			var accelerator = gpu.Accelerator;
			// lock (accelerator)
			// {
			gpu.CalculateOrderWideExponentBuffer = accelerator.Allocate1D<GpuUInt128>(newSize);
			gpu.CalculateOrderWideRemainderBuffer = accelerator.Allocate1D<GpuUInt128>(newSize);
			gpu.CalculateOrderWideExponentBufferView = gpu.CalculateOrderWideExponentBuffer.View;
			gpu.CalculateOrderWideRemainderBufferView = gpu.CalculateOrderWideRemainderBuffer.View;
			// }			
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	public static void EnsurePrimeOrderCalculatorCapacity(this PrimeOrderCalculatorAccelerator gpu, int factorsCount, int primeTesterCapacity)
	{
		Accelerator accelerator = gpu.Accelerator;

		MemoryBuffer1D<KeyValuePair<ulong, int>, Stride1D.Dense> pow2ModEntriesToTestOnDevice = gpu.Pow2ModEntriesToTestOnDevice;
		if (pow2ModEntriesToTestOnDevice.Length < factorsCount)
		{
			// lock (accelerator)
			// {
				pow2ModEntriesToTestOnDevice.Dispose();
				pow2ModEntriesToTestOnDevice = accelerator.Allocate1D<KeyValuePair<ulong, int>>(factorsCount);
				gpu.Pow2ModEntriesToTestOnDevice = pow2ModEntriesToTestOnDevice;
				gpu.Pow2ModEntriesToTestOnDeviceView = pow2ModEntriesToTestOnDevice.View;
			// }
		}

		MemoryBuffer1D<ulong, Stride1D.Dense> input = gpu.Input;
		if (input.Length < primeTesterCapacity)
		{
			// lock (accelerator)
			// {
				input.Dispose();

				input = accelerator.Allocate1D<ulong>(primeTesterCapacity);
				gpu.Input = input;
				gpu.InputView = input.View;
			// }
		}

		MemoryBuffer1D<byte, Stride1D.Dense> outputByte = gpu.OutputByte;
		if (outputByte.Length < primeTesterCapacity)
		{
			// lock (accelerator)
			// {
				outputByte.Dispose();

				outputByte = accelerator.Allocate1D<byte>(primeTesterCapacity);
				gpu.OutputByte = outputByte;
				gpu.OutputByteView = outputByte.View;
			// }
		}

		MemoryBuffer1D<int, Stride1D.Dense> outputInt = gpu.OutputInt;
		if (outputInt.Length < primeTesterCapacity)
		{
			// lock (accelerator)
			// {
				outputInt.Dispose();

				outputInt = accelerator.Allocate1D<int>(primeTesterCapacity);
				gpu.OutputInt = outputInt;
				gpu.OutputIntView = outputInt.View;
			// }
		}
	}

	public static void EnsurePartialFactorCapacity(this PrimeOrderCalculatorAccelerator gpu, int newSize)
	{
		var accelerator = gpu.Accelerator;

		var outputUlong = gpu.OutputUlong;
		if (newSize > outputUlong.Length)
		{
			outputUlong.Dispose();

			outputUlong = accelerator.Allocate1D<ulong>(newSize);
			gpu.OutputUlong = outputUlong;
			gpu.OutputUlongView = outputUlong.View;
		}

		var outputInt = gpu.OutputInt;
		if (newSize > outputInt.Length)
		{
			outputInt.Dispose();
			
			outputInt = accelerator.Allocate1D<int>(newSize);
			gpu.OutputInt = outputInt;
			gpu.OutputIntView = outputInt.View;
		}
	}

	public static void EnsureSmallPrimeFactorSlotsCapacity(this PrimeOrderCalculatorAccelerator gpu, int newSize)
	{
		// Console.WriteLine($"Resizing GPU scratch buffer from pool ({buffer.SmallPrimeFactorPrimeSlots.Length} / {smallPrimeFactorSlotCount}), ({buffer.SpecialMaxFactors.Length}/{specialMaxFactorCapacity})");
		var accelerator = gpu.Accelerator;
		MemoryBuffer1D<ulong, Stride1D.Dense> outputUlong = gpu.OutputUlong;
		if (outputUlong.Length < newSize)
		{
			// lock (accelerator)
			// {
				outputUlong.Dispose();

				outputUlong = accelerator.Allocate1D<ulong>(newSize);
				gpu.OutputUlong = outputUlong;
				gpu.OutputUlongView = outputUlong.View;
			// }
		}

		MemoryBuffer1D<int, Stride1D.Dense> outputInt = gpu.OutputInt;
		if (outputInt.Length < newSize)
		{
			// lock (accelerator)
			// {
				outputInt.Dispose();

				outputInt = accelerator.Allocate1D<int>(newSize);
				gpu.OutputInt = outputInt;
				gpu.OutputIntView = outputInt.View;
			// }
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	public static void EnsureUlongInputOutputCapacity(this PrimeOrderCalculatorAccelerator gpu, int newSize)
	{
		var accelerator = gpu.Accelerator;
		MemoryBuffer1D<ulong, Stride1D.Dense> buffer = gpu.Input;
		if (buffer.Length < newSize)
		{
			// lock (accelerator)
			// {
				buffer.Dispose();

				buffer = accelerator.Allocate1D<ulong>(newSize);
				gpu.Input = buffer;
				gpu.InputView = buffer.View;
			// }
		}

		buffer = gpu.OutputUlong;
		if (buffer.Length < newSize)
		{
			// lock (accelerator)
			// {
				buffer.Dispose();

				buffer = accelerator.Allocate1D<ulong>(newSize);
				gpu.OutputUlong = buffer;
				gpu.OutputUlongView = buffer.View;
			// }
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
	public static BitContradictionBuffers EnsureBitContradictionCapacity(this PrimeOrderCalculatorAccelerator gpu, int maxBatchIndexWords, int maxQWords, int delta8CacheConcurrentThreads)
	{
		var accelerator = gpu.Accelerator;

		MemoryBuffer1D<ulong, Stride1D.Dense> bufferUlong = gpu.Input;
		if (bufferUlong.Length < maxBatchIndexWords)
		{
			// lock (accelerator)
			// {
				bufferUlong.Dispose();

				bufferUlong = accelerator.Allocate1D<ulong>(maxBatchIndexWords);
				gpu.Input = bufferUlong;

			// }
		}

		MemoryBuffer1D<int, Stride1D.Dense> bufferInt = gpu.OutputInt;
		maxQWords <<= 1;
		if (bufferUlong.Length < maxQWords)
		{
			// lock (accelerator)
			// {
				bufferInt.Dispose();

				bufferInt = accelerator.Allocate1D<int>(maxQWords);
				gpu.OutputInt = bufferInt;
			// }
		}

		bufferInt = gpu.StackIndexBuffer;
		int requiredCapacity = BitContradictionKernels.GetDeltaCacheKeyLength(delta8CacheConcurrentThreads);
		if (bufferInt.Length < requiredCapacity)
		{
			bufferInt.Dispose();
			bufferInt = accelerator.Allocate1D<int>(requiredCapacity);
			gpu.StackIndexBuffer = bufferInt;

			bufferInt = gpu.StackExponentBuffer;
			bufferInt.Dispose();
			bufferInt = accelerator.Allocate1D<int>(BitContradictionKernels.GetDeltaCacheIntLength(delta8CacheConcurrentThreads));
			gpu.StackExponentBuffer = bufferInt;
		}

		return new(
			gpu.Input.View,
			gpu.OutputInt.View,
			gpu.StackIndexBuffer.View,
			gpu.StackExponentBuffer.View
		);
	}
}

public readonly struct BitContradictionBuffers(
	in ArrayView1D<ulong, Stride1D.Dense> batchIndexWords,
	in ArrayView1D<int, Stride1D.Dense> foundQWords,
	in ArrayView1D<int, Stride1D.Dense> delta8CacheKeys,
	in ArrayView1D<int, Stride1D.Dense> delta8CacheValues
)
{
	public readonly ArrayView1D<ulong, Stride1D.Dense> BatchIndexWordsView = batchIndexWords;
	public readonly ArrayView1D<int,  Stride1D.Dense> FoundQWordsView = foundQWords;
	public readonly ArrayView1D<int, Stride1D.Dense> Delta8CacheKeysView = delta8CacheKeys;
	public readonly ArrayView1D<int, Stride1D.Dense> Delta8CacheValuesView = delta8CacheValues;
}

public sealed class PrimeOrderCalculatorAccelerator
{
	#region Pool

	// [ThreadStatic]
	private static ConcurrentFixedCapacityStack<PrimeOrderCalculatorAccelerator>? _pool;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static PrimeOrderCalculatorAccelerator Rent(int primeTesterCapacity)
	{
		var pool = _pool ??= new(PerfectNumberConstants.DefaultThreadPoolCapacity);
		if (pool.TryPop(out var gpu))
		{
			gpu.EnsurePrimeOrderCalculatorCapacity(PerfectNumberConstants.DefaultFactorsBuffer, primeTesterCapacity);
			return gpu;
		}

		int acceleratorIndex = AcceleratorPool.Shared.Rent();
		return new(acceleratorIndex, PerfectNumberConstants.DefaultFactorsBuffer, primeTesterCapacity, PerfectNumberConstants.DefaultSmallPrimeFactorSlotCount, PerfectNumberConstants.DefaultSpecialMaxFactorCapacity);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static void Return(PrimeOrderCalculatorAccelerator gpu) => _pool!.Push(gpu);

	internal static void Clear()
	{
		var pool = _pool;
		if (pool != null)
		{
			while (pool.Pop() is { } gpu)
			{
				gpu.Dispose();
			}
		}
	}

	internal static void DisposeAll()
	{
		var pool = _pool;
		if (pool != null)
		{
			while (pool.Pop() is { } gpu)
			{
				gpu.Dispose();
			}
		}
	}

	internal static void WarmUp()
	{
		Accelerator[] accelerators = AcceleratorPool.Shared.Accelerators;
		int acceleratorCount = EnvironmentConfiguration.RollingAccelerators;
		if (acceleratorCount > accelerators.Length)
		{
			acceleratorCount = accelerators.Length;
		}
		for (var i = 0; i < acceleratorCount; i++)
		{
			Console.WriteLine($"Preparing accelerator {i}...");
			var accelerator = accelerators[i];
			AcceleratorStreamPool.WarmUp(i);
			// Don't take this from the pool as quick uploads of data to the accelerator consumes much of GPU's memory and throws.
			AcceleratorStream stream = accelerator.CreateStream();
			if (SkipOrderTablesAndKernels)
			{
				stream.Synchronize();
				stream.Dispose();
				continue;
			}

			// TODO: Review which tables are needed when final execution path is defined.
			LastDigitGpuTables.WarmUp(i, stream);
			_kernels[i] = new CalculatorKernels(accelerator);
			HeuristicCombinedGpuTables.WarmUp(i, stream);
			SmallPrimeFactorGpuTables.WarmUp(i, stream);
			// SharedHeuristicGpuTables.EnsureStaticTables(accelerator, stream);
			// _ = GpuKernelPool.GetOrAddKernels(accelerator, stream, KernelType.None);
			// KernelContainer kernels = GpuKernelPool.GetOrAddKernels(accelerator, stream);
			// GpuStaticTableInitializer.EnsureStaticTables(accelerator, kernels, stream);
			stream.Synchronize();
			stream.Dispose();
		}
	}

	#endregion
	#region Static Tables

	public readonly ArrayView1D<uint, Stride1D.Dense> DevicePrimesLastOne;
	public readonly ArrayView1D<uint, Stride1D.Dense> DevicePrimesLastSeven;
	public readonly ArrayView1D<uint, Stride1D.Dense> DevicePrimesLastThree;
	public readonly ArrayView1D<uint, Stride1D.Dense> DevicePrimesLastNine;
	public readonly ArrayView1D<ulong, Stride1D.Dense> DevicePrimesPow2LastOne;
	public readonly ArrayView1D<ulong, Stride1D.Dense> DevicePrimesPow2LastSeven;
	public readonly ArrayView1D<ulong, Stride1D.Dense> DevicePrimesPow2LastThree;
	public readonly ArrayView1D<ulong, Stride1D.Dense> DevicePrimesPow2LastNine;

	public readonly HeuristicCombinedGpuViews DivisorTables;

	public readonly ArrayView1D<uint, Stride1D.Dense> SmallPrimeFactorPrimes;
	public readonly ArrayView1D<ulong, Stride1D.Dense> SmallPrimeFactorSquares;

	#endregion

	private static readonly Accelerator[] _accelerators = AcceleratorPool.Shared.Accelerators;

	public static bool SkipOrderTablesAndKernels;


	public readonly Accelerator Accelerator;
	public readonly int AcceleratorIndex;
	
	public MemoryBuffer1D<ulong, Stride1D.Dense> Input;
	public ArrayView1D<ulong, Stride1D.Dense> InputView;
	
	public MemoryBuffer1D<byte, Stride1D.Dense> OutputByte;
	public ArrayView1D<byte, Stride1D.Dense> OutputByteView;

	public MemoryBuffer1D<ulong, Stride1D.Dense> OutputUlong;
	public readonly MemoryBuffer1D<ulong, Stride1D.Dense> OutputUlong2;
	public ArrayView1D<ulong, Stride1D.Dense> OutputUlongView;
	public readonly ArrayView<ulong> OutputUlongView2;

	public MemoryBuffer1D<int, Stride1D.Dense> OutputInt;
	public readonly MemoryBuffer1D<int, Stride1D.Dense> OutputInt2;	
	public ArrayView1D<int, Stride1D.Dense> OutputIntView;
	public readonly ArrayView1D<int, Stride1D.Dense> OutputIntView2;

	public MemoryBuffer1D<KeyValuePair<ulong, int>, Stride1D.Dense> Pow2ModEntriesToTestOnDevice;
	public ArrayView1D<KeyValuePair<ulong, int>, Stride1D.Dense> Pow2ModEntriesToTestOnDeviceView;

	public readonly MemoryBuffer1D<ulong, Stride1D.Dense> WorkFactorBuffer;
	public readonly MemoryBuffer1D<ulong, Stride1D.Dense> CandidateBuffer;
	public MemoryBuffer1D<int, Stride1D.Dense> StackIndexBuffer;
	public MemoryBuffer1D<int, Stride1D.Dense> StackExponentBuffer;
	public readonly MemoryBuffer1D<ulong, Stride1D.Dense> StackProductBuffer;

	public OrderKernelBuffers CalculateOrderKernelBuffers;
	public readonly ArrayView1D<ulong, Stride1D.Dense> WorkFactorBufferView;
	public readonly ArrayView1D<ulong, Stride1D.Dense> CandidateBufferView;
	public readonly ArrayView1D<int, Stride1D.Dense> StackIndexBufferView;
	public readonly ArrayView1D<int, Stride1D.Dense> StackExponentBufferView;
	public readonly ArrayView1D<ulong, Stride1D.Dense> StackProductBufferView;

	public MemoryBuffer1D<GpuUInt128, Stride1D.Dense> CalculateOrderWideExponentBuffer;
	public MemoryBuffer1D<GpuUInt128, Stride1D.Dense> CalculateOrderWideRemainderBuffer;
	public ArrayView1D<GpuUInt128, Stride1D.Dense> CalculateOrderWideExponentBufferView;
	public ArrayView1D<GpuUInt128, Stride1D.Dense> CalculateOrderWideRemainderBufferView;

	public Dictionary<ulong, int> Pow2ModEntriesToTestOnHost = [];

	private static readonly CalculatorKernels[] _kernels = new CalculatorKernels[_accelerators.Length];

	private sealed class CalculatorKernels(Accelerator accelerator)
	{
		// public readonly Action<AcceleratorStream, Index1D, ulong, CalculateOrderKernelConfig, ulong, ulong, ulong, ulong, ulong, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, OrderKernelBuffers> CalculateOrderKernelLauncher = KernelUtil.GetKernel(accelerator.LoadAutoGroupedStreamKernel<Index1D, ulong, CalculateOrderKernelConfig, ulong, ulong, ulong, ulong, ulong, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, OrderKernelBuffers>(PrimeOrderKernels.CalculateOrderKernel)).CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, CalculateOrderKernelConfig, ulong, ulong, ulong, ulong, ulong, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, OrderKernelBuffers>>();

		public readonly Kernel CheckFactorsKernel = KernelUtil.GetKernel(accelerator.LoadStreamKernel<int, ulong, ArrayView1D<KeyValuePair<ulong, int>, Stride1D.Dense>, ulong, ulong, ulong, ulong, ulong, ArrayView1D<ulong, Stride1D.Dense>>(PrimeOrderKernels.CheckFactorsKernel));

		public readonly Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, ulong, ulong, ulong, ulong, ulong, ArrayView1D<ulong, Stride1D.Dense>> ConvertToStandardKernelLauncher = KernelUtil.GetKernel(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<ulong, Stride1D.Dense>, ulong, ulong, ulong, ulong, ulong, ArrayView1D<ulong, Stride1D.Dense>>(Pow2MontgomeryKernels.Pow2MontgomeryKernelConvertToStandard)).CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, ulong, ulong, ulong, ulong, ulong, ArrayView1D<ulong, Stride1D.Dense>>>();

		// public readonly Action<AcceleratorStream, Index1D, byte, ArrayView<int>, ulong, ulong, HeuristicCombinedGpuViews> HeuristicCombinedTrialDivisionKernelLauncher = KernelUtil.GetKernel(accelerator.LoadAutoGroupedStreamKernel<Index1D, byte, ArrayView<int>, ulong, ulong, HeuristicCombinedGpuViews>(PrimeTesterKernels.HeuristicTrialCombinedDivisionKernel)).CreateLauncherDelegate<Action<AcceleratorStream, Index1D, byte, ArrayView<int>, ulong, ulong, HeuristicCombinedGpuViews>>();
		public readonly Action<AcceleratorStream, Index1D, ArrayView<int>, ulong, ulong, ArrayView<ulong>, ArrayView<ulong>> HeuristicCombinedTrialDivisionKernelLauncher = KernelUtil.GetKernel(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ulong, ulong, ArrayView<ulong>, ArrayView<ulong>>(PrimeTesterKernels.HeuristicTrialCombinedDivisionKernel)).CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView<int>, ulong, ulong, ArrayView<ulong>, ArrayView<ulong>>>();

		public readonly Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, ulong, ulong, ulong, ulong, ulong, ArrayView1D<ulong, Stride1D.Dense>> KeepMontgomeryKernelLauncher = KernelUtil.GetKernel(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<ulong, Stride1D.Dense>, ulong, ulong, ulong, ulong, ulong, ArrayView1D<ulong, Stride1D.Dense>>(Pow2MontgomeryKernels.Pow2MontgomeryKernelKeepMontgomery)).CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, ulong, ulong, ulong, ulong, ulong, ArrayView1D<ulong, Stride1D.Dense>>>();

		// public readonly Action<AcceleratorStream, Index1D, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, int, ulong, uint, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<byte, Stride1D.Dense>> PartialFactorKernelLauncher = KernelUtil.GetKernel(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, int, ulong, uint, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<byte, Stride1D.Dense>>(PrimeOrderKernels.PartialFactorKernel)).CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, int, ulong, uint, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<byte, Stride1D.Dense>>>();

		// public readonly Kernel PollardRhoKernel = KernelUtil.GetKernel(accelerator.LoadStreamKernel<ulong, int, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<byte, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>(Pow2MontgomeryKernels.TryPollardRhoKernel));

		public readonly Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, ulong, ulong, ulong, ulong, ulong, ArrayView1D<ulong, Stride1D.Dense>> Pow2ModKernelLauncher = KernelUtil.GetKernel(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<ulong, Stride1D.Dense>, ulong, ulong, ulong, ulong, ulong, ArrayView1D<ulong, Stride1D.Dense>>(PrimeOrderKernels.Pow2ModKernel)).CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, ulong, ulong, ulong, ulong, ulong, ArrayView1D<ulong, Stride1D.Dense>>>();

		public readonly Action<AcceleratorStream, Index1D, ArrayView1D<GpuUInt128, Stride1D.Dense>, GpuUInt128, ArrayView1D<GpuUInt128, Stride1D.Dense>> Pow2ModWideKernelLauncher = KernelUtil.GetKernel(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<GpuUInt128, Stride1D.Dense>, GpuUInt128, ArrayView1D<GpuUInt128, Stride1D.Dense>>(PrimeOrderKernels.Pow2ModKernelWide)).CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView1D<GpuUInt128, Stride1D.Dense>, GpuUInt128, ArrayView1D<GpuUInt128, Stride1D.Dense>>>();

		// public readonly Action<AcceleratorStream, Index1D, ArrayView<ulong>, ArrayView<byte>> SharesFactorKernelLauncher = KernelUtil.GetKernel(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ArrayView<byte>>(PrimeTesterKernels.SharesFactorKernel)).CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView<ulong>, ArrayView<byte>>>();

		public readonly Action<AcceleratorStream, Index1D, ulong, uint, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>> SmallPrimeFactorKernelLauncher = KernelUtil.GetKernel(accelerator.LoadAutoGroupedStreamKernel<Index1D, ulong, uint, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>(SmallPrimeFactorKernels.SmallPrimeFactorKernelScan)).CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, uint, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>>();

		// public readonly Action<AcceleratorStream, Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>> SmallPrimeSieveKernelLauncher = KernelUtil.GetKernel(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>>(PrimeTesterKernels.SmallPrimeSieveKernel)).CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>>>();

		public readonly Action<AcceleratorStream, Index1D, ulong, ArrayView<ulong>, int, ulong, ArrayView<ulong>> SpecialMaxKernelLauncher = KernelUtil.GetKernel(accelerator.LoadAutoGroupedStreamKernel<Index1D, ulong, ArrayView<ulong>, int, ulong, ArrayView<ulong>>(PrimeOrderGpuHeuristics.EvaluateSpecialMaxCandidatesKernel)).CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, ArrayView<ulong>, int, ulong, ArrayView<ulong>>>();
	}

	// public readonly Action<AcceleratorStream, Index1D, ulong, CalculateOrderKernelConfig, ulong, ulong, ulong, ulong, ulong, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, OrderKernelBuffers> CalculateOrderKernelLauncher;
	public readonly Kernel CheckFactorsKernel;
	public readonly Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, ulong, ulong, ulong, ulong, ulong, ArrayView1D<ulong, Stride1D.Dense>> ConvertToStandardKernelLauncher;
	public readonly Action<AcceleratorStream, Index1D, ArrayView<int>, ulong, ulong, ArrayView<ulong>, ArrayView<ulong>> HeuristicCombinedTrialDivisionKernelLauncher;
	public readonly Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, ulong, ulong, ulong, ulong, ulong, ArrayView1D<ulong, Stride1D.Dense>> KeepMontgomeryKernelLauncher;
	public readonly Action<AcceleratorStream, Index1D, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, int, ulong, uint, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<byte, Stride1D.Dense>> PartialFactorKernelLauncher;
	public readonly Kernel PollardRhoKernel;
	public readonly Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, ulong, ulong, ulong, ulong, ulong, ArrayView1D<ulong, Stride1D.Dense>> Pow2ModKernelLauncher;
	public readonly Action<AcceleratorStream, Index1D, ArrayView1D<GpuUInt128, Stride1D.Dense>, GpuUInt128, ArrayView1D<GpuUInt128, Stride1D.Dense>> Pow2ModWideKernelLauncher;
	public readonly Action<AcceleratorStream, Index1D, ArrayView<ulong>, ArrayView<byte>> SharesFactorKernelLauncher;
	public readonly Action<AcceleratorStream, Index1D, ulong, uint, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>> SmallPrimeFactorKernelLauncher;
	public readonly Action<AcceleratorStream, Index1D, ArrayView<ulong>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<uint>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>> SmallPrimeSieveKernelLauncher;
	public readonly Action<AcceleratorStream, Index1D, ulong, ArrayView<ulong>, int, ulong, ArrayView<ulong>> SpecialMaxKernelLauncher;

	public PrimeOrderCalculatorAccelerator(int acceleratorIndex, int factorsCount, int primeTesterCapacity, int smallPrimeFactorSlotCount, int specialMaxFactorCapacity)
	{
		var accelerator = _accelerators[acceleratorIndex];
		Accelerator = accelerator;
		AcceleratorIndex = acceleratorIndex;

		Input = accelerator.Allocate1D<ulong>(Math.Max(Math.Max(specialMaxFactorCapacity, primeTesterCapacity), PrimeOrderConstants.MaxGpuBatchSize));
		InputView = Input.View;

		OutputByte = Accelerator.Allocate1D<byte>(primeTesterCapacity);
		OutputByteView = OutputByte.View;

		OutputInt = Accelerator.Allocate1D<int>(Math.Max(Math.Max(primeTesterCapacity, smallPrimeFactorSlotCount), PrimeOrderConstants.GpuSmallPrimeFactorSlots));
		OutputIntView = OutputInt.View;

		OutputInt2 = accelerator.Allocate1D<int>(PrimeOrderConstants.GpuSmallPrimeFactorSlots);			
		OutputIntView2 = OutputInt2.View;

		OutputUlong = accelerator.Allocate1D<ulong>(Math.Max(Math.Max(Math.Max(specialMaxFactorCapacity, smallPrimeFactorSlotCount), PrimeOrderConstants.MaxGpuBatchSize), PrimeOrderConstants.GpuSmallPrimeFactorSlots));
		OutputUlongView = OutputUlong.View;

		OutputUlong2 = accelerator.Allocate1D<ulong>(1);
		OutputUlongView2 = OutputUlong2.View;

		Pow2ModEntriesToTestOnDevice = accelerator.Allocate1D<KeyValuePair<ulong, int>>(factorsCount);
		Pow2ModEntriesToTestOnDeviceView = Pow2ModEntriesToTestOnDevice.View;

		WorkFactorBuffer = accelerator.Allocate1D<ulong>(PrimeOrderConstants.GpuSmallPrimeFactorSlots);
		CandidateBuffer = accelerator.Allocate1D<ulong>(PrimeOrderConstants.HeuristicCandidateLimit);
		StackIndexBuffer = accelerator.Allocate1D<int>(PrimeOrderConstants.HeuristicStackCapacity);
		StackExponentBuffer = accelerator.Allocate1D<int>(PrimeOrderConstants.HeuristicStackCapacity);
		StackProductBuffer = accelerator.Allocate1D<ulong>(PrimeOrderConstants.HeuristicStackCapacity);

		WorkFactorBufferView = WorkFactorBuffer.View;
		CandidateBufferView = CandidateBuffer.View;
		StackIndexBufferView = StackIndexBuffer.View;
		StackExponentBufferView = StackExponentBuffer.View;
		StackProductBufferView = StackProductBuffer.View;

		CalculateOrderKernelBuffers = new OrderKernelBuffers(
			OutputUlongView,
			OutputIntView,
			WorkFactorBufferView,
			OutputIntView2,
			CandidateBufferView,
			StackIndexBufferView,
			StackExponentBufferView,
			StackProductBufferView,
			OutputUlongView2,
			OutputByteView
		);

		CalculateOrderWideExponentBuffer = accelerator.Allocate1D<GpuUInt128>(PrimeOrderConstants.MaxGpuBatchSize);
		CalculateOrderWideRemainderBuffer = accelerator.Allocate1D<GpuUInt128>(PrimeOrderConstants.MaxGpuBatchSize);
		CalculateOrderWideExponentBufferView = CalculateOrderWideExponentBuffer.View;
		CalculateOrderWideRemainderBufferView = CalculateOrderWideRemainderBuffer.View;

		if (SkipOrderTablesAndKernels)
		{
			DevicePrimesLastOne = default;
			DevicePrimesLastSeven = default;
			DevicePrimesLastThree = default;
			DevicePrimesLastNine = default;
			DevicePrimesPow2LastOne = default;
			DevicePrimesPow2LastSeven = default;
			DevicePrimesPow2LastThree = default;
			DevicePrimesPow2LastNine = default;
			SmallPrimeFactorPrimes = default;
			SmallPrimeFactorSquares = default;
			CheckFactorsKernel = default;
			ConvertToStandardKernelLauncher = default;
			HeuristicCombinedTrialDivisionKernelLauncher = default;
			Pow2ModKernelLauncher = default;
			Pow2ModWideKernelLauncher = default;
			SmallPrimeFactorKernelLauncher = default;
			SpecialMaxKernelLauncher = default;
			return;
		}

		LastDigitGpuTables lastDigitSharedTables = LastDigitGpuTables.GetStaticTables(acceleratorIndex);
		DevicePrimesLastOne = lastDigitSharedTables.DevicePrimesLastOne.View;
		DevicePrimesLastSeven = lastDigitSharedTables.DevicePrimesLastSeven.View;
		DevicePrimesLastThree = lastDigitSharedTables.DevicePrimesLastThree.View;
		DevicePrimesLastNine = lastDigitSharedTables.DevicePrimesLastNine.View;
		DevicePrimesPow2LastOne = lastDigitSharedTables.DevicePrimesPow2LastOne.View;
		DevicePrimesPow2LastSeven = lastDigitSharedTables.DevicePrimesPow2LastSeven.View;
		DevicePrimesPow2LastThree = lastDigitSharedTables.DevicePrimesPow2LastThree.View;
		DevicePrimesPow2LastNine = lastDigitSharedTables.DevicePrimesPow2LastNine.View;

		// DivisorTables = HeuristicCombinedGpuTables
		// 	.GetStaticTables(acceleratorIndex)
		// 	.CreateViews();

		var smallPrimeFactorTables = SmallPrimeFactorGpuTables.GetStaticTables(acceleratorIndex);

		if (smallPrimeFactorTables.HasValue)
		{
			SmallPrimeFactorGpuTables tables = smallPrimeFactorTables.Value;
			SmallPrimeFactorPrimes = tables.Primes.View;
			SmallPrimeFactorSquares = tables.Squares.View;
		}

		var kernels = _kernels[acceleratorIndex];

		// CalculateOrderKernelLauncher = kernels.CalculateOrderKernelLauncher;
		CheckFactorsKernel = kernels.CheckFactorsKernel;
		ConvertToStandardKernelLauncher = kernels.ConvertToStandardKernelLauncher;
		HeuristicCombinedTrialDivisionKernelLauncher = kernels.HeuristicCombinedTrialDivisionKernelLauncher;
		// KeepMontgomeryKernelLauncher = kernels.KeepMontgomeryKernelLauncher;
		// PartialFactorKernelLauncher = kernels.PartialFactorKernelLauncher;
		// PollardRhoKernel = kernels.PollardRhoKernel;
		Pow2ModKernelLauncher = kernels.Pow2ModKernelLauncher;
		Pow2ModWideKernelLauncher = kernels.Pow2ModWideKernelLauncher;
		// SharesFactorKernelLauncher = kernels.SharesFactorKernelLauncher;
		SmallPrimeFactorKernelLauncher = kernels.SmallPrimeFactorKernelLauncher;
		// SmallPrimeSieveKernelLauncher = kernels.SmallPrimeSieveKernelLauncher;
		SpecialMaxKernelLauncher = kernels.SpecialMaxKernelLauncher;
	}

	public void Dispose()
	{
		Input.Dispose();
		OutputByte.Dispose();
		OutputInt.Dispose();
		OutputInt2.Dispose();
		OutputUlong.Dispose();
		OutputUlong2.Dispose();
		
		Pow2ModEntriesToTestOnDevice.Dispose();

		WorkFactorBuffer.Dispose();
		CandidateBuffer.Dispose();
		StackIndexBuffer.Dispose();
		StackExponentBuffer.Dispose();
		StackProductBuffer.Dispose();

		CalculateOrderWideExponentBuffer.Dispose();
		CalculateOrderWideRemainderBuffer.Dispose();

		// These resources are shared between GPU leases
		// Stream.Dispose();
		// Accelerator.Dispose();
		// Context.Dispose();
	}
}
