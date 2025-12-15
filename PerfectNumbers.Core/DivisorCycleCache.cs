using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu;
using PerfectNumbers.Core.Gpu.Accelerators;

namespace PerfectNumbers.Core;

public sealed class DivisorCycleCache
{
	private const byte ByteZero = 0;
	private const byte ByteOne = 1;
	private const int StackBufferThreshold = 128;
	private readonly Accelerator _accelerator;
	private readonly int _acceleratorIndex;
	private readonly Action<AcceleratorStream, Index1D, int, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<byte, Stride1D.Dense>> _gpuKernel;
	private volatile ulong[] _snapshot;
	private readonly int _divisorCyclesBatchSize;
	private static int _sharedDivisorCyclesBatchSize = GpuConstants.GpuCycleStepsPerInvocation;

	private const int CycleCacheTrackingLimit = 500_000;
	private static readonly HashSet<ulong> CycleCacheTrackedDivisors = [];
	private static readonly object CycleCacheTrackingLock = new();

	public static void SetDivisorCyclesBatchSize(int divisorCyclesBatchSize)
	{
		_sharedDivisorCyclesBatchSize = Math.Max(1, divisorCyclesBatchSize);
	}

	public static readonly DivisorCycleCache Shared = new(_sharedDivisorCyclesBatchSize);

	public int PreferredBatchSize => _divisorCyclesBatchSize;

	private DivisorCycleCache(int divisorCyclesBatchSize)
	{
		_divisorCyclesBatchSize = Math.Max(1, divisorCyclesBatchSize);
		var acceleratorIndex = AcceleratorPool.Shared.Rent();
		_acceleratorIndex = acceleratorIndex;
		Accelerator accelerator = AcceleratorPool.Shared.Accelerators[acceleratorIndex];
		_accelerator = accelerator;
		_gpuKernel = LoadKernel(accelerator);
		_snapshot = MersenneDivisorCycles.Shared.ExportSmallCyclesSnapshot();
	}

	public void RefreshSnapshot()
	{
		_snapshot = MersenneDivisorCycles.Shared.ExportSmallCyclesSnapshot();
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public ulong GetCycleLengthCpu(ulong divisor, in MontgomeryDivisorData divisorData)
	{
		ulong[] snapshot = _snapshot;

		if (divisor < (ulong)snapshot.Length)
		{
			ulong cached = snapshot[divisor];
			ArgumentOutOfRangeException.ThrowIfZero(cached);

			return cached;
		}

		return MersenneDivisorCycles.CalculateCycleLengthCpu(divisor, divisorData);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public ulong GetCycleLengthHybrid(PrimeOrderCalculatorAccelerator gpu, ulong divisor, in MontgomeryDivisorData divisorData, bool skipPrimeOrderHeuristic = false)
	{
		if (!skipPrimeOrderHeuristic)
		{
			return GetCycleLengthHybrid(gpu, divisor, divisorData);
		}

		// The skipPrimeOrderHeuristic path is only used for divisors produced by the CPU by-divisor scanner,
		// which always generates values greater than one. Keep this guard disabled here to avoid repeating the
		// validation already performed by the general GetCycleLengths path.
		// if (divisor <= 1UL)
		// {
		//     throw new InvalidDataException("Divisor must be > 1");
		// }

		ulong[] snapshot = _snapshot;
		ulong cycleLength;
		if (divisor < (ulong)snapshot.Length)
		{
			cycleLength = snapshot[divisor];
			ArgumentOutOfRangeException.ThrowIfZero(cycleLength);

			return cycleLength;
		}

		cycleLength = MersenneDivisorCycles.CalculateCycleLengthHybrid(gpu, divisor, divisorData, skipPrimeOrderHeuristic: true);
		return cycleLength;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public ulong GetCycleLengthGpu(PrimeOrderCalculatorAccelerator gpu, ulong divisor, in MontgomeryDivisorData	divisorData, bool skipPrimeOrderHeuristic = false)
	{
		if (!skipPrimeOrderHeuristic)
		{
			Span<ulong> result = stackalloc ulong[1];
			ReadOnlySpan<ulong> singleDivisor = stackalloc ulong[1] { divisor };
			GetCycleLengthsGpu(singleDivisor, result);
			return result[0];
		}

		// The skipPrimeOrderHeuristic path is only used for divisors produced by the CPU by-divisor scanner,
		// which always generates values greater than one. Keep this guard disabled here to avoid repeating the
		// validation already performed by the general GetCycleLengths path.
		// if (divisor <= 1UL)
		// {
		//     throw new InvalidDataException("Divisor must be > 1");
		// }

		ulong[] snapshot = _snapshot;
		ulong cycleLength;
		if (divisor < (ulong)snapshot.Length)
		{
			cycleLength = snapshot[divisor];
			ArgumentOutOfRangeException.ThrowIfZero(cycleLength);

			return cycleLength;
		}

		cycleLength = MersenneDivisorCycles.CalculateCycleLengthGpu(gpu, divisor, divisorData, skipPrimeOrderHeuristic: true);
		return cycleLength;
	}

	public void GetCycleLengthsCpu(ReadOnlySpan<ulong> divisors, Span<ulong> cycles)
	{
		ulong[] snapshot = _snapshot;
		int length = divisors.Length;
		int[]? rentedMissing = null;
		Span<int> missingBuffer = length <= StackBufferThreshold
			? stackalloc int[length]
			: new Span<int>(rentedMissing = ThreadStaticPools.IntPool.Rent(length), 0, length);

		int missingCount = 0;

		for (int i = 0; i < length; i++)
		{
			ulong divisor = divisors[i];

			if (divisor < (ulong)snapshot.Length)
			{
				ulong cached = snapshot[divisor];
				ArgumentOutOfRangeException.ThrowIfZero(cached);

				cycles[i] = cached;
			}
			else
			{
				missingBuffer[missingCount++] = i;
			}
		}

		if (missingCount != 0)
		{
			ReadOnlySpan<int> missingIndices = missingBuffer[..missingCount];
			ComputeCyclesCpu(divisors, cycles, missingIndices);
		}

		if (rentedMissing is not null)
		{
			ThreadStaticPools.IntPool.Return(rentedMissing, clearArray: false);
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public ulong GetCycleLengthHybrid(PrimeOrderCalculatorAccelerator gpu, ulong divisor, in MontgomeryDivisorData divisorData)
	{
		ulong[] snapshot = _snapshot;

		if (divisor < (ulong)snapshot.Length)
		{
			ulong cached = snapshot[divisor];
			ArgumentOutOfRangeException.ThrowIfZero(cached);

			return cached;
		}

		return MersenneDivisorCycles.CalculateCycleLengthHybrid(gpu, divisor, divisorData);
	}

	public void GetCycleLengthsHybrid(PrimeOrderCalculatorAccelerator gpu, ReadOnlySpan<ulong> divisors, Span<ulong> cycles)
	{
		ulong[] snapshot = _snapshot;
		int length = divisors.Length;
		int[]? rentedMissing = null;
		Span<int> missingBuffer = length <= StackBufferThreshold
			? stackalloc int[length]
			: new Span<int>(rentedMissing = ThreadStaticPools.IntPool.Rent(length), 0, length);

		int missingCount = 0;

		for (int i = 0; i < length; i++)
		{
			ulong divisor = divisors[i];

			if (divisor < (ulong)snapshot.Length)
			{
				ulong cached = snapshot[divisor];
				ArgumentOutOfRangeException.ThrowIfZero(cached);

				cycles[i] = cached;
			}
			else
			{
				missingBuffer[missingCount++] = i;
			}
		}

		if (missingCount != 0)
		{
			ReadOnlySpan<int> missingIndices = missingBuffer[..missingCount];
			ComputeCyclesHybrid(gpu, divisors, cycles, missingIndices);
		}

		if (rentedMissing is not null)
		{
			ThreadStaticPools.IntPool.Return(rentedMissing, clearArray: false);
		}
	}

	public void GetCycleLengthsGpu(ReadOnlySpan<ulong> divisors, Span<ulong> cycles)
	{
		ulong[] snapshot = _snapshot;
		int length = divisors.Length;
		int[]? rentedMissing = null;
		Span<int> missingBuffer = length <= StackBufferThreshold
			? stackalloc int[length]
			: new Span<int>(rentedMissing = ThreadStaticPools.IntPool.Rent(length), 0, length);

		int missingCount = 0;

		for (int i = 0; i < length; i++)
		{
			ulong divisor = divisors[i];

			if (divisor < (ulong)snapshot.Length)
			{
				ulong cached = snapshot[divisor];
				ArgumentOutOfRangeException.ThrowIfZero(cached);

				cycles[i] = cached;
			}
			else
			{
				missingBuffer[missingCount++] = i;
			}
		}

		if (missingCount != 0)
		{
			ReadOnlySpan<int> missingIndices = missingBuffer[..missingCount];
			ComputeCyclesGpu(divisors, cycles, missingIndices);
		}

		if (rentedMissing is not null)
		{
			ThreadStaticPools.IntPool.Return(rentedMissing, clearArray: false);
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void ComputeCyclesCpu(ReadOnlySpan<ulong> divisors, Span<ulong> cycles, ReadOnlySpan<int> indices)
	{
		for (int i = 0; i < indices.Length; i++)
		{
			int targetIndex = indices[i];
			ulong divisor = divisors[targetIndex];
			MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(divisor);
			cycles[targetIndex] = MersenneDivisorCycles.CalculateCycleLengthCpu(divisor, divisorData);
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void ComputeCyclesHybrid(PrimeOrderCalculatorAccelerator gpu, ReadOnlySpan<ulong> divisors, Span<ulong> cycles, ReadOnlySpan<int> indices)
	{
		for (int i = 0; i < indices.Length; i++)
		{
			int targetIndex = indices[i];
			ulong divisor = divisors[targetIndex];
			MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(divisor);
			cycles[targetIndex] = MersenneDivisorCycles.CalculateCycleLengthHybrid(gpu, divisor, divisorData);
		}
	}

	private void ComputeCyclesGpu(ReadOnlySpan<ulong> divisors, Span<ulong> cycles, ReadOnlySpan<int> indices)
	{
		int count = indices.Length;
		ulong[]? rentedDivisors = null;
		Span<ulong> workDivisors = count <= StackBufferThreshold
			? stackalloc ulong[count]
			: new Span<ulong>(rentedDivisors = ThreadStaticPools.UlongPool.Rent(count), 0, count);

		ulong[]? rentedResults = null;
		Span<ulong> workResults = count <= StackBufferThreshold
			? stackalloc ulong[count]
			: new Span<ulong>(rentedResults = ThreadStaticPools.UlongPool.Rent(count), 0, count);

		for (int i = 0; i < count; i++)
		{
			workDivisors[i] = divisors[indices[i]];
		}

		ComputeCyclesGpuCore(workDivisors, workResults);

		for (int i = 0; i < count; i++)
		{
			cycles[indices[i]] = workResults[i];
		}

		if (rentedDivisors is not null)
		{
			ThreadStaticPools.UlongPool.Return(rentedDivisors, clearArray: false);
			ThreadStaticPools.UlongPool.Return(rentedResults!, clearArray: false);
		}
	}

	private const int DefaultCapacity = 128;

	[ThreadStatic]
	private static MemoryBuffer1D<ulong, Stride1D.Dense>? _divisorBuffer;

	[ThreadStatic]
	private static MemoryBuffer1D<ulong, Stride1D.Dense>? _powBuffer;

	[ThreadStatic]
	private static MemoryBuffer1D<ulong, Stride1D.Dense>? _orderBuffer;

	[ThreadStatic]
	private static MemoryBuffer1D<ulong, Stride1D.Dense>? _resultBuffer;

	[ThreadStatic]
	private static MemoryBuffer1D<byte, Stride1D.Dense>? _statusBuffer;

	private void EnsureCapacity(int requiredCapacity)
	{
		if (_divisorBuffer is not null && requiredCapacity <= _divisorBuffer.Length)
		{
			return;
		}

		if (_divisorBuffer != null)
		{
			_divisorBuffer.Dispose();
			_powBuffer!.Dispose();
			_orderBuffer!.Dispose();
			_resultBuffer!.Dispose();
			_statusBuffer!.Dispose();
		}

		Accelerator accelerator = _accelerator;
		// lock (accelerator)
		{
			_divisorBuffer = accelerator.Allocate1D<ulong>(requiredCapacity);
			_powBuffer = accelerator.Allocate1D<ulong>(requiredCapacity);
			_orderBuffer = accelerator.Allocate1D<ulong>(requiredCapacity);
			_resultBuffer = accelerator.Allocate1D<ulong>(requiredCapacity);
			_statusBuffer = accelerator.Allocate1D<byte>(requiredCapacity);
		}
	}

	private void ComputeCyclesGpuCore(ReadOnlySpan<ulong> divisors, Span<ulong> destination)
	{
		int length = divisors.Length;

		EnsureCapacity(length);

		ulong[]? rentedPow = null;
		Span<ulong> powSpan = length <= StackBufferThreshold
			? stackalloc ulong[length]
			: new Span<ulong>(rentedPow = ThreadStaticPools.UlongPool.Rent(length), 0, length);

		ulong[]? rentedOrder = null;
		Span<ulong> orderSpan = length <= StackBufferThreshold
			? stackalloc ulong[length]
			: new Span<ulong>(rentedOrder = ThreadStaticPools.UlongPool.Rent(length), 0, length);

		ulong[]? rentedResult = null;
		Span<ulong> resultSpan = length <= StackBufferThreshold
			? stackalloc ulong[length]
			: new Span<ulong>(rentedResult = ThreadStaticPools.UlongPool.Rent(length), 0, length);

		byte[]? rentedStatus = null;
		Span<byte> statusSpan = length <= StackBufferThreshold
			? stackalloc byte[length]
			: new Span<byte>(rentedStatus = ThreadStaticPools.BytePool.Rent(length), 0, length);

		for (int i = 0; i < length; i++)
		{
			ulong divisor = divisors[i];
			ulong initialPow = 2UL;
			if (initialPow >= divisor)
			{
				initialPow -= divisor;
			}

			powSpan[i] = initialPow;
			orderSpan[i] = 1UL;
			resultSpan[i] = 0UL;
			statusSpan[i] = ByteZero;
		}

		// GpuPrimeWorkLimiter.Acquire();
		int pending;
		var kernel = _gpuKernel;
		var acceleratorIndex = _acceleratorIndex;
		var stream = AcceleratorStreamPool.Rent(acceleratorIndex);

		_divisorBuffer!.View.CopyFromCPU(stream, divisors);
		_powBuffer!.View.CopyFromCPU(stream, powSpan);
		_orderBuffer!.View.CopyFromCPU(stream, orderSpan);
		_resultBuffer!.View.CopyFromCPU(stream, resultSpan);
		_statusBuffer!.View.CopyFromCPU(stream, statusSpan);

		do
		{
			kernel(stream, length, _divisorCyclesBatchSize, _divisorBuffer.View, _powBuffer.View, _orderBuffer.View, _resultBuffer.View, _statusBuffer.View);

			_statusBuffer.View.CopyToCPU(stream, statusSpan);
			stream.Synchronize();

			pending = 0;
			for (int i = 0; i < length; i++)
			{
				if (statusSpan[i] == ByteZero)
				{
					pending++;
				}
			}
		}
		while (pending > 0);

		_resultBuffer.View.CopyToCPU(stream, resultSpan);
		stream.Synchronize();
		AcceleratorStreamPool.Return(acceleratorIndex, stream);

		resultSpan.CopyTo(destination);
		if (rentedPow is not null)
		{
			ThreadStaticPools.UlongPool.Return(rentedPow, clearArray: false);
			ThreadStaticPools.UlongPool.Return(rentedOrder!, clearArray: false);
			ThreadStaticPools.UlongPool.Return(rentedResult!, clearArray: false);
			ThreadStaticPools.BytePool.Return(rentedStatus!, clearArray: false);
		}

		// GpuPrimeWorkLimiter.Release();
	}

	private static Action<AcceleratorStream, Index1D, int, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<byte, Stride1D.Dense>> LoadKernel(Accelerator accelerator)
	{
		var loaded = accelerator.LoadAutoGroupedStreamKernel<Index1D, int, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<byte, Stride1D.Dense>>(DivisorCycleKernels.GpuAdvanceDivisorCyclesKernel);
		var kernel = KernelUtil.GetKernel(loaded);
		return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, int, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<byte, Stride1D.Dense>>>();
	}

	private sealed class AcceleratorReferenceComparer : IEqualityComparer<Accelerator>
	{
		internal static AcceleratorReferenceComparer Instance { get; } = new();

		public bool Equals(Accelerator? x, Accelerator? y) => ReferenceEquals(x, y);

		public int GetHashCode(Accelerator obj) => RuntimeHelpers.GetHashCode(obj);
	}
}




