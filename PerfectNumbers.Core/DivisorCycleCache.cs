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
	private volatile bool _useGpuGeneration = true;
	private readonly int _divisorCyclesBatchSize;
	private static int _sharedDivisorCyclesBatchSize = GpuConstants.GpuCycleStepsPerInvocation;

	private const int CycleCacheTrackingLimit = 500_000;
	private static readonly HashSet<ulong> CycleCacheTrackedDivisors = new();
	private static readonly object CycleCacheTrackingLock = new();
	private static ulong CycleCacheHitCount;
	private static bool CycleCacheTrackingDisabled;

	public static void SetDivisorCyclesBatchSize(int divisorCyclesBatchSize)
	{
		_sharedDivisorCyclesBatchSize = Math.Max(1, divisorCyclesBatchSize);
	}

	public static DivisorCycleCache Shared { get; } = new DivisorCycleCache(_sharedDivisorCyclesBatchSize);

	public int PreferredBatchSize => _divisorCyclesBatchSize;

	// private static readonly Accelerator[] _accelerators;

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

	public void ConfigureGeneratorDevice(bool useGpu)
	{
		_useGpuGeneration = useGpu;
	}

	public void RefreshSnapshot()
	{
		_snapshot = MersenneDivisorCycles.Shared.ExportSmallCyclesSnapshot();
	}

	public ulong GetCycleLength(ulong divisor, bool skipPrimeOrderHeuristic = false)
	{
		if (!skipPrimeOrderHeuristic)
		{
			Span<ulong> result = stackalloc ulong[1];
			ReadOnlySpan<ulong> singleDivisor = stackalloc ulong[1] { divisor };
			GetCycleLengths(singleDivisor, result);
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
		if (divisor < (ulong)snapshot.Length)
		{
			ulong cached = snapshot[divisor];
			if (cached == 0UL)
			{
				throw new InvalidDataException($"Divisor cycle is missing for {divisor}");
			}

			return cached;
		}

		MontgomeryDivisorData divisorData = MontgomeryDivisorData.FromModulus(divisor);
		return MersenneDivisorCycles.CalculateCycleLength(divisor, divisorData, skipPrimeOrderHeuristic: true);
	}

	public void GetCycleLengths(ReadOnlySpan<ulong> divisors, Span<ulong> cycles)
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
			if (divisor <= 1UL)
			{
				throw new InvalidDataException("Divisor must be > 1");
			}

			if (divisor < (ulong)snapshot.Length)
			{
				ulong cached = snapshot[divisor];
				if (cached == 0UL)
				{
					throw new InvalidDataException($"Divisor cycle is missing for {divisor}");
				}

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
			if (_useGpuGeneration)
			{
				ComputeCyclesGpu(divisors, cycles, missingIndices);
			}
			else
			{
				ComputeCyclesCpu(divisors, cycles, missingIndices);
			}
		}

		if (rentedMissing is not null)
		{
			ThreadStaticPools.IntPool.Return(rentedMissing, clearArray: false);
		}
	}

	private static void ObserveCycleCacheDivisor(ulong divisor)
	{
		bool logHit = false;
		ulong hitNumber = 0UL;
		bool trackingDisabledNow = false;

		lock (CycleCacheTrackingLock)
		{
			if (CycleCacheTrackingDisabled)
			{
				return;
			}

			if (!CycleCacheTrackedDivisors.Add(divisor))
			{
				CycleCacheHitCount++;
				hitNumber = CycleCacheHitCount;
				logHit = true;
			}
			else if (CycleCacheTrackedDivisors.Count >= CycleCacheTrackingLimit)
			{
				CycleCacheTrackingDisabled = true;
				CycleCacheTrackedDivisors.Clear();
				trackingDisabledNow = true;
			}
		}

		if (logHit)
		{
			Console.WriteLine($"Cycle cache hit for divisor {divisor} ({hitNumber})");
		}
		else if (trackingDisabledNow)
		{
			Console.WriteLine("Cycle cache hit tracking disabled after exceeding the tracking limit.");
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
			cycles[targetIndex] = MersenneDivisorCycles.CalculateCycleLength(divisor, divisorData);
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
	private static MemoryBuffer1D<ulong, Stride1D.Dense>? divisorBuffer;

	[ThreadStatic]
	private static MemoryBuffer1D<ulong, Stride1D.Dense>? powBuffer;

	[ThreadStatic]
	private static MemoryBuffer1D<ulong, Stride1D.Dense>? orderBuffer;

	[ThreadStatic]
	private static MemoryBuffer1D<ulong, Stride1D.Dense>? resultBuffer;

	[ThreadStatic]
	private static MemoryBuffer1D<byte, Stride1D.Dense>? statusBuffer;

	private void EnsureCapacity(int requiredCapacity)
	{
		if (requiredCapacity <= (divisorBuffer?.Length ?? 0))
		{
			return;
		}

		if (divisorBuffer != null)
		{
			divisorBuffer.Dispose();
			powBuffer!.Dispose();
			orderBuffer!.Dispose();
			resultBuffer!.Dispose();
			statusBuffer!.Dispose();
		}

		Accelerator accelerator = _accelerator;
		lock(accelerator)
		{
			divisorBuffer = accelerator.Allocate1D<ulong>(requiredCapacity);
			powBuffer = accelerator.Allocate1D<ulong>(requiredCapacity);
			orderBuffer = accelerator.Allocate1D<ulong>(requiredCapacity);
			resultBuffer = accelerator.Allocate1D<ulong>(requiredCapacity);
			statusBuffer = accelerator.Allocate1D<byte>(requiredCapacity);
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

		divisorBuffer!.View.CopyFromCPU(stream, divisors);
		powBuffer!.View.CopyFromCPU(stream, powSpan);
		orderBuffer!.View.CopyFromCPU(stream, orderSpan);
		resultBuffer!.View.CopyFromCPU(stream, resultSpan);
		statusBuffer!.View.CopyFromCPU(stream, statusSpan);

		do
		{
			kernel(stream, length, _divisorCyclesBatchSize, divisorBuffer.View, powBuffer.View, orderBuffer.View, resultBuffer.View, statusBuffer.View);

			statusBuffer.View.CopyToCPU(stream, statusSpan);
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

		resultBuffer.View.CopyToCPU(stream, resultSpan);
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




