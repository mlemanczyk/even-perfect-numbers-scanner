using System.Buffers;
using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

public sealed class MersenneNumberDivisorByDivisorAccelerator
{
	internal MersenneNumberDivisorByDivisorAccelerator(Accelerator accelerator, int capacity)
	{
		ArrayPool<ulong> ulongPool = ThreadStaticPools.UlongPool;
		ArrayPool<int> intPool = ThreadStaticPools.IntPool;
		ArrayPool<GpuDivisorPartialData> partialPool = ThreadStaticPools.GpuDivisorPartialDataPool;

		Divisors = ulongPool.Rent(capacity);
		Exponents = ulongPool.Rent(capacity);
		DivisorData = partialPool.Rent(capacity);
		Offsets = intPool.Rent(capacity);
		Counts = intPool.Rent(capacity);
		Cycles = ulongPool.Rent(capacity);

		// lock (accelerator)
		{
			DivisorDataBuffer = accelerator.Allocate1D<GpuDivisorPartialData>(capacity);
			OffsetBuffer = accelerator.Allocate1D<int>(capacity);
			CountBuffer = accelerator.Allocate1D<int>(capacity);
			CycleBuffer = accelerator.Allocate1D<ulong>(capacity);
			ExponentBuffer = accelerator.Allocate1D<ulong>(capacity);
			HitsBuffer = accelerator.Allocate1D<byte>(capacity);
			HitIndexBuffer = accelerator.Allocate1D<int>(1);

			CheckDivisorKernel = KernelUtil.GetKernel(accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<GpuDivisorPartialData>, ArrayView<int>, ArrayView<int>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>, ArrayView<int>>(DivisorByDivisorKernels.CheckKernel));
		}
	}

	internal MemoryBuffer1D<GpuDivisorPartialData, Stride1D.Dense> DivisorDataBuffer;
	internal MemoryBuffer1D<int, Stride1D.Dense> OffsetBuffer;
	internal MemoryBuffer1D<int, Stride1D.Dense> CountBuffer;
	internal MemoryBuffer1D<ulong, Stride1D.Dense> CycleBuffer;
	internal MemoryBuffer1D<ulong, Stride1D.Dense> ExponentBuffer;
	internal MemoryBuffer1D<byte, Stride1D.Dense> HitsBuffer;
	internal MemoryBuffer1D<int, Stride1D.Dense> HitIndexBuffer;

	internal ulong[] Divisors;
	internal ulong[] Exponents;
	internal GpuDivisorPartialData[] DivisorData;
	internal int[] Offsets;
	internal int[] Counts;
	internal ulong[] Cycles;
	internal readonly Kernel CheckDivisorKernel;

	public void EnsureCapacity(Accelerator accelerator, int requiredCapacity)
	{
		if (DivisorDataBuffer.Length < requiredCapacity)
		{
			ReallocateBuffers(accelerator, requiredCapacity);
		}
	}

	private void ReallocateBuffers(Accelerator accelerator, int capacity)
	{
		DivisorDataBuffer.Dispose();
		OffsetBuffer.Dispose();
		CountBuffer.Dispose();
		CycleBuffer.Dispose();
		ExponentBuffer.Dispose();
		HitsBuffer.Dispose();
		HitIndexBuffer.Dispose();

		// lock (accelerator)
		{
			DivisorDataBuffer = accelerator.Allocate1D<GpuDivisorPartialData>(capacity);
			OffsetBuffer = accelerator.Allocate1D<int>(capacity);
			CountBuffer = accelerator.Allocate1D<int>(capacity);
			CycleBuffer = accelerator.Allocate1D<ulong>(capacity);
			ExponentBuffer = accelerator.Allocate1D<ulong>(capacity);
			HitsBuffer = accelerator.Allocate1D<byte>(capacity);
			HitIndexBuffer = accelerator.Allocate1D<int>(1);
		}

		ArrayPool<ulong> ulongPool = ThreadStaticPools.UlongPool;
		ArrayPool<int> intPool = ThreadStaticPools.IntPool;
		ArrayPool<GpuDivisorPartialData> partialPool = ThreadStaticPools.GpuDivisorPartialDataPool;
		ulongPool.Return(Divisors, clearArray: false);
		ulongPool.Return(Exponents, clearArray: false);
		partialPool.Return(DivisorData, clearArray: false);
		intPool.Return(Offsets, clearArray: false);
		intPool.Return(Counts, clearArray: false);
		ulongPool.Return(Cycles, clearArray: false);

		Divisors = ulongPool.Rent(capacity);
		Exponents = ulongPool.Rent(capacity);
		DivisorData = partialPool.Rent(capacity);
		Offsets = intPool.Rent(capacity);
		Counts = intPool.Rent(capacity);
		Cycles = ulongPool.Rent(capacity);
	}

	public void Dispose()
	{
		DivisorDataBuffer.Dispose();
		OffsetBuffer.Dispose();
		CountBuffer.Dispose();
		CycleBuffer.Dispose();
		ExponentBuffer.Dispose();
		HitsBuffer.Dispose();
		HitIndexBuffer.Dispose();

		ArrayPool<ulong> ulongPool = ThreadStaticPools.UlongPool;
		ArrayPool<int> intPool = ThreadStaticPools.IntPool;
		ArrayPool<GpuDivisorPartialData> partialPool = ThreadStaticPools.GpuDivisorPartialDataPool;
		ulongPool.Return(Divisors, clearArray: false);
		ulongPool.Return(Exponents, clearArray: false);
		partialPool.Return(DivisorData, clearArray: false);
		intPool.Return(Offsets, clearArray: false);
		intPool.Return(Counts, clearArray: false);
		ulongPool.Return(Cycles, clearArray: false);
	}
}
