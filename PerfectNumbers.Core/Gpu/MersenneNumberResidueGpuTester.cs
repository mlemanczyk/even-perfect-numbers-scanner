using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu.Accelerators;

namespace PerfectNumbers.Core.Gpu;

public class MersenneNumberResidueGpuTester(bool useGpuOrder)
{
	private readonly bool _useGpuOrder = useGpuOrder;

	[ThreadStatic]
	private static Dictionary<Accelerator, Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>>? _pow2ModKernel;

	[ThreadStatic]
	private static Queue<ResidueResources>? _resourcePools;

	// TODO: Integrate MersenneDivisorCycles.Shared to consult cycle lengths for small q (<= 4M) and fast-reject
	// candidates without launching heavy kernels. Consider a device-visible constant buffer per-accelerator
	// via GpuKernelPool to avoid host round-trips.
	// TODO: When cycles are missing for larger q values, compute only the single required cycle on the device
	// requested by the caller, skip queuing additional block generation, and keep the snapshot cache untouched.

	// GPU residue variant: check 2^p % q == 1 for q = 2*p*k + 1.
	public void Scan(PrimeOrderCalculatorAccelerator gpu, ulong exponent, UInt128 twoP, LastDigit lastDigit, UInt128 maxK, ref bool isPrime)
	{
		// GpuPrimeWorkLimiter.Acquire();
		var acceleratorIndex = gpu.AcceleratorIndex;
		var accelerator = gpu.Accelerator;

		// Monitor.Enter(gpuLease.ExecutionLock);

		var stream = AcceleratorStreamPool.Rent(acceleratorIndex);
		var kernel = GetPow2ModKernel(accelerator);
		var resources = RentResources(accelerator, GpuConstants.ScanBatchSize);
		var gpuKernels = GpuKernelPool.GetOrAddKernels(acceleratorIndex, stream, KernelType.SmallCycles | KernelType.SmallPrimes);

		var orderBuffer = resources.OrderBuffer;
		ulong[] orderArray = resources.OrderArray;
		int batchSize = resources.Capacity;

		// Ensure device has small cycles and primes tables for in-kernel lookup
		// TODO: Pass-in the stream down the path
		var smallCyclesView = GpuKernelPool.GetSmallCyclesOnDevice(gpuKernels);
		ResiduePrimeViews primeViews = GpuKernelPool.GetSmallPrimesOnDevice(gpuKernels);

		UInt128 kStart = 1UL;
		UInt128 limit = maxK + UInt128.One;
		byte last = lastDigit == LastDigit.Seven ? (byte)1 : (byte)0;
		twoP.Mod10_8_5_3(out ulong step10, out ulong step8, out ulong step5, out ulong step3);
		step10 = step10.Mod10();

		GpuUInt128 twoPGpu = (GpuUInt128)twoP;
		UInt128 batchSize128 = (UInt128)batchSize;
		UInt128 fullBatchStep = twoP * batchSize128;
		UInt128 q = twoP * kStart + UInt128.One;

		while (kStart < limit && Volatile.Read(ref isPrime))
		{
			UInt128 remaining = limit - kStart;
			int currentSize = remaining > batchSize128 ? batchSize : (int)remaining;
			Span<ulong> orders = orderArray.AsSpan(0, currentSize);
			ref ulong ordersRef = ref MemoryMarshal.GetReference(orders);

			q.Mod10_8_5_3(out ulong q0m10, out ulong q0m8, out ulong q0m5, out ulong q0m3);
			var kernelArgs = new ResidueAutomatonArgs(q0m10, step10, q0m8, step8, q0m3, step3, q0m5, step5);

			kernel(stream, currentSize, exponent, twoPGpu, (GpuUInt128)kStart, last, 0UL,
				kernelArgs, orderBuffer.View, smallCyclesView, primeViews.LastOne, primeViews.LastSeven, primeViews.LastOnePow2, primeViews.LastSevenPow2);

			orderBuffer.View.CopyToCPU(stream, ref ordersRef, currentSize);
			stream.Synchronize();
			if (!Volatile.Read(ref isPrime))
			{
				break;
			}

			bool compositeFound = false;
			for (int i = 0; i < currentSize; i++)
			{
				if (orders[i] != 0UL)
				{
					Volatile.Write(ref isPrime, false);
					compositeFound = true;
					break;
				}
			}

			if (compositeFound)
			{
				break;
			}

			UInt128 processed = (UInt128)currentSize;
			kStart += processed;
			if (kStart < limit)
			{
				if (processed == batchSize128)
				{
					q += fullBatchStep;
				}
				else
				{
					q += twoP * processed; // TODO: Swap this multiplication for the shared cycle stepping helper once residue cycles are mandatory so we can reuse the cached remainder ladder instead of performing UInt128 multiply operations.
				}
			}
		}

		ReturnResources(resources);
		AcceleratorStreamPool.Return(acceleratorIndex, stream);
		// Monitor.Exit(gpuLease.ExecutionLock);
		gpuKernels.Dispose();
		// GpuPrimeWorkLimiter.Release();
	}

	private static Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>> GetPow2ModKernel(Accelerator accelerator)
	{

		var pool = _pow2ModKernel ?? [];
		if (pool.TryGetValue(accelerator, out var cached))
		{
			return cached;
		}

		cached = CreatePow2ModKernel(accelerator);
		pool[accelerator] = cached;
		return cached;
	}

	private static Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>> CreatePow2ModKernel(Accelerator accelerator)
	{
		var loaded = accelerator.LoadAutoGroupedStreamKernel<Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>(Pow2ModKernels.Pow2ModKernelScan);

		var kernel = (Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>)Delegate.CreateDelegate(
			typeof(Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>),
			loaded.Target,
			loaded.Method);
		return kernel;
	}


	private static ResidueResources RentResources(Accelerator accelerator, int capacity)
	{
		var resourcePools = _resourcePools ??= new();
		if (resourcePools.TryDequeue(out var bag))
		{
			bag.EnsureCapacity(accelerator, capacity);
			return bag;
		}

		return new ResidueResources(accelerator, capacity);
	}

	private static void ReturnResources(ResidueResources resources)
	{
		var resourcePools = _resourcePools ??= new();
		resourcePools.Enqueue(resources);
	}

	private sealed class ResidueResources
	{
		internal ResidueResources(Accelerator accelerator, int capacity)
		{
			Capacity = Math.Max(1, capacity);
			// lock (accelerator)
			{
				OrderBuffer = accelerator.Allocate1D<ulong>(Capacity);
			}

			OrderArray = ThreadStaticPools.UlongPool.Rent(Capacity);
		}

		internal void EnsureCapacity(Accelerator accelerator, int capacity)
		{
			if (OrderBuffer.Length >= capacity)
			{
				return;
			}

			OrderBuffer.Dispose();
			ThreadStaticPools.UlongPool.Return(OrderArray);

			// lock (accelerator)
			{
				OrderBuffer = accelerator.Allocate1D<ulong>(capacity);
			}

			OrderArray = ThreadStaticPools.UlongPool.Rent(capacity);
			Capacity = capacity;
		}

		internal MemoryBuffer1D<ulong, Stride1D.Dense> OrderBuffer { get; private set; }

		internal ulong[] OrderArray { get; private set; }

		internal int Capacity { get; private set; }

		public void Dispose()
		{
			OrderBuffer.Dispose();
			ThreadStaticPools.UlongPool.Return(OrderArray, clearArray: false);
		}
	}

	private sealed class AcceleratorReferenceComparer : IEqualityComparer<Accelerator>
	{
		public static AcceleratorReferenceComparer Instance { get; } = new();

		public bool Equals(Accelerator? x, Accelerator? y) => ReferenceEquals(x, y);

		public int GetHashCode(Accelerator obj) => RuntimeHelpers.GetHashCode(obj);
	}


}