using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

public class MersenneNumberResidueGpuTester(bool useGpuOrder)
{
    private readonly bool _useGpuOrder = useGpuOrder;
    private readonly ConcurrentBag<GpuContextPool.GpuContextLease> _acceleratorPool = new();
    private readonly ConcurrentDictionary<Accelerator, Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>> _pow2ModKernelCache = new(AcceleratorReferenceComparer.Instance);
    private readonly ConcurrentDictionary<Accelerator, ConcurrentBag<ResidueResources>> _resourcePools = new(AcceleratorReferenceComparer.Instance);

    // TODO: Integrate MersenneDivisorCycles.Shared to consult cycle lengths for small q (<= 4M) and fast-reject
    // candidates without launching heavy kernels. Consider a device-visible constant buffer per-accelerator
    // via GpuKernelPool to avoid host round-trips.
    // TODO: When cycles are missing for larger q values, compute only the single required cycle on the device
    // requested by the caller, skip queuing additional block generation, and keep the snapshot cache untouched.

    // GPU residue variant: check 2^p % q == 1 for q = 2*p*k + 1.
    public void Scan(ulong exponent, UInt128 twoP, bool lastIsSeven, UInt128 maxK, ref bool isPrime)
    {
        var limiter = GpuPrimeWorkLimiter.Acquire();
        var gpuLease = RentAccelerator();
        var accelerator = gpuLease.Accelerator;

        Monitor.Enter(gpuLease.ExecutionLock);

        var stream = accelerator.CreateStream();
        var kernel = GetPow2ModKernel(accelerator);
        var resources = RentResources(accelerator, GpuConstants.ScanBatchSize);

        var orderBuffer = resources.OrderBuffer;
        ulong[] orderArray = resources.OrderArray;
        int batchSize = resources.Capacity;

        // Ensure device has small cycles and primes tables for in-kernel lookup
        var smallCyclesView = GpuKernelPool.EnsureSmallCyclesOnDevice(accelerator);
        ResiduePrimeViews primeViews = GpuKernelPool.EnsureSmallPrimesOnDevice(accelerator);

        UInt128 kStart = 1UL;
        UInt128 limit = maxK + UInt128.One;
        byte last = lastIsSeven ? (byte)1 : (byte)0;
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

            stream.Synchronize();
            orderBuffer.View.CopyToCPU(ref ordersRef, currentSize);
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

        ReturnResources(accelerator, resources);
        stream.Dispose();
        Monitor.Exit(gpuLease.ExecutionLock);
        ReturnAccelerator(gpuLease);
        limiter.Dispose();
    }

    private Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>> GetPow2ModKernel(Accelerator accelerator) =>
        _pow2ModKernelCache.GetOrAdd(
            accelerator,
            static acc =>
            {
                var loaded = acc.LoadAutoGroupedStreamKernel<Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>(Pow2ModKernels.Pow2ModKernelScan);
                return (Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>)Delegate.CreateDelegate(
                    typeof(Action<AcceleratorStream, Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<ulong>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>),
                    loaded.Target,
                    loaded.Method);
            });

    private GpuContextPool.GpuContextLease RentAccelerator()
    {
        if (_acceleratorPool.TryTake(out var lease))
        {
            return lease;
        }

        return GpuContextPool.RentPreferred(preferCpu: !_useGpuOrder);
    }

    private void ReturnAccelerator(GpuContextPool.GpuContextLease lease) => _acceleratorPool.Add(lease);

    private ResidueResources RentResources(Accelerator accelerator, int capacity)
    {
        var bag = _resourcePools.GetOrAdd(accelerator, static _ => new ConcurrentBag<ResidueResources>());
        while (bag.TryTake(out var resources))
        {
            if (resources.Capacity >= capacity)
			{
				// The caller always overwrites the required elements. We don't need to worry about clearing the output buffer.
                // resources.OrderBuffer.MemSetToZero();
                return resources;
            }

            resources.Dispose();
        }

        var created = new ResidueResources(accelerator, capacity);
		// The caller always overwrites the required elements. We don't need to worry about clearing the output buffer.
        // created.OrderBuffer.MemSetToZero();
        return created;
    }

    private void ReturnResources(Accelerator accelerator, ResidueResources resources) =>
        _resourcePools.GetOrAdd(accelerator, static _ => new ConcurrentBag<ResidueResources>()).Add(resources);

    private sealed class ResidueResources : IDisposable
    {
        internal ResidueResources(Accelerator accelerator, int capacity)
        {
            Capacity = Math.Max(1, capacity);
            OrderBuffer = accelerator.Allocate1D<ulong>(Capacity);
            OrderArray = ArrayPool<ulong>.Shared.Rent(Capacity);
        }

        internal MemoryBuffer1D<ulong, Stride1D.Dense> OrderBuffer { get; }

        internal ulong[] OrderArray { get; }

        internal int Capacity { get; }

        public void Dispose()
        {
            OrderBuffer.Dispose();
            ArrayPool<ulong>.Shared.Return(OrderArray, clearArray: false);
        }
    }

    private sealed class AcceleratorReferenceComparer : IEqualityComparer<Accelerator>
    {
        public static AcceleratorReferenceComparer Instance { get; } = new();

        public bool Equals(Accelerator? x, Accelerator? y) => ReferenceEquals(x, y);

        public int GetHashCode(Accelerator obj) => RuntimeHelpers.GetHashCode(obj);
    }


}
