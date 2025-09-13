using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;
using static PerfectNumbers.Core.Gpu.GpuContextPool;

namespace PerfectNumbers.Core.Gpu;

public readonly struct ResidueAutomatonArgs
{
    public readonly ulong Q0M10;
    public readonly ulong Step10;
    public readonly ulong Q0M8;
    public readonly ulong Step8;
    public readonly ulong Q0M3;
    public readonly ulong Step3;
    public readonly ulong Q0M5;
    public readonly ulong Step5;

    public ResidueAutomatonArgs(ulong q0m10, ulong step10, ulong q0m8, ulong step8, ulong q0m3, ulong step3, ulong q0m5, ulong step5)
    {
        Q0M10 = q0m10; Step10 = step10; Q0M8 = q0m8; Step8 = step8; Q0M3 = q0m3; Step3 = step3; Q0M5 = q0m5; Step5 = step5;
    }
}

public sealed class KernelContainer
{
	// Serializes first-time initialization of kernels/buffers per accelerator.
	public Action<Index1D, ulong, ulong, ArrayView<GpuUInt128>, ArrayView<ulong>>? Order;
    public Action<Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong,
        ulong, ulong, ulong, ulong, ArrayView<ulong>, ArrayView1D<uint, Stride1D.Dense>>? Incremental;
    public Action<Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong,
        ResidueAutomatonArgs, ArrayView<ulong>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>? Pow2Mod;
	public Action<Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong,
		ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<uint, Stride1D.Dense>>? IncrementalOrder;
    public Action<Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong,
        ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<uint, Stride1D.Dense>>? Pow2ModOrder;

    // Optional device buffer with small divisor cycles (<= 4M). Index = divisor, value = cycle length.
    public MemoryBuffer1D<uint, Stride1D.Dense>? SmallCycles;
    public MemoryBuffer1D<uint, Stride1D.Dense>? SmallPrimes;
    public MemoryBuffer1D<ulong, Stride1D.Dense>? SmallPrimesPow2;

	public static T InitOnce<T>(ref T? slot, Func<T> factory) where T : class
	{
		var current = Volatile.Read(ref slot);
		if (current is not null)
		{
			return current;
		}

		current = Volatile.Read(ref slot);
		if (current is null)
		{
			current = factory();
			Volatile.Write(ref slot, current);
		}
		return current;
	}
}

public ref struct GpuKernelLease(IDisposable limiter, GpuContextLease gpu, KernelContainer kernels)
{
	private bool disposedValue;
	private IDisposable? _limiter = limiter;
	private GpuContextLease? _gpu = gpu;
	private KernelContainer _kernels = kernels;

	public readonly Accelerator Accelerator => _gpu?.Accelerator ?? throw new NullReferenceException("GPU context is already released");

	public readonly Action<Index1D, ulong, ulong, ArrayView<GpuUInt128>, ArrayView<ulong>> OrderKernel
	{
		get
		{
			var accel = Accelerator; // avoid capturing 'this' in lambda
			lock (_gpu!.Value.KernelInitLock)
			{
				return KernelContainer.InitOnce(ref _kernels.Order, () => accel.LoadAutoGroupedStreamKernel<Index1D, ulong, ulong, ArrayView<GpuUInt128>, ArrayView<ulong>>(OrderKernelScan));
			}
		}
	}

    public readonly Action<Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<ulong>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>> Pow2ModKernel
        {
                get
                {
                        var accel = Accelerator;
                        lock (_gpu!.Value.KernelInitLock)
                        {
                                return KernelContainer.InitOnce(ref _kernels.Pow2Mod, () => accel.LoadAutoGroupedStreamKernel<Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<ulong>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>>(Pow2ModKernelScan));
                        }
                }
        }

    public readonly Action<Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ulong, ulong, ulong, ulong, ArrayView<ulong>, ArrayView1D<uint, Stride1D.Dense>> IncrementalKernel
	{
		get
		{
			var accel = Accelerator;
			lock (_gpu!.Value.KernelInitLock)
			{
				return KernelContainer.InitOnce(ref _kernels.Incremental, () => accel.LoadAutoGroupedStreamKernel<Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ulong, ulong, ulong, ulong, ArrayView<ulong>, ArrayView1D<uint, Stride1D.Dense>>(IncrementalKernelScan));
			}
		}
	}

	public readonly Action<Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<uint, Stride1D.Dense>> IncrementalOrderKernel
	{
		get
		{
			var accel = Accelerator;
			lock (_gpu!.Value.KernelInitLock)
			{
				return KernelContainer.InitOnce(ref _kernels.IncrementalOrder, () => accel.LoadAutoGroupedStreamKernel<Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<uint, Stride1D.Dense>>(IncrementalOrderKernelScan));
			}
		}
	}

	public readonly Action<Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<uint, Stride1D.Dense>> Pow2ModOrderKernel
	{
		get
		{
			var accel = Accelerator;
			lock (_gpu!.Value.KernelInitLock)
			{
				return KernelContainer.InitOnce(ref _kernels.Pow2ModOrder, () => accel.LoadAutoGroupedStreamKernel<Index1D, ulong, GpuUInt128, GpuUInt128, byte, ulong, ResidueAutomatonArgs, ArrayView<int>, ArrayView1D<uint, Stride1D.Dense>>(Pow2ModOrderKernelScan));
			}
		}
	}

	// TODO: Plumb the small-cycles device buffer into all kernels that can benefit
	// (some already accept it). Consider a compact type (byte/ushort) for memory footprint.

    private static void IncrementalKernelScan(Index1D index, ulong exponent, GpuUInt128 twoP, GpuUInt128 kStart, byte lastIsSeven, ulong divMul,
        ulong q0m10, ulong q0m8, ulong q0m3, ulong q0m5, ArrayView<ulong> orders,
        ArrayView1D<uint, Stride1D.Dense> smallCycles)
    {
        ulong idx = (ulong)index.X;
		// residue automaton
        ulong step10 = ((exponent % 10UL) << 1) % 10UL;
        ulong r10 = q0m10 + (step10 * idx) % 10UL;
		r10 -= (r10 >= 10UL) ? 10UL : 0UL;
		bool shouldCheck = r10 != 5UL; // skip q ≡ 5 (mod 10); other residues handled by r8/r3/r5
		if (shouldCheck)
		{
            ulong step8 = ((exponent & 7UL) << 1) & 7UL;
            ulong r8 = (q0m8 + ((step8 * idx) & 7UL)) & 7UL;
			if (r8 != 1UL && r8 != 7UL)
			{
				shouldCheck = false;
			}
            else
            {
                ulong step3 = ((exponent % 3UL) << 1) % 3UL;
                ulong r3 = q0m3 + (step3 * (idx % 3UL)); r3 -= (r3 >= 3UL) ? 3UL : 0UL;
                if (r3 == 0UL)
                {
                    shouldCheck = false;
                }
                else
                {
                    ulong step5Loc = ((exponent % 5UL) << 1) % 5UL;
                    ulong r5 = q0m5 + (step5Loc * (idx % 5UL)); if (r5 >= 5UL) r5 -= 5UL;
                    if (r5 == 0UL)
                    {
                        shouldCheck = false;
                    }
                }
            }
		}

		if (!shouldCheck)
		{
			orders[index] = 0UL;
			return;
		}

        GpuUInt128 k = kStart + (GpuUInt128)idx;
        GpuUInt128 q = twoP.Mul64(k) + GpuUInt128.One;
        
        // Small-cycles in-kernel early rejection from device table
        if (q.High == 0UL && q.Low <= (ulong)(smallCycles.Length - 1))
        {
            uint cyc = smallCycles[(int)q.Low];
            if (cyc != 0U && (exponent % cyc) == 0UL)
            {
                orders[index] = exponent;
                return;
            }
        }

        GpuUInt128 phi = q - GpuUInt128.One;
        if (phi.High != 0UL)
        {
            orders[index] = 0UL;
            return;
        }

        ulong phi64 = phi.Low;
        if (GpuUInt128.Pow2Mod(phi64, q) != GpuUInt128.One)
        {
            orders[index] = 0UL;
            return;
        }

        GpuUInt128 halfPow = GpuUInt128.Pow2Mod(phi64 >> 1, q) - GpuUInt128.One;
        if (GpuUInt128.BinaryGcd(halfPow, q) != GpuUInt128.One)
        {
            orders[index] = 0UL;
            return;
        }

        ulong div = phi64.FastDiv64(exponent, divMul);
        GpuUInt128 divPow = GpuUInt128.Pow2Mod(div, q) - GpuUInt128.One;
        if (GpuUInt128.BinaryGcd(divPow, q) != GpuUInt128.One)
        {
            orders[index] = 0UL;
            return;
        }

        orders[index] = 1UL;
    }

    private static void OrderKernelScan(Index1D index, ulong exponent, ulong divMul, ArrayView<GpuUInt128> qs, ArrayView<ulong> orders)
    {
        GpuUInt128 q = qs[index];
        GpuUInt128 phi = q - GpuUInt128.One;
        if (phi.High != 0UL)
        {
            orders[index] = 0UL;
            return;
        }

        ulong phi64 = phi.Low;
        GpuUInt128 pow = GpuUInt128.Pow2Mod(phi64, q);
        if (pow != GpuUInt128.One)
        {
            orders[index] = 0UL;
            return;
        }

        GpuUInt128 halfPow = GpuUInt128.Pow2Mod(phi64 >> 1, q) - GpuUInt128.One;
        if (GpuUInt128.BinaryGcd(halfPow, q) != GpuUInt128.One)
        {
            orders[index] = 0UL;
            return;
        }

        ulong div = phi64.FastDiv64(exponent, divMul);
        GpuUInt128 divPow = GpuUInt128.Pow2Mod(div, q) - GpuUInt128.One;
        if (GpuUInt128.BinaryGcd(divPow, q) != GpuUInt128.One)
        {
            orders[index] = 0UL;
            return;
        }

        orders[index] = exponent;
    }

	private void Dispose(bool disposing)
	{
		if (!disposedValue)
		{
			if (disposing)
			{
				_gpu?.Dispose();
				_gpu = null;

				_limiter?.Dispose();
				_limiter = null;
			}

			// free unmanaged resources (unmanaged objects) and override finalizer
			// set large fields to null
			disposedValue = true;
		}
	}

    private static void Pow2ModKernelScan(Index1D index, ulong exponent, GpuUInt128 twoP, GpuUInt128 kStart, byte lastIsSeven, ulong _,
        ResidueAutomatonArgs ra, ArrayView<ulong> orders,
        ArrayView1D<uint, Stride1D.Dense> smallCycles, ArrayView1D<uint, Stride1D.Dense> smallPrimes,
        ArrayView1D<ulong, Stride1D.Dense> smallPrimesPow2)
    {
        ulong idx = (ulong)index.X;
        ulong r10 = ra.Q0M10 + (ra.Step10 * idx) % 10UL; r10 -= (r10 >= 10UL) ? 10UL : 0UL;
        bool shouldCheck = r10 != 5UL;
        if (shouldCheck)
        {
            ulong r8 = (ra.Q0M8 + ((ra.Step8 * idx) & 7UL)) & 7UL;
            if (r8 != 1UL && r8 != 7UL)
            {
                shouldCheck = false;
            }
            else
            {
                ulong r3 = ra.Q0M3 + (ra.Step3 * (idx % 3UL)); r3 -= (r3 >= 3UL) ? 3UL : 0UL;
                if (r3 == 0UL)
                {
                    shouldCheck = false;
                }
                else
                {
                    ulong r5 = ra.Q0M5 + (ra.Step5 * (idx % 5UL)); if (r5 >= 5UL) r5 -= 5UL;
                    if (r5 == 0UL)
                    {
                        shouldCheck = false;
                    }
                }
            }
        }
        if (!shouldCheck)
        {
            orders[index] = 0UL;
            return;
        }

        GpuUInt128 k = kStart + (GpuUInt128)idx;
        GpuUInt128 q = twoP.Mul64(k) + GpuUInt128.One;
        if (q.High == 0UL && q.Low <= (ulong)(smallCycles.Length - 1))
        {
            uint cyc = smallCycles[(int)q.Low];
            if (cyc != 0U && (exponent % cyc) == 0UL)
            {
                orders[index] = exponent;
                return;
            }
        }
        if (GpuUInt128.Pow2Minus1Mod(exponent, q) != GpuUInt128.Zero)
        {
            orders[index] = 0UL;
            return;
        }

        int primesLen = (int)smallPrimes.Length;
        for (int i = 0; i < primesLen; i++)
        {
            ulong square = smallPrimesPow2[i];
            if (new GpuUInt128(0UL, square) > q)
            {
                break;
            }
            ulong prime = smallPrimes[i];
            if (Mod128By64(q, prime) == 0UL)
            {
                orders[index] = 0UL;
                return;
            }
        }

        orders[index] = exponent;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong Mod128By64(GpuUInt128 value, ulong modulus)
    {
        ulong rem = 0UL;
        ulong part = value.High;
        for (int i = 0; i < 64; i++)
        {
            rem = (rem << 1) | (part >> 63);
            if (rem >= modulus)
            {
                rem -= modulus;
            }
            part <<= 1;
        }

        part = value.Low;
        for (int i = 0; i < 64; i++)
        {
            rem = (rem << 1) | (part >> 63);
            if (rem >= modulus)
            {
                rem -= modulus;
            }
            part <<= 1;
        }

        return rem;
    }

    // Device-friendly small cycle length for 2 mod divisor
    private static ulong CalculateCycleLengthSmall(ulong divisor)
    {
        if ((divisor & (divisor - 1UL)) == 0UL)
            return 1UL;

        ulong order = 1UL, pow = 2UL;
        while (pow != 1UL)
        {
            pow <<= 1;
            if (pow > divisor)
                pow -= divisor;

            order++;
        }

        return order;
    }

    private static void IncrementalOrderKernelScan(Index1D index, ulong exponent, GpuUInt128 twoP, GpuUInt128 kStart, byte lastIsSeven, ulong divMul,
		ResidueAutomatonArgs ra, ArrayView<int> found, ArrayView1D<uint, Stride1D.Dense> smallCycles)
    {
        ulong idx = (ulong)index.X;
		ulong r10 = ra.Q0M10 + (ra.Step10 * idx) % 10UL; r10 -= (r10 >= 10UL) ? 10UL : 0UL;
		bool shouldCheck = r10 != 5UL;
		if (shouldCheck)
		{
			ulong r8 = (ra.Q0M8 + ((ra.Step8 * idx) & 7UL)) & 7UL;
			if (r8 != 1UL && r8 != 7UL)
			{
				shouldCheck = false;
			}
			else
			{
				ulong r3 = ra.Q0M3 + (ra.Step3 * (idx % 3UL)); r3 -= (r3 >= 3UL) ? 3UL : 0UL;
				if (r3 == 0UL)
				{
					shouldCheck = false;
				}
				else
				{
					ulong r5 = ra.Q0M5 + (ra.Step5 * (idx % 5UL)); if (r5 >= 5UL) r5 -= 5UL;
					if (r5 == 0UL)
					{
						shouldCheck = false;
					}
				}
			}
		}
		if (!shouldCheck)
		{
			return;
		}

        GpuUInt128 k = kStart + (GpuUInt128)idx;
        GpuUInt128 q = twoP.Mul64(k) + GpuUInt128.One;

        // Small-cycles in-kernel early rejection from device table
        if (q.High == 0UL && q.Low <= (ulong)(smallCycles.Length - 1))
        {
            uint cyc = smallCycles[(int)q.Low];
            if (cyc != 0U && (exponent % (ulong)cyc) == 0UL)
            {
                return;
            }
        }
        GpuUInt128 phi = q - GpuUInt128.One;
        if (phi.High != 0UL)
        {
            return;
        }

        ulong phi64 = phi.Low;
        if (GpuUInt128.Pow2Mod(phi64, q) != GpuUInt128.One)
        {
            return;
        }

        GpuUInt128 halfPow = GpuUInt128.Pow2Mod(phi64 >> 1, q) - GpuUInt128.One;
        if (GpuUInt128.BinaryGcd(halfPow, q) != GpuUInt128.One)
        {
            return;
        }

        ulong div = phi64.FastDiv64(exponent, divMul);
        GpuUInt128 divPow = GpuUInt128.Pow2Mod(div, q) - GpuUInt128.One;
        if (GpuUInt128.BinaryGcd(divPow, q) != GpuUInt128.One)
        {
            return;
        }

        Atomic.Or(ref found[0], 1);
    }

    private static void Pow2ModOrderKernelScan(Index1D index, ulong exponent, GpuUInt128 twoP, GpuUInt128 kStart, byte lastIsSeven, ulong _,
		ResidueAutomatonArgs ra, ArrayView<int> found, ArrayView1D<uint, Stride1D.Dense> smallCycles)
    {
        ulong idx = (ulong)index.X;
		ulong r10 = ra.Q0M10 + (ra.Step10 * idx) % 10UL; r10 -= (r10 >= 10UL) ? 10UL : 0UL;
		bool shouldCheck = r10 != 5UL;
		if (shouldCheck)
		{
			ulong r8 = (ra.Q0M8 + ((ra.Step8 * idx) & 7UL)) & 7UL;
			if (r8 != 1UL && r8 != 7UL)
			{
				shouldCheck = false;
			}
			else
			{
				ulong r3 = ra.Q0M3 + (ra.Step3 * (idx % 3UL)); r3 -= (r3 >= 3UL) ? 3UL : 0UL;
				if (r3 == 0UL)
				{
					shouldCheck = false;
				}
				else
				{
					ulong r5 = ra.Q0M5 + (ra.Step5 * (idx % 5UL)); if (r5 >= 5UL) r5 -= 5UL;
					if (r5 == 0UL)
					{
						shouldCheck = false;
					}
				}
			}
		}
		if (!shouldCheck)
		{
			return;
		}

        GpuUInt128 k = kStart + (GpuUInt128)idx;
        GpuUInt128 q = twoP.Mul64(k) + GpuUInt128.One;
        // Small-cycles in-kernel early rejection from device table
        if (q.High == 0UL && q.Low <= (ulong)(smallCycles.Length - 1))
        {
            uint cyc = smallCycles[(int)q.Low];
            if (cyc != 0U && (exponent % (ulong)cyc) == 0UL)
            {
                return;
            }
        }
        if (GpuUInt128.Pow2Mod(exponent, q) != GpuUInt128.One)
        {
            return;
        }

        Atomic.Or(ref found[0], 1);
    }

	public void Dispose()
	{
		// Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
		Dispose(disposing: true);
		// Enabled this when the struct is converted to a class
		// GC.SuppressFinalize(this);
	}
}

public class GpuKernelPool
{
    private static readonly ConcurrentDictionary<Accelerator, KernelContainer> KernelCache = new();

    private static KernelContainer GetKernels(Accelerator accelerator)
    {
        return KernelCache.GetOrAdd(accelerator, _ => new KernelContainer());
    }

    // Ensures the small cycles table is uploaded to the device for the given accelerator.
    // Returns the ArrayView to pass into kernels (when kernels are extended to accept it).
    public static ArrayView1D<uint, Stride1D.Dense> EnsureSmallCyclesOnDevice(Accelerator accelerator)
    {
        var kernels = GetKernels(accelerator);
        if (kernels.SmallCycles is { } buffer)
        {
            return buffer.View;
        }

        // Ensure single upload per accelerator even if multiple threads race here.
        lock (kernels)
        {
            if (kernels.SmallCycles is { } existing)
            {
                return existing.View;
            }

            var host = MersenneDivisorCycles.Shared.ExportSmallCyclesSnapshot();
            var device = accelerator.Allocate1D<uint>(host.Length);
            device.View.CopyFromCPU(host);
            kernels.SmallCycles = device;
            return device.View;
        }
    }

    public static (ArrayView1D<uint, Stride1D.Dense> primes, ArrayView1D<ulong, Stride1D.Dense> primesPow2) EnsureSmallPrimesOnDevice(Accelerator accelerator)
    {
        var kernels = GetKernels(accelerator);
        if (kernels.SmallPrimes is { } p && kernels.SmallPrimesPow2 is { } p2)
        {
            return (p.View, p2.View);
        }

        lock (kernels)
        {
            if (kernels.SmallPrimes is { } existing && kernels.SmallPrimesPow2 is { } existingPow2)
            {
                return (existing.View, existingPow2.View);
            }

            var hostPrimes = PrimesGenerator.SmallPrimes;
            var hostPow2 = PrimesGenerator.SmallPrimesPow2;
            var devicePrimes = accelerator.Allocate1D<uint>(hostPrimes.Length);
            devicePrimes.View.CopyFromCPU(hostPrimes);
            var devicePow2 = accelerator.Allocate1D<ulong>(hostPow2.Length);
            devicePow2.View.CopyFromCPU(hostPow2);
            kernels.SmallPrimes = devicePrimes;
            kernels.SmallPrimesPow2 = devicePow2;
            return (devicePrimes.View, devicePow2.View);
        }
    }

	public static GpuKernelLease GetKernel(bool useGpuOrder)
	{
		var limiter = GpuPrimeWorkLimiter.Acquire();
		var gpu = RentPreferred(preferCpu: !useGpuOrder);
		var accelerator = gpu.Accelerator;
		var kernels = GetKernels(accelerator);
		return new(limiter, gpu, kernels);
	}

	/// <summary>
	/// Runs a GPU action with an acquired accelerator and stream.
	/// </summary>
	/// <param name="action">Action to run with (Accelerator, Stream).</param>
    public static void Run(Action<Accelerator, AcceleratorStream> action)
    {
        var lease = GetKernel(useGpuOrder: true);
        var accelerator = lease.Accelerator;
        var stream = accelerator.CreateStream();
        action(accelerator, stream);
        stream.Dispose();
        lease.Dispose();

    }
}
