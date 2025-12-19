using ILGPU;
using ILGPU.Runtime;
using PerfectNumbers.Core.Gpu.Accelerators;

namespace PerfectNumbers.Core.Gpu;

/// <summary>
/// Provides GPU-accelerated divisibility checks for Mersenne numbers with prime exponents p >= 31.
/// </summary>
public sealed class MersenneNumberDivisorGpuTester
{
	[ThreadStatic]
	private static Dictionary<Accelerator, Queue<MemoryBuffer1D<byte, Stride1D.Dense>>>? _resultBuffers;

	private static Queue<MemoryBuffer1D<byte, Stride1D.Dense>> GetQueue(Accelerator accelerator)
	{
		var pool = _resultBuffers ??= [];
		if (!pool.TryGetValue(accelerator, out var bufferQueue))
		{
			bufferQueue = [];
			pool[accelerator] = bufferQueue;
		}

		return bufferQueue;
	}

	private static Action<AcceleratorStream, Index1D, ulong, ReadOnlyGpuUInt128, ArrayView<byte>> GetDivisorKernel(Accelerator accelerator)
	{
		var loaded = accelerator.LoadAutoGroupedStreamKernel<Index1D, ulong, ReadOnlyGpuUInt128, ArrayView<byte>>(DivisorKernels.Kernel);

		var kernel = KernelUtil.GetKernel(loaded);

		return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ulong, ReadOnlyGpuUInt128, ArrayView<byte>>>();
	}

	public static void BuildDivisorCandidates()
	{
		ulong[] snapshot = MersenneDivisorCyclesGpu.Shared.ExportSmallCyclesSnapshot();
		(ulong divisor, uint cycle)[] list = new (ulong divisor, uint cycle)[snapshot.Length / 2];
		ulong cycle;
		int count = 0, i, snapshotLength = snapshot.Length;
		for (i = 3; i < snapshotLength; i += 2)
		{
			cycle = snapshot[i];
			if (cycle == 0U)
			{
				continue;
			}

			list[count++] = ((ulong)i, (uint)cycle);
		}

		_divisorCandidates = count == 0 ? [] : list[..count];
	}

	private static readonly Accelerator[] _accelerators = AcceleratorPool.Shared.Accelerators;

	public bool IsDivisible(ulong exponent, in ReadOnlyGpuUInt128 divisor)
	{
		// var accelerator = SharedGpuContext.CreateAccelerator();
		var acceleratorIndex = AcceleratorPool.Shared.Rent();
		var accelerator = _accelerators[acceleratorIndex];
		var stream = accelerator.CreateStream();
		Queue<MemoryBuffer1D<byte, Stride1D.Dense>> resultBuffers = GetQueue(accelerator);
		if (!resultBuffers.TryDequeue(out var resultBuffer))
		{
			// lock (accelerator)
			{
				resultBuffer = accelerator.Allocate1D<byte>(1);
			}
		}

		// There is no point in clearing this buffer. We always override item [0] and never use it beyond item [0]
		// resultBuffer.MemSetToZero(stream);
		var divisorKernel = GetDivisorKernel(accelerator);
		divisorKernel(stream, 1, exponent, divisor, resultBuffer.View);
		byte result = 0;
		resultBuffer.View.CopyToCPU(stream, ref result, 1);
		stream.Synchronize();
		bool divisible = result != 0;
		resultBuffers.Enqueue(resultBuffer);
		stream.Dispose();
		// AcceleratorPool.Shared.Return(accelerator);
		// accelerator.Dispose();
		return divisible;
	}

	private static (ulong divisor, uint cycle)[]? _divisorCandidates = [];

	public bool IsPrime(PrimeOrderCalculatorAccelerator gpu, ulong p, UInt128 d, ulong divisorCyclesSearchLimit, out bool divisorsExhausted)
	{
		ReadOnlyGpuUInt128 readOnlyDivisor;

		if (d != UInt128.Zero)
		{
			readOnlyDivisor = new ReadOnlyGpuUInt128(d);
			if (IsDivisible(p, in readOnlyDivisor))
			{
				divisorsExhausted = true;
				return false;
			}

			divisorsExhausted = false;
			return true;
		}

		if (_divisorCandidates is { Length: > 0 } candidates)
		{
			int candidateCount = candidates.Length;
			for (int index = 0; index < candidateCount; index++)
			{
				(ulong candidateDivisor, uint cycle) = candidates[index];
				if (p % cycle != 0UL)
				{
					continue;
				}

				readOnlyDivisor = new ReadOnlyGpuUInt128(candidateDivisor); // Reusing readOnlyDivisor for candidate divisors.
				if (IsDivisible(p, in readOnlyDivisor))
				{
					divisorsExhausted = true;
					return false;
				}
			}
		}

		UInt128 kMul2;
		UInt128 maxK2 = UInt128.MaxValue / ((UInt128)p << 1);
		ulong limit = divisorCyclesSearchLimit;
		if ((UInt128)limit > maxK2)
		{
			limit = (ulong)maxK2;
		}

		for (ulong k2 = 1UL; k2 <= limit; k2++)
		{
			kMul2 = (UInt128)k2 << 1;
			UInt128 candidate = checked(kMul2 * p);
			d = checked(candidate + UInt128.One);
			if (p < 64UL && d == ((UInt128)1 << (int)p) - UInt128.One)
			{
				continue;
			}

			UInt128 cycle128 = MersenneDivisorCyclesGpu.GetCycle(gpu, d);
			if ((UInt128)p % cycle128 != UInt128.Zero)
			{
				continue;
			}

			readOnlyDivisor = new ReadOnlyGpuUInt128(d); // Reusing readOnlyDivisor for generated divisors.
			if (IsDivisible(p, in readOnlyDivisor))
			{
				divisorsExhausted = true;
				return false;
			}
		}

		divisorsExhausted = (UInt128)divisorCyclesSearchLimit >= maxK2;
		return true;
	}

}

