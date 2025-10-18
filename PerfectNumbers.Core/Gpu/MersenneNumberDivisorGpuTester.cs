using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

/// <summary>
/// Provides GPU-accelerated divisibility checks for Mersenne numbers with prime exponents p >= 31.
/// </summary>
public sealed class MersenneNumberDivisorGpuTester
{
	private readonly ConcurrentDictionary<Accelerator, Action<Index1D, ulong, ReadOnlyGpuUInt128, ArrayView<byte>>> _kernelCache = new();
	private readonly ConcurrentDictionary<Accelerator, MemoryBuffer1D<byte, Stride1D.Dense>> _resultBuffers = new();

	private Action<Index1D, ulong, ReadOnlyGpuUInt128, ArrayView<byte>> GetKernel(Accelerator accelerator) =>
			_kernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ulong, ReadOnlyGpuUInt128, ArrayView<byte>>(DivisorKernels.Kernel));

        public static void BuildDivisorCandidates()
        {
                ulong[] snapshot = MersenneDivisorCycles.Shared.ExportSmallCyclesSnapshot();
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

    public bool IsDivisible(ulong exponent, in ReadOnlyGpuUInt128 divisor)
    {
        var gpu = GpuContextPool.RentPreferred(preferCpu: false);
        var accelerator = gpu.Accelerator;
        var kernel = GetKernel(accelerator);
        var resultBuffer = _resultBuffers.GetOrAdd(accelerator, acc => acc.Allocate1D<byte>(1));
        resultBuffer.MemSetToZero();
        kernel(1, exponent, divisor, resultBuffer.View);
        accelerator.Synchronize();
        Span<byte> result = stackalloc byte[1];
        resultBuffer.View.CopyToCPU(ref result[0], 1);
        bool divisible = result[0] != 0;
        result[0] = 0;
        resultBuffer.View.CopyFromCPU(ref result[0], 1);
        accelerator.Synchronize();
        gpu.Dispose();
        return divisible;
    }

        private static (ulong divisor, uint cycle)[]? _divisorCandidates = Array.Empty<(ulong divisor, uint cycle)>();

    public bool IsPrime(ulong p, UInt128 d, ulong divisorCyclesSearchLimit, out bool divisorsExhausted)
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
        var divisorCycles = MersenneDivisorCycles.Shared;
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

            UInt128 cycle128 = MersenneDivisorCycles.GetCycle(d);
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

