using System;

namespace PerfectNumbers.Core.Gpu;

/// <summary>
/// Uses GPU kernels to incrementally scan Mersenne candidates for prime exponents p >= 31.
/// </summary>
public class MersenneNumberIncrementalGpuTester(GpuKernelType kernelType, bool useGpuOrder)
{
    private readonly GpuKernelType _kernelType = kernelType;
    private readonly bool _useGpuOrder = useGpuOrder;

    public void Scan(ulong exponent, UInt128 twoP, bool lastIsSeven, UInt128 maxK, ref bool isPrime)
    {
        throw new NotImplementedException($"GPU incremental scanning requires the device cycle heuristics implementation (kernel: {_kernelType}, GPU order: {_useGpuOrder}).");
    }
}
