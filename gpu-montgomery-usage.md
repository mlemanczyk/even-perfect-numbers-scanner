# GPU Modular Arithmetic in the `--mersenne=bydivisor` Path

The GPU implementation now routes single-use exponent checks through standard modular arithmetic while keeping Montgomery reduction only where repeated multiplications reuse the same modulus.

## Kernels using standard modular arithmetic

- `CheckKernel` in `MersenneNumberDivisorByDivisorGpuTester` evaluates each divisor candidate with `Pow2ModBinaryGpu` and compares the result directly against 1. 【F:PerfectNumbers.Core/Gpu/MersenneNumberDivisorByDivisorGpuTester.cs†L561-L645】【F:PerfectNumbers.Core/ULongExtensions.cs†L314-L339】
- `ComputeExponentKernel` in the same tester precomputes batched residues with the plain helper so CPU lookups receive standard-domain results. 【F:PerfectNumbers.Core/Gpu/MersenneNumberDivisorByDivisorGpuTester.cs†L780-L791】【F:PerfectNumbers.Core/ULongExtensions.cs†L314-L339】
- `ComputeResiduesKernel` inside `MersenneNumberDivisorResidueGpuEvaluator` now mirrors that logic when building residue tables for mixed CPU/GPU scans. 【F:PerfectNumbers.Core/Gpu/MersenneNumberDivisorResidueGpuEvaluator.cs†L102-L114】【F:PerfectNumbers.Core/ULongExtensions.cs†L314-L339】
- The order-search helpers in `PrimeOrderGpuHeuristics` (`TrySpecialMaxKernel`, `ExponentLoweringKernel`, and `Pow2EqualsOneKernel`) also rely on `Pow2ModBinaryGpu` for their equality checks. 【F:PerfectNumbers.Core/Gpu/PrimeOrderGpuHeuristics.cs†L798-L866】【F:PerfectNumbers.Core/Gpu/PrimeOrderGpuHeuristics.cs†L1596-L1598】

## Kernels still using Montgomery arithmetic

- `MontgomeryOddPowerKernelScan` in `GpuKernelPool` continues to build windowed odd-power tables with `MontgomeryMultiply`, which amortizes the Montgomery setup over many multiplications per modulus. 【F:PerfectNumbers.Core/Gpu/GpuKernelPool.cs†L200-L237】【F:PerfectNumbers.Core/ULongExtensions.cs†L532-L559】

## Shared helper

- `Pow2ModBinaryGpu` in `ULongExtensions` implements the GPU-friendly binary exponentiation loop that powers the standard modular path across these kernels. 【F:PerfectNumbers.Core/ULongExtensions.cs†L314-L339】
