# ProcessEightBitWindows pow2mod windowed small-cycle usage

## Overview
ProcessEightBitWindows powers both the CPU and GPU implementations of `Pow2ModWindowed`. The CLI configures the execution mode in `Program.Main`, which instantiates a `MersenneNumberTester` that fans out to CPU or GPU testers depending on the `--mersenne`, `--mersenne-device`, and `--order` switches.【F:EvenPerfectBitScanner/Program.cs†L535-L555】【F:PerfectNumbers.Core/MersenneNumberTester.cs†L18-L49】【F:PerfectNumbers.Core/MersenneNumberTester.cs†L241-L312】 This note documents where the precomputed "small cycle" table still matters after removing it from the ProcessEightBitWindows pow2mod kernels.

## CPU execution path
* When `--mersenne-device=cpu` or the tester falls back to CPU scanners, pow2mod calls stay on the host (`MersenneNumberIncrementalCpuTester`, `MersenneNumberOrderCpuTester`, and the Lucas–Lehmer prescan). Each pow2 evaluation uses the shared windowed helper and optionally reuses the cached divisor cycles provided by `MersenneDivisorCycles`. Missing cycle data simply means the helper evaluates the full exponent.【F:PerfectNumbers.Core/Cpu/MersenneNumberIncrementalCpuTester.cs†L8-L89】【F:PerfectNumbers.Core/Cpu/MersenneNumberOrderCpuTester.cs†L8-L73】【F:PerfectNumbers.Core/ULongExtensions.cs†L311-L383】【F:PerfectNumbers.Core/ULongExtensions.cs†L512-L582】

## GPU execution path
* `GpuKernelPool` continues to expose the cycle-free ProcessEightBitWindows kernel, which remains prewarmed alongside the small-prime tables.【F:PerfectNumbers.Core/Gpu/GpuKernelPool.cs†L124-L207】【F:PerfectNumbers.Core/Gpu/GpuKernelPool.cs†L518-L623】【F:PerfectNumbers.Core/Gpu/GpuKernelPool.cs†L900-L928】
* The small-cycle kernel is temporarily idle; all production callers use the warmed variant and skip uploading the cycle snapshot so `GpuSmallCycleKernelLimiter` stays dormant.【F:PerfectNumbers.Core/Gpu/MersenneNumberResidueGpuTester.cs†L8-L104】【F:PerfectNumbers.Core/Gpu/GpuKernelPool.cs†L900-L928】【F:PerfectNumbers.Core/Gpu/GpuSmallCycleKernelLimiter.cs†L1-L51】
* When profiling requires the pooled kernel again, re-enable the leases and restore the cycle uploads so the limiter can enforce the `--primes-with-smallcycles-threads` cap.【F:EvenPerfectBitScanner/Program.cs†L66-L74】【F:PerfectNumbers.Core/Gpu/GpuKernelPool.cs†L900-L928】

## Mixed CPU/GPU configurations
* Mixed modes (for example CPU scanning with GPU order checks) route pow2mod calls through the same helpers. CPU scanners remain free to consult cached cycles, while GPU probes run the cycle-free kernels described above.【F:PerfectNumbers.Core/MersenneNumberTester.cs†L261-L312】【F:PerfectNumbers.Core/Gpu/MersenneNumberOrderGpuTester.cs†L8-L66】

## Conclusions
* Small cycles remain optional for CPU pow2mod helpers, but the GPU scanners currently skip the pooled kernel entirely.
* The default GPU workflow stays with the cycle-free kernel; reintroduce the pooled variant once the early-exit savings justify the extra device memory.
