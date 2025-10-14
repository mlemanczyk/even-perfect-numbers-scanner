# ProcessEightBitWindows pow2mod windowed small-cycle usage

## Overview
ProcessEightBitWindows powers both the CPU and GPU implementations of `Pow2ModWindowed`. The CLI configures the execution mode in `Program.Main`, which instantiates a `MersenneNumberTester` that fans out to CPU or GPU testers depending on the `--mersenne`, `--mersenne-device`, and `--order` switches.【F:EvenPerfectBitScanner/Program.cs†L535-L555】【F:PerfectNumbers.Core/MersenneNumberTester.cs†L18-L49】【F:PerfectNumbers.Core/MersenneNumberTester.cs†L241-L312】 This note documents where the precomputed "small cycle" table still matters after removing it from the ProcessEightBitWindows pow2mod kernels.

## CPU execution path
* When `--mersenne-device=cpu` or the tester falls back to CPU scanners, pow2mod calls stay on the host (`MersenneNumberIncrementalCpuTester`, `MersenneNumberOrderCpuTester`, and the Lucas–Lehmer prescan). Each pow2 evaluation uses the shared windowed helper and optionally reuses the cached divisor cycles provided by `MersenneDivisorCycles`. Missing cycle data simply means the helper evaluates the full exponent.【F:PerfectNumbers.Core/Cpu/MersenneNumberIncrementalCpuTester.cs†L8-L89】【F:PerfectNumbers.Core/Cpu/MersenneNumberOrderCpuTester.cs†L8-L73】【F:PerfectNumbers.Core/ULongExtensions.cs†L311-L383】【F:PerfectNumbers.Core/ULongExtensions.cs†L512-L582】

## GPU execution path
* `GpuKernelPool` now exposes two ProcessEightBitWindows kernels: a warmed variant that only depends on the small-prime tables and a pooled variant that also consumes the device small-cycle snapshot.【F:PerfectNumbers.Core/Gpu/GpuKernelPool.cs†L124-L207】【F:PerfectNumbers.Core/Gpu/GpuKernelPool.cs†L401-L520】
* Callers that benefit from the cycle table (for example the residue tester) acquire the pooled kernel via `GpuKernelPool.GetKernel(..., requiresSmallCycles: true)`, upload the snapshot once per accelerator, and return the lease so the number of in-memory kernels never exceeds the `--primes-with-smallcycles-threads` limit.【F:PerfectNumbers.Core/Gpu/MersenneNumberResidueGpuTester.cs†L20-L86】【F:EvenPerfectBitScanner/Program.cs†L66-L74】【F:PerfectNumbers.Core/Gpu/GpuSmallCycleKernelLimiter.cs†L1-L51】
* Warmup still preloads only the cycle-free variant alongside the prime tables; the small-cycle kernel is compiled on demand when a lease is acquired.【F:PerfectNumbers.Core/Gpu/GpuKernelPool.cs†L900-L928】

## Mixed CPU/GPU configurations
* Mixed modes (for example CPU scanning with GPU order checks) route pow2mod calls through the same helpers. CPU scanners remain free to consult cached cycles, while GPU probes run the cycle-free kernels described above.【F:PerfectNumbers.Core/MersenneNumberTester.cs†L261-L312】【F:PerfectNumbers.Core/Gpu/MersenneNumberOrderGpuTester.cs†L8-L66】

## Conclusions
* Small cycles remain optional for CPU pow2mod helpers and are now gated on a shared GPU pool so only a bounded number of kernels preload the device snapshot.
* The default GPU workflow stays with the cycle-free kernel; requesting the pooled variant trades a small amount of contention for the early exits provided by the cached cycle lengths.
