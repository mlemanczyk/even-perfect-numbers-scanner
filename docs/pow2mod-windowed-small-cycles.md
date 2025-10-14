# ProcessEightBitWindows pow2mod windowed small-cycle usage

## Overview
ProcessEightBitWindows powers both the CPU and GPU implementations of `Pow2ModWindowed`. The CLI configures the execution mode in `Program.Main`, which instantiates a `MersenneNumberTester` that fans out to CPU or GPU testers depending on the `--mersenne`, `--mersenne-device`, and `--order` switches.【F:EvenPerfectBitScanner/Program.cs†L535-L555】【F:PerfectNumbers.Core/MersenneNumberTester.cs†L18-L49】【F:PerfectNumbers.Core/MersenneNumberTester.cs†L241-L312】 This note traces the pow2mod call path and documents when the precomputed "small cycle" table is required.

## CPU execution path
* When `--mersenne-device=cpu` or the tester falls back to CPU scanners, pow2mod calls stay on the host (`MersenneNumberIncrementalCpuTester`, `MersenneNumberOrderCpuTester`, and the Lucas–Lehmer prescan). Each pow2 evaluation uses the shared windowed helper, optionally reduced by whatever cycle length `MersenneDivisorCycles` can provide, but the exponentiation itself only depends on the modulus.【F:PerfectNumbers.Core/Cpu/MersenneNumberIncrementalCpuTester.cs†L8-L89】【F:PerfectNumbers.Core/Cpu/MersenneNumberOrderCpuTester.cs†L8-L73】【F:PerfectNumbers.Core/ULongExtensions.cs†L311-L383】【F:PerfectNumbers.Core/ULongExtensions.cs†L512-L582】
* If a cycle lookup misses, the rotation parameter passed to `Pow2ModWindowed` stays at the original exponent, so the host still computes a correct residue without the table. The small-cycle snapshot therefore acts solely as a performance hint on CPU runs.

## GPU execution path
* When GPU scanning is enabled (`--mersenne-device=gpu` or GPU order mode), the residue, incremental, and order testers all upload the small-cycle snapshot once per accelerator via `GpuKernelPool.EnsureSmallCyclesOnDevice` before launching the ProcessEightBitWindows kernels.【F:PerfectNumbers.Core/Gpu/MersenneNumberResidueGpuTester.cs†L20-L63】【F:PerfectNumbers.Core/Gpu/MersenneNumberIncrementalGpuTester.cs†L18-L78】【F:PerfectNumbers.Core/Gpu/MersenneNumberOrderGpuTester.cs†L12-L53】【F:PerfectNumbers.Core/Gpu/GpuKernelPool.cs†L838-L853】
* Inside `Pow2ModWindowedKernelScan` and `Pow2ModWindowedOrderKernelScan`, the device table is consulted only to reject exponents that are not multiples of the cached cycle length. Regardless of that shortcut, every lane still runs the full windowed pow2 reduction and reports a composite when the residue differs from one.【F:PerfectNumbers.Core/Gpu/GpuKernelPool.cs†L470-L505】【F:PerfectNumbers.Core/Gpu/GpuKernelPool.cs†L706-L719】 As a result, missing device-side small cycles would merely eliminate the early exits; correctness is unaffected.

## Mixed CPU/GPU configurations
* Mixed modes (for example CPU scanning with GPU order checks) route pow2mod calls to the same helpers described above: CPU scanners rely on the host implementation, whereas GPU order/residue probes continue to prefetch the small-cycle table before invoking the ProcessEightBitWindows kernels.【F:PerfectNumbers.Core/MersenneNumberTester.cs†L261-L312】【F:PerfectNumbers.Core/Gpu/MersenneNumberOrderGpuTester.cs†L12-L53】
* The by-divisor pipeline reuses `Pow2ModWindowedGpu` for candidate evaluation and does not touch the kernel-based ladder, so it never requires the device snapshot.【F:PerfectNumbers.Core/Gpu/MersenneNumberDivisorByDivisorGpuTester.cs†L804-L818】【F:PerfectNumbers.Core/ULongExtensions.cs†L416-L482】

## Conclusions
* Small cycles are **not** required for the pow2mod windowed arithmetic itself; they only enable fast-path rejection when a cached cycle divides the exponent.
* Uploading the small-cycle table to the GPU remains worthwhile as a performance optimization, but the pow2mod kernels continue to produce correct answers even if the upload is skipped or unavailable.
