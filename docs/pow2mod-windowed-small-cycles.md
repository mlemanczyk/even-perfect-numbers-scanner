# ProcessEightBitWindows pow2mod windowed small-cycle usage

## Overview
ProcessEightBitWindows powers both the CPU and GPU implementations of `Pow2ModWindowed`. The CLI configures the execution mode in `Program.Main`, which instantiates a `MersenneNumberTester` that fans out to CPU or GPU testers depending on the `--mersenne`, `--mersenne-device`, and `--order` switches.【F:EvenPerfectBitScanner/Program.cs†L535-L555】【F:PerfectNumbers.Core/MersenneNumberTester.cs†L18-L49】【F:PerfectNumbers.Core/MersenneNumberTester.cs†L241-L312】 This note documents where the precomputed "small cycle" table still matters after removing it from the ProcessEightBitWindows pow2mod kernels.

## CPU execution path
* When `--mersenne-device=cpu` or the tester falls back to CPU scanners, pow2mod calls stay on the host (`MersenneNumberIncrementalCpuTester`, `MersenneNumberOrderCpuTester`, and the Lucas–Lehmer prescan). Each pow2 evaluation uses the shared windowed helper and optionally reuses the cached divisor cycles provided by `MersenneDivisorCycles`. Missing cycle data simply means the helper evaluates the full exponent.【F:PerfectNumbers.Core/Cpu/MersenneNumberIncrementalCpuTester.cs†L8-L89】【F:PerfectNumbers.Core/Cpu/MersenneNumberOrderCpuTester.cs†L8-L73】【F:PerfectNumbers.Core/ULongExtensions.cs†L311-L383】【F:PerfectNumbers.Core/ULongExtensions.cs†L512-L582】

## GPU execution path
* GPU residue, incremental, and order testers now request only the small-prime tables before launching the ProcessEightBitWindows kernels; no small-cycle upload is required.【F:PerfectNumbers.Core/Gpu/MersenneNumberResidueGpuTester.cs†L20-L67】【F:PerfectNumbers.Core/Gpu/MersenneNumberIncrementalGpuTester.cs†L30-L83】【F:PerfectNumbers.Core/Gpu/MersenneNumberOrderGpuTester.cs†L8-L66】
* The pow2mod kernels themselves no longer consult a device-side cycle table—each lane constructs `q = 2pk + 1` and immediately runs the windowed ladder, so correctness depends only on the modulus.【F:PerfectNumbers.Core/Gpu/GpuKernelPool.cs†L400-L474】【F:PerfectNumbers.Core/Gpu/GpuKernelPool.cs†L520-L586】
* Warmup preloads the ProcessEightBitWindows kernels and the small-prime buffers so later launches remain branch free, but no longer touches the removed cycle snapshot.【F:PerfectNumbers.Core/Gpu/GpuKernelPool.cs†L740-L780】

## Mixed CPU/GPU configurations
* Mixed modes (for example CPU scanning with GPU order checks) route pow2mod calls through the same helpers. CPU scanners remain free to consult cached cycles, while GPU probes run the cycle-free kernels described above.【F:PerfectNumbers.Core/MersenneNumberTester.cs†L261-L312】【F:PerfectNumbers.Core/Gpu/MersenneNumberOrderGpuTester.cs†L8-L66】

## Conclusions
* Small cycles remain a CPU-side optimization that can shorten pow2mod exponents when the divisor cache produces a hit.
* GPU pow2mod kernels now rely solely on the ProcessEightBitWindows ladder and do not require any cycle metadata to produce correct residues.
