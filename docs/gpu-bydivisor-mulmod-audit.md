# GPU By-Divisor `MulMod` Usage Audit

This note tracks every `MulMod` call site in the GPU-only `--mersenne=bydivisor` pipeline that changed in the recent optimizer work. It now records where the host path swapped over to the fast helpers and which kernel paths already aligned with the benchmark winners.

## CPU helpers invoked by the GPU path

| Location | Notes |
| --- | --- |
| `ULongExtensions.MulMod64Gpu` | Marked as deprecated to steer new code toward `MulMod64` on the CPU and `GpuUInt128.MulMod` inside GPU kernels. The shim remains only for legacy callers that require the GPU-compatible arithmetic façade.【F:PerfectNumbers.Core/ULongExtensions.Gpu.cs†L1-L33】 |
| `ULongExtensions.Pow2ModWindowedGpu` | Migrated to call `MulMod64` for every square and multiply, eliminating the 6–18× slowdown that the shim introduced while executing on the host.【F:PerfectNumbers.Core/ULongExtensions.Gpu.cs†L39-L109】 |
| `ULongExtensions.InitializeStandardOddPowers` | Populates the odd-power table through `MulMod64`, so the host setup costs now match the CPU benchmark baseline.【F:PerfectNumbers.Core/ULongExtensions.Gpu.cs†L286-L309】 |

All host-executed `MulMod` operations in the GPU pipeline now target the optimized CPU helper.

## GPU kernel and device-side code paths

| Location | Notes |
| --- | --- |
| `ULongExtensions.Pow2ModWindowedGpuKernel` | Uses the in-place `GpuUInt128.MulMod` helper, which benchmarks as the fastest device-compatible option when the modulus fits in 64 bits—the case for divisor scans.【F:PerfectNumbers.Core/ULongExtensions.Gpu.cs†L112-L149】【F:Benchmarks.txt†L604-L614】 |
| `ULongExtensions.ComputeWindowedOddPowerGpuKernel` | Also relies on the in-place helper while iterating the odd-power ladder, so no slower variants are in play.【F:PerfectNumbers.Core/ULongExtensions.Gpu.cs†L153-L181】【F:Benchmarks.txt†L604-L614】 |
| `ExponentRemainderStepperGpu.AdvanceStateGpu` | Advances residues via `_currentResidue.MulMod` with the cached modulus. This matches the in-place approach validated by the GPU benchmarks, so nothing to fix here.【F:PerfectNumbers.Core/Gpu/ExponentRemainderStepperGpu.cs†L99-L111】【F:Benchmarks.txt†L604-L614】 |

No other GPU-side `MulMod` calls exist in the by-divisor path, so the device kernels already use the most efficient helper available.
