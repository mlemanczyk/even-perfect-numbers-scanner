# GPU By-Divisor `MulMod` Usage Audit

This note tracks every `MulMod` call site in the GPU-only `--mersenne=bydivisor` pipeline that changed in the recent optimizer work. It flags host-side helpers that still lean on the slower GPU shim and confirms which kernel paths already align with the fastest benchmarked helpers.

## CPU helpers invoked by the GPU path

| Location | Notes |
| --- | --- |
| `ULongExtensions.MulMod64Gpu` | Still routes through the legacy GPU-compatible shim. Benchmarks show the standard `MulMod64` implementation runs 6–18× faster on dense inputs, so callers that execute on the CPU should migrate to the baseline helper.【F:PerfectNumbers.Core/ULongExtensions.Gpu.cs†L21-L27】【F:Benchmarks.txt†L24-L46】 |
| `ULongExtensions.Pow2ModWindowedGpu` | Squares and multiplies via `MulMod64Gpu`, so every ladder step pays the same 6–18× penalty noted above. The method executes entirely on the host, so it should also be switched to the baseline `MulMod64` helper in a follow-up.【F:PerfectNumbers.Core/ULongExtensions.Gpu.cs†L39-L109】【F:Benchmarks.txt†L24-L46】 |
| `ULongExtensions.InitializeStandardOddPowers` | Populates the CPU-side table with repeated `MulMod64Gpu` calls. Replacing those calls with `MulMod64` will eliminate the overhead once the surrounding helper is updated.【F:PerfectNumbers.Core/ULongExtensions.Gpu.cs†L288-L309】【F:Benchmarks.txt†L24-L46】 |

These are the only host-executed `MulMod` operations remaining in the GPU pipeline, so they form the to-do list for the next pass.

## GPU kernel and device-side code paths

| Location | Notes |
| --- | --- |
| `ULongExtensions.Pow2ModWindowedGpuKernel` | Uses the in-place `GpuUInt128.MulMod` helper, which benchmarks as the fastest device-compatible option when the modulus fits in 64 bits—the case for divisor scans.【F:PerfectNumbers.Core/ULongExtensions.Gpu.cs†L112-L149】【F:Benchmarks.txt†L604-L614】 |
| `ULongExtensions.ComputeWindowedOddPowerGpuKernel` | Also relies on the in-place helper while iterating the odd-power ladder, so no slower variants are in play.【F:PerfectNumbers.Core/ULongExtensions.Gpu.cs†L153-L181】【F:Benchmarks.txt†L604-L614】 |
| `ExponentRemainderStepperGpu.AdvanceStateGpu` | Advances residues via `_currentResidue.MulMod` with the cached modulus. This matches the in-place approach validated by the GPU benchmarks, so nothing to fix here.【F:PerfectNumbers.Core/Gpu/ExponentRemainderStepperGpu.cs†L104-L111】【F:Benchmarks.txt†L604-L614】 |

No other GPU-side `MulMod` calls exist in the by-divisor path, so the device kernels already use the most efficient helper available.
