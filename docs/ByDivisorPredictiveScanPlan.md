# By-Divisor Adaptive Scan Plan

## Goals
- [x] Add deterministic class-based strategy selection for --mersenne=bydivisor (CPU/Hybrid first) using k~q/(2p) stats per 10-bit suffix.
- [x] Persist and reuse class statistics (k10 percentiles, gap probability, stability) to drive k-start/k-step decisions; rebuild from recorded divisors when stats files are missing or stale.
- [x] Skip trivial divisors and previously found divisors for a given p when rebuilding stats or scanning.
- [x] Introduce a classifier component that categorizes p -> class params and exposes recommended scan ranges/steps for k.
- [x] Integrate classifier into by-divisor CPU/hybrid scan path to choose aggressive k progression (cheap low-k sweep + geometric escalation to class targets).
- [x] Update persistence to append new findings and refresh the model for future runs.
- [x] Keep GPU path unchanged for now; guard hooks for future extension.

## Background / Inputs
- Existing results file (p,divisor,searchedMersenne,detailedCheck,passedAllTests) already loaded in Program.cs (by-divisor flow requires --filter-p and reads prior results).
- Divisor form: q = 2*k*p + 1. We need per-class stats of q/p percentiles -> k percentiles (k10 ~= P10/2, k50 ~= median/2, k75 ~= P75/2) and gap probability (missing rows = gaps).
- Classes keyed by suffix s = p & 1023 (10 LSBs). Stability classification available when rolling envelope data exists; otherwise default to "unknown/assume stable".
- Aggressive strategy (Spec 2):
  - Phase A: always scan cheap low k up to K_cheap.
  - Phase B: if not found, jump to class start k_start = max(K_cheap+1, ceil(gamma50*k50)), escalate by rho1 until k_target = ceil(gamma75*k75), then rho2.
  - Hard/gappy classes (gap_prob >= 0.75 or escaping) use more aggressive multipliers (gamma50=1.0, gamma75=1.4, rho2=1.8). Default gamma50=0.8, gamma75=1.2, rho1=1.35, rho2=1.6.

## Planned Artifacts
- New docs/stats files to persist model (e.g., docs/stats/bydivisor_k10_model.json or CSV) with suffix->{k50,k75,gap_prob,stability,timestamps}.
- Runtime cache file near results/state directory (e.g., Checks/bydivisor-kmodel.bin or similar) for quick load/update.
- Helper to rebuild model from prior results file when model file missing; updates appended when new divisors found.

## Implementation Tasks
1) Data model and persistence
- [x] Define DTO/structs for per-suffix stats (k50,k75,gap_prob,stability tier, sample counts, last-updated).
- [x] Choose on-disk format (JSON/CSV) stored under docs/stats and runtime writable copy under Checks/ (or alongside results). Implement load/save with versioning.
- [x] Add builder that reads existing results (p,q,passed) and computes percentiles + gap probability (skip trivial q, enforce q == 1 mod 2p and pow check when needed or reuse flags).
- [x] Handle missing model files by building from results; merge newly found divisors via rebuild after a run.

2) Classifier
- [x] Implement classifier service (ByDivisorClassModel) returning class params for p: suffix lookup, fallbacks when stats absent.
- [x] Expose recommendation API for scanning: GetScanPlan(p) -> {kCheap, kStart, kTarget, rho1, rho2} with deterministic defaults.
- [x] Flag classes as hard/gappy using thresholds (gap_prob, stability) to pick aggressive multipliers.

3) Integration with by-divisor CPU/Hybrid flow
- [x] Feed classifier with loaded model at Program start (by-divisor path) using the already loaded results file; share between CPU/Hybrid testers.
- [x] Extend divisor scan session to use Phase A/B plan: cheap linear sweep up to K_cheap, then geometric progression per class (k_start -> k_target with rho1, then rho2) instead of flat increments.
- [x] Ensure pow2-groups path remains unchanged; apply to sequential path first (CPU/Hybrid). Guard GPU path (no-op for now).
- [x] Skip trivial divisors and previously recorded divisors for p when scanning/rebuilding stats.
- [x] Persist updated model after processing primes (additions from new divisors found during run).

4) Validation
- [ ] Add targeted unit/functional tests (xUnit + FluentAssertions) for classifier defaults, model rebuild from synthetic results, and scan plan generation for easy vs hard classes.
- [ ] Document how to run fast tests (Category=Fast) and keep runtime under 2 minutes.

## Open Questions / Assumptions
- Results file path reused for stats rebuild; assumes divisor column contains the found q (non-zero only when divisor found).
- Stability classification file presence: if not shipped, treat as stable; allow future plug-in when rolling envelopes added.
- K_cheap initial guess TBD (needs perf check); provisional: small constant (e.g., 10_000) until benchmarked.
