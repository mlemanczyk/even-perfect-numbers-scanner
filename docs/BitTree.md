# Bit-Tree Divisor Scan

Plan and notes for the `DivisorSet.BitTree` / `--bydivisor-k-increment=bittree` mode.

## Goal
- Divisor search via joint reconstruction of unknown bits in both the divisor \(q\) (only LSB=1 is known initially) and the cofactor \(a\), matching column-wise carries in \(q \cdot a = 2^p - 1\).
- Explore the full bit tree (all admissible \(q,a\) combinations) with early pruning from parity reachability, carry budgets, and length limits; no recursion (explicit stack/queue).
- Report primes \(p\) whose entire search space exhausts with no successful branch first.

## Operational assumptions
- \(p\) is odd; we target prime-like divisors with \(q \equiv 1 \pmod{2p}\) but start from bit patterns (q LSB = 1) and let the tree explore.
- Work in joint (a,q) space, not tied to a single \(k\). All admissible \(q\) consistent with \(2pk+1\) should be reachable through the bit tree.
- Iterative DFS/BFS (explicit stack/queue), no recursion. Process columns streaming (no full 138M+ bit buffers).

## Implementation steps
- [x] Add `BitTree` mode to CLI (`--bydivisor-k-increment`) and state repositories/files; ensure distinct state extensions.
- [x] Wire mode through CPU/Hybrid/GPU paths with `.bittree` state files.
- [x] Implement core bit-tree solver (frontier/BFS variant):
  - Joint state: column index, carry, partial `q` (LSB=1), partial `a`, bounds on `a` length (≤ p - bitlen(q)+1), checks against `allowedMax`.
  - Branching: choose next `q` bit (0/1), derive `a` bit from parity+carry; prune on length overflow, parity unreachable, carry cannot collapse, or `q` beyond allowedMax.
  - Termination: success iff at/beyond column `p` with carry=0 and no further contributions from `(a,q)`; otherwise contradiction.
- [x] Integrate BitTree solver into `CheckDivisorsBitTree`; legacy per-`k` loop removed for this mode.
- [ ] Logging: emit Decided/Divides and contradiction reasons; throw if solver cannot decide.
- [ ] Tests:
  - Unit tests on small \(p\) for success/failure paths.
  - Integration test selecting `DivisorSet.BitTree` to ensure composites are detected and primes pass.
- [ ] Docs/help: mention the new mode in help text and README references.

## Variants considered

### Variant A (implemented baseline)
- Explicit stack-based DFS over `(a,q)` bits with reachability/pruning on parity and carry budgets.
- Processes columns sequentially without recursion; suited to small/medium `p` in current form.

### Variant B (frontier/windowed exploration — in progress)
- Implemented: frontier/BFS over branches with sliding window (suffix) storage for `(a,q)` and per-branch carry, bounded by `allowedMax` and `p`; window slides forward to cap memory and drop obsolete history.
- Planned enhancement: tune window size/adaptive policy and add optional frontier compression (baseline/delta) for `p ≥ 138M` and many threads.

### Variant C (frontier with shared baseline/deltas — planned)
- For each processed column maintain a shared baseline state (e.g., minimal carry/bit sums across the frontier).
- Each branch stores only its delta relative to the baseline (carry delta and the minimal differing `(a,q)` suffix), shrinking per-branch memory.
- When compressing the frontier after a window, pick the baseline that minimises aggregate delta size (e.g., minimal sum of carries/bit-counts across surviving branches).
- Objective: reduce memory further for highly parallel runs (many threads) while preserving full branch coverage.

## Current status
- Variant A solver is integrated and invoked from `DivisorSet.BitTree` (branch scan path); legacy per-`k` loop is disabled for this mode.
- Variant B is not implemented yet; design notes above.

## Next steps
- Implement frontier/windowed solver (Variant B) and add mode toggle if needed.
- Prototype baseline/delta compression for the frontier (Variant C) and measure memory savings under parallel load.
- Expand tests to cover both variants on small `p` and ensure integration keeps reporting decisions for all candidates.
