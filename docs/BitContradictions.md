# Bit-Contradiction Divisor Scan

Plan and notes for the `DivisorSet.BitContradiction` / `--bydivisor-k-increment=bitcontradiction` mode.

## Goal
- Divisor search via joint reconstruction of bits in \(q = 1 + 2pk\) and column-wise carries in \(q \cdot a = 2^p - 1\).
- Early local contradictions (parity reachability, carry budgets, length limits for \(a\)) without recursion.
- Report primes \(p\) whose search space exhausts with no successful branch first.

## Operational assumptions
- \(p\) is odd; we target prime divisors \(q \equiv 1 \pmod{2p}\).
- Work in \(k\)-space (since \(q = 2pk + 1\)).
- Iterative DFS/BFS (explicit stack/queue), no recursion.

## Implementation steps
- [x] Add `BitContradiction` mode to CLI (`--bydivisor-k-increment`) and state repositories/files.
- [x] Wire mode through CPU/Hybrid/GPU paths with `.bitcontradiction` state files.
- [x] Add core interval/propagation primitives (carry ranges, column bounds) with unit tests.
- [x] Wire BitContradiction mode into the scan path with a bit-level divisibility pre-check for small \(p\) before falling back to the standard divisor scan.
- [x] Replace the pre-check with a full forward-only column solver (no fixed column cap), reusable in tests.
- [x] Emit detailed log when the solver prunes a divisor with a concrete contradiction reason.
- [x] Unit tests on small \(p\) covering both success and no-divisor paths.
- [ ] (Optional) Integrate class-model stats to influence \(k_t\) branching order if helpful.
- [ ] Persist solver-specific state (FASTER) in `Checks\bitcontradiction.solver.faster` if we decide to retain per-prime solver telemetry.

## Current minimal version
- Mode is selectable and persists its own state in FASTER/`.bitcontradiction` files.
- Uses the stable by-divisor scan pipeline as a placeholder until the dedicated solver is completed.

## Next steps
- Implement interval-based propagation and iterative DFS on \(k\) bits (no recursion).
- Add tests for carry/length contradictions and for detecting a real divisor.
