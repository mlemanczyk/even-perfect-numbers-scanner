# `--mersenne=bydivisor` Branch Review

This checklist documents each conditional executed along the `--mersenne=bydivisor` path and records the redundancy clean-up that keeps the hot code free of duplicated or unnecessary `if` statements.

## Driver (`EvenPerfectBitScanner/Program.cs`)

- ✅ `_byDivisorStartPrime` now resolves through a single ternary expression instead of a paired `if`/`else`, removing the redundant second branch that always reset the field to zero.
- ✅ The by-divisor block-size override uses a conditional assignment in place of a standalone `if`, avoiding repeated checks of the same flag while keeping the behavior identical.
- ✅ The remaining by-divisor conditionals configure required guards (filter enforcement, candidate loading, execution routing) and are executed exactly once per run, so no duplicates remain.

## CPU dispatcher (`PerfectNumbers.Core/MersenneNumberDivisorByDivisorTester.cs`)

- ✅ The empty-candidate diagnostic now builds the message with a ternary expression, eliminating the duplicated `if (applyStartPrime)` message branches while keeping the wording intact.
- ✅ All other conditionals (start-prime filtering, previous result pruning, worker selection) participate in distinct decision points and therefore remain unchanged.

## GPU dispatcher (`PerfectNumbers.Core/Gpu/MersenneNumberDivisorByDivisorGpuTester.cs`)

- ✅ The divisor session no longer re-checks whether the kernel was initialized after `EnsureExecutionResourcesLocked` populates it; the redundant guard was replaced by a direct access that relies on the earlier setup.
- ✅ The surviving conditionals cover genuinely different execution states (cycle resolution, buffer sizing, hit interpretation) without repeating the same predicate.

With these updates every conditional encountered by the `--mersenne=bydivisor` workflow is either unique to its decision point or consolidated into a single expression.
