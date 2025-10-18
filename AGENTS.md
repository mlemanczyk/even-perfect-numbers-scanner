# Repository Guidelines

- Only solution: `EvenPerfectScanner.sln` (console app scanning numeric ranges).
- Keep changes confined to the relevant solution/project unless instructed otherwise.
- Assume each solution has or will have its own tests and build configuration; avoid cross-project edits unless required.

## Build and Test

- Build: `dotnet build EvenPerfectScanner.sln`.
- Run only tests related to the modified solution.
- Run tests without `--no-build`.
- Write tests with xUnit + FluentAssertions.
- Confirm test runtimes are reasonable; skip suites expected to run excessively long.
- Always apply timeout guards and `Category=Fast` filters when running tests.

## File Editing

- Do not use `apply_patch.cs` until TODOs lift the ban; edit files with shell tools and stage via git.
- Intended usage (once re-enabled): `dotnet script apply_patch.cs -- <patch-file> [target-directory]`.
- Expect the script to run in local/remote Codex environments on Windows and Linux.
- Supported options (when allowed):
  - `--check`
  - `--ignore-whitespace`
  - `--ignore-eol`
  - `--ignore-line-endings`
  - `--target <root>`
  - `--profile`
  - `--replace-line <path> <line> <expected> <new>` (use `_` for expected to skip validation)
- Build patches in standard `git diff` format:
  - Include `diff --git`, `---`, `+++`, and `@@` headers with numeric ranges.
  - Exclude `***` or `*** Begin/End Patch` wrappers.
- Run `--check` before applying patches; apply only after a successful check.
- Keep patches small, precise, and regenerated after each edit; confirm target lines with `Get-Content` or `nl -ba`.
- Record exact start lines for each hunk and double-check with `nl -ba`.
- Prefer `--replace-line` for single-line edits after capturing the target line number.
- Regenerate diffs after successful patch applications and whenever files shift.
- Use `--ignore-whitespace` / `--ignore-line-endings` only when validation noise blocks `--check`.
- Maintain unique patch filenames and delete applied ones.
- Investigate failed hunks immediately and rebuild diffs instead of retrying stale context.
- Prefer single-purpose patches.
- Use `nl -ba` to show line numbers when needed.
- Use four spaces for code block indentation.
- Create patches from the repo root with relative paths.
- Do not apply changes manually or via other scripts unless modifying `apply_patch.cs`.
- If `apply_patch.cs` has issues or lacks features, fix or enhance it and continue using it.

## Test Execution Time Policy

- Estimate runtimes before launching tests or samples.
- Do not run tests expected to exceed 2 minutes unless explicitly authorized.
- Never impose the two-minute guard on benchmarks or profiling runs; let them finish unless obviously stuck.
- Keep timeouts external to the test code.
- Use targeted filters such as `--filter FullyQualifiedName~<ClassOrMethod>` or `--filter "FullyQualifiedName=Namespace.Class.Method"`.
- Keep GPU-heavy tests minimal; skip kernels needing >2 minutes for compile/warmup.
- In local Codex tasks, run only fast, newly added tests and list recommended existing tests instead.

### 120-second Guard Examples

- Windows (PowerShell):
  - `$proj = "<path-to>.csproj"`
  - `$filter = 'FullyQualifiedName~SomeTests'`
  - `$job = Start-Job -ScriptBlock { param($p,$f) dotnet test $p -c Debug --filter $f } -ArgumentList $proj,$filter`
  - `if (-not (Wait-Job $job -Timeout 120)) { Stop-Job $job; throw 'Test run exceeded 120s timeout.' }`
  - `Receive-Job $job`
- Linux/macOS (bash):
  - `timeout 120s dotnet test <path-to>.csproj -c Debug --filter "Category=Fast&FullyQualifiedName~SomeTests"`
  - Always specify `timeout 120s` (or shorter) and include `Category=Fast` in the filter.
  - Treat exit code 124 as a timeout.
- Codex/agent runs: wrap `dotnet test` commands in shell guards above.
- Filter examples: `--filter "FullyQualifiedName~PrimeTesterGcdTests"`, `--filter "FullyQualifiedName=EvenPerfectBitScanner.Tests.ResultsFileNameTests.BuildResultsFileName_contains_the_range"`.

## Coding Standards

- Use four-space indentation; increase one level per nested block.
- Indent wrapped parameters/expressions by two indentation units.
- Reuse existing variables when prior values are obsolete; call out meaning changes in comments to keep intent clear and register pressure low.
- Avoid try/finally blocks solely for resource release unless omitting finally harms clarity or maintainability; ensure every path releases resources.
- Limit the divisor cycle cache to one in-memory block plus the on-disk snapshot; recompute on misses, discard transient results, and avoid background generation or extra caches.
- Write code, comments, commit messages, branch names, and PR descriptions in American English.
- Preserve relevant TODO notes and explanatory comments unless invalid.
- Prefer structs/classes with public fields over auto-properties for hot-path data holders.
- Pass mutable structs as parameters with the `in` modifier whenever the call site and usage allow it (skip when constraints such as lambda captures require a copy).
- Omit the `in` modifier when passing immutable structs as parameters.
- Remove or extend trivial wrapper methods that only call another method.

## Performance Checklist

- Favor dedicated kernels over flag-controlled/parameterized ones; duplicate code when it improves speed.
- Remove `if` statements inside hot loops when possible.
- Replace runtime flags in hot paths with dedicated method variants.
- Reuse variables instead of allocating new ones; document meaning changes.
- Extract functionality into separate classes and place reusable parts in `*.Core` projects.
- Rent arrays from `ArrayPool<T>` when stack allocation is impractical and return them after use.
- Provide cancellation-aware and cancellation-free variants of hot methods; handle cancellation in outer scopes.
- Prefer `for` loops over `foreach` to avoid enumerator allocations.
- Declare loop control variables outside loops and reuse them.
- Cache arrays, static/read-only arrays, and collection counts in locals before loops.
- Declare variables together outside loops when feasible and reuse them.
- Avoid repeated casting within loops by precomputing or using wider variables.

## GPU Notes

- ILGPU 1.5.3 checks: `%` and integer division supported for 32-bit/64-bit integers; `UInt128` remains unsupported inside kernels. Use the `GpuUInt128` struct for GPU-side emulation and extend it whenever required arithmetic helpers are missing.
- Arithmetic in GPU kernels covers only 32-bit and 64-bit primitives; rely on `GpuUInt128` together with its read-only counterpart `ReadOnlyGpuUInt128` to emulate 128-bit behavior.
- CPU code should default to `UInt128` instead of `GpuUInt128` unless the same calculation executes millions or billions of times, in which case prefer mutable `GpuUInt128` when benchmarks show no material regression; benchmark summaries and recommendations live in `Benchmarks.md`.
- GPU kernels should favor mutable `GpuUInt128` and `ReadOnlyGpuUInt128` when the necessary operations exist on the accelerator and benchmarks do not show a dramatic advantage for `UInt128`.
- Perform conversions from `GpuUInt128` or other structs to `ReadOnlyGpuUInt128` outside hot loops whenever possible; push the conversion toward the top of the call chain to avoid repeated costs.
- For each numeric type, review available helpers in `UIntExtensions`, `ULongExtensions`, `UInt128Extensions`, and `GpuUInt128`; add new operations to the extension class or struct that matches the type.
- Arrays/vectors of `bool` unsupported; use `byte`.
- `UInt128` multiplication, division, bit shifts unsupported in kernels; implement manually.
- GPU context pooling supported and recommended.
- Numeric probe results (ILGPU 1.5.3, CPU accelerator):
  - `System.Numerics.BigInteger`: unsupported (`InternalCompilerException`).
  - `PeterO.Numbers.EInteger`: unsupported (`InternalCompilerException`).
  - `PeterO.Numbers.ERational`: unsupported (`InternalCompilerException`, lacks remainder API).

## Patch Preparation Reminders

- Manually review every patch before `--check`; remove unintended whitespace noise.
- Capture context with `nl -ba` before composing each hunk; include ≥3 context lines when available.
- Normalize trailing whitespace early or rely on ignore flags temporarily, then rerun without ignores to confirm the final diff.
- Regenerate patches from scratch after working tree changes; avoid editing old diffs.
- Note that the parser ignores blank metadata lines between headers/hunks; keep headers compact unless tools add blanks.

## apply_patch.cs Profiling Snapshot (2025-10-04)

- `dotnet script apply_patch.cs -- --profile --check /tmp/agents_todo_update.diff`.
- Sample patch: 2 files, 4 hunks, 31 lines, 3468 bytes.
- Timings: read 0.45 ms, parse 2.96 ms, apply 2.96 ms.
- Next focus: stream the patch reader and reuse pooled buffers to reduce parse/apply cost.

## Mersenne Divisor Guidance

- Prime divisor `q` of `2^p - 1` satisfies `q ≡ 1 (mod 2p)` with minimal `q = 2p + 1` for `k = 1`.
- Target for at least `p` in `[138,000,000, 200,000,000]`, minimal `q < 400,000,001` (29 bits).
- Target for at least `p = 600,000,000`, minimal `q ≤ 1,200,000,001` (31 bits).
- Larger `k` values can grow `q` up to nearly `2^p - 1`; plan heuristics/storage for wider integers even though the noted bit widths cover the first admissible divisors.
- Worst-case divisor width: `p - 1` bits (`⌊(2^p - 1) / 2⌋ = 2^{p-1} - 1`).
- `k_max ≈ 2^{p-2} / p` for large `p`.

## Practical Kernel Guidance

- Use native `%` and `/` for 32-bit/64-bit integers when efficient.
- For `UInt128` in kernels, avoid `%`/`/`; prefer Montgomery/Barrett or Mersenne folding.
- Prefer Mersenne folding for `2^p - 1` moduli to remove branches in hot loops.

## Heuristic Multiply-Shift Benchmarks (2025-10-04)

- Run `dotnet run -c Release --project tools/HeuristicMultiplyShiftTiming/HeuristicMultiplyShiftTiming.csproj` to compare `ULongExtensions` helpers without the BenchmarkDotNet harness.
- Average ns per call over 200,000,000 iterations (identical checksums):
  - `NearOverflowShift`: `MultiplyShiftRight` ≈ 2.37 ns, shift-first ≈ 4.71 ns, naive ≈ 0.85 ns.
  - `HalfRange`: `MultiplyShiftRight` ≈ 2.11 ns, shift-first ≈ 4.15 ns, naive ≈ 0.70 ns.
  - `MixedBits`: `MultiplyShiftRight` ≈ 1.43 ns, shift-first ≈ 2.83 ns, naive ≈ 0.60 ns.
  - `Sparse`: `MultiplyShiftRight` ≈ 1.57 ns, shift-first ≈ 2.84 ns, naive ≈ 0.59 ns.
- Shift-first doubles CPU cost but stays within 64-bit arithmetic and is the GPU fallback when avoiding `UInt128` overflow.
