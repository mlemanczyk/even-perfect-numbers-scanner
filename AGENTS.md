# Repository Guidelines

This repository contains only one solution:

- `EvenPerfectScanner.sln` – a console application that scans numeric ranges.

Changes should be isolated to the relevant solution and project. When working on one
project, avoid modifying the other unless explicitly required. Each solution
has or will have its own test suite and build configuration.

To build `EvenPerfectScanner` run:

```bash
dotnet build EvenPerfectScanner.sln
```

Run only the tests related to the solution you modify. Always run tests without --no-build parameter to avoid run failures due to missing assembly files. Tests should be always written using XUnit with FluentAssertions. Before executing any tests, make sure their running time is reasonable; skip test suites that are expected to take excessively long. When running tests, always provide both an explicit timeout guard and the Category=Fast filter so that only the fast suite executes.

## Making Changes in the Repository
To make any changes to the files in the repository, you should always utilize a script created for it to overcome environment / scripting issues. The script is called `apply_patch.cs` and has the following syntax:

```bash
dotnet script apply_patch.cs -- <patch-file> [target-directory]
```

> **Temporary workflow notice:** Until the pending apply_patch.cs performance and correctness TODOs are complete, do not invoke the script in this repository. Edit files directly with shell tools (sed, tee, cat, python, etc.) and stage the results via git. Re-enable the helper only after the TODO tracker explicitly calls out that the ban has been lifted.

It supports the standard `git diff` patch files as input. You should be able to run it both in the local and remote `Codex` environments. You should expect it to work both in `Windows` and `Linux` environments. Make sure that you include `diff --git` lines, headers and file prefixes in the patch files to avoid applying issues.

`apply_patch.cs` supports several parameters, depending on your needs:
* `--check` - allows checking if the patch will work without making changes to the files
* `--ignore-whitespace`
* `--ignore-eol`
* `--ignore-line-endings`
* `--target` - allows specifying target root directory for the patch
* `--profile` - capture read/parse/apply timings for each patch file.
* `--replace-line <path> <line-number> <expected-text> <new-text>` - replace a single line by absolute number (use `_` as the expected text to skip validation)

* Build patches in standard `git diff` format.
    * It supports `---`, `+++`, and `@@`.
    * It doesn't support `***` or `*** Begin/End Patch` wrappers.
    * It requires `diff --git` at the beginning.
    * Patch header needs the actual line numbers.
    * When hand-writing a diff, always include the numeric ranges in every `@@ -start,count +start,count @@` header; missing or zero counts will cause `--check` to fail immediately.
* Always check if the patch will work by using `--check` parameter, without making changes to the files, which would force patch changes. Try applying the patch only when `--check` passed.
* Build patches in small, line-precise hunks; confirm target lines with `Get-Content` or `nl -ba` before writing the diff to avoid context mismatches.
* Capture the exact start lines for each hunk while drafting the `@@ -start,length +start,length @@` header; double-check them with `nl -ba` so the first verification succeeds.
* For one-off line edits, prefer the `--replace-line <path> <line-number> <expected-text> <new-text>` option. Pass `_` as the expected text to skip validation when necessary.
* Before calling `--replace-line`, capture the target number with `nl -ba` (or an equivalent command) so the replacement lands exactly once without retrying.
* After every successful patch application, regenerate a fresh diff from the updated files before preparing the next patch so the hunk headers and context reflect the latest line numbers.
* If the patch script flags repeated mismatches, regenerate a fresh diff for the updated file instead of tweaking stale context blocks.
* When `--check` reveals that only whitespace or EOL normalization differs, rerun it with `--ignore-whitespace` and/or `--ignore-line-endings` so validation concentrates on the meaningful edits.
* Regenerate the diff whenever the file shifts—never reuse an old hunk after other edits.
* Keep patch filenames unique and delete applied files to prevent accidental re-use.
* When a hunk fails, inspect the around-lines immediately and adjust rather than retrying the same diff.
* Prefer single-purpose patches (one logical change per file) so rollbacks or fixes stay focused.
* Use `nl -ba` to show line numbers, when needed.
* Always use 4 spaces for indentation of code blocks.

If you want to make any changes, create a patch file and run the script to apply it. Always run it from the root directory and reference any modified files with relative paths respectively in the patch.

Do **not** attempt to apply the changes manually, or with `Python`, or with `PowerShell` scripts, or `cat` or any other way, unless you apply changes to `apply_patch.cs` script itself. If you identify any issues with `apply_patch.cs` script, you should propose fixes to resolve them and continue using the script after updates. If you spot any missing, but required features, propose enhancements and continue using the script after updates, too.

### Test execution time policy

- Always estimate runtime before launching any tests or long-running samples. If a test or ad-hoc run is likely to exceed 2 minutes, do not run it unless the user explicitly requests a longer run and provides justification.
- Never apply the two-minute guard to benchmark or profiling runs; allow them to finish and capture the summary unless the console log shows the job is clearly stuck.
- Do not hardcode timeouts inside test code. Enforce time limits from the runner/shell layer so developers can run tests locally without artificial constraints.
- Prefer targeted filters over full suites to keep runs short: `--filter FullyQualifiedName~<ClassOrMethod>` or `--filter "FullyQualifiedName=Namespace.Class.Method"`.
- Keep GPU-heavy tests minimal and scoped. If a kernel compile or warmup is expected to take >2 minutes, skip running it during regular iterations.
- When working on a task in a local Codex environment, do not run any existing unit tests, except for fast, new, added tests, unless you're instructed otherwise. In the task summary list the names of the recommended unit tests to validate the changes, instead.

#### How to run tests within the 2-minute limit

- Windows (PowerShell):
  - Run a single class or test with a 120s guard:
    - `$proj = "<path-to>.csproj"`
    - `$filter = 'FullyQualifiedName~SomeTests'  # or FullyQualifiedName=Namespace.Class.Method`
    - `$job = Start-Job -ScriptBlock { param($p,$f) dotnet test $p -c Debug --filter $f } -ArgumentList $proj,$filter`
    - `if (-not (Wait-Job $job -Timeout 120)) { Stop-Job $job; throw 'Test run exceeded 120s timeout.' }`
    - `Receive-Job $job`

- Linux/macOS (bash):
  - Use GNU timeout (install coreutils if needed on macOS):
    - `timeout 120s dotnet test <path-to>.csproj -c Debug --filter "Category=Fast&FullyQualifiedName~SomeTests"`
    - Always specify `timeout 120s` (or a shorter limit when appropriate) and include `Category=Fast` in every filter.
    - Note: exit code 124 indicates timeout; handle accordingly in scripts.

- Codex/agent runs: when invoked as separate steps where inline time control isn’t available, wrap calls through the shell using the guards above rather than calling `dotnet test` directly from the tool.

Examples:
- Quick GCD check: `--filter "FullyQualifiedName~PrimeTesterGcdTests"`
- Single test method: `--filter "FullyQualifiedName=EvenPerfectBitScanner.Tests.ResultsFileNameTests.BuildResultsFileName_contains_reduction_and_devices"`

Always implement the functionality to achieve the highest possible performance, even if that means allowing for some code duplication. E.g. use List<T> instead of IList<T>, IEnumerable<T> etc.; calculate modulo "%" result in the most efficient way, use aggresive inlining when appropriate and so on. Always take arrays from pools, when it's desired and possible depending on the required array size.

Don't simplify the functionality and / or algorithms to behave differently, unless you're resolving a bug that you've found. Always attempt preservint the functionality, is it's designed in iterations and there is a reason for specific behaviors / implementations.

The only .Net SDK available in Codex environment is .Net 8.0. It'll be periodically updated to newer versions as they become available. Specify target framework to the one currently available, but always target all .csproj files against all .Net versions currently supported by MS, using the latest official release. .Net 6.0 is no longer supported, so projects should target only .Net 8.0 and .Net 9.0, with 8+ being the default.

Always enter blank line after closing "}" parenthesis, unless there is a consecutive closing "}" bracket. IF's, loops, in practice all code blocks should be put in their "{ (...) }" blocks. Even if there is only a single line in such a code block. Always use the latest officially released language features. All .csproj files should define nullable = enabled and we should always use "?" and/or "!" where appropriate.

Never ever run tests marked with trait Category = Slow. Always invoke `dotnet test` (or equivalent wrappers) with both an explicit timeout and the `Category=Fast` filter; never run tests without specifying these arguments. If there are no categories assigned yet, categorize them before running or skip the execution.

There is only one agents file - this one. Don't search for another one.

Always use indentation with four space characters in source code files, even if a file currently uses tabs or appears to rely on eight-space indentation. Increase indentation level by exactly one indentation unit (four spaces) per nested block. When parameters or long expressions wrap, indent the continuation with two indentation units for clarity.
Explicitly prefer reusing existing variables once their previous values are no longer required instead of declaring new locals of the same type. Call out the variable reuse in comments when its meaning changes so the intent stays clear and register pressure remains minimal.
Do not introduce try/finally blocks solely to release pooled resources. Restructure the code so that every return path releases the resources without relying on finally unless the control flow would become unclear or unmaintainable.

Keep the divisor cycle cache limited to a single block in memory (only the on-disk snapshot). When a lookup misses, compute the required cycle on the configured device and discard the transient result instead of scheduling background generation or caching additional blocks.

All code, comments, commit messages, branch names and PR descriptions must be written in American English.

- When modifying existing files, preserve all relevant TODO notes and explanatory comments unless they are no longer valid.
- Prefer defining data holders as structs or classes with public fields over auto-properties when performance on the hot path is a concern.
- Do not add or keep methods that simply wrap another method call without adding any logic; inline those calls or extend the wrapper with meaningful work.

## Performance implementation checklist

- Prefer dedicated kernels over flag-controlled or parameterized kernels. Avoid conditional branches in kernels and tight loops by creating separate kernels when necessary.
- Remove `if` statements inside hot loops whenever possible. Duplicated code in additional kernels or methods is acceptable when it improves speed.
- Replace runtime flags in hot paths with dedicated method variants to eliminate conditional branches.
- Reuse existing variables of the same type instead of allocating new ones if some become unused. Add comments whenever a variable's meaning changes.
- Extract functionality into separate classes instead of growing existing classes. Place reusable components into `*.Core` projects.
- Rent arrays from `ArrayPool<T>` when stack allocation is impractical or less efficient, and return them after use.
- Provide cancellation-aware and cancellation-free variants of hot methods so cancellation tokens are not checked in every loop iteration; handle cancellation in outer scopes.
- Prefer `for` loops over `foreach` to reduce enumerator allocations.
- Declare loop control variables outside the loop and reuse them across loops when possible.
- Cache static or read-only arrays in local variables before looping to avoid repeated class field lookups.
- Store collection `Count` or array `Length` in a local variable before the loop and use the cached value.
- Declare variables together outside loops when feasible and reuse them instead of creating new ones.
- Avoid repeated casting within loops by precomputing or using wider variables when necessary.

## GPU-specific notes

The following checks were executed with ILGPU 1.5.3 on CPU and OpenCL accelerators:

- [x] `%` operator (kernels): supported for 32-bit and 64-bit integer types; not supported for `UInt128` inside kernels.
- [x] Integer division (kernels): supported for 32-bit and 64-bit types; not supported for `UInt128` inside kernels.
- [x] Arrays or vectors of `bool`: unsupported (`NotSupportedException`); use `byte` instead.
- [x] `UInt128` multiplication: unsupported in kernels (`AggregateException/NullReferenceException`); manual implementation required.
- [x] `UInt128` division: unsupported in kernels; manual implementation required.
- [x] `UInt128` bit shifts: attempts failed in kernels; avoid inside device code.
- [x] GPU context pooling: supported and recommended (validated in a separate program).

Additional numeric-type probe results (`tools/GpuNumericTypeProbe.csx`, ILGPU 1.5.3 on the CPU accelerator):

- `System.Numerics.BigInteger`: unsupported. Kernels that exercise addition, subtraction, multiplication, division, modulo, and `CompareTo` all fail with `InternalCompilerException`, indicating fundamental codegen gaps for the type.
- `PeterO.Numbers.EInteger`: unsupported. Attempts to compile kernels that call addition, subtraction, multiplication, division, remainder, and `CompareTo` throw `InternalCompilerException`, so none of the arithmetic helpers are currently usable on the device.
- `PeterO.Numbers.ERational`: unsupported. Kernels that call addition, subtraction, multiplication, division, and `CompareTo` trigger `InternalCompilerException`; the type also lacks a remainder/modulo API for device usage.

Patch-preparation checklist for reliable `--check` runs:

- Manually review every patch file before running `--check`; confirm the hunks reflect only the intended edits and eliminate accidental whitespace-only noise ahead of time.
- Always capture context with `nl -ba` before composing each hunk so the `@@` header line numbers match the target file on the first try.
- Ensure every hunk carries at least three unchanged context lines above and below the edits whenever the surrounding file provides them; regenerate the diff if the file changed since the previous attempt so the context stays accurate.
- Trim or normalize trailing whitespace while drafting the patch, or rely on `--ignore-whitespace` during verification when formatting noise is unavoidable, then re-run without ignores to confirm the final diff matches exactly.
- Prefer regenerating the patch from scratch whenever the working tree changes instead of editing previous diffs; stale context is the most common source of mismatches.

- The patch parser now ignores blank metadata lines between `diff --git`, `---`, and `+++` headers and skips stray blank separators between hunks, so you no longer need to insert dummy empty lines to satisfy the checker. Keep the header compact unless your diff tool emits those blank lines naturally.

### apply_patch.cs profiling snapshot (2025-10-04)

- Latest timing check: `dotnet script apply_patch.cs -- --profile --check /tmp/agents_todo_update.diff`.
- Patch characteristics: 2 files, 4 hunks, 31 lines, 3468 bytes.
- Timings: read 0.45 ms, parse 2.96 ms, apply 2.96 ms.
- Focus next on streaming the patch reader and pooled buffer reuse to reduce the parse APPLY bottleneck highlighted above.

Mersenne divisor size estimates:

- Any prime divisor `q` of a Mersenne number `2^p - 1` must satisfy `q ≡ 1 (mod 2p)`. The minimal `q` therefore appears at `q = 2p + 1` when the multiplier `k = 1` yields a prime.
- For candidate exponents `p` in the `[138,000,000, 200,000,000]` range, the minimal compliant divisor lands below `2 · 200,000,000 + 1 = 400,000,001`, which fits in 29 bits (`2^29 = 536,870,912`).
- Extending the search up to `p = 600,000,000` pushes the minimal divisor to `1,200,000,001`, requiring 31 bits (`2^31 = 2,147,483,648`).
- Larger `k` values (and thus larger `q`) are possible—up to nearly `2^p - 1`—so the heuristics and storage should plan for wider integers when deep trial factoring is necessary, but the noted bit widths cover the first admissible divisors for these exponent bands.

- Because every proper divisor of `2^p - 1` is at most `(2^p - 1) / 2`, the worst-case bit-width for any admissible `q` is `p - 1` bits (since `⌊(2^p - 1) / 2⌋ = 2^{p-1} - 1`). Consequently we need roughly `199,999,999` bits when `p ≤ 200,000,000` and `599,999,999` bits when `p ≤ 600,000,000`. In terms of the `q = 2kp + 1` form, this translates to `k_max = ⌊((2^p - 1) / 2 - 1) / (2p)⌋`, which is approximately `2^{p-2} / p` for large `p`.

Practical guidance:
- Use native `%` and `/` in kernels when working with 32-bit or 64-bit integers if it is the most efficient option for the target backend.
- For `UInt128` in kernels, avoid `%`/`/`; implement reductions via Montgomery/Barrett, or use Mersenne-specific folding for mod `2^p-1`.
- Prefer Mersenne folding over general modulo for `2^p-1` moduli to remove branches in hot loops.

- Heuristic multiply-shift benchmarks (2025-10-04):
  - Run `dotnet run -c Release --project tools/HeuristicMultiplyShiftTiming/HeuristicMultiplyShiftTiming.csproj` to compare the `ULongExtensions` helpers without the BenchmarkDotNet harness.
  - Timings below report the average nanoseconds per call over 200,000,000 iterations; all variants produced identical checksums, confirming consistent arithmetic.
    - `NearOverflowShift`: `MultiplyShiftRight` ≈ 2.37 ns, shift-first ≈ 4.71 ns, naive `(value * multiplier) >> shift` ≈ 0.85 ns.
    - `HalfRange`: `MultiplyShiftRight` ≈ 2.11 ns, shift-first ≈ 4.15 ns, naive ≈ 0.70 ns.
    - `MixedBits`: `MultiplyShiftRight` ≈ 1.43 ns, shift-first ≈ 2.83 ns, naive ≈ 0.60 ns.
    - `Sparse`: `MultiplyShiftRight` ≈ 1.57 ns, shift-first ≈ 2.84 ns, naive ≈ 0.59 ns.
  - The shift-first rewrite roughly doubles the per-call cost on CPU, but it stays within 64-bit arithmetic and is the preferred fallback for GPU heuristics when avoiding `UInt128` overflow is mandatory.
