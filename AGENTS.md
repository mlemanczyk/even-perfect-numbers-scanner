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

### Test execution time policy

- Always estimate runtime before launching any tests or long-running samples. If a test or ad-hoc run is likely to exceed 2 minutes, do not run it unless the user explicitly requests a longer run and provides justification.
- Do not hardcode timeouts inside test code. Enforce time limits from the runner/shell layer so developers can run tests locally without artificial constraints.
- Prefer targeted filters over full suites to keep runs short: `--filter FullyQualifiedName~<ClassOrMethod>` or `--filter "FullyQualifiedName=Namespace.Class.Method"`.
- Keep GPU-heavy tests minimal and scoped. If a kernel compile or warmup is expected to take >2 minutes, skip running it during regular iterations.

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

Keep the divisor cycle cache limited to a single block in memory (only the on-disk snapshot). When a lookup misses, compute the required cycle on the configured device and discard the transient result instead of scheduling background generation or caching additional blocks.

All code, comments, commit messages, branch names and PR descriptions must be written in American English.

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

Practical guidance:
- Use native `%` and `/` in kernels when working with 32-bit or 64-bit integers if it is the most efficient option for the target backend.
- For `UInt128` in kernels, avoid `%`/`/`; implement reductions via Montgomery/Barrett, or use Mersenne-specific folding for mod `2^p-1`.
- Prefer Mersenne folding over general modulo for `2^p-1` moduli to remove branches in hot loops.
