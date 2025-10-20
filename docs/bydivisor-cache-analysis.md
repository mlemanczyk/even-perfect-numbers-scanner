# CPU by-divisor cache reuse analysis

## Scope
The EvenPerfectBitScanner switches to the by-divisor implementation when the CLI selects `--mersenne=bydivisor`. In CPU-only runs it instantiates `MersenneNumberDivisorByDivisorCpuTester` and feeds it the primes loaded from the filter file together with any previously recorded outcomes. 【F:EvenPerfectBitScanner/Program.cs†L216-L305】

Each candidate exponent `p` is a prime at least `138,000,000`, the divisors tested for that exponent are generated as `q = 2·k·p + 1` with strictly increasing integer `k ≥ 1`, and both the prime and divisor streams are monotonically increasing with no repetitions. The small-cycle snapshot bundled with the binaries only covers divisors up to four million. 【F:PerfectNumbers.Core/Cpu/MersenneNumberDivisorByDivisorCpuTester.cs†L179-L214】【F:PerfectNumbers.Core/PerfectNumberConstants.cs†L3-L16】

## Caches touched in this path

### Previous results skip map
When the results file already exists, the loader records every stored `p` in a dictionary so the new run can skip those primes entirely. 【F:EvenPerfectBitScanner/Program.cs†L286-L305】

* **Observed reuse:** Entries are only revisited if the user reruns the scan on the same results file. Within a single execution the filter contains each prime at most once, so the dictionary does not experience repeated lookups for identical keys beyond that replay scenario.

### Montgomery divisor data creation
Every admissible divisor constructs its Montgomery reduction constants directly before evaluating the cycle length, both in the per-divisor loop and when the reusable scan session validates cached state. 【F:PerfectNumbers.Core/Cpu/MersenneNumberDivisorByDivisorCpuTester.cs†L188-L218】【F:PerfectNumbers.Core/Cpu/MersenneCpuDivisorScanSession.cs†L27-L69】【F:PerfectNumbers.Core/MontgomeryDivisorData.cs†L32-L45】

* **Observed reuse:** The helper still runs for every divisor, but because `q = 2·k·p + 1` never repeats, no value-based caching is retained. The direct construction simply reflects that reusing previously computed constants provided no benefit on this path.

### Divisor cycle cache
If the dedicated cycle solver produces a result, it is returned immediately. Otherwise the code falls back to `DivisorCycleCache.GetCycleLength` with `skipPrimeOrderHeuristic=true`. 【F:PerfectNumbers.Core/Cpu/MersenneNumberDivisorByDivisorCpuTester.cs†L190-L218】 This cache stores a preloaded snapshot for divisors up to four million and computes larger cycles on demand. 【F:PerfectNumbers.Core/DivisorCycleCache.cs†L32-L80】

* **Observed reuse:** The tester only queries the snapshot for divisors at or below four million; larger values invoke `MersenneDivisorCycles.CalculateCycleLength` directly. The by-divisor workload never emits small divisors, so the cache is effectively idle and every large divisor triggers a fresh computation without persisting the result.

### Factorization cache
Previous revisions attached a per-prime factorization cache, but profiling showed virtually no reuse and the added bookkeeping slowed the scan. The implementation now recomputes every factorization on demand and documents why the cache should not return. 【F:PerfectNumbers.Core/Cpu/MersenneNumberDivisorByDivisorCpuTester.cs†L101-L264】【F:PerfectNumbers.Core/MersenneDivisorCycles.cs†L305-L399】

* **Current behavior:** Each divisor recomputes the factorization for the exponent and `k` values using thread-static scratch dictionaries only. Inline comments note that the retired cache produced zero hits in production runs. 【F:PerfectNumbers.Core/Cpu/MersenneNumberDivisorByDivisorCpuTester.cs†L101-L264】【F:PerfectNumbers.Core/MersenneDivisorCycles.cs†L321-L399】
* **Instrumentation:** `MersenneNumberDivisorByDivisorCpuTester` and `MersenneDivisorCycles` keep static tracking sets that log to the console whenever a repeated exponent appears, incrementing a shared hit counter. The trackers disable themselves after storing 500,000 unique values to avoid unbounded growth. 【F:PerfectNumbers.Core/Cpu/MersenneNumberDivisorByDivisorCpuTester.cs†L15-L125】【F:PerfectNumbers.Core/Cpu/MersenneNumberDivisorByDivisorCpuTester.cs†L266-L301】【F:PerfectNumbers.Core/MersenneDivisorCycles.cs†L18-L39】【F:PerfectNumbers.Core/MersenneDivisorCycles.cs†L321-L369】
* **Rationale:** The divisor stream never repeats values during a scan, so caching results provided no benefit while increasing synchronization and pooling overhead. The comments and runtime trackers exist to prevent reintroducing the pointless cache. 【F:PerfectNumbers.Core/MersenneDivisorCycles.cs†L321-L399】

### Factorization helper pools
The factorization helpers rely on thread-static pools for temporary dictionaries, reusable factor entries, and key-value buffers. These pools only recycle storage; they deliberately clear their contents before handing objects to the next caller. 【F:PerfectNumbers.Core/ThreadStaticPools.cs†L270-L316】

* **Observed reuse:** The pools successfully avoid new allocations, but they do not provide value-based caching. All number-theoretic results are recomputed for every divisor except the cached exponent entry described above.

## Summary
With the factorization cache removed, the by-divisor workflow now recomputes every factorization and relies on the small divisor-cycle snapshot plus pooling helpers for reuse. Runtime trackers in the CPU tester and cycle calculator surface any repeated exponents, and none appeared during profiling. Every branch of the factorization workflow—including the direct prime-order success case, the fallback that invokes `DivisorCycleCache`, and the recursive Pollard–Rho reducer—was reviewed in code; no remaining cache-related areas of this path require inspection. 【F:PerfectNumbers.Core/Cpu/MersenneNumberDivisorByDivisorCpuTester.cs†L15-L125】【F:PerfectNumbers.Core/Cpu/MersenneNumberDivisorByDivisorCpuTester.cs†L266-L301】【F:PerfectNumbers.Core/MersenneDivisorCycles.cs†L18-L39】【F:PerfectNumbers.Core/MersenneDivisorCycles.cs†L321-L369】
