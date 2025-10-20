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
Previous revisions attached a per-prime factorization cache, but profiling showed virtually no reuse and the added bookkeeping slowed the scan. The implementation now recomputes every factorization on demand and documents why the cache should not return. 【F:PerfectNumbers.Core/Cpu/MersenneNumberDivisorByDivisorCpuTester.cs†L152-L221】【F:PerfectNumbers.Core/MersenneDivisorCycles.cs†L308-L351】

* **Current behavior:** Each divisor recomputes the factorization for the exponent and `k` values using thread-static scratch dictionaries only. Inline comments note that the retired cache produced zero hits in production runs. 【F:PerfectNumbers.Core/MersenneDivisorByDivisorCpuTester.cs†L150-L204】【F:PerfectNumbers.Core/MersenneDivisorCycles.cs†L332-L347】
* **Rationale:** The divisor stream never repeats values during a scan, so caching results provided no benefit while increasing synchronization and pooling overhead. The comments exist to prevent reintroducing the pointless cache. 【F:PerfectNumbers.Core/MersenneDivisorCycles.cs†L336-L343】

### Factorization helper pools
The factorization helpers rely on thread-static pools for temporary dictionaries, reusable factor entries, and key-value buffers. These pools only recycle storage; they deliberately clear their contents before handing objects to the next caller. 【F:PerfectNumbers.Core/ThreadStaticPools.cs†L270-L316】

* **Observed reuse:** The pools successfully avoid new allocations, but they do not provide value-based caching. All number-theoretic results are recomputed for every divisor except the cached exponent entry described above.

### Primality probe cache
`PrimeOrderCalculator` only retains primality results for inputs at or below the four-million snapshot threshold, matching the strict small-factor limit. Larger values—including the monotonically increasing `p` and `q` candidates—skip the dictionary entirely, so the cache only serves repeatedly observed small factors uncovered while factoring `k`. 【F:PerfectNumbers.Core/PrimeOrderCalculator.Cpu.cs†L676-L704】

* **Observed reuse:** The bounded cache keeps quick answers for recurring small primes such as `3` or `5`, while unique large composites are evaluated once and discarded. This prevents the cache from growing with one-off exponent or divisor candidates and aligns its scope with the actual reuse opportunities.

## Summary
With the factorization cache removed, the bounded ≤4,000,000 primality dictionary remains the only meaningful reuse on the CPU by-divisor path. All other caches either hold a single most-recent value, serve small divisors that never appear in this workload, or are limited to memory pooling without retaining computed results. The fallback that bypasses the prime-order solver still funnels every large divisor into a fresh cycle computation, so no additional reuse occurs there. Every branch of the factorization workflow—including the direct prime-order success case, the fallback that invokes `DivisorCycleCache`, and the recursive Pollard–Rho reducer—was reviewed in code; no remaining cache-related areas of this path require inspection.
