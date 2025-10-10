using System.Collections.Concurrent;
using System.Numerics;

namespace PerfectNumbers.Core.Cpu;

public sealed class MersenneNumberDivisorByDivisorCpuTester : IMersenneNumberDivisorByDivisorTester
{
    private readonly object _sync = new();
    private readonly ConcurrentBag<DivisorScanSession> _sessionPool = [];
    private ulong _divisorLimit;
    private ulong _lastStatusDivisor;
    private bool _isConfigured;
    private int _batchSize = 1_024;


    public int BatchSize
    {
        get => _batchSize;
        set => _batchSize = Math.Max(1, value);
    }

    public void ConfigureFromMaxPrime(ulong maxPrime)
    {
        lock (_sync)
        {
            _divisorLimit = ComputeDivisorLimitFromMaxPrime(maxPrime);
            _lastStatusDivisor = 0UL;
            _isConfigured = true;
        }
    }

    public ulong DivisorLimit
    {
        get
        {
            lock (_sync)
            {
                if (!_isConfigured)
                {
                    throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
                }

                return _divisorLimit;
            }
        }
    }

    public ulong GetAllowedMaxDivisor(ulong prime)
    {
        lock (_sync)
        {
            if (!_isConfigured)
            {
                throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
            }

            return ComputeAllowedMaxDivisor(prime, _divisorLimit);
        }
    }

    public bool IsPrime(ulong prime, out bool divisorsExhausted)
    {
        ulong allowedMax;
        lock (_sync)
        {
            if (!_isConfigured)
            {
                throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
            }

            allowedMax = ComputeAllowedMaxDivisor(prime, _divisorLimit);
        }

        if (allowedMax < 3UL)
        {
            divisorsExhausted = true;
            return true;
        }

        ulong processedCount = 0UL;
        ulong lastProcessed = 0UL;
        bool processedAll = false;

        bool composite = CheckDivisors(
            prime,
            allowedMax,
            out lastProcessed,
            out processedAll,
            out processedCount);

        if (processedCount > 0UL)
        {
            lock (_sync)
            {
                UpdateStatusUnsafe(processedCount);
            }
        }

        if (composite)
        {
            divisorsExhausted = true;
            return false;
        }

        divisorsExhausted = processedAll || composite;
        return true;
    }

    public void PrepareCandidates(ReadOnlySpan<ulong> primes, Span<ulong> allowedMaxValues)
    {
        if (allowedMaxValues.Length < primes.Length)
        {
            throw new ArgumentException("allowedMaxValues span must be at least as long as primes span.", nameof(allowedMaxValues));
        }

        ulong divisorLimit;

        lock (_sync)
        {
            if (!_isConfigured)
            {
                throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
            }

            divisorLimit = _divisorLimit;
        }

        for (int index = 0; index < primes.Length; index++)
        {
            allowedMaxValues[index] = ComputeAllowedMaxDivisor(primes[index], divisorLimit);
        }
    }

    public IMersenneNumberDivisorByDivisorTester.IDivisorScanSession CreateDivisorSession()
    {
        lock (_sync)
        {
            if (!_isConfigured)
            {
                throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
            }

            if (_sessionPool.TryTake(out DivisorScanSession? session))
            {
                session.Reset();
                return session;
            }

            return new DivisorScanSession(this);
        }
    }

    private void ReturnSession(DivisorScanSession session)
    {
        _sessionPool.Add(session);
    }

    private bool CheckDivisors(
        ulong prime,
        ulong allowedMax,
        out ulong lastProcessed,
        out bool processedAll,
        out ulong processedCount)
    {
        lastProcessed = 0UL;
        processedCount = 0UL;
        processedAll = false;

        if (allowedMax < 3UL)
        {
            return false;
        }

        UInt128 step = (UInt128)prime << 1;
        if (step == UInt128.Zero)
        {
            processedAll = true;
            return false;
        }

        UInt128 limit = allowedMax;
        UInt128 divisor = step + UInt128.One;
        if (divisor > limit)
        {
            processedAll = true;
            return false;
        }

        Dictionary<ulong, MersenneDivisorCycles.FactorCacheEntry>? factorCache = null;
        DivisorCycleCache cycleCache = DivisorCycleCache.Shared;

        byte step10 = (byte)(step % 10UL);
        byte step8 = (byte)(step % 8UL);
        byte step5 = (byte)(step % 5UL);
        byte step3 = (byte)(step % 3UL);
        byte step7 = (byte)(step % 7UL);
        byte step11 = (byte)(step % 11UL);

        byte remainder10 = (byte)(divisor % 10UL);
        byte remainder8 = (byte)(divisor % 8UL);
        byte remainder5 = (byte)(divisor % 5UL);
        byte remainder3 = (byte)(divisor % 3UL);
        byte remainder7 = (byte)(divisor % 7UL);
        byte remainder11 = (byte)(divisor % 11UL);

        // Keep the divisibility filters aligned with the divisor-cycle generator so the
        // CPU path never requests cycles that were skipped during cache creation.
        bool lastIsSeven = (prime & 3UL) == 3UL;
        bool hasExactMersenne;
        ulong mersenne;
        if (prime <= 63UL)
        {
			mersenne = (1UL << (int)prime) - 1UL;
            hasExactMersenne = true;
        }
        else if (prime == 64UL)
        {
            mersenne = ulong.MaxValue;
            hasExactMersenne = true;
        }
        else
        {
            mersenne = 0UL;
            hasExactMersenne = false;
        }

        while (divisor <= limit)
        {
            ulong candidate = (ulong)divisor;
            processedCount++;
            lastProcessed = candidate;

            bool admissible = lastIsSeven
                ? (remainder10 == 3 || remainder10 == 7 || remainder10 == 9)
                : (remainder10 == 1 || remainder10 == 3 || remainder10 == 9);

            if (admissible && (remainder8 == 1 || remainder8 == 7) && remainder3 != 0 && remainder5 != 0 && remainder7 != 0 && remainder11 != 0)
            {
                MontgomeryDivisorData divisorData = MontgomeryDivisorDataCache.Get(candidate);
                ulong divisorCycle;

                if (candidate <= PerfectNumberConstants.MaxQForDivisorCycles)
                {
                    divisorCycle = cycleCache.GetCycleLength(candidate);
                }
                else
                {
                    factorCache ??= new Dictionary<ulong, MersenneDivisorCycles.FactorCacheEntry>(8);
                    if (!MersenneDivisorCycles.TryCalculateCycleLengthForExponentCpu(candidate, prime, divisorData, factorCache, out divisorCycle) || divisorCycle == 0UL)
                    {
                        divisorCycle = cycleCache.GetCycleLength(candidate);
                    }
                }

                if (divisorCycle == prime)
                {
                    // A cycle equal to the tested exponent (which is prime in this path)
                    // guarantees that the candidate divides the corresponding Mersenne
                    // number because the order of 2 modulo the divisor is exactly p.
                    processedAll = true;
                    return true;
                }

                if (divisorCycle == 0UL)
                {
                    Console.WriteLine($"Divisor cycle was not calculated for {prime}");
                }
            }

            ulong increments = 1UL;
            if (divisor > UInt128.One)
            {
                ulong skip = ComputeBitSkipIncrements(
                    step,
                    divisor,
                    limit,
                    hasExactMersenne,
                    mersenne);
                if (skip > 1UL)
                {
                    increments = skip;
                }
            }

            UInt128 incrementAmount = step * increments;
            divisor += incrementAmount;
            remainder10 = AdvanceRemainder(remainder10, step10, 10, increments);
            remainder8 = AdvanceRemainder(remainder8, step8, 8, increments);
            remainder5 = AdvanceRemainder(remainder5, step5, 5, increments);
            remainder3 = AdvanceRemainder(remainder3, step3, 3, increments);
            remainder7 = AdvanceRemainder(remainder7, step7, 7, increments);
            remainder11 = AdvanceRemainder(remainder11, step11, 11, increments);
        }

        processedAll = divisor > limit;
        return false;
    }

    private static byte CheckDivisor(ulong prime, ulong divisorCycle, in MontgomeryDivisorData divisorData)
    {
        ulong residue = prime.Pow2MontgomeryModWithCycleCpu(divisorCycle, divisorData);
        return residue == 1UL ? (byte)1 : (byte)0;
    }

    private static byte AddMod(byte value, byte delta, byte modulus)
    {
        int sum = value + delta;
        if (sum >= modulus)
        {
            sum -= modulus;
        }

        return (byte)sum;
    }

    private static byte AdvanceRemainder(byte remainder, byte delta, byte modulus, ulong increments)
    {
        if (increments == 0UL || modulus == 0)
        {
            return remainder;
        }

        ulong modulusValue = modulus;
        ulong scaledDelta = (delta * (increments % modulusValue)) % modulusValue;
        byte reducedDelta = (byte)scaledDelta;
        int sum = remainder + reducedDelta;
        if (sum >= modulus)
        {
            sum -= modulus;
        }

        return (byte)sum;
    }

    private static ulong ComputeBitSkipIncrements(
        UInt128 step,
        UInt128 divisor,
        UInt128 limit,
        bool hasExactMersenne,
        ulong mersenne)
    {
        if (step == UInt128.Zero || divisor <= UInt128.One || limit <= divisor || limit <= UInt128.One)
        {
            return 1UL;
        }

        UInt128 maxK = (limit - UInt128.One) / step;
        if (maxK == UInt128.Zero)
        {
            return 1UL;
        }

        UInt128 currentK = (divisor - UInt128.One) / step;
        if (currentK >= maxK)
        {
            return 1UL;
        }

        if (hasExactMersenne && divisor <= (UInt128)ulong.MaxValue && step <= (UInt128)ulong.MaxValue)
        {
            ulong divisor64 = (ulong)divisor;
            ulong step64 = (ulong)step;
            if (step64 == 0UL)
            {
                return 1UL;
            }

            ulong quotient = mersenne / divisor64;
            if (quotient <= 1UL)
            {
                return 1UL;
            }

            int bitLength = BitOperations.Log2(quotient) + 1;
            if (bitLength <= 1)
            {
                return 1UL;
            }

            int shift = bitLength - 1;
            if (shift >= 64)
            {
                return 1UL;
            }

            ulong denominator = 1UL << shift;
            if (denominator == 0UL)
            {
                return 1UL;
            }

            ulong baseBound = mersenne / denominator;
            UInt128 minDivisor = (UInt128)baseBound + UInt128.One;
            if (minDivisor <= divisor)
            {
                return 1UL;
            }

            UInt128 minNumerator = minDivisor - UInt128.One;
            UInt128 targetK = DivideRoundUp128(minNumerator, step);
            if (targetK <= currentK + UInt128.One || targetK > maxK)
            {
                return 1UL;
            }

            UInt128 increments128 = targetK - currentK;
            if (increments128 <= UInt128.One || increments128 > (UInt128)ulong.MaxValue)
            {
                return 1UL;
            }

            return (ulong)increments128;
        }

        int divisorLog = GetLog2(divisor);
        int nextBitIndex = divisorLog + 1;
        if (nextBitIndex >= 128)
        {
            return 1UL;
        }

        UInt128 nextPower = UInt128.One << nextBitIndex;
        if (nextPower <= divisor)
        {
            return 1UL;
        }

        UInt128 delta = nextPower - divisor;
        UInt128 incrementsCandidate = DivideRoundUp128(delta, step);
        if (incrementsCandidate <= UInt128.One)
        {
            return 1UL;
        }

        UInt128 remainingK = maxK - currentK;
        if (incrementsCandidate > remainingK)
        {
            return 1UL;
        }

        UInt128 projectedDivisor = divisor + step * incrementsCandidate;
        if (projectedDivisor > limit)
        {
            return 1UL;
        }

        if (incrementsCandidate > (UInt128)ulong.MaxValue)
        {
            return 1UL;
        }

        return (ulong)incrementsCandidate;
    }

    private static int GetLog2(UInt128 value)
    {
        if (value <= (UInt128)ulong.MaxValue)
        {
            return BitOperations.Log2((ulong)value);
        }

        ulong upper = (ulong)(value >> 64);
        if (upper == 0UL)
        {
            return BitOperations.Log2((ulong)value);
        }

        return 64 + BitOperations.Log2(upper);
    }

    private static UInt128 DivideRoundUp128(UInt128 numerator, UInt128 divisor)
    {
        if (divisor == UInt128.Zero)
        {
            return UInt128.MaxValue;
        }

        if (numerator == UInt128.Zero)
        {
            return UInt128.Zero;
        }

        UInt128 quotient = numerator / divisor;
        UInt128 remainder = numerator % divisor;
        if (remainder == UInt128.Zero)
        {
            return quotient;
        }

        if (quotient == UInt128.MaxValue)
        {
            return UInt128.MaxValue;
        }

        return quotient + UInt128.One;
    }

    private static ulong DivideRoundUp128(UInt128 numerator, ulong divisor)
    {
        if (numerator == UInt128.Zero)
        {
            return 0UL;
        }

        if (divisor <= 1UL)
        {
            if (divisor == 1UL)
            {
                if (numerator > (UInt128)ulong.MaxValue)
                {
                    return ulong.MaxValue;
                }

                return (ulong)numerator;
            }

            return ulong.MaxValue;
        }

        UInt128 quotient = numerator / divisor;
        UInt128 remainder = numerator % divisor;

        if (quotient > (UInt128)ulong.MaxValue)
        {
            return ulong.MaxValue;
        }

        ulong result = (ulong)quotient;
        if (remainder != UInt128.Zero)
        {
            if (result == ulong.MaxValue)
            {
                return ulong.MaxValue;
            }

            result++;
        }

        return result;
    }

    private static ulong DivideRoundUp(ulong value, ulong divisor)
    {
        if (divisor == 0UL)
        {
            return ulong.MaxValue;
        }

        ulong quotient = value / divisor;
        if (value % divisor != 0UL)
        {
            quotient++;
        }

        return quotient;
    }

    private void UpdateStatusUnsafe(ulong processedCount)
    {
        if (processedCount == 0UL)
        {
            return;
        }

        ulong interval = PerfectNumberConstants.ConsoleInterval;
        if (interval == 0UL)
        {
            _lastStatusDivisor = 0UL;
            return;
        }

        ulong total = _lastStatusDivisor + processedCount;
        if (total >= interval)
        {
            do
            {
                total -= interval;
            }
            while (total >= interval);
        }

        _lastStatusDivisor = total;
    }

    private static ulong ComputeDivisorLimitFromMaxPrime(ulong maxPrime)
    {
        if (maxPrime <= 1UL)
        {
            return 0UL;
        }

        if (maxPrime - 1UL >= 64UL)
        {
            return ulong.MaxValue;
        }

        return (1UL << (int)(maxPrime - 1UL)) - 1UL;
    }

    private static ulong ComputeAllowedMaxDivisor(ulong prime, ulong divisorLimit)
    {
        if (prime <= 1UL)
        {
            return 0UL;
        }

        if (prime - 1UL >= 64UL)
        {
            return divisorLimit;
        }

        return Math.Min((1UL << (int)(prime - 1UL)) - 1UL, divisorLimit);
    }

    private sealed class DivisorScanSession : IMersenneNumberDivisorByDivisorTester.IDivisorScanSession
    {
        private readonly MersenneNumberDivisorByDivisorCpuTester _owner;
        private readonly Dictionary<ulong, PrimeDivisorState> _primeStates = new();
        private bool _disposed;

        internal DivisorScanSession(MersenneNumberDivisorByDivisorCpuTester owner)
        {
            _owner = owner;
        }

        internal void Reset()
        {
            _disposed = false;
            _primeStates.Clear();
        }

        public void CheckDivisor(ulong divisor, in MontgomeryDivisorData divisorData, ulong divisorCycle, ReadOnlySpan<ulong> primes, Span<byte> hits)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(DivisorScanSession));
            }

            int length = primes.Length;
            if (length == 0)
            {
                return;
            }

            MontgomeryDivisorData cachedData = divisorData;
            if (cachedData.Modulus != divisor)
            {
                bool reusedDivisorData = false;
                if (length != 0)
                {
                    for (int stateIndex = 0; stateIndex < length; stateIndex++)
                    {
                        ulong statePrime = primes[stateIndex];
                        if (!_primeStates.TryGetValue(statePrime, out PrimeDivisorState cachedState) || !cachedState.HasState)
                        {
                            continue;
                        }

                        if (cachedState.LastDivisor != divisor)
                        {
                            continue;
                        }

                        cachedData = new MontgomeryDivisorData(
                            divisor,
                            cachedState.NPrime,
                            cachedState.MontgomeryOne,
                            cachedState.MontgomeryTwo,
                            cachedState.MontgomeryTwoSquared);
                        reusedDivisorData = true;
                        break;
                    }
                }

                if (!reusedDivisorData)
                {
                    cachedData = MontgomeryDivisorDataCache.Get(divisor);
                }
            }

            if (divisorCycle == 0UL)
            {
                divisorCycle = DivisorCycleCache.Shared.GetCycleLength(divisor);
                if (divisorCycle == 0UL)
                {
                    hits.Clear();
                    return;
                }
            }

            var cycleStepper = new CycleRemainderStepper(divisorCycle);
            ExponentRemainderStepper exponentStepper = default;
            bool exponentStepperInitialized = false;
            byte hitValue = 0; // Reused for every prime processed in this batch.

            bool EnsureExponentStepperInitialized(ulong currentDivisor, PrimeDivisorState? stateOverride)
            {
                if (exponentStepperInitialized)
                {
                    return true;
                }

                if (stateOverride.HasValue && stateOverride.Value.HasState && stateOverride.Value.LastDivisor == currentDivisor)
                {
                    PrimeDivisorState stateValue = stateOverride.Value;
                    var stateData = new MontgomeryDivisorData(
                        currentDivisor,
                        stateValue.NPrime,
                        stateValue.MontgomeryOne,
                        stateValue.MontgomeryTwo,
                        stateValue.MontgomeryTwoSquared);
                    exponentStepper = new ExponentRemainderStepper(stateData);
                }
                else
                {
                    exponentStepper = new ExponentRemainderStepper(cachedData);
                }

                if (!exponentStepper.IsValidModulus)
                {
                    return false;
                }

                exponentStepperInitialized = true;
                return true;
            }

            bool TrySynchronizeExponentStepper(ulong exponent, ulong currentDivisor, ref PrimeDivisorState state, out byte hitFromState)
            {
                hitFromState = 0;
                if (!state.HasState || state.LastDivisor != currentDivisor)
                {
                    return false;
                }

                EnsureMontgomeryResidue(exponent, currentDivisor, ref state);

                if (!EnsureExponentStepperInitialized(currentDivisor, state))
                {
                    return false;
                }

                if (!exponentStepper.TryInitializeFromMontgomeryResult(exponent, state.MontgomeryResidue, out bool isUnity))
                {
                    return false;
                }

                hitFromState = isUnity ? (byte)1 : (byte)0;
                return true;
            }

            ulong remainder = cycleStepper.Initialize(primes[0]);
            if (remainder == 0UL)
            {
                if (TryComputeAnalyticHit(primes[0], divisor, cachedData, out hitValue, out PrimeDivisorState analyticState))
                {
                    hits[0] = hitValue;
                }
                else
                {
                    if (!analyticState.HasState && _primeStates.TryGetValue(primes[0], out analyticState))
                    {
                        // Reusing analyticState to reflect the cached snapshot for seeding the exponent stepper.
                    }

                    if (TrySynchronizeExponentStepper(primes[0], divisor, ref analyticState, out byte synchronizedHit))
                    {
                        hits[0] = synchronizedHit;
                    }
                    else
                    {
                        // Rely on the zero cycle remainder to mark the divisor as a factor and drop any cached snapshot.
                        hits[0] = 1;
                        _primeStates.Remove(primes[0]);
                    }
                }
            }
            else
            {
                hits[0] = 0;
            }

            for (int i = 1; i < length; i++)
            {
                remainder = cycleStepper.ComputeNext(primes[i]);
                if (remainder != 0UL)
                {
                    hits[i] = 0;
                    continue;
                }

                if (TryComputeAnalyticHit(primes[i], divisor, cachedData, out hitValue, out PrimeDivisorState analyticState))
                {
                    hits[i] = hitValue;
                    continue;
                }

                if (!analyticState.HasState && _primeStates.TryGetValue(primes[i], out analyticState))
                {
                    // Reusing analyticState to hydrate the cached snapshot before invoking the exponent stepper.
                }

                if (TrySynchronizeExponentStepper(primes[i], divisor, ref analyticState, out byte synchronizedHit))
                {
                    hits[i] = synchronizedHit;
                    continue;
                }

                // Zero cycle remainder implies the divisor's order matches the exponent, so record the hit and discard stale state.
                hits[i] = 1;
                _primeStates.Remove(primes[i]);
            }
        }


        
        private bool TryComputeAnalyticHit(ulong prime, ulong divisor, in MontgomeryDivisorData divisorData, out byte hit, out PrimeDivisorState updatedState)
        {
            hit = 0;
            updatedState = default;
        
            if (!TryGetMersenneMinusOne(prime, out ulong mersenne))
            {
                return false;
            }
        
            if (divisor <= 1UL)
            {
                return false;
            }
        
            ulong step = prime << 1;
            if (step == 0UL)
            {
                return false;
            }
        
            if (!_primeStates.TryGetValue(prime, out PrimeDivisorState state) || !state.HasState)
            {
                state = CreatePrimeState(step, divisor, mersenne, divisorData);
                // Reusing state to store the freshly initialized snapshot for this prime.
                _primeStates[prime] = state;
                updatedState = state;
                hit = state.Remainder == 0UL ? (byte)1 : (byte)0;
                return true;
            }
        
            if (state.Step != step || divisor <= state.LastDivisor)
            {
                state = CreatePrimeState(step, divisor, mersenne, divisorData);
                // Reusing state to replace the stale snapshot because the divisor progression reset.
                _primeStates[prime] = state;
                updatedState = state;
                hit = state.Remainder == 0UL ? (byte)1 : (byte)0;
                return true;
            }
        
            ulong difference = divisor - state.LastDivisor;
            if (difference == 0UL)
            {
                updatedState = state;
                hit = state.Remainder == 0UL ? (byte)1 : (byte)0;
                return true;
            }
        
            if (difference % step != 0UL)
            {
                state = CreatePrimeState(step, divisor, mersenne, divisorData);
                // Reusing state again to recover from a non-uniform divisor increment.
                _primeStates[prime] = state;
                updatedState = state;
                hit = state.Remainder == 0UL ? (byte)1 : (byte)0;
                return true;
            }
        
            ulong increments = difference / step;
            ulong quotient = state.Quotient;
            ulong remainder = state.Remainder;
            ulong currentDivisor = state.LastDivisor;
            ulong reciprocal = state.Reciprocal;
            MontgomeryDivisorData divisorDataCopy = divisorData;
            PrimeDivisorState localUpdatedState = default;
            byte localHit = 0;

            bool ResetState()
            {
                state = CreatePrimeState(step, divisor, mersenne, divisorDataCopy);
                // Reusing state to rebuild the analytic snapshot after the fast path failed.
                _primeStates[prime] = state;
                localUpdatedState = state;
                localHit = state.Remainder == 0UL ? (byte)1 : (byte)0;
                return true;
            }

            bool TryComputeDelta(
                UInt128 numeratorValue,
                ulong divisorValue,
                ref ulong reciprocalValue,
                bool allowExactFallback,
                out ulong deltaValue,
                out UInt128 compensatedProductValue)
            {
                deltaValue = 0UL;
                compensatedProductValue = UInt128.Zero;

                if (numeratorValue == UInt128.Zero)
                {
                    return true;
                }

                ulong workingReciprocal = reciprocalValue;
                UInt128 workingProduct = UInt128.Zero;
                ulong candidateDelta = DivideRoundUpUsingReciprocal(numeratorValue, divisorValue, workingReciprocal);
                if (candidateDelta == ulong.MaxValue)
                {
                    workingReciprocal = ComputeReciprocalEstimate(divisorValue);
                    candidateDelta = DivideRoundUpUsingReciprocal(numeratorValue, divisorValue, workingReciprocal);

                    if (candidateDelta == ulong.MaxValue)
                    {
                        ulong refinedReciprocal = RefineReciprocalEstimate(workingReciprocal, divisorValue);
                        if (refinedReciprocal != 0UL && refinedReciprocal != ulong.MaxValue)
                        {
                            workingReciprocal = refinedReciprocal;
                            candidateDelta = DivideRoundUpUsingReciprocal(numeratorValue, divisorValue, workingReciprocal);
                        }
                    }
                }

                if (candidateDelta == ulong.MaxValue)
                {
                    if (!allowExactFallback
                        || !TryComputeDeltaExact(numeratorValue, divisorValue, ref workingReciprocal, out candidateDelta, out workingProduct))
                    {
                        return false;
                    }
                }
                else
                {
                    workingProduct = (UInt128)candidateDelta * divisorValue;
                }

                reciprocalValue = workingReciprocal;
                compensatedProductValue = workingProduct;
                deltaValue = candidateDelta;
                return true;
            }

            bool TryComputeDeltaExact(
                UInt128 numeratorValue,
                ulong divisorValue,
                ref ulong reciprocalValue,
                out ulong deltaValue,
                out UInt128 compensatedProductValue)
            {
                deltaValue = DivideRoundUp128(numeratorValue, divisorValue);
                compensatedProductValue = UInt128.Zero;
                if (deltaValue == ulong.MaxValue)
                {
                    return false;
                }

                compensatedProductValue = (UInt128)deltaValue * divisorValue;

                ulong refreshedReciprocal = ComputeReciprocalEstimate(divisorValue);
                if (refreshedReciprocal != 0UL && refreshedReciprocal != ulong.MaxValue)
                {
                    ulong refinedReciprocal = RefineReciprocalEstimate(refreshedReciprocal, divisorValue);
                    if (refinedReciprocal != 0UL && refinedReciprocal != ulong.MaxValue)
                    {
                        reciprocalValue = refinedReciprocal;
                    }
                    else
                    {
                        reciprocalValue = refreshedReciprocal;
                    }
                }

                return true;
            }

            for (ulong index = 0UL; index < increments; index++)
            {
                ulong nextDivisor = currentDivisor + step;
                reciprocal = RefineReciprocalEstimate(reciprocal, nextDivisor);

                UInt128 stepTimesQuotient = (UInt128)step * quotient;
                ulong previousRemainder = remainder;
                ulong delta = 0UL;
                UInt128 compensatedProduct = UInt128.Zero;
                if (stepTimesQuotient > remainder)
                {
                    UInt128 numerator = stepTimesQuotient - previousRemainder;
                    if (!TryComputeDelta(numerator, nextDivisor, ref reciprocal, true, out delta, out compensatedProduct))
                    {
                        // Reusing state to rebuild the analytic snapshot after the fast path failed to produce a delta.
                        ResetState();
                        updatedState = localUpdatedState;
                        hit = localHit;
                        return true;
                    }

                    if (delta > quotient)
                    {
                        if (!TryComputeDeltaExact(numerator, nextDivisor, ref reciprocal, out delta, out compensatedProduct)
                            || delta > quotient)
                        {
                            // Rebuilding the analytic snapshot after the analytic delta exceeded the cached quotient.
                            ResetState();
                            updatedState = localUpdatedState;
                            hit = localHit;
                            return true;
                        }
                    }

                    UInt128 reducedRemainder = compensatedProduct - numerator;
                    if (reducedRemainder >= nextDivisor)
                    {
                        if (!TryComputeDeltaExact(numerator, nextDivisor, ref reciprocal, out delta, out compensatedProduct))
                        {
                            // Reusing state to rebuild the analytic snapshot after the remainder correction failed.
                            ResetState();
                            updatedState = localUpdatedState;
                            hit = localHit;
                            return true;
                        }

                        if (delta > quotient)
                        {
                            // Rebuilding the analytic snapshot after the corrected delta exceeded the cached quotient.
                            ResetState();
                            updatedState = localUpdatedState;
                            hit = localHit;
                            return true;
                        }

                        reducedRemainder = compensatedProduct - numerator;
                        if (reducedRemainder >= nextDivisor)
                        {
                            // Reusing state to rebuild the analytic snapshot after the compensated remainder escaped twice.
                            ResetState();
                            updatedState = localUpdatedState;
                            hit = localHit;
                            return true;
                        }
                    }

                    remainder = (ulong)reducedRemainder;
                }

                if (delta > quotient)
                {
                    UInt128 numeratorForCorrection = UInt128.Zero;
                    if (stepTimesQuotient > previousRemainder)
                    {
                        numeratorForCorrection = stepTimesQuotient - previousRemainder;
                    }

                    if (numeratorForCorrection != UInt128.Zero
                        && TryComputeDeltaExact(numeratorForCorrection, nextDivisor, ref reciprocal, out ulong correctedDelta, out UInt128 correctedProduct)
                        && correctedDelta <= quotient)
                    {
                        delta = correctedDelta;
                        UInt128 correctedRemainder = correctedProduct - numeratorForCorrection;
                        if (correctedRemainder >= nextDivisor)
                        {
                            correctedRemainder -= nextDivisor;
                            if (correctedRemainder >= nextDivisor)
                            {
                                // Reusing state to rebuild the analytic snapshot after the corrected remainder escaped twice.
                                ResetState();
                                updatedState = localUpdatedState;
                                hit = localHit;
                                return true;
                            }
                        }

                        remainder = (ulong)correctedRemainder;
                    }
                    else
                    {
                        // Rebuilding the analytic snapshot after the analytic delta exhausted the quotient.
                        ResetState();
                        updatedState = localUpdatedState;
                        hit = localHit;
                        return true;
                    }
                }

                quotient -= delta;
                currentDivisor = nextDivisor;
            }

            ulong residue = ComputeResidueFromRemainder(remainder, divisor);
            state.LastDivisor = divisor;
            state.Quotient = quotient;
            state.Remainder = remainder;
            state.Reciprocal = reciprocal;
            state.NPrime = divisorData.NPrime;
            state.MontgomeryOne = divisorData.MontgomeryOne;
            state.MontgomeryTwo = divisorData.MontgomeryTwo;
            state.MontgomeryTwoSquared = divisorData.MontgomeryTwoSquared;
            UpdateMontgomeryResidueSnapshot(ref state, residue, divisorData);
            updatedState = state;
            _primeStates[prime] = state;
        
            hit = remainder == 0UL ? (byte)1 : (byte)0;
            return true;
        }
        
        private static ulong ComputeReciprocalEstimate(ulong divisor)
        {
            if (divisor <= 1UL)
            {
                return ulong.MaxValue;
            }

            UInt128 scaledOne = UInt128.One << 64;
            UInt128 estimate = scaledOne / divisor;
            if (scaledOne % divisor != UInt128.Zero)
            {
                estimate += UInt128.One;
            }

            if (estimate == UInt128.Zero)
            {
                return 1UL;
            }

            if (estimate > ulong.MaxValue)
            {
                return ulong.MaxValue;
            }

            return (ulong)estimate;
        }

        private static ulong RefineReciprocalEstimate(ulong reciprocal, ulong divisor)
        {
            if (divisor <= 1UL)
            {
                return ulong.MaxValue;
            }

            if (reciprocal == 0UL)
            {
                return ComputeReciprocalEstimate(divisor);
            }

            UInt128 reciprocal128 = reciprocal;
            UInt128 divisor128 = divisor;
            UInt128 product = divisor128 * reciprocal128;
            UInt128 scaled = product >> 64;
            UInt128 twoScaled = (UInt128)2 << 64;
            if (scaled >= twoScaled)
            {
                return ComputeReciprocalEstimate(divisor);
            }

            UInt128 correction = twoScaled - scaled;
            UInt128 refined = (reciprocal128 * correction) >> 64;
            if (refined == UInt128.Zero)
            {
                return 1UL;
            }

            if (refined > ulong.MaxValue)
            {
                return ulong.MaxValue;
            }

            return (ulong)refined;
        }

        private static ulong DivideRoundUpUsingReciprocal(UInt128 numerator, ulong divisor, ulong reciprocal)
        {
            if (numerator == UInt128.Zero)
            {
                return 0UL;
            }

            if (divisor <= 1UL)
            {
                return ulong.MaxValue;
            }

            if (reciprocal == 0UL)
            {
                return ulong.MaxValue;
            }

            ulong high = (ulong)(numerator >> 64);
            ulong low = (ulong)numerator;
            if (high >= divisor)
            {
                return ulong.MaxValue;
            }

            UInt128 highProduct = (UInt128)high * reciprocal;
            UInt128 lowProduct = (UInt128)low * reciprocal;
            UInt128 estimateValue = highProduct + (lowProduct >> 64);

            if (estimateValue > (UInt128)ulong.MaxValue)
            {
                return ulong.MaxValue;
            }

            ulong estimate = (ulong)estimateValue;
            UInt128 product = (UInt128)estimate * divisor;

            while (product > numerator)
            {
                if (estimate == 0UL)
                {
                    return 0UL;
                }

                estimate--;
                product -= divisor;
            }

            UInt128 remainder = numerator - product;
            while (remainder >= divisor)
            {
                remainder -= divisor;
                estimate++;
                if (estimate == ulong.MaxValue)
                {
                    return ulong.MaxValue;
                }
            }

            if (remainder == UInt128.Zero)
            {
                return estimate;
            }

            if (estimate == ulong.MaxValue)
            {
                return ulong.MaxValue;
            }

            return estimate + 1UL;
        }

        private static PrimeDivisorState CreatePrimeState(ulong step, ulong divisor, ulong mersenne, in MontgomeryDivisorData divisorData)
        {
            PrimeDivisorState state = new()
            {
                Step = step,
                LastDivisor = divisor,
                Reciprocal = ComputeReciprocalEstimate(divisor),
                NPrime = divisorData.NPrime,
                MontgomeryOne = divisorData.MontgomeryOne,
                MontgomeryTwo = divisorData.MontgomeryTwo,
                MontgomeryTwoSquared = divisorData.MontgomeryTwoSquared,
                HasState = true,
            };

            ulong quotient = mersenne / divisor;
            ulong remainder = mersenne % divisor;
            state.Quotient = quotient;
            state.Remainder = remainder;
            ulong residue = ComputeResidueFromRemainder(remainder, divisor);
            UpdateMontgomeryResidueSnapshot(ref state, residue, divisorData);
            return state;
        }

        private static bool TryGetMersenneMinusOne(ulong exponent, out ulong mersenne)
        {
            if (exponent <= 63UL)
            {
                mersenne = (1UL << (int)exponent) - 1UL;
                return true;
            }

            if (exponent == 64UL)
            {
                mersenne = ulong.MaxValue;
                return true;
            }

            mersenne = 0UL;
            return false;
        }

        private static ulong ComputeResidueFromRemainder(ulong remainder, ulong divisor)
        {
            ulong residue = remainder + 1UL;
            if (residue >= divisor)
            {
                residue -= divisor;
            }

            return residue;
        }

        private static void UpdateMontgomeryResidueSnapshot(ref PrimeDivisorState state, ulong residue, in MontgomeryDivisorData divisorData)
        {
            state.MontgomeryResidueValid = true;
            if (residue == 0UL)
            {
                state.MontgomeryResidue = 0UL;
                return;
            }

            if (residue == 1UL)
            {
                state.MontgomeryResidue = divisorData.MontgomeryOne;
                return;
            }

            if (residue == 2UL)
            {
                state.MontgomeryResidue = divisorData.MontgomeryTwo;
                return;
            }

            ulong modulus = divisorData.Modulus;
            if (residue >= modulus)
            {
                // Reusing residue to hold the normalized value before entering Montgomery space.
                residue -= modulus;
            }

            // Reusing residue again to capture the Montgomery representation for the cached snapshot.
            residue = residue.MontgomeryMultiply(divisorData.MontgomeryTwoSquared, modulus, divisorData.NPrime);
            state.MontgomeryResidue = residue;
        }

        private static ulong ConvertResidueToMontgomery(ulong residue, in MontgomeryDivisorData divisorData)
        {
            ulong modulus = divisorData.Modulus;
            if (modulus <= 1UL)
            {
                return 0UL;
            }

            if (residue >= modulus)
            {
                residue -= modulus;
            }

            if (residue == 0UL)
            {
                return 0UL;
            }

            if (residue == 1UL)
            {
                return divisorData.MontgomeryOne;
            }

            if (residue == 2UL)
            {
                return divisorData.MontgomeryTwo;
            }

            ulong montgomeryTwoSquared = divisorData.MontgomeryTwoSquared;
            ulong nPrime = divisorData.NPrime;
            return residue.MontgomeryMultiply(montgomeryTwoSquared, modulus, nPrime);
        }

        private void EnsureMontgomeryResidue(ulong exponent, ulong currentDivisor, ref PrimeDivisorState state)
        {
            if (state.MontgomeryResidueValid)
            {
                return;
            }

            var divisorData = new MontgomeryDivisorData(
                currentDivisor,
                state.NPrime,
                state.MontgomeryOne,
                state.MontgomeryTwo,
                state.MontgomeryTwoSquared);

            ulong residue = ComputeResidueFromRemainder(state.Remainder, currentDivisor);
            state.MontgomeryResidue = ConvertResidueToMontgomery(residue, divisorData);
            state.MontgomeryResidueValid = true;
            _primeStates[exponent] = state;
        }

        private struct PrimeDivisorState
        {
            public ulong Step;
            public ulong LastDivisor;
            public ulong Quotient;
            public ulong Remainder;
            public ulong Reciprocal;
            public ulong MontgomeryResidue;
            public ulong NPrime;
            public ulong MontgomeryOne;
            public ulong MontgomeryTwo;
            public ulong MontgomeryTwoSquared;
            public bool MontgomeryResidueValid;
            public bool HasState;
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            _owner.ReturnSession(this);
        }
    }
}


