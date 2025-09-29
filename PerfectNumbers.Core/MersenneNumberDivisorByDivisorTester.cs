using System;
using System.Buffers;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Open.Numeric.Primes;

namespace PerfectNumbers.Core;

public static class MersenneNumberDivisorByDivisorTester
{
        private const int ByDivisorStateActive = 0;
        private const int ByDivisorStateComposite = 1;
        private const int ByDivisorStateCompleted = 2;
        private const int ByDivisorStateCompletedDetailed = 3;
        private const int DivisorAllocationBlockSize = 64;

        public static void Run(
                List<ulong> candidates,
                IMersenneNumberDivisorByDivisorTester tester,
                Dictionary<ulong, (bool DetailedCheck, bool PassedAllTests)>? previousResults,
                ulong startPrime,
                Action markComposite,
                Action clearComposite,
                Action<ulong, bool, bool, bool> printResult,
                int threadCount)
        {
                if (candidates.Count == 0)
                {
                        Console.WriteLine("No candidates were provided for --mersenne=bydivisor.");
                        return;
                }

                bool applyStartPrime = startPrime > 0UL;
                int skippedByPreviousResults = 0;
                List<ByDivisorPrimeState> states = new(candidates.Count);
                ulong maxPrime = 0UL;

                using IEnumerator<ulong> primeEnumerator = Prime.Numbers.GetEnumerator();
                bool hasPrime = primeEnumerator.MoveNext();
                ulong currentPrime = hasPrime ? primeEnumerator.Current : 0UL;

                for (int index = 0; index < candidates.Count; index++)
                {
                        ulong candidate = candidates[index];

                        if (applyStartPrime && candidate < startPrime)
                        {
                                continue;
                        }

                        if (previousResults is not null && previousResults.ContainsKey(candidate))
                        {
                                skippedByPreviousResults++;
                                continue;
                        }

                        if (candidate <= 1UL)
                        {
                                markComposite();
                                printResult(candidate, false, false, false);
                                continue;
                        }

                        while (hasPrime && currentPrime < candidate)
                        {
                                hasPrime = primeEnumerator.MoveNext();
                                if (!hasPrime)
                                {
                                        break;
                                }

                                currentPrime = primeEnumerator.Current;
                        }

                        if (!hasPrime || currentPrime != candidate)
                        {
                                markComposite();
                                printResult(candidate, false, false, false);
                                continue;
                        }

                        if (candidate > maxPrime)
                        {
                                maxPrime = candidate;
                        }

                        states.Add(new ByDivisorPrimeState
                        {
                                Prime = candidate,
                        });
                }

                if (skippedByPreviousResults > 0)
                {
                        Console.WriteLine($"Skipped {skippedByPreviousResults.ToString(CultureInfo.InvariantCulture)} candidates excluded by previous results.");
                }

                if (states.Count == 0)
                {
                        if (applyStartPrime)
                        {
                                Console.WriteLine($"No primes greater than or equal to {startPrime.ToString(CultureInfo.InvariantCulture)} were found for --mersenne=bydivisor.");
                        }
                        else
                        {
                                Console.WriteLine("No prime candidates remain for --mersenne=bydivisor after filtering.");
                        }

                        return;
                }

                if (maxPrime <= 1UL)
                {
                        Console.WriteLine("The filter specified by --filter-p must contain at least one prime exponent greater than 1 for --mersenne=bydivisor.");
                        return;
                }

                tester.ConfigureFromMaxPrime(maxPrime);

                int stateCount = states.Count;
                ulong[] primeBatch = ArrayPool<ulong>.Shared.Rent(stateCount);
                ulong[] allowedMaxBatch = ArrayPool<ulong>.Shared.Rent(stateCount);

                try
                {
                        Span<ulong> primeBatchSpan = primeBatch.AsSpan(0, stateCount);
                        Span<ulong> allowedMaxSpan = allowedMaxBatch.AsSpan(0, stateCount);

                        for (int batchIndex = 0; batchIndex < stateCount; batchIndex++)
                        {
                                primeBatchSpan[batchIndex] = states[batchIndex].Prime;
                        }

                        tester.PrepareCandidates(primeBatchSpan, allowedMaxSpan);

                        List<ByDivisorPrimeState> filteredStates = new(stateCount);
                        for (int stateIndex = 0; stateIndex < stateCount; stateIndex++)
                        {
                                ulong prime = primeBatchSpan[stateIndex];
                                ulong allowedMax = allowedMaxSpan[stateIndex];
                                if (allowedMax < 3UL)
                                {
                                        clearComposite();
                                        printResult(prime, true, true, true);
                                        continue;
                                }

                                filteredStates.Add(new ByDivisorPrimeState
                                {
                                        Prime = prime,
                                        AllowedMax = allowedMax,
                                        Completed = false,
                                        Composite = false,
                                        DetailedCheck = false,
                                });
                        }

                        states = filteredStates;
                }
                finally
                {
                        ArrayPool<ulong>.Shared.Return(allowedMaxBatch, clearArray: true);
                        ArrayPool<ulong>.Shared.Return(primeBatch, clearArray: true);
                }

                if (states.Count == 0)
                {
                        if (applyStartPrime)
                        {
                                Console.WriteLine($"No primes greater than or equal to {startPrime.ToString(CultureInfo.InvariantCulture)} were found for --mersenne=bydivisor.");
                        }

                        return;
                }

                stateCount = states.Count;
                ulong[] primeValues = new ulong[stateCount];
                ulong[] allowedMaxValues = new ulong[stateCount];
                int[] stateFlags = new int[stateCount];

                for (int stateIndex = 0; stateIndex < stateCount; stateIndex++)
                {
                        ByDivisorPrimeState currentState = states[stateIndex];
                        primeValues[stateIndex] = currentState.Prime;
                        allowedMaxValues[stateIndex] = currentState.AllowedMax;
                        stateFlags[stateIndex] = ByDivisorStateActive;
                }

                states.Clear();

                ulong divisorLimit = tester.DivisorLimit;
                ulong nextDivisor = 3UL;
                long finalDivisorBits = unchecked((long)3UL);
                int divisorsExhaustedFlag = 0;
                int finalizerState = 0;
                int finalizationCompleted = 0;
                int remainingStates = stateCount;
                int activeStartIndex = 0;
                long[] activeStateMask = new long[(stateCount + 63) >> 6];

                for (int maskStateIndex = 0; maskStateIndex < stateCount; maskStateIndex++)
                {
                        int wordIndex = maskStateIndex >> 6;
                        int bitIndex = maskStateIndex & 63;
                        activeStateMask[wordIndex] |= 1L << bitIndex;
                }

                Task[] workers = new Task[Math.Max(1, threadCount)];

                for (int workerIndex = 0; workerIndex < workers.Length; workerIndex++)
                {
                        int capturedStateCount = stateCount;
                        workers[workerIndex] = Task.Factory.StartNew(() =>
                        {
                                using var session = tester.CreateDivisorSession();
                                byte[] hitsBuffer = ArrayPool<byte>.Shared.Rent(capturedStateCount);
                                ulong[] primeBuffer = ArrayPool<ulong>.Shared.Rent(capturedStateCount);
                                int[] indexBuffer = ArrayPool<int>.Shared.Rent(capturedStateCount);
                                PendingResult[] completionsBuffer = ArrayPool<PendingResult>.Shared.Rent(capturedStateCount);
                                PendingResult[] compositesBuffer = ArrayPool<PendingResult>.Shared.Rent(capturedStateCount);
                                int completionsCount = 0;
                                int compositesCount = 0;
                                ulong localDivisorCursor = 0UL;
                                int localDivisorsRemaining = 0;
                                bool exhausted = false;
                                ulong divisor;
                                int activeCount;
                                Span<byte> hitsSpan = default;
                                int hitIndex = 0;
                                int index;
                                bool useDivisorCycles = tester.UseDivisorCycles;
                                ulong divisorCycle = 0UL;

                                try
                                {
                                        while (true)
                                        {
                                                if (Volatile.Read(ref remainingStates) == 0)
                                                {
                                                        if (Volatile.Read(ref finalizationCompleted) != 0 || Volatile.Read(ref divisorsExhaustedFlag) == 0)
                                                        {
                                                                break;
                                                        }
                                                }

                                                divisor = AcquireNextDivisor(ref nextDivisor, divisorLimit, ref divisorsExhaustedFlag, ref finalDivisorBits, out exhausted, ref localDivisorCursor, ref localDivisorsRemaining);
                                                if (divisor != 0UL)
                                                {
                                                        DivisorCycleCache.CycleBlock? cycleBlock = null;
                                                        if (useDivisorCycles)
                                                        {
                                                                cycleBlock = DivisorCycleCache.Shared.Acquire(divisor);
                                                                divisorCycle = cycleBlock.GetCycle(divisor);
                                                        }
                                                        else
                                                        {
                                                                divisorCycle = 0UL;
                                                        }

                                                        activeCount = BuildPrimeBuffer(
                                                                divisor,
                                                                primeValues,
                                                                allowedMaxValues,
                                                                stateFlags,
                                                                primeBuffer,
                                                                indexBuffer,
                                                                completionsBuffer,
                                                                ref completionsCount,
                                                                ref remainingStates,
                                                                activeStateMask,
                                                                ref activeStartIndex,
                                                                markComposite,
                                                                clearComposite,
                                                                printResult);

                                                        if (completionsCount > 0)
                                                        {
                                                                FlushPendingResults(completionsBuffer, ref completionsCount, markComposite, clearComposite, printResult);
                                                        }

                                                        if (activeCount == 0)
                                                        {
                                                                cycleBlock?.Dispose();
                                                                continue;
                                                        }

                                                        hitsSpan = hitsBuffer.AsSpan(0, activeCount);
                                                        hitsSpan.Clear();
                                                        session.CheckDivisor(divisor, divisorCycle, primeBuffer.AsSpan(0, activeCount), hitsSpan);

                                                        for (hitIndex = 0; hitIndex < activeCount; hitIndex++)
                                                        {
                                                                if (hitsSpan[hitIndex] == 0)
                                                                {
                                                                        continue;
                                                                }

                                                                index = indexBuffer[hitIndex];
                                                                if (Interlocked.CompareExchange(ref stateFlags[index], ByDivisorStateComposite, ByDivisorStateActive) == ByDivisorStateActive)
                                                                {
                                                                        ClearActiveMask(activeStateMask, index);
                                                                        Interlocked.Decrement(ref remainingStates);
                                                                        compositesBuffer[compositesCount++] = new PendingResult(primeValues[index], detailedCheck: true, passedAllTests: false);
                                                                }

                                                                if (compositesCount == compositesBuffer.Length)
                                                                {
                                                                        FlushPendingResults(compositesBuffer, ref compositesCount, markComposite, clearComposite, printResult);
                                                                }
                                                        }

                                                        if (compositesCount > 0)
                                                        {
                                                                FlushPendingResults(compositesBuffer, ref compositesCount, markComposite, clearComposite, printResult);
                                                        }

                                                        cycleBlock?.Dispose();

                                                        continue;
                                                }

                                                if (!exhausted)
                                                {
                                                        if (Volatile.Read(ref remainingStates) == 0)
                                                        {
                                                                break;
                                                        }

                                                        Thread.Yield();
                                                        continue;
                                                }

                                                if (Interlocked.CompareExchange(ref finalizerState, 1, 0) == 0)
                                                {
                                                        FinalizeRemainingStates(
                                                                primeValues,
                                                                allowedMaxValues,
                                                                stateFlags,
                                                                ref remainingStates,
                                                                ref finalDivisorBits,
                                                                completionsBuffer,
                                                                ref completionsCount,
                                                                activeStateMask,
                                                                clearComposite,
                                                                printResult,
                                                                markComposite);

                                                        if (completionsCount > 0)
                                                        {
                                                                FlushPendingResults(completionsBuffer, ref completionsCount, markComposite, clearComposite, printResult);
                                                        }

                                                        Volatile.Write(ref finalizationCompleted, 1);
                                                }
                                                else
                                                {
                                                        while (Volatile.Read(ref finalizationCompleted) == 0 && Volatile.Read(ref remainingStates) > 0)
                                                        {
                                                                Thread.Yield();
                                                        }
                                                }

                                                if (Volatile.Read(ref remainingStates) == 0)
                                                {
                                                        break;
                                                }
                                        }
                                }
                                finally
                                {
                                        if (completionsCount > 0)
                                        {
                                                FlushPendingResults(completionsBuffer, ref completionsCount, markComposite, clearComposite, printResult);
                                        }

                                        if (compositesCount > 0)
                                        {
                                                FlushPendingResults(compositesBuffer, ref compositesCount, markComposite, clearComposite, printResult);
                                        }

                                        ArrayPool<PendingResult>.Shared.Return(compositesBuffer, clearArray: true);
                                        ArrayPool<PendingResult>.Shared.Return(completionsBuffer, clearArray: true);
                                        ArrayPool<int>.Shared.Return(indexBuffer, clearArray: true);
                                        ArrayPool<ulong>.Shared.Return(primeBuffer, clearArray: true);
                                        ArrayPool<byte>.Shared.Return(hitsBuffer, clearArray: true);
                                }
                        }, TaskCreationOptions.LongRunning);
                }

                Task.WaitAll(workers);
        }

        private static ulong AcquireNextDivisor(
                ref ulong nextDivisor,
                ulong divisorLimit,
                ref int divisorsExhaustedFlag,
                ref long finalDivisorBits,
                out bool exhausted,
                ref ulong localDivisorCursor,
                ref int localDivisorsRemaining)
        {
                ref long nextDivisorBits = ref Unsafe.As<ulong, long>(ref nextDivisor);
                ulong blockStride = unchecked(DivisorAllocationBlockSize * 2UL);

                if (TryAcquireLocalDivisor(ref localDivisorCursor, ref localDivisorsRemaining, out ulong localDivisor))
                {
                        exhausted = false;
                        return localDivisor;
                }

                while (true)
                {
                        if (Volatile.Read(ref divisorsExhaustedFlag) != 0)
                        {
                                exhausted = true;
                                return 0UL;
                        }

                        long currentBits = Volatile.Read(ref nextDivisorBits);
                        ulong currentValue = unchecked((ulong)currentBits);
                        if (currentValue > divisorLimit)
                        {
                                if (Interlocked.CompareExchange(ref divisorsExhaustedFlag, 1, 0) == 0)
                                {
                                        Volatile.Write(ref finalDivisorBits, currentBits);
                                }

                                exhausted = true;
                                return 0UL;
                        }

                        ulong maximumNext = divisorLimit >= ulong.MaxValue - 1UL ? ulong.MaxValue : divisorLimit + 2UL;
                        ulong requestedNext = currentValue > ulong.MaxValue - blockStride ? ulong.MaxValue : currentValue + blockStride;
                        if (requestedNext > maximumNext)
                        {
                                requestedNext = maximumNext;
                        }

                        long nextBits = unchecked((long)requestedNext);
                        if (Interlocked.CompareExchange(ref nextDivisorBits, nextBits, currentBits) != currentBits)
                        {
                                continue;
                        }

                        ulong available = requestedNext - currentValue;
                        if (available == 0UL)
                        {
                                continue;
                        }

                        int count = (int)(available >> 1);
                        if (count <= 0)
                        {
                                continue;
                        }

                        localDivisorCursor = currentValue;
                        localDivisorsRemaining = count;

                        if (TryAcquireLocalDivisor(ref localDivisorCursor, ref localDivisorsRemaining, out ulong candidate))
                        {
                                exhausted = false;
                                return candidate;
                        }

                        localDivisorCursor = 0UL;
                        localDivisorsRemaining = 0;
                }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool TryAcquireLocalDivisor(ref ulong cursor, ref int remaining, out ulong divisor)
        {
                while (remaining > 0)
                {
                        ulong candidate = cursor;
                        cursor += 2UL;
                        remaining--;

                        if (IsAllowedDivisorCandidate(candidate))
                        {
                                divisor = candidate;
                                return true;
                        }
                }

                divisor = 0UL;
                return false;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsAllowedDivisorCandidate(ulong divisor)
        {
                if ((divisor & 1UL) == 0UL)
                {
                        return false;
                }

                if ((divisor % 3UL) == 0UL || (divisor % 5UL) == 0UL || (divisor % 7UL) == 0UL || (divisor % 11UL) == 0UL)
                {
                        return false;
                }

                return true;
        }

        private static int BuildPrimeBuffer(
                ulong divisor,
                in ulong[] primeValues,
                in ulong[] allowedMaxValues,
                int[] stateFlags,
                ulong[] primeBuffer,
                int[] indexBuffer,
                PendingResult[] completionsBuffer,
                ref int completionsCount,
                ref int remainingStates,
                long[] activeStateMask,
                ref int activeStartIndex,
                Action markComposite,
                Action clearComposite,
                Action<ulong, bool, bool, bool> printResult)
        {
                int length = primeValues.Length;
                int startIndex;
                while (true)
                {
                        startIndex = Volatile.Read(ref activeStartIndex);
                        if (startIndex >= length || allowedMaxValues[startIndex] >= divisor)
                        {
                                break;
                        }

                        if (Interlocked.CompareExchange(ref activeStartIndex, startIndex + 1, startIndex) != startIndex)
                        {
                                continue;
                        }

                        int state = Volatile.Read(ref stateFlags[startIndex]);
                        if (state == ByDivisorStateActive)
                        {
                                ClearActiveMask(activeStateMask, startIndex);
                                Interlocked.Decrement(ref remainingStates);
                                if (completionsCount == completionsBuffer.Length)
                                {
                                        FlushPendingResults(completionsBuffer, ref completionsCount, markComposite, clearComposite, printResult);
                                }
                                completionsBuffer[completionsCount++] = new PendingResult(primeValues[startIndex], detailedCheck: true, passedAllTests: true);
                        }
                        else
                        {
                                ClearActiveMask(activeStateMask, startIndex);
                        }
                }

                int index = startIndex;
                int activeCount = 0;
                while (index < length)
                {
                        int wordIndex = index >> 6;
                        ulong word = unchecked((ulong)activeStateMask[wordIndex]);
                        if (word == 0UL)
                        {
                                word = unchecked((ulong)Volatile.Read(ref activeStateMask[wordIndex]));
                                if (word == 0UL)
                                {
                                        index = (wordIndex + 1) << 6;
                                        continue;
                                }
                        }

                        int bitOffset = index & 63;
                        if (bitOffset != 0)
                        {
                                word &= ulong.MaxValue << bitOffset;
                                if (word == 0UL)
                                {
                                        index = (wordIndex + 1) << 6;
                                        continue;
                                }
                        }

                        while (word != 0UL)
                        {
                                int candidateIndex = (wordIndex << 6) + BitOperations.TrailingZeroCount(word);
                                if (candidateIndex >= length)
                                {
                                        return activeCount;
                                }

                                if (stateFlags[candidateIndex] == ByDivisorStateActive)
                                {
                                        primeBuffer[activeCount] = primeValues[candidateIndex];
                                        indexBuffer[activeCount] = candidateIndex;
                                        activeCount++;
                                }
                                else
                                {
                                        ClearActiveMask(activeStateMask, candidateIndex);
                                }

                                word &= word - 1UL;
                        }

                        index = (wordIndex + 1) << 6;
                }

                return activeCount;
        }

        private static void FinalizeRemainingStates(
                ulong[] primeValues,
                ulong[] allowedMaxValues,
                int[] stateFlags,
                ref int remainingStates,
                ref long finalDivisorBits,
                PendingResult[] completionsBuffer,
                ref int completionsCount,
                long[] activeStateMask,
                Action clearComposite,
                Action<ulong, bool, bool, bool> printResult,
                Action markComposite)
        {
                for (int finalizeIndex = 0; finalizeIndex < primeValues.Length; finalizeIndex++)
                {
                        if (stateFlags[finalizeIndex] != ByDivisorStateActive)
                        {
                                continue;
                        }

                        ulong finalDivisor = unchecked((ulong)Volatile.Read(ref finalDivisorBits));
                        bool detailed = finalDivisor > allowedMaxValues[finalizeIndex];

                        if (Interlocked.CompareExchange(ref stateFlags[finalizeIndex], detailed ? ByDivisorStateCompletedDetailed : ByDivisorStateCompleted, ByDivisorStateActive) != ByDivisorStateActive)
                        {
                                continue;
                        }

                        ClearActiveMask(activeStateMask, finalizeIndex);
                        Interlocked.Decrement(ref remainingStates);
                        if (completionsCount == completionsBuffer.Length)
                        {
                                FlushPendingResults(completionsBuffer, ref completionsCount, markComposite, clearComposite, printResult);
                        }

                        completionsBuffer[completionsCount++] = new PendingResult(primeValues[finalizeIndex], detailedCheck: detailed, passedAllTests: true);
                }
        }

        private static void FlushPendingResults(
                PendingResult[] buffer,
                ref int count,
                Action markComposite,
                Action clearComposite,
                Action<ulong, bool, bool, bool> printResult)
        {
                for (int flushIndex = 0; flushIndex < count; flushIndex++)
                {
                        PendingResult result = buffer[flushIndex];
                        if (result.PassedAllTests)
                        {
                                clearComposite();
                        }
                        else
                        {
                                markComposite();
                        }

                        printResult(result.Prime, true, result.DetailedCheck, result.PassedAllTests);
                }

                count = 0;
        }

        private static void ClearActiveMask(long[] activeStateMask, int index)
        {
                int wordIndex = index >> 6;
                long bit = 1L << (index & 63);

                while (true)
                {
                        long current = Volatile.Read(ref activeStateMask[wordIndex]);
                        if ((current & bit) == 0)
                        {
                                return;
                        }

                        if (Interlocked.CompareExchange(ref activeStateMask[wordIndex], current & ~bit, current) == current)
                        {
                                return;
                        }
                }
        }

        private struct ByDivisorPrimeState
        {
                internal ulong Prime;
                internal ulong AllowedMax;
                internal bool Completed;
                internal bool Composite;
                internal bool DetailedCheck;
        }

        private readonly struct PendingResult
        {
                internal PendingResult(ulong prime, bool detailedCheck, bool passedAllTests)
                {
                        Prime = prime;
                        DetailedCheck = detailedCheck;
                        PassedAllTests = passedAllTests;
                }

                internal ulong Prime { get; }

                internal bool DetailedCheck { get; }

                internal bool PassedAllTests { get; }
        }
}
