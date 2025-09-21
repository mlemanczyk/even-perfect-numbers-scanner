using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

public sealed class MersenneNumberDivisorByDivisorGpuTester
{
        private readonly List<ulong> _divisors = new();
        private readonly List<ulong> _remainders = new();
        private readonly List<ulong> _lastPrimes = new();
        private readonly object _sync = new();
        private ulong _divisorLimit;
        private bool _isConfigured;

        private readonly ConcurrentDictionary<Accelerator, Action<Index1D, ulong, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>>> _updateKernelCache = new();
        private readonly ConcurrentDictionary<Accelerator, Action<Index1D, ulong, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>>> _initialKernelCache = new();

        private Action<Index1D, ulong, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>> GetUpdateKernel(Accelerator accelerator) =>
                        _updateKernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ulong, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>>(UpdateKernel));

        private Action<Index1D, ulong, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>> GetInitialKernel(Accelerator accelerator) =>
                        _initialKernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ulong, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>>(InitialKernel));

        public void ConfigureFromMaxPrime(ulong maxPrime)
        {
                lock (_sync)
                {
                        if (_divisors.Count > 0)
                        {
                                _divisors.Clear();
                                _remainders.Clear();
                                _lastPrimes.Clear();
                        }

                        _divisorLimit = ComputeDivisorLimitFromMaxPrime(maxPrime);
                        _isConfigured = true;
                }
        }

        public bool IsPrime(ulong prime, out bool divisorsExhausted)
        {
                lock (_sync)
                {
                        if (!_isConfigured)
                        {
                                throw new InvalidOperationException("ConfigureFromMaxPrime must be called before using the tester.");
                        }

                        ulong allowedMax = ComputeAllowedMaxDivisor(prime);
                        if (allowedMax < 3UL)
                        {
                                divisorsExhausted = true;
                                return true;
                        }

                        bool composite = UpdateExistingDivisors(prime, allowedMax);
                        if (!composite)
                        {
                                composite = ExtendDivisors(prime, allowedMax);
                        }

                        divisorsExhausted = true;
                        return !composite;
                }
        }

        private bool UpdateExistingDivisors(ulong prime, ulong allowedMax)
        {
                int count = _divisors.Count;
                if (count == 0)
                {
                        return false;
                }

                var gpuLease = GpuContextPool.RentPreferred(preferCpu: false);
                var accelerator = gpuLease.Accelerator;
                var kernel = GetUpdateKernel(accelerator);
                var divisorsBuffer = accelerator.Allocate1D<ulong>(count);
                var remaindersBuffer = accelerator.Allocate1D<ulong>(count);
                var lastPrimesBuffer = accelerator.Allocate1D<ulong>(count);
                var hitsBuffer = accelerator.Allocate1D<byte>(count);

                Span<ulong> divisorsSpan = CollectionsMarshal.AsSpan(_divisors);
                Span<ulong> remaindersSpan = CollectionsMarshal.AsSpan(_remainders);
                Span<ulong> lastPrimesSpan = CollectionsMarshal.AsSpan(_lastPrimes);

                divisorsBuffer.View.CopyFromCPU(ref MemoryMarshal.GetReference(divisorsSpan), count);
                remaindersBuffer.View.CopyFromCPU(ref MemoryMarshal.GetReference(remaindersSpan), count);
                lastPrimesBuffer.View.CopyFromCPU(ref MemoryMarshal.GetReference(lastPrimesSpan), count);
                hitsBuffer.MemSetToZero();

                kernel(count, prime, divisorsBuffer.View, remaindersBuffer.View, lastPrimesBuffer.View, hitsBuffer.View);
                accelerator.Synchronize();

                remaindersBuffer.View.CopyToCPU(ref MemoryMarshal.GetReference(remaindersSpan), count);
                lastPrimesBuffer.View.CopyToCPU(ref MemoryMarshal.GetReference(lastPrimesSpan), count);

                byte[] hits = ArrayPool<byte>.Shared.Rent(count);
                Span<byte> hitsSpan = hits.AsSpan(0, count);
                hitsBuffer.View.CopyToCPU(ref MemoryMarshal.GetReference(hitsSpan), count);

                bool composite = false;
                for (int i = 0; i < count; i++)
                {
                        if (divisorsSpan[i] > allowedMax)
                        {
                                break;
                        }

                        if (hitsSpan[i] != 0)
                        {
                                composite = true;
                                break;
                        }
                }

                ArrayPool<byte>.Shared.Return(hits, clearArray: true);

                hitsBuffer.Dispose();
                lastPrimesBuffer.Dispose();
                remaindersBuffer.Dispose();
                divisorsBuffer.Dispose();
                gpuLease.Dispose();

                return composite;
        }

        private bool ExtendDivisors(ulong prime, ulong allowedMax)
        {
                ulong currentMax = _divisors.Count == 0 ? 1UL : _divisors[^1];
                ulong start = currentMax < 3UL ? 3UL : currentMax + 2UL;
                if (start > allowedMax)
                {
                        return false;
                }

                List<ulong> newDivisors = new();
                for (ulong divisor = start; divisor <= allowedMax; divisor += 2UL)
                {
                        newDivisors.Add(divisor);
                }

                int count = newDivisors.Count;
                if (count == 0)
                {
                        return false;
                }

                var gpuLease = GpuContextPool.RentPreferred(preferCpu: false);
                var accelerator = gpuLease.Accelerator;
                var kernel = GetInitialKernel(accelerator);
                var divisorsBuffer = accelerator.Allocate1D<ulong>(count);
                var remaindersBuffer = accelerator.Allocate1D<ulong>(count);
                var hitsBuffer = accelerator.Allocate1D<byte>(count);

                Span<ulong> newDivisorsSpan = CollectionsMarshal.AsSpan(newDivisors);
                divisorsBuffer.View.CopyFromCPU(ref MemoryMarshal.GetReference(newDivisorsSpan), count);
                hitsBuffer.MemSetToZero();

                kernel(count, prime, divisorsBuffer.View, remaindersBuffer.View, hitsBuffer.View);
                accelerator.Synchronize();

                ulong[] newRemainders = ArrayPool<ulong>.Shared.Rent(count);
                byte[] hits = ArrayPool<byte>.Shared.Rent(count);
                Span<ulong> newRemaindersSpan = newRemainders.AsSpan(0, count);
                Span<byte> hitsSpan = hits.AsSpan(0, count);
                remaindersBuffer.View.CopyToCPU(ref MemoryMarshal.GetReference(newRemaindersSpan), count);
                hitsBuffer.View.CopyToCPU(ref MemoryMarshal.GetReference(hitsSpan), count);

                bool composite = false;
                for (int i = 0; i < count; i++)
                {
                        ulong divisor = newDivisorsSpan[i];
                        ulong remainder = newRemaindersSpan[i];
                        _divisors.Add(divisor);
                        _remainders.Add(remainder);
                        _lastPrimes.Add(prime);
                        if (!composite && hitsSpan[i] != 0)
                        {
                                composite = true;
                        }
                }

                ArrayPool<ulong>.Shared.Return(newRemainders, clearArray: true);
                ArrayPool<byte>.Shared.Return(hits, clearArray: true);

                hitsBuffer.Dispose();
                remaindersBuffer.Dispose();
                divisorsBuffer.Dispose();
                gpuLease.Dispose();

                return composite;
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

        private ulong ComputeAllowedMaxDivisor(ulong prime)
        {
                if (prime <= 1UL)
                {
                        return 0UL;
                }

                if (prime - 1UL >= 64UL)
                {
                        return _divisorLimit;
                }

                ulong maxByPrime = (1UL << (int)(prime - 1UL)) - 1UL;
                return Math.Min(maxByPrime, _divisorLimit);
        }

        private static void UpdateKernel(Index1D index, ulong prime, ArrayView<ulong> divisors, ArrayView<ulong> remainders, ArrayView<ulong> lastPrimes, ArrayView<byte> hits)
        {
                ulong divisor = divisors[index];
                if (divisor <= 1UL || (divisor & 1UL) == 0UL)
                {
                        hits[index] = 0;
                        return;
                }

                ulong lastPrime = lastPrimes[index];
                ulong remainder = remainders[index] % divisor;
                ulong step = prime >= lastPrime ? prime - lastPrime : 0UL;
                if (step == 0UL)
                {
                        hits[index] = remainder == 0UL ? (byte)1 : (byte)0;
                        return;
                }

                ulong pow = Pow2Mod(step, divisor);
                ulong temp = remainder + 1UL;
                if (temp >= divisor)
                {
                        temp -= divisor;
                }

                ulong product = MulMod(temp, pow, divisor);
                ulong newRemainder = product == 0UL ? divisor - 1UL : product - 1UL;
                bool isZero = newRemainder == 0UL;
                remainders[index] = newRemainder;
                lastPrimes[index] = prime;
                hits[index] = isZero ? (byte)1 : (byte)0;
        }

        private static void InitialKernel(Index1D index, ulong prime, ArrayView<ulong> divisors, ArrayView<ulong> remainders, ArrayView<byte> hits)
        {
                ulong divisor = divisors[index];
                if (divisor <= 1UL || (divisor & 1UL) == 0UL)
                {
                        remainders[index] = 0UL;
                        hits[index] = 0;
                        return;
                }

                ulong pow = Pow2Mod(prime, divisor);
                ulong remainder = pow == 0UL ? divisor - 1UL : (pow == 1UL ? 0UL : pow - 1UL);
                remainders[index] = remainder;
                hits[index] = remainder == 0UL ? (byte)1 : (byte)0;
        }

        private static ulong Pow2Mod(ulong exponent, ulong modulus)
        {
                if (modulus <= 1UL)
                {
                        return 0UL;
                }

                ulong result = 1UL % modulus;
                ulong baseVal = 2UL % modulus;
                ulong exp = exponent;
                while (exp > 0UL)
                {
                        if ((exp & 1UL) != 0UL)
                        {
                                result = MulMod(result, baseVal, modulus);
                        }

                        exp >>= 1;
                        if (exp == 0UL)
                        {
                                break;
                        }

                        baseVal = MulMod(baseVal, baseVal, modulus);
                }

                return result;
        }

        private static ulong MulMod(ulong a, ulong b, ulong modulus)
        {
                if (modulus == 0UL)
                {
                        return 0UL;
                }

                ulong result = 0UL;
                ulong x = a % modulus;
                ulong y = b;
                while (y > 0UL)
                {
                        if ((y & 1UL) != 0UL)
                        {
                                result += x;
                                if (result >= modulus)
                                {
                                        result -= modulus;
                                }
                        }

                        x <<= 1;
                        if (x >= modulus)
                        {
                                x -= modulus;
                        }

                        y >>= 1;
                }

                return result;
        }
}

