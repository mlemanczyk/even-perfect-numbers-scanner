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
        private int _gpuBatchSize = GpuConstants.ScanBatchSize;

        private readonly List<ulong> _divisors = new();
        private readonly List<ulong> _remainders = new();
        private readonly List<ulong> _lastPrimes = new();
        private readonly object _sync = new();
        private ulong _divisorLimit;
        private bool _isConfigured;
        private ulong _lastStatusDivisor;

        private readonly ConcurrentDictionary<Accelerator, Action<Index1D, ulong, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>>> _updateKernelCache = new();
        private readonly ConcurrentDictionary<Accelerator, Action<Index1D, ulong, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>>> _initialKernelCache = new();

        private Action<Index1D, ulong, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>> GetUpdateKernel(Accelerator accelerator) =>
                        _updateKernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ulong, ArrayView<ulong>, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>>(UpdateKernel));

        private Action<Index1D, ulong, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>> GetInitialKernel(Accelerator accelerator) =>
                        _initialKernelCache.GetOrAdd(accelerator, acc => acc.LoadAutoGroupedStreamKernel<Index1D, ulong, ArrayView<ulong>, ArrayView<ulong>, ArrayView<byte>>(InitialKernel));

        public int GpuBatchSize
        {
                get => _gpuBatchSize;
                set => _gpuBatchSize = Math.Max(1, value);
        }

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
                        _lastStatusDivisor = 0UL;
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

                var divisorsSpan = CollectionsMarshal.AsSpan(_divisors);
                var remaindersSpan = CollectionsMarshal.AsSpan(_remainders);
                var lastPrimesSpan = CollectionsMarshal.AsSpan(_lastPrimes);

                var gpuLease = GpuContextPool.RentPreferred(preferCpu: false);
                var accelerator = gpuLease.Accelerator;
                var kernel = GetUpdateKernel(accelerator);

                int batchCapacity = Math.Min(_gpuBatchSize, count);
                var divisorsBuffer = accelerator.Allocate1D<ulong>(batchCapacity);
                var remaindersBuffer = accelerator.Allocate1D<ulong>(batchCapacity);
                var lastPrimesBuffer = accelerator.Allocate1D<ulong>(batchCapacity);
                var hitsBuffer = accelerator.Allocate1D<byte>(batchCapacity);

                byte[] hits = ArrayPool<byte>.Shared.Rent(batchCapacity);

                bool composite = false;
                ulong lastProcessed = 0UL;

                try
                {
                        int processed = 0;
                        while (processed < count)
                        {
                                int batchSize = Math.Min(batchCapacity, count - processed);
                                var divisorView = divisorsBuffer.View.SubView(0, batchSize);
                                var remainderView = remaindersBuffer.View.SubView(0, batchSize);
                                var lastPrimeView = lastPrimesBuffer.View.SubView(0, batchSize);
                                var hitsView = hitsBuffer.View.SubView(0, batchSize);

                                Span<ulong> divisorBatch = divisorsSpan.Slice(processed, batchSize);
                                Span<ulong> remainderBatch = remaindersSpan.Slice(processed, batchSize);
                                Span<ulong> lastPrimeBatch = lastPrimesSpan.Slice(processed, batchSize);

                                divisorView.CopyFromCPU(ref MemoryMarshal.GetReference(divisorBatch), batchSize);
                                remainderView.CopyFromCPU(ref MemoryMarshal.GetReference(remainderBatch), batchSize);
                                lastPrimeView.CopyFromCPU(ref MemoryMarshal.GetReference(lastPrimeBatch), batchSize);
                                hitsView.MemSetToZero();

                                kernel(batchSize, prime, divisorView, remainderView, lastPrimeView, hitsView);
                                accelerator.Synchronize();

                                remainderView.CopyToCPU(ref MemoryMarshal.GetReference(remainderBatch), batchSize);
                                lastPrimeView.CopyToCPU(ref MemoryMarshal.GetReference(lastPrimeBatch), batchSize);

                                Span<byte> hitsSpan = hits.AsSpan(0, batchSize);
                                hitsView.CopyToCPU(ref MemoryMarshal.GetReference(hitsSpan), batchSize);

                                for (int i = 0; i < batchSize; i++)
                                {
                                        ulong divisor = divisorBatch[i];
                                        if (divisor > allowedMax)
                                        {
                                                processed = count;
                                                break;
                                        }

                                        lastProcessed = divisor;
                                        if (hitsSpan[i] != 0)
                                        {
                                                composite = true;
                                                processed = count;
                                                break;
                                        }
                                }

                                if (composite)
                                {
                                        break;
                                }

                                if (processed < count)
                                {
                                        processed += batchSize;
                                }
                        }
                }
                finally
                {
                        ArrayPool<byte>.Shared.Return(hits, clearArray: true);
                        hitsBuffer.Dispose();
                        lastPrimesBuffer.Dispose();
                        remaindersBuffer.Dispose();
                        divisorsBuffer.Dispose();
                        gpuLease.Dispose();
                }

                if (lastProcessed != 0UL)
                {
                        ReportStatus(lastProcessed);
                }

                return composite;
        }

        private bool ExtendDivisors(ulong prime, ulong allowedMax)
        {
                int existingCount = _divisors.Count;
                allowedMax = Math.Min(allowedMax, _divisorLimit);

                ulong currentMax = existingCount == 0 ? 1UL : _divisors[^1];
                ulong start = currentMax < 3UL ? 3UL : currentMax + 2UL;
                if (start > allowedMax)
                {
                        return false;
                }

                var gpuLease = GpuContextPool.RentPreferred(preferCpu: false);
                var accelerator = gpuLease.Accelerator;
                var kernel = GetInitialKernel(accelerator);

                int batchCapacity = _gpuBatchSize;
                var divisorsBuffer = accelerator.Allocate1D<ulong>(batchCapacity);
                var remaindersBuffer = accelerator.Allocate1D<ulong>(batchCapacity);
                var hitsBuffer = accelerator.Allocate1D<byte>(batchCapacity);

                ulong[] remainders = ArrayPool<ulong>.Shared.Rent(batchCapacity);
                byte[] hits = ArrayPool<byte>.Shared.Rent(batchCapacity);
                ulong[] divisors = ArrayPool<ulong>.Shared.Rent(batchCapacity);

                bool composite = false;
                ulong lastProcessed = 0UL;
                ulong divisor = start;

                try
                {
                        while (divisor <= allowedMax)
                        {
                                int batchSize = batchCapacity;
                                int index = 0;
                                while (index < batchSize && divisor <= allowedMax)
                                {
                                        divisors[index++] = divisor;
                                        divisor += 2UL;
                                }

                                if (index == 0)
                                {
                                        break;
                                }

                                var divisorView = divisorsBuffer.View.SubView(0, index);
                                var remainderView = remaindersBuffer.View.SubView(0, index);
                                var hitsView = hitsBuffer.View.SubView(0, index);

                                Span<ulong> divisorSpan = divisors.AsSpan(0, index);
                                divisorView.CopyFromCPU(ref MemoryMarshal.GetReference(divisorSpan), index);
                                hitsView.MemSetToZero();

                                kernel(index, prime, divisorView, remainderView, hitsView);
                                accelerator.Synchronize();

                                Span<ulong> remainderSpan = remainders.AsSpan(0, index);
                                Span<byte> hitsSpan = hits.AsSpan(0, index);
                                remainderView.CopyToCPU(ref MemoryMarshal.GetReference(remainderSpan), index);
                                hitsView.CopyToCPU(ref MemoryMarshal.GetReference(hitsSpan), index);

                                for (int i = 0; i < index; i++)
                                {
                                        ulong newDivisor = divisorSpan[i];
                                        _divisors.Add(newDivisor);
                                        _remainders.Add(remainderSpan[i]);
                                        _lastPrimes.Add(prime);
                                        lastProcessed = newDivisor;
                                        if (!composite && hitsSpan[i] != 0)
                                        {
                                                composite = true;
                                        }
                                }

                                if (composite)
                                {
                                        break;
                                }
                        }
                }
                finally
                {
                        ArrayPool<ulong>.Shared.Return(divisors, clearArray: true);
                        ArrayPool<ulong>.Shared.Return(remainders, clearArray: true);
                        ArrayPool<byte>.Shared.Return(hits, clearArray: true);
                        hitsBuffer.Dispose();
                        remaindersBuffer.Dispose();
                        divisorsBuffer.Dispose();
                        gpuLease.Dispose();
                }

                if (lastProcessed != 0UL)
                {
                        ReportStatus(lastProcessed);
                }

                return composite;
        }

        private void ReportStatus(ulong divisor)
        {
                if (divisor <= _lastStatusDivisor)
                {
                        return;
                }

                _lastStatusDivisor = divisor;
                Console.WriteLine($"...processed by-divisor candidate = {divisor}");
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

