using System;
using System.Buffers;
using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;
using System.Numerics;
using PerfectNumbers.Core;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using ILGPU.Runtime.OpenCL;
namespace PerfectNumbers.Core.Gpu;

internal enum GpuPow2ModStatus
{
    Success,
    Overflow,
    Unavailable,
}

internal static partial class PrimeOrderGpuHeuristics
{
    private static readonly ConcurrentDictionary<ulong, byte> OverflowedPrimes = new();
    private static readonly ConcurrentDictionary<UInt128, byte> OverflowedPrimesWide = new();
    private static readonly ConcurrentDictionary<Accelerator, Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>> Pow2ModKernelCache = new();
    private static readonly ConcurrentDictionary<Accelerator, Action<AcceleratorStream, Index1D, ArrayView1D<GpuUInt128, Stride1D.Dense>, GpuUInt128, ArrayView1D<GpuUInt128, Stride1D.Dense>>> Pow2ModKernelWideCache = new();
    private static readonly ConcurrentDictionary<Accelerator, Action<AcceleratorStream, Index1D, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, int, ulong, uint, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<byte, Stride1D.Dense>>> PartialFactorKernelCache = new();
    private static readonly ConcurrentDictionary<Accelerator, OrderKernelLauncher> OrderKernelCache = new();
    private static readonly ConcurrentDictionary<Accelerator, SmallPrimeDeviceCache> SmallPrimeDeviceCaches = new();

    public readonly struct OrderKernelConfig(ulong previousOrder, byte hasPreviousOrder, uint smallFactorLimit, int maxPowChecks, int mode)
    {
        public readonly ulong PreviousOrder = previousOrder;
        public readonly byte HasPreviousOrder = hasPreviousOrder;
        public readonly uint SmallFactorLimit = smallFactorLimit;
        public readonly int MaxPowChecks = maxPowChecks;
        public readonly int Mode = mode;
    }

    public readonly struct OrderKernelBuffers(
        ArrayView1D<ulong, Stride1D.Dense> phiFactors,
        ArrayView1D<int, Stride1D.Dense> phiExponents,
        ArrayView1D<ulong, Stride1D.Dense> workFactors,
        ArrayView1D<int, Stride1D.Dense> workExponents,
        ArrayView1D<ulong, Stride1D.Dense> candidates,
        ArrayView1D<int, Stride1D.Dense> stackIndex,
        ArrayView1D<int, Stride1D.Dense> stackExponent,
        ArrayView1D<ulong, Stride1D.Dense> stackProduct,
        ArrayView1D<ulong, Stride1D.Dense> result,
        ArrayView1D<byte, Stride1D.Dense> status)
    {
        public readonly ArrayView1D<ulong, Stride1D.Dense> PhiFactors = phiFactors;
        public readonly ArrayView1D<int, Stride1D.Dense> PhiExponents = phiExponents;
        public readonly ArrayView1D<ulong, Stride1D.Dense> WorkFactors = workFactors;
        public readonly ArrayView1D<int, Stride1D.Dense> WorkExponents = workExponents;
        public readonly ArrayView1D<ulong, Stride1D.Dense> Candidates = candidates;
        public readonly ArrayView1D<int, Stride1D.Dense> StackIndex = stackIndex;
        public readonly ArrayView1D<int, Stride1D.Dense> StackExponent = stackExponent;
        public readonly ArrayView1D<ulong, Stride1D.Dense> StackProduct = stackProduct;
        public readonly ArrayView1D<ulong, Stride1D.Dense> Result = result;
        public readonly ArrayView1D<byte, Stride1D.Dense> Status = status;
    }

    private delegate void OrderKernelLauncher(
        AcceleratorStream stream,
        Index1D extent,
        ulong prime,
        OrderKernelConfig config,
        MontgomeryDivisorData divisor,
        ArrayView1D<uint, Stride1D.Dense> primes,
        ArrayView1D<ulong, Stride1D.Dense> squares,
        int primeCount,
        OrderKernelBuffers buffers);

    private sealed class SmallPrimeDeviceCache
    {
        public MemoryBuffer1D<uint, Stride1D.Dense>? Primes;
        public MemoryBuffer1D<ulong, Stride1D.Dense>? Squares;
        public int Count;
    }

    private const int WideStackThreshold = 8;
    private static PrimeOrderGpuCapability s_capability = PrimeOrderGpuCapability.Default;

    private const int Pow2WindowSizeBits = 8;
    private const int Pow2WindowOddPowerCount = 1 << (Pow2WindowSizeBits - 1);
    private const ulong Pow2WindowFallbackThreshold = 32UL;
    private const int HeuristicCandidateLimit = 512;
    private const int HeuristicStackCapacity = 256;

    private const int GpuSmallPrimeFactorSlots = 64;

    internal static ConcurrentDictionary<ulong, byte> OverflowRegistry => OverflowedPrimes;
    internal static ConcurrentDictionary<UInt128, byte> OverflowRegistryWide => OverflowedPrimesWide;

    internal static void OverrideCapabilitiesForTesting(PrimeOrderGpuCapability capability)
    {
        s_capability = capability;
    }

    internal static void ResetCapabilitiesForTesting()
    {
        s_capability = PrimeOrderGpuCapability.Default;
    }

    private static SmallPrimeDeviceCache GetSmallPrimeDeviceCache(Accelerator accelerator)
    {
        return SmallPrimeDeviceCaches.GetOrAdd(accelerator, static acc =>
        {
            uint[] primes = PrimesGenerator.SmallPrimes;
            ulong[] squares = PrimesGenerator.SmallPrimesPow2;
            var primeBuffer = acc.Allocate1D<uint>(primes.Length);
            primeBuffer.View.CopyFromCPU(primes);
            var squareBuffer = acc.Allocate1D<ulong>(squares.Length);
            squareBuffer.View.CopyFromCPU(squares);
            return new SmallPrimeDeviceCache
            {
                Primes = primeBuffer,
                Squares = squareBuffer,
                Count = primes.Length,
            };
        });
    }

    private static Action<AcceleratorStream, Index1D, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, int, ulong, uint, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<byte, Stride1D.Dense>> GetPartialFactorKernel(Accelerator accelerator)
    {
        return PartialFactorKernelCache.GetOrAdd(accelerator, static accel =>
        {
            var loaded = accel.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, int, ulong, uint, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<byte, Stride1D.Dense>>(PartialFactorKernel);
            var kernel = KernelUtil.GetKernel(loaded);
            return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, int, ulong, uint, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, ArrayView1D<byte, Stride1D.Dense>>>();
        });
    }

    public static bool TryPartialFactor(
        ulong value,
        uint limit,
        Span<ulong> primeTargets,
        Span<int> exponentTargets,
        out int factorCount,
        out ulong remaining,
        out bool fullyFactored)
    {
        factorCount = 0;
        remaining = value;
        fullyFactored = false;

        if (primeTargets.Length == 0 || exponentTargets.Length == 0)
        {
            return false;
        }

        primeTargets.Clear();
        exponentTargets.Clear();

        if (!TryLaunchPartialFactorKernel(
                value,
                limit,
                primeTargets,
                exponentTargets,
                out int extracted,
                out ulong leftover,
                out bool kernelFullyFactored))
        {
            return false;
        }

        int capacity = Math.Min(primeTargets.Length, exponentTargets.Length);
        if (extracted > capacity)
        {
            factorCount = 0;
            remaining = value;
            fullyFactored = false;
            return false;
        }

        factorCount = extracted;
        remaining = leftover;
        fullyFactored = kernelFullyFactored && leftover == 1UL;
        return true;
    }

    private static bool TryLaunchPartialFactorKernel(
        ulong value,
        uint limit,
        Span<ulong> primeTargets,
        Span<int> exponentTargets,
        out int factorCount,
        out ulong remaining,
        out bool fullyFactored)
    {
        factorCount = 0;
        remaining = value;
        fullyFactored = false;

        try
        {
            var lease = GpuKernelPool.GetKernel(useGpuOrder: true);
            var execution = lease.EnterExecutionScope();
            Accelerator accelerator = lease.Accelerator;
            AcceleratorStream stream = lease.Stream;

            var kernel = GetPartialFactorKernel(accelerator);
            SmallPrimeDeviceCache cache = GetSmallPrimeDeviceCache(accelerator);

            var factorBuffer = accelerator.Allocate1D<ulong>(primeTargets.Length);
            var exponentBuffer = accelerator.Allocate1D<int>(exponentTargets.Length);
            var countBuffer = accelerator.Allocate1D<int>(1);
            var remainingBuffer = accelerator.Allocate1D<ulong>(1);
            var fullyFactoredBuffer = accelerator.Allocate1D<byte>(1);

            factorBuffer.MemSetToZero();
            exponentBuffer.MemSetToZero();
            countBuffer.MemSetToZero();
            remainingBuffer.MemSetToZero();
            fullyFactoredBuffer.MemSetToZero();

            kernel(
                stream,
                1,
                cache.Primes!.View,
                cache.Squares!.View,
                cache.Count,
                primeTargets.Length,
                value,
                limit,
                factorBuffer.View,
                exponentBuffer.View,
                countBuffer.View,
                remainingBuffer.View,
                fullyFactoredBuffer.View);

            stream.Synchronize();

            countBuffer.View.CopyToCPU(ref factorCount, 1);
            factorCount = Math.Min(factorCount, primeTargets.Length);
            factorBuffer.View.CopyToCPU(ref MemoryMarshal.GetReference(primeTargets), primeTargets.Length);
            exponentBuffer.View.CopyToCPU(ref MemoryMarshal.GetReference(exponentTargets), exponentTargets.Length);
            remainingBuffer.View.CopyToCPU(ref remaining, 1);

            byte fullyFactoredFlag = 0;
            fullyFactoredBuffer.View.CopyToCPU(ref fullyFactoredFlag, 1);
            fullyFactored = fullyFactoredFlag != 0;

            factorBuffer.Dispose();
            exponentBuffer.Dispose();
            countBuffer.Dispose();
            remainingBuffer.Dispose();
            fullyFactoredBuffer.Dispose();
            execution.Dispose();
            lease.Dispose();
            return true;
        }
        catch (CLException ex)
        {
            Console.WriteLine($"GPU ERROR ({ex.Error}): {ex.Message}");
            factorCount = 0;
            remaining = value;
            fullyFactored = false;
            return false;
        }
        catch (Exception ex) when (ex is AcceleratorException or InternalCompilerException or NotSupportedException or InvalidOperationException or AggregateException)
        {
            Console.WriteLine($"GPU ERROR: {ex.Message}");
            factorCount = 0;
            remaining = value;
            fullyFactored = false;
            return false;
        }
    }

    private static void PartialFactorKernel(
        Index1D index,
        ArrayView1D<uint, Stride1D.Dense> primes,
        ArrayView1D<ulong, Stride1D.Dense> squares,
        int primeCount,
        int slotCount,
        ulong value,
        uint limit,
        ArrayView1D<ulong, Stride1D.Dense> factorsOut,
        ArrayView1D<int, Stride1D.Dense> exponentsOut,
        ArrayView1D<int, Stride1D.Dense> countOut,
        ArrayView1D<ulong, Stride1D.Dense> remainingOut,
        ArrayView1D<byte, Stride1D.Dense> fullyFactoredOut)
    {
        if (index != 0)
        {
            return;
        }

        uint effectiveLimit = limit == 0 ? uint.MaxValue : limit;
        ulong remainingLocal = value;
        int count = 0;

        for (int i = 0; i < primeCount && count < slotCount; i++)
        {
            uint primeCandidate = primes[i];
            if (primeCandidate > effectiveLimit)
            {
                break;
            }

            ulong primeSquare = squares[i];
            if (primeSquare != 0UL && primeSquare > remainingLocal)
            {
                break;
            }

            ulong primeValue = primeCandidate;
            if (primeValue == 0UL || (remainingLocal % primeValue) != 0UL)
            {
                continue;
            }

            int exponent = 0;
            do
            {
                remainingLocal /= primeValue;
                exponent++;
            }
            while ((remainingLocal % primeValue) == 0UL);

            factorsOut[count] = primeValue;
            exponentsOut[count] = exponent;
            count++;
        }

        countOut[0] = count;
        remainingOut[0] = remainingLocal;
        fullyFactoredOut[0] = remainingLocal == 1UL ? (byte)1 : (byte)0;
    }
    public static GpuPow2ModStatus TryPow2Mod(ulong exponent, ulong prime, out ulong remainder, in MontgomeryDivisorData divisorData)
    {
        Span<ulong> exponents = stackalloc ulong[1];
        Span<ulong> remainders = stackalloc ulong[1];
        exponents[0] = exponent;

        GpuPow2ModStatus status = TryPow2ModBatch(exponents, prime, remainders, divisorData);
        remainder = remainders[0];
        return status;
    }

    public static GpuPow2ModStatus TryPow2ModBatch(ReadOnlySpan<ulong> exponents, ulong prime, Span<ulong> remainders, in MontgomeryDivisorData divisorData)
    {
        // This will never occur in production code
        // if (exponents.Length == 0)
        // {
        //     return GpuPow2ModStatus.Success;
        // }

        // This will never occur in production code
        // if (remainders.Length < exponents.Length)
        // {
        //     throw new ArgumentException("Remainder span is shorter than the exponent span.", nameof(remainders));
        // }

        // This will never occur in production code
        // if (remainders.Length > exponents.Length)
        //     throw new ArgumentException("Remainder span is longer than the exponent span.", nameof(remainders));

        // We need to clear more, but we save on allocations    
        // Span<ulong> target = remainders[..exponents.Length];
        remainders.Clear();

        // This will never occur in production code
        // if (prime <= 1UL)
        // {
        //     return GpuPow2ModStatus.Unavailable;
        // }

        ConcurrentDictionary<ulong, byte> overflowRegistry = OverflowedPrimes;

        if (overflowRegistry.ContainsKey(prime))
        {
            return GpuPow2ModStatus.Overflow;
        }

        PrimeOrderGpuCapability capability = s_capability;

        if (prime.GetBitLength() > capability.ModulusBits)
        {
            overflowRegistry[prime] = 0;
            return GpuPow2ModStatus.Overflow;
        }

        for (int i = 0; i < exponents.Length; i++)
        {
            if (exponents[i].GetBitLength() > capability.ExponentBits)
            {
                return GpuPow2ModStatus.Overflow;
            }
        }

        bool computed = TryComputeOnGpu(exponents, prime, divisorData, remainders);
        return computed ? GpuPow2ModStatus.Success : GpuPow2ModStatus.Unavailable;
    }

    public static GpuPow2ModStatus TryPow2Mod(in UInt128 exponent, in UInt128 prime, out UInt128 remainder)
    {
        Span<UInt128> exponents = stackalloc UInt128[1];
        Span<UInt128> remainders = stackalloc UInt128[1];
        exponents[0] = exponent;

        GpuPow2ModStatus status = TryPow2ModBatch(exponents, prime, remainders);
        remainder = remainders[0];
        return status;
    }

    public static GpuPow2ModStatus TryPow2ModBatch(ReadOnlySpan<UInt128> exponents, UInt128 prime, Span<UInt128> remainders)
    {
        return TryPow2ModBatchInternal(exponents, prime, remainders);
    }

    internal static bool TryCalculateOrder(
        ulong prime,
        ulong? previousOrder,
        PrimeOrderCalculator.PrimeOrderSearchConfig config,
        in MontgomeryDivisorData divisorData,
        out ulong order)
    {
        order = 0UL;

        var lease = GpuKernelPool.GetKernel(useGpuOrder: true);
        var execution = lease.EnterExecutionScope();
        Accelerator accelerator = lease.Accelerator;
        AcceleratorStream stream = lease.Stream;

        var kernel = GetOrderKernel(accelerator);
        SmallPrimeDeviceCache cache = GetSmallPrimeDeviceCache(accelerator);

        var phiFactorBuffer = accelerator.Allocate1D<ulong>(GpuSmallPrimeFactorSlots);
        var phiExponentBuffer = accelerator.Allocate1D<int>(GpuSmallPrimeFactorSlots);
        var workFactorBuffer = accelerator.Allocate1D<ulong>(GpuSmallPrimeFactorSlots);
        var workExponentBuffer = accelerator.Allocate1D<int>(GpuSmallPrimeFactorSlots);
        var candidateBuffer = accelerator.Allocate1D<ulong>(HeuristicCandidateLimit);
        var stackIndexBuffer = accelerator.Allocate1D<int>(HeuristicStackCapacity);
        var stackExponentBuffer = accelerator.Allocate1D<int>(HeuristicStackCapacity);
        var stackProductBuffer = accelerator.Allocate1D<ulong>(HeuristicStackCapacity);
        var resultBuffer = accelerator.Allocate1D<ulong>(1);
        var statusBuffer = accelerator.Allocate1D<byte>(1);

        phiFactorBuffer.MemSetToZero();
        phiExponentBuffer.MemSetToZero();
        workFactorBuffer.MemSetToZero();
        workExponentBuffer.MemSetToZero();
        candidateBuffer.MemSetToZero();
        stackIndexBuffer.MemSetToZero();
        stackExponentBuffer.MemSetToZero();
        stackProductBuffer.MemSetToZero();
        resultBuffer.MemSetToZero();
        statusBuffer.MemSetToZero();

        uint limit = config.SmallFactorLimit == 0 ? uint.MaxValue : config.SmallFactorLimit;
        byte hasPrevious = previousOrder.HasValue ? (byte)1 : (byte)0;
        ulong previousValue = previousOrder ?? 0UL;

        var kernelConfig = new OrderKernelConfig(previousValue, hasPrevious, limit, config.MaxPowChecks, (int)config.Mode);
        var buffers = new OrderKernelBuffers(
            phiFactorBuffer.View,
            phiExponentBuffer.View,
            workFactorBuffer.View,
            workExponentBuffer.View,
            candidateBuffer.View,
            stackIndexBuffer.View,
            stackExponentBuffer.View,
            stackProductBuffer.View,
            resultBuffer.View,
            statusBuffer.View);

        kernel(
            stream,
            1,
            prime,
            kernelConfig,
            divisorData,
            cache.Primes!.View,
            cache.Squares!.View,
            cache.Count,
            buffers);

        stream.Synchronize();

        byte status = 0;
        statusBuffer.View.CopyToCPU(ref status, 1);

        PrimeOrderKernelStatus kernelStatus = (PrimeOrderKernelStatus)status;
        if (kernelStatus == PrimeOrderKernelStatus.Fallback)
        {
            DisposeResources();
            lease.Dispose();
            return false;
        }

        if (kernelStatus == PrimeOrderKernelStatus.PollardOverflow)
        {
            DisposeResources();
            lease.Dispose();
            throw new InvalidOperationException("GPU Pollard Rho stack overflow; increase HeuristicStackCapacity.");
        }

        resultBuffer.View.CopyToCPU(ref order, 1);

        if (kernelStatus == PrimeOrderKernelStatus.FactoringFailure)
        {
            order = 0UL;
        }

        DisposeResources();
        lease.Dispose();

        return order != 0UL;

        void DisposeResources()
        {
            phiFactorBuffer.Dispose();
            phiExponentBuffer.Dispose();
            workFactorBuffer.Dispose();
            workExponentBuffer.Dispose();
            candidateBuffer.Dispose();
            stackIndexBuffer.Dispose();
            stackExponentBuffer.Dispose();
            stackProductBuffer.Dispose();
            resultBuffer.Dispose();
            statusBuffer.Dispose();
            execution.Dispose();
        }
    }

    private static OrderKernelLauncher GetOrderKernel(Accelerator accelerator)
    {
        return OrderKernelCache.GetOrAdd(accelerator, static accel =>
        {
            var loaded = accel.LoadAutoGroupedStreamKernel<Index1D, ulong, OrderKernelConfig, MontgomeryDivisorData, ArrayView1D<uint, Stride1D.Dense>, ArrayView1D<ulong, Stride1D.Dense>, int, OrderKernelBuffers>(CalculateOrderKernel);
            var kernel = KernelUtil.GetKernel(loaded);
            return kernel.CreateLauncherDelegate<OrderKernelLauncher>();
        });
    }

    private enum PrimeOrderKernelStatus : byte
    {
        Fallback = 0,
        Found = 1,
        HeuristicUnresolved = 2,
        PollardOverflow = 3,
        FactoringFailure = 4,
    }

    private readonly struct CandidateKey
    {
        public CandidateKey(int primary, long secondary, long tertiary)
        {
            Primary = primary;
            Secondary = secondary;
            Tertiary = tertiary;
        }

        public int Primary { get; }

        public long Secondary { get; }

        public long Tertiary { get; }

        public int CompareTo(CandidateKey other)
        {
            if (Primary != other.Primary)
            {
                return Primary.CompareTo(other.Primary);
            }

            if (Secondary != other.Secondary)
            {
                return Secondary.CompareTo(other.Secondary);
            }

            return Tertiary.CompareTo(other.Tertiary);
        }
    }

    private static void CalculateOrderKernel(
        Index1D index,
        ulong prime,
        OrderKernelConfig config,
        MontgomeryDivisorData divisor,
        ArrayView1D<uint, Stride1D.Dense> primes,
        ArrayView1D<ulong, Stride1D.Dense> squares,
        int primeCount,
        OrderKernelBuffers buffers)
    {
        if (index != 0)
        {
            return;
        }

        ArrayView1D<ulong, Stride1D.Dense> phiFactors = buffers.PhiFactors;
        ArrayView1D<int, Stride1D.Dense> phiExponents = buffers.PhiExponents;
        ArrayView1D<ulong, Stride1D.Dense> workFactors = buffers.WorkFactors;
        ArrayView1D<int, Stride1D.Dense> workExponents = buffers.WorkExponents;
        ArrayView1D<ulong, Stride1D.Dense> candidates = buffers.Candidates;
        ArrayView1D<int, Stride1D.Dense> stackIndex = buffers.StackIndex;
        ArrayView1D<int, Stride1D.Dense> stackExponent = buffers.StackExponent;
        ArrayView1D<ulong, Stride1D.Dense> stackProduct = buffers.StackProduct;
        ArrayView1D<ulong, Stride1D.Dense> resultOut = buffers.Result;
        ArrayView1D<byte, Stride1D.Dense> statusOut = buffers.Status;

        uint limit = config.SmallFactorLimit == 0 ? uint.MaxValue : config.SmallFactorLimit;
        ulong previousOrder = config.PreviousOrder;
        byte hasPreviousOrder = config.HasPreviousOrder;
        int maxPowChecks = config.MaxPowChecks;
        int mode = config.Mode;

        statusOut[0] = (byte)PrimeOrderKernelStatus.Fallback;

        if (prime <= 3UL)
        {
            ulong orderValue = prime == 3UL ? 2UL : 1UL;
            resultOut[0] = orderValue;
            statusOut[0] = (byte)PrimeOrderKernelStatus.Found;
            return;
        }

        ulong phi = prime - 1UL;

        int phiFactorCount = FactorWithSmallPrimes(phi, limit, primes, squares, primeCount, phiFactors, phiExponents, out ulong phiRemaining);
        if (phiRemaining != 1UL)
        {
            // Reuse stackProduct as the Pollard Rho stack so factoring stays within this kernel.
            if (!TryFactorWithPollardKernel(
                    phiRemaining,
                    limit,
                    primes,
                    squares,
                    primeCount,
                    phiFactors,
                    phiExponents,
                    ref phiFactorCount,
                    stackProduct,
                    statusOut))
            {
                if (statusOut[0] != (byte)PrimeOrderKernelStatus.PollardOverflow)
                {
                    resultOut[0] = CalculateByDoublingKernel(prime);
                }

                return;
            }
        }

        SortFactors(phiFactors, phiExponents, phiFactorCount);

        if (TrySpecialMaxKernel(phi, prime, phiFactors, phiFactorCount, divisor))
        {
            resultOut[0] = phi;
            statusOut[0] = (byte)PrimeOrderKernelStatus.Found;
            return;
        }

        ulong candidateOrder = InitializeStartingOrderKernel(prime, phi, divisor);
        candidateOrder = ExponentLoweringKernel(candidateOrder, prime, phiFactors, phiExponents, phiFactorCount, divisor);

        if (TryConfirmOrderKernel(
                prime,
                candidateOrder,
                divisor,
                limit,
                primes,
                squares,
                primeCount,
                workFactors,
                workExponents,
                stackProduct,
                statusOut))
        {
            resultOut[0] = candidateOrder;
            statusOut[0] = (byte)PrimeOrderKernelStatus.Found;
            return;
        }

        bool isStrict = mode == 1;
        if (isStrict)
        {
            ulong strictOrder = CalculateByDoublingKernel(prime);
            resultOut[0] = strictOrder;
            statusOut[0] = (byte)PrimeOrderKernelStatus.Found;
            return;
        }

        if (TryHeuristicFinishKernel(
                prime,
                candidateOrder,
                previousOrder,
                hasPreviousOrder,
                divisor,
                limit,
                maxPowChecks,
                primes,
                squares,
                primeCount,
                workFactors,
                workExponents,
                candidates,
                stackIndex,
                stackExponent,
                stackProduct,
                statusOut,
                out ulong confirmedOrder))
        {
            resultOut[0] = confirmedOrder;
            statusOut[0] = (byte)PrimeOrderKernelStatus.Found;
            return;
        }

        ulong fallbackOrder = CalculateByDoublingKernel(prime);
        resultOut[0] = fallbackOrder;
        statusOut[0] = (byte)PrimeOrderKernelStatus.HeuristicUnresolved;
    }

    private static int FactorWithSmallPrimes(
        ulong value,
        uint limit,
        ArrayView1D<uint, Stride1D.Dense> primes,
        ArrayView1D<ulong, Stride1D.Dense> squares,
        int primeCount,
        ArrayView1D<ulong, Stride1D.Dense> factors,
        ArrayView1D<int, Stride1D.Dense> exponents,
        out ulong remaining)
    {
        remaining = value;
        int factorCount = 0;
        long factorLength = factors.Length;
        long exponentLength = exponents.Length;
        int capacity = factorLength < exponentLength ? (int)factorLength : (int)exponentLength;

        for (int i = 0; i < primeCount && remaining > 1UL && factorCount < capacity; i++)
        {
            uint primeCandidate = primes[i];
            if (primeCandidate == 0U || primeCandidate > limit)
            {
                break;
            }

            ulong square = squares[i];
            if (square != 0UL && square > remaining)
            {
                break;
            }

            ulong primeValue = primeCandidate;
            if ((remaining % primeValue) != 0UL)
            {
                continue;
            }

            int exponent = 0;
            do
            {
                remaining /= primeValue;
                exponent++;
            }
            while ((remaining % primeValue) == 0UL);

            factors[factorCount] = primeValue;
            exponents[factorCount] = exponent;
            factorCount++;
        }

        for (int i = factorCount; i < factors.Length; i++)
        {
            factors[i] = 0UL;
        }

        for (int i = factorCount; i < exponents.Length; i++)
        {
            exponents[i] = 0;
        }

        return factorCount;
    }

    private static void SortFactors(ArrayView1D<ulong, Stride1D.Dense> factors, ArrayView1D<int, Stride1D.Dense> exponents, int count)
    {
        for (int i = 1; i < count; i++)
        {
            ulong factor = factors[i];
            int exponent = exponents[i];
            int j = i - 1;

            while (j >= 0 && factors[j] > factor)
            {
                factors[j + 1] = factors[j];
                exponents[j + 1] = exponents[j];
                j--;
            }

            factors[j + 1] = factor;
            exponents[j + 1] = exponent;
        }
    }

    private static bool TrySpecialMaxKernel(
        ulong phi,
        ulong prime,
        ArrayView1D<ulong, Stride1D.Dense> factors,
        int factorCount,
        in MontgomeryDivisorData divisor)
    {
        for (int i = 0; i < factorCount; i++)
        {
            ulong factor = factors[i];
            // The factorization routine emits strictly greater-than-one factors for phi on this path, so the
            // defensive continue would never execute.
            // if (factor <= 1UL)
            // {
            //     continue;
            // }

            ulong reduced = phi / factor;
            if (reduced.Pow2ModWindowedGpu(divisor.Modulus) == 1UL)
            {
                return false;
            }
        }

        return true;
    }

    private static ulong InitializeStartingOrderKernel(ulong prime, ulong phi, in MontgomeryDivisorData divisor)
    {
        ulong order = phi;
        ulong residue = prime & 7UL;
        if (residue == 1UL || residue == 7UL)
        {
            ulong half = phi >> 1;
            if ((half).Pow2ModWindowedGpu(divisor.Modulus) == 1UL)
            {
                order = half;
            }
        }

        return order;
    }

    private static ulong ExponentLoweringKernel(
        ulong order,
        ulong prime,
        ArrayView1D<ulong, Stride1D.Dense> factors,
        ArrayView1D<int, Stride1D.Dense> exponents,
        int factorCount,
        in MontgomeryDivisorData divisor)
    {
        for (int i = 0; i < factorCount; i++)
        {
            ulong primeFactor = factors[i];
            int exponent = exponents[i];
            // Factorization never yields unity on the scanning path, so skip the redundant guard.
            // if (primeFactor <= 1UL)
            // {
            //     continue;
            // }

            for (int iteration = 0; iteration < exponent; iteration++)
            {
                if ((order % primeFactor) != 0UL)
                {
                    break;
                }

                ulong reduced = order / primeFactor;
                if ((reduced).Pow2ModWindowedGpu(divisor.Modulus) == 1UL)
                {
                    order = reduced;
                    continue;
                }

                break;
            }
        }

        return order;
    }

    private static bool TryConfirmOrderKernel(
        ulong prime,
        ulong order,
        in MontgomeryDivisorData divisor,
        uint limit,
        ArrayView1D<uint, Stride1D.Dense> primes,
        ArrayView1D<ulong, Stride1D.Dense> squares,
        int primeCount,
        ArrayView1D<ulong, Stride1D.Dense> factors,
        ArrayView1D<int, Stride1D.Dense> exponents,
        ArrayView1D<ulong, Stride1D.Dense> compositeStack,
        ArrayView1D<byte, Stride1D.Dense> statusOut)
    {
        // Order values handed to the kernel are strictly positive on the production path, so the zero guard stays commented
        // out to remove the redundant branch.
        // if (order == 0UL)
        // {
        //     return false;
        // }

        if ((order).Pow2ModWindowedGpu(divisor.Modulus) != 1UL)
        {
            return false;
        }

        int factorCount = FactorWithSmallPrimes(order, limit, primes, squares, primeCount, factors, exponents, out ulong remaining);
        if (remaining != 1UL)
        {
            if (!TryFactorWithPollardKernel(
                    remaining,
                    limit,
                    primes,
                    squares,
                    primeCount,
                    factors,
                    exponents,
                    ref factorCount,
                    compositeStack,
                    statusOut))
            {
                return false;
            }
        }

        SortFactors(factors, exponents, factorCount);

        for (int i = 0; i < factorCount; i++)
        {
            ulong primeFactor = factors[i];
            int exponent = exponents[i];
            // Factorization never yields one on the by-divisor path, so the defensive continue stays commented out.
            // if (primeFactor <= 1UL)
            // {
            //     continue;
            // }

            ulong reduced = order;
            for (int iteration = 0; iteration < exponent; iteration++)
            {
                if ((reduced % primeFactor) != 0UL)
                {
                    break;
                }

                reduced /= primeFactor;
                if ((reduced).Pow2ModWindowedGpu(divisor.Modulus) == 1UL)
                {
                    return false;
                }
            }
        }

        return true;
    }

    private static bool TryHeuristicFinishKernel(
        ulong prime,
        ulong order,
        ulong previousOrder,
        byte hasPreviousOrder,
        in MontgomeryDivisorData divisor,
        uint limit,
        int maxPowChecks,
        ArrayView1D<uint, Stride1D.Dense> primes,
        ArrayView1D<ulong, Stride1D.Dense> squares,
        int primeCount,
        ArrayView1D<ulong, Stride1D.Dense> workFactors,
        ArrayView1D<int, Stride1D.Dense> workExponents,
        ArrayView1D<ulong, Stride1D.Dense> candidates,
        ArrayView1D<int, Stride1D.Dense> stackIndex,
        ArrayView1D<int, Stride1D.Dense> stackExponent,
        ArrayView1D<ulong, Stride1D.Dense> stackProduct,
        ArrayView1D<byte, Stride1D.Dense> statusOut,
        out ulong confirmedOrder)
    {
        confirmedOrder = 0UL;

        if (order <= 1UL)
        {
            return false;
        }

        int factorCount = FactorWithSmallPrimes(order, limit, primes, squares, primeCount, workFactors, workExponents, out ulong remaining);
        if (remaining != 1UL)
        {
            // Reuse stackProduct as the Pollard Rho stack while factoring the order candidates.
            if (!TryFactorWithPollardKernel(
                    remaining,
                    limit,
                    primes,
                    squares,
                    primeCount,
                    workFactors,
                    workExponents,
                    ref factorCount,
                    stackProduct,
                    statusOut))
            {
                return false;
            }
        }

        SortFactors(workFactors, workExponents, factorCount);

        long candidateCapacity = candidates.Length;
        int candidateLimit = candidateCapacity < HeuristicCandidateLimit ? (int)candidateCapacity : HeuristicCandidateLimit;
        int candidateCount = BuildCandidatesKernel(order, workFactors, workExponents, factorCount, candidates, stackIndex, stackExponent, stackProduct, candidateLimit);
        if (candidateCount == 0)
        {
            return false;
        }

        SortCandidatesKernel(prime, previousOrder, hasPreviousOrder != 0, candidates, candidateCount);

        int powBudget = maxPowChecks <= 0 ? candidateCount : maxPowChecks;
        // The heuristics always reserve at least one check, so the zero-budget fallback remains commented out.
        // if (powBudget <= 0)
        // {
        //     powBudget = candidateCount;
        // }

        int powUsed = 0;

        for (int i = 0; i < candidateCount && powUsed < powBudget; i++)
        {
            ulong candidate = candidates[i];
            // Candidate generation skips 1 by construction, so the defensive continue stays commented out.
            // if (candidate <= 1UL)
            // {
            //     continue;
            // }

            if (powUsed >= powBudget)
            {
                break;
            }

            powUsed++;
            if ((candidate).Pow2ModWindowedGpu(divisor.Modulus) != 1UL)
            {
                continue;
            }

            if (!TryConfirmCandidateKernel(
                    prime,
                    candidate,
                    divisor,
                    limit,
                    primes,
                    squares,
                    primeCount,
                    workFactors,
                    workExponents,
                    stackProduct,
                    statusOut,
                    ref powUsed,
                    powBudget))
            {
                continue;
            }

            confirmedOrder = candidate;
            return true;
        }

        return false;
    }

    private static int BuildCandidatesKernel(
        ulong order,
        ArrayView1D<ulong, Stride1D.Dense> factors,
        ArrayView1D<int, Stride1D.Dense> exponents,
        int factorCount,
        ArrayView1D<ulong, Stride1D.Dense> candidates,
        ArrayView1D<int, Stride1D.Dense> stackIndex,
        ArrayView1D<int, Stride1D.Dense> stackExponent,
        ArrayView1D<ulong, Stride1D.Dense> stackProduct,
        int limit)
    {
        long stackIndexLength = stackIndex.Length;
        long stackExponentLength = stackExponent.Length;
        long stackProductLength = stackProduct.Length;
        if (factorCount == 0 || limit <= 0 || stackIndexLength == 0L || stackExponentLength == 0L || stackProductLength == 0L)
        {
            return 0;
        }

        int stackCapacity = stackIndexLength < stackExponentLength ? (int)stackIndexLength : (int)stackExponentLength;
        if (stackProductLength < stackCapacity)
        {
            stackCapacity = (int)stackProductLength;
        }

        int candidateCount = 0;
        int stackTop = 0;

        stackIndex[0] = 0;
        stackExponent[0] = 0;
        stackProduct[0] = 1UL;
        stackTop = 1;

        while (stackTop > 0)
        {
            stackTop--;
            int index = stackIndex[stackTop];
            int exponent = stackExponent[stackTop];
            ulong product = stackProduct[stackTop];

            if (index >= factorCount)
            {
                if (product != 1UL && product != order && candidateCount < limit)
                {
                    ulong candidate = order / product;
                    if (candidate > 1UL && candidate < order)
                    {
                        candidates[candidateCount] = candidate;
                        candidateCount++;
                    }
                }

                continue;
            }

            int maxExponent = exponents[index];
            if (exponent > maxExponent)
            {
                continue;
            }

            if (stackTop >= stackCapacity)
            {
                return candidateCount;
            }

            stackIndex[stackTop] = index + 1;
            stackExponent[stackTop] = 0;
            stackProduct[stackTop] = product;
            stackTop++;

            if (exponent == maxExponent)
            {
                continue;
            }

            ulong primeFactor = factors[index];
            if (primeFactor == 0UL || product > order / primeFactor)
            {
                continue;
            }

            if (stackTop >= stackCapacity)
            {
                return candidateCount;
            }

            stackIndex[stackTop] = index;
            stackExponent[stackTop] = exponent + 1;
            stackProduct[stackTop] = product * primeFactor;
            stackTop++;
        }

        return candidateCount;
    }

    private static void SortCandidatesKernel(
        ulong prime,
        ulong previousOrder,
        bool hasPrevious,
        ArrayView1D<ulong, Stride1D.Dense> candidates,
        int count)
    {
        for (int i = 1; i < count; i++)
        {
            ulong value = candidates[i];
            CandidateKey key = BuildCandidateKey(value, prime, previousOrder, hasPrevious);
            int j = i - 1;

            while (j >= 0)
            {
                CandidateKey other = BuildCandidateKey(candidates[j], prime, previousOrder, hasPrevious);
                if (other.CompareTo(key) <= 0)
                {
                    break;
                }

                candidates[j + 1] = candidates[j];
                j--;
            }

            candidates[j + 1] = value;
        }
    }

    private static CandidateKey BuildCandidateKey(ulong value, ulong prime, ulong previousOrder, bool hasPrevious)
    {
        int group = GetGroup(value, prime);
        if (group == 0)
        {
            return new CandidateKey(int.MaxValue, long.MaxValue, long.MaxValue);
        }

        ulong reference = hasPrevious ? previousOrder : 0UL;
        bool isGe = !hasPrevious || value >= reference;
        int previousGroup = hasPrevious ? GetGroup(reference, prime) : 1;
        int primary = ComputePrimary(group, isGe, previousGroup);
        long secondary;
        long tertiary;

        if (group == 3)
        {
            secondary = -(long)value;
            tertiary = -(long)value;
        }
        else
        {
            ulong distance = hasPrevious ? (value > reference ? value - reference : reference - value) : value;
            secondary = (long)distance;
            tertiary = (long)value;
        }

        return new CandidateKey(primary, secondary, tertiary);
    }

    private static int GetGroup(ulong value, ulong prime)
    {
        ulong threshold1 = prime >> 3;
        if (value <= threshold1)
        {
            return 1;
        }

        ulong threshold2 = prime >> 2;
        if (value <= threshold2)
        {
            return 2;
        }

        ulong threshold3 = (prime * 3UL) >> 3;
        if (value <= threshold3)
        {
            return 3;
        }

        return 0;
    }

    private static int ComputePrimary(int group, bool isGe, int previousGroup)
    {
        int groupOffset;
        switch (group)
        {
            case 1:
                groupOffset = 0;
                break;
            case 2:
                groupOffset = 2;
                break;
            case 3:
                groupOffset = 4;
                break;
            default:
                groupOffset = 6;
                break;
        }

        if (group == previousGroup)
        {
            if (group == 3)
            {
                return groupOffset + (isGe ? 0 : 3);
            }

            return groupOffset + (isGe ? 0 : 1);
        }

        return groupOffset + (isGe ? 0 : 1);
    }

    private static bool TryConfirmCandidateKernel(
        ulong prime,
        ulong candidate,
        in MontgomeryDivisorData divisor,
        uint limit,
        ArrayView1D<uint, Stride1D.Dense> primes,
        ArrayView1D<ulong, Stride1D.Dense> squares,
        int primeCount,
        ArrayView1D<ulong, Stride1D.Dense> factors,
        ArrayView1D<int, Stride1D.Dense> exponents,
        ArrayView1D<ulong, Stride1D.Dense> compositeStack,
        ArrayView1D<byte, Stride1D.Dense> statusOut,
        ref int powUsed,
        int powBudget)
    {
        int factorCount = FactorWithSmallPrimes(candidate, limit, primes, squares, primeCount, factors, exponents, out ulong remaining);
        if (remaining != 1UL)
        {
            if (!TryFactorWithPollardKernel(
                    remaining,
                    limit,
                    primes,
                    squares,
                    primeCount,
                    factors,
                    exponents,
                    ref factorCount,
                    compositeStack,
                    statusOut))
            {
                return false;
            }
        }

        SortFactors(factors, exponents, factorCount);

        for (int i = 0; i < factorCount; i++)
        {
            ulong primeFactor = factors[i];
            int exponent = exponents[i];
            // Factorization never yields one for candidate orders, so keep the guard commented out to avoid re-checking inputs.
            // if (primeFactor <= 1UL)
            // {
            //     continue;
            // }

            ulong reduced = candidate;
            for (int iteration = 0; iteration < exponent; iteration++)
            {
                if ((reduced % primeFactor) != 0UL)
                {
                    break;
                }

                reduced /= primeFactor;
                if (powBudget > 0 && powUsed >= powBudget)
                {
                    return false;
                }

                powUsed++;
                if ((reduced).Pow2ModWindowedGpu(divisor.Modulus) == 1UL)
                {
                    return false;
                }
            }
        }

        return true;
    }

    private static bool TryFactorWithPollardKernel(
        ulong initial,
        uint limit,
        ArrayView1D<uint, Stride1D.Dense> primes,
        ArrayView1D<ulong, Stride1D.Dense> squares,
        int primeCount,
        ArrayView1D<ulong, Stride1D.Dense> factors,
        ArrayView1D<int, Stride1D.Dense> exponents,
        ref int factorCount,
        ArrayView1D<ulong, Stride1D.Dense> compositeStack,
        ArrayView1D<byte, Stride1D.Dense> statusOut)
    {
        // Composite candidates sent here are always greater than one during production scans, so the trivial-return branch stays
        // commented out.
        // if (initial <= 1UL)
        // {
        //     return true;
        // }

        int stackCapacity = (int)compositeStack.Length;
        if (stackCapacity <= 0)
        {
            statusOut[0] = (byte)PrimeOrderKernelStatus.PollardOverflow;
            return false;
        }

        int stackTop = 0;
        compositeStack[stackTop] = initial;
        stackTop++;

        while (stackTop > 0)
        {
            stackTop--;
            ulong composite = compositeStack[stackTop];
            if (composite <= 1UL)
            {
                continue;
            }

            if (!PeelSmallPrimesKernel(
                    composite,
                    limit,
                    primes,
                    squares,
                    primeCount,
                    factors,
                    exponents,
                    ref factorCount,
                    out ulong remaining))
            {
                statusOut[0] = (byte)PrimeOrderKernelStatus.PollardOverflow;
                return false;
            }

            if (remaining == 1UL)
            {
                continue;
            }

            ulong factor = PollardRhoKernel(remaining);
            if (factor <= 1UL || factor == remaining)
            {
                if (!TryAppendFactorKernel(factors, exponents, ref factorCount, remaining, 1))
                {
                    statusOut[0] = (byte)PrimeOrderKernelStatus.PollardOverflow;
                    return false;
                }

                continue;
            }

            ulong quotient = remaining / factor;
            if (stackTop + 2 > stackCapacity)
            {
                statusOut[0] = (byte)PrimeOrderKernelStatus.PollardOverflow;
                return false;
            }

            compositeStack[stackTop] = factor;
            compositeStack[stackTop + 1] = quotient;
            stackTop += 2;
        }

        return true;
    }

    private static bool PeelSmallPrimesKernel(
        ulong value,
        uint limit,
        ArrayView1D<uint, Stride1D.Dense> primes,
        ArrayView1D<ulong, Stride1D.Dense> squares,
        int primeCount,
        ArrayView1D<ulong, Stride1D.Dense> factors,
        ArrayView1D<int, Stride1D.Dense> exponents,
        ref int factorCount,
        out ulong remaining)
    {
        ulong remainingLocal = value;

        for (int i = 0; i < primeCount && remainingLocal > 1UL; i++)
        {
            uint primeCandidate = primes[i];
            if (primeCandidate == 0U || primeCandidate > limit)
            {
                break;
            }

            ulong square = squares[i];
            if (square != 0UL && square > remainingLocal)
            {
                break;
            }

            ulong primeValue = primeCandidate;
            if ((remainingLocal % primeValue) != 0UL)
            {
                continue;
            }

            int exponent = 0;
            do
            {
                remainingLocal /= primeValue;
                exponent++;
            }
            while ((remainingLocal % primeValue) == 0UL);

            if (!TryAppendFactorKernel(factors, exponents, ref factorCount, primeValue, exponent))
            {
                remaining = value;
                return false;
            }
        }

        remaining = remainingLocal;
        return true;
    }

    private static bool TryAppendFactorKernel(
        ArrayView1D<ulong, Stride1D.Dense> factors,
        ArrayView1D<int, Stride1D.Dense> exponents,
        ref int count,
        ulong prime,
        int exponent)
    {
        if (prime <= 1UL || exponent <= 0)
        {
            return true;
        }

        for (int i = 0; i < count; i++)
        {
            if (factors[i] == prime)
            {
                exponents[i] += exponent;
                return true;
            }
        }

        int capacity = (int)factors.Length;
        if (count >= capacity)
        {
            return false;
        }

        factors[count] = prime;
        exponents[count] = exponent;
        count++;
        return true;
    }

    private static ulong MulModKernel(ulong left, ulong right, ulong modulus)
    {
        GpuUInt128 product = new GpuUInt128(left);
        return product.MulMod(right, modulus);
    }

    private static ulong PollardRhoKernel(ulong value)
    {
        if ((value & 1UL) == 0UL)
        {
            return 2UL;
        }

        ulong c = 1UL;
        while (true)
        {
            ulong x = 2UL;
            ulong y = 2UL;
            ulong d = 1UL;

            while (d == 1UL)
            {
                x = AdvancePolynomialKernel(x, c, value);
                y = AdvancePolynomialKernel(y, c, value);
                y = AdvancePolynomialKernel(y, c, value);

                ulong diff = x > y ? x - y : y - x;
                d = BinaryGcdKernel(diff, value);
            }

            if (d == value)
            {
                c++;
                if (c == 0UL)
                {
                    c = 1UL;
                }

                continue;
            }

            return d;
        }
    }

    private static ulong AdvancePolynomialKernel(ulong x, ulong c, ulong modulus)
    {
        ulong squared = MulModKernel(x, x, modulus);
        GpuUInt128 accumulator = new GpuUInt128(squared);
        accumulator.AddMod(c, modulus);
        return accumulator.Low;
    }

    private static ulong BinaryGcdKernel(ulong a, ulong b)
    {
        if (a == 0UL)
        {
            return b;
        }

        if (b == 0UL)
        {
            return a;
        }

        int shift = BitOperations.TrailingZeroCount(a | b);
        ulong aLocal = a >> BitOperations.TrailingZeroCount(a);
        ulong bLocal = b;

        while (true)
        {
            bLocal >>= BitOperations.TrailingZeroCount(bLocal);
            if (aLocal > bLocal)
            {
                ulong temp = aLocal;
                aLocal = bLocal;
                bLocal = temp;
            }

            bLocal -= aLocal;
            if (bLocal == 0UL)
            {
                return aLocal << shift;
            }
        }
    }

    private static ulong CalculateByDoublingKernel(ulong prime)
    {
        ulong order = 1UL;
        ulong value = 2UL % prime;

        while (value != 1UL)
        {
            value <<= 1;
            if (value >= prime)
            {
                value -= prime;
            }

            order++;
        }

        return order;
    }

    private static GpuPow2ModStatus TryPow2ModBatchInternal(ReadOnlySpan<UInt128> exponents, UInt128 prime, Span<UInt128> remainders)
    {
        // Residue batches are never empty on production workloads, so skip the zero-length success shortcut.
        // if (exponents.Length == 0)
        // {
        //     return GpuPow2ModStatus.Success;
        // }

        if (remainders.Length < exponents.Length)
        {
            throw new ArgumentException("Remainder span is shorter than the exponent span.", nameof(remainders));
        }

        Span<UInt128> target = remainders.Slice(0, exponents.Length);
        target.Clear();

        ConcurrentDictionary<UInt128, byte> overflowRegistryWide = OverflowedPrimesWide;

        if (overflowRegistryWide.ContainsKey(prime))
        {
            return GpuPow2ModStatus.Overflow;
        }

        PrimeOrderGpuCapability capability = s_capability;

        if (prime.GetBitLength() > capability.ModulusBits)
        {
            overflowRegistryWide[prime] = 0;
            return GpuPow2ModStatus.Overflow;
        }

        for (int i = 0; i < exponents.Length; i++)
        {
            if (exponents[i].GetBitLength() > capability.ExponentBits)
            {
                return GpuPow2ModStatus.Overflow;
            }
        }

        bool computed = TryComputeOnGpuWide(exponents, prime, target);
        return computed ? GpuPow2ModStatus.Success : GpuPow2ModStatus.Unavailable;
    }

    private static bool TryComputeOnGpu(ReadOnlySpan<ulong> exponents, ulong prime, in MontgomeryDivisorData divisorData, Span<ulong> results)
    {
        var lease = GpuKernelPool.GetKernel(useGpuOrder: true);
        var execution = lease.EnterExecutionScope();
        Accelerator accelerator = lease.Accelerator;
        AcceleratorStream stream = lease.Stream;
        var kernel = GetPow2ModKernel(accelerator);
        var exponentBuffer = accelerator.Allocate1D<ulong>(exponents.Length);
        var remainderBuffer = accelerator.Allocate1D<ulong>(exponents.Length);

        exponentBuffer.View.CopyFromCPU(ref MemoryMarshal.GetReference(exponents), exponents.Length);
        remainderBuffer.MemSetToZero();

        try
        {
            kernel(stream, exponents.Length, exponentBuffer.View, divisorData, remainderBuffer.View);
            stream.Synchronize();
        }
        catch (Exception)
        {
            Console.WriteLine($"Exception for {prime} and exponents: {string.Join(",", exponents.ToArray())}.");
            exponentBuffer.Dispose();
            remainderBuffer.Dispose();
            execution.Dispose();
            lease.Dispose();
            throw;
        }

        remainderBuffer.View.CopyToCPU(ref MemoryMarshal.GetReference(results), exponents.Length);
        exponentBuffer.Dispose();
        remainderBuffer.Dispose();
        execution.Dispose();
        lease.Dispose();
        return true;
    }

    private static void ComputePow2ModCpu(ReadOnlySpan<ulong> exponents, ulong prime, in MontgomeryDivisorData divisorData, Span<ulong> results)
    {
        int length = exponents.Length;
        ulong modulus = divisorData.Modulus;
        _ = prime; // modulus and divisorData both describe the same q; retain the parameter to preserve the public signature.
        for (int i = 0; i < length; i++)
        {
            results[i] = exponents[i].Pow2ModWindowedCpu(modulus);
        }
    }

    private static bool TryComputeOnGpuWide(ReadOnlySpan<UInt128> exponents, UInt128 prime, Span<UInt128> results)
    {
        int length = exponents.Length;
        // Production launches always carry work, so keep the empty-batch shortcut commented out.
        // if (length == 0)
        // {
        //     return true;
        // }

        GpuUInt128[]? rentedExponents = null;
        GpuUInt128[]? rentedResults = null;
        var lease = GpuKernelPool.GetKernel(useGpuOrder: true);
        var execution = lease.EnterExecutionScope();

        try
        {
            Accelerator accelerator = lease.Accelerator;
            AcceleratorStream stream = lease.Stream;
            var kernel = GetPow2ModWideKernel(accelerator);
            var exponentBuffer = accelerator.Allocate1D<GpuUInt128>(length);
            var remainderBuffer = accelerator.Allocate1D<GpuUInt128>(length);

            Span<GpuUInt128> exponentSpan = length <= WideStackThreshold
                ? stackalloc GpuUInt128[length]
                : new Span<GpuUInt128>(rentedExponents = ArrayPool<GpuUInt128>.Shared.Rent(length), 0, length);

            for (int i = 0; i < length; i++)
            {
                exponentSpan[i] = (GpuUInt128)exponents[i];
            }

            exponentBuffer.View.CopyFromCPU(ref MemoryMarshal.GetReference(exponentSpan), length);
            remainderBuffer.MemSetToZero();

            GpuUInt128 modulus = (GpuUInt128)prime;
            kernel(stream, length, exponentBuffer.View, modulus, remainderBuffer.View);

            stream.Synchronize();

            Span<GpuUInt128> resultSpan = length <= WideStackThreshold
                ? stackalloc GpuUInt128[length]
                : new Span<GpuUInt128>(rentedResults = ArrayPool<GpuUInt128>.Shared.Rent(length), 0, length);

            remainderBuffer.View.CopyToCPU(ref MemoryMarshal.GetReference(resultSpan), length);

            for (int i = 0; i < length; i++)
            {
                results[i] = (UInt128)resultSpan[i];
            }

            exponentBuffer.Dispose();
            remainderBuffer.Dispose();
            return true;
        }
        catch (Exception)
        {
            Console.WriteLine($"Exception when computing {prime} with exponents: {string.Join(", ", exponents.ToArray())}");
            throw;
        }
        finally
        {
            // Small batches use the stack-allocated fast path, leaving the rented arrays null. Retain the guards so pooled
            // buffers are only returned when they were actually leased.
            if (rentedExponents is not null)
            {
                ArrayPool<GpuUInt128>.Shared.Return(rentedExponents, clearArray: false);
            }

            if (rentedResults is not null)
            {
                ArrayPool<GpuUInt128>.Shared.Return(rentedResults, clearArray: false);
            }

            execution.Dispose();
            lease.Dispose();
        }
    }

    private static void ComputePow2ModCpuWide(ReadOnlySpan<UInt128> exponents, UInt128 prime, Span<UInt128> results)
    {
        int length = exponents.Length;
        for (int i = 0; i < length; i++)
        {
            results[i] = exponents[i].Pow2MontgomeryModWindowed(prime);
        }
    }

    private static Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>> GetPow2ModKernel(Accelerator accelerator)
    {
        return Pow2ModKernelCache.GetOrAdd(accelerator, static accel =>
        {
            var loaded = accel.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>(Pow2ModKernel);
            var kernel = KernelUtil.GetKernel(loaded);
            return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView1D<ulong, Stride1D.Dense>, MontgomeryDivisorData, ArrayView1D<ulong, Stride1D.Dense>>>();
        });
    }

    private static void Pow2ModKernel(Index1D index, ArrayView1D<ulong, Stride1D.Dense> exponents, MontgomeryDivisorData divisor, ArrayView1D<ulong, Stride1D.Dense> remainders)
    {
        ulong modulus = divisor.Modulus;
        // The by-divisor heuristics call this kernel only with odd prime moduli, so the defensive branch stays commented to
        // document the invariant without spending a branch in the kernel.
        // if (modulus <= 1UL || (modulus & 1UL) == 0UL)
        // {
        //     remainders[index] = 0UL;
        //     return;
        // }

        ulong exponent = exponents[index];
        remainders[index] = exponent.Pow2ModWindowedGpu(modulus);
    }

    private static Action<AcceleratorStream, Index1D, ArrayView1D<GpuUInt128, Stride1D.Dense>, GpuUInt128, ArrayView1D<GpuUInt128, Stride1D.Dense>> GetPow2ModWideKernel(Accelerator accelerator)
    {
        return Pow2ModKernelWideCache.GetOrAdd(accelerator, static accel =>
        {
            var loaded = accel.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<GpuUInt128, Stride1D.Dense>, GpuUInt128, ArrayView1D<GpuUInt128, Stride1D.Dense>>(Pow2ModKernelWide);
            var kernel = KernelUtil.GetKernel(loaded);
            return kernel.CreateLauncherDelegate<Action<AcceleratorStream, Index1D, ArrayView1D<GpuUInt128, Stride1D.Dense>, GpuUInt128, ArrayView1D<GpuUInt128, Stride1D.Dense>>>();
        });
    }

    private static void Pow2ModKernelWide(Index1D index, ArrayView1D<GpuUInt128, Stride1D.Dense> exponents, GpuUInt128 modulus, ArrayView1D<GpuUInt128, Stride1D.Dense> remainders)
    {
        GpuUInt128 exponent = exponents[index];
        remainders[index] = Pow2ModKernelCore(exponent, modulus);
    }

    private static GpuUInt128[] InitializeOddPowersTable(GpuUInt128 baseValue, GpuUInt128 modulus, int oddPowerCount)
    {
        GpuUInt128[] result = new GpuUInt128[PerfectNumberConstants.MaxOddPowersCount];
        result[0] = baseValue;
        if (oddPowerCount == 1)
        {
            return result;
        }

        // Reusing baseValue to hold base^2 for the shared odd-power ladder that follows.
        baseValue.MulMod(baseValue, modulus);

        // TODO: We can calculate baseValue % modulus before loop and use it to increase ladderEntry calculation speed - we'll reuse the base for incremental calculations.
        GpuUInt128 current = baseValue;

        // We're manually assigning each field to prevent the compiler to initialize each field twice due to auto-initialization. We're using the action to lower the code base size.
        for (int i = 1; i < oddPowerCount; i++)
        {
            current.MulMod(baseValue, modulus);
            result[i] = current;
        }

        return result;
    }

    private static GpuUInt128 Pow2ModKernelCore(GpuUInt128 exponent, GpuUInt128 modulus)
    {
        // This should never happen in production code.
        // if (modulus == GpuUInt128.One)
        // {
        //     return GpuUInt128.Zero;
        // }

        // This should never happen in production code.
        // if (exponent.IsZero)
        // {
        //     return GpuUInt128.One;
        // }

        GpuUInt128 baseValue = UInt128Numbers.TwoGpu;

        // This should never happen in production code - 2 should never be greater or equal modulus.
        // if (baseValue.CompareTo(modulus) >= 0)
        // {
        //     baseValue.Sub(modulus);
        // }

        if (ShouldUseSingleBit(exponent))
        {
            return Pow2MontgomeryModSingleBit(exponent, modulus, baseValue);
        }

        int index = exponent.GetBitLength();
        int windowSize = GetWindowSize(index);
        index--;

        int oddPowerCount = 1 << (windowSize - 1);
        GpuUInt128[] oddPowers = InitializeOddPowersTable(baseValue, modulus, oddPowerCount);
        GpuUInt128 result = GpuUInt128.One;

        int windowStart;
        ulong windowValue;
        while (index >= 0)
        {
            if (!IsBitSet(exponent, index))
            {
                result.MulMod(result, modulus);
                index--;
                continue;
            }

            windowStart = index - windowSize + 1;
            if (windowStart < 0)
            {
                windowStart = 0;
            }

            while (!IsBitSet(exponent, windowStart))
            {
                windowStart++;
            }

            // We're reusing oddPowerCount as windowBitCount here to lower registry pressure & avoid additional allocation
            oddPowerCount = index - windowStart + 1;
            index = windowStart - 1;
            windowValue = ExtractWindowValue(exponent, windowStart, oddPowerCount);

            // We're reusing windowStart as square here to lower registry pressure & avoid additional allocation
            for (windowStart = 0; windowStart < oddPowerCount; windowStart++)
            {
                result.MulMod(result, modulus);
            }

            // We're reusing windowStart as tableIndex here to lower registry pressure & avoid additional allocation
            windowStart = (int)((windowValue - 1UL) >> 1);
            result.MulMod(oddPowers[windowStart], modulus);
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool ShouldUseSingleBit(GpuUInt128 exponent) => exponent.High == 0UL && exponent.Low <= Pow2WindowFallbackThreshold;

    private static int GetWindowSize(int bitLength)
    {
        if (bitLength <= Pow2WindowSizeBits)
        {
            return Math.Max(bitLength, 1);
        }

        if (bitLength <= 23)
        {
            return 4;
        }

        if (bitLength <= 79)
        {
            return 5;
        }

        if (bitLength <= 239)
        {
            return 6;
        }

        if (bitLength <= 671)
        {
            return 7;
        }

        return Pow2WindowSizeBits;
    }

    private static GpuUInt128 Pow2MontgomeryModSingleBit(GpuUInt128 exponent, GpuUInt128 modulus, GpuUInt128 baseValue)
    {
        GpuUInt128 result = GpuUInt128.One;

        while (!exponent.IsZero)
        {
            if ((exponent.Low & 1UL) != 0UL)
            {
                result.MulMod(baseValue, modulus);
            }

            exponent.ShiftRight(1);
            if (exponent.IsZero)
            {
                break;
            }

            // Reusing baseValue to store the squared base for the next iteration.
            baseValue.MulMod(baseValue, modulus);
        }

        return result;
    }

    private static bool IsBitSet(GpuUInt128 value, int bitIndex)
    {
        if (bitIndex >= 64)
        {
            return ((value.High >> (bitIndex - 64)) & 1UL) != 0UL;
        }

        return ((value.Low >> bitIndex) & 1UL) != 0UL;
    }

    private static ulong ExtractWindowValue(GpuUInt128 exponent, int windowStart, int windowBitCount)
    {
        if (windowStart != 0)
        {
            exponent.ShiftRight(windowStart);
            ulong mask = (1UL << windowBitCount) - 1UL;
            return exponent.Low & mask;
        }

        ulong directMask = (1UL << windowBitCount) - 1UL;
        return exponent.Low & directMask;
    }

    internal readonly record struct PrimeOrderGpuCapability(int ModulusBits, int ExponentBits)
    {
        public static PrimeOrderGpuCapability Default => new(128, 128);
    }
}
