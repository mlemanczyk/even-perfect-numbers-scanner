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

    private static GpuPow2ModStatus TryPow2ModBatchInternal(ReadOnlySpan<UInt128> exponents, UInt128 prime, Span<UInt128> remainders)
    {
        if (exponents.Length == 0)
        {
            return GpuPow2ModStatus.Success;
        }

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
        for (int i = 0; i < length; i++)
        {
            results[i] = Pow2ModCpu(exponents[i], prime, divisorData);
        }
    }

    private static bool TryComputeOnGpuWide(ReadOnlySpan<UInt128> exponents, UInt128 prime, Span<UInt128> results)
    {
        int length = exponents.Length;
        if (length == 0)
        {
            return true;
        }

        GpuUInt128[]? rentedExponents = null;
        GpuUInt128[]? rentedResults = null;
        var lease = GpuKernelPool.GetKernel(useGpuOrder: true);

        try
        {
            var execution = lease.EnterExecutionScope();
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
            execution.Dispose();
            return true;
        }
        catch (Exception)
        {
            Console.WriteLine($"Exception when computing {prime} with exponents: {string.Join(", ", exponents.ToArray())}");
            throw;
        }
        finally
        {
            if (rentedExponents is not null)
            {
                ArrayPool<GpuUInt128>.Shared.Return(rentedExponents, clearArray: false);
            }

            if (rentedResults is not null)
            {
                ArrayPool<GpuUInt128>.Shared.Return(rentedResults, clearArray: false);
            }

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

    private static ulong Pow2ModCpu(ulong exponent, ulong modulus, in MontgomeryDivisorData divisorData)
    {
        if (modulus <= 1UL)
        {
            return 0UL;
        }

        return ULongExtensions.Pow2MontgomeryModWindowedGpu(divisorData, exponent, keepMontgomery: false);
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




    internal readonly record struct PrimeOrderGpuCapability(int ModulusBits, int ExponentBits)
    {
        public static PrimeOrderGpuCapability Default => new(128, 128);
    }
}
