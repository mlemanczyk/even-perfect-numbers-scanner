using System;
using System.Buffers;
using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;
using System.Numerics;
using PerfectNumbers.Core;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
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

    public readonly struct OrderKernelConfig
    {
        public OrderKernelConfig(ulong previousOrder, byte hasPreviousOrder, uint smallFactorLimit, int maxPowChecks, int mode)
        {
            PreviousOrder = previousOrder;
            HasPreviousOrder = hasPreviousOrder;
            SmallFactorLimit = smallFactorLimit;
            MaxPowChecks = maxPowChecks;
            Mode = mode;
        }

        public ulong PreviousOrder { get; }

        public byte HasPreviousOrder { get; }

        public uint SmallFactorLimit { get; }

        public int MaxPowChecks { get; }

        public int Mode { get; }
    }

    public readonly struct OrderKernelBuffers
    {
        public OrderKernelBuffers(
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
            PhiFactors = phiFactors;
            PhiExponents = phiExponents;
            WorkFactors = workFactors;
            WorkExponents = workExponents;
            Candidates = candidates;
            StackIndex = stackIndex;
            StackExponent = stackExponent;
            StackProduct = stackProduct;
            Result = result;
            Status = status;
        }

        public ArrayView1D<ulong, Stride1D.Dense> PhiFactors { get; }

        public ArrayView1D<int, Stride1D.Dense> PhiExponents { get; }

        public ArrayView1D<ulong, Stride1D.Dense> WorkFactors { get; }

        public ArrayView1D<int, Stride1D.Dense> WorkExponents { get; }

        public ArrayView1D<ulong, Stride1D.Dense> Candidates { get; }

        public ArrayView1D<int, Stride1D.Dense> StackIndex { get; }

        public ArrayView1D<int, Stride1D.Dense> StackExponent { get; }

        public ArrayView1D<ulong, Stride1D.Dense> StackProduct { get; }

        public ArrayView1D<ulong, Stride1D.Dense> Result { get; }

        public ArrayView1D<byte, Stride1D.Dense> Status { get; }
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


    private struct Pow2OddPowerTable
    {
        public GpuUInt128 Element0;
        public GpuUInt128 Element1;
        public GpuUInt128 Element2;
        public GpuUInt128 Element3;
        public GpuUInt128 Element4;
        public GpuUInt128 Element5;
        public GpuUInt128 Element6;
        public GpuUInt128 Element7;
        public GpuUInt128 Element8;
        public GpuUInt128 Element9;
        public GpuUInt128 Element10;
        public GpuUInt128 Element11;
        public GpuUInt128 Element12;
        public GpuUInt128 Element13;
        public GpuUInt128 Element14;
        public GpuUInt128 Element15;
        public GpuUInt128 Element16;
        public GpuUInt128 Element17;
        public GpuUInt128 Element18;
        public GpuUInt128 Element19;
        public GpuUInt128 Element20;
        public GpuUInt128 Element21;
        public GpuUInt128 Element22;
        public GpuUInt128 Element23;
        public GpuUInt128 Element24;
        public GpuUInt128 Element25;
        public GpuUInt128 Element26;
        public GpuUInt128 Element27;
        public GpuUInt128 Element28;
        public GpuUInt128 Element29;
        public GpuUInt128 Element30;
        public GpuUInt128 Element31;
        public GpuUInt128 Element32;
        public GpuUInt128 Element33;
        public GpuUInt128 Element34;
        public GpuUInt128 Element35;
        public GpuUInt128 Element36;
        public GpuUInt128 Element37;
        public GpuUInt128 Element38;
        public GpuUInt128 Element39;
        public GpuUInt128 Element40;
        public GpuUInt128 Element41;
        public GpuUInt128 Element42;
        public GpuUInt128 Element43;
        public GpuUInt128 Element44;
        public GpuUInt128 Element45;
        public GpuUInt128 Element46;
        public GpuUInt128 Element47;
        public GpuUInt128 Element48;
        public GpuUInt128 Element49;
        public GpuUInt128 Element50;
        public GpuUInt128 Element51;
        public GpuUInt128 Element52;
        public GpuUInt128 Element53;
        public GpuUInt128 Element54;
        public GpuUInt128 Element55;
        public GpuUInt128 Element56;
        public GpuUInt128 Element57;
        public GpuUInt128 Element58;
        public GpuUInt128 Element59;
        public GpuUInt128 Element60;
        public GpuUInt128 Element61;
        public GpuUInt128 Element62;
        public GpuUInt128 Element63;
        public GpuUInt128 Element64;
        public GpuUInt128 Element65;
        public GpuUInt128 Element66;
        public GpuUInt128 Element67;
        public GpuUInt128 Element68;
        public GpuUInt128 Element69;
        public GpuUInt128 Element70;
        public GpuUInt128 Element71;
        public GpuUInt128 Element72;
        public GpuUInt128 Element73;
        public GpuUInt128 Element74;
        public GpuUInt128 Element75;
        public GpuUInt128 Element76;
        public GpuUInt128 Element77;
        public GpuUInt128 Element78;
        public GpuUInt128 Element79;
        public GpuUInt128 Element80;
        public GpuUInt128 Element81;
        public GpuUInt128 Element82;
        public GpuUInt128 Element83;
        public GpuUInt128 Element84;
        public GpuUInt128 Element85;
        public GpuUInt128 Element86;
        public GpuUInt128 Element87;
        public GpuUInt128 Element88;
        public GpuUInt128 Element89;
        public GpuUInt128 Element90;
        public GpuUInt128 Element91;
        public GpuUInt128 Element92;
        public GpuUInt128 Element93;
        public GpuUInt128 Element94;
        public GpuUInt128 Element95;
        public GpuUInt128 Element96;
        public GpuUInt128 Element97;
        public GpuUInt128 Element98;
        public GpuUInt128 Element99;
        public GpuUInt128 Element100;
        public GpuUInt128 Element101;
        public GpuUInt128 Element102;
        public GpuUInt128 Element103;
        public GpuUInt128 Element104;
        public GpuUInt128 Element105;
        public GpuUInt128 Element106;
        public GpuUInt128 Element107;
        public GpuUInt128 Element108;
        public GpuUInt128 Element109;
        public GpuUInt128 Element110;
        public GpuUInt128 Element111;
        public GpuUInt128 Element112;
        public GpuUInt128 Element113;
        public GpuUInt128 Element114;
        public GpuUInt128 Element115;
        public GpuUInt128 Element116;
        public GpuUInt128 Element117;
        public GpuUInt128 Element118;
        public GpuUInt128 Element119;
        public GpuUInt128 Element120;
        public GpuUInt128 Element121;
        public GpuUInt128 Element122;
        public GpuUInt128 Element123;
        public GpuUInt128 Element124;
        public GpuUInt128 Element125;
        public GpuUInt128 Element126;
        public GpuUInt128 Element127;

        public GpuUInt128 this[int index]
        {
            readonly get
            {
                return GetElement(index);
            }

            set
            {
                SetElement(index, value);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private readonly GpuUInt128 GetElement(int index) => index switch
        {
            0 => Element0,
            1 => Element1,
            2 => Element2,
            3 => Element3,
            4 => Element4,
            5 => Element5,
            6 => Element6,
            7 => Element7,
            8 => Element8,
            9 => Element9,
            10 => Element10,
            11 => Element11,
            12 => Element12,
            13 => Element13,
            14 => Element14,
            15 => Element15,
            16 => Element16,
            17 => Element17,
            18 => Element18,
            19 => Element19,
            20 => Element20,
            21 => Element21,
            22 => Element22,
            23 => Element23,
            24 => Element24,
            25 => Element25,
            26 => Element26,
            27 => Element27,
            28 => Element28,
            29 => Element29,
            30 => Element30,
            31 => Element31,
            32 => Element32,
            33 => Element33,
            34 => Element34,
            35 => Element35,
            36 => Element36,
            37 => Element37,
            38 => Element38,
            39 => Element39,
            40 => Element40,
            41 => Element41,
            42 => Element42,
            43 => Element43,
            44 => Element44,
            45 => Element45,
            46 => Element46,
            47 => Element47,
            48 => Element48,
            49 => Element49,
            50 => Element50,
            51 => Element51,
            52 => Element52,
            53 => Element53,
            54 => Element54,
            55 => Element55,
            56 => Element56,
            57 => Element57,
            58 => Element58,
            59 => Element59,
            60 => Element60,
            61 => Element61,
            62 => Element62,
            63 => Element63,
            64 => Element64,
            65 => Element65,
            66 => Element66,
            67 => Element67,
            68 => Element68,
            69 => Element69,
            70 => Element70,
            71 => Element71,
            72 => Element72,
            73 => Element73,
            74 => Element74,
            75 => Element75,
            76 => Element76,
            77 => Element77,
            78 => Element78,
            79 => Element79,
            80 => Element80,
            81 => Element81,
            82 => Element82,
            83 => Element83,
            84 => Element84,
            85 => Element85,
            86 => Element86,
            87 => Element87,
            88 => Element88,
            89 => Element89,
            90 => Element90,
            91 => Element91,
            92 => Element92,
            93 => Element93,
            94 => Element94,
            95 => Element95,
            96 => Element96,
            97 => Element97,
            98 => Element98,
            99 => Element99,
            100 => Element100,
            101 => Element101,
            102 => Element102,
            103 => Element103,
            104 => Element104,
            105 => Element105,
            106 => Element106,
            107 => Element107,
            108 => Element108,
            109 => Element109,
            110 => Element110,
            111 => Element111,
            112 => Element112,
            113 => Element113,
            114 => Element114,
            115 => Element115,
            116 => Element116,
            117 => Element117,
            118 => Element118,
            119 => Element119,
            120 => Element120,
            121 => Element121,
            122 => Element122,
            123 => Element123,
            124 => Element124,
            125 => Element125,
            126 => Element126,
            127 => Element127,
            _ => Element0,
        };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void SetElement(int index, GpuUInt128 value)
        {
            switch (index)
            {
            case 0:
                Element0 = value;
                return;
            case 1:
                Element1 = value;
                return;
            case 2:
                Element2 = value;
                return;
            case 3:
                Element3 = value;
                return;
            case 4:
                Element4 = value;
                return;
            case 5:
                Element5 = value;
                return;
            case 6:
                Element6 = value;
                return;
            case 7:
                Element7 = value;
                return;
            case 8:
                Element8 = value;
                return;
            case 9:
                Element9 = value;
                return;
            case 10:
                Element10 = value;
                return;
            case 11:
                Element11 = value;
                return;
            case 12:
                Element12 = value;
                return;
            case 13:
                Element13 = value;
                return;
            case 14:
                Element14 = value;
                return;
            case 15:
                Element15 = value;
                return;
            case 16:
                Element16 = value;
                return;
            case 17:
                Element17 = value;
                return;
            case 18:
                Element18 = value;
                return;
            case 19:
                Element19 = value;
                return;
            case 20:
                Element20 = value;
                return;
            case 21:
                Element21 = value;
                return;
            case 22:
                Element22 = value;
                return;
            case 23:
                Element23 = value;
                return;
            case 24:
                Element24 = value;
                return;
            case 25:
                Element25 = value;
                return;
            case 26:
                Element26 = value;
                return;
            case 27:
                Element27 = value;
                return;
            case 28:
                Element28 = value;
                return;
            case 29:
                Element29 = value;
                return;
            case 30:
                Element30 = value;
                return;
            case 31:
                Element31 = value;
                return;
            case 32:
                Element32 = value;
                return;
            case 33:
                Element33 = value;
                return;
            case 34:
                Element34 = value;
                return;
            case 35:
                Element35 = value;
                return;
            case 36:
                Element36 = value;
                return;
            case 37:
                Element37 = value;
                return;
            case 38:
                Element38 = value;
                return;
            case 39:
                Element39 = value;
                return;
            case 40:
                Element40 = value;
                return;
            case 41:
                Element41 = value;
                return;
            case 42:
                Element42 = value;
                return;
            case 43:
                Element43 = value;
                return;
            case 44:
                Element44 = value;
                return;
            case 45:
                Element45 = value;
                return;
            case 46:
                Element46 = value;
                return;
            case 47:
                Element47 = value;
                return;
            case 48:
                Element48 = value;
                return;
            case 49:
                Element49 = value;
                return;
            case 50:
                Element50 = value;
                return;
            case 51:
                Element51 = value;
                return;
            case 52:
                Element52 = value;
                return;
            case 53:
                Element53 = value;
                return;
            case 54:
                Element54 = value;
                return;
            case 55:
                Element55 = value;
                return;
            case 56:
                Element56 = value;
                return;
            case 57:
                Element57 = value;
                return;
            case 58:
                Element58 = value;
                return;
            case 59:
                Element59 = value;
                return;
            case 60:
                Element60 = value;
                return;
            case 61:
                Element61 = value;
                return;
            case 62:
                Element62 = value;
                return;
            case 63:
                Element63 = value;
                return;
            case 64:
                Element64 = value;
                return;
            case 65:
                Element65 = value;
                return;
            case 66:
                Element66 = value;
                return;
            case 67:
                Element67 = value;
                return;
            case 68:
                Element68 = value;
                return;
            case 69:
                Element69 = value;
                return;
            case 70:
                Element70 = value;
                return;
            case 71:
                Element71 = value;
                return;
            case 72:
                Element72 = value;
                return;
            case 73:
                Element73 = value;
                return;
            case 74:
                Element74 = value;
                return;
            case 75:
                Element75 = value;
                return;
            case 76:
                Element76 = value;
                return;
            case 77:
                Element77 = value;
                return;
            case 78:
                Element78 = value;
                return;
            case 79:
                Element79 = value;
                return;
            case 80:
                Element80 = value;
                return;
            case 81:
                Element81 = value;
                return;
            case 82:
                Element82 = value;
                return;
            case 83:
                Element83 = value;
                return;
            case 84:
                Element84 = value;
                return;
            case 85:
                Element85 = value;
                return;
            case 86:
                Element86 = value;
                return;
            case 87:
                Element87 = value;
                return;
            case 88:
                Element88 = value;
                return;
            case 89:
                Element89 = value;
                return;
            case 90:
                Element90 = value;
                return;
            case 91:
                Element91 = value;
                return;
            case 92:
                Element92 = value;
                return;
            case 93:
                Element93 = value;
                return;
            case 94:
                Element94 = value;
                return;
            case 95:
                Element95 = value;
                return;
            case 96:
                Element96 = value;
                return;
            case 97:
                Element97 = value;
                return;
            case 98:
                Element98 = value;
                return;
            case 99:
                Element99 = value;
                return;
            case 100:
                Element100 = value;
                return;
            case 101:
                Element101 = value;
                return;
            case 102:
                Element102 = value;
                return;
            case 103:
                Element103 = value;
                return;
            case 104:
                Element104 = value;
                return;
            case 105:
                Element105 = value;
                return;
            case 106:
                Element106 = value;
                return;
            case 107:
                Element107 = value;
                return;
            case 108:
                Element108 = value;
                return;
            case 109:
                Element109 = value;
                return;
            case 110:
                Element110 = value;
                return;
            case 111:
                Element111 = value;
                return;
            case 112:
                Element112 = value;
                return;
            case 113:
                Element113 = value;
                return;
            case 114:
                Element114 = value;
                return;
            case 115:
                Element115 = value;
                return;
            case 116:
                Element116 = value;
                return;
            case 117:
                Element117 = value;
                return;
            case 118:
                Element118 = value;
                return;
            case 119:
                Element119 = value;
                return;
            case 120:
                Element120 = value;
                return;
            case 121:
                Element121 = value;
                return;
            case 122:
                Element122 = value;
                return;
            case 123:
                Element123 = value;
                return;
            case 124:
                Element124 = value;
                return;
            case 125:
                Element125 = value;
                return;
            case 126:
                Element126 = value;
                return;
            case 127:
                Element127 = value;
                return;
            default:
                // ILGPU kernels cannot throw exceptions, and callers guarantee the index range.
                return;
            }
        }
    }

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
            try
            {
                using var execution = lease.EnterExecutionScope();
                Accelerator accelerator = lease.Accelerator;
                AcceleratorStream stream = lease.Stream;

                var kernel = GetPartialFactorKernel(accelerator);
                SmallPrimeDeviceCache cache = GetSmallPrimeDeviceCache(accelerator);

                using var factorBuffer = accelerator.Allocate1D<ulong>(primeTargets.Length);
                using var exponentBuffer = accelerator.Allocate1D<int>(exponentTargets.Length);
                using var countBuffer = accelerator.Allocate1D<int>(1);
                using var remainingBuffer = accelerator.Allocate1D<ulong>(1);
                using var fullyFactoredBuffer = accelerator.Allocate1D<byte>(1);

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

                return true;
            }
            finally
            {
                lease.Dispose();
            }
        }
        catch (Exception ex) when (ex is AcceleratorException or InternalCompilerException or NotSupportedException or InvalidOperationException or AggregateException)
        {
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
    public static GpuPow2ModStatus TryPow2Mod(ulong exponent, ulong prime, out ulong remainder)
    {
        Span<ulong> exponents = stackalloc ulong[1];
        Span<ulong> remainders = stackalloc ulong[1];
        exponents[0] = exponent;

        GpuPow2ModStatus status = TryPow2ModBatch(exponents, prime, remainders);
        remainder = remainders[0];
        return status;
    }

    public static GpuPow2ModStatus TryPow2ModBatch(ReadOnlySpan<ulong> exponents, ulong prime, Span<ulong> remainders)
    {
        if (exponents.Length == 0)
        {
            return GpuPow2ModStatus.Success;
        }

        if (remainders.Length < exponents.Length)
        {
            throw new ArgumentException("Remainder span is shorter than the exponent span.", nameof(remainders));
        }

        Span<ulong> target = remainders.Slice(0, exponents.Length);
        target.Clear();

        if (prime <= 1UL)
        {
            return GpuPow2ModStatus.Unavailable;
        }

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

        bool computed = TryComputeOnGpu(exponents, prime, target);
        return computed ? GpuPow2ModStatus.Success : GpuPow2ModStatus.Unavailable;
    }

    public static GpuPow2ModStatus TryPow2Mod(UInt128 exponent, UInt128 prime, out UInt128 remainder)
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
        out PrimeOrderCalculator.PrimeOrderResult result)
    {
        result = default;

        try
        {
            var lease = GpuKernelPool.GetKernel(useGpuOrder: true);
            try
            {
                using var execution = lease.EnterExecutionScope();
                Accelerator accelerator = lease.Accelerator;
                AcceleratorStream stream = lease.Stream;

                var kernel = GetOrderKernel(accelerator);
                SmallPrimeDeviceCache cache = GetSmallPrimeDeviceCache(accelerator);

                using var phiFactorBuffer = accelerator.Allocate1D<ulong>(GpuSmallPrimeFactorSlots);
                using var phiExponentBuffer = accelerator.Allocate1D<int>(GpuSmallPrimeFactorSlots);
                using var workFactorBuffer = accelerator.Allocate1D<ulong>(GpuSmallPrimeFactorSlots);
                using var workExponentBuffer = accelerator.Allocate1D<int>(GpuSmallPrimeFactorSlots);
                using var candidateBuffer = accelerator.Allocate1D<ulong>(HeuristicCandidateLimit);
                using var stackIndexBuffer = accelerator.Allocate1D<int>(HeuristicStackCapacity);
                using var stackExponentBuffer = accelerator.Allocate1D<int>(HeuristicStackCapacity);
                using var stackProductBuffer = accelerator.Allocate1D<ulong>(HeuristicStackCapacity);
                using var resultBuffer = accelerator.Allocate1D<ulong>(1);
                using var statusBuffer = accelerator.Allocate1D<byte>(1);

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
                    return false;
                }

                if (kernelStatus == PrimeOrderKernelStatus.PollardOverflow)
                {
                    throw new InvalidOperationException("GPU Pollard Rho stack overflow; increase HeuristicStackCapacity.");
                }

                ulong order = 0UL;
                resultBuffer.View.CopyToCPU(ref order, 1);

                PrimeOrderCalculator.PrimeOrderStatus finalStatus = kernelStatus == PrimeOrderKernelStatus.Found
                    ? PrimeOrderCalculator.PrimeOrderStatus.Found
                    : PrimeOrderCalculator.PrimeOrderStatus.HeuristicUnresolved;

                result = new PrimeOrderCalculator.PrimeOrderResult(finalStatus, order);
                return true;
            }
            finally
            {
                lease.Dispose();
            }
        }
        catch (Exception ex) when (ex is AcceleratorException or InternalCompilerException or NotSupportedException or InvalidOperationException or AggregateException)
        {
            result = default;
            return false;
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
            if (factor <= 1UL)
            {
                continue;
            }

            ulong reduced = phi / factor;
            if (Pow2EqualsOneKernel(reduced, prime, divisor))
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
            if (Pow2EqualsOneKernel(half, prime, divisor))
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
            if (primeFactor <= 1UL)
            {
                continue;
            }

            for (int iteration = 0; iteration < exponent; iteration++)
            {
                if ((order % primeFactor) != 0UL)
                {
                    break;
                }

                ulong reduced = order / primeFactor;
                if (Pow2EqualsOneKernel(reduced, prime, divisor))
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
        if (order == 0UL)
        {
            return false;
        }

        if (!Pow2EqualsOneKernel(order, prime, divisor))
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
            if (primeFactor <= 1UL)
            {
                continue;
            }

            ulong reduced = order;
            for (int iteration = 0; iteration < exponent; iteration++)
            {
                if ((reduced % primeFactor) != 0UL)
                {
                    break;
                }

                reduced /= primeFactor;
                if (Pow2EqualsOneKernel(reduced, prime, divisor))
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
        if (powBudget <= 0)
        {
            powBudget = candidateCount;
        }

        int powUsed = 0;

        for (int i = 0; i < candidateCount && powUsed < powBudget; i++)
        {
            ulong candidate = candidates[i];
            if (candidate <= 1UL)
            {
                continue;
            }

            if (powUsed >= powBudget)
            {
                break;
            }

            powUsed++;
            if (!Pow2EqualsOneKernel(candidate, prime, divisor))
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
            if (primeFactor <= 1UL)
            {
                continue;
            }

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
                if (Pow2EqualsOneKernel(reduced, prime, divisor))
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
        if (initial <= 1UL)
        {
            return true;
        }

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

    private static bool Pow2EqualsOneKernel(ulong exponent, ulong prime, in MontgomeryDivisorData divisor)
    {
        return exponent.Pow2MontgomeryModWindowed(divisor, keepMontgomery: false) == 1UL;
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

    private static bool TryComputeOnGpu(ReadOnlySpan<ulong> exponents, ulong prime, Span<ulong> results)
    {
        try
        {
            var lease = GpuKernelPool.GetKernel(useGpuOrder: true);
            try
            {
                using var execution = lease.EnterExecutionScope();
                Accelerator accelerator = lease.Accelerator;
                AcceleratorStream stream = lease.Stream;
                var kernel = GetPow2ModKernel(accelerator);
                using var exponentBuffer = accelerator.Allocate1D<ulong>(exponents.Length);
                using var remainderBuffer = accelerator.Allocate1D<ulong>(exponents.Length);

                exponentBuffer.View.CopyFromCPU(ref MemoryMarshal.GetReference(exponents), exponents.Length);
                remainderBuffer.MemSetToZero();

                MontgomeryDivisorData divisor = MontgomeryDivisorDataCache.Get(prime);
                kernel(stream, exponents.Length, exponentBuffer.View, divisor, remainderBuffer.View);

                stream.Synchronize();

                remainderBuffer.View.CopyToCPU(ref MemoryMarshal.GetReference(results), results.Length);
                return true;
            }
            finally
            {
                lease.Dispose();
            }
        }
        catch (Exception ex) when (ex is AcceleratorException or InternalCompilerException or NotSupportedException or InvalidOperationException or AggregateException)
        {
            ComputePow2ModCpu(exponents, prime, results);
            return true;
        }
    }

    private static void ComputePow2ModCpu(ReadOnlySpan<ulong> exponents, ulong prime, Span<ulong> results)
    {
        int length = exponents.Length;
        for (int i = 0; i < length; i++)
        {
            results[i] = Pow2ModCpu(exponents[i], prime);
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
            using var execution = lease.EnterExecutionScope();
            Accelerator accelerator = lease.Accelerator;
            AcceleratorStream stream = lease.Stream;
            var kernel = GetPow2ModWideKernel(accelerator);
            using var exponentBuffer = accelerator.Allocate1D<GpuUInt128>(length);
            using var remainderBuffer = accelerator.Allocate1D<GpuUInt128>(length);

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

            return true;
        }
        catch (Exception ex) when (ex is AcceleratorException or InternalCompilerException or NotSupportedException or InvalidOperationException or AggregateException)
        {
            ComputePow2ModCpuWide(exponents, prime, results);
            return true;
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

    private static ulong Pow2ModCpu(ulong exponent, ulong modulus)
    {
        if (modulus <= 1UL)
        {
            return 0UL;
        }

        MontgomeryDivisorData divisor = MontgomeryDivisorDataCache.Get(modulus);
        return exponent.Pow2MontgomeryModWindowed(divisor, keepMontgomery: false);
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
        ulong exponent = exponents[index];
        remainders[index] = exponent.Pow2MontgomeryModWindowed(divisor, keepMontgomery: false);
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

    private static GpuUInt128 Pow2ModKernelCore(GpuUInt128 exponent, GpuUInt128 modulus)
    {
        if (modulus == GpuUInt128.One)
        {
            return GpuUInt128.Zero;
        }

        if (exponent.IsZero)
        {
            return GpuUInt128.One;
        }

        GpuUInt128 baseValue = new GpuUInt128(2UL);
        if (baseValue.CompareTo(modulus) >= 0)
        {
            baseValue.Sub(modulus);
        }

        if (ShouldUseSingleBit(exponent))
        {
            return Pow2MontgomeryModSingleBit(exponent, modulus, baseValue);
        }

        int bitLength = exponent.GetBitLength();
        int windowSize = GetWindowSize(bitLength);
        int oddPowerCount = 1 << (windowSize - 1);

        Pow2OddPowerTable oddPowers = default;
        InitializeOddPowers(ref oddPowers, baseValue, modulus, oddPowerCount);

        GpuUInt128 result = GpuUInt128.One;
        int index = bitLength - 1;

        while (index >= 0)
        {
            if (!IsBitSet(exponent, index))
            {
                result.MulMod(result, modulus);
                index--;
                continue;
            }

            int windowStart = index - windowSize + 1;
            if (windowStart < 0)
            {
                windowStart = 0;
            }

            while (!IsBitSet(exponent, windowStart))
            {
                windowStart++;
            }

            int windowBitCount = index - windowStart + 1;
            for (int square = 0; square < windowBitCount; square++)
            {
                result.MulMod(result, modulus);
            }

            ulong windowValue = ExtractWindowValue(exponent, windowStart, windowBitCount);
            int tableIndex = (int)((windowValue - 1UL) >> 1);
            GpuUInt128 factor = oddPowers[tableIndex];
            result.MulMod(factor, modulus);

            index = windowStart - 1;
        }

        return result;
    }

    private static bool ShouldUseSingleBit(in GpuUInt128 exponent) => exponent.High == 0UL && exponent.Low <= Pow2WindowFallbackThreshold;

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

    private static void InitializeOddPowers(ref Pow2OddPowerTable oddPowers, GpuUInt128 baseValue, GpuUInt128 modulus, int oddPowerCount)
    {
        oddPowers[0] = baseValue;
        if (oddPowerCount == 1)
        {
            return;
        }

        // Reusing baseValue to hold base^2 for the shared odd-power ladder that follows.
        baseValue.MulMod(baseValue, modulus);
        for (int i = 1; i < oddPowerCount; i++)
        {
            GpuUInt128 ladderEntry = oddPowers[i - 1];
            ladderEntry.MulMod(baseValue, modulus);
            oddPowers[i] = ladderEntry;
        }
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

    private static bool IsBitSet(in GpuUInt128 value, int bitIndex)
    {
        if (bitIndex >= 64)
        {
            return ((value.High >> (bitIndex - 64)) & 1UL) != 0UL;
        }

        return ((value.Low >> bitIndex) & 1UL) != 0UL;
    }

    private static ulong ExtractWindowValue(in GpuUInt128 exponent, int windowStart, int windowBitCount)
    {
        if (windowStart != 0)
        {
            GpuUInt128 shifted = exponent;
            shifted.ShiftRight(windowStart);
            ulong mask = (1UL << windowBitCount) - 1UL;
            return shifted.Low & mask;
        }

        ulong directMask = (1UL << windowBitCount) - 1UL;
        return exponent.Low & directMask;
    }

    internal readonly record struct PrimeOrderGpuCapability(int ModulusBits, int ExponentBits)
    {
        public static PrimeOrderGpuCapability Default => new(128, 128);
    }
}
