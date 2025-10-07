using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using PerfectNumbers.Core.Gpu;

namespace EvenPerfectBitScanner.Benchmarks
{
    public readonly struct Pow2OddPowerTableDelegate
    {
        private static readonly Action _emptyAction = () => { };

        public Pow2OddPowerTableDelegate(GpuUInt128 baseValue, GpuUInt128 modulus, int oddPowerCount)
        {
            Element0 = baseValue;
            GpuUInt128 current;
            Action transform;
            if (oddPowerCount > 1)
            {
                // Reusing baseValue to hold base^2 for the shared odd-power ladder that follows.
                baseValue.MulMod(baseValue, modulus);
                // TODO: We can calculate baseValue % modulus before loop and use it to increase ladderEntry calculation speed - we'll reuse the base for incremental calculations.
                current = baseValue;
                transform = () => current.MulMod(baseValue, modulus);
            }
            else
            {
                current = GpuUInt128.Zero;
                transform = _emptyAction;
            }

            // We're manually assigning each field to prevent the compiler to initialize each field twice due to auto-initialization. We're using the action to lower the code base size.
            transform();
            Element1 = current;
            transform();
            Element2 = current;
            transform();
            Element3 = current;
            transform();
            Element4 = current;
            transform();
            Element5 = current;
            transform();
            Element6 = current;
            transform();
            Element7 = current;
            transform();
            Element8 = current;
            transform();
            Element9 = current;
            transform();
            Element10 = current;
            transform();
            Element11 = current;
            transform();
            Element12 = current;
            transform();
            Element13 = current;
            transform();
            Element14 = current;
            transform();
            Element15 = current;
            transform();
            Element16 = current;
            transform();
            Element17 = current;
            transform();
            Element18 = current;
            transform();
            Element19 = current;
            transform();
            Element20 = current;
            transform();
            Element21 = current;
            transform();
            Element22 = current;
            transform();
            Element23 = current;
            transform();
            Element24 = current;
            transform();
            Element25 = current;
            transform();
            Element26 = current;
            transform();
            Element27 = current;
            transform();
            Element28 = current;
            transform();
            Element29 = current;
            transform();
            Element30 = current;
            transform();
            Element31 = current;
            transform();
            Element32 = current;
            transform();
            Element33 = current;
            transform();
            Element34 = current;
            transform();
            Element35 = current;
            transform();
            Element36 = current;
            transform();
            Element37 = current;
            transform();
            Element38 = current;
            transform();
            Element39 = current;
            transform();
            Element40 = current;
            transform();
            Element41 = current;
            transform();
            Element42 = current;
            transform();
            Element43 = current;
            transform();
            Element44 = current;
            transform();
            Element45 = current;
            transform();
            Element46 = current;
            transform();
            Element47 = current;
            transform();
            Element48 = current;
            transform();
            Element49 = current;
            transform();
            Element50 = current;
            transform();
            Element51 = current;
            transform();
            Element52 = current;
            transform();
            Element53 = current;
            transform();
            Element54 = current;
            transform();
            Element55 = current;
            transform();
            Element56 = current;
            transform();
            Element57 = current;
            transform();
            Element58 = current;
            transform();
            Element59 = current;
            transform();
            Element60 = current;
            transform();
            Element61 = current;
            transform();
            Element62 = current;
            transform();
            Element63 = current;
            transform();
            Element64 = current;
            transform();
            Element65 = current;
            transform();
            Element66 = current;
            transform();
            Element67 = current;
            transform();
            Element68 = current;
            transform();
            Element69 = current;
            transform();
            Element70 = current;
            transform();
            Element71 = current;
            transform();
            Element72 = current;
            transform();
            Element73 = current;
            transform();
            Element74 = current;
            transform();
            Element75 = current;
            transform();
            Element76 = current;
            transform();
            Element77 = current;
            transform();
            Element78 = current;
            transform();
            Element79 = current;
            transform();
            Element80 = current;
            transform();
            Element81 = current;
            transform();
            Element82 = current;
            transform();
            Element83 = current;
            transform();
            Element84 = current;
            transform();
            Element85 = current;
            transform();
            Element86 = current;
            transform();
            Element87 = current;
            transform();
            Element88 = current;
            transform();
            Element89 = current;
            transform();
            Element90 = current;
            transform();
            Element91 = current;
            transform();
            Element92 = current;
            transform();
            Element93 = current;
            transform();
            Element94 = current;
            transform();
            Element95 = current;
            transform();
            Element96 = current;
            transform();
            Element97 = current;
            transform();
            Element98 = current;
            transform();
            Element99 = current;
            transform();
            Element100 = current;
            transform();
            Element101 = current;
            transform();
            Element102 = current;
            transform();
            Element103 = current;
            transform();
            Element104 = current;
            transform();
            Element105 = current;
            transform();
            Element106 = current;
            transform();
            Element107 = current;
            transform();
            Element108 = current;
            transform();
            Element109 = current;
            transform();
            Element110 = current;
            transform();
            Element111 = current;
            transform();
            Element112 = current;
            transform();
            Element113 = current;
            transform();
            Element114 = current;
            transform();
            Element115 = current;
            transform();
            Element116 = current;
            transform();
            Element117 = current;
            transform();
            Element118 = current;
            transform();
            Element119 = current;
            transform();
            Element120 = current;
            transform();
            Element121 = current;
            transform();
            Element122 = current;
            transform();
            Element123 = current;
            transform();
            Element124 = current;
            transform();
            Element125 = current;
            transform();
            Element126 = current;
            transform();
            Element127 = current;
        }

        public readonly GpuUInt128 Element0;
        public readonly GpuUInt128 Element1;
        public readonly GpuUInt128 Element2;
        public readonly GpuUInt128 Element3;
        public readonly GpuUInt128 Element4;
        public readonly GpuUInt128 Element5;
        public readonly GpuUInt128 Element6;
        public readonly GpuUInt128 Element7;
        public readonly GpuUInt128 Element8;
        public readonly GpuUInt128 Element9;
        public readonly GpuUInt128 Element10;
        public readonly GpuUInt128 Element11;
        public readonly GpuUInt128 Element12;
        public readonly GpuUInt128 Element13;
        public readonly GpuUInt128 Element14;
        public readonly GpuUInt128 Element15;
        public readonly GpuUInt128 Element16;
        public readonly GpuUInt128 Element17;
        public readonly GpuUInt128 Element18;
        public readonly GpuUInt128 Element19;
        public readonly GpuUInt128 Element20;
        public readonly GpuUInt128 Element21;
        public readonly GpuUInt128 Element22;
        public readonly GpuUInt128 Element23;
        public readonly GpuUInt128 Element24;
        public readonly GpuUInt128 Element25;
        public readonly GpuUInt128 Element26;
        public readonly GpuUInt128 Element27;
        public readonly GpuUInt128 Element28;
        public readonly GpuUInt128 Element29;
        public readonly GpuUInt128 Element30;
        public readonly GpuUInt128 Element31;
        public readonly GpuUInt128 Element32;
        public readonly GpuUInt128 Element33;
        public readonly GpuUInt128 Element34;
        public readonly GpuUInt128 Element35;
        public readonly GpuUInt128 Element36;
        public readonly GpuUInt128 Element37;
        public readonly GpuUInt128 Element38;
        public readonly GpuUInt128 Element39;
        public readonly GpuUInt128 Element40;
        public readonly GpuUInt128 Element41;
        public readonly GpuUInt128 Element42;
        public readonly GpuUInt128 Element43;
        public readonly GpuUInt128 Element44;
        public readonly GpuUInt128 Element45;
        public readonly GpuUInt128 Element46;
        public readonly GpuUInt128 Element47;
        public readonly GpuUInt128 Element48;
        public readonly GpuUInt128 Element49;
        public readonly GpuUInt128 Element50;
        public readonly GpuUInt128 Element51;
        public readonly GpuUInt128 Element52;
        public readonly GpuUInt128 Element53;
        public readonly GpuUInt128 Element54;
        public readonly GpuUInt128 Element55;
        public readonly GpuUInt128 Element56;
        public readonly GpuUInt128 Element57;
        public readonly GpuUInt128 Element58;
        public readonly GpuUInt128 Element59;
        public readonly GpuUInt128 Element60;
        public readonly GpuUInt128 Element61;
        public readonly GpuUInt128 Element62;
        public readonly GpuUInt128 Element63;
        public readonly GpuUInt128 Element64;
        public readonly GpuUInt128 Element65;
        public readonly GpuUInt128 Element66;
        public readonly GpuUInt128 Element67;
        public readonly GpuUInt128 Element68;
        public readonly GpuUInt128 Element69;
        public readonly GpuUInt128 Element70;
        public readonly GpuUInt128 Element71;
        public readonly GpuUInt128 Element72;
        public readonly GpuUInt128 Element73;
        public readonly GpuUInt128 Element74;
        public readonly GpuUInt128 Element75;
        public readonly GpuUInt128 Element76;
        public readonly GpuUInt128 Element77;
        public readonly GpuUInt128 Element78;
        public readonly GpuUInt128 Element79;
        public readonly GpuUInt128 Element80;
        public readonly GpuUInt128 Element81;
        public readonly GpuUInt128 Element82;
        public readonly GpuUInt128 Element83;
        public readonly GpuUInt128 Element84;
        public readonly GpuUInt128 Element85;
        public readonly GpuUInt128 Element86;
        public readonly GpuUInt128 Element87;
        public readonly GpuUInt128 Element88;
        public readonly GpuUInt128 Element89;
        public readonly GpuUInt128 Element90;
        public readonly GpuUInt128 Element91;
        public readonly GpuUInt128 Element92;
        public readonly GpuUInt128 Element93;
        public readonly GpuUInt128 Element94;
        public readonly GpuUInt128 Element95;
        public readonly GpuUInt128 Element96;
        public readonly GpuUInt128 Element97;
        public readonly GpuUInt128 Element98;
        public readonly GpuUInt128 Element99;
        public readonly GpuUInt128 Element100;
        public readonly GpuUInt128 Element101;
        public readonly GpuUInt128 Element102;
        public readonly GpuUInt128 Element103;
        public readonly GpuUInt128 Element104;
        public readonly GpuUInt128 Element105;
        public readonly GpuUInt128 Element106;
        public readonly GpuUInt128 Element107;
        public readonly GpuUInt128 Element108;
        public readonly GpuUInt128 Element109;
        public readonly GpuUInt128 Element110;
        public readonly GpuUInt128 Element111;
        public readonly GpuUInt128 Element112;
        public readonly GpuUInt128 Element113;
        public readonly GpuUInt128 Element114;
        public readonly GpuUInt128 Element115;
        public readonly GpuUInt128 Element116;
        public readonly GpuUInt128 Element117;
        public readonly GpuUInt128 Element118;
        public readonly GpuUInt128 Element119;
        public readonly GpuUInt128 Element120;
        public readonly GpuUInt128 Element121;
        public readonly GpuUInt128 Element122;
        public readonly GpuUInt128 Element123;
        public readonly GpuUInt128 Element124;
        public readonly GpuUInt128 Element125;
        public readonly GpuUInt128 Element126;
        public readonly GpuUInt128 Element127;

        public readonly GpuUInt128 this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return index switch
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
            }
        }
    }

    public readonly struct Pow2OddPowerTableReturn
    {
        public Pow2OddPowerTableReturn(GpuUInt128 baseValue, GpuUInt128 modulus, int oddPowerCount)
        {
            Element0 = baseValue;
            if (oddPowerCount == 1)
            {
                return;
            }

            // Reusing baseValue to hold base^2 for the shared odd-power ladder that follows.
            baseValue.MulMod(baseValue, modulus);
            // TODO: We can calculate baseValue % modulus before loop and use it to increase ladderEntry calculation speed - we'll reuse the base for incremental calculations.
            GpuUInt128 current = baseValue;

            // We're manually assigning each field to prevent the compiler to initialize each field twice due to auto-initialization. We're using the action to lower the code base size.
            current.MulMod(baseValue, modulus);
            Element1 = current;
            current.MulMod(baseValue, modulus);
            Element2 = current;
            current.MulMod(baseValue, modulus);
            Element3 = current;
            current.MulMod(baseValue, modulus);
            Element4 = current;
            current.MulMod(baseValue, modulus);
            Element5 = current;
            current.MulMod(baseValue, modulus);
            Element6 = current;
            current.MulMod(baseValue, modulus);
            Element7 = current;
            current.MulMod(baseValue, modulus);
            Element8 = current;
            current.MulMod(baseValue, modulus);
            Element9 = current;
            current.MulMod(baseValue, modulus);
            Element10 = current;
            current.MulMod(baseValue, modulus);
            Element11 = current;
            current.MulMod(baseValue, modulus);
            Element12 = current;
            current.MulMod(baseValue, modulus);
            Element13 = current;
            current.MulMod(baseValue, modulus);
            Element14 = current;
            current.MulMod(baseValue, modulus);
            Element15 = current;
            current.MulMod(baseValue, modulus);
            Element16 = current;
            current.MulMod(baseValue, modulus);
            Element17 = current;
            current.MulMod(baseValue, modulus);
            Element18 = current;
            current.MulMod(baseValue, modulus);
            Element19 = current;
            current.MulMod(baseValue, modulus);
            Element20 = current;
            current.MulMod(baseValue, modulus);
            Element21 = current;
            current.MulMod(baseValue, modulus);
            Element22 = current;
            current.MulMod(baseValue, modulus);
            Element23 = current;
            current.MulMod(baseValue, modulus);
            Element24 = current;
            current.MulMod(baseValue, modulus);
            Element25 = current;
            current.MulMod(baseValue, modulus);
            Element26 = current;
            current.MulMod(baseValue, modulus);
            Element27 = current;
            current.MulMod(baseValue, modulus);
            Element28 = current;
            current.MulMod(baseValue, modulus);
            Element29 = current;
            current.MulMod(baseValue, modulus);
            Element30 = current;
            current.MulMod(baseValue, modulus);
            Element31 = current;
            current.MulMod(baseValue, modulus);
            Element32 = current;
            current.MulMod(baseValue, modulus);
            Element33 = current;
            current.MulMod(baseValue, modulus);
            Element34 = current;
            current.MulMod(baseValue, modulus);
            Element35 = current;
            current.MulMod(baseValue, modulus);
            Element36 = current;
            current.MulMod(baseValue, modulus);
            Element37 = current;
            current.MulMod(baseValue, modulus);
            Element38 = current;
            current.MulMod(baseValue, modulus);
            Element39 = current;
            current.MulMod(baseValue, modulus);
            Element40 = current;
            current.MulMod(baseValue, modulus);
            Element41 = current;
            current.MulMod(baseValue, modulus);
            Element42 = current;
            current.MulMod(baseValue, modulus);
            Element43 = current;
            current.MulMod(baseValue, modulus);
            Element44 = current;
            current.MulMod(baseValue, modulus);
            Element45 = current;
            current.MulMod(baseValue, modulus);
            Element46 = current;
            current.MulMod(baseValue, modulus);
            Element47 = current;
            current.MulMod(baseValue, modulus);
            Element48 = current;
            current.MulMod(baseValue, modulus);
            Element49 = current;
            current.MulMod(baseValue, modulus);
            Element50 = current;
            current.MulMod(baseValue, modulus);
            Element51 = current;
            current.MulMod(baseValue, modulus);
            Element52 = current;
            current.MulMod(baseValue, modulus);
            Element53 = current;
            current.MulMod(baseValue, modulus);
            Element54 = current;
            current.MulMod(baseValue, modulus);
            Element55 = current;
            current.MulMod(baseValue, modulus);
            Element56 = current;
            current.MulMod(baseValue, modulus);
            Element57 = current;
            current.MulMod(baseValue, modulus);
            Element58 = current;
            current.MulMod(baseValue, modulus);
            Element59 = current;
            current.MulMod(baseValue, modulus);
            Element60 = current;
            current.MulMod(baseValue, modulus);
            Element61 = current;
            current.MulMod(baseValue, modulus);
            Element62 = current;
            current.MulMod(baseValue, modulus);
            Element63 = current;
            current.MulMod(baseValue, modulus);
            Element64 = current;
            current.MulMod(baseValue, modulus);
            Element65 = current;
            current.MulMod(baseValue, modulus);
            Element66 = current;
            current.MulMod(baseValue, modulus);
            Element67 = current;
            current.MulMod(baseValue, modulus);
            Element68 = current;
            current.MulMod(baseValue, modulus);
            Element69 = current;
            current.MulMod(baseValue, modulus);
            Element70 = current;
            current.MulMod(baseValue, modulus);
            Element71 = current;
            current.MulMod(baseValue, modulus);
            Element72 = current;
            current.MulMod(baseValue, modulus);
            Element73 = current;
            current.MulMod(baseValue, modulus);
            Element74 = current;
            current.MulMod(baseValue, modulus);
            Element75 = current;
            current.MulMod(baseValue, modulus);
            Element76 = current;
            current.MulMod(baseValue, modulus);
            Element77 = current;
            current.MulMod(baseValue, modulus);
            Element78 = current;
            current.MulMod(baseValue, modulus);
            Element79 = current;
            current.MulMod(baseValue, modulus);
            Element80 = current;
            current.MulMod(baseValue, modulus);
            Element81 = current;
            current.MulMod(baseValue, modulus);
            Element82 = current;
            current.MulMod(baseValue, modulus);
            Element83 = current;
            current.MulMod(baseValue, modulus);
            Element84 = current;
            current.MulMod(baseValue, modulus);
            Element85 = current;
            current.MulMod(baseValue, modulus);
            Element86 = current;
            current.MulMod(baseValue, modulus);
            Element87 = current;
            current.MulMod(baseValue, modulus);
            Element88 = current;
            current.MulMod(baseValue, modulus);
            Element89 = current;
            current.MulMod(baseValue, modulus);
            Element90 = current;
            current.MulMod(baseValue, modulus);
            Element91 = current;
            current.MulMod(baseValue, modulus);
            Element92 = current;
            current.MulMod(baseValue, modulus);
            Element93 = current;
            current.MulMod(baseValue, modulus);
            Element94 = current;
            current.MulMod(baseValue, modulus);
            Element95 = current;
            current.MulMod(baseValue, modulus);
            Element96 = current;
            current.MulMod(baseValue, modulus);
            Element97 = current;
            current.MulMod(baseValue, modulus);
            Element98 = current;
            current.MulMod(baseValue, modulus);
            Element99 = current;
            current.MulMod(baseValue, modulus);
            Element100 = current;
            current.MulMod(baseValue, modulus);
            Element101 = current;
            current.MulMod(baseValue, modulus);
            Element102 = current;
            current.MulMod(baseValue, modulus);
            Element103 = current;
            current.MulMod(baseValue, modulus);
            Element104 = current;
            current.MulMod(baseValue, modulus);
            Element105 = current;
            current.MulMod(baseValue, modulus);
            Element106 = current;
            current.MulMod(baseValue, modulus);
            Element107 = current;
            current.MulMod(baseValue, modulus);
            Element108 = current;
            current.MulMod(baseValue, modulus);
            Element109 = current;
            current.MulMod(baseValue, modulus);
            Element110 = current;
            current.MulMod(baseValue, modulus);
            Element111 = current;
            current.MulMod(baseValue, modulus);
            Element112 = current;
            current.MulMod(baseValue, modulus);
            Element113 = current;
            current.MulMod(baseValue, modulus);
            Element114 = current;
            current.MulMod(baseValue, modulus);
            Element115 = current;
            current.MulMod(baseValue, modulus);
            Element116 = current;
            current.MulMod(baseValue, modulus);
            Element117 = current;
            current.MulMod(baseValue, modulus);
            Element118 = current;
            current.MulMod(baseValue, modulus);
            Element119 = current;
            current.MulMod(baseValue, modulus);
            Element120 = current;
            current.MulMod(baseValue, modulus);
            Element121 = current;
            current.MulMod(baseValue, modulus);
            Element122 = current;
            current.MulMod(baseValue, modulus);
            Element123 = current;
            current.MulMod(baseValue, modulus);
            Element124 = current;
            current.MulMod(baseValue, modulus);
            Element125 = current;
            current.MulMod(baseValue, modulus);
            Element126 = current;
            current.MulMod(baseValue, modulus);
            Element127 = current;
        }

        public readonly GpuUInt128 Element0;
        public readonly GpuUInt128 Element1;
        public readonly GpuUInt128 Element2;
        public readonly GpuUInt128 Element3;
        public readonly GpuUInt128 Element4;
        public readonly GpuUInt128 Element5;
        public readonly GpuUInt128 Element6;
        public readonly GpuUInt128 Element7;
        public readonly GpuUInt128 Element8;
        public readonly GpuUInt128 Element9;
        public readonly GpuUInt128 Element10;
        public readonly GpuUInt128 Element11;
        public readonly GpuUInt128 Element12;
        public readonly GpuUInt128 Element13;
        public readonly GpuUInt128 Element14;
        public readonly GpuUInt128 Element15;
        public readonly GpuUInt128 Element16;
        public readonly GpuUInt128 Element17;
        public readonly GpuUInt128 Element18;
        public readonly GpuUInt128 Element19;
        public readonly GpuUInt128 Element20;
        public readonly GpuUInt128 Element21;
        public readonly GpuUInt128 Element22;
        public readonly GpuUInt128 Element23;
        public readonly GpuUInt128 Element24;
        public readonly GpuUInt128 Element25;
        public readonly GpuUInt128 Element26;
        public readonly GpuUInt128 Element27;
        public readonly GpuUInt128 Element28;
        public readonly GpuUInt128 Element29;
        public readonly GpuUInt128 Element30;
        public readonly GpuUInt128 Element31;
        public readonly GpuUInt128 Element32;
        public readonly GpuUInt128 Element33;
        public readonly GpuUInt128 Element34;
        public readonly GpuUInt128 Element35;
        public readonly GpuUInt128 Element36;
        public readonly GpuUInt128 Element37;
        public readonly GpuUInt128 Element38;
        public readonly GpuUInt128 Element39;
        public readonly GpuUInt128 Element40;
        public readonly GpuUInt128 Element41;
        public readonly GpuUInt128 Element42;
        public readonly GpuUInt128 Element43;
        public readonly GpuUInt128 Element44;
        public readonly GpuUInt128 Element45;
        public readonly GpuUInt128 Element46;
        public readonly GpuUInt128 Element47;
        public readonly GpuUInt128 Element48;
        public readonly GpuUInt128 Element49;
        public readonly GpuUInt128 Element50;
        public readonly GpuUInt128 Element51;
        public readonly GpuUInt128 Element52;
        public readonly GpuUInt128 Element53;
        public readonly GpuUInt128 Element54;
        public readonly GpuUInt128 Element55;
        public readonly GpuUInt128 Element56;
        public readonly GpuUInt128 Element57;
        public readonly GpuUInt128 Element58;
        public readonly GpuUInt128 Element59;
        public readonly GpuUInt128 Element60;
        public readonly GpuUInt128 Element61;
        public readonly GpuUInt128 Element62;
        public readonly GpuUInt128 Element63;
        public readonly GpuUInt128 Element64;
        public readonly GpuUInt128 Element65;
        public readonly GpuUInt128 Element66;
        public readonly GpuUInt128 Element67;
        public readonly GpuUInt128 Element68;
        public readonly GpuUInt128 Element69;
        public readonly GpuUInt128 Element70;
        public readonly GpuUInt128 Element71;
        public readonly GpuUInt128 Element72;
        public readonly GpuUInt128 Element73;
        public readonly GpuUInt128 Element74;
        public readonly GpuUInt128 Element75;
        public readonly GpuUInt128 Element76;
        public readonly GpuUInt128 Element77;
        public readonly GpuUInt128 Element78;
        public readonly GpuUInt128 Element79;
        public readonly GpuUInt128 Element80;
        public readonly GpuUInt128 Element81;
        public readonly GpuUInt128 Element82;
        public readonly GpuUInt128 Element83;
        public readonly GpuUInt128 Element84;
        public readonly GpuUInt128 Element85;
        public readonly GpuUInt128 Element86;
        public readonly GpuUInt128 Element87;
        public readonly GpuUInt128 Element88;
        public readonly GpuUInt128 Element89;
        public readonly GpuUInt128 Element90;
        public readonly GpuUInt128 Element91;
        public readonly GpuUInt128 Element92;
        public readonly GpuUInt128 Element93;
        public readonly GpuUInt128 Element94;
        public readonly GpuUInt128 Element95;
        public readonly GpuUInt128 Element96;
        public readonly GpuUInt128 Element97;
        public readonly GpuUInt128 Element98;
        public readonly GpuUInt128 Element99;
        public readonly GpuUInt128 Element100;
        public readonly GpuUInt128 Element101;
        public readonly GpuUInt128 Element102;
        public readonly GpuUInt128 Element103;
        public readonly GpuUInt128 Element104;
        public readonly GpuUInt128 Element105;
        public readonly GpuUInt128 Element106;
        public readonly GpuUInt128 Element107;
        public readonly GpuUInt128 Element108;
        public readonly GpuUInt128 Element109;
        public readonly GpuUInt128 Element110;
        public readonly GpuUInt128 Element111;
        public readonly GpuUInt128 Element112;
        public readonly GpuUInt128 Element113;
        public readonly GpuUInt128 Element114;
        public readonly GpuUInt128 Element115;
        public readonly GpuUInt128 Element116;
        public readonly GpuUInt128 Element117;
        public readonly GpuUInt128 Element118;
        public readonly GpuUInt128 Element119;
        public readonly GpuUInt128 Element120;
        public readonly GpuUInt128 Element121;
        public readonly GpuUInt128 Element122;
        public readonly GpuUInt128 Element123;
        public readonly GpuUInt128 Element124;
        public readonly GpuUInt128 Element125;
        public readonly GpuUInt128 Element126;
        public readonly GpuUInt128 Element127;

        public readonly GpuUInt128 this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]            
            get
            {
                return index switch
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
            }
        }
    }

    public readonly struct Pow2OddPowerTableLoop
    {
        public Pow2OddPowerTableLoop(GpuUInt128 baseValue, GpuUInt128 modulus, int oddPowerCount)
        {
            Element0 = baseValue;
            if (oddPowerCount == 1)
            {
                return;
            }

            // Reusing baseValue to hold base^2 for the shared odd-power ladder that follows.
            baseValue.MulMod(baseValue, modulus);
            // TODO: We can calculate baseValue % modulus before loop and use it to increase ladderEntry calculation speed - we'll reuse the base for incremental calculations.
            GpuUInt128 current = baseValue;

            // We're manually assigning each field to prevent the compiler to initialize each field twice due to auto-initialization. We're using the action to lower the code base size.
            for (int i = 1; i < oddPowerCount; i++)
            {
                current.MulMod(baseValue, modulus);
                switch (i)
                {
                    case 1:
                        Element1 = current;
                        break;
                    case 2:
                        Element2 = current;
                        break;
                    case 3:
                        Element3 = current;
                        break;
                    case 4:
                        Element4 = current;
                        break;
                    case 5:
                        Element5 = current;
                        break;
                    case 6:
                        Element6 = current;
                        break;
                    case 7:
                        Element7 = current;
                        break;
                    case 8:
                        Element8 = current;
                        break;
                    case 9:
                        Element9 = current;
                        break;
                    case 10:
                        Element10 = current;
                        break;
                    case 11:
                        Element11 = current;
                        break;
                    case 12:
                        Element12 = current;
                        break;
                    case 13:
                        Element13 = current;
                        break;
                    case 14:
                        Element14 = current;
                        break;
                    case 15:
                        Element15 = current;
                        break;
                    case 16:
                        Element16 = current;
                        break;
                    case 17:
                        Element17 = current;
                        break;
                    case 18:
                        Element18 = current;
                        break;
                    case 19:
                        Element19 = current;
                        break;
                    case 20:
                        Element20 = current;
                        break;
                    case 21:
                        Element21 = current;
                        break;
                    case 22:
                        Element22 = current;
                        break;
                    case 23:
                        Element23 = current;
                        break;
                    case 24:
                        Element24 = current;
                        break;
                    case 25:
                        Element25 = current;
                        break;
                    case 26:
                        Element26 = current;
                        break;
                    case 27:
                        Element27 = current;
                        break;
                    case 28:
                        Element28 = current;
                        break;
                    case 29:
                        Element29 = current;
                        break;
                    case 30:
                        Element30 = current;
                        break;
                    case 31:
                        Element31 = current;
                        break;
                    case 32:
                        Element32 = current;
                        break;
                    case 33:
                        Element33 = current;
                        break;
                    case 34:
                        Element34 = current;
                        break;
                    case 35:
                        Element35 = current;
                        break;
                    case 36:
                        Element36 = current;
                        break;
                    case 37:
                        Element37 = current;
                        break;
                    case 38:
                        Element38 = current;
                        break;
                    case 39:
                        Element39 = current;
                        break;
                    case 40:
                        Element40 = current;
                        break;
                    case 41:
                        Element41 = current;
                        break;
                    case 42:
                        Element42 = current;
                        break;
                    case 43:
                        Element43 = current;
                        break;
                    case 44:
                        Element44 = current;
                        break;
                    case 45:
                        Element45 = current;
                        break;
                    case 46:
                        Element46 = current;
                        break;
                    case 47:
                        Element47 = current;
                        break;
                    case 48:
                        Element48 = current;
                        break;
                    case 49:
                        Element49 = current;
                        break;
                    case 50:
                        Element50 = current;
                        break;
                    case 51:
                        Element51 = current;
                        break;
                    case 52:
                        Element52 = current;
                        break;
                    case 53:
                        Element53 = current;
                        break;
                    case 54:
                        Element54 = current;
                        break;
                    case 55:
                        Element55 = current;
                        break;
                    case 56:
                        Element56 = current;
                        break;
                    case 57:
                        Element57 = current;
                        break;
                    case 58:
                        Element58 = current;
                        break;
                    case 59:
                        Element59 = current;
                        break;
                    case 60:
                        Element60 = current;
                        break;
                    case 61:
                        Element61 = current;
                        break;
                    case 62:
                        Element62 = current;
                        break;
                    case 63:
                        Element63 = current;
                        break;
                    case 64:
                        Element64 = current;
                        break;
                    case 65:
                        Element65 = current;
                        break;
                    case 66:
                        Element66 = current;
                        break;
                    case 67:
                        Element67 = current;
                        break;
                    case 68:
                        Element68 = current;
                        break;
                    case 69:
                        Element69 = current;
                        break;
                    case 70:
                        Element70 = current;
                        break;
                    case 71:
                        Element71 = current;
                        break;
                    case 72:
                        Element72 = current;
                        break;
                    case 73:
                        Element73 = current;
                        break;
                    case 74:
                        Element74 = current;
                        break;
                    case 75:
                        Element75 = current;
                        break;
                    case 76:
                        Element76 = current;
                        break;
                    case 77:
                        Element77 = current;
                        break;
                    case 78:
                        Element78 = current;
                        break;
                    case 79:
                        Element79 = current;
                        break;
                    case 80:
                        Element80 = current;
                        break;
                    case 81:
                        Element81 = current;
                        break;
                    case 82:
                        Element82 = current;
                        break;
                    case 83:
                        Element83 = current;
                        break;
                    case 84:
                        Element84 = current;
                        break;
                    case 85:
                        Element85 = current;
                        break;
                    case 86:
                        Element86 = current;
                        break;
                    case 87:
                        Element87 = current;
                        break;
                    case 88:
                        Element88 = current;
                        break;
                    case 89:
                        Element89 = current;
                        break;
                    case 90:
                        Element90 = current;
                        break;
                    case 91:
                        Element91 = current;
                        break;
                    case 92:
                        Element92 = current;
                        break;
                    case 93:
                        Element93 = current;
                        break;
                    case 94:
                        Element94 = current;
                        break;
                    case 95:
                        Element95 = current;
                        break;
                    case 96:
                        Element96 = current;
                        break;
                    case 97:
                        Element97 = current;
                        break;
                    case 98:
                        Element98 = current;
                        break;
                    case 99:
                        Element99 = current;
                        break;
                    case 100:
                        Element100 = current;
                        break;
                    case 101:
                        Element101 = current;
                        break;
                    case 102:
                        Element102 = current;
                        break;
                    case 103:
                        Element103 = current;
                        break;
                    case 104:
                        Element104 = current;
                        break;
                    case 105:
                        Element105 = current;
                        break;
                    case 106:
                        Element106 = current;
                        break;
                    case 107:
                        Element107 = current;
                        break;
                    case 108:
                        Element108 = current;
                        break;
                    case 109:
                        Element109 = current;
                        break;
                    case 110:
                        Element110 = current;
                        break;
                    case 111:
                        Element111 = current;
                        break;
                    case 112:
                        Element112 = current;
                        break;
                    case 113:
                        Element113 = current;
                        break;
                    case 114:
                        Element114 = current;
                        break;
                    case 115:
                        Element115 = current;
                        break;
                    case 116:
                        Element116 = current;
                        break;
                    case 117:
                        Element117 = current;
                        break;
                    case 118:
                        Element118 = current;
                        break;
                    case 119:
                        Element119 = current;
                        break;
                    case 120:
                        Element120 = current;
                        break;
                    case 121:
                        Element121 = current;
                        break;
                    case 122:
                        Element122 = current;
                        break;
                    case 123:
                        Element123 = current;
                        break;
                    case 124:
                        Element124 = current;
                        break;
                    case 125:
                        Element125 = current;
                        break;
                    case 126:
                        Element126 = current;
                        break;
                    case 127:
                        Element127 = current;
                        break;
                }
            }
        }

        public readonly GpuUInt128 Element0;
        public readonly GpuUInt128 Element1;
        public readonly GpuUInt128 Element2;
        public readonly GpuUInt128 Element3;
        public readonly GpuUInt128 Element4;
        public readonly GpuUInt128 Element5;
        public readonly GpuUInt128 Element6;
        public readonly GpuUInt128 Element7;
        public readonly GpuUInt128 Element8;
        public readonly GpuUInt128 Element9;
        public readonly GpuUInt128 Element10;
        public readonly GpuUInt128 Element11;
        public readonly GpuUInt128 Element12;
        public readonly GpuUInt128 Element13;
        public readonly GpuUInt128 Element14;
        public readonly GpuUInt128 Element15;
        public readonly GpuUInt128 Element16;
        public readonly GpuUInt128 Element17;
        public readonly GpuUInt128 Element18;
        public readonly GpuUInt128 Element19;
        public readonly GpuUInt128 Element20;
        public readonly GpuUInt128 Element21;
        public readonly GpuUInt128 Element22;
        public readonly GpuUInt128 Element23;
        public readonly GpuUInt128 Element24;
        public readonly GpuUInt128 Element25;
        public readonly GpuUInt128 Element26;
        public readonly GpuUInt128 Element27;
        public readonly GpuUInt128 Element28;
        public readonly GpuUInt128 Element29;
        public readonly GpuUInt128 Element30;
        public readonly GpuUInt128 Element31;
        public readonly GpuUInt128 Element32;
        public readonly GpuUInt128 Element33;
        public readonly GpuUInt128 Element34;
        public readonly GpuUInt128 Element35;
        public readonly GpuUInt128 Element36;
        public readonly GpuUInt128 Element37;
        public readonly GpuUInt128 Element38;
        public readonly GpuUInt128 Element39;
        public readonly GpuUInt128 Element40;
        public readonly GpuUInt128 Element41;
        public readonly GpuUInt128 Element42;
        public readonly GpuUInt128 Element43;
        public readonly GpuUInt128 Element44;
        public readonly GpuUInt128 Element45;
        public readonly GpuUInt128 Element46;
        public readonly GpuUInt128 Element47;
        public readonly GpuUInt128 Element48;
        public readonly GpuUInt128 Element49;
        public readonly GpuUInt128 Element50;
        public readonly GpuUInt128 Element51;
        public readonly GpuUInt128 Element52;
        public readonly GpuUInt128 Element53;
        public readonly GpuUInt128 Element54;
        public readonly GpuUInt128 Element55;
        public readonly GpuUInt128 Element56;
        public readonly GpuUInt128 Element57;
        public readonly GpuUInt128 Element58;
        public readonly GpuUInt128 Element59;
        public readonly GpuUInt128 Element60;
        public readonly GpuUInt128 Element61;
        public readonly GpuUInt128 Element62;
        public readonly GpuUInt128 Element63;
        public readonly GpuUInt128 Element64;
        public readonly GpuUInt128 Element65;
        public readonly GpuUInt128 Element66;
        public readonly GpuUInt128 Element67;
        public readonly GpuUInt128 Element68;
        public readonly GpuUInt128 Element69;
        public readonly GpuUInt128 Element70;
        public readonly GpuUInt128 Element71;
        public readonly GpuUInt128 Element72;
        public readonly GpuUInt128 Element73;
        public readonly GpuUInt128 Element74;
        public readonly GpuUInt128 Element75;
        public readonly GpuUInt128 Element76;
        public readonly GpuUInt128 Element77;
        public readonly GpuUInt128 Element78;
        public readonly GpuUInt128 Element79;
        public readonly GpuUInt128 Element80;
        public readonly GpuUInt128 Element81;
        public readonly GpuUInt128 Element82;
        public readonly GpuUInt128 Element83;
        public readonly GpuUInt128 Element84;
        public readonly GpuUInt128 Element85;
        public readonly GpuUInt128 Element86;
        public readonly GpuUInt128 Element87;
        public readonly GpuUInt128 Element88;
        public readonly GpuUInt128 Element89;
        public readonly GpuUInt128 Element90;
        public readonly GpuUInt128 Element91;
        public readonly GpuUInt128 Element92;
        public readonly GpuUInt128 Element93;
        public readonly GpuUInt128 Element94;
        public readonly GpuUInt128 Element95;
        public readonly GpuUInt128 Element96;
        public readonly GpuUInt128 Element97;
        public readonly GpuUInt128 Element98;
        public readonly GpuUInt128 Element99;
        public readonly GpuUInt128 Element100;
        public readonly GpuUInt128 Element101;
        public readonly GpuUInt128 Element102;
        public readonly GpuUInt128 Element103;
        public readonly GpuUInt128 Element104;
        public readonly GpuUInt128 Element105;
        public readonly GpuUInt128 Element106;
        public readonly GpuUInt128 Element107;
        public readonly GpuUInt128 Element108;
        public readonly GpuUInt128 Element109;
        public readonly GpuUInt128 Element110;
        public readonly GpuUInt128 Element111;
        public readonly GpuUInt128 Element112;
        public readonly GpuUInt128 Element113;
        public readonly GpuUInt128 Element114;
        public readonly GpuUInt128 Element115;
        public readonly GpuUInt128 Element116;
        public readonly GpuUInt128 Element117;
        public readonly GpuUInt128 Element118;
        public readonly GpuUInt128 Element119;
        public readonly GpuUInt128 Element120;
        public readonly GpuUInt128 Element121;
        public readonly GpuUInt128 Element122;
        public readonly GpuUInt128 Element123;
        public readonly GpuUInt128 Element124;
        public readonly GpuUInt128 Element125;
        public readonly GpuUInt128 Element126;
        public readonly GpuUInt128 Element127;

        public readonly GpuUInt128 this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => index switch
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
        }
    }

    public readonly struct Pow2OddPowerTableSetter
    {
        public Pow2OddPowerTableSetter(GpuUInt128 baseValue, GpuUInt128 modulus, int oddPowerCount)
        {
            Element0 = baseValue;
            if (oddPowerCount == 1)
            {
                return;
            }

            // Reusing baseValue to hold base^2 for the shared odd-power ladder that follows.
            baseValue.MulMod(baseValue, modulus);
            // TODO: We can calculate baseValue % modulus before loop and use it to increase ladderEntry calculation speed - we'll reuse the base for incremental calculations.
            GpuUInt128 current = baseValue;

            // We're manually assigning each field to prevent the compiler to initialize each field twice due to auto-initialization. We're using the action to lower the code base size.
            for (int i = 1; i < oddPowerCount; i++)
            {
                current.MulMod(baseValue, modulus);
                this[i] = current;
            }
        }

        public readonly GpuUInt128 Element0;
        public readonly GpuUInt128 Element1;
        public readonly GpuUInt128 Element2;
        public readonly GpuUInt128 Element3;
        public readonly GpuUInt128 Element4;
        public readonly GpuUInt128 Element5;
        public readonly GpuUInt128 Element6;
        public readonly GpuUInt128 Element7;
        public readonly GpuUInt128 Element8;
        public readonly GpuUInt128 Element9;
        public readonly GpuUInt128 Element10;
        public readonly GpuUInt128 Element11;
        public readonly GpuUInt128 Element12;
        public readonly GpuUInt128 Element13;
        public readonly GpuUInt128 Element14;
        public readonly GpuUInt128 Element15;
        public readonly GpuUInt128 Element16;
        public readonly GpuUInt128 Element17;
        public readonly GpuUInt128 Element18;
        public readonly GpuUInt128 Element19;
        public readonly GpuUInt128 Element20;
        public readonly GpuUInt128 Element21;
        public readonly GpuUInt128 Element22;
        public readonly GpuUInt128 Element23;
        public readonly GpuUInt128 Element24;
        public readonly GpuUInt128 Element25;
        public readonly GpuUInt128 Element26;
        public readonly GpuUInt128 Element27;
        public readonly GpuUInt128 Element28;
        public readonly GpuUInt128 Element29;
        public readonly GpuUInt128 Element30;
        public readonly GpuUInt128 Element31;
        public readonly GpuUInt128 Element32;
        public readonly GpuUInt128 Element33;
        public readonly GpuUInt128 Element34;
        public readonly GpuUInt128 Element35;
        public readonly GpuUInt128 Element36;
        public readonly GpuUInt128 Element37;
        public readonly GpuUInt128 Element38;
        public readonly GpuUInt128 Element39;
        public readonly GpuUInt128 Element40;
        public readonly GpuUInt128 Element41;
        public readonly GpuUInt128 Element42;
        public readonly GpuUInt128 Element43;
        public readonly GpuUInt128 Element44;
        public readonly GpuUInt128 Element45;
        public readonly GpuUInt128 Element46;
        public readonly GpuUInt128 Element47;
        public readonly GpuUInt128 Element48;
        public readonly GpuUInt128 Element49;
        public readonly GpuUInt128 Element50;
        public readonly GpuUInt128 Element51;
        public readonly GpuUInt128 Element52;
        public readonly GpuUInt128 Element53;
        public readonly GpuUInt128 Element54;
        public readonly GpuUInt128 Element55;
        public readonly GpuUInt128 Element56;
        public readonly GpuUInt128 Element57;
        public readonly GpuUInt128 Element58;
        public readonly GpuUInt128 Element59;
        public readonly GpuUInt128 Element60;
        public readonly GpuUInt128 Element61;
        public readonly GpuUInt128 Element62;
        public readonly GpuUInt128 Element63;
        public readonly GpuUInt128 Element64;
        public readonly GpuUInt128 Element65;
        public readonly GpuUInt128 Element66;
        public readonly GpuUInt128 Element67;
        public readonly GpuUInt128 Element68;
        public readonly GpuUInt128 Element69;
        public readonly GpuUInt128 Element70;
        public readonly GpuUInt128 Element71;
        public readonly GpuUInt128 Element72;
        public readonly GpuUInt128 Element73;
        public readonly GpuUInt128 Element74;
        public readonly GpuUInt128 Element75;
        public readonly GpuUInt128 Element76;
        public readonly GpuUInt128 Element77;
        public readonly GpuUInt128 Element78;
        public readonly GpuUInt128 Element79;
        public readonly GpuUInt128 Element80;
        public readonly GpuUInt128 Element81;
        public readonly GpuUInt128 Element82;
        public readonly GpuUInt128 Element83;
        public readonly GpuUInt128 Element84;
        public readonly GpuUInt128 Element85;
        public readonly GpuUInt128 Element86;
        public readonly GpuUInt128 Element87;
        public readonly GpuUInt128 Element88;
        public readonly GpuUInt128 Element89;
        public readonly GpuUInt128 Element90;
        public readonly GpuUInt128 Element91;
        public readonly GpuUInt128 Element92;
        public readonly GpuUInt128 Element93;
        public readonly GpuUInt128 Element94;
        public readonly GpuUInt128 Element95;
        public readonly GpuUInt128 Element96;
        public readonly GpuUInt128 Element97;
        public readonly GpuUInt128 Element98;
        public readonly GpuUInt128 Element99;
        public readonly GpuUInt128 Element100;
        public readonly GpuUInt128 Element101;
        public readonly GpuUInt128 Element102;
        public readonly GpuUInt128 Element103;
        public readonly GpuUInt128 Element104;
        public readonly GpuUInt128 Element105;
        public readonly GpuUInt128 Element106;
        public readonly GpuUInt128 Element107;
        public readonly GpuUInt128 Element108;
        public readonly GpuUInt128 Element109;
        public readonly GpuUInt128 Element110;
        public readonly GpuUInt128 Element111;
        public readonly GpuUInt128 Element112;
        public readonly GpuUInt128 Element113;
        public readonly GpuUInt128 Element114;
        public readonly GpuUInt128 Element115;
        public readonly GpuUInt128 Element116;
        public readonly GpuUInt128 Element117;
        public readonly GpuUInt128 Element118;
        public readonly GpuUInt128 Element119;
        public readonly GpuUInt128 Element120;
        public readonly GpuUInt128 Element121;
        public readonly GpuUInt128 Element122;
        public readonly GpuUInt128 Element123;
        public readonly GpuUInt128 Element124;
        public readonly GpuUInt128 Element125;
        public readonly GpuUInt128 Element126;
        public readonly GpuUInt128 Element127;

        public readonly GpuUInt128 this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => index switch
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
            init
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
    }

    [MemoryDiagnoser]
    [SimpleJob(RuntimeMoniker.Net80)]
    public class Pow2OddPowerTableBenchmarks
    {
        private static readonly GpuUInt128 BaseValue = new GpuUInt128(3, 123456789);
        private static readonly GpuUInt128 Modulus = new GpuUInt128(17, 987654321);
        private static readonly int OddPowerCount = 128;

        [Benchmark(Baseline = true)]
        public Pow2OddPowerTableLoop CreateLoop() => new Pow2OddPowerTableLoop(BaseValue, Modulus, OddPowerCount);

        [Benchmark]
        public Pow2OddPowerTableSetter CreateSetter() => new Pow2OddPowerTableSetter(BaseValue, Modulus, OddPowerCount);

        [Benchmark]
        public Pow2OddPowerTableDelegate CreateDelegate() => new Pow2OddPowerTableDelegate(BaseValue, Modulus, OddPowerCount);

        [Benchmark]
        public Pow2OddPowerTableReturn CreateReturn() => new Pow2OddPowerTableReturn(BaseValue, Modulus, OddPowerCount);
    }
}