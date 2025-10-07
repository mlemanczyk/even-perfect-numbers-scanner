using System.Runtime.CompilerServices;
namespace PerfectNumbers.Core.Gpu;

internal static partial class PrimeOrderGpuHeuristics
{
    internal readonly struct Pow2OddPowerTableGpuUInt128
    {
        public Pow2OddPowerTableGpuUInt128(GpuUInt128 baseValue, GpuUInt128 modulus, int oddPowerCount)
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
}
