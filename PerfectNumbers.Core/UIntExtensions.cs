using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public static class UIntExtensions
{
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Mod3(this uint value) => PerfectNumbersMath.FastRemainder3(value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Mod5(this uint value) => PerfectNumbersMath.FastRemainder5(value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Mod7(this uint value)
        {
                uint remainder = 0U;
                uint temp = value;

                while (temp != 0U)
                {
                        remainder += temp & 7U;
                        temp >>= 3;
                        if (remainder >= 7U)
                        {
                                remainder -= 7U;
                        }
                }

                return remainder >= 7U ? remainder - 7U : remainder;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Mod8(this uint value) => value & 7U;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Mod10(this uint value)
        {
                return value - (uint)(((ulong)value * 0xCCCCCCCDUL) >> 35) * 10U;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Mod11(this uint value)
        {
                uint remainder = 0U;
                uint temp = value;

                while (temp != 0U)
                {
                        remainder += temp & 1023U;
                        temp >>= 10;
                        remainder -= 11U * (remainder / 11U);
                }

                return remainder;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Mod128(this uint value) => value & 127U;
}

