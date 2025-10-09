using System;
using System.Buffers;
using System.Numerics;
using System.Security.Cryptography;
using PerfectNumbers.Core;
using UInt128 = System.UInt128;

namespace PerfectNumbers.Core.Cpu;

internal sealed class MersennePollardBrent
{
    private enum ArithmeticMode
    {
        Unsupported,
        UInt64,
        UInt128,
        BigInteger,
    }

    private readonly ArithmeticMode _mode;
    private readonly int _bitLength;
    private readonly int _byteLength;
    private readonly int _bitRemainder;
    private readonly byte _maskByte;
    private readonly BigInteger _modulus;
    private readonly BigInteger _mask;
    private readonly ulong _modulus64;
    private readonly ulong _mask64;
    private readonly UInt128 _modulus128;
    private readonly UInt128 _mask128;
    private readonly int _batchSize;
    private readonly bool _isSupported;

    public MersennePollardBrent(ulong exponent, int batchSize)
    {
        if (exponent < 3UL || exponent > int.MaxValue)
        {
            _mode = ArithmeticMode.Unsupported;
            _isSupported = false;
            _bitLength = 0;
            _byteLength = 0;
            _bitRemainder = 0;
            _maskByte = 0;
            _modulus = BigInteger.Zero;
            _mask = BigInteger.Zero;
            _modulus64 = 0UL;
            _mask64 = 0UL;
            _modulus128 = UInt128.Zero;
            _mask128 = UInt128.Zero;
            _batchSize = 0;
            return;
        }

        _isSupported = true;
        _bitLength = (int)exponent;
        _byteLength = (_bitLength + 7) / 8;
        _bitRemainder = _bitLength & 7;
        _maskByte = _bitRemainder == 0 ? (byte)0xFF : (byte)((1 << _bitRemainder) - 1);
        _batchSize = Math.Clamp(batchSize, 8, 1024);

        if (_bitLength <= 64)
        {
            _mode = ArithmeticMode.UInt64;
            if (_bitLength == 64)
            {
                _modulus64 = ulong.MaxValue;
                _mask64 = ulong.MaxValue;
            }
            else
            {
                _modulus64 = (1UL << _bitLength) - 1UL;
                _mask64 = _modulus64;
            }

            _modulus128 = UInt128.Zero;
            _mask128 = UInt128.Zero;
            _modulus = BigInteger.Zero;
            _mask = BigInteger.Zero;
            return;
        }

        if (_bitLength <= 128)
        {
            _mode = ArithmeticMode.UInt128;
            if (_bitLength == 128)
            {
                _modulus128 = UInt128.MaxValue;
                _mask128 = UInt128.MaxValue;
            }
            else
            {
                _modulus128 = (UInt128.One << _bitLength) - UInt128.One;
                _mask128 = _modulus128;
            }

            _modulus64 = 0UL;
            _mask64 = 0UL;
            _modulus = BigInteger.Zero;
            _mask = BigInteger.Zero;
            return;
        }

        _mode = ArithmeticMode.BigInteger;
        _modulus = (BigInteger.One << _bitLength) - BigInteger.One;
        _mask = _modulus;
        _modulus64 = 0UL;
        _mask64 = 0UL;
        _modulus128 = UInt128.Zero;
        _mask128 = UInt128.Zero;
    }

    public bool IsSupported => _isSupported;

    public bool TryFindFactor(
        ulong allowedMax,
        int maxRestarts,
        int maxIterationsPerRestart,
        out ulong factor,
        out ulong iterationCount)
    {
        factor = 0UL;
        iterationCount = 0UL;

        if (!_isSupported)
        {
            return false;
        }

        if (allowedMax <= 1UL)
        {
            return false;
        }

        return _mode switch
        {
            ArithmeticMode.UInt64 => TryFindFactorUInt64(allowedMax, maxRestarts, maxIterationsPerRestart, out factor, out iterationCount),
            ArithmeticMode.UInt128 => TryFindFactorUInt128(allowedMax, maxRestarts, maxIterationsPerRestart, out factor, out iterationCount),
            ArithmeticMode.BigInteger => TryFindFactorBigInteger(allowedMax, maxRestarts, maxIterationsPerRestart, out factor, out iterationCount),
            _ => false,
        };
    }

    private bool TryFindFactorUInt64(
        ulong allowedMax,
        int maxRestarts,
        int maxIterationsPerRestart,
        out ulong factor,
        out ulong iterationCount)
    {
        factor = 0UL;
        iterationCount = 0UL;

        int restarts = Math.Max(1, maxRestarts);
        int iterationBudget = Math.Max(_batchSize, maxIterationsPerRestart);

        for (int restart = 0; restart < restarts; restart++)
        {
            ulong x = NextRandomSeed64();
            ulong y = x;
            ulong c = NextRandomConstant64();
            ulong q = 1UL;
            ulong g = 1UL;
            ulong ys = 0UL;
            int r = 1;
            ulong localIterations = 0UL;

            while (g == 1UL && localIterations < (ulong)iterationBudget)
            {
                x = y;
                for (int i = 0; i < r && localIterations < (ulong)iterationBudget; i++)
                {
                    y = Iterate64(y, c, ref localIterations);
                }

                if (localIterations >= (ulong)iterationBudget)
                {
                    break;
                }

                int k = 0;
                while (k < r && g == 1UL && localIterations < (ulong)iterationBudget)
                {
                    ys = y;
                    int t = Math.Min(_batchSize, r - k);
                    for (int i = 0; i < t; i++)
                    {
                        y = Iterate64(y, c, ref localIterations);
                        ulong diff = AbsSub64(y, x);
                        if (diff == 0UL)
                        {
                            continue;
                        }

                        q = MulMod64(q, diff);
                    }

                    g = BinaryGcd64(q, _modulus64);
                    k += t;
                }

                r <<= 1;
            }

            if (g == _modulus64)
            {
                do
                {
                    ys = Iterate64(ys, c, ref localIterations);
                    ulong diff = AbsSub64(ys, x);
                    if (diff == 0UL)
                    {
                        continue;
                    }

                    g = BinaryGcd64(diff, _modulus64);
                }
                while (g == 1UL);
            }

            iterationCount += localIterations;

            if (g == 1UL || g == _modulus64)
            {
                continue;
            }

            if (g > 1UL && g <= allowedMax)
            {
                factor = g;
                return true;
            }
        }

        return false;
    }

    private bool TryFindFactorUInt128(
        ulong allowedMax,
        int maxRestarts,
        int maxIterationsPerRestart,
        out ulong factor,
        out ulong iterationCount)
    {
        factor = 0UL;
        iterationCount = 0UL;

        int restarts = Math.Max(1, maxRestarts);
        int iterationBudget = Math.Max(_batchSize, maxIterationsPerRestart);
        UInt128 allowedMaxWide = (UInt128)allowedMax;

        for (int restart = 0; restart < restarts; restart++)
        {
            UInt128 x = NextRandomSeed128();
            UInt128 y = x;
            UInt128 c = NextRandomConstant128();
            UInt128 q = UInt128.One;
            UInt128 g = UInt128.One;
            UInt128 ys = UInt128.Zero;
            int r = 1;
            ulong localIterations = 0UL;

            while (g == UInt128.One && localIterations < (ulong)iterationBudget)
            {
                x = y;
                for (int i = 0; i < r && localIterations < (ulong)iterationBudget; i++)
                {
                    y = Iterate128(y, c, ref localIterations);
                }

                if (localIterations >= (ulong)iterationBudget)
                {
                    break;
                }

                int k = 0;
                while (k < r && g == UInt128.One && localIterations < (ulong)iterationBudget)
                {
                    ys = y;
                    int t = Math.Min(_batchSize, r - k);
                    for (int i = 0; i < t; i++)
                    {
                        y = Iterate128(y, c, ref localIterations);
                        UInt128 diff = AbsSub128(y, x);
                        if (diff == UInt128.Zero)
                        {
                            continue;
                        }

                        q = MulMod128(q, diff);
                    }

                    g = q.BinaryGcd(_modulus128);
                    k += t;
                }

                r <<= 1;
            }

            if (g == _modulus128)
            {
                do
                {
                    ys = Iterate128(ys, c, ref localIterations);
                    UInt128 diff = AbsSub128(ys, x);
                    if (diff == UInt128.Zero)
                    {
                        continue;
                    }

                    g = diff.BinaryGcd(_modulus128);
                }
                while (g == UInt128.One);
            }

            iterationCount += localIterations;

            if (g == UInt128.One || g == _modulus128)
            {
                continue;
            }

            if (g > UInt128.One && g <= allowedMaxWide)
            {
                factor = (ulong)g;
                return true;
            }
        }

        return false;
    }

    private bool TryFindFactorBigInteger(
        ulong allowedMax,
        int maxRestarts,
        int maxIterationsPerRestart,
        out ulong factor,
        out ulong iterationCount)
    {
        factor = 0UL;
        iterationCount = 0UL;

        int restarts = Math.Max(1, maxRestarts);
        int iterationBudget = Math.Max(_batchSize, maxIterationsPerRestart);

        for (int restart = 0; restart < restarts; restart++)
        {
            BigInteger x = NextRandomSeedBigInteger();
            BigInteger y = x;
            BigInteger c = NextRandomConstantBigInteger();
            BigInteger q = BigInteger.One;
            BigInteger g = BigInteger.One;
            BigInteger ys = BigInteger.Zero;
            int r = 1;
            ulong localIterations = 0UL;

            while (g.IsOne && localIterations < (ulong)iterationBudget)
            {
                x = y;
                for (int i = 0; i < r && localIterations < (ulong)iterationBudget; i++)
                {
                    y = IterateBigInteger(y, c, ref localIterations);
                }

                if (localIterations >= (ulong)iterationBudget)
                {
                    break;
                }

                int k = 0;
                while (k < r && g.IsOne && localIterations < (ulong)iterationBudget)
                {
                    ys = y;
                    int t = Math.Min(_batchSize, r - k);
                    for (int i = 0; i < t; i++)
                    {
                        y = IterateBigInteger(y, c, ref localIterations);
                        BigInteger diff = AbsSubBigInteger(y, x);
                        if (diff.IsZero)
                        {
                            continue;
                        }

                        q = MulModBigInteger(q, diff);
                    }

                    g = BigInteger.GreatestCommonDivisor(q, _modulus);
                    k += t;
                }

                r <<= 1;
            }

            if (g == _modulus)
            {
                do
                {
                    ys = IterateBigInteger(ys, c, ref localIterations);
                    BigInteger diff = AbsSubBigInteger(ys, x);
                    if (diff.IsZero)
                    {
                        continue;
                    }

                    g = BigInteger.GreatestCommonDivisor(diff, _modulus);
                }
                while (g.IsOne);
            }

            iterationCount += localIterations;

            if (g.IsOne || g == _modulus)
            {
                continue;
            }

            if (g <= ulong.MaxValue)
            {
                ulong candidate = (ulong)g;
                if (candidate > 1UL && candidate <= allowedMax)
                {
                    factor = candidate;
                    return true;
                }
            }
        }

        return false;
    }

    private ulong Iterate64(ulong value, ulong c, ref ulong iterationCounter)
    {
        iterationCounter++;
        value = SquareMod64(value);
        value = AddMod64(value, c);
        return value;
    }

    private UInt128 Iterate128(UInt128 value, UInt128 c, ref ulong iterationCounter)
    {
        iterationCounter++;
        value = SquareMod128(value);
        value = AddMod128(value, c);
        return value;
    }

    private BigInteger IterateBigInteger(BigInteger value, BigInteger c, ref ulong iterationCounter)
    {
        iterationCounter++;
        value = SquareModBigInteger(value);
        value = AddModBigInteger(value, c);
        return value;
    }

    private ulong AddMod64(ulong a, ulong b)
    {
        UInt128 sum = (UInt128)a + b;
        return Reduce64(sum);
    }

    private UInt128 AddMod128(UInt128 a, UInt128 b)
    {
        UInt128 sum = a + b;
        bool carryOut = sum < a;
        if (!carryOut && sum < _modulus128)
        {
            return sum;
        }

        ulong p3 = 0UL;
        ulong p2 = carryOut ? 1UL : 0UL;
        ulong p1 = (ulong)(sum >> 64);
        ulong p0 = (ulong)sum;
        return ReduceProductBitwise(p3, p2, p1, p0, _modulus128);
    }

    private BigInteger AddModBigInteger(BigInteger a, BigInteger b)
    {
        return ReduceBigInteger(a + b);
    }

    private ulong SquareMod64(ulong value)
    {
        UInt128 square = (UInt128)value * value;
        return Reduce64(square);
    }

    private UInt128 SquareMod128(UInt128 value)
    {
        return MulMod128(value, value);
    }

    private BigInteger SquareModBigInteger(BigInteger value)
    {
        return ReduceBigInteger(value * value);
    }

    private ulong MulMod64(ulong a, ulong b)
    {
        UInt128 product = (UInt128)a * b;
        return Reduce64(product);
    }

    private UInt128 MulMod128(UInt128 a, UInt128 b)
    {
        MultiplyFull(a, b, out ulong p3, out ulong p2, out ulong p1, out ulong p0);
        return ReduceProductBitwise(p3, p2, p1, p0, _modulus128);
    }

    private BigInteger MulModBigInteger(BigInteger a, BigInteger b)
    {
        if (b == _modulus)
        {
            return a;
        }

        if (a == _modulus)
        {
            a = BigInteger.Zero;
        }

        return ReduceBigInteger(a * b);
    }

    private ulong Reduce64(UInt128 value)
    {
        if (_bitLength == 64)
        {
            UInt128 combined = (UInt128)(ulong)value + (value >> 64);
            UInt128 modulus = _modulus64;
            if (combined >= modulus)
            {
                combined -= modulus;
                if (combined >= modulus)
                {
                    combined -= modulus;
                }
            }

            return (ulong)combined;
        }

        UInt128 mask = _mask64;
        UInt128 modulusValue = _modulus64;
        UInt128 working = value;
        while (true)
        {
            UInt128 high = working >> _bitLength;
            if (high == UInt128.Zero)
            {
                break;
            }

            working = (working & mask) + high;
        }

        if (working >= modulusValue)
        {
            working -= modulusValue;
            if (working >= modulusValue)
            {
                working %= modulusValue;
                if (working >= modulusValue)
                {
                    working -= modulusValue;
                }
            }
        }

        return (ulong)working;
    }

    private BigInteger ReduceBigInteger(BigInteger value)
    {
        if (value.Sign < 0)
        {
            value %= _modulus;
            if (value.Sign < 0)
            {
                value += _modulus;
            }

            return value;
        }

        while (true)
        {
            BigInteger high = value >> _bitLength;
            if (high.IsZero)
            {
                break;
            }

            value &= _mask;
            value += high;
        }

        if (value >= _modulus)
        {
            value -= _modulus;
            if (value >= _modulus)
            {
                value %= _modulus;
                if (value.Sign < 0)
                {
                    value += _modulus;
                }
            }
        }

        return value;
    }

    private ulong NextRandomSeed64()
    {
        ulong value;
        do
        {
            value = NextRandomBelowModulus64();
        }
        while (value <= 1UL);

        return value;
    }

    private UInt128 NextRandomSeed128()
    {
        UInt128 value;
        do
        {
            value = NextRandomBelowModulus128();
        }
        while (value <= UInt128.One);

        return value;
    }

    private BigInteger NextRandomSeedBigInteger()
    {
        BigInteger value;
        do
        {
            value = NextRandomBelowModulusBigInteger();
        }
        while (value <= BigInteger.One);

        return value;
    }

    private ulong NextRandomConstant64()
    {
        ulong value;
        do
        {
            value = NextRandomBelowModulus64();
        }
        while (value == 0UL);

        return value;
    }

    private UInt128 NextRandomConstant128()
    {
        UInt128 value;
        do
        {
            value = NextRandomBelowModulus128();
        }
        while (value == UInt128.Zero);

        return value;
    }

    private BigInteger NextRandomConstantBigInteger()
    {
        BigInteger value;
        do
        {
            value = NextRandomBelowModulusBigInteger();
        }
        while (value.IsZero);

        return value;
    }

    private ulong NextRandomBelowModulus64()
    {
        if (_byteLength == 0)
        {
            return 0UL;
        }

        Span<byte> buffer = stackalloc byte[_byteLength];
        ulong candidate;
        do
        {
            RandomNumberGenerator.Fill(buffer);
            if (_bitRemainder != 0)
            {
                buffer[_byteLength - 1] &= _maskByte;
            }

            candidate = (ulong)ReadUInt128LittleEndian(buffer);
        }
        while (candidate >= _modulus64);

        return candidate;
    }

    private UInt128 NextRandomBelowModulus128()
    {
        if (_byteLength == 0)
        {
            return UInt128.Zero;
        }

        Span<byte> buffer = stackalloc byte[_byteLength];
        UInt128 candidate;
        do
        {
            RandomNumberGenerator.Fill(buffer);
            if (_bitRemainder != 0)
            {
                buffer[_byteLength - 1] &= _maskByte;
            }

            candidate = ReadUInt128LittleEndian(buffer);
        }
        while (candidate >= _modulus128);

        return candidate;
    }

    private BigInteger NextRandomBelowModulusBigInteger()
    {
        if (_byteLength == 0)
        {
            return BigInteger.Zero;
        }

        byte[] buffer = ArrayPool<byte>.Shared.Rent(_byteLength);
        try
        {
            BigInteger value;
            do
            {
                RandomNumberGenerator.Fill(buffer.AsSpan(0, _byteLength));
                if (_bitRemainder != 0)
                {
                    buffer[_byteLength - 1] &= _maskByte;
                }

                value = new BigInteger(buffer, isUnsigned: true, isBigEndian: false);
            }
            while (value >= _modulus);

            return value;
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer, clearArray: true);
        }
    }

    private static ulong AbsSub64(ulong a, ulong b)
    {
        return a >= b ? a - b : b - a;
    }

    private static UInt128 AbsSub128(UInt128 a, UInt128 b)
    {
        return a >= b ? a - b : b - a;
    }

    private static BigInteger AbsSubBigInteger(BigInteger a, BigInteger b)
    {
        BigInteger diff = a - b;
        return diff.Sign < 0 ? -diff : diff;
    }

    private static ulong BinaryGcd64(ulong u, ulong v)
    {
        if (u == 0UL)
        {
            return v;
        }

        if (v == 0UL)
        {
            return u;
        }

        int shift = BitOperations.TrailingZeroCount(u | v);
        u >>= BitOperations.TrailingZeroCount(u);
        do
        {
            v >>= BitOperations.TrailingZeroCount(v);
            if (u > v)
            {
                (u, v) = (v, u);
            }

            v -= u;
        }
        while (v != 0UL);

        return u << shift;
    }

    private static void MultiplyFull(UInt128 left, UInt128 right, out ulong p3, out ulong p2, out ulong p1, out ulong p0)
    {
        ulong leftLow = (ulong)left;
        ulong leftHigh = (ulong)(left >> 64);
        ulong rightLow = (ulong)right;
        ulong rightHigh = (ulong)(right >> 64);

        (ulong High, ulong Low) partial0 = Multiply64(leftLow, rightLow);
        (ulong High, ulong Low) partial1 = Multiply64(leftLow, rightHigh);
        (ulong High, ulong Low) partial2 = Multiply64(leftHigh, rightLow);
        (ulong High, ulong Low) partial3 = Multiply64(leftHigh, rightHigh);

        p0 = partial0.Low;
        ulong carry = partial0.High;
        ulong sum = carry + partial1.Low;
        ulong carryMid = sum < partial1.Low ? 1UL : 0UL;
        sum += partial2.Low;
        if (sum < partial2.Low)
        {
            carryMid++;
        }

        p1 = sum;
        sum = partial1.High + partial2.High;
        ulong carryHigh = sum < partial2.High ? 1UL : 0UL;
        sum += partial3.Low;
        if (sum < partial3.Low)
        {
            carryHigh++;
        }

        sum += carryMid;
        if (sum < carryMid)
        {
            carryHigh++;
        }

        p2 = sum;
        p3 = partial3.High + carryHigh;
    }

    private static (ulong High, ulong Low) Multiply64(ulong left, ulong right)
    {
        ulong a0 = (uint)left;
        ulong a1 = left >> 32;
        ulong b0 = (uint)right;
        ulong b1 = right >> 32;

        ulong lo = a0 * b0;
        ulong mid1 = a1 * b0;
        b0 = a0 * b1;
        b1 *= a1;

        a0 = (lo >> 32) + (uint)mid1 + (uint)b0;
        a1 = (lo & 0xFFFFFFFFUL) | (a0 << 32);
        b1 += (mid1 >> 32) + (b0 >> 32) + (a0 >> 32);

        return (b1, a1);
    }

    private static UInt128 ReduceProductBitwise(ulong p3, ulong p2, ulong p1, ulong p0, UInt128 modulus)
    {
        UInt128 remainder = UInt128.Zero;
        for (int bit = 63; bit >= 0; bit--)
        {
            remainder <<= 1;
            remainder |= (UInt128)((p3 >> bit) & 1UL);
            if (remainder >= modulus)
            {
                remainder -= modulus;
            }
        }

        for (int bit = 63; bit >= 0; bit--)
        {
            remainder <<= 1;
            remainder |= (UInt128)((p2 >> bit) & 1UL);
            if (remainder >= modulus)
            {
                remainder -= modulus;
            }
        }

        for (int bit = 63; bit >= 0; bit--)
        {
            remainder <<= 1;
            remainder |= (UInt128)((p1 >> bit) & 1UL);
            if (remainder >= modulus)
            {
                remainder -= modulus;
            }
        }

        for (int bit = 63; bit >= 0; bit--)
        {
            remainder <<= 1;
            remainder |= (UInt128)((p0 >> bit) & 1UL);
            if (remainder >= modulus)
            {
                remainder -= modulus;
            }
        }

        return remainder;
    }

    private static UInt128 ReadUInt128LittleEndian(ReadOnlySpan<byte> buffer)
    {
        UInt128 value = UInt128.Zero;
        for (int i = buffer.Length - 1; i >= 0; i--)
        {
            value <<= 8;
            value |= buffer[i];
        }

        return value;
    }
}
