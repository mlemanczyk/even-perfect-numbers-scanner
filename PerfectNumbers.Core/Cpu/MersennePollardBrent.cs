using System;
using System.Buffers;
using System.Numerics;
using System.Security.Cryptography;

namespace PerfectNumbers.Core.Cpu;

internal sealed class MersennePollardBrent
{
    private readonly int _bitLength;
    private readonly int _byteLength;
    private readonly int _bitRemainder;
    private readonly byte _maskByte;
    private readonly BigInteger _modulus;
    private readonly BigInteger _mask;
    private readonly int _batchSize;
    private readonly bool _isSupported;

    public MersennePollardBrent(ulong exponent, int batchSize)
    {
        if (exponent < 3UL || exponent > int.MaxValue)
        {
            _isSupported = false;
            _bitLength = 0;
            _byteLength = 0;
            _bitRemainder = 0;
            _maskByte = 0;
            _modulus = BigInteger.Zero;
            _mask = BigInteger.Zero;
            _batchSize = 0;
            return;
        }

        _isSupported = true;
        _bitLength = (int)exponent;
        _byteLength = (_bitLength + 7) / 8;
        _bitRemainder = _bitLength & 7;
        _maskByte = _bitRemainder == 0 ? (byte)0xFF : (byte)((1 << _bitRemainder) - 1);
        _modulus = (BigInteger.One << _bitLength) - BigInteger.One;
        _mask = _modulus;
        _batchSize = Math.Clamp(batchSize, 8, 1024);
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

        int restarts = Math.Max(1, maxRestarts);
        int iterationBudget = Math.Max(_batchSize, maxIterationsPerRestart);

        for (int restart = 0; restart < restarts; restart++)
        {
            BigInteger x = NextRandomSeed();
            BigInteger y = x;
            BigInteger c = NextRandomConstant();
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
                    y = Iterate(y, c, ref localIterations);
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
                        y = Iterate(y, c, ref localIterations);
                        BigInteger diff = AbsSub(y, x);
                        if (diff.IsZero)
                        {
                            continue;
                        }

                        q = MulMod(q, diff);
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
                    ys = Iterate(ys, c, ref localIterations);
                    BigInteger diff = AbsSub(ys, x);
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

    private BigInteger Iterate(BigInteger value, BigInteger c, ref ulong iterationCounter)
    {
        iterationCounter++;
        value = SquareMod(value);
        value = AddMod(value, c);
        return value;
    }

    private BigInteger AddMod(BigInteger a, BigInteger b)
    {
        return Reduce(a + b);
    }

    private BigInteger SquareMod(BigInteger value)
    {
        return Reduce(value * value);
    }

    private BigInteger MulMod(BigInteger a, BigInteger b)
    {
        if (b == _modulus)
        {
            return a;
        }

        if (a == _modulus)
        {
            a = BigInteger.Zero;
        }

        return Reduce(a * b);
    }

    private BigInteger Reduce(BigInteger value)
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

    private BigInteger NextRandomSeed()
    {
        BigInteger value;
        do
        {
            value = NextRandomBelowModulus();
        }
        while (value <= BigInteger.One);

        return value;
    }

    private BigInteger NextRandomConstant()
    {
        BigInteger value;
        do
        {
            value = NextRandomBelowModulus();
        }
        while (value.IsZero);

        return value;
    }

    private BigInteger NextRandomBelowModulus()
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

    private static BigInteger AbsSub(BigInteger a, BigInteger b)
    {
        BigInteger diff = a - b;
        return diff.Sign < 0 ? -diff : diff;
    }
}
