using System;
using System.Numerics;
using FluentAssertions;
using PerfectNumbers.Core.Gpu.Kernels;
using Xunit;

namespace PerfectNumbers.Core.Tests;

public class BitContradictionKernelsTests
{
    [Theory]
    [Trait("Category", "Fast")]
    [InlineData(31UL, 0UL, 0)]
    [InlineData(31UL, 0UL, 5)]
    [InlineData(61UL, 7UL, 3)]
    [InlineData(36721UL, 1UL, 0)]
    public void TryBuildQWordsFromBatchIndex_matches_BigInteger(ulong exponent, ulong batchIndex, int batchOffset)
    {
        int wordCount = BitContradictionKernels.MaxQWordCount;
        ulong[] batchIndexWords = new ulong[wordCount];
        batchIndexWords[0] = batchIndex;

        ulong[] qWords = new ulong[wordCount];
        bool ok = BitContradictionKernels.TryBuildQWordsFromBatchIndex(
            exponent,
            batchIndexWords,
            wordCount,
            batchOffset,
            qWords,
            out int qBitLen);

        ok.Should().BeTrue();

        BigInteger expectedK = ((BigInteger)batchIndex * BitContradictionKernels.BatchCount) + BigInteger.One + batchOffset;
        BigInteger expectedQ = (expectedK * (exponent << 1)) + BigInteger.One;
        BigInteger actualQ = ReadBigIntegerFromWords(qWords);

        actualQ.Should().Be(expectedQ);
        qBitLen.Should().Be(GetBitLength(expectedQ));
    }

    private static BigInteger ReadBigIntegerFromWords(ReadOnlySpan<ulong> words)
    {
        byte[] bytes = new byte[words.Length * sizeof(ulong)];
        for (int w = 0; w < words.Length; w++)
        {
            ulong word = words[w];
            int byteIndex = w << 3;
            bytes[byteIndex] = (byte)word;
            bytes[byteIndex + 1] = (byte)(word >> 8);
            bytes[byteIndex + 2] = (byte)(word >> 16);
            bytes[byteIndex + 3] = (byte)(word >> 24);
            bytes[byteIndex + 4] = (byte)(word >> 32);
            bytes[byteIndex + 5] = (byte)(word >> 40);
            bytes[byteIndex + 6] = (byte)(word >> 48);
            bytes[byteIndex + 7] = (byte)(word >> 56);
        }

        return new BigInteger(bytes, isUnsigned: true, isBigEndian: false);
    }

    private static int GetBitLength(BigInteger value)
    {
        if (value.IsZero)
        {
            return 0;
        }

        byte[] bytes = value.ToByteArray(isUnsigned: true, isBigEndian: false);
        int last = bytes[^1];
        int bits = (bytes.Length - 1) * 8;
        while (last != 0)
        {
            bits++;
            last >>= 1;
        }

        return bits;
    }
}
