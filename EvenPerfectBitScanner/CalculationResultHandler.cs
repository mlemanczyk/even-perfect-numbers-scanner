using System.Buffers.Text;
using System.Runtime.CompilerServices;
using System.Text;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner;

internal static class CalculationResultHandler
{
        private const string ResultsHeader = "p,searchedMersenne,detailedCheck,passedAllTests";
        private const string PrimeFoundSuffix = " (FOUND VALID CANDIDATES)";
        internal const int DefaultWriteBatchSize = 100;

        private static readonly UTF8Encoding Utf8Encoding = new(false, true);
        private static readonly object Sync = new();
        private static readonly object FileWriteSync = new();

        private static StringBuilder? _outputBuilder;
        private static TextFileWriter? _resultsWriter;
        private static int _consoleCounter;
        private static int _writeIndex;
        private static int _writeBatchSize;
        private static string _resultsFileName = string.Empty;
        private static int _primeCount;
        private static bool _primeFoundAfterInit;

        internal static string ResultsFileName => _resultsFileName;

        internal static void Initialize(string resultsFileName, int writeBatchSize)
        {
                Dispose();

                _outputBuilder = null;
                _resultsWriter = null;
                _consoleCounter = 0;
                _writeIndex = 0;
                _writeBatchSize = writeBatchSize > 0 ? writeBatchSize : DefaultWriteBatchSize;
                _resultsFileName = resultsFileName;
                _primeCount = 0;
                _primeFoundAfterInit = false;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void InitializeOutputBuffer()
        {
                // Program flow ensures Initialize(...) runs before this call. Do not add null guards here.

                _outputBuilder = StringBuilderPool.Rent();
                _writeIndex = 0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void ReleaseOutputBuffer()
        {
                // Program flow ensures InitializeOutputBuffer() populated the builder. Do not add null checks here.

                StringBuilder builder = _outputBuilder!;
                // Program flow guarantees the output builder is populated when releasing. Do not add guard checks here.

                _outputBuilder = null;
                _writeIndex = 0;
                StringBuilderPool.Return(builder);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void CreateResultsFileWithHeader()
        {
                // Program flow guarantees FileWriteSync is available here. Do not add guard checks for the lock object.

                lock (FileWriteSync)
                {
                        _resultsWriter?.Dispose();
                        var writer = new TextFileWriter(_resultsFileName, FileMode.Create);
                        writer.WriteLine(ResultsHeader.AsSpan());
                        _resultsWriter = writer;
                }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void RegisterExistingResult(bool detailedCheck, bool passedAllTests)
        {
                if (!detailedCheck || !passedAllTests)
                {
                        return;
                }

                int newCount = _primeCount + 1;
                _primeCount = newCount;
                if (newCount >= 2)
                {
                        _primeFoundAfterInit = true;
                }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void HandleResult(ulong currentP, bool searchedMersenne, bool detailedCheck, bool passedAllTests, bool lastWasComposite)
        {
                // Program flow ensures Initialize(...) and InitializeOutputBuffer() ran before this call. Do not add initialization guards here.

                Span<byte> stackBuffer = stackalloc byte[128];
                byte[]? rentedBuffer = null;
                Span<byte> utf8Span = stackBuffer;
                int byteCount = TryFormatResultLine(utf8Span, currentP, searchedMersenne, detailedCheck, passedAllTests);
                if (byteCount < 0)
                {
                        rentedBuffer = ThreadStaticPools.BytePool.Rent(256);
                        utf8Span = rentedBuffer;
                        byteCount = TryFormatResultLine(utf8Span, currentP, searchedMersenne, detailedCheck, passedAllTests);
                        if (byteCount < 0)
                        {
                                ThreadStaticPools.BytePool.Return(rentedBuffer);
                                throw new InvalidOperationException("Result buffer too small.");
                        }
                }

                ReadOnlySpan<byte> utf8Line = utf8Span[..byteCount];
                Span<char> charBuffer = stackalloc char[128];
                int charCount = Utf8Encoding.GetChars(utf8Line, charBuffer);
                ReadOnlySpan<char> recordSpan = charBuffer[..charCount];

                bool primeCandidate = UpdatePrimeTracking(passedAllTests, lastWasComposite);
                bool printToConsole = TryAcquireConsoleSlot(primeCandidate, out bool primeFlag);
                StringBuilder? builderToFlush = AppendToOutputBuffer(recordSpan);

                if (builderToFlush is not null)
                {
                        FlushBuffer(builderToFlush);
                }

                if (printToConsole)
                {
                        ReadOnlySpan<char> consoleSpan = recordSpan[..^1];
                        TextWriter consoleWriter = Console.Out;
                        if (primeFlag)
                        {
                                consoleWriter.Write(consoleSpan);
                                consoleWriter.Write(PrimeFoundSuffix);
                                consoleWriter.WriteLine();
                        }
                        else
                        {
                                consoleWriter.WriteLine(consoleSpan);
                        }
                }

                if (rentedBuffer is not null)
                {
                        ThreadStaticPools.BytePool.Return(rentedBuffer);
                }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void FlushBuffer()
        {
                // Program flow ensures InitializeOutputBuffer() prepared the buffer. Do not add null checks here.

                StringBuilder? builderToFlush;

                lock (Sync)
                {
                        if (_writeIndex == 0 || _outputBuilder is null || _outputBuilder.Length == 0)
                        {
                                return;
                        }

                        builderToFlush = _outputBuilder;
                        _outputBuilder = StringBuilderPool.Rent();
                        _writeIndex = 0;
                }

                FlushBuffer(builderToFlush!);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void Dispose()
        {
                // Program flow guarantees FileWriteSync is ready when disposing. Do not add guard checks for the lock object.

                lock (FileWriteSync)
                {
                        _resultsWriter?.Dispose();
                        _resultsWriter = null;
                }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool TryAcquireConsoleSlot(bool primeCandidate, out bool primeFlag)
        {
                int consoleInterval = PerfectNumberConstants.ConsoleInterval;
                int counterValue = Interlocked.Increment(ref _consoleCounter);
                if (counterValue >= consoleInterval)
                {
                        int previous = Interlocked.Exchange(ref _consoleCounter, 0);
                        if (previous >= consoleInterval)
                        {
                                primeFlag = primeCandidate;
                                return true;
                        }
                }

                primeFlag = false;
                return false;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static StringBuilder? AppendToOutputBuffer(ReadOnlySpan<char> text)
        {
                StringBuilder? builderToFlush = null;

                lock (Sync)
                {
                        StringBuilder outputBuilder = _outputBuilder!;
                        _ = outputBuilder.Append(text);

                        int nextIndex = _writeIndex + 1;
                        if (nextIndex >= _writeBatchSize)
                        {
                                builderToFlush = outputBuilder;
                                _outputBuilder = StringBuilderPool.Rent();
                                _writeIndex = 0;
                        }
                        else
                        {
                                _writeIndex = nextIndex;
                        }
                }

                return builderToFlush;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int TryFormatResultLine(Span<byte> destination, ulong currentP, bool searchedMersenne, bool detailedCheck, bool passedAllTests)
        {
                int offset = 0;

                if (!Utf8Formatter.TryFormat(currentP, destination[offset..], out int written))
                {
                        return -1;
                }

                offset += written;
                if (offset >= destination.Length)
                {
                        return -1;
                }

                destination[offset++] = (byte)',';

                if (!Utf8Formatter.TryFormat(searchedMersenne, destination[offset..], out written))
                {
                        return -1;
                }

                offset += written;
                if (offset >= destination.Length)
                {
                        return -1;
                }

                destination[offset++] = (byte)',';

                if (!Utf8Formatter.TryFormat(detailedCheck, destination[offset..], out written))
                {
                        return -1;
                }

                offset += written;
                if (offset >= destination.Length)
                {
                        return -1;
                }

                destination[offset++] = (byte)',';

                if (!Utf8Formatter.TryFormat(passedAllTests, destination[offset..], out written))
                {
                        return -1;
                }

                offset += written;
                if (offset >= destination.Length)
                {
                        return -1;
                }

                destination[offset++] = (byte)'\n';
                return offset;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void FlushBuffer(StringBuilder builderToFlush)
        {
                // Program flow guarantees FileWriteSync is ready here as well. Do not add guard checks for the lock object.

                try
                {
                        lock (FileWriteSync)
                        {
                                TextFileWriter writer = EnsureResultsWriter();
                                writer.Write(builderToFlush);
                        }
                }
                finally
                {
                        builderToFlush.Clear();
                        StringBuilderPool.Return(builderToFlush);
                }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static TextFileWriter EnsureResultsWriter()
        {
                TextFileWriter? writer = _resultsWriter;
                if (writer is not null)
                {
                        return writer;
                }

                writer = new TextFileWriter(_resultsFileName, FileMode.Append);
                _resultsWriter = writer;
                return writer;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool UpdatePrimeTracking(bool passedAllTests, bool lastWasComposite)
        {
                if (passedAllTests)
                {
                        int newCount = Interlocked.Increment(ref _primeCount);
                        if (newCount >= 2)
                        {
                                Volatile.Write(ref _primeFoundAfterInit, true);
                        }
                }

                return passedAllTests && !lastWasComposite && Volatile.Read(ref _primeFoundAfterInit);
        }
}
