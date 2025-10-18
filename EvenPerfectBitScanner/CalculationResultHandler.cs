using System.Buffers.Text;
using System.IO;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner;

internal struct CalculationResultHandler
{
    private const string ResultsHeader = "p,searchedMersenne,detailedCheck,passedAllTests";
    private const string PrimeFoundSuffix = " (FOUND VALID CANDIDATES)";
    internal const int DefaultWriteBatchSize = 100;

    private static readonly UTF8Encoding Utf8Encoding = new(false, true);

    private readonly object _sync;
    private readonly object _fileWriteSync;
    private StringBuilder? _outputBuilder;
    private TextFileWriter? _resultsWriter;
    private int _consoleCounter;
    private int _writeIndex;
    private readonly int _writeBatchSize;
    private readonly string _resultsFileName;
    private int _primeCount;
    private bool _primeFoundAfterInit;

    private bool IsInitialized => _sync is not null;
    private bool HasFileSync => _fileWriteSync is not null;

    internal CalculationResultHandler(string resultsFileName, int writeBatchSize)
    {
        _sync = new object();
        _fileWriteSync = new object();
        _outputBuilder = null;
        _resultsWriter = null;
        _consoleCounter = 0;
        _writeIndex = 0;
        _writeBatchSize = writeBatchSize > 0 ? writeBatchSize : DefaultWriteBatchSize;
        _resultsFileName = resultsFileName;
        _primeCount = 0;
        _primeFoundAfterInit = false;
    }

    internal string ResultsFileName => _resultsFileName;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal void InitializeOutputBuffer()
    {
        // Program flow ensures the handler is initialized before this call. Do not uncomment the guard below.
        // if (!IsInitialized)
        // {
        //     return;
        // }

        _outputBuilder = StringBuilderPool.Rent();
        _writeIndex = 0;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal void ReleaseOutputBuffer()
    {
        // Program flow ensures the handler is initialized before this call. Do not uncomment the guard below.
        // if (!IsInitialized)
        // {
        //     return;
        // }

        StringBuilder builder = _outputBuilder!;
        // Program flow guarantees the output builder is populated when releasing. Do not uncomment the guard below.
        // if (builder is null)
        // {
        //     return;
        // }

        _outputBuilder = null;
        _writeIndex = 0;
        StringBuilderPool.Return(builder);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal void CreateResultsFileWithHeader()
    {
        // Program flow guarantees the file synchronization primitives are available here. Do not uncomment the guard below.
        // if (!HasFileSync)
        // {
        //     return;
        // }

        lock (_fileWriteSync)
        {
            _resultsWriter?.Dispose();
            var writer = new TextFileWriter(_resultsFileName, FileMode.Create);
            writer.WriteLine(ResultsHeader.AsSpan());
            _resultsWriter = writer;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal void RegisterExistingResult(bool detailedCheck, bool passedAllTests)
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
    internal void HandleResult(ulong currentP, bool searchedMersenne, bool detailedCheck, bool passedAllTests, bool lastWasComposite)
    {
        // Program flow ensures the handler is initialized before this call. Do not uncomment the guard below.
        // if (!IsInitialized)
        // {
        //     return;
        // }

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
    internal void FlushBuffer()
    {
        // Program flow ensures the handler is initialized before this call. Do not uncomment the guard below.
        // if (!IsInitialized)
        // {
        //     return;
        // }

        StringBuilder? builderToFlush;

        lock (_sync)
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
    internal void Dispose()
    {
        // Program flow guarantees the file synchronization primitives are available here. Do not uncomment the guard below.
        // if (!HasFileSync)
        // {
        //     return;
        // }

        lock (_fileWriteSync)
        {
            _resultsWriter?.Dispose();
            _resultsWriter = null;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private bool TryAcquireConsoleSlot(bool primeCandidate, out bool primeFlag)
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
    private StringBuilder? AppendToOutputBuffer(ReadOnlySpan<char> text)
    {
        StringBuilder? builderToFlush = null;

        lock (_sync)
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
    private int TryFormatResultLine(Span<byte> destination, ulong currentP, bool searchedMersenne, bool detailedCheck, bool passedAllTests)
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
    private void FlushBuffer(StringBuilder builderToFlush)
    {
        // Program flow guarantees the file synchronization primitives are available here. Do not uncomment the guard below.
        // if (!HasFileSync)
        // {
        //     builderToFlush.Clear();
        //     StringBuilderPool.Return(builderToFlush);
        //     return;
        // }

        try
        {
            lock (_fileWriteSync)
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
    private TextFileWriter EnsureResultsWriter()
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
    private bool UpdatePrimeTracking(bool passedAllTests, bool lastWasComposite)
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
