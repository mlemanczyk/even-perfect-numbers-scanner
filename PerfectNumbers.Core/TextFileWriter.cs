using System.Text;

namespace PerfectNumbers.Core;

/// <summary>
/// Provides thread-safe file writing for both text and binary data.
/// </summary>
public sealed class TextFileWriter
{
    private readonly FileStream _stream;
    private StreamWriter? _textWriter;
    private BinaryWriter? _binaryWriter;
    private readonly object _lock = new();

    public TextFileWriter(string path, FileMode mode)
    {
        var options = new FileStreamOptions
        {
            Access = FileAccess.Write,
            Mode = mode,
            Share = FileShare.Read
        };
        _stream = new FileStream(path, options);
    }

    public long Position
    {
        get
        {
            // TODO: Surface a lock-free position snapshot (for example via Interlocked.Read on a cached long)
            // so the high-frequency status probes measured in the writer throughput profiling stop contending on
            // the flush lock while runs follow the buffered pipeline planned for the scanner fast path.
            lock (_lock)
            {
                return _stream.Position;
            }
        }
    }

    private StreamWriter TextWriter => _textWriter ??= new StreamWriter(_stream, leaveOpen: true);
    private BinaryWriter BinaryWriter => _binaryWriter ??= new BinaryWriter(_stream, System.Text.Encoding.UTF8, leaveOpen: true);

    /// <summary>
    /// Clears the file by truncating it to zero length.
    /// </summary>
    public void Clear()
    {
        lock (_lock)
        {
            // TODO: Use the pooled writer buffers planned for the TextFileWriter fast path so truncation flushes
            // reuse the span caches identified alongside the Mod10_8_5_3Benchmarks instead of forcing synchronous
            // SetLength/Flush work on every restart.
            _stream.SetLength(0);
            _stream.Position = 0;
            _textWriter?.Flush();
            _binaryWriter?.Flush();
        }
    }

    /// <summary>
    /// Writes a line of text to the file.
    /// </summary>
    public void WriteLine(string line)
    {
        lock (_lock)
        {
            // TODO: Buffer these writes through an ArrayPool-backed accumulator so we only flush when the
            // batch is full; flushing per line shows up as a serialization bottleneck once the scanner emits
            // millions of candidate summaries.
            TextWriter.WriteLine(line);
            TextWriter.Flush();
        }
    }

    public void WriteLine(ReadOnlySpan<char> line)
    {
        lock (_lock)
        {
            StreamWriter writer = TextWriter;
            writer.Write(line);
            writer.WriteLine();
            writer.Flush();
        }
    }

    public void Write(ReadOnlySpan<char> text)
    {
        lock (_lock)
        {
            StreamWriter writer = TextWriter;
            writer.Write(text);
            writer.Flush();
        }
    }

    public void Write(StringBuilder builder)
    {
        lock (_lock)
        {
            StreamWriter writer = TextWriter;
            foreach (ReadOnlyMemory<char> chunk in builder.GetChunks())
            {
                writer.Write(chunk.Span);
            }

            writer.Flush();
        }
    }

    /// <summary>
    /// Executes a binary writing action within the internal lock.
    /// </summary>
    public void WriteBinary(Action<BinaryWriter> action)
    {
        lock (_lock)
        {
            // TODO: Buffer binary writes with the same ArrayPool-backed accumulator planned for text so GPU result
            // dumps can batch disk flushes instead of flushing on every invocation.
            action(BinaryWriter);
            BinaryWriter.Flush();
        }
    }

    public void Dispose()
    {
        // TODO: Return the writers to a pooled wrapper once the buffered pipeline lands so disposing the
        // TextFileWriter mirrors the zero-allocation strategy validated by the Mod10_8_5_3Benchmarks helpers.
        _binaryWriter?.Dispose();
        _textWriter?.Dispose();
        _stream.Dispose();
    }
}

