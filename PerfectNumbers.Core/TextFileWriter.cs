namespace PerfectNumbers.Core;

/// <summary>
/// Provides thread-safe file writing for both text and binary data.
/// </summary>
public sealed class TextFileWriter : IDisposable
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
        get { lock (_lock) { return _stream.Position; } }
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
            TextWriter.WriteLine(line);
            TextWriter.Flush();
        }
    }

    /// <summary>
    /// Executes a binary writing action within the internal lock.
    /// </summary>
    public void WriteBinary(Action<BinaryWriter> action)
    {
        lock (_lock)
        {
            action(BinaryWriter);
            BinaryWriter.Flush();
        }
    }

    public void Dispose()
    {
        _binaryWriter?.Dispose();
        _textWriter?.Dispose();
        _stream.Dispose();
    }
}

