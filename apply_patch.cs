#nullable enable
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

internal sealed record PatchFile(string OriginalPath, string ModifiedPath, List<PatchHunk> Hunks, PatchFileType Type);

internal enum PatchFileType
{
    Modify,
    Add,
    Delete
}

internal sealed record PatchHunk(int OriginalStart, int OriginalLength, int ModifiedStart, int ModifiedLength, List<PatchLine> Lines);

internal abstract record PatchLine(string Content, bool HasTrailingNewLine);

internal sealed record ContextLine(string Content, bool HasTrailingNewLine) : PatchLine(Content, HasTrailingNewLine);

internal sealed record AddedLine(string Content, bool HasTrailingNewLine) : PatchLine(Content, HasTrailingNewLine);

internal sealed record RemovedLine(string Content, bool HasTrailingNewLine) : PatchLine(Content, HasTrailingNewLine);

internal sealed class FileSnapshot
{
    public FileSnapshot(List<string> lines, string newLine, bool hadTrailingNewLine)
    {
        Lines = lines;
        NewLine = newLine;
        HadTrailingNewLine = hadTrailingNewLine;
    }

    public List<string> Lines { get; }

    public string NewLine { get; }

    public bool HadTrailingNewLine { get; set; }
}


internal readonly record struct PatchApplicationOptions(bool CheckOnly, bool IgnoreWhitespace, bool IgnoreLineEndings);

internal sealed record PatchRunResult(string PatchFilePath, bool Success, string? ErrorMessage);

internal sealed record PatchProfileEntry(
    string PatchFilePath,
    int FileCount,
    int HunkCount,
    int LineCount,
    long PatchBytes,
    TimeSpan ReadTime,
    TimeSpan ParseTime,
    TimeSpan ApplyTime);

internal sealed record LineReplacement(string RelativePath, int LineNumber, string? ExpectedContent, string NewContent);

internal sealed class LineReplacementException : Exception
{
    public LineReplacementException(LineReplacement replacement, string message)
        : base(message)
    {
        Replacement = replacement;
    }

    public LineReplacement Replacement { get; }
}

var args = Args?.ToArray() ?? Array.Empty<string>();
if (args.Length == 0)
{
    PrintUsage();
    Environment.Exit(1);
}

bool checkOnly = false;
bool ignoreWhitespace = false;
bool ignoreLineEndings = false;
bool profile = false;
string? explicitTargetRoot = null;
var patchFileArguments = new List<string>();
var lineReplacements = new List<LineReplacement>();

for (int i = 0; i < args.Length; i++)
{
    var argument = args[i];

    if (string.Equals(argument, "--check", StringComparison.Ordinal))
    {
        checkOnly = true;
        continue;
    }

    if (string.Equals(argument, "--ignore-whitespace", StringComparison.Ordinal))
    {
        ignoreWhitespace = true;
        continue;
    }

    if (string.Equals(argument, "--ignore-eol", StringComparison.Ordinal) || string.Equals(argument, "--ignore-line-endings", StringComparison.Ordinal))
    {
        ignoreLineEndings = true;
        continue;
    }

    if (string.Equals(argument, "--profile", StringComparison.Ordinal))
    {
        profile = true;
        continue;
    }

    if (string.Equals(argument, "--target", StringComparison.Ordinal))
    {
        if (i + 1 >= args.Length)
        {
            Console.Error.WriteLine("Missing value for --target option.");
            Environment.Exit(1);
        }

        explicitTargetRoot = Path.GetFullPath(args[++i]);
        continue;
    }

    if (string.Equals(argument, "--replace-line", StringComparison.Ordinal))
    {
        if (i + 4 >= args.Length)
        {
            Console.Error.WriteLine("--replace-line requires <path> <line-number> <expected-text> <new-text> arguments.");
            Environment.Exit(1);
        }

        var replacementPath = args[++i];

        if (!int.TryParse(args[++i], NumberStyles.Integer, CultureInfo.InvariantCulture, out var replacementLine) || replacementLine <= 0)
        {
            Console.Error.WriteLine("--replace-line requires a positive integer line number.");
            Environment.Exit(1);
        }

        var expected = args[++i];

        if (string.Equals(expected, "_", StringComparison.Ordinal))
        {
            expected = null;
        }

        var newContent = args[++i];
        lineReplacements.Add(new LineReplacement(replacementPath, replacementLine, expected, newContent));
        continue;
    }

    patchFileArguments.Add(argument);
}

if (patchFileArguments.Count == 0 && lineReplacements.Count == 0)
{
    PrintUsage();
    Environment.Exit(1);
}

string targetRoot;

if (explicitTargetRoot is not null)
{
    targetRoot = explicitTargetRoot;
}
else if (patchFileArguments.Count > 1)
{
    var potentialTarget = patchFileArguments[^1];

    if (Directory.Exists(potentialTarget))
    {
        targetRoot = Path.GetFullPath(potentialTarget);
        patchFileArguments.RemoveAt(patchFileArguments.Count - 1);
    }
    else
    {
        targetRoot = Directory.GetCurrentDirectory();
    }
}
else
{
    targetRoot = Directory.GetCurrentDirectory();
}

if (patchFileArguments.Count == 0 && lineReplacements.Count == 0)
{
    Console.Error.WriteLine("No patch files or line replacements specified.");
    Environment.Exit(1);
}

targetRoot = Path.GetFullPath(targetRoot);

if (!Directory.Exists(targetRoot))
{
    Console.Error.WriteLine($"Target directory '{targetRoot}' does not exist.");
    Environment.Exit(1);
}

var patchFilePaths = patchFileArguments.Select(Path.GetFullPath).ToList();
var applicationOptions = new PatchApplicationOptions(checkOnly, ignoreWhitespace, ignoreLineEndings);
var patchResults = new List<PatchRunResult>();
List<PatchProfileEntry>? patchProfiles = profile ? new List<PatchProfileEntry>() : null;

foreach (var patchFilePath in patchFilePaths)
{
    try
    {
        if (!File.Exists(patchFilePath))
        {
            throw new FileNotFoundException($"Patch file '{patchFilePath}' does not exist.", patchFilePath);
        }

        Stopwatch? stopwatch = null;
        TimeSpan readDuration = default;
        TimeSpan parseDuration = default;
        TimeSpan applyDuration = default;
        long patchBytes = 0L;
        int fileCount = 0;
        int hunkCount = 0;
        int lineCount = 0;

        if (profile)
        {
            stopwatch = Stopwatch.StartNew();
            patchBytes = new FileInfo(patchFilePath).Length;
        }

        var parsedFiles = ParsePatchFromFile(patchFilePath);

        if (stopwatch is not null)
        {
            readDuration = TimeSpan.Zero;
            parseDuration = stopwatch.Elapsed;
            fileCount = parsedFiles.Count;

            foreach (var parsedFile in parsedFiles)
            {
                hunkCount += parsedFile.Hunks.Count;

                foreach (var hunk in parsedFile.Hunks)
                {
                    lineCount += hunk.Lines.Count;
                }
            }

            stopwatch.Restart();
        }

        foreach (var patchFile in parsedFiles)
        {
            ApplyPatch(targetRoot, patchFile, applicationOptions);
        }

        if (stopwatch is not null && patchProfiles is not null)
        {
            applyDuration = stopwatch.Elapsed;
            patchProfiles.Add(new PatchProfileEntry(patchFilePath, fileCount, hunkCount, lineCount, patchBytes, readDuration, parseDuration, applyDuration));
        }

        patchResults.Add(new PatchRunResult(patchFilePath, true, null));

        if (!applicationOptions.CheckOnly)
        {
            TryDeletePatchFile(patchFilePath);
        }
    }
    catch (Exception ex)
    {
        patchResults.Add(new PatchRunResult(patchFilePath, false, ex.Message));
    }
}

if (lineReplacements.Count > 0)
{
    var replacementResults = ApplyLineReplacements(targetRoot, lineReplacements, applicationOptions);
    patchResults.AddRange(replacementResults);
}

foreach (var result in patchResults)
{
    var displayPath = GetDisplayPath(result.PatchFilePath);

    if (result.Success)
    {
        if (applicationOptions.CheckOnly)
        {
            Console.WriteLine($"Patch '{displayPath}' verified successfully. Ready to apply.");
        }
        else
        {
            Console.WriteLine($"Patch '{displayPath}' applied successfully.");
        }
    }
    else
    {
        Console.Error.WriteLine($"Patch '{displayPath}' failed: {result.ErrorMessage}");
    }
}

if (patchProfiles is { Count: > 0 })
{
    Console.WriteLine("Patch profiling summary:");

    foreach (var entry in patchProfiles)
    {
        var displayPath = GetDisplayPath(entry.PatchFilePath);
        Console.WriteLine(
            $"- {displayPath}: read {entry.ReadTime.TotalMilliseconds:F2} ms, parse {entry.ParseTime.TotalMilliseconds:F2} ms, apply {entry.ApplyTime.TotalMilliseconds:F2} ms " +
            $"(files={entry.FileCount}, hunks={entry.HunkCount}, lines={entry.LineCount}, bytes={entry.PatchBytes})");
    }

    var totalFiles = patchProfiles.Sum(static e => e.FileCount);
    var totalHunks = patchProfiles.Sum(static e => e.HunkCount);
    var totalLines = patchProfiles.Sum(static e => e.LineCount);
    var totalBytes = patchProfiles.Sum(static e => e.PatchBytes);
    var totalRead = TimeSpan.FromTicks(patchProfiles.Sum(static e => e.ReadTime.Ticks));
    var totalParse = TimeSpan.FromTicks(patchProfiles.Sum(static e => e.ParseTime.Ticks));
    var totalApply = TimeSpan.FromTicks(patchProfiles.Sum(static e => e.ApplyTime.Ticks));

    Console.WriteLine(
        $"Totals: read {totalRead.TotalMilliseconds:F2} ms, parse {totalParse.TotalMilliseconds:F2} ms, apply {totalApply.TotalMilliseconds:F2} ms " +
        $"(files={totalFiles}, hunks={totalHunks}, lines={totalLines}, bytes={totalBytes})");
}

if (patchResults.Any(static r => !r.Success))
{
    Environment.Exit(1);
}

static List<PatchFile> ParsePatchFromFile(string patchFilePath)
{
    using var stream = File.OpenRead(patchFilePath);
    using var reader = new StreamReader(stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true, bufferSize: 16_384, leaveOpen: false);
    using var lineReader = new PatchLineReader(reader);
    var files = ParsePatch(lineReader);
    return files;
}

static List<PatchFile> ParsePatch(PatchLineReader reader)
{
    var files = new List<PatchFile>();
    string? diffHeader = null;
    string? originalPath = null;
    string? modifiedPath = null;
    List<PatchHunk>? hunks = null;

    while (reader.TryRead(out var currentLine))
    {
        if (currentLine is null)
        {
            break;
        }

        if (string.IsNullOrWhiteSpace(currentLine))
        {
            continue;
        }

        if (!currentLine.StartsWith("diff --git ", StringComparison.Ordinal))
        {
            if (diffHeader is null)
            {
                continue;
            }

            if (currentLine.StartsWith("--- ", StringComparison.Ordinal))
            {
                originalPath = NormalizePath(currentLine.AsSpan(4));
                continue;
            }

            if (currentLine.StartsWith("+++ ", StringComparison.Ordinal))
            {
                modifiedPath = NormalizePath(currentLine.AsSpan(4));
                continue;
            }

            if (currentLine.StartsWith("@@", StringComparison.Ordinal))
            {
                hunks ??= new List<PatchHunk>();
                var hunk = ParseHunk(currentLine, reader);
                hunks.Add(hunk);
                continue;
            }

            continue;
        }

        if (diffHeader is not null)
        {
            FinalizePatchFile(files, diffHeader, originalPath, modifiedPath, hunks);
        }

        diffHeader = currentLine;
        originalPath = null;
        modifiedPath = null;
        hunks = new List<PatchHunk>();
    }

    if (diffHeader is not null)
    {
        FinalizePatchFile(files, diffHeader, originalPath, modifiedPath, hunks);
    }

    return files;
}

static PatchHunk ParseHunk(string header, PatchLineReader reader)
{
    var (originalStart, originalLength, modifiedStart, modifiedLength) = ParseHunkHeader(header);
    var patchLines = new List<PatchLine>();

    while (reader.TryPeek(out var nextLine))
    {
        if (nextLine is null)
        {
            break;
        }

        if (nextLine.StartsWith("diff --git ", StringComparison.Ordinal) ||
            nextLine.StartsWith("@@", StringComparison.Ordinal) ||
            nextLine.StartsWith("+++ ", StringComparison.Ordinal) ||
            nextLine.StartsWith("--- ", StringComparison.Ordinal))
        {
            break;
        }

        if (!reader.TryRead(out var consumed) || consumed is null)
        {
            break;
        }

        var line = consumed;

        if (line.StartsWith("+", StringComparison.Ordinal))
        {
            patchLines.Add(new AddedLine(line.Substring(1), true));
            continue;
        }

        if (line.StartsWith("-", StringComparison.Ordinal))
        {
            patchLines.Add(new RemovedLine(line.Substring(1), true));
            continue;
        }

        if (line.StartsWith(" ", StringComparison.Ordinal))
        {
            patchLines.Add(new ContextLine(line.Substring(1), true));
            continue;
        }

        if (line.StartsWith("\\ No newline at end of file", StringComparison.Ordinal))
        {
            MarkLastLineWithoutTrailingNewline(patchLines);
            continue;
        }

        if (line.Length == 0)
        {
            // Skip stray blank separators so hunks without an empty line still parse correctly.
            continue;
        }

        throw new InvalidOperationException($"Unexpected line in hunk: '{line}'");
    }

    return new PatchHunk(originalStart, originalLength, modifiedStart, modifiedLength, patchLines);
}

static void FinalizePatchFile(
    List<PatchFile> files,
    string diffHeader,
    string? originalPath,
    string? modifiedPath,
    List<PatchHunk>? hunks)
{
    if (originalPath is null || modifiedPath is null)
    {
        throw new InvalidOperationException($"Missing file paths for patch header: {diffHeader}");
    }

    PatchFileType type = PatchFileType.Modify;

    if (IsDevNull(originalPath) && !IsDevNull(modifiedPath))
    {
        type = PatchFileType.Add;
    }
    else if (!IsDevNull(originalPath) && IsDevNull(modifiedPath))
    {
        type = PatchFileType.Delete;
    }

    files.Add(new PatchFile(originalPath, modifiedPath, hunks ?? new List<PatchHunk>(), type));
}

private sealed class PatchLineReader : IDisposable
{
    private const int ReadBufferSize = 4096;
    private const int InitialLineBufferSize = 256;

    private readonly StreamReader _reader;
    private readonly char[] _readBuffer;
    private char[] _lineBuffer;
    private int _readLength;
    private int _readIndex;
    private string? _bufferedLine;
    private bool _hasBufferedLine;
    private bool _disposed;

    public PatchLineReader(StreamReader reader)
    {
        _reader = reader;
        _readBuffer = ArrayPool<char>.Shared.Rent(ReadBufferSize);
        _lineBuffer = ArrayPool<char>.Shared.Rent(InitialLineBufferSize);
    }

    public bool TryPeek(out string? line)
    {
        if (!_hasBufferedLine)
        {
            if (!TryReadLineInternal(out _bufferedLine))
            {
                line = null;
                return false;
            }

            _hasBufferedLine = true;
        }

        line = _bufferedLine;
        return line is not null;
    }

    public bool TryRead(out string? line)
    {
        if (_hasBufferedLine)
        {
            line = _bufferedLine;
            _bufferedLine = null;
            _hasBufferedLine = false;
            return line is not null;
        }

        return TryReadLineInternal(out line);
    }

    private bool TryReadLineInternal(out string? line)
    {
        if (_disposed)
        {
            line = null;
            return false;
        }

        int length = 0;

        while (true)
        {
            if (!TryReadChar(out char current))
            {
                if (length == 0)
                {
                    line = null;
                    return false;
                }

                break;
            }

            if (current == '\r')
            {
                if ((_readIndex < _readLength) || EnsureCharBuffer())
                {
                    if (_readIndex < _readLength && _readBuffer[_readIndex] == '\n')
                    {
                        _readIndex++;
                    }
                }

                break;
            }

            if (current == '\n')
            {
                break;
            }

            EnsureLineCapacity(length, length + 1);
            _lineBuffer[length++] = current;
        }

        line = new string(_lineBuffer, 0, length);
        return true;
    }

    private bool TryReadChar(out char value)
    {
        if (!EnsureCharBuffer())
        {
            value = '\0';
            return false;
        }

        value = _readBuffer[_readIndex++];
        return true;
    }

    private bool EnsureCharBuffer()
    {
        if (_readIndex < _readLength)
        {
            return true;
        }

        if (_disposed)
        {
            return false;
        }

        _readLength = _reader.Read(_readBuffer, 0, _readBuffer.Length);
        _readIndex = 0;
        return _readLength > 0;
    }

    private void EnsureLineCapacity(int currentLength, int requiredLength)
    {
        if (requiredLength <= _lineBuffer.Length)
        {
            return;
        }

        int newSize = _lineBuffer.Length * 2;
        while (newSize < requiredLength)
        {
            newSize *= 2;
        }

        char[] newBuffer = ArrayPool<char>.Shared.Rent(newSize);
        Array.Copy(_lineBuffer, newBuffer, currentLength);
        ArrayPool<char>.Shared.Return(_lineBuffer);
        _lineBuffer = newBuffer;
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        ArrayPool<char>.Shared.Return(_readBuffer);
        ArrayPool<char>.Shared.Return(_lineBuffer);
    }
}

static void MarkLastLineWithoutTrailingNewline(List<PatchLine> lines)
{
    if (lines.Count == 0)
    {
        return;
    }

    var last = lines[^1];

    lines[^1] = last switch
    {
        ContextLine context => context with { HasTrailingNewLine = false },
        AddedLine added => added with { HasTrailingNewLine = false },
        RemovedLine removed => removed with { HasTrailingNewLine = false },
        _ => last
    };
}

static (int originalStart, int originalLength, int modifiedStart, int modifiedLength) ParseHunkHeader(string header)
{
    var span = header.AsSpan();

    int firstSpace = span.IndexOf(' ');

    if (firstSpace < 0)
    {
        throw new InvalidOperationException($"Invalid hunk header: '{header}'.");
    }

    span = span[(firstSpace + 1)..];

    int secondSpace = span.IndexOf(' ');

    if (secondSpace < 0)
    {
        throw new InvalidOperationException($"Invalid hunk header: '{header}'.");
    }

    var originalRange = span[..secondSpace];
    span = span[(secondSpace + 1)..];

    int thirdSpace = span.IndexOf(' ');
    ReadOnlySpan<char> modifiedRange;

    if (thirdSpace < 0)
    {
        modifiedRange = span;
    }
    else
    {
        modifiedRange = span[..thirdSpace];
    }

    var (originalStart, originalLength) = ParseRange(originalRange);
    var (modifiedStart, modifiedLength) = ParseRange(modifiedRange);
    return (originalStart, originalLength, modifiedStart, modifiedLength);
}

static (int start, int length) ParseRange(ReadOnlySpan<char> range)
{
    if (range.Length == 0)
    {
        throw new InvalidOperationException("Range descriptor cannot be empty.");
    }

    if (range[0] != '-' && range[0] != '+')
    {
        throw new InvalidOperationException($"Unexpected range prefix: '{range[0]}' in '{range.ToString()}'.");
    }

    var numericPortion = range[1..];
    int commaIndex = numericPortion.IndexOf(',');
    ReadOnlySpan<char> startText;
    ReadOnlySpan<char> lengthText;

    if (commaIndex >= 0)
    {
        startText = numericPortion[..commaIndex];
        lengthText = numericPortion[(commaIndex + 1)..];
    }
    else
    {
        startText = numericPortion;
        lengthText = "1";
    }

    return (int.Parse(startText, NumberStyles.Integer, CultureInfo.InvariantCulture), int.Parse(lengthText, NumberStyles.Integer, CultureInfo.InvariantCulture));
}

static void ApplyPatch(string root, PatchFile patchFile, PatchApplicationOptions options)
{
    string targetPath = DetermineTargetPath(root, patchFile);

    switch (patchFile.Type)
    {
        case PatchFileType.Add:
            ApplyAdd(targetPath, patchFile, options);
            break;
        case PatchFileType.Delete:
            ApplyDelete(targetPath, patchFile, options);
            break;
        default:
            ApplyModify(targetPath, patchFile, options);
            break;
    }
}

static string DetermineTargetPath(string root, PatchFile patchFile)
{
    string path = IsDevNull(patchFile.ModifiedPath) ? patchFile.OriginalPath : patchFile.ModifiedPath;
    var normalized = NormalizeRelativePath(path);
    return Path.Combine(root, normalized);
}

static string NormalizeRelativePath(string path)
{
    if (path.StartsWith("a/", StringComparison.Ordinal) || path.StartsWith("b/", StringComparison.Ordinal))
    {
        return path.Substring(2);
    }

    return path;
}

static string NormalizePath(ReadOnlySpan<char> headerPath)
{
    var trimmed = headerPath.Trim();

    if (trimmed.StartsWith("a/", StringComparison.Ordinal) || trimmed.StartsWith("b/", StringComparison.Ordinal))
    {
        trimmed = trimmed[2..];
    }

    return new string(trimmed);
}

static string NormalizePath(string headerPath) => NormalizePath(headerPath.AsSpan());

static bool IsDevNull(string? path)
{
    return path is null || path.Equals("/dev/null", StringComparison.Ordinal);
}

static void ApplyAdd(string targetPath, PatchFile patchFile, PatchApplicationOptions options)
{
    if (File.Exists(targetPath))
    {
        throw new InvalidOperationException($"Cannot add '{targetPath}' because it already exists.");
    }

    var directory = Path.GetDirectoryName(targetPath);

    if (!string.IsNullOrEmpty(directory) && !options.CheckOnly)
    {
        Directory.CreateDirectory(directory);
    }

    if (options.CheckOnly)
    {
        return;
    }

    var content = BuildContentFromHunks(patchFile.Hunks);
    File.WriteAllText(targetPath, content);
}

static void ApplyDelete(string targetPath, PatchFile patchFile, PatchApplicationOptions options)
{
    if (!File.Exists(targetPath))
    {
        throw new InvalidOperationException($"Cannot delete '{targetPath}' because it does not exist.");
    }

    var snapshot = ReadAllLinesPreservingNewlines(targetPath);
    ApplyHunks(snapshot, patchFile.Hunks, options);

    if (!options.CheckOnly)
    {
        File.Delete(targetPath);
    }
}

static void ApplyModify(string targetPath, PatchFile patchFile, PatchApplicationOptions options)
{
    if (!File.Exists(targetPath))
    {
        throw new InvalidOperationException($"Cannot modify '{targetPath}' because it does not exist.");
    }

    var snapshot = ReadAllLinesPreservingNewlines(targetPath);
    ApplyHunks(snapshot, patchFile.Hunks, options);

    if (!options.CheckOnly)
    {
        WriteAllLinesPreservingNewlines(targetPath, snapshot);
    }
}

static string BuildContentFromHunks(List<PatchHunk> hunks)
{
    var addedLines = hunks.SelectMany(h => h.Lines.OfType<AddedLine>()).ToList();

    if (addedLines.Count == 0)
    {
        return string.Empty;
    }

    var builder = new StringBuilder();

    for (int i = 0; i < addedLines.Count; i++)
    {
        var added = addedLines[i];
        builder.Append(added.Content);

        if (added.HasTrailingNewLine || i < addedLines.Count - 1)
        {
            builder.Append(Environment.NewLine);
        }
    }

    return builder.ToString();
}

static void ApplyHunks(FileSnapshot snapshot, List<PatchHunk> hunks, PatchApplicationOptions options)
{
    var lines = snapshot.Lines;
    int lineOffset = 0;
    bool? trailingNewLineOverride = null;

    foreach (var hunk in hunks)
    {
        int targetIndex = hunk.OriginalStart - 1 + lineOffset;
        int relativeLine = 0;

        if (targetIndex < 0)
        {
            throw new InvalidOperationException($"Invalid hunk target index computed for original line {hunk.OriginalStart}.");
        }

        foreach (var patchLine in hunk.Lines)
        {
            switch (patchLine)
            {
                case ContextLine context:
                {
                    if (targetIndex >= lines.Count || !LinesMatch(lines[targetIndex], context.Content, options))
                    {
                        var actual = targetIndex < lines.Count ? lines[targetIndex] : null;
                        ThrowLineMismatch($"Context mismatch applying hunk starting at original line {hunk.OriginalStart + relativeLine}.", context.Content, actual, options);
                    }

                    targetIndex++;
                    trailingNewLineOverride = context.HasTrailingNewLine;
                    relativeLine++;
                    break;
                }
                case RemovedLine removed:
                {
                    if (targetIndex >= lines.Count || !LinesMatch(lines[targetIndex], removed.Content, options))
                    {
                        var actual = targetIndex < lines.Count ? lines[targetIndex] : null;
                        ThrowLineMismatch($"Removal mismatch applying hunk starting at original line {hunk.OriginalStart + relativeLine}.", removed.Content, actual, options);
                    }

                    lines.RemoveAt(targetIndex);
                    lineOffset--;
                    trailingNewLineOverride = removed.HasTrailingNewLine;
                    relativeLine++;
                    break;
                }
                case AddedLine added:
                {
                    lines.Insert(targetIndex, added.Content);
                    targetIndex++;
                    lineOffset++;
                    trailingNewLineOverride = added.HasTrailingNewLine;
                    break;
                }
            }
        }
    }

    if (trailingNewLineOverride.HasValue)
    {
        snapshot.HadTrailingNewLine = trailingNewLineOverride.Value;
    }
}

static FileSnapshot ReadAllLinesPreservingNewlines(string path)
{
    var text = File.ReadAllText(path);

    if (text.Length == 0)
    {
        return new FileSnapshot(new List<string>(), Environment.NewLine, false);
    }

    var newLine = text.Contains("\r\n", StringComparison.Ordinal) ? "\r\n" : "\n";
    var split = text.Split(new[] { "\r\n", "\n" }, StringSplitOptions.None).ToList();
    bool hadTrailingNewLine = text.EndsWith(newLine, StringComparison.Ordinal);

    if (hadTrailingNewLine && split.Count > 0 && split[^1].Length == 0)
    {
        split.RemoveAt(split.Count - 1);
    }

    return new FileSnapshot(split, newLine, hadTrailingNewLine);
}

static void WriteAllLinesPreservingNewlines(string path, FileSnapshot snapshot)
{
    var content = string.Join(snapshot.NewLine, snapshot.Lines);

    if (snapshot.HadTrailingNewLine)
    {
        content += snapshot.NewLine;
    }

    File.WriteAllText(path, content);
}


static bool LinesMatch(string actual, string expected, PatchApplicationOptions options)
{
    if (options.IgnoreLineEndings)
    {
        actual = NormalizeLineEndings(actual);
        expected = NormalizeLineEndings(expected);
    }

    if (options.IgnoreWhitespace)
    {
        actual = NormalizeWhitespace(actual);
        expected = NormalizeWhitespace(expected);
    }

    return string.Equals(actual, expected, StringComparison.Ordinal);
}

static string NormalizeWhitespace(string value)
{
    if (value.Length == 0)
    {
        return value;
    }

    var builder = new StringBuilder(value.Length);

    foreach (var ch in value)
    {
        if (!char.IsWhiteSpace(ch))
        {
            builder.Append(ch);
        }
    }

    return builder.ToString();
}

static string NormalizeLineEndings(string value)
{
    if (value.IndexOf('\r', StringComparison.Ordinal) < 0)
    {
        return value;
    }

    return value.Replace("\r", string.Empty, StringComparison.Ordinal);
}

static string FormatLineForMessage(string line)
{
    return '\'' + line
        .Replace("\\", "\\\\", StringComparison.Ordinal)
        .Replace("\t", "\\t", StringComparison.Ordinal)
        .Replace("\r", "\\r", StringComparison.Ordinal)
        .Replace("\n", "\\n", StringComparison.Ordinal)
        .Replace("'", "\'", StringComparison.Ordinal) + '\'';
}

static void ThrowLineMismatch(string message, string expected, string? actual, PatchApplicationOptions options)
{
    var builder = new StringBuilder();
    builder.AppendLine(message);
    builder.Append("Expected: ").AppendLine(FormatLineForMessage(expected));
    builder.Append("Actual  : ").AppendLine(actual is null ? "<missing line>" : FormatLineForMessage(actual));

    if (options.IgnoreWhitespace || options.IgnoreLineEndings)
    {
        builder.Append("Comparison settings: ");

        if (options.IgnoreWhitespace && options.IgnoreLineEndings)
        {
            builder.Append("ignoring whitespace and line endings.");
        }
        else if (options.IgnoreWhitespace)
        {
            builder.Append("ignoring whitespace.");
        }
        else
        {
            builder.Append("ignoring line endings.");
        }

        builder.AppendLine();
    }

    throw new InvalidOperationException(builder.ToString());
}

static string GetDisplayPath(string path)
{
    try
    {
        var current = Directory.GetCurrentDirectory();
        var relative = Path.GetRelativePath(current, path);

        if (!relative.Contains("..", StringComparison.Ordinal))
        {
            return relative;
        }
    }
    catch
    {
        return path;
    }

    return path;
}

static void TryDeletePatchFile(string patchFilePath)
{
    try
    {
        if (File.Exists(patchFilePath))
        {
            File.Delete(patchFilePath);
        }
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"Warning: Failed to remove patch file '{GetDisplayPath(patchFilePath)}': {ex.Message}");
    }
}

static List<PatchRunResult> ApplyLineReplacements(string root, List<LineReplacement> replacements, PatchApplicationOptions options)
{
    var results = new List<PatchRunResult>();

    foreach (var group in replacements.GroupBy(static r => NormalizeRelativePath(r.RelativePath), StringComparer.Ordinal))
    {
        var normalizedPath = group.Key;
        var replacementsForFile = group.OrderBy(static r => r.LineNumber).ToList();

        if (replacementsForFile.Count == 0)
        {
            continue;
        }

        try
        {
            var targetPath = Path.Combine(root, normalizedPath);

            if (!File.Exists(targetPath))
            {
                var displayPath = GetDisplayPath(targetPath);
                throw new LineReplacementException(replacementsForFile[0], $"Cannot replace line because file '{displayPath}' does not exist.");
            }

            var snapshot = ReadAllLinesPreservingNewlines(targetPath);
            var lines = snapshot.Lines;
            var pending = new List<(int Index, string NewContent)>();

            foreach (var replacement in replacementsForFile)
            {
                if (replacement.LineNumber <= 0)
                {
                    throw new LineReplacementException(replacement, $"Line numbers must be positive. Received {replacement.LineNumber} for '{GetDisplayPath(targetPath)}'.");
                }

                int index = replacement.LineNumber - 1;

                if (index >= lines.Count)
                {
                    throw new LineReplacementException(replacement, $"File '{GetDisplayPath(targetPath)}' contains only {lines.Count} lines. Cannot replace line {replacement.LineNumber}.");
                }

                if (replacement.ExpectedContent is not null)
                {
                    var actual = lines[index];

                    if (!LinesMatch(actual, replacement.ExpectedContent, options))
                    {
                        try
                        {
                            ThrowLineMismatch($"Line {replacement.LineNumber} mismatch in '{GetDisplayPath(targetPath)}'.", replacement.ExpectedContent, actual, options);
                        }
                        catch (InvalidOperationException ex)
                        {
                            throw new LineReplacementException(replacement, ex.Message);
                        }
                    }
                }

                pending.Add((index, replacement.NewContent));
            }

            if (!options.CheckOnly)
            {
                foreach (var (index, newContent) in pending)
                {
                    lines[index] = newContent;
                }

                WriteAllLinesPreservingNewlines(targetPath, snapshot);
            }

            foreach (var replacement in replacementsForFile)
            {
                results.Add(new PatchRunResult(BuildReplacementDisplay(replacement), true, null));
            }
        }
        catch (LineReplacementException ex)
        {
            results.Add(new PatchRunResult(BuildReplacementDisplay(ex.Replacement), false, ex.Message));
        }
        catch (Exception ex)
        {
            var fallback = replacementsForFile[0];
            results.Add(new PatchRunResult(BuildReplacementDisplay(fallback), false, ex.Message));
        }
    }

    return results;
}

static string BuildReplacementDisplay(LineReplacement replacement)
{
    var normalizedPath = NormalizeRelativePath(replacement.RelativePath);
    return $"replace-line:{normalizedPath}:{replacement.LineNumber}";
}

static void PrintUsage()
{
    Console.Error.WriteLine("Usage: dotnet script apply_patch.cs -- [options] <patch-file> [<patch-file> ...] [target-directory]");
    Console.Error.WriteLine();
    Console.Error.WriteLine("Options:");
    Console.Error.WriteLine("  --check                Validate patches without modifying files.");
    Console.Error.WriteLine("  --ignore-whitespace    Ignore whitespace differences when matching context.");
    Console.Error.WriteLine("  --ignore-eol           Ignore line-ending differences when matching context.");
    Console.Error.WriteLine("  --profile              Collect read/parse/apply timings for each patch file.");
    Console.Error.WriteLine("  --target <directory>   Explicitly set the target directory.");
    Console.Error.WriteLine("  --replace-line <path> <line-number> <expected-text> <new-text>");
    Console.Error.WriteLine("                        Replace a single line by absolute number; use '_' as expected-text to skip validation.");
}

