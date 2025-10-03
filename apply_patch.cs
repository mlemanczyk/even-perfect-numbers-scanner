#nullable enable
using System;
using System.Collections.Generic;
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

var args = Args?.ToArray() ?? Array.Empty<string>();
if (args.Length == 0)
{
    PrintUsage();
    Environment.Exit(1);
}

bool checkOnly = false;
bool ignoreWhitespace = false;
bool ignoreLineEndings = false;
string? explicitTargetRoot = null;
var patchFileArguments = new List<string>();

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

    patchFileArguments.Add(argument);
}

if (patchFileArguments.Count == 0)
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

if (patchFileArguments.Count == 0)
{
    Console.Error.WriteLine("No patch files specified.");
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

foreach (var patchFilePath in patchFilePaths)
{
    try
    {
        if (!File.Exists(patchFilePath))
        {
            throw new FileNotFoundException($"Patch file '{patchFilePath}' does not exist.", patchFilePath);
        }

        var patchText = File.ReadAllLines(patchFilePath);
        var parsedFiles = ParsePatch(patchText);

        foreach (var patchFile in parsedFiles)
        {
            ApplyPatch(targetRoot, patchFile, applicationOptions);
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

foreach (var result in patchResults)
{
    var displayPath = GetDisplayPath(result.PatchFilePath);

    if (result.Success)
    {
        Console.WriteLine($"Patch '{displayPath}' applied successfully.");
    }
    else
    {
        Console.Error.WriteLine($"Patch '{displayPath}' failed: {result.ErrorMessage}");
    }
}

if (patchResults.Any(static r => !r.Success))
{
    Environment.Exit(1);
}

static List<PatchFile> ParsePatch(IReadOnlyList<string> lines)
{
    var files = new List<PatchFile>();
    int index = 0;

    while (index < lines.Count)
    {
        if (!lines[index].StartsWith("diff --git ", StringComparison.Ordinal))
        {
            index++;
            continue;
        }

        var diffHeader = lines[index];
        index++;

        string? originalPath = null;
        string? modifiedPath = null;
        var hunks = new List<PatchHunk>();
        PatchFileType type = PatchFileType.Modify;

        while (index < lines.Count)
        {
            var line = lines[index];

            if (line.StartsWith("diff --git ", StringComparison.Ordinal) && hunks.Count > 0)
            {
                break;
            }

            if (line.StartsWith("--- ", StringComparison.Ordinal))
            {
                originalPath = NormalizePath(line.Substring(4));
                index++;
                continue;
            }

            if (line.StartsWith("+++ ", StringComparison.Ordinal))
            {
                modifiedPath = NormalizePath(line.Substring(4));
                index++;
                continue;
            }

            if (line.StartsWith("@@", StringComparison.Ordinal))
            {
                var hunk = ParseHunk(lines, ref index);
                hunks.Add(hunk);
                continue;
            }

            index++;
        }

        if (originalPath is null || modifiedPath is null)
        {
            throw new InvalidOperationException($"Missing file paths for patch header: {diffHeader}");
        }

        if (IsDevNull(originalPath) && !IsDevNull(modifiedPath))
        {
            type = PatchFileType.Add;
        }
        else if (!IsDevNull(originalPath) && IsDevNull(modifiedPath))
        {
            type = PatchFileType.Delete;
        }

        files.Add(new PatchFile(originalPath, modifiedPath, hunks, type));
    }

    return files;
}

static PatchHunk ParseHunk(IReadOnlyList<string> lines, ref int index)
{
    var header = lines[index];
    var (originalStart, originalLength, modifiedStart, modifiedLength) = ParseHunkHeader(header);
    index++;

    var patchLines = new List<PatchLine>();

    while (index < lines.Count)
    {
        var line = lines[index];

        if (line.StartsWith("diff --git ", StringComparison.Ordinal) || line.StartsWith("@@", StringComparison.Ordinal))
        {
            break;
        }

        if (line.StartsWith("+++ ", StringComparison.Ordinal) || line.StartsWith("--- ", StringComparison.Ordinal))
        {
            break;
        }

        if (line.StartsWith("+", StringComparison.Ordinal))
        {
            patchLines.Add(new AddedLine(line.Substring(1), true));
            index++;
            continue;
        }

        if (line.StartsWith("-", StringComparison.Ordinal))
        {
            patchLines.Add(new RemovedLine(line.Substring(1), true));
            index++;
            continue;
        }

        if (line.StartsWith(" ", StringComparison.Ordinal))
        {
            patchLines.Add(new ContextLine(line.Substring(1), true));
            index++;
            continue;
        }

        if (line.StartsWith("\\ No newline at end of file", StringComparison.Ordinal))
        {
            MarkLastLineWithoutTrailingNewline(patchLines);
            index++;
            continue;
        }

        if (line.Length == 0)
        {
            patchLines.Add(new ContextLine(string.Empty, true));
            index++;
            continue;
        }

        throw new InvalidOperationException($"Unexpected line in hunk: '{line}'");
    }

    return new PatchHunk(originalStart, originalLength, modifiedStart, modifiedLength, patchLines);
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
    // Format: @@ -start,length +start,length @@ optional text
    int firstSpace = header.IndexOf(' ');
    int secondSpace = header.IndexOf(' ', firstSpace + 1);
    var originalRange = header.Substring(firstSpace + 1, secondSpace - firstSpace - 1);
    var modifiedRangeEnd = header.IndexOf(' ', secondSpace + 1);
    string modifiedRange;

    if (modifiedRangeEnd == -1)
    {
        modifiedRange = header.Substring(secondSpace + 1).TrimEnd('@').Trim();
    }
    else
    {
        modifiedRange = header.Substring(secondSpace + 1, modifiedRangeEnd - secondSpace - 1);
    }

    var (originalStart, originalLength) = ParseRange(originalRange);
    var (modifiedStart, modifiedLength) = ParseRange(modifiedRange);
    return (originalStart, originalLength, modifiedStart, modifiedLength);
}

static (int start, int length) ParseRange(string range)
{
    // Format: -start,length or +start,length
    int commaIndex = range.IndexOf(',');
    string startText;
    string lengthText;

    if (commaIndex >= 0)
    {
        startText = range.Substring(1, commaIndex - 1);
        lengthText = range[(commaIndex + 1)..];
    }
    else
    {
        startText = range.Substring(1);
        lengthText = "1";
    }

    return (int.Parse(startText, CultureInfo.InvariantCulture), int.Parse(lengthText, CultureInfo.InvariantCulture));
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

static string NormalizePath(string headerPath)
{
    var trimmed = headerPath.Trim();

    if (trimmed.StartsWith("a/", StringComparison.Ordinal) || trimmed.StartsWith("b/", StringComparison.Ordinal))
    {
        return trimmed.Substring(2);
    }

    return trimmed;
}

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

static void PrintUsage()
{
    Console.Error.WriteLine("Usage: dotnet script apply_patch.cs -- [options] <patch-file> [<patch-file> ...] [target-directory]");
    Console.Error.WriteLine();
    Console.Error.WriteLine("Options:");
    Console.Error.WriteLine("  --check                Validate patches without modifying files.");
    Console.Error.WriteLine("  --ignore-whitespace    Ignore whitespace differences when matching context.");
    Console.Error.WriteLine("  --ignore-eol           Ignore line-ending differences when matching context.");
    Console.Error.WriteLine("  --target <directory>   Explicitly set the target directory.");
}

