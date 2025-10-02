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

if (args.Length == 0)
{
    Console.Error.WriteLine("Usage: dotnet script apply_patch.cs -- <patch-file> [target-directory]");
    Environment.Exit(1);
}

var patchFilePath = args[0];
var targetRoot = args.Length > 1 ? Path.GetFullPath(args[1]) : Directory.GetCurrentDirectory();

if (!File.Exists(patchFilePath))
{
    Console.Error.WriteLine($"Patch file '{patchFilePath}' does not exist.");
    Environment.Exit(1);
}

if (!Directory.Exists(targetRoot))
{
    Console.Error.WriteLine($"Target directory '{targetRoot}' does not exist.");
    Environment.Exit(1);
}

var patchText = File.ReadAllLines(patchFilePath);
var parsedFiles = ParsePatch(patchText);

foreach (var patchFile in parsedFiles)
{
    ApplyPatch(targetRoot, patchFile);
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

static void ApplyPatch(string root, PatchFile patchFile)
{
    string targetPath = DetermineTargetPath(root, patchFile);

    switch (patchFile.Type)
    {
        case PatchFileType.Add:
            ApplyAdd(targetPath, patchFile);
            break;
        case PatchFileType.Delete:
            ApplyDelete(targetPath, patchFile);
            break;
        default:
            ApplyModify(targetPath, patchFile);
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

static void ApplyAdd(string targetPath, PatchFile patchFile)
{
    if (File.Exists(targetPath))
    {
        throw new InvalidOperationException($"Cannot add '{targetPath}' because it already exists.");
    }

    var directory = Path.GetDirectoryName(targetPath);

    if (!string.IsNullOrEmpty(directory))
    {
        Directory.CreateDirectory(directory);
    }
    var content = BuildContentFromHunks(patchFile.Hunks);
    File.WriteAllText(targetPath, content);
}

static void ApplyDelete(string targetPath, PatchFile patchFile)
{
    if (!File.Exists(targetPath))
    {
        throw new InvalidOperationException($"Cannot delete '{targetPath}' because it does not exist.");
    }

    var snapshot = ReadAllLinesPreservingNewlines(targetPath);
    ApplyHunks(snapshot, patchFile.Hunks);
    File.Delete(targetPath);
}

static void ApplyModify(string targetPath, PatchFile patchFile)
{
    if (!File.Exists(targetPath))
    {
        throw new InvalidOperationException($"Cannot modify '{targetPath}' because it does not exist.");
    }

    var snapshot = ReadAllLinesPreservingNewlines(targetPath);
    ApplyHunks(snapshot, patchFile.Hunks);
    WriteAllLinesPreservingNewlines(targetPath, snapshot);
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

static void ApplyHunks(FileSnapshot snapshot, List<PatchHunk> hunks)
{
    var lines = snapshot.Lines;
    int lineOffset = 0;
    bool? trailingNewLineOverride = null;

    foreach (var hunk in hunks)
    {
        int targetIndex = hunk.OriginalStart - 1 + lineOffset;

        foreach (var patchLine in hunk.Lines)
        {
            switch (patchLine)
            {
                case ContextLine context:
                    if (targetIndex >= lines.Count || !string.Equals(lines[targetIndex], context.Content, StringComparison.Ordinal))
                    {
                        throw new InvalidOperationException($"Context mismatch applying hunk at line {hunk.OriginalStart}.");
                    }

                    targetIndex++;
                    trailingNewLineOverride = context.HasTrailingNewLine;
                    break;
                case RemovedLine removed:
                    if (targetIndex >= lines.Count || !string.Equals(lines[targetIndex], removed.Content, StringComparison.Ordinal))
                    {
                        throw new InvalidOperationException($"Removal mismatch applying hunk at line {hunk.OriginalStart}.");
                    }

                    lines.RemoveAt(targetIndex);
                    lineOffset--;
                    break;
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
