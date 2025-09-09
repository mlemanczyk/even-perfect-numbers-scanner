using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public static class RleBlacklist
{
    private static readonly object Sync = new();
    private static volatile bool _loaded;
    private static HashSet<string>? _patterns;

    public static void Load(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return;
        }

        if (_loaded)
        {
            return;
        }

        lock (Sync)
        {
            if (_loaded)
            {
                return;
            }

            var set = new HashSet<string>(StringComparer.Ordinal);
            foreach (var line in File.ReadLines(path))
            {
                if (string.IsNullOrWhiteSpace(line))
                {
                    continue;
                }

                var span = line.AsSpan().Trim();
                if (span.Length == 0 || span[0] == '#')
                {
                    continue;
                }

                // CSV support: Results/rle_blacklist_composite_only_end7_base100k_validated_primes1M.csv
                // Header: rle_pattern,label,count_in_composites_end7_le_100k,appears_in_primes_le_1M
                // Row example: "[1, 1, 3]",composite,12,False
                // We only include rows with label==composite and appears_in_primes_le_1M==False.
                bool handledCsv = false;
                int commaIdx = span.IndexOf(',');
                if (commaIdx > 0)
                {
                    ReadOnlySpan<char> firstField;
                    int startNext;
                    if (span[0] == '"')
                    {
                        int endQuote = span.Slice(1).IndexOf('"');
                        if (endQuote > 0)
                        {
                            endQuote += 1; // adjust to original span
                            firstField = span.Slice(1, endQuote - 1);
                            // move to char after closing quote, expect comma
                            int after = endQuote + 1;
                            if (after < span.Length && span[after] == ',')
                            {
                                startNext = after + 1;
                            }
                            else
                            {
                                startNext = commaIdx + 1;
                            }
                        }
                        else
                        {
                            firstField = span.Slice(0, commaIdx);
                            startNext = commaIdx + 1;
                        }
                    }
                    else
                    {
                        firstField = span.Slice(0, commaIdx);
                        startNext = commaIdx + 1;
                    }

                    // Parse remaining CSV fields (label, count, appears_in_primes)
                    // label
                    int comma2 = span.Slice(startNext).IndexOf(',');
                    if (comma2 > 0)
                    {
                        ReadOnlySpan<char> label = span.Slice(startNext, comma2).Trim();
                        int startCount = startNext + comma2 + 1;
                        int comma3 = span.Slice(startCount).IndexOf(',');
                        if (comma3 > 0)
                        {
                            // Read appears_in_primes flag
                            int startFlag = startCount + comma3 + 1;
                            ReadOnlySpan<char> appears = span.Slice(startFlag).Trim().Trim('"');
                            bool composite = label.Equals("composite", StringComparison.OrdinalIgnoreCase);
                            bool appearsInPrimes = appears.Equals("True", StringComparison.OrdinalIgnoreCase);
                            if (composite && !appearsInPrimes)
                            {
                                var key = NormalizeBracketList(firstField);
                                if (key != null)
                                {
                                    set.Add(key);
                                }
                            }

                            handledCsv = true;
                        }
                    }
                }

                if (handledCsv)
                {
                    continue;
                }

                // Accept patterns as space/comma/dash-separated integers; optionally prefixed with firstbit|.
                var normalized = NormalizePattern(span);
                if (normalized != null)
                {
                    set.Add(normalized);
                }
            }

            _patterns = set;
            _loaded = true;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsLoaded() => _loaded && _patterns != null && _patterns.Count != 0;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool Matches(ulong p)
    {
        if (!IsLoaded())
        {
            return false;
        }

        string key = BuildRleKey(p);
        return _patterns!.Contains(key);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static string BuildRleKey(ulong value)
    {
        // Build RLE from MSB to LSB runs.
        int bitLen = 64 - int.CreateChecked(ulong.LeadingZeroCount(value));
        if (bitLen <= 0)
        {
            return "";
        }

        Span<int> runs = stackalloc int[64];
        int runCount = 0;
        int currentBit = (int)((value >> (bitLen - 1)) & 1UL);
        int count = 1;
        for (int i = bitLen - 2; i >= 0; i--)
        {
            int b = (int)((value >> i) & 1UL);
            if (b == currentBit)
            {
                count++;
            }
            else
            {
                runs[runCount++] = count;
                currentBit = b;
                count = 1;
            }
        }

        runs[runCount++] = count;

        // Key format: first bit value; then run lengths joined by '-'
        // Example: for binary 101100 -> first=1, runs: 1,1,2,2 -> "1|1-1-2-2"
        // This preserves starting bit which some datasets may rely on.
        var sb = StringBuilderPool.Rent();
        try
        {
            _ = sb.Append(((value >> (bitLen - 1)) & 1UL) == 0 ? '0' : '1');
            _ = sb.Append('|');
            for (int i = 0; i < runCount; i++)
            {
                if (i != 0)
                {
                    _ = sb.Append('-');
                }

                _ = sb.Append(runs[i]);
            }

            return sb.ToString();
        }
        finally
        {
            StringBuilderPool.Return(sb);
        }
    }

    private static string? NormalizePattern(ReadOnlySpan<char> span)
    {
        // Accept optional "firstbit|a-b-c" or just "a b c" / "a,b,c".
        int pipe = span.IndexOf('|');
        int start = 0;
        char first = '1';
        if (pipe >= 0)
        {
            if (pipe > 0 && (span[0] == '0' || span[0] == '1'))
            {
                first = span[0];
            }

            start = pipe + 1;
        }

        Span<int> tmp = stackalloc int[128];
        int count = 0;
        int i = start;
        while (i <= span.Length)
        {
            // Read token until separator: space/comma/dash or end
            int j = i;
            while (j < span.Length)
            {
                char c = span[j];
                if (c == ' ' || c == '\t' || c == ',' || c == ';' || c == '-')
                {
                    break;
                }

                j++;
            }

            if (j > i)
            {
                if (int.TryParse(span.Slice(i, j - i), out int v) && v > 0)
                {
                    tmp[count++] = v;
                }
                else
                {
                    return null;
                }
            }

            if (j >= span.Length)
            {
                break;
            }

            i = j + 1;
        }

        if (count == 0)
        {
            return null;
        }

        // Build normalized key
        var sb = StringBuilderPool.Rent();
        try
        {
            _ = sb.Append(first);
            _ = sb.Append('|');
            for (int k = 0; k < count; k++)
            {
                if (k != 0)
                {
                    _ = sb.Append('-');
                }

                _ = sb.Append(tmp[k]);
            }

            return sb.ToString();
        }
        finally
        {
            StringBuilderPool.Return(sb);
        }
    }

    private static string? NormalizeBracketList(ReadOnlySpan<char> span)
    {
        // Expect a JSON-like list: [1, 2, 3]
        int l = span.IndexOf('[');
        int r = span.LastIndexOf(']');
        if (l < 0 || r <= l)
        {
            return null;
        }

        ReadOnlySpan<char> inner = span.Slice(l + 1, r - l - 1);
        Span<int> tmp = stackalloc int[128];
        int count = 0;
        int i = 0;
        while (i <= inner.Length)
        {
            while (i < inner.Length && (inner[i] == ' ' || inner[i] == '\t' || inner[i] == ','))
            {
                i++;
            }

            if (i >= inner.Length)
            {
                break;
            }

            int j = i;
            while (j < inner.Length && inner[j] != ',' && inner[j] != ' ' && inner[j] != '\t')
            {
                j++;
            }

            if (j > i)
            {
                if (int.TryParse(inner.Slice(i, j - i), out int v) && v > 0)
                {
                    tmp[count++] = v;
                }
                else
                {
                    return null;
                }
            }

            i = j + 1;
        }

        if (count == 0)
        {
            return null;
        }

        // Assume first bit is 1 (MSB is 1 for any positive number), so starting run is ones.
        var sb = StringBuilderPool.Rent();
        try
        {
            _ = sb.Append('1');
            _ = sb.Append('|');
            for (int k = 0; k < count; k++)
            {
                if (k != 0)
                {
                    _ = sb.Append('-');
                }

                _ = sb.Append(tmp[k]);
            }

            return sb.ToString();
        }
        finally
        {
            StringBuilderPool.Return(sb);
        }
    }
}
