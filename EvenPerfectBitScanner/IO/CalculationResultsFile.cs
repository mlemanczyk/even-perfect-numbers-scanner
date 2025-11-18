using Open.Collections;
using PerfectNumbers.Core;
using PerfectNumbers.Core.Gpu;

namespace EvenPerfectBitScanner.IO;

internal static class CalculationResultsFile
{
        internal static List<ulong> LoadCandidatesWithinRange(string candidateFile, UInt128 maxPrimeLimit, bool maxPrimeConfigured, out int skippedByLimit)
        {
                List<ulong> candidates = [];
                skippedByLimit = 0;
                using FileStream readStream = new(candidateFile, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
                using StreamReader reader = new(readStream);
                string? line;
                ReadOnlySpan<char> trimmed;
                while ((line = reader.ReadLine()) is not null)
                {
                        if (string.IsNullOrWhiteSpace(line))
                        {
                                continue;
                        }

                        trimmed = line.AsSpan().Trim();
                        if (trimmed.IsEmpty || trimmed[0] == '#')
                        {
                                continue;
                        }

                        int length = trimmed.Length;
                        int index = 0;
                        if (!maxPrimeConfigured)
                        {
                                while (index < length)
                                {
                                        char current = trimmed[index];
                                        uint digit = (uint)(current - '0');
                                        if (digit > 9U)
                                        {
                                                index++;
                                                continue;
                                        }

                                        int start = index++;
                                        while (index < length && (uint)(trimmed[index] - '0') <= 9U)
                                        {
                                                index++;
                                        }

                                        if (Utf8CliParser.TryParseUInt64(trimmed[start..index], out ulong parsed))
                                        {
                                                candidates.Add(parsed);
                                        }
                                }
                        }
                        else
                        {
                                UInt128 limit = maxPrimeLimit;
                                while (index < length)
                                {
                                        char current = trimmed[index];
                                        uint digit = (uint)(current - '0');
                                        if (digit > 9U)
                                        {
                                                index++;
                                                continue;
                                        }

                                        int start = index++;
                                        while (index < length && (uint)(trimmed[index] - '0') <= 9U)
                                        {
                                                index++;
                                        }

                                        if (Utf8CliParser.TryParseUInt64(trimmed[start..index], out ulong parsed))
                                        {
                                                if ((UInt128)parsed <= limit)
                                                {
                                                        candidates.Add(parsed);
                                                }
                                                else
                                                {
                                                        skippedByLimit++;
                                                }
                                        }
                                }
                        }
                }

                return candidates;
        }

        internal static void EnumerateCandidates(string resultsFileName, Action<ulong, bool, bool> lineProcessorAction)
        {
                using FileStream readStream = new(resultsFileName, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
                using StreamReader reader = new(readStream);
                string? line;
                bool headerSkipped = false;
                ReadOnlySpan<char> span;
                int first = 0;
                int second = 0;
                int third = 0;
                ulong parsedP = 0UL;
                ReadOnlySpan<char> detailedSpan = default;
                ReadOnlySpan<char> passedAllTestsSpan = default;
                bool detailed = false;
                bool passedAllTests = false;
                while ((line = reader.ReadLine()) != null)
                {
                        if (!headerSkipped)
                        {
                                headerSkipped = true;
                                continue;
                        }

                        if (string.IsNullOrWhiteSpace(line))
                        {
                                continue;
                        }

                        span = line.AsSpan();
                        first = span.IndexOf(',');
                        if (first < 0)
                        {
                                continue;
                        }
                        parsedP = Utf8CliParser.ParseUInt64(span[..first]);
                        span = span[(first + 1)..];
                        second = span.IndexOf(',');
                        if (second < 0)
                        {
                                continue;
                        }

                        span = span[(second + 1)..];
                        third = span.IndexOf(',');
                        if (third < 0)
                        {
                                continue;
                        }

                        detailedSpan = span[..third];
                        passedAllTestsSpan = span[(third + 1)..];

                        if (Utf8CliParser.TryParseBoolean(detailedSpan, out detailed) && Utf8CliParser.TryParseBoolean(passedAllTestsSpan, out passedAllTests))
                        {
                                lineProcessorAction(parsedP, detailed, passedAllTests);
                        }
                }
        }

        internal static string BuildFileName(
                bool bitInc,
                int threads,
                int block,
                GpuKernelType kernelType,
                bool useLucasFlag,
                bool useDivisorFlag,
                bool useByDivisorFlag,
                bool mersenneOnGpu,
                bool useOrder,
                bool useGcd,
                NttBackend nttBackend,
                int gpuPrimeThreads,
                int llSlice,
                int gpuScanBatch,
                ulong warmupLimit,
                ModReductionMode reduction,
                string mersenneDevice,
                string primesDevice,
                string orderDevice)
        {
                string inc = bitInc ? "bit" : "add";
                string mers = useDivisorFlag
                        ? "divisor"
                        : (useLucasFlag
                                ? "lucas"
                                : (useByDivisorFlag
                                        ? "bydivisor"
                                        : (kernelType == GpuKernelType.Pow2Mod ? "pow2mod" : "incremental")));
                string ntt = nttBackend == NttBackend.Staged ? "staged" : "reference";
                string red = reduction switch { ModReductionMode.Mont64 => "mont64", ModReductionMode.Barrett128 => "barrett128", ModReductionMode.GpuUInt128 => "uint128", _ => "auto" };
                string order = useOrder ? "order-on" : "order-off";
                string gcd = useGcd ? "gcd-on" : "gcd-off";
                Span<char> initialBuffer = stackalloc char[256];
                var builder = new PooledValueStringBuilder(initialBuffer);
                try
                {
                        builder.Append("even_perfect_bit_scan_inc-");
                        builder.Append(inc);
                        builder.Append("_thr-");
                        builder.Append(threads);
                        builder.Append("_blk-");
                        builder.Append(block);
                        builder.Append("_mers-");
                        builder.Append(mers);
                        builder.Append("_mersdev-");
                        builder.Append(mersenneDevice);
                        builder.Append("_ntt-");
                        builder.Append(ntt);
                        builder.Append("_red-");
                        builder.Append(red);
                        builder.Append("_primesdev-");
                        builder.Append(primesDevice);
                        builder.Append('_');
                        builder.Append(order);
                        builder.Append("_orderdev-");
                        builder.Append(orderDevice);
                        builder.Append('_');
                        builder.Append(gcd);
                        builder.Append("_gputh-");
                        builder.Append(gpuPrimeThreads);
                        builder.Append("_llslice-");
                        builder.Append(llSlice);
                        builder.Append("_scanb-");
                        builder.Append(gpuScanBatch);
                        builder.Append("_warm-");
                        builder.Append(warmupLimit);
                        builder.Append(".csv");
                        return builder.ToString();
                }
                finally
                {
                        builder.Dispose();
                }
        }
}
