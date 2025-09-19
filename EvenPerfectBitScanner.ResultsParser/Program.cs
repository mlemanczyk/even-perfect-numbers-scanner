using System.Globalization;
using System.Text;
using System.Threading.Tasks;
using Open.Numeric.Primes;

namespace EvenPerfectBitScanner.ResultsParser;

internal static class Program
{
	private static readonly HashSet<string> HelpOptions = new(StringComparer.OrdinalIgnoreCase)
		{
				"--?",
				"-?",
				"-help",
				"--help",
				"--h",
				"-h",
				"/?",
				"/h",
				"/help"
		};

	private const string DefaultHeader = "p,searchedMersenne,detailedCheck,passedAllTests";

        private static int Main(string[] args)
        {
                if (args.Length == 0)
                {
                        PrintHelp();
			return 1;
		}

		if (args.Length == 1 && IsHelpOption(args[0]))
		{
			PrintHelp();
			return 0;
		}

		string inputPath = args[0];
		if (IsHelpOption(inputPath))
		{
			PrintHelp();
			return 0;
		}

		if (!File.Exists(inputPath))
		{
			Console.Error.WriteLine($"Input file '{inputPath}' was not found.");
			return 1;
		}

                if (!TryParseProcessingArguments(
                                args.AsSpan(1),
                                out ulong pMin,
                                out ulong pMax,
                                out int threadCount,
                                out int bufferSize))
                {
                        return 1;
                }

                if (pMin > pMax)
                {
                        Console.Error.WriteLine("The --p-min value cannot be greater than --p-max.");
                        return 1;
                }

                if (threadCount < 1)
                {
                        Console.Error.WriteLine("The --threads value must be a positive integer.");
                        return 1;
                }

                if (bufferSize < 1)
                {
                        Console.Error.WriteLine("The --buffer-size value must be a positive integer.");
                        return 1;
                }

                try
                {
                        ProcessFile(inputPath, pMin, pMax, threadCount, bufferSize);
                        return 0;
                }
                catch (Exception exception)
                {
                        Console.Error.WriteLine($"Processing failed: {exception.Message}");
			return 1;
		}
	}

	private static bool IsHelpOption(string value) => HelpOptions.Contains(value);

        private static bool TryParseProcessingArguments(
                        ReadOnlySpan<string> arguments,
                        out ulong pMin,
                        out ulong pMax,
                        out int threadCount,
                        out int bufferSize)
        {
                pMin = 0UL;
                pMax = ulong.MaxValue;
                threadCount = Environment.ProcessorCount;
                bufferSize = 4096;

                if (arguments.Length == 0)
                {
                        return true;
                }

                int index = 0;
                while (index < arguments.Length)
                {
                        string argument = arguments[index];
                        if (!TryReadOptionValue(arguments, ref index, out string option, out string value))
                        {
                                return false;
                        }

                        try
                        {
                                switch (option)
                                {
                                        case "--p-min":
                                                pMin = ParseRangeValue(option, value);
                                                break;
                                        case "--p-max":
                                                pMax = ParseRangeValue(option, value);
                                                break;
                                        case "--threads":
                                                threadCount = ParsePositiveInt(option, value);
                                                break;
                                        case "--buffer-size":
                                                bufferSize = ParsePositiveInt(option, value);
                                                break;
                                        default:
                                                Console.Error.WriteLine($"Unsupported option '{option}'.");
                                                PrintHelp();
                                                return false;
                                }
                        }
                        catch (FormatException exception)
                        {
                                Console.Error.WriteLine(exception.Message);
                                PrintHelp();
                                return false;
                        }
                }

                return true;
        }

        private static ulong ParseRangeValue(in string option, in string value)
        {
                if (!ulong.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out ulong parsed))
                {
                        throw new FormatException($"Value '{value}' for option '{option}' is not a valid non-negative integer.");
                }

                return parsed;
        }

        private static int ParsePositiveInt(in string option, in string value)
        {
                if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed) || parsed < 1)
                {
                        throw new FormatException($"Value '{value}' for option '{option}' is not a valid positive integer.");
                }

                return parsed;
        }

        private static bool TryReadOptionValue(
                        ReadOnlySpan<string> arguments,
                        ref int index,
                        out string option,
                        out string value)
        {
                option = string.Empty;
                value = string.Empty;
                string current = arguments[index];
                int equalsIndex = current.IndexOf('=');
                if (equalsIndex <= 0 || equalsIndex == current.Length - 1)
                {
                        Console.Error.WriteLine($"Option '{current}' must use the --name=value syntax.");
                        PrintHelp();
                        return false;
                }

                option = current[..equalsIndex];
                value = current[(equalsIndex + 1)..];
                index++;
                return true;
        }

	private static void PrintHelp()
	{
		Console.WriteLine("EvenPerfectBitScanner.ResultsParser");
		Console.WriteLine("Parses CSV results produced by the candidate scanner.");
		Console.WriteLine();
		Console.WriteLine("Usage:");
                Console.WriteLine(
                        "  EvenPerfectBitScanner.ResultsParser <results.csv> [--p-min=<value>] [--p-max=<value>] [--threads=<value>] [--buffer-size=<value>]");
                Console.WriteLine();
                Console.WriteLine("The program accepts the following help switches:");
                Console.WriteLine("  --?, -?, /?, -help, --help, --h, -h, /h, /help");
                Console.WriteLine();
                Console.WriteLine("Optional range arguments allow filtering primes by exponent:");
                Console.WriteLine("  --p-min=<value>  - include entries with p >= value");
                Console.WriteLine("  --p-max=<value>  - include entries with p <= value");
                Console.WriteLine();
                Console.WriteLine("Parallel processing options:");
                Console.WriteLine("  --threads=<value>      - number of worker threads used for parsing and splitting (default: logical cores)");
                Console.WriteLine("  --buffer-size=<value>  - number of records read into each parsing batch (default: 4096)");
                Console.WriteLine("Values must be provided using the --option=value syntax.");
                Console.WriteLine();
                Console.WriteLine("Outputs prime results into four files placed next to the source file:");
		Console.WriteLine("  raw-primes-<name>              - prime entries in the original order");
		Console.WriteLine("  sorted-primes-<name>           - prime entries sorted by p");
		Console.WriteLine("  sorted-primes-passed-<name>    - sorted prime entries with passedAllTests = true");
		Console.WriteLine("  sorted-primes-rejected-<name>  - sorted prime entries with passedAllTests = false");
		Console.WriteLine();
		Console.WriteLine("Example:");
                Console.WriteLine("  EvenPerfectBitScanner.ResultsParser results.csv --p-min=89 --p-max=107");
	}

        private static void ProcessFile(in string inputPath, ulong pMin, ulong pMax, int threadCount, int bufferSize)
        {
                using StreamReader reader = new(inputPath);
                string? headerLine = reader.ReadLine();
                if (headerLine is null)
                {
                        Console.WriteLine("Input file did not contain any data. Created empty outputs using the default header.");
                        string fallbackHeader = DefaultHeader;
                        string rawFallbackPath = BuildOutputPath(inputPath, "raw-primes-");
                        WriteCsv(rawFallbackPath, fallbackHeader, File.Exists(rawFallbackPath), []);
                        CreateEmptyOutputs(inputPath, fallbackHeader);
                        return;
                }

                string header = string.IsNullOrWhiteSpace(headerLine) ? DefaultHeader : headerLine.Trim();

                List<CandidateResult> candidates = LoadCandidates(reader, pMin, pMax, threadCount, bufferSize);

                List<CandidateResult> sortedPrimeResults = ExtractSortedPrimeCandidates(candidates);
                List<CandidateResult> primeResults = BuildRawPrimeResults(candidates, sortedPrimeResults);

                string rawOutputPath = BuildOutputPath(inputPath, "raw-primes-");
                bool append = File.Exists(rawOutputPath);
                WriteCsv(rawOutputPath, header, append, primeResults);

                if (sortedPrimeResults.Count == 0)
                {
                        Console.WriteLine("No prime entries were found in the input file.");
                        Console.WriteLine($"Created an empty prime snapshot at: {rawOutputPath}");
                        CreateEmptyOutputs(inputPath, header);
                        return;
                }

                Console.WriteLine("Sorting candidates...");
                SortByPrime(sortedPrimeResults);

                Console.WriteLine("Splitting results...");
                SplitResults(sortedPrimeResults, threadCount, out List<CandidateResult> passedResults, out List<CandidateResult> rejectedResults);

                Console.WriteLine("Saving files...");
                string sortedOutputPath = BuildOutputPath(inputPath, "sorted-primes-");
                string passedOutputPath = BuildOutputPath(inputPath, "sorted-primes-passed-");
                string rejectedOutputPath = BuildOutputPath(inputPath, "sorted-primes-rejected-");

                append = File.Exists(sortedOutputPath);
                WriteCsv(sortedOutputPath, header, append, sortedPrimeResults);
                append = File.Exists(passedOutputPath);
                WriteCsv(passedOutputPath, header, append, passedResults);
                append = File.Exists(rejectedOutputPath);
                WriteCsv(rejectedOutputPath, header, append, rejectedResults);

                Console.WriteLine($"Processed {sortedPrimeResults.Count} prime entr{(sortedPrimeResults.Count == 1 ? "y" : "ies")}.");
                Console.WriteLine($"Prime results (raw order): {rawOutputPath}");
                Console.WriteLine($"Prime results (sorted): {sortedOutputPath}");
                Console.WriteLine($"Prime results (passed): {passedOutputPath}");
                Console.WriteLine($"Prime results (rejected): {rejectedOutputPath}");
        }

	private static void CreateEmptyOutputs(in string inputPath, in string header)
	{
		string sortedOutputPath = BuildOutputPath(inputPath, "sorted-primes-");
		string passedOutputPath = BuildOutputPath(inputPath, "sorted-primes-passed-");
		string rejectedOutputPath = BuildOutputPath(inputPath, "sorted-primes-rejected-");

		WriteCsv(sortedOutputPath, header, false, []);
		WriteCsv(passedOutputPath, header, false, []);
		WriteCsv(rejectedOutputPath, header, false, []);
	}

	private static CandidateResult ParseLine(in string line, ulong pMin, ulong pMax)
	{
		ReadOnlySpan<char> span = line.AsSpan();
		int firstCommaIndex = span.IndexOf(',');
		int lastCommaIndex = span.LastIndexOf(',');
		if (firstCommaIndex < 0 || lastCommaIndex <= firstCommaIndex)
		{
			throw new FormatException("Input line does not contain expected comma separators.");
		}

		ulong p = ulong.Parse(span[..firstCommaIndex], NumberStyles.Integer, CultureInfo.InvariantCulture);
		if (p < pMin || p > pMax)
		{
			return CandidateResult.Empty;
		}

		bool passedAllTests = bool.Parse(span[(lastCommaIndex + 1)..]);
		return new CandidateResult(p, passedAllTests, line);
	}

	private static string BuildOutputPath(in string inputPath, in string prefix)
	{
		string fileName = Path.GetFileName(inputPath);
		string prefixedName = prefix + fileName;
		string? directory = Path.GetDirectoryName(inputPath);
		if (string.IsNullOrEmpty(directory))
		{
			return prefixedName;
		}

		return Path.Combine(directory, prefixedName);
	}

        private static void WriteCsv(in string path, in string header, bool append, IEnumerable<CandidateResult> entries)
        {
                using StreamWriter writer = new(path, append, Encoding.UTF8);
                if (!append)
                {
                        writer.WriteLine(header);
                }

                foreach (CandidateResult entry in entries)
                {
                        writer.WriteLine(entry.Csv);
                }
        }

        private static List<CandidateResult> LoadCandidates(
                        StreamReader reader,
                        ulong pMin,
                        ulong pMax,
                        int threadCount,
                        int bufferSize)
        {
                List<CandidateResult> candidates = [];
                if (bufferSize < 1)
                {
                        return candidates;
                }

                int workerLimit = Math.Max(1, threadCount);
                List<Task<ChunkResult>> activeChunks = new(workerLimit);
                Dictionary<int, ChunkResult> completedChunks = new();
                int nextChunkToFlush = 0;
                int consoleProgress = 0;
                ulong lastReportedCandidate = 0UL;

                string[] buffer = new string[bufferSize];
                int buffered = 0;
                int chunkIndex = 0;

                void FlushChunk(in ChunkResult chunk)
                {
                        if (chunk.Count == 0)
                        {
                                return;
                        }

                        CandidateResult[] items = chunk.Items;
                        int count = chunk.Count;
                        candidates.EnsureCapacity(candidates.Count + count);
                        for (int i = 0; i < count; i++)
                        {
                                CandidateResult candidate = items[i];
                                candidates.Add(candidate);
                                lastReportedCandidate = candidate.P;
                                if (++consoleProgress == 10_000)
                                {
                                        Console.WriteLine($"Processed {lastReportedCandidate}");
                                        consoleProgress = 0;
                                }
                        }
                }

                void TryFlush()
                {
                        while (completedChunks.Remove(nextChunkToFlush, out ChunkResult ready))
                        {
                                FlushChunk(in ready);
                                nextChunkToFlush++;
                        }
                }

                void WaitForAny()
                {
                        Task<ChunkResult> completed = Task.WhenAny(activeChunks).GetAwaiter().GetResult();
                        activeChunks.Remove(completed);
                        ChunkResult chunk = completed.GetAwaiter().GetResult();
                        completedChunks.Add(chunk.Index, chunk);
                        TryFlush();
                }

                void ScheduleChunk(string[] chunkLines)
                {
                        int currentIndex = chunkIndex++;
                        Task<ChunkResult> task = Task.Run(() => ProcessChunk(chunkLines, currentIndex, pMin, pMax));
                        activeChunks.Add(task);
                        if (activeChunks.Count == workerLimit)
                        {
                                WaitForAny();
                        }
                }

                void DispatchBuffer(int length)
                {
                        string[] chunkLines;
                        if (length == buffer.Length)
                        {
                                chunkLines = buffer;
                                buffer = new string[bufferSize];
                        }
                        else
                        {
                                chunkLines = new string[length];
                                Array.Copy(buffer, chunkLines, length);
                        }

                        buffered = 0;
                        ScheduleChunk(chunkLines);
                }

                string? line;
                while ((line = reader.ReadLine()) is not null)
                {
                        if (string.IsNullOrWhiteSpace(line))
                        {
                                continue;
                        }

                        buffer[buffered++] = line;
                        if (buffered == bufferSize)
                        {
                                DispatchBuffer(buffered);
                        }
                }

                if (buffered > 0)
                {
                        DispatchBuffer(buffered);
                }

                while (activeChunks.Count > 0)
                {
                        WaitForAny();
                }

                TryFlush();
                return candidates;
        }

        private static ChunkResult ProcessChunk(string[] lines, int index, ulong pMin, ulong pMax)
        {
                CandidateResult[] items = new CandidateResult[lines.Length];
                int count = 0;
                ulong pEmpty = CandidateResult.Empty.P;
                for (int i = 0; i < lines.Length; i++)
                {
                        string line = lines[i];
                        if (string.IsNullOrWhiteSpace(line))
                        {
                                continue;
                        }

                        CandidateResult candidate = ParseLine(line, pMin, pMax);
                        if (candidate.P == pEmpty)
                        {
                                continue;
                        }

                        items[count++] = candidate;
                }

                return new ChunkResult(index, items, count);
        }

        private static List<CandidateResult> ExtractSortedPrimeCandidates(List<CandidateResult> candidates)
        {
                if (candidates.Count == 0)
                {
                        return [];
                }

                List<CandidateResult> sortedCandidates = [.. candidates];
                sortedCandidates.Sort(CandidateResultComparer.Instance);

                List<CandidateResult> primeCandidates = [];
                using IEnumerator<ulong> primeEnumerator = Prime.Numbers.GetEnumerator();
                if (!primeEnumerator.MoveNext())
                {
                        throw new InvalidOperationException("Prime generator did not provide any values.");
                }

                ulong currentPrime = primeEnumerator.Current;
                foreach (CandidateResult candidate in sortedCandidates)
                {
                        ulong candidateP = candidate.P;
                        while (currentPrime < candidateP)
                        {
                                if (!primeEnumerator.MoveNext())
                                {
                                        return primeCandidates;
                                }

                                currentPrime = primeEnumerator.Current;
                        }

                        if (currentPrime == candidateP)
                        {
                                primeCandidates.Add(candidate);
                        }
                }

                return primeCandidates;
        }

        private static List<CandidateResult> BuildRawPrimeResults(
                        List<CandidateResult> candidates,
                        List<CandidateResult> sortedPrimeResults)
        {
                int primeCount = sortedPrimeResults.Count;
                if (primeCount == 0)
                {
                        return [];
                }

                Dictionary<ulong, int> remainingCounts = new(primeCount);
                for (int i = 0; i < primeCount; i++)
                {
                        CandidateResult candidate = sortedPrimeResults[i];
                        if (remainingCounts.TryGetValue(candidate.P, out int count))
                        {
                                remainingCounts[candidate.P] = count + 1;
                                continue;
                        }

                        remainingCounts.Add(candidate.P, 1);
                }

                List<CandidateResult> rawPrimes = new(primeCount);
                for (int i = 0; i < candidates.Count; i++)
                {
                        CandidateResult candidate = candidates[i];
                        if (!remainingCounts.TryGetValue(candidate.P, out int count))
                        {
                                continue;
                        }

                        rawPrimes.Add(candidate);
                        if (count == 1)
                        {
                                remainingCounts.Remove(candidate.P);
                                continue;
                        }

                        remainingCounts[candidate.P] = count - 1;
                }

                return rawPrimes;
        }

        private static void SortByPrime(List<CandidateResult> results)
        {
                if (results.Count < 2)
                {
                        return;
                }

                results.Sort(CandidateResultComparer.Instance);
        }

        private static void SplitResults(
                        List<CandidateResult> sortedResults,
                        int threadCount,
                        out List<CandidateResult> passedResults,
                        out List<CandidateResult> rejectedResults)
        {
                int count = sortedResults.Count;
                if (count == 0)
                {
                        passedResults = [];
                        rejectedResults = [];
                        return;
                }

                if (threadCount <= 1 || count < threadCount * 4)
                {
                        passedResults = new List<CandidateResult>(count);
                        rejectedResults = new List<CandidateResult>(count);
                        for (int i = 0; i < count; i++)
                        {
                                CandidateResult candidate = sortedResults[i];
                                if (candidate.PassedAllTests)
                                {
                                        passedResults.Add(candidate);
                                        continue;
                                }

                                rejectedResults.Add(candidate);
                        }

                        return;
                }

                int workerCount = Math.Min(threadCount, count);
                List<CandidateResult>[] passedSegments = new List<CandidateResult>[workerCount];
                List<CandidateResult>[] rejectedSegments = new List<CandidateResult>[workerCount];
                Task[] tasks = new Task[workerCount];

                int start = 0;
                int baseChunk = count / workerCount;
                int remainder = count % workerCount;
                for (int worker = 0; worker < workerCount; worker++)
                {
                        int length = baseChunk + (worker < remainder ? 1 : 0);
                        int chunkStart = start;
                        start += length;

                        if (length == 0)
                        {
                                passedSegments[worker] = [];
                                rejectedSegments[worker] = [];
                                tasks[worker] = Task.CompletedTask;
                                continue;
                        }

                        tasks[worker] = Task.Run(() =>
                        {
                                List<CandidateResult> localPassed = new(length);
                                List<CandidateResult> localRejected = new(length);
                                int end = chunkStart + length;
                                for (int i = chunkStart; i < end; i++)
                                {
                                        CandidateResult candidate = sortedResults[i];
                                        if (candidate.PassedAllTests)
                                        {
                                                localPassed.Add(candidate);
                                                continue;
                                        }

                                        localRejected.Add(candidate);
                                }

                                passedSegments[worker] = localPassed;
                                rejectedSegments[worker] = localRejected;
                        });
                }

                Task.WaitAll(tasks);

                passedResults = new List<CandidateResult>(count);
                rejectedResults = new List<CandidateResult>(count);
                for (int worker = 0; worker < workerCount; worker++)
                {
                        if (passedSegments[worker].Count > 0)
                        {
                                passedResults.AddRange(passedSegments[worker]);
                        }

                        if (rejectedSegments[worker].Count > 0)
                        {
                                rejectedResults.AddRange(rejectedSegments[worker]);
                        }
                }
        }

        private sealed class CandidateResultComparer : IComparer<CandidateResult>
        {
                public static readonly CandidateResultComparer Instance = new();

                public int Compare(CandidateResult x, CandidateResult y)
                {
                        if (x.P == y.P)
                        {
                                return 0;
                        }

                        return x.P < y.P ? -1 : 1;
                }
        }

        private readonly struct ChunkResult
        {
                public readonly int Index;
                public readonly CandidateResult[] Items;
                public readonly int Count;

                public ChunkResult(int index, CandidateResult[] items, int count)
                {
                        Index = index;
                        Items = items;
                        Count = count;
                }
        }

        private readonly record struct CandidateResult(ulong P, bool PassedAllTests, string Csv)
        {
                public static readonly CandidateResult Empty = new(0UL, false, "<empty>");

                public readonly string Csv = Csv;
		public readonly ulong P = P;
		public readonly bool PassedAllTests = PassedAllTests;
	}
}

