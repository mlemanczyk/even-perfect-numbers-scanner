using System.Globalization;
using System.Text;
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

        private const string DefaultHeader = "p,searchedMersenne,detailedCheck,isPerfect";

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

                if (args.Length != 1)
                {
                        Console.Error.WriteLine("Only a single CSV file path is supported.");
                        PrintHelp();
                        return 1;
                }

                string inputPath = args[0];

                if (!File.Exists(inputPath))
                {
                        Console.Error.WriteLine($"Input file '{inputPath}' was not found.");
                        return 1;
                }

                try
                {
                        ProcessFile(inputPath);
                        return 0;
                }
                catch (Exception exception)
                {
                        Console.Error.WriteLine($"Processing failed: {exception.Message}");
                        return 1;
                }
        }

        private static bool IsHelpOption(string value)
        {
                return HelpOptions.Contains(value);
        }

        private static void PrintHelp()
        {
                Console.WriteLine("EvenPerfectBitScanner.ResultsParser");
                Console.WriteLine("Parses CSV results produced by the candidate scanner.");
                Console.WriteLine();
                Console.WriteLine("Usage:");
                Console.WriteLine("  EvenPerfectBitScanner.ResultsParser <results.csv>");
                Console.WriteLine();
                Console.WriteLine("The program accepts the following help switches:");
                Console.WriteLine("  --?, -?, /?, -help, --help, --h, -h, /h, /help");
                Console.WriteLine();
                Console.WriteLine("Outputs prime results into four files placed next to the source file:");
                Console.WriteLine("  raw-primes-<name>              - prime entries in the original order");
                Console.WriteLine("  sorted-primes-<name>           - prime entries sorted by p");
                Console.WriteLine("  sorted-primes-passed-<name>    - sorted prime entries with isPerfect = true");
                Console.WriteLine("  sorted-primes-rejected-<name>  - sorted prime entries with isPerfect = false");
                Console.WriteLine();
                Console.WriteLine("Example:");
                Console.WriteLine("  EvenPerfectBitScanner.ResultsParser results.csv");
        }

        private static void ProcessFile(string inputPath)
        {
                using IEnumerator<string> enumerator = File.ReadLines(inputPath).GetEnumerator();
                if (!enumerator.MoveNext())
                {
                        Console.WriteLine("Input file did not contain any data. Created empty outputs using the default header.");
                        string fallbackHeader = DefaultHeader;
                        string rawFallbackPath = BuildOutputPath(inputPath, "raw-primes-");
                        WriteCsv(rawFallbackPath, fallbackHeader, Array.Empty<CandidateResult>());
                        CreateEmptyOutputs(inputPath, fallbackHeader);
                        return;
                }

                string headerLine = enumerator.Current;
                string header = string.IsNullOrWhiteSpace(headerLine) ? DefaultHeader : headerLine.Trim();

                using IEnumerator<ulong> primeEnumerator = Prime.Numbers.GetEnumerator();
                if (!primeEnumerator.MoveNext())
                {
                        throw new InvalidOperationException("Prime generator did not provide any values.");
                }

                ulong currentPrime = primeEnumerator.Current;
                List<CandidateResult> pendingResults = new();
                List<CandidateResult> primeResults = new();
                while (enumerator.MoveNext())
                {
                        string currentLine = enumerator.Current;
                        if (string.IsNullOrWhiteSpace(currentLine))
                        {
                                continue;
                        }

                        CandidateResult result = ParseLine(currentLine);
                        pendingResults.Add(result);
                        DrainPrimeMatches(pendingResults, primeResults, ref currentPrime, primeEnumerator, false);
                }

                DrainPrimeMatches(pendingResults, primeResults, ref currentPrime, primeEnumerator, true);

                string rawOutputPath = BuildOutputPath(inputPath, "raw-primes-");
                WriteCsv(rawOutputPath, header, primeResults);

                if (primeResults.Count == 0)
                {
                        Console.WriteLine("No prime entries were found in the input file.");
                        Console.WriteLine($"Created an empty prime snapshot at: {rawOutputPath}");
                        CreateEmptyOutputs(inputPath, header);
                        return;
                }

                List<CandidateResult> sortedResults = new(primeResults);
                BubbleSort(sortedResults);

                List<CandidateResult> passedResults = new();
                List<CandidateResult> rejectedResults = new();

                for (int i = 0; i < sortedResults.Count; i++)
                {
                        CandidateResult candidate = sortedResults[i];
                        if (candidate.IsPerfect)
                        {
                                passedResults.Add(candidate);
                        }
                        else
                        {
                                rejectedResults.Add(candidate);
                        }
                }

                string sortedOutputPath = BuildOutputPath(inputPath, "sorted-primes-");
                string passedOutputPath = BuildOutputPath(inputPath, "sorted-primes-passed-");
                string rejectedOutputPath = BuildOutputPath(inputPath, "sorted-primes-rejected-");

                WriteCsv(sortedOutputPath, header, sortedResults);
                WriteCsv(passedOutputPath, header, passedResults);
                WriteCsv(rejectedOutputPath, header, rejectedResults);

                Console.WriteLine($"Processed {primeResults.Count} prime entr" +
                        $"{(primeResults.Count == 1 ? "y" : "ies")}.");
                Console.WriteLine($"Prime results (raw order): {rawOutputPath}");
                Console.WriteLine($"Prime results (sorted): {sortedOutputPath}");
                Console.WriteLine($"Prime results (passed): {passedOutputPath}");
                Console.WriteLine($"Prime results (rejected): {rejectedOutputPath}");
        }

        private static void CreateEmptyOutputs(string inputPath, string header)
        {
                string sortedOutputPath = BuildOutputPath(inputPath, "sorted-primes-");
                string passedOutputPath = BuildOutputPath(inputPath, "sorted-primes-passed-");
                string rejectedOutputPath = BuildOutputPath(inputPath, "sorted-primes-rejected-");

                WriteCsv(sortedOutputPath, header, Array.Empty<CandidateResult>());
                WriteCsv(passedOutputPath, header, Array.Empty<CandidateResult>());
                WriteCsv(rejectedOutputPath, header, Array.Empty<CandidateResult>());
        }

        private static CandidateResult ParseLine(string line)
        {
                string trimmedLine = line.Trim();
                ReadOnlySpan<char> span = trimmedLine.AsSpan();
                int firstCommaIndex = span.IndexOf(',');
                int lastCommaIndex = span.LastIndexOf(',');
                if (firstCommaIndex < 0 || lastCommaIndex <= firstCommaIndex)
                {
                        throw new FormatException("Input line does not contain expected comma separators.");
                }

                ReadOnlySpan<char> pSpan = span[..firstCommaIndex];
                ReadOnlySpan<char> isPerfectSpan = span[(lastCommaIndex + 1)..];
                ulong p = ulong.Parse(pSpan, NumberStyles.Integer, CultureInfo.InvariantCulture);
                bool isPerfect = bool.Parse(isPerfectSpan);
                return new CandidateResult(p, isPerfect, trimmedLine);
        }

        private static string BuildOutputPath(string inputPath, string prefix)
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

        private static void WriteCsv(string path, string header, IEnumerable<CandidateResult> entries)
        {
                using StreamWriter writer = new(path, false, Encoding.UTF8);
                writer.WriteLine(header);
                foreach (CandidateResult entry in entries)
                {
                        writer.WriteLine(entry.ToCsv());
                }
        }

        private static void BubbleSort(List<CandidateResult> results)
        {
                if (results.Count < 2)
                {
                        return;
                }

                int length = results.Count;
                bool swapped;
                do
                {
                        swapped = false;
                        for (int i = 1; i < length; i++)
                        {
                                if (results[i - 1].P <= results[i].P)
                                {
                                        continue;
                                }

                                (results[i - 1], results[i]) = (results[i], results[i - 1]);
                                swapped = true;
                        }

                        length--;
                }
                while (swapped);
        }

        private static void DrainPrimeMatches(
                List<CandidateResult> pendingResults,
                List<CandidateResult> primeResults,
                ref ulong currentPrime,
                IEnumerator<ulong> primeEnumerator,
                bool inputCompleted)
        {
                if (pendingResults.Count == 0)
                {
                        return;
                }

                RemoveEntriesBelowPrime(pendingResults, currentPrime);

                while (pendingResults.Count > 0)
                {
                        bool extractedPrime = ExtractPrimeEntries(pendingResults, primeResults, currentPrime);
                        if (extractedPrime)
                        {
                                if (!TryAdvancePrime(primeEnumerator, ref currentPrime))
                                {
                                        pendingResults.Clear();
                                        return;
                                }

                                RemoveEntriesBelowPrime(pendingResults, currentPrime);
                                continue;
                        }

                        if (!inputCompleted)
                        {
                                return;
                        }

                        if (!TryAdvancePrime(primeEnumerator, ref currentPrime))
                        {
                                pendingResults.Clear();
                                return;
                        }

                        RemoveEntriesBelowPrime(pendingResults, currentPrime);
                }
        }

        private static bool ExtractPrimeEntries(
                List<CandidateResult> pendingResults,
                List<CandidateResult> primeResults,
                ulong prime)
        {
                bool found = false;
                for (int i = 0; i < pendingResults.Count; i++)
                {
                        CandidateResult candidate = pendingResults[i];
                        if (candidate.P != prime)
                        {
                                continue;
                        }

                        primeResults.Add(candidate);
                        pendingResults.RemoveAt(i);
                        i--;
                        found = true;
                }

                return found;
        }

        private static void RemoveEntriesBelowPrime(List<CandidateResult> pendingResults, ulong currentPrime)
        {
                for (int i = 0; i < pendingResults.Count; i++)
                {
                        if (pendingResults[i].P >= currentPrime)
                        {
                                continue;
                        }

                        pendingResults.RemoveAt(i);
                        i--;
                }
        }

        private static bool TryAdvancePrime(IEnumerator<ulong> primeEnumerator, ref ulong currentPrime)
        {
                if (!primeEnumerator.MoveNext())
                {
                        currentPrime = ulong.MaxValue;
                        return false;
                }

                currentPrime = primeEnumerator.Current;
                return true;
        }

        private readonly record struct CandidateResult(ulong P, bool IsPerfect, string Csv)
        {
                public string ToCsv()
                {
                        return Csv;
                }
        }
}

