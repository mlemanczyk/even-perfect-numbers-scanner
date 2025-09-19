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

		if (!TryParseRangeArguments(args.AsSpan(1), out ulong pMin, out ulong pMax))
		{
			return 1;
		}

		if (pMin > pMax)
		{
			Console.Error.WriteLine("The --p-min value cannot be greater than --p-max.");
			return 1;
		}

		try
		{
			ProcessFile(inputPath, pMin, pMax);
			return 0;
		}
		catch (Exception exception)
		{
			Console.Error.WriteLine($"Processing failed: {exception.Message}");
			return 1;
		}
	}

	private static bool IsHelpOption(string value) => HelpOptions.Contains(value);

	private static bool TryParseRangeArguments(ReadOnlySpan<string> arguments, out ulong pMin, out ulong pMax)
	{
		pMin = 0UL;
		pMax = ulong.MaxValue;
		if (arguments.Length == 0)
		{
			return true;
		}

		if ((arguments.Length & 1) == 1)
		{
			Console.Error.WriteLine("Each range option must be followed by a numeric value.");
			PrintHelp();
			return false;
		}

		int index;
		string option, optionAndValue, value;
		for (int i = 0; i < arguments.Length; i += 2)
		{
			optionAndValue = arguments[i];
			index = optionAndValue.IndexOf('=');
			option = optionAndValue[..index];
			value = optionAndValue[(index + 1)..];

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

	private static ulong ParseRangeValue(string option, string value)
	{
		if (!ulong.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out ulong parsed))
		{
			throw new FormatException($"Value '{value}' for option '{option}' is not a valid non-negative integer.");
		}

		return parsed;
	}

	private static void PrintHelp()
	{
		Console.WriteLine("EvenPerfectBitScanner.ResultsParser");
		Console.WriteLine("Parses CSV results produced by the candidate scanner.");
		Console.WriteLine();
		Console.WriteLine("Usage:");
		Console.WriteLine("  EvenPerfectBitScanner.ResultsParser <results.csv> [--p-min <value>] [--p-max <value>]");
		Console.WriteLine();
		Console.WriteLine("The program accepts the following help switches:");
		Console.WriteLine("  --?, -?, /?, -help, --help, --h, -h, /h, /help");
		Console.WriteLine();
		Console.WriteLine("Optional range arguments allow filtering primes by exponent:");
		Console.WriteLine("  --p-min <value>  - include entries with p >= value");
		Console.WriteLine("  --p-max <value>  - include entries with p <= value");
		Console.WriteLine();
		Console.WriteLine("Outputs prime results into four files placed next to the source file:");
		Console.WriteLine("  raw-primes-<name>              - prime entries in the original order");
		Console.WriteLine("  sorted-primes-<name>           - prime entries sorted by p");
		Console.WriteLine("  sorted-primes-passed-<name>    - sorted prime entries with passedAllTests = true");
		Console.WriteLine("  sorted-primes-rejected-<name>  - sorted prime entries with passedAllTests = false");
		Console.WriteLine();
		Console.WriteLine("Example:");
		Console.WriteLine("  EvenPerfectBitScanner.ResultsParser results.csv --p-min 89 --p-max 107");
	}

	private static void ProcessFile(string inputPath, ulong pMin, ulong pMax)
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
		int consoleProgress = 0;
		while (enumerator.MoveNext())
		{
			string currentLine = enumerator.Current;
			if (string.IsNullOrWhiteSpace(currentLine))
			{
				continue;
			}

			CandidateResult result = ParseLine(currentLine);
			if (result.P < pMin || result.P > pMax)
			{
				continue;
			}

			InsertPendingResult(pendingResults, result);
			DrainPrimeMatches(pendingResults, primeResults, ref currentPrime, primeEnumerator, false);
			if (++consoleProgress == 10_000)
			{
				Console.WriteLine($"Processed {result.P}");
				consoleProgress = 0;
			}
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

		List<CandidateResult> sortedResults = [.. primeResults];
		Console.WriteLine("Sorting candidates...");
		BubbleSort(sortedResults);

		Console.WriteLine("Splitting results...");
		List<CandidateResult> passedResults = new(sortedResults.Count);
		List<CandidateResult> rejectedResults = new(sortedResults.Count);

		for (int i = 0; i < sortedResults.Count; i++)
		{
			CandidateResult candidate = sortedResults[i];
			if (candidate.PassedAllTests)
			{
				passedResults.Add(candidate);
			}
			else
			{
				rejectedResults.Add(candidate);
			}
		}

		Console.WriteLine("Saving files...");
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

		WriteCsv(sortedOutputPath, header, []);
		WriteCsv(passedOutputPath, header, []);
		WriteCsv(rejectedOutputPath, header, []);
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
		ReadOnlySpan<char> passedAllTestsSpan = span[(lastCommaIndex + 1)..];
		ulong p = ulong.Parse(pSpan, NumberStyles.Integer, CultureInfo.InvariantCulture);
		bool passedAllTests = bool.Parse(passedAllTestsSpan);
		return new CandidateResult(p, passedAllTests, trimmedLine);
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

		bool extractedPrime;
		while (pendingResults.Count > 0)
		{
			extractedPrime = ExtractPrimeEntries(pendingResults, primeResults, currentPrime);
			if (extractedPrime)
			{
				if (!TryAdvancePrime(primeEnumerator, ref currentPrime, pendingResults))
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

			if (!TryAdvancePrime(primeEnumerator, ref currentPrime, pendingResults))
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
		if (pendingResults.Count == 0)
		{
			return false;
		}

		int startIndex = FindFirstIndexAtLeast(pendingResults, prime);
		if (startIndex >= pendingResults.Count)
		{
			return false;
		}

		CandidateResult firstCandidate = pendingResults[startIndex];
		if (firstCandidate.P != prime)
		{
			return false;
		}

		int endIndex = startIndex + 1;
		while (endIndex < pendingResults.Count && pendingResults[endIndex].P == prime)
		{
			endIndex++;
		}

		for (int i = startIndex; i < endIndex; i++)
		{
			primeResults.Add(pendingResults[i]);
		}

		pendingResults.RemoveRange(startIndex, endIndex - startIndex);
		return true;
	}

	private static void RemoveEntriesBelowPrime(List<CandidateResult> pendingResults, ulong currentPrime)
	{
		if (pendingResults.Count == 0)
		{
			return;
		}

		int removeCount = FindFirstIndexAtLeast(pendingResults, currentPrime);
		if (removeCount == 0)
		{
			return;
		}

		pendingResults.RemoveRange(0, removeCount);
	}

	private static bool TryAdvancePrime(
			IEnumerator<ulong> primeEnumerator,
			ref ulong currentPrime,
			List<CandidateResult> pendingResults)
	{
		ulong minimumPending = pendingResults.Count > 0 ? pendingResults[0].P : 0UL;
		do
		{
			if (!primeEnumerator.MoveNext())
			{
				currentPrime = ulong.MaxValue;
				return false;
			}

			currentPrime = primeEnumerator.Current;
		}
		while (pendingResults.Count > 0 && currentPrime < minimumPending);

		return true;
	}

	private static void InsertPendingResult(List<CandidateResult> pendingResults, CandidateResult result) => pendingResults.Insert(FindFirstIndexAtLeast(pendingResults, result.P), result);

	private static int FindFirstIndexAtLeast(List<CandidateResult> pendingResults, ulong value)
	{
		int low = 0,
			high = pendingResults.Count - 1,
			mid;

		while (low <= high)
		{
			mid = low + ((high - low) >> 1);
			if (pendingResults[mid].P < value)
			{
				low = mid + 1;
				continue;
			}

			high = mid - 1;
		}

		return low;
	}

	private readonly record struct CandidateResult(ulong P, bool PassedAllTests, string Csv)
	{
		public string ToCsv() => Csv;

		public readonly string Csv = Csv;
		public readonly ulong P = P;
		public readonly bool PassedAllTests = PassedAllTests;
	}
}

