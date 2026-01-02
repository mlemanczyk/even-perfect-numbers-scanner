using System.Globalization;
using System.Numerics;
using System.Text;

namespace BuildDivisorTransitions;

internal static class Program
{
    private const string OutputFileName = "NextKForecastTraning.csv";

    private static int Main(string[] args)
    {
        if (args.Length != 2)
        {
            PrintUsage();
            return 1;
        }

        string candidatesPath = args[0];
        string divisorsPath = args[1];

        if (!File.Exists(candidatesPath))
        {
            Console.Error.WriteLine($"Candidates file not found: {candidatesPath}");
            return 1;
        }

        if (!File.Exists(divisorsPath))
        {
            Console.Error.WriteLine($"Divisors file not found: {divisorsPath}");
            return 1;
        }

        List<long> candidatePs = LoadCandidates(candidatesPath);
        Dictionary<long, BigInteger> divisors = LoadDivisors(divisorsPath);
        List<Transition> transitions = BuildTransitions(candidatePs, divisors);

        WriteOutput(transitions);

        Console.WriteLine($"Saved {transitions.Count} transitions to {OutputFileName}.");
        return 0;
    }

    private static void PrintUsage()
    {
        Console.WriteLine("Usage: BuildDivisorTransitions <candidatesCsv> <divisorsCsv>");
        Console.WriteLine("  candidatesCsv: CSV with columns p,searchedMersenne,detailedCheck,passedAllTests");
        Console.WriteLine("  divisorsCsv:   CSV with columns p,divisor");
    }

    private static List<long> LoadCandidates(string path)
    {
        var result = new List<long>();
        var seen = new HashSet<long>();

        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
        using var reader = new StreamReader(stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true);

        while (!reader.EndOfStream)
        {
            string? line = reader.ReadLine();
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            string trimmedLine = line.Trim();
            if (trimmedLine.StartsWith("p", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            string[] parts = trimmedLine.Split(',', StringSplitOptions.TrimEntries);
            if (parts.Length == 0)
            {
                continue;
            }

            if (!long.TryParse(parts[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out long p))
            {
                throw new FormatException($"Invalid p value '{parts[0]}' in {path}.");
            }

            if (seen.Add(p))
            {
                result.Add(p);
            }
        }

        return result;
    }

    private static Dictionary<long, BigInteger> LoadDivisors(string path)
    {
        var result = new Dictionary<long, BigInteger>();

        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
        using var reader = new StreamReader(stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true);

        while (!reader.EndOfStream)
        {
            string? line = reader.ReadLine();
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            string trimmedLine = line.Trim();
            if (trimmedLine.StartsWith("p", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            string[] parts = trimmedLine.Split(',', StringSplitOptions.TrimEntries);
            if (parts.Length < 2)
            {
                throw new FormatException($"Line '{line}' in {path} does not contain both p and divisor columns.");
            }

            if (!long.TryParse(parts[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out long p))
            {
                throw new FormatException($"Invalid p value '{parts[0]}' in {path}.");
            }

            if (!BigInteger.TryParse(parts[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out BigInteger divisor))
            {
                throw new FormatException($"Invalid divisor value '{parts[1]}' for p={p} in {path}.");
            }

            if (IsProperMersenneDivisor(p, divisor))
            {
                result[p] = divisor;
            }
            else
            {
                Console.Error.WriteLine($"Skipping invalid divisor entry p={p}, divisor={divisor} (not a proper divisor of 2^{p}-1 or fails powmod check).");
            }
        }

        return result;
    }

    private static List<Transition> BuildTransitions(IReadOnlyList<long> candidatePs, IReadOnlyDictionary<long, BigInteger> divisors)
    {
        var transitions = new List<Transition>();

        for (int i = 0; i < candidatePs.Count - 1; i++)
        {
            long currentP = candidatePs[i];
            long nextP = candidatePs[i + 1];

            if (!divisors.TryGetValue(currentP, out BigInteger currentDivisor))
            {
                continue;
            }

            if (!divisors.TryGetValue(nextP, out BigInteger nextDivisor))
            {
                continue;
            }

            BigInteger currentK = CalculateK(currentP, currentDivisor);
            BigInteger nextK = CalculateK(nextP, nextDivisor);
            bool isNextKHigher = nextK > currentK;
            transitions.Add(new Transition(currentP, currentDivisor, LastDigit(currentP), currentK, LastMersenneDigit(currentP), isNextKHigher));
        }

        return transitions;
    }

    private static BigInteger CalculateK(long p, BigInteger divisor)
    {
        BigInteger numerator = divisor - BigInteger.One;
        BigInteger denominator = new BigInteger(p) << 1;

        if (denominator <= BigInteger.Zero)
        {
            throw new InvalidOperationException($"Computed 2p is not positive for p={p}.");
        }

        BigInteger remainder = numerator % denominator;
        if (remainder != BigInteger.Zero)
        {
            throw new InvalidOperationException($"Divisor {divisor} is not congruent to 1 mod 2p for p={p}.");
        }

        BigInteger k = numerator / denominator;
        if (k < BigInteger.One)
        {
            throw new InvalidOperationException($"Calculated k={k} is less than 1 for p={p}, divisor={divisor}.");
        }

        return k;
    }

    private static bool IsProperMersenneDivisor(long p, in BigInteger divisor)
    {
        if (p <= 1 || divisor <= BigInteger.One)
        {
            return false;
        }

        int bitLength = GetBitLength(divisor);
        if (bitLength > p)
        {
            return false;
        }

        if (bitLength == p && IsPowerOfTwo(divisor + BigInteger.One))
        {
            // Divisor equals the Mersenne number itself.
            return false;
        }

        return BigInteger.ModPow(2, p, divisor) == BigInteger.One;
    }

    private static int GetBitLength(in BigInteger value)
    {
        if (value.IsZero)
        {
            return 0;
        }

        byte[] bytes = value.ToByteArray(isUnsigned: true, isBigEndian: false);
        byte msb = bytes[^1];
        return ((bytes.Length - 1) * 8) + (BitOperations.Log2(msb) + 1);
    }

    private static bool IsPowerOfTwo(in BigInteger value)
    {
        return value > BigInteger.One && (value & (value - BigInteger.One)) == BigInteger.Zero;
    }

    private static void WriteOutput(IReadOnlyCollection<Transition> transitions)
    {
        using var writer = new StreamWriter(OutputFileName, false, Encoding.UTF8);
        writer.WriteLine("p,divisor,lastDigit,k,lastMersenneDigit,isNextKHigher");

        foreach (Transition transition in transitions)
        {
            writer.WriteLine($"{transition.P.ToString(CultureInfo.InvariantCulture)},{FormatBigInteger(transition.Divisor)},{transition.LastDigit},{FormatBigInteger(transition.CurrentK)},{transition.LastMersenneDigit},{(transition.IsNextKHigher ? "1" : "0")}");
        }
    }

    private static int LastDigit(long p) => (int)(Math.Abs(p) % 10);

    private static int LastMersenneDigit(long p)
    {
        long mod = p & 3;
        return mod == 1 ? 1 : 7; // odd primes give p%4 == 1 or 3; 2^p mod 10 cycles -> 1 or 7 after subtracting 1
    }

    private static string FormatBigInteger(in BigInteger value) => value.ToString(CultureInfo.InvariantCulture);

    private sealed record Transition(long P, BigInteger Divisor, int LastDigit, BigInteger CurrentK, int LastMersenneDigit, bool IsNextKHigher);
}
