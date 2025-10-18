using System.Globalization;
using System.IO;

namespace EvenPerfectBitScanner.Candidates;

internal static class CalculationTestTime
{
    private const string TestTimeFileName = "even_perfect_test_time.txt";

    internal static void Report(TimeSpan elapsed, string? resultsDirectory)
    {
        double elapsedSeconds = elapsed.TotalSeconds;
        double elapsedMilliseconds = elapsed.TotalMilliseconds;
        Console.WriteLine($"Test elapsed time: {elapsedSeconds:F3} s");

        string filePath = GetFilePath(resultsDirectory);
        bool hasPrevious = TryRead(filePath, out double previousMilliseconds);
        string elapsedText = elapsedMilliseconds.ToString("F3", CultureInfo.InvariantCulture);

        if (!hasPrevious)
        {
            Write(filePath, elapsedMilliseconds);
            Console.WriteLine($"FIRST TIME: {elapsedText} ms");
            return;
        }

        string previousText = previousMilliseconds.ToString("F3", CultureInfo.InvariantCulture);
        if (elapsedMilliseconds < previousMilliseconds)
        {
            Write(filePath, elapsedMilliseconds);
            Console.WriteLine($"BETTER TIME: {elapsedText} ms (previous {previousText} ms)");
        }
        else
        {
            Console.WriteLine($"WORSE TIME: {elapsedText} ms (best {previousText} ms)");
        }
    }

    private static string GetFilePath(string? resultsDirectory)
    {
        if (string.IsNullOrEmpty(resultsDirectory))
        {
            return TestTimeFileName;
        }

        return Path.Combine(resultsDirectory!, TestTimeFileName);
    }

    private static bool TryRead(string filePath, out double milliseconds)
    {
        if (!File.Exists(filePath))
        {
            milliseconds = 0.0;
            return false;
        }

        string content = File.ReadAllText(filePath).Trim();
        if (content.Length == 0)
        {
            milliseconds = 0.0;
            return false;
        }

        return double.TryParse(content, NumberStyles.Float, CultureInfo.InvariantCulture, out milliseconds);
    }

    private static void Write(string filePath, double milliseconds)
    {
        string? directory = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        string formatted = milliseconds.ToString("F6", CultureInfo.InvariantCulture);
        File.WriteAllText(filePath, formatted);
    }
}
