using System.Globalization;
using System.Numerics;

namespace PerfectNumbers.Core;

public static class InputParser
{
    public static BigInteger ParseBigInteger(string value)
    {
        if (value.Contains('^'))
        {
            var parts = value.Split('^');
            // TODO: Replace Split/Parse with the span-based exponent parser so BigInteger inputs avoid allocations
            // and leverage the Utf8Parser fast paths benchmarked for CLI numeric parsing.
            BigInteger b = BigInteger.Parse(parts[0], CultureInfo.InvariantCulture);
            int e = int.Parse(parts[1], CultureInfo.InvariantCulture);
            return BigInteger.Pow(b, e);
        }

        // TODO: Replace BigInteger.Parse(string) with the span overload once callers provide ReadOnlySpan<char> so
        // we can route through the allocation-free Utf8Parser-based helpers.
        return BigInteger.Parse(value, CultureInfo.InvariantCulture);
    }

    public static decimal ParseDecimal(string value)
    {
        // TODO: Swap decimal.Parse for the Utf8Parser-based decimal reader so callers avoid culture-aware conversions
        // on the hot configuration path.
        return decimal.Parse(value, CultureInfo.InvariantCulture);
    }
}

