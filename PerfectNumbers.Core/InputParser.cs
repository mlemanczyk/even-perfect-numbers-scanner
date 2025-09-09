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
            BigInteger b = BigInteger.Parse(parts[0], CultureInfo.InvariantCulture);
            int e = int.Parse(parts[1], CultureInfo.InvariantCulture);
            return BigInteger.Pow(b, e);
        }

        return BigInteger.Parse(value, CultureInfo.InvariantCulture);
    }

    public static decimal ParseDecimal(string value)
    {
        return decimal.Parse(value, CultureInfo.InvariantCulture);
    }
}

