#r "nuget: ILGPU, 1.5.3"
#r "nuget: PeterO.Numbers, 1.8.1"
#r "nuget: Open.Numeric.Primes, 4.0.4"

using System;
using System.Numerics;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using PeterO.Numbers;
using Open.Numeric.Primes;

var context = Context.CreateDefault();
var accelerator = context.CreateCPUAccelerator(0);

try
{
    Console.WriteLine("ILGPU numeric type support probe (CPU accelerator)");
    Console.WriteLine();

    ProbeResult[] results =
    [
        ProbeBigInteger(accelerator),
        ProbeEInteger(accelerator),
        ProbeERational(accelerator),
        ProbeOpenNumericPrimesIsPrime(accelerator)
    ];

    foreach (ProbeResult result in results)
    {
        Console.WriteLine($"{result.Type}: {(result.Supported ? "supported" : "unsupported")}");
        Console.WriteLine($"  Details: {result.Details}");
        Console.WriteLine();
    }
}
finally
{
    accelerator.Dispose();
    context.Dispose();
}

record ProbeResult(string Type, bool Supported, string Details)
{
    public static ProbeResult Success(string type, string details) => new(type, true, details);

    public static ProbeResult Failure(string type, Exception exception) =>
        new(type, false, $"{exception.GetType().Name}: {exception.Message}");

    public static ProbeResult Failure(string type, string details) => new(type, false, details);
}

static ProbeResult ProbeBigInteger(Accelerator accelerator)
{
    try
    {
        var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>>(BigIntegerKernel);
        using var buffer = accelerator.Allocate1D<int>(1);
        var extent = new Index1D((int)buffer.Length);
        kernel(extent, buffer.View);
        accelerator.Synchronize();
        return ProbeResult.Success(
            "System.Numerics.BigInteger",
            "Addition, subtraction, multiplication, division, and modulo compile on the CPU accelerator.");
    }
    catch (Exception ex)
    {
        return ProbeResult.Failure("System.Numerics.BigInteger", ex);
    }
}

static ProbeResult ProbeEInteger(Accelerator accelerator)
{
    try
    {
        var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>>(EIntegerKernel);
        using var buffer = accelerator.Allocate1D<int>(1);
        kernel(new Index1D((int)buffer.Length), buffer.View);
        accelerator.Synchronize();
        return ProbeResult.Success(
            "PeterO.Numbers.EInteger",
            "Addition, subtraction, multiplication, division, and remainder compile on the CPU accelerator.");
    }
    catch (Exception ex)
    {
        return ProbeResult.Failure("PeterO.Numbers.EInteger", ex);
    }
}

static ProbeResult ProbeERational(Accelerator accelerator)
{
    bool hasRemainder = HasERationalRemainder();

    try
    {
        var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>>(ERationalKernel);
        using var buffer = accelerator.Allocate1D<int>(1);
        kernel(new Index1D((int)buffer.Length), buffer.View);
        accelerator.Synchronize();

        string details = "Addition, subtraction, multiplication, and division compile on the CPU accelerator.";
        details += hasRemainder
            ? " ERational exposes a remainder API that requires further investigation."
            : " ERational does not expose a remainder/modulo API.";
        return ProbeResult.Success("PeterO.Numbers.ERational", details);
    }
    catch (Exception ex)
    {
        string details = $"{ex.GetType().Name}: {ex.Message}";
        if (!hasRemainder)
        {
            details += " ERational does not expose a remainder/modulo API.";
        }

        return ProbeResult.Failure("PeterO.Numbers.ERational", details);
    }
}

static ProbeResult ProbeOpenNumericPrimesIsPrime(Accelerator accelerator)
{
    try
    {
        var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>>(OpenNumericPrimesIsPrimeKernel);
        using var buffer = accelerator.Allocate1D<int>(4);
        kernel(new Index1D((int)buffer.Length), buffer.View);
        accelerator.Synchronize();
        return ProbeResult.Success(
            "Open.Numeric.Primes.Prime.Numbers.IsPrime",
            "IsPrime compiles and runs on the CPU accelerator kernel.");
    }
    catch (Exception ex)
    {
        return ProbeResult.Failure("Open.Numeric.Primes.Prime.Numbers.IsPrime", ex);
    }
}

static void OpenNumericPrimesIsPrimeKernel(Index1D index, ArrayView1D<int, Stride1D.Dense> data)
{
    ulong value = (ulong)(index + 2);
    bool isPrime = Prime.Numbers.IsPrime(value);
    data[index] = isPrime ? 1 : 0;
}

static void BigIntegerKernel(Index1D index, ArrayView1D<int, Stride1D.Dense> data)
{
    BigInteger value = new BigInteger(data[index]);
    value += BigInteger.One;
    value -= BigInteger.One;
    value *= new BigInteger(2);
    value /= new BigInteger(3);
    value %= new BigInteger(5);
    int comparison = value.CompareTo(BigInteger.Zero);
    data[index] = comparison;
}

static void EIntegerKernel(Index1D index, ArrayView1D<int, Stride1D.Dense> data)
{
    EInteger value = EInteger.FromInt32(data[index]);
    value = value + EInteger.One;
    value = value - EInteger.One;
    value = value * EInteger.FromInt32(2);
    value = value.Divide(EInteger.FromInt32(3));
    value = value.Remainder(EInteger.FromInt32(5));
    int comparison = value.CompareTo(EInteger.Zero);
    data[index] = comparison;
}

static void ERationalKernel(Index1D index, ArrayView1D<int, Stride1D.Dense> data)
{
    ERational value = ERational.FromInt32(data[index]);
    ERational adjusted = (value + ERational.FromInt32(1)) - ERational.FromInt32(1);
    ERational product = adjusted * ERational.FromInt32(2);
    ERational quotient = product / ERational.FromInt32(3);
    int comparison = quotient.CompareTo(ERational.FromInt32(0));
    data[index] = comparison;
}

static bool HasERationalRemainder()
{
    foreach (var method in typeof(ERational).GetMethods())
    {
        if (string.Equals(method.Name, "Remainder", StringComparison.Ordinal))
        {
            return true;
        }
    }

    return false;
}
