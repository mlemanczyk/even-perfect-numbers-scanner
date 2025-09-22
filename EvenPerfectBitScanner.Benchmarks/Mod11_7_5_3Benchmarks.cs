using BenchmarkDotNet.Attributes;
using PerfectNumbers.Core;

namespace EvenPerfectBitScanner.Benchmarks;

[MemoryDiagnoser]
public class Mod11_7_5_3Benchmarks
{
    [Params(3UL, 8191UL, 131071UL, 2147483647UL)]
    public ulong Value { get; set; }

    [Benchmark(Baseline = true)]
    public (ulong Mod11, ulong Mod7, ulong Mod5, ulong Mod3) ModuloOperator()
    {
        ulong value = Value;
        return (value % 11UL, value % 7UL, value % 5UL, value % 3UL);
    }

    [Benchmark]
    public (ulong Mod11, ulong Mod7, ulong Mod5, ulong Mod3) ExtensionMethods()
    {
        ulong value = Value;
        return (value.Mod11(), value.Mod7(), value.Mod5(), value.Mod3());
    }

    [Benchmark]
    public (ulong Mod11, ulong Mod7, ulong Mod5, ulong Mod3) CombinedMethod()
    {
        Value.Mod11_7_5_3(out ulong mod11, out ulong mod7, out ulong mod5, out ulong mod3);
        return (mod11, mod7, mod5, mod3);
    }
}
