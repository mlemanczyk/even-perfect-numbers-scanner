namespace EvenPerfectBitScanner.Benchmarks;

public enum ModNumericType
{
    UInt32,
    UInt64,
    UInt128,
}

public readonly struct ModBenchmarkCase
{
    public ModBenchmarkCase(ModNumericType type, UInt128 value, string display)
    {
        Type = type;
        Value = value;
        Display = display;
    }

    public ModNumericType Type { get; }

    public UInt128 Value { get; }

    private string Display { get; }

    public override string ToString()
    {
        return Display;
    }
}
