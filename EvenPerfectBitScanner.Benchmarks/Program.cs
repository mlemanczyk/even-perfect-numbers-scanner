using PerfectNumbers.Core.Gpu;

namespace EvenPerfectBitScanner.Benchmarks;

public static class Program
{
    public static void Main(string[] args)
    {
        // var config = DefaultConfig.Instance.WithOptions(ConfigOptions.JoinSummary | ConfigOptions.DisableOptimizationsValidator);
        // BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args, config);

        Pow2ModBenchmarks benchmarks = new()
        {
            Scale = Pow2ModBenchmarks.InputScale.Large
        };

        GpuPrimeWorkLimiter.SetLimit(1);
        benchmarks.Setup();

        // Console.WriteLine("Press ENTER to run");
        // Console.ReadLine();

        // foreach (int i in Enumerable.Range(0, 100_000))
        // {
        //     benchmarks.MontgomeryWithPrecomputedCycle();
        // }
    }
}

