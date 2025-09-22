using System;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;

namespace EvenPerfectBitScanner.Benchmarks;

public static class Program
{
    public static void Main(string[] args)
    {
		var config = DefaultConfig.Instance.WithOptions(ConfigOptions.JoinSummary);
        BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args, config);
    }
}

