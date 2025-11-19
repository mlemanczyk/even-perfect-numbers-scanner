namespace PerfectNumbers.Core;

public static class PerfectNumberConstants
{
    public const ulong BiggestKnownEvenPerfectP = 136279841UL;
	public const int ConsoleInterval = 100_000;
	public const int DefaultFactorsBuffer = 32;
	public const int DefaultSmallPrimeFactorSlotCount = 64;
	public const int DefaultSpecialMaxFactorCapacity = 1024;
    public const ulong ExtraDivisorCycleSearchLimit = 64UL;
	public const int GpuFrequency = 768 - 1;
    public const int MaxQForDivisorCycles = 4_000_000;
    public static readonly uint PrimesLimit = 1_000_000; //(ulong)Array.MaxLength;// 1_000_000;
    public const int MaxOddPowersCount = 128;
	public const int PooledArrayThreshold = 64;
		public static readonly int RollingAccelerators = 298; //SharedGpuContext.Device.MaxNumThreadsPerGroup;
	public const int ThreadsByAccelerator = 1;
    // TODO: Load these limits from the benchmark-driven configuration so CPU and GPU scans stay aligned with the optimal
    // divisor-cycle datasets we generate offline.
    // TODO: Promote these magic numbers into a runtime profile derived from EvenPerfectBitScanner.Benchmarks so we can retune
    // them automatically when the optimal ranges for 2kp+1 >= 138M change.
}
	