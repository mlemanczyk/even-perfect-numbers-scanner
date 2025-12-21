namespace PerfectNumbers.Core.Gpu;

public static class PrimeOrderConstants
{
	public const int GpuSmallPrimeFactorSlots = 64;
	public const int HeuristicCandidateLimit = 512;
	public const int HeuristicStackCapacity = 256;
	public const int MaxGpuBatchSize = 256;
	public const ulong Pow2WindowFallbackThreshold = 32UL;
	public const int Pow2WindowOddPowerCount = 1 << (Pow2WindowSizeBits - 1);
	public const int Pow2WindowSizeBits = 8;
	public const int WideStackThreshold = 12;
	public const int MaxCompositeSlots = 256;
	public const int ExponentSlotLimit = 256;
	public const int PrimeSlotLimit = 256;
	public const int ScratchSlots = 64;

}
