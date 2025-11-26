namespace PerfectNumbers.Core.Gpu.Kernels;

public enum PrimeOrderKernelStatus : byte
{
	Fallback = 0,
	Found = 1,
	HeuristicUnresolved = 2,
	PollardOverflow = 3,
	FactoringFailure = 4,
}
