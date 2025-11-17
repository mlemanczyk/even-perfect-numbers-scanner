namespace PerfectNumbers.Core.Gpu;

[Flags]
public enum KernelType
{
	None,
	SmallCycles,
	SmallPrimes,
	Pow2ModKernelScan,
	OrderKernelScan,
	IncrementalKernelScan,
	IncrementalOrderKernelScan,
	Pow2ModOrderKernelScan
}
