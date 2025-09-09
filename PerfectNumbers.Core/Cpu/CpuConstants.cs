namespace PerfectNumbers.Core.Cpu;

public static class CpuConstants
{
	// Allowed q mod 10 sets depending on last digit of M_p
	public const int LastSevenMask10 = (1 << 7) | (1 << 9);
	// For M_p ending with 1, restrict to {1,3,9}; 7 considered only for q == 7 (not possible for large p)
	public const int LastOneMask10 = (1 << 1) | (1 << 3) | (1 << 9);
}