namespace PerfectNumbers.Core.Gpu;

public readonly struct ResidueAutomatonArgs(
	ulong q0m10, ulong step10, ulong q0m8, ulong step8, ulong q0m3, ulong step3, ulong q0m5, ulong step5
)
{
	public readonly ulong Q0M10 = q0m10;
	public readonly ulong Step10 = step10;
	public readonly ulong Q0M8 = q0m8;
	public readonly ulong Step8 = step8;
	public readonly ulong Q0M3 = q0m3;
	public readonly ulong Step3 = step3;
	public readonly ulong Q0M5 = q0m5;
	public readonly ulong Step5 = step5;
}
