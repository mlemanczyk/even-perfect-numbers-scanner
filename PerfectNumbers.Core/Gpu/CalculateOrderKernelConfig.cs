namespace PerfectNumbers.Core.Gpu;

public readonly struct CalculateOrderKernelConfig(ulong previousOrder, byte hasPreviousOrder, uint smallFactorLimit, int maxPowChecks, bool strictMode)
{
	public readonly ulong PreviousOrder = previousOrder;
	public readonly byte HasPreviousOrder = hasPreviousOrder;
	public readonly uint SmallFactorLimit = smallFactorLimit;
	public readonly int MaxPowChecks = maxPowChecks;
	public readonly int Mode = strictMode ? 1 : 0;
}
