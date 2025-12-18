using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core.Cpu;

internal struct GpuUInt128WorkSet
{
	public GpuUInt128 Step;
	public GpuUInt128 Divisor;
	public GpuUInt128 Limit;
}
