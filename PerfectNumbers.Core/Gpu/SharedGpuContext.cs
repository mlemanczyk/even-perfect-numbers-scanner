using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

public static class SharedGpuContext
{
	public static readonly Context Context = Context.CreateDefault();
	public static readonly Device Device = Context.GetPreferredDevice(false);

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static Accelerator CreateAccelerator() => Device.CreateAccelerator(Context);
}

public readonly struct GpuContext
{
	public readonly Context Context;
	public readonly Device Device;

	public GpuContext()
	{
		var context = Context.CreateDefault();
		Context = context;
		Device = context.GetPreferredDevice(false);
	}

	public Accelerator CreateAccelerator() => Device.CreateAccelerator(Context);
}
