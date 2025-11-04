using System;
using System.Threading;
using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

public static class SharedGpuContext
{
    private static readonly Lazy<Context> s_context = new(() => ILGPU.Context.CreateDefault(), LazyThreadSafetyMode.ExecutionAndPublication);
    private static readonly Lazy<Device> s_device = new(() => s_context.Value.GetPreferredDevice(false), LazyThreadSafetyMode.ExecutionAndPublication);
    private static readonly Lazy<Accelerator> s_accelerator = new(() => s_device.Value.CreateAccelerator(s_context.Value), LazyThreadSafetyMode.ExecutionAndPublication);

    internal static Context Context => s_context.Value;

    internal static Device Device => s_device.Value;

    internal static Accelerator Accelerator => s_accelerator.Value;
}
