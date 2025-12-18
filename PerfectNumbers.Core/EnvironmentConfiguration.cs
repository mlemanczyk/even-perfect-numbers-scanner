using System.Numerics;
using System.Runtime.CompilerServices;
using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core;

#pragma warning disable CA2211 // The fields are settable on purpose for the best performance

public static class EnvironmentConfiguration
{
	public static ulong DivisorLimit = ulong.MaxValue;
	public static int GpuBatchSize = GpuConstants.ScanBatchSize;
	public static int GpuRatio = PerfectNumberConstants.GpuRatio;
	public static BigInteger MinK = BigInteger.One;
	public static ComputationDevice OrderDevice = ComputationDevice.Cpu;
	public static int RollingAccelerators = PerfectNumberConstants.RollingAccelerators;
	public static string? StateFilePath;
	public static bool UseGpuOrder;
	public static bool UseHybridOrder;
	public static bool UseCpuOrder;

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static void Initialize()
	{
		UseGpuOrder = OrderDevice == ComputationDevice.Gpu;
		UseHybridOrder = OrderDevice == ComputationDevice.Hybrid;
		UseCpuOrder = OrderDevice == ComputationDevice.Cpu;
	}
}

#pragma warning restore CA2211 // Non-constant fields should not be visible
