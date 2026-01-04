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
	public static bool UsePow2GroupDivisors;
	public static int ByDivisorSpecialRange;
	public static ByDivisor.ByDivisorClassModel? ByDivisorClassModel;
	public static ByDivisor.ByDivisorScanTuning ByDivisorScanTuning = ByDivisor.ByDivisorScanTuning.Default;
	public static double ByDivisorCheapKLimit = 10_000d;
	public static Persistence.ByDivisorClassModelRepository? ByDivisorClassModelRepository;
	public static Persistence.KStateRepository? ByDivisorKStateRepository;
	public static Persistence.KStateRepository? ByDivisorCheapKStateRepository;
	public static Persistence.KStateRepository? ByDivisorSpecialStateRepository;
	public static Persistence.KStateRepository? ByDivisorGroupsStateRepository;
	public static Persistence.Pow2Minus1StateRepository? ByDivisorPow2Minus1Repository;

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static void Initialize()
	{
		UseGpuOrder = OrderDevice == ComputationDevice.Gpu;
		UseHybridOrder = OrderDevice == ComputationDevice.Hybrid;
		UseCpuOrder = OrderDevice == ComputationDevice.Cpu;
	}
}

#pragma warning restore CA2211 // Non-constant fields should not be visible
