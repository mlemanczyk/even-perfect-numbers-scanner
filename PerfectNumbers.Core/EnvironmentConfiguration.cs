using System.Numerics;
using System.Runtime.CompilerServices;
using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core;

#pragma warning disable CA2211 // Non-constant fields should not be visible

public static class EnvironmentConfiguration
{
	public static ulong DivisorLimit = ulong.MaxValue;
	public static int GpuBatchSize = GpuConstants.ScanBatchSize;
	public static int GpuPrimeThreads = 1;
	public static int GpuRatio = PerfectNumberConstants.GpuRatio;
	public static ComputationDevice MersenneDevice = ComputationDevice.Cpu;
	public static BigInteger MinK = BigInteger.One;
	public static ComputationDevice OrderDevice = ComputationDevice.Cpu;
	public static int RollingAccelerators = PerfectNumberConstants.RollingAccelerators;
	public static string? StateFilePath;
	public static int ThreadCount;
	public static bool UseGpu;
	public static bool UseGpuOrder;
	public static bool UseHybridOrder;
	public static bool UseCpuOrder;
	public static bool UsePow2GroupDivisors;
	public static int ByDivisorSpecialRange;
	public static ByDivisor.ByDivisorClassModel? ByDivisorClassModel;
	public static ByDivisor.ByDivisorScanTuning ByDivisorScanTuning = ByDivisor.ByDivisorScanTuning.Default;
	public static double ByDivisorCheapKLimit = 10_000d;
	public static ByDivisorKIncrementMode ByDivisorKIncrementMode = ByDivisorKIncrementMode.Sequential;
	public static Persistence.ByDivisorClassModelRepository? ByDivisorClassModelRepository;
	public static Persistence.KStateRepository? ByDivisorKStateRepository;
	public static Persistence.KStateRepository? ByDivisorCheapKStateRepository;
	public static Persistence.KStateRepository? ByDivisorSpecialStateRepository;
	public static Persistence.KStateRepository? ByDivisorGroupsStateRepository;
	public static Persistence.Pow2Minus1StateRepository? ByDivisorPow2Minus1Repository;
	public static Persistence.BitContradictionStateRepository? BitContradictionStateRepository;
	public static CalculationMethod CalculationMethod = CalculationMethod.Residue;

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public static void Initialize()
	{
		if (MersenneDevice == ComputationDevice.Gpu &&
			ByDivisorKIncrementMode == ByDivisorKIncrementMode.BitContradiction)
		{
			OrderDevice = ComputationDevice.Gpu;
		}

		UseGpuOrder = OrderDevice == ComputationDevice.Gpu;
		UseHybridOrder = OrderDevice == ComputationDevice.Hybrid;
		UseCpuOrder = OrderDevice == ComputationDevice.Cpu;

		UseGpu =
			UseGpuOrder || UseHybridOrder ||
			MersenneDevice is ComputationDevice.Hybrid or ComputationDevice.Gpu;

		RollingAccelerators = UseGpu 
			? Math.Min(Math.Min(
			PerfectNumberConstants.RollingAccelerators, GpuPrimeThreads), ThreadCount)
			: 0;

		UsePow2GroupDivisors = ByDivisorKIncrementMode == ByDivisorKIncrementMode.Pow2Groups;
	}
}

#pragma warning restore CA2211 // Non-constant fields should not be visible
