using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public sealed class PrimeTesterByLastDigit
{
	[ThreadStatic]
	private static BidirectionalCycleRemainderStepper _mod10Stepper;

	[ThreadStatic]
	private static bool _mod10Initialized;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static byte GetMod10(ulong value)
	{
		if (!_mod10Initialized || _mod10Stepper.Modulus != 10UL)
		{
			_mod10Stepper = new BidirectionalCycleRemainderStepper(10UL);
			_mod10Stepper.Initialize(value);
			_mod10Initialized = true;
			return (byte)_mod10Stepper.Step(value);
		}

		return (byte)_mod10Stepper.Step(value);
	}

	public static bool IsPrimeCpu(ulong n)
	{
		// The below IFs never trigger in production code of EvenPerfectBitScanner on --mersenne=bydivisor path.
		// if (n <= 1UL)
		// {
		// 	throw new InvalidOperationException("PrimeTester.IsPrime encountered the sentinel input 2.");
		// 	// return false;
		// }

		// if (n == 2UL)
		// {
		// 	throw new InvalidOperationException("PrimeTester.IsPrime encountered the sentinel input 2.");
		// }

		// EvenPerfectBitScanner streams exponents starting at 136,279,841, so the Mod10/GCD guard never fires on the
		// production path. Leave the logic commented out as instrumentation for diagnostic builds.
		// bool sharesMaxExponentFactor = n.Mod10() == 1UL && SharesFactorWithMaxExponent(n);
		// result &= !sharesMaxExponentFactor;

		uint[] smallPrimeDivisors;
		ulong[] smallPrimeDivisorsMul;
		byte nMod10 = GetMod10(n);

		switch (nMod10)
		{
			case 1:
				smallPrimeDivisors = PrimesGenerator.SmallPrimesLastOne;
				smallPrimeDivisorsMul = PrimesGenerator.SmallPrimesPow2LastOne;
				break;
			case 3:
				smallPrimeDivisors = DivisorGenerator.SmallPrimesLastThree;
				smallPrimeDivisorsMul = DivisorGenerator.SmallPrimesPow2LastThree;
				break;
			case 7:
				smallPrimeDivisors = PrimesGenerator.SmallPrimesLastSeven;
				smallPrimeDivisorsMul = PrimesGenerator.SmallPrimesPow2LastSeven;
				break;
			default:
				smallPrimeDivisors = DivisorGenerator.SmallPrimesLastNine;
				smallPrimeDivisorsMul = DivisorGenerator.SmallPrimesPow2LastNine;
				break;
		}

		int smallPrimeDivisorsLength = smallPrimeDivisors.Length;

		AscendingDivisorRemainderStepper stepper = new(n);
		for (int i = 0; i < smallPrimeDivisorsLength; i++)
		{
			ulong value = smallPrimeDivisorsMul[i];
			if (!stepper.ShouldContinue(value))
			{
				break;
			}

			uint divisor = smallPrimeDivisors[i];
			if (stepper.Divides(divisor))
			{
				return false;
			}
		}

		return true;
	}

	// [MethodImpl(MethodImplOptions.AggressiveInlining)]
	// public static bool IsPrimeGpu(PrimeOrderCalculatorAccelerator gpu, ulong n)
	// {
	// 	byte flag = 0;
	// 	var inputView = gpu.InputView;
	// 	var outputView = gpu.OutputByteView;

	// 	int acceleratorIndex = gpu.AcceleratorIndex;
	// 	AcceleratorStream stream = AcceleratorStreamPool.Rent(acceleratorIndex);
	// 	inputView.CopyFromCPU(stream, ref n, 1);

	// 	var kernelLauncher = gpu.SmallPrimeSieveKernelLauncher;
		
	// 	kernelLauncher(
	// 					stream,
	// 					1,
	// 					inputView,
	// 					gpu.DevicePrimesLastOne,
	// 					gpu.DevicePrimesLastSeven,
	// 					gpu.DevicePrimesLastThree,
	// 					gpu.DevicePrimesLastNine,
	// 					gpu.DevicePrimesPow2LastOne,
	// 					gpu.DevicePrimesPow2LastSeven,
	// 					gpu.DevicePrimesPow2LastThree,
	// 					gpu.DevicePrimesPow2LastNine,
	// 					outputView);

	// 	outputView.CopyToCPU(stream, ref flag, 1);
	// 	stream.Synchronize();
	// 	AcceleratorStreamPool.Return(acceleratorIndex, stream);

	// 	return flag != 0;
	// }

	// public static int GpuBatchSize { get; set; } = 262_144;

	// [MethodImpl(MethodImplOptions.AggressiveInlining)]
	// public static void IsPrimeBatchGpu(PrimeOrderCalculatorAccelerator gpu, ReadOnlySpan<ulong> values, Span<byte> results)
	// {
	// 	int acceleratorIndex = gpu.AcceleratorIndex;
	// 	int totalLength = values.Length;
	// 	int batchSize = GpuBatchSize;

	// 	var inputView = gpu.InputView;
	// 	var outputView = gpu.OutputByteView;

	// 	int pos = 0;
	// 	AcceleratorStream stream = AcceleratorStreamPool.Rent(acceleratorIndex);
	// 	var kernelLauncher = gpu.SmallPrimeSieveKernelLauncher;

	// 	while (pos < totalLength)
	// 	{
	// 		int remaining = totalLength - pos;
	// 		int count = remaining > batchSize ? batchSize : remaining;

	// 		var valueSlice = values.Slice(pos, count);
	// 		inputView.CopyFromCPU(stream, valueSlice);

	// 		kernelLauncher(
	// 				stream,
	// 				count,
	// 				inputView,
	// 				gpu.DevicePrimesLastOne,
	// 				gpu.DevicePrimesLastSeven,
	// 				gpu.DevicePrimesLastThree,
	// 				gpu.DevicePrimesLastNine,
	// 				gpu.DevicePrimesPow2LastOne,
	// 				gpu.DevicePrimesPow2LastSeven,
	// 				gpu.DevicePrimesPow2LastThree,
	// 				gpu.DevicePrimesPow2LastNine,
	// 				outputView);

	// 		var resultSlice = results.Slice(pos, count);
	// 		outputView.CopyToCPU(stream, resultSlice);

	// 		pos += count;
	// 	}

	// 	stream.Synchronize();
	// 	AcceleratorStreamPool.Return(acceleratorIndex, stream);
	// }
}
