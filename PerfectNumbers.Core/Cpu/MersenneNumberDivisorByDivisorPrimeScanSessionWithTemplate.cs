using System.Globalization;
using System.Globalization;
using System.IO;
using System.Numerics;
using PerfectNumbers.Core.Gpu;
using PerfectNumbers.Core.Gpu.Accelerators;
using System.Runtime.CompilerServices;


#if DivisorSet_Pow2Groups
using TesterCpu = PerfectNumbers.Core.Cpu.MersenneNumberDivisorByDivisorCpuTesterWithForPow2GroupsDivisorSetForCpuOrder;
using TesterHybrid = PerfectNumbers.Core.Cpu.MersenneNumberDivisorByDivisorCpuTesterWithForPow2GroupsDivisorSetForHybridOrder;
using TesterGpu = PerfectNumbers.Core.Cpu.MersenneNumberDivisorByDivisorCpuTesterWithForPow2GroupsDivisorSetForGpuOrder;
#elif DivisorSet_Predictive
using TesterCpu = PerfectNumbers.Core.Cpu.MersenneNumberDivisorByDivisorCpuTesterWithForPredictiveDivisorSetForCpuOrder;
using TesterHybrid = PerfectNumbers.Core.Cpu.MersenneNumberDivisorByDivisorCpuTesterWithForPredictiveDivisorSetForHybridOrder;
using TesterGpu = PerfectNumbers.Core.Cpu.MersenneNumberDivisorByDivisorCpuTesterWithForPredictiveDivisorSetForGpuOrder;
#else
using TesterCpu = PerfectNumbers.Core.Cpu.MersenneNumberDivisorByDivisorCpuTesterWithForOneByOneDivisorSetForCpuOrder;
using TesterHybrid = PerfectNumbers.Core.Cpu.MersenneNumberDivisorByDivisorCpuTesterWithForOneByOneDivisorSetForHybridOrder;
using TesterGpu = PerfectNumbers.Core.Cpu.MersenneNumberDivisorByDivisorCpuTesterWithForOneByOneDivisorSetForGpuOrder;
#endif

namespace PerfectNumbers.Core.Cpu;

[EnumDependentTemplate(typeof(DivisorSet))]
[DeviceDependentTemplate(typeof(ComputationDevice))]
[NameSuffix("Order")]
internal struct MersenneNumberDivisorByDivisorPrimeScanSessionWithTemplate
{
#if DEVICE_GPU
	private TesterGpu _tester;
#elif DEVICE_HYBRID
	private TesterHybrid _tester;
#else
	private TesterCpu _tester;
#endif

	private readonly Action _markComposite;
	private readonly Action _clearComposite;
	private readonly Action<ulong, bool, bool, bool, BigInteger> _printResult;

	public MersenneNumberDivisorByDivisorPrimeScanSessionWithTemplate(
#if DEVICE_GPU
		in TesterGpu prototype,
#elif DEVICE_HYBRID
		in TesterHybrid prototype,
#else
		in TesterCpu prototype,
#endif
		Action markComposite,
		Action clearComposite,
		Action<ulong, bool, bool, bool, BigInteger> printResult)
	{
		_tester = new()
		{
			DivisorLimit = prototype.DivisorLimit,
			MinK = EnvironmentConfiguration.MinK,
		};

		_markComposite = markComposite;
		_clearComposite = clearComposite;
		_printResult = printResult;
	}

	public void ProcessPrime(ulong prime)
	{
		try
		{
			Console.WriteLine($"Processing {prime}");
			string extension = ".bin";
#if DivisorSet_Predictive
			extension = ".predictive";
#endif
			string stateFile = Path.Combine(
				PerfectNumberConstants.ByDivisorStateDirectory,
				prime.ToString(CultureInfo.InvariantCulture) + extension);

			ConfigureForPrime(stateFile);

			bool isPrime = _tester.IsPrime(prime, out bool divisorsExhausted, out BigInteger divisor);

			if (!isPrime)
			{
				if (!string.IsNullOrEmpty(stateFile) && File.Exists(stateFile))
				{
					File.Delete(stateFile);
				}

				_markComposite();
				_printResult(prime, true, true, false, divisor);
				Console.WriteLine($"Finished processing {prime}");
				return;
			}

			_clearComposite();
			_printResult(prime, true, divisorsExhausted, true, divisor);
			Console.WriteLine($"Finished processing {prime}");
		}
		catch (Exception ex)
		{
			Console.WriteLine($"Error processing {ex.Message} {ex.StackTrace}");
			Environment.Exit(1);
		}
	}

	private void ConfigureForPrime(string stateFile)
	{
		BigInteger resumeK = EnvironmentConfiguration.MinK;
		if (File.Exists(stateFile))
		{
			if (MersenneNumberDivisorByDivisorTester.TryReadLastSavedK(stateFile, out BigInteger lastK))
			{
				resumeK = lastK + BigInteger.One;
				_tester.ResumeFromState(stateFile, lastK, resumeK);
			}
			else
			{
				_tester.ResumeFromState(stateFile, BigInteger.Zero, resumeK);
			}
		}
		else
		{
			_tester.ResumeFromState(stateFile, BigInteger.Zero, resumeK);
		}

		_tester.ResetStateTracking();
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public void Dispose()
	{
#if DEVICE_GPU || DEVICE_HYBRID
		PrimeOrderCalculatorAccelerator.Return(_tester.Accelerator);
#endif
	}
}
