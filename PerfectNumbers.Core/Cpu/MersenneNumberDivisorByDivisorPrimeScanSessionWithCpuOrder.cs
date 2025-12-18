using System.Globalization;
using System.Numerics;
using PerfectNumbers.Core.Gpu;

namespace PerfectNumbers.Core.Cpu;

internal struct MersenneNumberDivisorByDivisorPrimeScanSessionWithCpuOrder(
	in MersenneNumberDivisorByDivisorCpuTesterWithCpuOrder prototype,
	Action markComposite,
	Action clearComposite,
	Action<ulong, bool, bool, bool, BigInteger> printResult)
{
	private MersenneNumberDivisorByDivisorCpuTesterWithCpuOrder _tester = new()
	{
		DivisorLimit = prototype.DivisorLimit,
		MinK = EnvironmentConfiguration.MinK,
	};

	private MersenneCpuDivisorScanSessionWithCpuOrder _divisorScanSession = new();

	private readonly Action _markComposite = markComposite;
	private readonly Action _clearComposite = clearComposite;
	private readonly Action<ulong, bool, bool, bool, BigInteger> _printResult = printResult;

	public void ProcessPrime(ulong prime)
	{
		try
		{
			Console.WriteLine($"Processing {prime}");
			string stateFile = Path.Combine(
				PerfectNumberConstants.ByDivisorStateDirectory,
				prime.ToString(CultureInfo.InvariantCulture) + ".bin");

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

	public void Dispose()
	{
	}
}
