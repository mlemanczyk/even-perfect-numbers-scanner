using System.Globalization;
using System.IO;
using System.Numerics;

namespace PerfectNumbers.Core.Gpu;

internal struct MersenneNumberDivisorByDivisorGpuPrimeScanSession
{
	private readonly MersenneNumberDivisorByDivisorGpuTester _tester;
	private readonly ulong _divisorLimit;
	private readonly Action _markComposite;
	private readonly Action _clearComposite;
	private readonly Action<ulong, bool, bool, bool, BigInteger> _printResult;

	public MersenneNumberDivisorByDivisorGpuPrimeScanSession(
		MersenneNumberDivisorByDivisorGpuTester prototype,
		Action markComposite,
		Action clearComposite,
		Action<ulong, bool, bool, bool, BigInteger> printResult)
	{
		_tester = new MersenneNumberDivisorByDivisorGpuTester
		{
			MinK = EnvironmentConfiguration.MinK,
		};
		_divisorLimit = prototype.DivisorLimit;
		_tester.SetDivisorLimitForClone(_divisorLimit);

		_markComposite = markComposite;
		_clearComposite = clearComposite;
		_printResult = printResult;
	}

	public void ProcessPrime(ulong prime)
	{
		try
		{
			Console.WriteLine($"Processing {prime}");
			string stateFile = prime.ToString(CultureInfo.InvariantCulture) + ".bin";

			ConfigureForPrime(stateFile);

			bool isPrime = _tester.IsPrime(prime, out bool divisorsExhausted, out BigInteger divisor);

			if (!isPrime)
			{
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
#if DivisorSet_TopDown
		BigInteger resumeK = EnvironmentConfiguration.MinK;
		BigInteger lastK = BigInteger.Zero;
		if (EnvironmentConfiguration.ByDivisorKStateRepository is not null)
		{
			if (ulong.TryParse(Path.GetFileNameWithoutExtension(stateFile), NumberStyles.None, CultureInfo.InvariantCulture, out ulong parsedPrime) &&
				EnvironmentConfiguration.ByDivisorKStateRepository.TryGet(parsedPrime, out BigInteger storedK) &&
				storedK > BigInteger.Zero)
			{
				lastK = storedK;
			}
		}

		_tester.ResumeFromState(stateFile, lastK, resumeK);
		_tester.ResetStateTracking();
		return;
#else
		BigInteger resumeK = EnvironmentConfiguration.MinK;
		BigInteger lastK = BigInteger.Zero;
		if (EnvironmentConfiguration.ByDivisorKStateRepository is not null)
		{
			if (ulong.TryParse(Path.GetFileNameWithoutExtension(stateFile), NumberStyles.None, CultureInfo.InvariantCulture, out ulong parsedPrime) &&
				EnvironmentConfiguration.ByDivisorKStateRepository.TryGet(parsedPrime, out BigInteger storedK) &&
				storedK > BigInteger.Zero)
			{
				lastK = storedK;
				resumeK = storedK + BigInteger.One;
			}
		}

		_tester.ResumeFromState(stateFile, lastK, resumeK);
		_tester.ResetStateTracking();
#endif
	}

	public void Dispose()
	{
	}
}
