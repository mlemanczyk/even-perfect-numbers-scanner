using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public sealed class EulerPrimeTester
{
    private readonly PrimeTester _primeTester;

    public EulerPrimeTester(PrimeTester primeTester)
    {
        _primeTester = primeTester;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool IsEulerPrime(ulong n)
    {
        // TODO: Inline this helper into callers so we skip the wrapper and pass the divisibility check directly to the optimized
        // prime tester path highlighted in the CPU benchmarks.
        return (n & 3UL) == 1UL && _primeTester.IsPrime(n, CancellationToken.None);
    }
}

