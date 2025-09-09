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
        return (n & 3UL) == 1UL && _primeTester.IsPrime(n, CancellationToken.None);
    }
}
