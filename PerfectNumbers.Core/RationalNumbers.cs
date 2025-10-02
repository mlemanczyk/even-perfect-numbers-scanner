using PeterO.Numbers;

namespace PerfectNumbers.Core;

public static class RationalNumbers
{
    public static ERational Two { get; } = ERational.FromInt32(2);
    // TODO: Precompute additional frequently used rationals (e.g., OneHalf, ThreeHalves) so hot alpha computations stop
    // constructing them via ERational.FromInt32 at runtime.
}

