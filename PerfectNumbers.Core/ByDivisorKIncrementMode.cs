namespace PerfectNumbers.Core;
#pragma warning disable CA2211 // The fields are settable on purpose for the best performance

public enum ByDivisorKIncrementMode
{
    Sequential,
    Pow2Groups,
    Predictive,
    Percentile,
    Additive,
    TopDown,
    BitContradiction,
    BitTree,
}

#pragma warning restore CA2211 // Non-constant fields should not be visible
