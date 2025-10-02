namespace PerfectNumbers.Core;

public enum AlphaMScannerCandidateStatus
{
    // TODO: Store these statuses as byte-coded constants so hot-path evaluations can stay branchless when scanning candidates.
    UNKNOWN,
    PRODUCT_ABOVE_TARGET,
    PRODUCT_BELOW_TARGET,
    ALPHA_P_AND_ALPHA_M_OUT_OF_RANGE,
    ALPHA_P_OUT_OF_RANGE,
    ALPHA_M_OUT_OF_RANGE,
    OK,
}

