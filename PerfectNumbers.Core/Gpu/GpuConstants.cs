namespace PerfectNumbers.Core.Gpu
{
        public static class GpuConstants
        {
                public const int ScanBatchSize = 2_097_152;
                public const int GpuCycleStepsPerInvocation = 256;
                // TODO: Auto-tune these GPU constants per accelerator so batch sizing aligns with the fastest kernel
                // configuration reported by the Pow2Mod and divisor-cycle benchmarks.
    }
}

