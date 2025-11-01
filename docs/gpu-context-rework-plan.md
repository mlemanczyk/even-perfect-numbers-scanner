# GPU context pooling and heuristic divisor updates

- [x] Review existing GPU context pooling in `PerfectNumbers.Core/PrimeTester.cs` and determine shared table allocations that can be scoped per trailing digit.
- [x] Split `PrimeTesterGpuContextPool` into trailing-digit specific pools so each lease only uploads the divisor tables required for that digit.
- [x] Update GPU warm-up and rental code paths to pick pools based on `n % 10` and ensure thread-safe reuse.
- [x] Replace square-table usage in GPU heuristic trial division with square-root limits so divisor tables no longer need squared values on the accelerator.
- [x] Remove GPU square buffers and adjust kernels, table types, and callers accordingly.
- [x] Build solution to confirm the new pooling and divisor logic compiles; tests documented in summary.
