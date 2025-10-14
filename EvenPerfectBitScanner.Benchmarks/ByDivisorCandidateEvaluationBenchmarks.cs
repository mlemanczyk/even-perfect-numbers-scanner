using System;
using BenchmarkDotNet.Attributes;
using PerfectNumbers.Core;
using PerfectNumbers.Core.Gpu;

namespace EvenPerfectBitScanner.Benchmarks;

[MemoryDiagnoser]
public class ByDivisorCandidateEvaluationBenchmarks : IDisposable
{
    private readonly MersenneNumberDivisorCandidateGpuEvaluator _gpuEvaluator;
    private readonly ulong[] _candidateBuffer;
    private readonly byte[] _maskBuffer;
    private readonly byte[] _gpuRemainders;
    private readonly byte[] _gpuSteps;
    private readonly byte[] _cpuRemainders;
    private readonly byte[] _cpuSteps;
    private readonly bool _lastIsSeven;
    private readonly ulong _startDivisor;
    private readonly ulong _step;
    private readonly ulong _limit;

    public ByDivisorCandidateEvaluationBenchmarks()
    {
        const int MaxBatchSize = 512;
        ulong prime = 138_000_001UL;
        _step = prime << 1;
        _startDivisor = _step + 1UL;
        _limit = _startDivisor + _step * (ulong)MaxBatchSize;
        _lastIsSeven = (prime & 3UL) == 3UL;

        _gpuEvaluator = new MersenneNumberDivisorCandidateGpuEvaluator(MaxBatchSize);
        _candidateBuffer = new ulong[MaxBatchSize];
        _maskBuffer = new byte[MaxBatchSize];
        _gpuRemainders = new byte[6];
        _gpuSteps = new byte[6];
        _cpuRemainders = new byte[6];
        _cpuSteps = new byte[6];

        InitializeState();
    }

    [Params(32, 128, 256)]
    public int Count { get; set; }

    [IterationSetup]
    public void ResetState()
    {
        InitializeState();
    }

    [Benchmark]
    public void EvaluateOnGpu()
    {
        Span<ulong> candidates = _candidateBuffer.AsSpan(0, Count);
        Span<byte> mask = _maskBuffer.AsSpan(0, Count);
        Span<byte> remainders = _gpuRemainders.AsSpan();
        ReadOnlySpan<byte> steps = _gpuSteps.AsSpan();

        _gpuEvaluator.EvaluateCandidates(
            _startDivisor,
            _step,
            _limit,
            remainders,
            steps,
            _lastIsSeven,
            candidates,
            mask);
    }

    [Benchmark]
    public void EvaluateOnCpu()
    {
        Span<ulong> candidates = _candidateBuffer.AsSpan(0, Count);
        Span<byte> mask = _maskBuffer.AsSpan(0, Count);
        Span<byte> remainders = _cpuRemainders.AsSpan();
        ReadOnlySpan<byte> steps = _cpuSteps.AsSpan();

        MersenneNumberDivisorCandidateCpuEvaluator.EvaluateCandidates(
            _startDivisor,
            _step,
            _limit,
            remainders,
            steps,
            _lastIsSeven,
            candidates,
            mask);
    }

    private void InitializeState()
    {
        _gpuRemainders[0] = (byte)(_startDivisor % 10UL);
        _gpuRemainders[1] = (byte)(_startDivisor % 8UL);
        _gpuRemainders[2] = (byte)(_startDivisor % 5UL);
        _gpuRemainders[3] = (byte)(_startDivisor % 3UL);
        _gpuRemainders[4] = (byte)(_startDivisor % 7UL);
        _gpuRemainders[5] = (byte)(_startDivisor % 11UL);

        _gpuSteps[0] = (byte)(_step % 10UL);
        _gpuSteps[1] = (byte)(_step % 8UL);
        _gpuSteps[2] = (byte)(_step % 5UL);
        _gpuSteps[3] = (byte)(_step % 3UL);
        _gpuSteps[4] = (byte)(_step % 7UL);
        _gpuSteps[5] = (byte)(_step % 11UL);

        _gpuRemainders.AsSpan().CopyTo(_cpuRemainders);
        _gpuSteps.AsSpan().CopyTo(_cpuSteps);
    }

    public void Dispose()
    {
        _gpuEvaluator.Dispose();
    }
}
