using System;
using System.Diagnostics;

namespace PerfectNumbers.Core;

internal readonly struct PrimeTestTimeLimit
{
    private readonly long _deadlineTimestamp;

    private PrimeTestTimeLimit(bool isActive, long deadlineTimestamp)
    {
        IsActive = isActive;
        _deadlineTimestamp = deadlineTimestamp;
    }

    internal bool IsActive { get; }

    internal static bool TryCreate(TimeSpan? timeLimit, out PrimeTestTimeLimit limit)
    {
        if (!timeLimit.HasValue)
        {
            limit = new PrimeTestTimeLimit(isActive: false, deadlineTimestamp: long.MaxValue);
            return true;
        }

        TimeSpan limitValue = timeLimit.GetValueOrDefault();
        if (limitValue <= TimeSpan.Zero)
        {
            limit = new PrimeTestTimeLimit(isActive: true, deadlineTimestamp: long.MinValue);
            return false;
        }

        double seconds = limitValue.TotalSeconds;
        if (double.IsInfinity(seconds))
        {
            limit = new PrimeTestTimeLimit(isActive: true, deadlineTimestamp: long.MaxValue);
            return true;
        }

        double scaled = seconds * Stopwatch.Frequency;
        long limitTicks = scaled >= long.MaxValue ? long.MaxValue : (long)scaled;
        long startTimestamp = Stopwatch.GetTimestamp();
        long deadline = startTimestamp <= long.MaxValue - limitTicks ? startTimestamp + limitTicks : long.MaxValue;
        if (deadline < startTimestamp)
        {
            deadline = long.MaxValue;
        }

        limit = new PrimeTestTimeLimit(isActive: true, deadlineTimestamp: deadline);
        return true;
    }

    internal bool HasExpired()
    {
        if (!IsActive)
        {
            return false;
        }

        if (_deadlineTimestamp == long.MaxValue)
        {
            return false;
        }

        if (_deadlineTimestamp == long.MinValue)
        {
            return true;
        }

        return Stopwatch.GetTimestamp() >= _deadlineTimestamp;
    }
}
