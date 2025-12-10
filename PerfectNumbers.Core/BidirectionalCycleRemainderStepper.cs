using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public struct BidirectionalCycleRemainderStepper(ulong modulus)
{
	private readonly ulong _modulus = modulus;
	private ulong _previous = 0UL;
	private ulong _remainder = 0UL;

	public readonly ulong Modulus
	{
		[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
		get
		{
			return _modulus;
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public void Initialize(ulong value)
	{
		_remainder = value.ReduceCycleRemainder(_modulus);
		_previous = value;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
	public ulong Step(ulong value)
	{
		long delta = unchecked((long)value - (long)_previous);
		_previous = value;
		if (delta == 0)
		{
			return _remainder;
		}

		if (delta > 0)
		{
			ulong step = ((ulong)delta).ReduceCycleRemainder(_modulus);
			ulong sum = _remainder + step;
			if (sum >= _modulus)
			{
				sum -= _modulus;
			}

			_remainder = sum;
			return _remainder;
		}

		ulong backward = ((ulong)-delta).ReduceCycleRemainder(_modulus);
		if (backward > _remainder)
		{
			ulong diff = backward - _remainder;
			ulong wrap = diff.ReduceCycleRemainder(_modulus);
			_remainder = wrap == 0 ? 0UL : _modulus - wrap;
			return _remainder;
		}

		_remainder -= backward;
		return _remainder;
	}
}
