using System.Runtime.CompilerServices;
using ILGPU;

namespace PerfectNumbers.Core.Gpu;

public static class ThreadStaticDeterministicRandomGpu
{
	[ThreadStatic]
	private static DeterministicRandomGpu? _exclusive;
	internal static DeterministicRandomGpu Exclusive => _exclusive ?? new DeterministicRandomGpu();
}

internal struct DeterministicRandomGpu
{
	private static long s_seedCounter;

	private ulong _state;
	public readonly ulong State
	{
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		get
		{
			return _state;
		}
	}

	public DeterministicRandomGpu()
	{
		_state = CreateThreadState();
	}

	public DeterministicRandomGpu(ulong initialState)
	{
		_state = initialState;
	}

	public ulong NextUInt64()
	{
		ulong state = _state;
		state ^= state >> 12;
		state ^= state << 25;
		state ^= state >> 27;
		_state = state;
		return state * 2685821657736338717UL;
	}

	public GpuUInt128 NextUInt128()
	{
		GpuUInt128 high = (GpuUInt128)NextUInt64();
		ReadOnlyGpuUInt128 low = (ReadOnlyGpuUInt128)NextUInt64();
		high.ShiftLeft(64);
		high.Add(low);
		return high;
	}

	public void SetState(ulong state)
	{
		_state = state;
	}

	private static ulong CreateThreadState()
	{
		ulong seed = (ulong)Atomic.Add(ref s_seedCounter, 1);
		seed += 0x9E3779B97F4A7C15UL;
		seed = (seed ^ (seed >> 30)) * 0xBF58476D1CE4E5B9UL;
		seed = (seed ^ (seed >> 27)) * 0x94D049BB133111EBUL;
		seed ^= seed >> 31;
		if (seed == 0UL)
		{
			seed = 0x9E3779B97F4A7C15UL;
		}

		return seed;
	}
}
