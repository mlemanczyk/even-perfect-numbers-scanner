using System;

namespace PerfectNumbers.Core.ByDivisor;

public	 static class BitContradictionTailCarryLut8
{
	private const int Rows = 8;
	private const int Columns = 8;
	private const int WindowBits = Rows + Columns - 1;
	private const int CarryStates = Rows + 1;
	private const byte FullMask = (1 << Rows) - 1;

	private static ushort[]? _table;
	private static readonly ushort[]?[] _tablesByMask = new ushort[(1 << Rows)][];

	public static void WarmUp()
	{
		_table ??= BuildTable(FullMask);
	}

	public static void WarmUpAll()
	{
		for (int mask = 0; mask < (1 << Rows); mask++)
		{
			_ = GetTable((byte)mask);
		}
	}

	public static ReadOnlySpan<ushort> Table => _table ?? throw new InvalidOperationException("BitContradictionTailCarryLut8 not initialized.");

	public static ReadOnlySpan<ushort> GetTable(byte rowMask)
	{
		ushort[]? table = _tablesByMask[rowMask];
		if (table is null)
		{
			table = BuildTable(rowMask);
			_tablesByMask[rowMask] = table;
			if (rowMask == FullMask)
			{
				_table ??= table;
			}
		}

		return table;
	}

	private static ushort[] BuildTable(byte rowMask)
	{
		int entries = (1 << WindowBits) * CarryStates;
		var table = new ushort[entries];
		int entryIndex = 0;

		for (int window = 0; window < (1 << WindowBits); window++)
		{
			for (int carryIn = 0; carryIn < CarryStates; carryIn++)
			{
				int carry = carryIn;
				ushort result = 0;

				for (int column = 0; column < Columns; column++)
				{
					int ones = 0;
					for (int shift = 0; shift < Rows; shift++)
					{
						if (((rowMask >> shift) & 1) == 0)
						{
							continue;
						}

						int windowIndex = column - shift + (Rows - 1);
						if ((uint)windowIndex < WindowBits && ((window >> windowIndex) & 1) != 0)
						{
							ones++;
						}
					}

					int sum = carry + ones;
					result |= (ushort)((sum & 1) << column);
					carry = sum >> 1;
				}

				table[entryIndex++] = (ushort)(result | (carry << 8));
			}
		}

		return table;
	}
}
