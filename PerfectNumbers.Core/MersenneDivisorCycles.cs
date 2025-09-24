using System.Buffers;
using System.Text;
using PerfectNumbers.Core.Gpu;
using ILGPU;
using ILGPU.Runtime;
using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

public class MersenneDivisorCycles
{
	private List<(ulong divisor, ulong cycleLength)> _table = [];
	// Lightweight read-mostly cache for small divisors (<= 4,000,000). 0 => unknown
	private uint[]? _smallCycles;

	public static MersenneDivisorCycles Shared { get; } = new MersenneDivisorCycles();

	public MersenneDivisorCycles()
	{
	}

	public void LoadFrom(string path)
	{
		EnsureSmallBuffer();
		List<(ulong divisor, ulong cycle)> cycles = [];
		using Stream outputStream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, BufferSize10M, useAsync: false);
		foreach (var (d, c) in EnumerateStream(outputStream))
		{
			cycles.Add((d, c));
			if (d <= PerfectNumberConstants.MaxQForDivisorCycles)
			{
				_smallCycles![(int)d] = (uint)Math.Min(uint.MaxValue, c == 0UL ? 1UL : c);
			}
		}

		cycles.Sort((a, b) => a.divisor < b.divisor ? -1 : (a.divisor > b.divisor ? 1 : 0));
		_table = cycles;
	}

	public const int SmallDivisorsMax = PerfectNumberConstants.MaxQForDivisorCycles;

	// Provides a snapshot array of small cycles [0..PerfectNumberConstants.MaxQForDivisorCycles]. Index is the divisor.
	// The returned array is a copy safe for use across threads and device uploads.
	public uint[] ExportSmallCyclesSnapshot()
	{
		EnsureSmallBuffer();
		uint[] snapshot = new uint[PerfectNumberConstants.MaxQForDivisorCycles + 1];
		Array.Copy(_smallCycles!, snapshot, snapshot.Length);
		return snapshot;
	}

	public static IEnumerable<(ulong divisor, ulong cycleLength)> EnumerateStream(Stream compressor)
	{
		// Binary pairs: (ulong divisor, ulong cycle)
		using var reader = new BinaryReader(compressor, Encoding.UTF8, leaveOpen: true);
		while (true)
		{
			ulong d, c;
			try
			{
				d = reader.ReadUInt64();
				c = reader.ReadUInt64();
			}
			catch (EndOfStreamException)
			{
				yield break;
			}

			yield return (d, c);
		}
	}

	public ulong GetCycle(ulong divisor)
	{
		// Fast-path: in-memory array for small divisors
		if (divisor <= PerfectNumberConstants.MaxQForDivisorCycles)
		{
			var arr = _smallCycles;
			if (arr is not null)
			{
				uint cached = arr[(int)divisor];
				if (cached != 0U)
				{
					return cached;
				}
			}
		}

		// Binary search for divisor in sorted _table
		int mid, right, left = 0;
		ulong d;
		List<(ulong divisor, ulong cycleLength)> cycles = _table;

		right = cycles.Count - 1;
		while (left <= right)
		{
			mid = left + ((right - left) >> 1);
			d = cycles[mid].divisor;

			if (d == divisor)
			{
				return cycles[mid].cycleLength;
			}

			if (d < divisor)
			{
				left = mid + 1;
			}
			else
			{
				right = mid - 1;
			}
		}

		return CalculateCycleLength(divisor);
	}

	public static UInt128 GetCycle(UInt128 divisor)
	{
		// For divisor = 2^k, cycle is 1
		if ((divisor & (divisor - UInt128.One)) == UInt128.Zero)
		{
			return UInt128.One;
		}

		// Otherwise, find order of 2 mod divisor
		UInt128 order = UInt128.One, pow = UInt128Numbers.Two;
		while (pow != UInt128.One)
		{
			pow <<= 1;
			if (pow >= divisor)
				pow -= divisor;

			order++;
		}

		return order;
	}

	public static (long nextPosition, long completeCount) FindLast(string path)
	{
		using Stream outputStream = new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.Read, BufferSize10M, useAsync: false);
		long count = 0;
		foreach (var _ in EnumerateStream(outputStream))
		{
			count++;
		}
		return (outputStream.Position, count);
	}

	private const int BufferSize10M = 10 * 1024 * 1024;

	public static void Generate(string path, ulong maxDivisor, int threads = 16)
	{
		// Binary-only generation of (divisor, cycle) pairs. CSV scaffolding removed.
		ulong start = 2;
		ulong blockSize = maxDivisor / (ulong)threads + 1UL;
		Task[] tasks = new Task[threads];
		using Stream outputStream = new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.Read, BufferSize10M, useAsync: false);
		// Start fresh to avoid mixing formats
		outputStream.SetLength(0L);
		outputStream.Position = 0L;
		using var writer = new BinaryWriter(outputStream, Encoding.UTF8, leaveOpen: false);
		for (var taskIndex = 0; taskIndex < threads; taskIndex++)
		{
			ulong threadStart = start + (ulong)taskIndex * blockSize;
			if (threadStart > maxDivisor)
			{
				tasks[taskIndex] = Task.CompletedTask;
				continue;
			}

			ulong threadEnd = threadStart + blockSize - 1UL;
			if (threadEnd > maxDivisor)
			{
				threadEnd = maxDivisor;
			}

			ulong localStart = threadStart;
			ulong localEnd = threadEnd;
			tasks[taskIndex] = Task.Run(() =>
			{
				ulong rangeLength = localEnd - localStart + 1UL;
				(ulong divisor, ulong cycle)[] localCycles = ArrayPool<(ulong, ulong)>.Shared.Rent(checked((int)rangeLength));
				int localCycleIndex = 0;

				try
				{
					for (ulong divisor = localStart; divisor <= localEnd; divisor++)
					{
						if ((divisor & 1UL) == 0UL)
						{
							continue;
						}

						if ((divisor % 3UL) == 0UL || (divisor % 5UL) == 0UL || (divisor % 7UL) == 0UL || (divisor % 11UL) == 0UL)
						{
							continue;
						}

						ulong cycle = CalculateCycleLength(divisor);
						localCycles[localCycleIndex++] = (divisor, cycle);
					}

					if (localCycleIndex == 0)
					{
						return;
					}

					lock (writer)
					{
						writer.BaseStream.Position = writer.BaseStream.Length;
						for (int i = 0; i < localCycleIndex; i++)
						{
							(ulong divisor, ulong cycle) = localCycles[i];
							writer.Write(divisor);
							writer.Write(cycle);
						}

						writer.Flush();
					}
				}
				finally
				{
					ArrayPool<(ulong, ulong)>.Shared.Return(localCycles, clearArray: false);
				}
			});
		}

		Task.WaitAll(tasks);
	}

	public static void GenerateGpu(string path, ulong maxDivisor, int batchSize = 1_000_000, long skipCount = 0L, long nextPosition = 0L)
	{
		// Prepare output file with header

		ulong start = 3UL;
		var locker = new object();

		var pool = ArrayPool<ulong>.Shared;
		ulong batchSizeUL = (ulong)batchSize, d, end;
		int count = (int)Math.Min(batchSizeUL, maxDivisor), i, idx;
		ulong[] divisors, outCycles, validDivisors;

		divisors = pool.Rent(count);
		outCycles = pool.Rent(count);
		var lease = GpuKernelPool.GetKernel(useGpuOrder: true);
		var accelerator = lease.Accelerator;
		var stream = accelerator.CreateStream();
		var kernel = accelerator.LoadAutoGroupedStreamKernel<
				Index1D,
				ArrayView1D<ulong, Stride1D.Dense>,
				ArrayView1D<ulong, Stride1D.Dense>>(GpuDivisorCycleKernel);


		using Stream outputStream = new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.Read, BufferSize10M, useAsync: true);
		if (nextPosition > 0L)
		{
			outputStream.Position = nextPosition;
		}
		else
		{
			outputStream.SetLength(0L);
			outputStream.Position = 0L;
		}

		using var writer = new BinaryWriter(outputStream, Encoding.UTF8, leaveOpen: false);
		try
		{
			while (skipCount > 0L && start <= maxDivisor)
			{
				if ((start & 1UL) == 0UL || (start % 3UL) == 0UL || (start % 5UL) == 0UL || (start % 7UL) == 0UL || (start % 11UL) == 0UL)
				{
					start++;
					continue;
				}

				start++;
				skipCount--;
			}

			while (start <= maxDivisor)
			{
				end = Math.Min(start + batchSizeUL - 1UL, maxDivisor);
				count = checked((int)(end - start + 1UL));

				idx = 0;
				for (d = start; d <= end; d++)
				{
					if ((d & 1UL) == 0UL)
						continue;

					divisors[idx++] = d;
				}

				validDivisors = divisors[..idx];

				// Use GpuKernelPool to get a kernel and context
				// GpuKernelPool.Run((accelerator, stream) =>
				var bufferDiv = accelerator.Allocate1D(validDivisors);
				var bufferCycle = accelerator.Allocate1D<ulong>(idx);

				kernel(
						idx,
						bufferDiv.View,
						bufferCycle.View);

				accelerator.Synchronize();

				bufferCycle.View.CopyToCPU(ref outCycles[0], idx);
				bufferCycle.Dispose();
				bufferDiv.Dispose();

				// Collect results in order
				for (i = 0; i < idx; i++)
				{
					d = divisors[i];

					if ((d % 3UL) == 0UL || (d % 5UL) == 0UL || (d % 7UL) == 0UL || (d % 11UL) == 0UL)
						continue;

					writer.Write(d);
					writer.Write(outCycles[i]);
				}

				writer.Flush();
				Console.WriteLine($"...processed divisor = {d}");
				start = end + 1UL;
			}
		}
		finally
		{
			pool.Return(divisors, clearArray: false);
			pool.Return(outCycles, clearArray: false);
		}

		stream.Dispose();
		lease.Dispose();

	}

	const byte ByteZero = 0;
	const byte ByteOne = 1;

	// GPU kernel for divisor cycle calculation
	static void GpuDivisorCycleKernel(
		Index1D index,
		ArrayView1D<ulong, Stride1D.Dense> divisors,
		ArrayView1D<ulong, Stride1D.Dense> outCycles)
	{
		int i = index.X;
		outCycles[i] = CalculateCycleLengthGpu(divisors[i]);
	}

	// GPU-friendly version of cycle length calculation
	public static ulong CalculateCycleLengthGpu(ulong divisor)
	{
		if ((divisor & (divisor - 1UL)) == 0UL)
			return 1UL;

		ulong order = 1UL, pow = 2UL;
		while (pow != 1UL)
		{
			pow <<= 1;
			if (pow >= divisor)
				pow -= divisor;

			order++;
		}

		return order;
	}

	public static ulong CalculateCycleLength(ulong divisor)
	{
		// For divisor = 2^k, cycle is 1
		if ((divisor & (divisor - 1UL)) == 0UL)
		{
			return 1UL;
		}

		// Otherwise, find order of 2 mod divisor
		ulong order = 1UL, pow = 2UL;
		while (pow != 1)
		{
			pow <<= 1;
			if (pow >= divisor)
				pow -= divisor;

			order++;
		}

		return order;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private void EnsureSmallBuffer()
	{
		if (_smallCycles is null)
		{
			_smallCycles = new uint[PerfectNumberConstants.MaxQForDivisorCycles + 1];
		}
	}
}
