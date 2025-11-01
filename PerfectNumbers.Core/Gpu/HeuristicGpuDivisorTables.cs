using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

internal readonly struct HeuristicGpuDivisorTables
{
        public static readonly HeuristicGpuDivisorTables Shared = default;

        public readonly ArrayView1D<ulong, Stride1D.Dense> GroupADivisors;
        public readonly ArrayView1D<ulong, Stride1D.Dense> GroupBDivisorsEnding1;
        public readonly ArrayView1D<ulong, Stride1D.Dense> GroupBDivisorsEnding7;
        public readonly ArrayView1D<ulong, Stride1D.Dense> GroupBDivisorsEnding9;

        public HeuristicGpuDivisorTables(
                ArrayView1D<ulong, Stride1D.Dense> groupADivisors,
                ArrayView1D<ulong, Stride1D.Dense> groupBDivisorsEnding1,
                ArrayView1D<ulong, Stride1D.Dense> groupBDivisorsEnding7,
                ArrayView1D<ulong, Stride1D.Dense> groupBDivisorsEnding9)
        {
                GroupADivisors = groupADivisors;
                GroupBDivisorsEnding1 = groupBDivisorsEnding1;
                GroupBDivisorsEnding7 = groupBDivisorsEnding7;
                GroupBDivisorsEnding9 = groupBDivisorsEnding9;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ArrayView1D<ulong, Stride1D.Dense> SelectGroupB(byte ending)
        {
                return ending switch
                {
                        1 => GroupBDivisorsEnding1,
                        7 => GroupBDivisorsEnding7,
                        9 => GroupBDivisorsEnding9,
                        _ => ArrayView1D<ulong, Stride1D.Dense>.Empty,
                };
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ArrayView1D<ulong, Stride1D.Dense> SelectDivisors(HeuristicGpuDivisorTableKind kind, byte ending)
        {
                return kind switch
                {
                        HeuristicGpuDivisorTableKind.GroupA => GroupADivisors,
                        HeuristicGpuDivisorTableKind.GroupBEnding1 => GroupBDivisorsEnding1,
                        HeuristicGpuDivisorTableKind.GroupBEnding7 => GroupBDivisorsEnding7,
                        HeuristicGpuDivisorTableKind.GroupBEnding9 => GroupBDivisorsEnding9,
                        _ => ArrayView1D<ulong, Stride1D.Dense>.Empty,
                };
        }

        internal static void InitializeShared(in HeuristicGpuDivisorTables tables)
        {
                Unsafe.AsRef(in Shared) = tables;
        }
}

internal readonly struct HeuristicGpuCombinedDivisorTables
{
        public static readonly HeuristicGpuCombinedDivisorTables Shared = default;

        public readonly ArrayView1D<ulong, Stride1D.Dense> CombinedDivisorsEnding1;
        public readonly ArrayView1D<ulong, Stride1D.Dense> CombinedDivisorsEnding3;
        public readonly ArrayView1D<ulong, Stride1D.Dense> CombinedDivisorsEnding7;
        public readonly ArrayView1D<ulong, Stride1D.Dense> CombinedDivisorsEnding9;

        public HeuristicGpuCombinedDivisorTables(
                ArrayView1D<ulong, Stride1D.Dense> combinedDivisorsEnding1,
                ArrayView1D<ulong, Stride1D.Dense> combinedDivisorsEnding3,
                ArrayView1D<ulong, Stride1D.Dense> combinedDivisorsEnding7,
                ArrayView1D<ulong, Stride1D.Dense> combinedDivisorsEnding9)
        {
                CombinedDivisorsEnding1 = combinedDivisorsEnding1;
                CombinedDivisorsEnding3 = combinedDivisorsEnding3;
                CombinedDivisorsEnding7 = combinedDivisorsEnding7;
                CombinedDivisorsEnding9 = combinedDivisorsEnding9;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ArrayView1D<ulong, Stride1D.Dense> SelectCombined(byte ending)
        {
                return ending switch
                {
                        1 => CombinedDivisorsEnding1,
                        3 => CombinedDivisorsEnding3,
                        7 => CombinedDivisorsEnding7,
                        9 => CombinedDivisorsEnding9,
                        _ => ArrayView1D<ulong, Stride1D.Dense>.Empty,
                };
        }

        internal static void InitializeShared(in HeuristicGpuCombinedDivisorTables tables)
        {
                Unsafe.AsRef(in Shared) = tables;
        }
}
