using System.Runtime.CompilerServices;
using ILGPU.Runtime;

namespace PerfectNumbers.Core.Gpu;

internal sealed class AcceleratorReferenceComparer : IEqualityComparer<Accelerator>
{
	internal static AcceleratorReferenceComparer Instance { get; } = new();

	public bool Equals(Accelerator? x, Accelerator? y) => ReferenceEquals(x, y);

	public int GetHashCode(Accelerator obj) => RuntimeHelpers.GetHashCode(obj);
}
