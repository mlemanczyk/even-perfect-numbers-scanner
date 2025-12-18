using System;

namespace PerfectNumbers.Core;

[AttributeUsage(AttributeTargets.Class | AttributeTargets.Struct, Inherited = false, AllowMultiple = false)]
public sealed class DeviceDependentTemplateAttribute : Attribute
{
    public DeviceDependentTemplateAttribute(Type enumType, string? suffix = null)
    {
        EnumType = enumType ?? throw new ArgumentNullException(nameof(enumType));
        Suffix = suffix;
    }

    public Type EnumType { get; }

    public string? Suffix { get; }
}
