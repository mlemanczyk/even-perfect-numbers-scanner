using System;

namespace PerfectNumbers.Core;

[AttributeUsage(AttributeTargets.Class | AttributeTargets.Struct, Inherited = false, AllowMultiple = false)]
public sealed class DeviceDependentTemplateAttribute : Attribute
{
    public DeviceDependentTemplateAttribute(Type enumType)
    {
        EnumType = enumType ?? throw new ArgumentNullException(nameof(enumType));
    }

    public Type EnumType { get; }
}

[AttributeUsage(AttributeTargets.Class | AttributeTargets.Struct, Inherited = false, AllowMultiple = true)]
public sealed class EnumDependentTemplateAttribute : Attribute
{
    public EnumDependentTemplateAttribute(Type enumType)
    {
        EnumType = enumType ?? throw new ArgumentNullException(nameof(enumType));
    }

    public Type EnumType { get; }
}

[AttributeUsage(AttributeTargets.Class | AttributeTargets.Struct, Inherited = false, AllowMultiple = false)]
public sealed class NameSuffixAttribute : Attribute
{
    public NameSuffixAttribute(string suffix)
    {
        Suffix = suffix ?? throw new ArgumentNullException(nameof(suffix));
    }

    public string Suffix { get; }
}
