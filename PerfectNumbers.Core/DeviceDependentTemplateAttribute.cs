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

