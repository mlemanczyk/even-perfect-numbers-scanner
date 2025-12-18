using System.Collections.Immutable;
using FluentAssertions;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Xunit;

namespace DeviceDependentGenerator.Tests;

public sealed class DeviceDependentGeneratorTests
{
    [Fact]
    public void Generates_classes_per_enum_value_and_filters_device_blocks()
    {
        const string attributeAndEnum = """
            using System;

            namespace PerfectNumbers.Core;

            [AttributeUsage(AttributeTargets.Class | AttributeTargets.Struct, Inherited = false, AllowMultiple = false)]
            public sealed class DeviceDependentTemplateAttribute : Attribute
            {
                public DeviceDependentTemplateAttribute(Type enumType) { }
            }

            public enum CalculationDevice
            {
                Cpu,
                Hybrid,
                Gpu,
            }
            """;

        const string template = """
            using System;

            namespace PerfectNumbers.Core;

            [DeviceDependentTemplate(typeof(CalculationDevice))]
            public static class PrimeCalculatorTemplate
            {
            #if DEVICE_CPU
                public static string Device => "CPU";
            #elif DEVICE_HYBRID
                public static string Device => "HYBRID";
            #elif DEVICE_GPU
                public static string Device => "GPU";
            #else
            #error Unknown device
            #endif
            }
            """;

        CSharpCompilation compilation = CreateCompilation(attributeAndEnum, template);

        var generator = new PerfectNumbers.Core.Generators.DeviceDependentGenerator();
        var parseOptions = (CSharpParseOptions)compilation.SyntaxTrees.First().Options;
        ISourceGenerator sourceGenerator = generator.AsSourceGenerator();
        GeneratorDriver driver = CSharpGeneratorDriver.Create(new[] { sourceGenerator }, parseOptions: parseOptions);
        driver = driver.RunGeneratorsAndUpdateCompilation(compilation, out _, out ImmutableArray<Diagnostic> diagnostics);

        diagnostics.Should().BeEmpty();

        GeneratorDriverRunResult runResult = driver.GetRunResult();
        runResult.Diagnostics.Should().BeEmpty();

        ImmutableArray<GeneratedSourceResult> generated = runResult.Results.Single().GeneratedSources;

        string cpu = generated.Single(s => s.HintName.EndsWith("PrimeCalculator.Cpu.generated.cs", StringComparison.Ordinal)).SourceText.ToString();
        string hybrid = generated.Single(s => s.HintName.EndsWith("PrimeCalculator.Hybrid.generated.cs", StringComparison.Ordinal)).SourceText.ToString();
        string gpu = generated.Single(s => s.HintName.EndsWith("PrimeCalculator.Gpu.generated.cs", StringComparison.Ordinal)).SourceText.ToString();

        cpu.Should().Contain("class PrimeCalculatorCpu");
        cpu.Should().Contain("=> \"CPU\"");
        cpu.Should().NotContain("#if");
        cpu.Should().NotContain("#endif");

        hybrid.Should().Contain("class PrimeCalculatorHybrid");
        hybrid.Should().Contain("=> \"HYBRID\"");
        hybrid.Should().NotContain("#if");
        hybrid.Should().NotContain("#endif");

        gpu.Should().Contain("class PrimeCalculatorGpu");
        gpu.Should().Contain("=> \"GPU\"");
        gpu.Should().NotContain("#if");
        gpu.Should().NotContain("#endif");
    }

    private static CSharpCompilation CreateCompilation(params string[] sources)
    {
        var parseOptions = new CSharpParseOptions(LanguageVersion.CSharp12);
        var trees = sources.Select((text, i) => CSharpSyntaxTree.ParseText(text, parseOptions, path: "Test" + i + ".cs"));

        var references = new[]
            {
                typeof(object),
                typeof(Attribute),
                typeof(Enumerable),
                typeof(System.Runtime.GCSettings),
            }
            .Select(static t => MetadataReference.CreateFromFile(t.Assembly.Location))
            .DistinctBy(static r => r.Display)
            .ToList();

        return CSharpCompilation.Create(
            assemblyName: "DeviceDependentGenerator.Tests.Compilation",
            syntaxTrees: trees,
            references: references,
            options: new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
    }
}
