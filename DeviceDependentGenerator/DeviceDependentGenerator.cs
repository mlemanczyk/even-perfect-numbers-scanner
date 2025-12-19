using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Text;

namespace PerfectNumbers.Core.Generators;

[Generator(LanguageNames.CSharp)]
public sealed class DeviceDependentGenerator : IIncrementalGenerator
{
    private const string DeviceDependentTemplateAttributeMetadataName = "PerfectNumbers.Core.DeviceDependentTemplateAttribute";
    private const string EnumDependentTemplateAttributeMetadataName = "PerfectNumbers.Core.EnumDependentTemplateAttribute";
    private const string NameSuffixAttributeMetadataName = "PerfectNumbers.Core.NameSuffixAttribute";

    private static readonly DiagnosticDescriptor TemplateEnumTypeMustBeEnum = new(
        id: "DDG001",
        title: "Template attribute requires an enum type",
        messageFormat: "Template attribute '{0}' requires an enum type, but '{1}' is not an enum.",
        category: "DeviceDependentGenerator",
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor TemplateMustBeTopLevel = new(
        id: "DDG002",
        title: "DeviceDependentTemplate must be top-level",
        messageFormat: "DeviceDependentTemplate can only be used on a top-level class/struct (not nested types).",
        category: "DeviceDependentGenerator",
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor TemplateNotFoundInDeviceParse = new(
        id: "DDG003",
        title: "Template not found after preprocessing",
        messageFormat: "Could not find template type '{0}' after preprocessing for combination '{1}'.",
        category: "DeviceDependentGenerator",
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    private static readonly DiagnosticDescriptor DuplicateSingleUseAttribute = new(
        id: "DDG004",
        title: "Template attribute used multiple times",
        messageFormat: "Attribute '{0}' can be applied only once to '{1}'.",
        category: "DeviceDependentGenerator",
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        IncrementalValuesProvider<TemplateInfo> templates = context.SyntaxProvider.CreateSyntaxProvider(
            predicate: static (node, _) => node is TypeDeclarationSyntax typeDecl && typeDecl.AttributeLists.Count > 0,
            transform: static (syntaxContext, cancellationToken) => TryGetTemplateInfo(syntaxContext, cancellationToken))
            .Where(static info => info is not null)!
            .Select(static (info, _) => info!);

        IncrementalValueProvider<GeneratorOptions> options = context.AnalyzerConfigOptionsProvider.Select(
            static (provider, _) => new GeneratorOptions(provider.GlobalOptions));

        IncrementalValueProvider<(Compilation Compilation, GeneratorOptions Options)> compilationAndOptions =
            context.CompilationProvider.Combine(options).Select(static (pair, _) => (pair.Left, pair.Right));

        context.RegisterSourceOutput(templates.Combine(compilationAndOptions), static (productionContext, pair) =>
        {
            GenerateForTemplate(productionContext, pair.Left, pair.Right.Compilation, pair.Right.Options);
        });
    }

    private static TemplateInfo? TryGetTemplateInfo(GeneratorSyntaxContext context, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        if (context.Node is not TypeDeclarationSyntax typeSyntax)
        {
            return null;
        }

        if (context.SemanticModel.GetDeclaredSymbol(typeSyntax, cancellationToken) is not INamedTypeSymbol templateSymbol)
        {
            return null;
        }

        return CollectTemplateInfo(templateSymbol, typeSyntax, cancellationToken);
    }

    private static void GenerateForTemplate(SourceProductionContext context, TemplateInfo template, Compilation compilation, GeneratorOptions options)
    {
        foreach (Diagnostic diagnostic in template.Diagnostics)
        {
            context.ReportDiagnostic(diagnostic);
        }

        if (!template.Valid || !template.HasAnyTemplateAttribute)
        {
            return;
        }

        if (template.DeclaringSyntaxReference?.GetSyntax(context.CancellationToken) is not TypeDeclarationSyntax templateSyntax)
        {
            return;
        }

        if (!IsTopLevel(templateSyntax))
        {
            context.ReportDiagnostic(Diagnostic.Create(TemplateMustBeTopLevel, templateSyntax.Identifier.GetLocation()));
            return;
        }

        string templateName = template.Name;
        string baseName = templateName.EndsWith("Template", StringComparison.Ordinal)
            ? templateName.Substring(0, templateName.Length - "Template".Length)
            : templateName;

        SourceText fileText = templateSyntax.SyntaxTree.GetText(context.CancellationToken);
        string filePath = templateSyntax.SyntaxTree.FilePath;
        CSharpParseOptions baseParseOptions = (CSharpParseOptions)templateSyntax.SyntaxTree.Options;
        SyntaxTree originalTree = templateSyntax.SyntaxTree;

        foreach (TemplateVariant variant in template.ExpandVariants())
        {
            CSharpParseOptions parseOptions = WithDefines(baseParseOptions, variant.PreprocessorSymbols);

            SyntaxTree deviceTree = CSharpSyntaxTree.ParseText(
                fileText.ToString(),
                parseOptions,
                filePath,
                encoding: Encoding.UTF8);

            CompilationUnitSyntax deviceRoot = deviceTree.GetCompilationUnitRoot(context.CancellationToken);
            TypeDeclarationSyntax? deviceTemplateSyntax = FindTemplateDeclaration(deviceRoot, templateName);
            if (deviceTemplateSyntax is null)
            {
                context.ReportDiagnostic(Diagnostic.Create(
                    TemplateNotFoundInDeviceParse,
                    templateSyntax.Identifier.GetLocation(),
                    templateName,
                    variant.CombinationDisplay));
                continue;
            }

            ReportDeviceSpecificTemplateErrors(context, deviceTree);

            string generatedTypeName = baseName + variant.NameSuffix + template.NameSuffix;
            TypeDeclarationSyntax generatedTypeSyntax = TransformTemplateType(deviceTemplateSyntax, generatedTypeName);

            CompilationUnitSyntax generatedRoot = BuildGeneratedFile(
                externs: deviceRoot.Externs,
                usings: deviceRoot.Usings,
                containingNamespace: template.ContainingNamespace,
                typeDeclaration: generatedTypeSyntax);

            generatedRoot = (CompilationUnitSyntax)new ConditionalDirectiveAndDisabledTextStripper().Visit(generatedRoot)!;
            generatedRoot = generatedRoot.NormalizeWhitespace();

            string generatedFileName = baseName + "." + variant.FileSuffix + template.NameSuffix + ".generated.cs";
            string hintName = options.BuildHintName(filePath, generatedFileName);
            context.AddSource(
                hintName,
                SourceText.From("// <auto-generated/>\n#nullable enable\n" + generatedRoot.ToFullString(), Encoding.UTF8));
        }
    }

    private static void ReportDeviceSpecificTemplateErrors(SourceProductionContext context, SyntaxTree deviceTree)
    {
        foreach (Diagnostic diagnostic in deviceTree.GetDiagnostics(context.CancellationToken))
        {
            if (diagnostic.Severity == DiagnosticSeverity.Error)
            {
                context.ReportDiagnostic(diagnostic);
            }
        }
    }

    private static bool IsTopLevel(TypeDeclarationSyntax typeDeclaration)
        => typeDeclaration.Parent is NamespaceDeclarationSyntax
            || typeDeclaration.Parent is FileScopedNamespaceDeclarationSyntax
            || typeDeclaration.Parent is CompilationUnitSyntax;

    private static CSharpParseOptions WithDefines(CSharpParseOptions baseParseOptions, IEnumerable<string> defines)
        => baseParseOptions.WithPreprocessorSymbols(baseParseOptions.PreprocessorSymbolNames.Concat(defines).Distinct());

    private static TypeDeclarationSyntax? FindTemplateDeclaration(CompilationUnitSyntax root, string templateName)
    {
        foreach (TypeDeclarationSyntax typeDecl in root.DescendantNodes().OfType<TypeDeclarationSyntax>())
        {
            if (string.Equals(typeDecl.Identifier.ValueText, templateName, StringComparison.Ordinal))
            {
                return typeDecl;
            }
        }

        return null;
    }

    private static TypeDeclarationSyntax TransformTemplateType(TypeDeclarationSyntax templateType, string generatedTypeName)
    {
        TypeDeclarationSyntax withoutAttribute = RemoveTemplateAttributes(templateType);
        TypeDeclarationSyntax renamed = (TypeDeclarationSyntax)new SelfIdentifierRewriter(templateType.Identifier.ValueText, generatedTypeName).Visit(withoutAttribute)!;
        return renamed.WithIdentifier(SyntaxFactory.Identifier(generatedTypeName));
    }

    private static TypeDeclarationSyntax RemoveTemplateAttributes(TypeDeclarationSyntax typeDeclaration)
    {
        if (typeDeclaration.AttributeLists.Count == 0)
        {
            return typeDeclaration;
        }

        var newLists = new List<AttributeListSyntax>(typeDeclaration.AttributeLists.Count);
        foreach (AttributeListSyntax list in typeDeclaration.AttributeLists)
        {
            SeparatedSyntaxList<AttributeSyntax> attributes = list.Attributes;
            var newAttributes = new List<AttributeSyntax>(attributes.Count);
            foreach (AttributeSyntax attribute in attributes)
            {
                if (IsTemplateAttribute(attribute.Name))
                {
                    continue;
                }

                newAttributes.Add(attribute);
            }

            if (newAttributes.Count == 0)
            {
                continue;
            }

            newLists.Add(list.WithAttributes(SyntaxFactory.SeparatedList(newAttributes)));
        }

        return typeDeclaration.WithAttributeLists(SyntaxFactory.List(newLists));
    }

    private sealed class SelfIdentifierRewriter : CSharpSyntaxRewriter
    {
        private readonly string templateName;
        private readonly string generatedName;

        public SelfIdentifierRewriter(string templateName, string generatedName)
        {
            this.templateName = templateName;
            this.generatedName = generatedName;
        }

        public override SyntaxToken VisitToken(SyntaxToken token)
        {
            if (token.IsKind(SyntaxKind.IdentifierToken) && string.Equals(token.ValueText, templateName, StringComparison.Ordinal))
            {
                return SyntaxFactory.Identifier(token.LeadingTrivia, generatedName, token.TrailingTrivia);
            }

            return base.VisitToken(token);
        }
    }

    private static bool IsTemplateAttribute(NameSyntax attributeName)
    {
        string text = attributeName.ToString();
        return text.EndsWith("DeviceDependentTemplate", StringComparison.Ordinal)
            || text.EndsWith("DeviceDependentTemplateAttribute", StringComparison.Ordinal)
            || text.EndsWith("EnumDependentTemplate", StringComparison.Ordinal)
            || text.EndsWith("EnumDependentTemplateAttribute", StringComparison.Ordinal)
            || text.EndsWith("NameSuffix", StringComparison.Ordinal)
            || text.EndsWith("NameSuffixAttribute", StringComparison.Ordinal);
    }

    private static CompilationUnitSyntax BuildGeneratedFile(
        SyntaxList<ExternAliasDirectiveSyntax> externs,
        SyntaxList<UsingDirectiveSyntax> usings,
        string containingNamespace,
        TypeDeclarationSyntax typeDeclaration)
    {
        MemberDeclarationSyntax member;
        if (string.IsNullOrEmpty(containingNamespace))
        {
            member = typeDeclaration;
        }
        else
        {
            member = SyntaxFactory
                .FileScopedNamespaceDeclaration(SyntaxFactory.ParseName(containingNamespace))
                .WithMembers(SyntaxFactory.SingletonList<MemberDeclarationSyntax>(typeDeclaration));
        }

        return SyntaxFactory
            .CompilationUnit()
            .WithExterns(externs)
            .WithUsings(usings)
            .WithMembers(SyntaxFactory.SingletonList(member));
    }

    private sealed class TemplateInfo
    {
        private TemplateInfo(
            INamedTypeSymbol templateType,
            ImmutableArray<EnumTemplateInfo> enumAttributes,
            DeviceTemplateInfo? deviceAttribute,
            string nameSuffix,
            ImmutableArray<Diagnostic> diagnostics)
        {
            TemplateType = templateType;
            EnumAttributes = enumAttributes;
            DeviceAttribute = deviceAttribute;
            NameSuffix = nameSuffix;
            Diagnostics = diagnostics;
        }

        public INamedTypeSymbol TemplateType { get; }
        public ImmutableArray<EnumTemplateInfo> EnumAttributes { get; }
        public DeviceTemplateInfo? DeviceAttribute { get; }
        public string NameSuffix { get; }
        public ImmutableArray<Diagnostic> Diagnostics { get; }

        public bool Valid => Diagnostics.IsDefaultOrEmpty;
        public bool HasAnyTemplateAttribute => EnumAttributes.Length > 0 || DeviceAttribute is not null;

        public string Name => TemplateType.Name;

        public string ContainingNamespace => TemplateType.ContainingNamespace?.IsGlobalNamespace == true
            ? string.Empty
            : TemplateType.ContainingNamespace.ToDisplayString();

        public SyntaxReference? DeclaringSyntaxReference => TemplateType.DeclaringSyntaxReferences.FirstOrDefault();

        public IEnumerable<TemplateVariant> ExpandVariants()
        {
            var combinations = ImmutableArray.Create(new TemplateVariantBuilder());

            foreach (EnumTemplateInfo enumInfo in EnumAttributes)
            {
                combinations = ExpandEnum(combinations, enumInfo);
            }

            bool hasEnumAttributes = EnumAttributes.Length > 0;

            if (DeviceAttribute is not null)
            {
                combinations = ExpandDevice(combinations, DeviceAttribute.Value, hasEnumAttributes);
            }

            foreach (TemplateVariantBuilder builder in combinations)
            {
                yield return new TemplateVariant(builder.Symbols, builder.NameSuffix);
            }
        }

        private static ImmutableArray<TemplateVariantBuilder> ExpandEnum(
            ImmutableArray<TemplateVariantBuilder> current,
            EnumTemplateInfo enumInfo)
        {
            var builder = ImmutableArray.CreateBuilder<TemplateVariantBuilder>(current.Length * enumInfo.Values.Length);
            foreach (TemplateVariantBuilder existing in current)
            {
                foreach (string value in enumInfo.Values)
                {
                    builder.Add(existing.Append(
                        symbol: $"{enumInfo.EnumType.Name}_{value}",
                        namePart: $"For{value}{enumInfo.EnumType.Name}"));
                }
            }

            return builder.ToImmutable();
        }

        private static ImmutableArray<TemplateVariantBuilder> ExpandDevice(
            ImmutableArray<TemplateVariantBuilder> current,
            DeviceTemplateInfo deviceInfo,
            bool prefixWithFor)
        {
            var builder = ImmutableArray.CreateBuilder<TemplateVariantBuilder>(current.Length * deviceInfo.Values.Length);
            foreach (TemplateVariantBuilder existing in current)
            {
                foreach (string value in deviceInfo.Values)
                {
                    string namePart = prefixWithFor ? $"For{value}" : value;
                    builder.Add(existing.Append(
                        symbol: "DEVICE_" + value.ToUpperInvariant(),
                        namePart: namePart));
                }
            }

            return builder.ToImmutable();
        }

        public static TemplateInfo Create(
            INamedTypeSymbol templateType,
            ImmutableArray<EnumTemplateInfo> enumAttributes,
            DeviceTemplateInfo? deviceAttribute,
            string? nameSuffix,
            ImmutableArray<Diagnostic> diagnostics)
            => new(
                templateType,
                enumAttributes,
                deviceAttribute,
                nameSuffix ?? string.Empty,
                diagnostics);
    }

    private readonly struct EnumTemplateInfo
    {
        public EnumTemplateInfo(INamedTypeSymbol enumType, ImmutableArray<string> values, Location attributeLocation)
        {
            EnumType = enumType;
            Values = values;
            AttributeLocation = attributeLocation;
        }

        public INamedTypeSymbol EnumType { get; }
        public ImmutableArray<string> Values { get; }
        public Location AttributeLocation { get; }
    }

    private readonly struct DeviceTemplateInfo
    {
        public DeviceTemplateInfo(INamedTypeSymbol enumType, ImmutableArray<string> values, Location attributeLocation)
        {
            EnumType = enumType;
            Values = values;
            AttributeLocation = attributeLocation;
        }

        public INamedTypeSymbol EnumType { get; }
        public ImmutableArray<string> Values { get; }
        public Location AttributeLocation { get; }
    }

    private readonly struct TemplateVariant
    {
        public TemplateVariant(ImmutableArray<string> preprocessorSymbols, string nameSuffix)
        {
            PreprocessorSymbols = preprocessorSymbols;
            NameSuffix = nameSuffix;
        }

        public ImmutableArray<string> PreprocessorSymbols { get; }
        public string NameSuffix { get; }
        public string FileSuffix => NameSuffix;
        public string CombinationDisplay => string.Join(", ", PreprocessorSymbols);
    }

    private sealed class TemplateVariantBuilder
    {
        public TemplateVariantBuilder()
        {
            Symbols = ImmutableArray<string>.Empty;
            NameSuffix = string.Empty;
        }

        private TemplateVariantBuilder(ImmutableArray<string> symbols, string nameSuffix)
        {
            Symbols = symbols;
            NameSuffix = nameSuffix;
        }

        public ImmutableArray<string> Symbols { get; }
        public string NameSuffix { get; }

        public TemplateVariantBuilder Append(string symbol, string namePart)
        {
            return new TemplateVariantBuilder(Symbols.Add(symbol), NameSuffix + namePart);
        }
    }

    private static TemplateInfo? CollectTemplateInfo(INamedTypeSymbol templateSymbol, TypeDeclarationSyntax typeSyntax, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        ImmutableArray<AttributeData> attributes = templateSymbol.GetAttributes();
        var enumAttributes = new List<EnumTemplateInfo>();
        DeviceTemplateInfo? deviceAttribute = null;
        string? nameSuffix = null;
        var diagnostics = ImmutableArray.CreateBuilder<Diagnostic>();

        AttributeData? nameSuffixAttribute = null;
        AttributeData? deviceTemplateAttribute = null;
        foreach (AttributeData attribute in attributes)
        {
            string? metadataName = attribute.AttributeClass?.ToDisplayString();
            if (metadataName is null)
            {
                continue;
            }

            if (string.Equals(metadataName, EnumDependentTemplateAttributeMetadataName, StringComparison.Ordinal))
            {
                enumAttributes.AddRange(CreateEnumTemplateInfo(attribute, TemplateEnumTypeMustBeEnum, diagnostics));
            }
            else if (string.Equals(metadataName, DeviceDependentTemplateAttributeMetadataName, StringComparison.Ordinal))
            {
                if (deviceTemplateAttribute is not null)
                {
                    diagnostics.Add(Diagnostic.Create(
                        DuplicateSingleUseAttribute,
                        attribute.ApplicationSyntaxReference?.GetSyntax(cancellationToken)?.GetLocation() ?? typeSyntax.GetLocation(),
                        DeviceDependentTemplateAttributeMetadataName,
                        templateSymbol.Name));
                    continue;
                }

                deviceTemplateAttribute = attribute;
            }
            else if (string.Equals(metadataName, NameSuffixAttributeMetadataName, StringComparison.Ordinal))
            {
                if (nameSuffixAttribute is not null)
                {
                    diagnostics.Add(Diagnostic.Create(
                        DuplicateSingleUseAttribute,
                        attribute.ApplicationSyntaxReference?.GetSyntax(cancellationToken)?.GetLocation() ?? typeSyntax.GetLocation(),
                        NameSuffixAttributeMetadataName,
                        templateSymbol.Name));
                    continue;
                }

                nameSuffixAttribute = attribute;
            }
        }

        if (deviceTemplateAttribute is not null)
        {
            var deviceInfo = CreateEnumTemplateInfo(deviceTemplateAttribute, TemplateEnumTypeMustBeEnum, diagnostics).FirstOrDefault();
            if (deviceInfo.EnumType is not null)
            {
                deviceAttribute = new DeviceTemplateInfo(deviceInfo.EnumType, deviceInfo.Values, deviceInfo.AttributeLocation);
            }
        }

        if (nameSuffixAttribute is not null)
        {
            nameSuffix = ExtractSuffix(nameSuffixAttribute);
        }

        if (!enumAttributes.Any() && deviceAttribute is null && nameSuffixAttribute is null)
        {
            return null;
        }

        ImmutableArray<EnumTemplateInfo> orderedEnums = enumAttributes
            .OrderBy(static e => e.AttributeLocation.SourceSpan.Start)
            .ToImmutableArray();

        return TemplateInfo.Create(
            templateSymbol,
            orderedEnums,
            deviceAttribute,
            nameSuffix,
            diagnostics.ToImmutable());
    }

    private static IEnumerable<EnumTemplateInfo> CreateEnumTemplateInfo(
        AttributeData attribute,
        DiagnosticDescriptor diagnosticDescriptor,
        ImmutableArray<Diagnostic>.Builder diagnostics)
    {
        if (attribute.ConstructorArguments.Length < 1)
        {
            return Array.Empty<EnumTemplateInfo>();
        }

        if (attribute.ConstructorArguments[0].Value is not INamedTypeSymbol enumTypeSymbol)
        {
            return Array.Empty<EnumTemplateInfo>();
        }

        if (enumTypeSymbol.TypeKind != TypeKind.Enum)
        {
            Location location = attribute.ApplicationSyntaxReference?.GetSyntax().GetLocation() ?? Location.None;
            diagnostics.Add(Diagnostic.Create(diagnosticDescriptor, location, attribute.AttributeClass?.Name ?? "TemplateAttribute", enumTypeSymbol.ToDisplayString()));
            return Array.Empty<EnumTemplateInfo>();
        }

        ImmutableArray<string> enumValues = enumTypeSymbol
            .GetMembers()
            .OfType<IFieldSymbol>()
            .Where(static f => !f.IsImplicitlyDeclared && f.HasConstantValue)
            .Select(static f => f.Name)
            .ToImmutableArray();

        Location attributeLocation = attribute.ApplicationSyntaxReference?.GetSyntax().GetLocation() ?? Location.None;
        return new[] { new EnumTemplateInfo(enumTypeSymbol, enumValues, attributeLocation) };
    }

    private static string? ExtractSuffix(AttributeData attributeData)
    {
        if (attributeData.ConstructorArguments.Length > 0)
        {
            return attributeData.ConstructorArguments[0].Value as string;
        }

        foreach (KeyValuePair<string, TypedConstant> argument in attributeData.NamedArguments)
        {
            if (string.Equals(argument.Key, "suffix", StringComparison.OrdinalIgnoreCase))
            {
                return argument.Value.Value as string;
            }
        }

        return null;
    }

    private readonly struct GeneratorOptions
    {
        private readonly AnalyzerConfigOptions globalOptions;

        public GeneratorOptions(AnalyzerConfigOptions globalOptions)
        {
            this.globalOptions = globalOptions;
        }

        public string BuildHintName(string templateFilePath, string generatedFileName)
        {
            if (!globalOptions.TryGetValue("build_property.ProjectDir", out string? projectDir) || string.IsNullOrWhiteSpace(projectDir))
            {
                return generatedFileName;
            }

            string? templateDirectory = Path.GetDirectoryName(templateFilePath);
            if (string.IsNullOrWhiteSpace(templateDirectory))
            {
                return generatedFileName;
            }

            string relativeDirectory = GetRelativePath(projectDir, templateDirectory);
            if (relativeDirectory is "." or "")
            {
                return generatedFileName;
            }

            if (relativeDirectory.StartsWith("..", StringComparison.Ordinal))
            {
                return generatedFileName;
            }

            relativeDirectory = relativeDirectory.Replace('\\', '/');
            return relativeDirectory.TrimEnd('/') + "/" + generatedFileName;
        }

        private static string GetRelativePath(string baseDirectory, string targetDirectory)
        {
            if (string.IsNullOrWhiteSpace(baseDirectory) || string.IsNullOrWhiteSpace(targetDirectory))
            {
                return targetDirectory;
            }

            if (!baseDirectory.EndsWith(Path.DirectorySeparatorChar.ToString(), StringComparison.Ordinal)
                && !baseDirectory.EndsWith(Path.AltDirectorySeparatorChar.ToString(), StringComparison.Ordinal))
            {
                baseDirectory += Path.DirectorySeparatorChar;
            }

            var baseUri = new Uri(baseDirectory, UriKind.Absolute);
            var targetUri = new Uri(targetDirectory, UriKind.Absolute);
            string relative = Uri.UnescapeDataString(baseUri.MakeRelativeUri(targetUri).ToString());
            return relative.Replace('/', Path.DirectorySeparatorChar);
        }
    }

    private sealed class ConditionalDirectiveAndDisabledTextStripper : CSharpSyntaxRewriter
    {
        public override SyntaxToken VisitToken(SyntaxToken token)
            => token.WithLeadingTrivia(Filter(token.LeadingTrivia)).WithTrailingTrivia(Filter(token.TrailingTrivia));

        private static SyntaxTriviaList Filter(SyntaxTriviaList list)
        {
            var builder = ImmutableArray.CreateBuilder<SyntaxTrivia>(list.Count);
            foreach (SyntaxTrivia trivia in list)
            {
                SyntaxKind kind = trivia.Kind();
                if (kind is SyntaxKind.DisabledTextTrivia
                    or SyntaxKind.IfDirectiveTrivia
                    or SyntaxKind.ElifDirectiveTrivia
                    or SyntaxKind.ElseDirectiveTrivia
                    or SyntaxKind.EndIfDirectiveTrivia)
                {
                    continue;
                }

                builder.Add(trivia);
            }

            return SyntaxFactory.TriviaList(builder.ToImmutable());
        }
    }
}
