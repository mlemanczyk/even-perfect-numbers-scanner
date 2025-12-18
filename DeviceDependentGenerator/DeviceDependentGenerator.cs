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

    private static readonly DiagnosticDescriptor EnumTypeMustBeEnum = new(
        id: "DDG001",
        title: "DeviceDependentTemplate requires an enum type",
        messageFormat: "DeviceDependentTemplate requires an enum type, but '{0}' is not an enum.",
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
        messageFormat: "Could not find template type '{0}' after preprocessing for device '{1}'.",
        category: "DeviceDependentGenerator",
        defaultSeverity: DiagnosticSeverity.Error,
        isEnabledByDefault: true);

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        IncrementalValuesProvider<TemplateInfo> templates = context.SyntaxProvider.ForAttributeWithMetadataName(
            fullyQualifiedMetadataName: DeviceDependentTemplateAttributeMetadataName,
            predicate: static (node, _) => node is TypeDeclarationSyntax,
            transform: static (attributeContext, cancellationToken) => TryGetTemplateInfo(attributeContext, cancellationToken))
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

    private static TemplateInfo? TryGetTemplateInfo(GeneratorAttributeSyntaxContext attributeContext, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        if (attributeContext.TargetSymbol is not INamedTypeSymbol templateSymbol)
        {
            return null;
        }

        if (attributeContext.Attributes.Length != 1)
        {
            return null;
        }

        AttributeData attributeData = attributeContext.Attributes[0];
        if (attributeData.ConstructorArguments.Length is not (1 or 2))
        {
            return null;
        }

        if (attributeData.ConstructorArguments[0].Value is not INamedTypeSymbol enumTypeSymbol)
        {
            return null;
        }

        string? suffix = null;
        if (attributeData.ConstructorArguments.Length == 2)
        {
            suffix = attributeData.ConstructorArguments[1].Value as string;
        }

        if (suffix is null && attributeData.NamedArguments.Length != 0)
        {
            foreach (KeyValuePair<string, TypedConstant> namedArgument in attributeData.NamedArguments)
            {
                if (string.Equals(namedArgument.Key, "suffix", StringComparison.OrdinalIgnoreCase))
                {
                    suffix = namedArgument.Value.Value as string;
                }
            }
        }

        if (enumTypeSymbol.TypeKind != TypeKind.Enum)
        {
            Location location = attributeContext.TargetNode.GetLocation();
            return TemplateInfo.CreateInvalidEnum(templateSymbol, enumTypeSymbol, location);
        }

        ImmutableArray<string> enumValues = enumTypeSymbol
            .GetMembers()
            .OfType<IFieldSymbol>()
            .Where(static f => !f.IsImplicitlyDeclared && f.HasConstantValue)
            .Select(static f => f.Name)
            .ToImmutableArray();

        Location attributeLocation = attributeContext.TargetNode.GetLocation();
        return TemplateInfo.CreateValid(templateSymbol, enumTypeSymbol, attributeLocation, enumValues, suffix);
    }

    private static void GenerateForTemplate(SourceProductionContext context, TemplateInfo template, Compilation compilation, GeneratorOptions options)
    {
        if (!template.Valid)
        {
            context.ReportDiagnostic(Diagnostic.Create(EnumTypeMustBeEnum, template.AttributeLocation, template.EnumType.ToDisplayString()));
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

        foreach (string enumValueName in template.EnumValues)
        {
            string deviceDefine = "DEVICE_" + enumValueName.ToUpperInvariant();
            CSharpParseOptions parseOptions = WithDeviceDefine(baseParseOptions, deviceDefine);

            SyntaxTree deviceTree = CSharpSyntaxTree.ParseText(
                fileText.ToString(),
                parseOptions,
                filePath,
                encoding: Encoding.UTF8);

            CompilationUnitSyntax deviceRoot = deviceTree.GetCompilationUnitRoot(context.CancellationToken);
            TypeDeclarationSyntax? deviceTemplateSyntax = FindTemplateDeclaration(deviceRoot, templateName);
            if (deviceTemplateSyntax is null)
            {
                context.ReportDiagnostic(Diagnostic.Create(TemplateNotFoundInDeviceParse, templateSyntax.Identifier.GetLocation(), templateName, enumValueName));
                continue;
            }

            ReportDeviceSpecificTemplateErrors(context, deviceTree);

            string generatedTypeName = baseName + enumValueName + template.Suffix;
            TypeDeclarationSyntax generatedTypeSyntax = TransformTemplateType(deviceTemplateSyntax, generatedTypeName);

            CompilationUnitSyntax generatedRoot = BuildGeneratedFile(
                externs: deviceRoot.Externs,
                usings: deviceRoot.Usings,
                containingNamespace: template.ContainingNamespace,
                typeDeclaration: generatedTypeSyntax);

            generatedRoot = (CompilationUnitSyntax)new ConditionalDirectiveAndDisabledTextStripper().Visit(generatedRoot)!;
            generatedRoot = generatedRoot.NormalizeWhitespace();

            string generatedFileName = baseName + "." + enumValueName + template.Suffix + ".generated.cs";
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

    private static CSharpParseOptions WithDeviceDefine(CSharpParseOptions baseParseOptions, string deviceDefine)
    {
        IEnumerable<string> symbols = baseParseOptions.PreprocessorSymbolNames;
        if (symbols.Contains(deviceDefine))
        {
            return baseParseOptions;
        }

        return baseParseOptions.WithPreprocessorSymbols(symbols.Concat(new[] { deviceDefine }));
    }

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
        TypeDeclarationSyntax withoutAttribute = RemoveDeviceDependentAttribute(templateType);
        return withoutAttribute.WithIdentifier(SyntaxFactory.Identifier(generatedTypeName));
    }

    private static TypeDeclarationSyntax RemoveDeviceDependentAttribute(TypeDeclarationSyntax typeDeclaration)
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
                if (IsDeviceDependentTemplateAttribute(attribute.Name))
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

    private static bool IsDeviceDependentTemplateAttribute(NameSyntax attributeName)
    {
        string text = attributeName.ToString();
        if (text.EndsWith("DeviceDependentTemplate", StringComparison.Ordinal))
        {
            return true;
        }

        return text.EndsWith("DeviceDependentTemplateAttribute", StringComparison.Ordinal);
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
            INamedTypeSymbol enumType,
            Location attributeLocation,
            ImmutableArray<string> enumValues,
            string? suffix,
            bool valid)
        {
            TemplateType = templateType;
            EnumType = enumType;
            AttributeLocation = attributeLocation;
            EnumValues = enumValues;
            Suffix = suffix ?? string.Empty;
            Valid = valid;
        }

        public INamedTypeSymbol TemplateType { get; }
        public INamedTypeSymbol EnumType { get; }
        public Location AttributeLocation { get; }
        public ImmutableArray<string> EnumValues { get; }
        public string Suffix { get; }
        public bool Valid { get; }

        public string Name => TemplateType.Name;

        public string ContainingNamespace => TemplateType.ContainingNamespace?.IsGlobalNamespace == true
            ? string.Empty
            : TemplateType.ContainingNamespace.ToDisplayString();

        public SyntaxReference? DeclaringSyntaxReference => TemplateType.DeclaringSyntaxReferences.FirstOrDefault();

        public static TemplateInfo CreateInvalidEnum(INamedTypeSymbol templateType, INamedTypeSymbol enumType, Location attributeLocation)
            => new(templateType, enumType, attributeLocation, enumValues: default, suffix: null, valid: false);

        public static TemplateInfo CreateValid(
            INamedTypeSymbol templateType,
            INamedTypeSymbol enumType,
            Location attributeLocation,
            ImmutableArray<string> enumValues,
            string? suffix)
            => new(templateType, enumType, attributeLocation, enumValues, suffix, valid: true);
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
