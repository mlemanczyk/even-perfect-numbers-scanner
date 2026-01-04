using System;
using System.IO;
using System.Text;
using System.Text.Json;
using FASTER.core;
using LightningDB;
using PerfectNumbers.Core.ByDivisor;

namespace PerfectNumbers.Core.Persistence;

/// <summary>
/// Persists the by-divisor class model using Microsoft.FASTER.Core with optional migration from LightningDB.
/// </summary>
public sealed class ByDivisorClassModelRepository : IDisposable
{
    private const string ModelKey = "bydivisor_model";
    private readonly FasterKV<string, string> _store;
    private readonly IDevice _logDevice;
    private readonly IDevice _objectLogDevice;
    private static readonly Encoding Utf8 = Encoding.UTF8;

    public string DatabasePath { get; }

    public static ByDivisorClassModelRepository Open(
        string directory,
        string repositoryFileName = "bydivisor_model.faster",
        string? lmdbPath = null,
        string? lmdbDirectory = null,
        bool deleteLmdbAfterMigration = true)
    {
        Directory.CreateDirectory(directory);
        string dbPath = Path.Combine(directory, repositoryFileName);
        var repo = new ByDivisorClassModelRepository(dbPath);
        string lmdbSource = lmdbPath ?? Path.ChangeExtension(dbPath, ".lmdb");
        if (!File.Exists(lmdbSource) && !Directory.Exists(lmdbSource) && !string.IsNullOrEmpty(lmdbDirectory))
        {
            lmdbSource = Path.Combine(lmdbDirectory, Path.GetFileName(Path.ChangeExtension(repositoryFileName, ".lmdb")));
        }
        repo.MigrateFromLmdb(lmdbSource, deleteLmdbAfterMigration);
        return repo;
    }

    private ByDivisorClassModelRepository(string path)
    {
        DatabasePath = path;
        _logDevice = Devices.CreateLogDevice(path + ".log");
        _objectLogDevice = Devices.CreateLogDevice(path + ".obj.log");
        _store = new FasterKV<string, string>(
            1L << 16,
            new LogSettings
            {
                LogDevice = _logDevice,
                ObjectLogDevice = _objectLogDevice,
            });
    }

    public void Save(ByDivisorClassModel model)
    {
        string json = JsonSerializer.Serialize(model, new JsonSerializerOptions { WriteIndented = true });
        using var session = _store.For(new SimpleFunctions<string, string>()).NewSession<SimpleFunctions<string, string>>();
        session.Upsert(ModelKey, json);
        session.CompletePending(true);
    }

    public bool TryLoad(out ByDivisorClassModel? model)
    {
        using var session = _store.For(new SimpleFunctions<string, string>()).NewSession<SimpleFunctions<string, string>>();
        if (session.Read(ModelKey, out string? json).Found && json is not null)
        {
            try
            {
                model = JsonSerializer.Deserialize<ByDivisorClassModel>(json);
                return model is not null;
            }
            catch
            {
                // ignore deserialization errors
            }
        }

        model = null;
        return false;
    }

    public void Commit()
    {
        _store.Log.FlushAndEvict(wait: true);
    }

    public void Dispose()
    {
        _store.Dispose();
        _logDevice.Dispose();
        _objectLogDevice.Dispose();
    }

    private void MigrateFromLmdb(string lmdbPath, bool deleteAfter)
    {
        try
        {
            if (!LmdbExists(lmdbPath))
            {
                Console.WriteLine($"No LMDB by-divisor model found at '{lmdbPath}', skipping migration.");
                return;
            }

            Console.WriteLine($"Migrating by-divisor model from LMDB at '{lmdbPath}' to FASTER...");
            using var env = new LightningEnvironment(lmdbPath) { MaxDatabases = 2 };
            env.Open();
            using var tx = env.BeginTransaction(TransactionBeginFlags.ReadOnly);
            var db = tx.OpenDatabase("bydivisor_model", new DatabaseConfiguration { Flags = DatabaseOpenFlags.None });
            if (tx.TryGet(db, Utf8.GetBytes(ModelKey), out byte[]? jsonBytes) && jsonBytes is not null)
            {
                try
                {
                    string json = Utf8.GetString(jsonBytes);
                    var session = _store.For(new SimpleFunctions<string, string>()).NewSession<SimpleFunctions<string, string>>();
                    session.Upsert(ModelKey, json);
                    session.CompletePending(true);
                    session.Dispose();
                }
                catch
                {
                    // ignore migration issues
                }
            }
            _store.Log.FlushAndEvict(wait: true);
            Console.WriteLine("Migration of by-divisor model to FASTER completed.");
        }
        catch
        {
            // ignore migration failures
        }

        if (deleteAfter)
        {
            Console.WriteLine($"Starting cleanup of LMDB by-divisor model at '{lmdbPath}'...");
            try
            {
                if (Directory.Exists(lmdbPath))
                {
                    Directory.Delete(lmdbPath, recursive: true);
                }
                else if (File.Exists(lmdbPath))
                {
                    File.Delete(lmdbPath);
                }
            }
            catch
            {
                // ignore cleanup failures
            }
        }
    }

    private static bool LmdbExists(string path)
    {
        if (Directory.Exists(path) || File.Exists(path))
        {
            return true;
        }

        string candidate = Path.Combine(path, "data.mdb");
        if (File.Exists(candidate))
        {
            return true;
        }

        string? directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(directory) && Directory.Exists(directory))
        {
            string sibling = Path.Combine(directory, "data.mdb");
            if (File.Exists(sibling))
            {
                return true;
            }
        }

        return false;
    }
}
