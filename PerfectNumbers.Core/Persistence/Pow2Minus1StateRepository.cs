using System;
using System.Globalization;
using System.IO;
using System.Text;
using FASTER.core;
using LightningDB;

namespace PerfectNumbers.Core.Persistence;

/// <summary>
/// Tracks primes for which the pow2-1 divisor check was completed, backed by Microsoft.FASTER.Core.
/// Supports one-time migration from LightningDB.
/// </summary>
public sealed class Pow2Minus1StateRepository : IDisposable
{
    private readonly FasterKV<string, string> _store;
    private readonly IDevice _logDevice;
    private readonly IDevice _objectLogDevice;
    private static readonly Encoding Utf8 = Encoding.UTF8;

    public string DatabasePath { get; }

    public static Pow2Minus1StateRepository Open(
        string databasePath,
        string lmdbPath = "pow2minus1.lmdb",
        string? lmdbDirectory = null,
        string lmdbDatabaseName = "pow2minus1",
        bool deleteLmdbAfterMigration = true)
    {
        string directory = Path.GetDirectoryName(databasePath) ?? ".";
        Directory.CreateDirectory(directory);

        var repo = new Pow2Minus1StateRepository(databasePath);

        if (!File.Exists(lmdbPath) && !Directory.Exists(lmdbPath) && !string.IsNullOrEmpty(lmdbDirectory))
        {
            lmdbPath = Path.Combine(lmdbDirectory, Path.GetFileName(lmdbPath));
        }

        repo.MigrateFromLmdb(lmdbPath, lmdbDatabaseName, deleteLmdbAfterMigration);
        return repo;
    }

    private Pow2Minus1StateRepository(string path)
    {
        DatabasePath = path;
        _logDevice = Devices.CreateLogDevice(path + ".log");
        _objectLogDevice = Devices.CreateLogDevice(path + ".obj.log");
        _store = new FasterKV<string, string>(
            1L << 18,
            new LogSettings
            {
                LogDevice = _logDevice,
                ObjectLogDevice = _objectLogDevice,
            });
    }

    public void MarkChecked(ulong prime)
    {
        using var session = _store.For(new SimpleFunctions<string, string>()).NewSession<SimpleFunctions<string, string>>();
        string key = prime.ToString(CultureInfo.InvariantCulture);
        session.Upsert(key, "checked");
        session.CompletePending(true);
    }

    public bool IsChecked(ulong prime)
    {
        using var session = _store.For(new SimpleFunctions<string, string>()).NewSession<SimpleFunctions<string, string>>();
        string key = prime.ToString(CultureInfo.InvariantCulture);
        var status = session.Read(key, out _);
        session.CompletePending(true);
        return status.Found;
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

    private void MigrateFromLmdb(string lmdbPath, string databaseName, bool deleteAfter)
    {
        try
        {
            if (!LmdbExists(lmdbPath))
            {
                Console.WriteLine($"No LMDB pow2minus1 state found at '{lmdbPath}', skipping migration.");
                return;
            }

            Console.WriteLine($"Migrating pow2minus1 state from LMDB at '{lmdbPath}' to FASTER...");
            using var env = new LightningEnvironment(lmdbPath) { MaxDatabases = 2 };
            env.Open();
            using var tx = env.BeginTransaction(TransactionBeginFlags.ReadOnly);
            var db = tx.OpenDatabase(databaseName, new DatabaseConfiguration { Flags = DatabaseOpenFlags.None });
            using var cursor = tx.CreateCursor(db);
            foreach (var entry in cursor.AsEnumerable())
            {
                string key = Utf8.GetString(entry.Item1.CopyToNewArray());
                if (ulong.TryParse(key, NumberStyles.None, CultureInfo.InvariantCulture, out ulong prime))
                {
                    MarkChecked(prime);
                }
            }
            _store.Log.FlushAndEvict(wait: true);
            Console.WriteLine("Migration of pow2minus1 state to FASTER completed.");
        }
        catch
        {
            // ignore migration issues
        }

        if (deleteAfter)
        {
            Console.WriteLine($"Starting cleanup of LMDB pow2minus1 state at '{lmdbPath}'...");
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
