using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Numerics;
using System.Text;
using FASTER.core;
using LightningDB;

namespace PerfectNumbers.Core.Persistence;

/// <summary>
/// Stores last-checked k values (BigInteger) per prime using Microsoft.FASTER.Core.
/// Supports one-time migration from an existing LightningDB (LMDB) store.
/// </summary>
public sealed class KStateRepository : IDisposable
{
    private readonly FasterKV<string, string> _store;
    private readonly IDevice _logDevice;
    private readonly IDevice _objectLogDevice;
    private static readonly Encoding Utf8 = Encoding.UTF8;

    public string DatabasePath { get; }

    public static KStateRepository Open(
        string directory,
        string repositoryFileName = "kstate.faster",
        string? lmdbPath = null,
        string? lmdbDirectory = null,
        string lmdbDatabaseName = "kstate",
        bool deleteLmdbAfterMigration = true)
    {
        Directory.CreateDirectory(directory);
        string dbPath = Path.Combine(directory, repositoryFileName);
        var repo = new KStateRepository(dbPath);

        string lmdbSource = lmdbPath ?? Path.ChangeExtension(dbPath, ".lmdb");
        if (!File.Exists(lmdbSource) && !Directory.Exists(lmdbSource) && !string.IsNullOrEmpty(lmdbDirectory))
        {
            lmdbSource = Path.Combine(lmdbDirectory, Path.GetFileName(Path.ChangeExtension(repositoryFileName, ".lmdb")));
        }

        repo.MigrateFromLmdb(lmdbSource, lmdbDatabaseName, deleteLmdbAfterMigration);
        return repo;
    }

    private KStateRepository(string path)
    {
        DatabasePath = path;
        _logDevice = Devices.CreateLogDevice(path + ".log");
        _objectLogDevice = Devices.CreateLogDevice(path + ".obj.log");
        _store = new FasterKV<string, string>(
            1L << 20,
            new LogSettings
            {
                LogDevice = _logDevice,
                ObjectLogDevice = _objectLogDevice,
            });
    }

	private volatile int _flushCount;

    public void Upsert(ulong prime, BigInteger k)
    {
        using var session = _store.For(new SimpleFunctions<string, string>()).NewSession<SimpleFunctions<string, string>>();
        string keyString = prime.ToString(CultureInfo.InvariantCulture);
        string valueString = k.ToString(CultureInfo.InvariantCulture);
        session.Upsert(keyString, valueString);
        session.CompletePending(true);
		var flushCount = ++_flushCount;
		if (flushCount >= PerfectNumberConstants.ByDivisorStateSaveInterval)
		{
			_flushCount = 0;
			_store.Log.FlushAndEvict(wait: true);
		}
		
    }

    public bool TryGet(ulong prime, out BigInteger value)
    {
        using var session = _store.For(new SimpleFunctions<string, string>()).NewSession<SimpleFunctions<string, string>>();
        string keyString = prime.ToString(CultureInfo.InvariantCulture);
        if (session.Read(keyString, out string? stored).Found &&
            stored is not null &&
            BigInteger.TryParse(stored, NumberStyles.Integer, CultureInfo.InvariantCulture, out value))
        {
            return true;
        }

        value = BigInteger.Zero;
        return false;
    }

    public void SeedFromStates(IEnumerable<(ulong Prime, BigInteger K)> states)
    {
        foreach (var (prime, k) in states)
        {
            Upsert(prime, k);
        }
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
                Console.WriteLine($"No LMDB k-state found at '{lmdbPath}', skipping migration.");
                return;
            }

            Console.WriteLine($"Migrating k-state from LMDB at '{lmdbPath}' to FASTER...");
            using var env = new LightningEnvironment(lmdbPath) { MaxDatabases = 2 };
            env.Open();
            using var tx = env.BeginTransaction(TransactionBeginFlags.ReadOnly);
            var db = tx.OpenDatabase(databaseName, new DatabaseConfiguration { Flags = DatabaseOpenFlags.None });
            using var cursor = tx.CreateCursor(db);
            foreach (var entry in cursor.AsEnumerable())
            {
                string keyString = Utf8.GetString(entry.Item1.CopyToNewArray());
                string valueString = Utf8.GetString(entry.Item2.CopyToNewArray());
                if (ulong.TryParse(keyString, NumberStyles.None, CultureInfo.InvariantCulture, out ulong prime) &&
                    BigInteger.TryParse(valueString, NumberStyles.Integer, CultureInfo.InvariantCulture, out BigInteger k) &&
                    k > BigInteger.Zero)
                {
                    Upsert(prime, k);
                }
            }
            _store.Log.FlushAndEvict(wait: true);
            Console.WriteLine("Migration of k-state to FASTER completed.");
        }
        catch
        {
            // Ignore migration failures; repo will simply start empty.
        }

        if (deleteAfter)
        {
            Console.WriteLine($"Starting cleanup of LMDB k-state at '{lmdbPath}'...");
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
