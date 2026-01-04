using System;
using System.Globalization;
using System.IO;
using System.Text;
using FASTER.core;

namespace PerfectNumbers.Core.Persistence;

/// <summary>
/// Stores per-prime BitContradiction solver state (arbitrary string payload) using Microsoft.FASTER.Core.
/// No LMDB migration is performed for this repository.
/// </summary>
public sealed class BitContradictionStateRepository : IDisposable
{
    private readonly FasterKV<string, string> _store;
    private readonly IDevice _logDevice;
    private readonly IDevice _objectLogDevice;
    private static readonly Encoding Utf8 = Encoding.UTF8;

    public string DatabasePath { get; }

    public static BitContradictionStateRepository Open(string directory, string repositoryFileName = "bitcontradiction.solver.faster")
    {
        Directory.CreateDirectory(directory);
        string dbPath = Path.Combine(directory, repositoryFileName);
        return new BitContradictionStateRepository(dbPath);
    }

    private BitContradictionStateRepository(string path)
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

    public void Upsert(ulong prime, string state)
    {
        using var session = _store.For(new SimpleFunctions<string, string>()).NewSession<SimpleFunctions<string, string>>();
        string keyString = prime.ToString(CultureInfo.InvariantCulture);
        session.Upsert(keyString, state);
        session.CompletePending(true);
    }

    public bool TryGet(ulong prime, out string? state)
    {
        using var session = _store.For(new SimpleFunctions<string, string>()).NewSession<SimpleFunctions<string, string>>();
        string keyString = prime.ToString(CultureInfo.InvariantCulture);
        if (session.Read(keyString, out string? stored).Found)
        {
            state = stored;
            return true;
        }

        state = null;
        return false;
    }

    public void Dispose()
    {
        _store.Dispose();
        _logDevice.Dispose();
        _objectLogDevice.Dispose();
    }
}
