using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Numerics;
using System.Text;
using FASTER.core;

namespace PerfectNumbers.Core.Persistence;

/// <summary>
/// Stores BitTree frontier batches using Microsoft.FASTER.Core.
/// Each batch is a byte[] payload containing a sequence of serialized states.
/// Maintains a simple head/tail cursor stored inside the KV store.
/// </summary>
public sealed class BitTreeFrontierRepository : IDisposable
{
    private readonly FasterKV<string, byte[]> _store;
    private readonly IDevice _logDevice;
    private readonly IDevice _objectLogDevice;
    private long _headIndex;
    private long _tailIndex;
    private const string HeadKey = "__head__";
    private const string TailKey = "__tail__";

    public string DatabasePath { get; }

    public static BitTreeFrontierRepository Open(string directory, string repositoryFileName = "bittree.frontier.faster")
    {
        Directory.CreateDirectory(directory);
        string dbPath = Path.Combine(directory, repositoryFileName);
        return new BitTreeFrontierRepository(dbPath);
    }

    private BitTreeFrontierRepository(string path)
    {
        DatabasePath = path;
        _logDevice = Devices.CreateLogDevice(path + ".log");
        _objectLogDevice = Devices.CreateLogDevice(path + ".obj.log");
        _store = new FasterKV<string, byte[]>(
            1L << 20,
            new LogSettings
            {
                LogDevice = _logDevice,
                ObjectLogDevice = _objectLogDevice,
            });

        LoadCursors();
    }

    public void AppendBatch(ReadOnlySpan<byte> batch)
    {
        using var session = _store.For(new SimpleFunctions<string, byte[]>()).NewSession<SimpleFunctions<string, byte[]>>();
        string key = BuildChunkKey(_tailIndex);
        session.Upsert(key, batch.ToArray());
        _tailIndex++;
        PersistCursors(session);
        session.CompletePending(true);
		_store.Log.FlushAndEvict(true);
    }

    public bool TryDequeueBatch(int maxCount, out List<byte[]> batches)
    {
        batches = new List<byte[]>();
        using var session = _store.For(new SimpleFunctions<string, byte[]>()).NewSession<SimpleFunctions<string, byte[]>>();

        while (_headIndex < _tailIndex && batches.Count < maxCount)
        {
            string key = BuildChunkKey(_headIndex);
            if (session.Read(key, out byte[]? stored).Found && stored is not null)
            {
                batches.Add(stored);
                session.Delete(key);
            }

            _headIndex++;
        }


        if (batches.Count > 0)
        {
            PersistCursors(session);
            session.CompletePending(true);
			_store.Log.FlushAndEvict(true);
            return true;
        }

        session.CompletePending(true);
		_store.Log.FlushAndEvict(true);
        return false;
    }

    public void Dispose()
    {
        _store.Dispose();
        _logDevice.Dispose();
        _objectLogDevice.Dispose();
    }

    private void LoadCursors()
    {
        using var session = _store.For(new SimpleFunctions<string, byte[]>()).NewSession<SimpleFunctions<string, byte[]>>();
        _headIndex = ReadCursor(session, HeadKey);
        _tailIndex = ReadCursor(session, TailKey);
        session.CompletePending(true);
    }

    private static long ReadCursor(ClientSession<string, byte[], byte[], byte[], Empty, SimpleFunctions<string, byte[]>> session, string key)
    {
        if (session.Read(key, out byte[]? stored).Found && stored is not null)
        {
            string text = Encoding.UTF8.GetString(stored);
            if (long.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out long value))
            {
                return value;
            }
        }

        return 0;
    }

    private void PersistCursors(ClientSession<string, byte[], byte[], byte[], Empty, SimpleFunctions<string, byte[]>> session)
    {
        session.Upsert(HeadKey, Encoding.UTF8.GetBytes(_headIndex.ToString(CultureInfo.InvariantCulture)));
        session.Upsert(TailKey, Encoding.UTF8.GetBytes(_tailIndex.ToString(CultureInfo.InvariantCulture)));
    }

    private static string BuildChunkKey(long index) => $"chunk-{index}";
}
