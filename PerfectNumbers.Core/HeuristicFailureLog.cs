using System.Collections.Concurrent;
using System.Globalization;
using System.Text;

namespace PerfectNumbers.Core
{
    internal static class HeuristicFailureLog
    {
        private const string LogFileName = "prime-order-heuristic-fallbacks.log";
        private const int FlushThreshold = 4096;
        private const int MaxBufferedEntries = 16;
        private static readonly ThreadLocal<StringBuilder?> s_threadBuilder = new(() => StringBuilderPool.Rent(), trackAllValues: true);
        private static readonly ThreadLocal<int> s_threadEntryCount = new(() => 0, trackAllValues: true);
        private static readonly object s_fileLock = new();
        private static readonly string s_logPath = Path.Combine(AppContext.BaseDirectory, LogFileName);
        private static readonly UTF8Encoding s_utf8NoBom = new(false);

        static HeuristicFailureLog()
        {
            AppDomain.CurrentDomain.ProcessExit += static (_, _) => FlushAll(force: true);
        }

        public static void Record(ulong prime, ulong? candidateOrder, HeuristicFailureReason reason)
        {
            string primeText = prime.ToString(CultureInfo.InvariantCulture);
            string? candidateText = candidateOrder.HasValue ? candidateOrder.Value.ToString(CultureInfo.InvariantCulture) : null;
            RecordInternal(primeText, candidateText, reason);
        }

        public static void Record(UInt128 prime, UInt128? candidateOrder, HeuristicFailureReason reason)
        {
            string primeText = prime.ToString(CultureInfo.InvariantCulture);
            string? candidateText = candidateOrder.HasValue ? candidateOrder.Value.ToString(CultureInfo.InvariantCulture) : null;
            RecordInternal(primeText, candidateText, reason);
        }

        private static void RecordInternal(string primeText, string? candidateText, HeuristicFailureReason reason)
        {
            StringBuilder? existing = s_threadBuilder.Value;
            StringBuilder builder = existing ?? StringBuilderPool.Rent();
            int count = s_threadEntryCount.Value + 1;
            bool forceFlush = count >= MaxBufferedEntries;
            builder.Append(DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture));
            builder.Append(" | prime=");
            builder.Append(primeText);
            builder.Append(" | reason=");
            builder.Append(reason);
            if (candidateText is not null)
            {
                builder.Append(" | candidateOrder=");
                builder.Append(candidateText);
            }

            builder.AppendLine();
            FlushBuilderIfNeeded(builder, forceFlush);
            if (builder.Length == 0)
            {
                count = 0;
            }

            s_threadEntryCount.Value = count;
            s_threadBuilder.Value = builder;
        }

        private static void FlushAll(bool force)
        {
            foreach (StringBuilder? builder in s_threadBuilder.Values)
            {
                if (builder is not null)
                {
                    FlushBuilderIfNeeded(builder, force);
                }
            }
        }

        private static void FlushBuilderIfNeeded(StringBuilder builder, bool force)
        {
            if (builder.Length == 0)
            {
                return;
            }

            if (!force && builder.Length < FlushThreshold)
            {
                return;
            }

            string text = builder.ToString();
            builder.Clear();
            lock (s_fileLock)
            {
                using FileStream stream = new FileStream(s_logPath, FileMode.Append, FileAccess.Write, FileShare.ReadWrite);
                using StreamWriter writer = new StreamWriter(stream, s_utf8NoBom);
                writer.Write(text);
            }
        }
    }
}