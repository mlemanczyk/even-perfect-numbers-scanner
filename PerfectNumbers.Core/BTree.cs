using System.Runtime.CompilerServices;

namespace PerfectNumbers.Core;

[method: MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
public sealed class BTree<TKey, TValue>()
{
    private const int HashByteCount = sizeof(int);
    private const int DefaultCapacity = 64;
    private const int SmallBufferCapacity = 4;
    private const int RootInlineThreshold = 2048;
    private const int RootInlineInitialCapacity = 32;
    private const int ShallowTreeThreshold = 1024;
    private const int ShallowHashByteCount = 1;
    private const int LeafInitialCapacity = 8;
    private static readonly EqualityComparer<TKey> _defaultComparer = EqualityComparer<TKey>.Default;

    private readonly TKey[] _smallKeys = new TKey[SmallBufferCapacity];
    private readonly TValue[] _smallValues = new TValue[SmallBufferCapacity];
    private int _smallCount;

    private Node[] _nodes = [];
    private int _nodeCount;

    public int Count;

	[MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Add(TKey key, TValue value)
    {
		TKey[] smallKeys = _smallKeys;
		TValue[] smallValues = _smallValues;
		int smallCount = _smallCount;
		if (smallCount < SmallBufferCapacity)
        {
			smallKeys[smallCount] = key;
			smallValues[smallCount] = value;
            _smallCount = smallCount + 1;
            Count++;
            return;
        }

        EnsureRoot();

		Node[] nodes = _nodes;
        if (smallCount > 0)
        {
			ref Node rootNode = ref nodes[0];
            for (int i = 0; i < smallCount; i++)
            {
                rootNode.AddKeyValue(smallKeys[i], smallValues[i], RootInlineInitialCapacity);
            }

            _smallCount = 0;
        }

        if (Count < RootInlineThreshold)
        {
            ref Node root = ref nodes[0];
            root.AddKeyValue(key, value, RootInlineInitialCapacity);
            Count++;
            return;
        }

        InsertIntoTree(GetHash(key), key, value, Count);
        Count++;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public int CopyTo(Span<TValue> destination)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(destination.Length, Count);

        var written = 0;

		int smallCount = _smallCount;
		TValue[] smallValues = _smallValues;
		for (int i = 0; i < smallCount; i++)
        {
			destination[written++] = smallValues[i];
        }

		int nodeCount = _nodeCount;
		if (nodeCount == 0)
        {
            return written;
        }

        FixedCapacityStack<int> stack = new(nodeCount);
        stack.Push(0);

        Node[]? nodes = _nodes;
        while (stack.Count > 0)
        {
            int index = stack.Pop();
            ref readonly Node node = ref nodes[index];
            written += node.CopyValues(destination[written..]);

            for (int childIndex = 0; childIndex < Node.ChildCount; childIndex++)
            {
                index = node.GetChild(childIndex);
                if (index != -1)
                {
                    stack.Push(index);
                }
            }
        }

        return written;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static uint GetHash(TKey key)
    {
        return key is not null
            ? unchecked((uint)_defaultComparer.GetHashCode(key))
            : 0u;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private void EnsureRoot()
    {
        if (_nodeCount != 0)
        {
            return;
        }

        _nodes = new Node[DefaultCapacity];
        _nodes[0] = new Node();
        _nodeCount = 1;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void InsertIntoTree(uint hash, TKey key, TValue value, int currentCount)
    {
        var nodes = _nodes;
        var currentIndex = 0;
        int depth = currentCount < ShallowTreeThreshold ? ShallowHashByteCount : HashByteCount;

        int nodeCount = _nodeCount;
        for (int i = 0; i < depth; i++)
        {
            ref Node node = ref nodes[currentIndex];
            int bucket = (byte)(hash >> (i << 3)) >> 5;
            int nextIndex = node.GetChild(bucket);
            if (nextIndex == -1)
            {
                nextIndex = nodeCount;
                nodes = EnsureNodeCapacity(nodes, nextIndex, nodeCount);
                node = ref nodes[currentIndex];
                nodes[nextIndex] = new Node();
                node.SetChild(bucket, nextIndex);
                nodeCount++;
            }

            currentIndex = nextIndex;
        }

		_nodes = nodes;
        _nodeCount = nodeCount;
        ref Node leaf = ref nodes[currentIndex];
        leaf.AddKeyValue(key, value, LeafInitialCapacity);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static Node[] EnsureNodeCapacity(in Node[] nodes, int requiredIndex, int liveNodeCount)
    {
        int currentLength = nodes.Length;
        if (requiredIndex < currentLength)
        {
            return nodes;
        }

        int targetLength = currentLength == 0 ? DefaultCapacity : currentLength;
        while (targetLength <= requiredIndex)
        {
            targetLength <<= 1;
        }

        Node[] newNodes = new Node[targetLength];
        if (liveNodeCount > currentLength)
        {
            liveNodeCount = currentLength;
        }

        Array.Copy(nodes, newNodes, liveNodeCount);
        return newNodes;
    }

    private struct Node
    {
        public const int ChildCount = 8;

        private TKey[]? _keys;
        private TValue[]? _values;
        private int _entryCount;
        private int[]? _children;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public readonly int GetChild(int slot)
        {
            int[]? children = _children;
            return children == null ? -1 : children[slot];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void SetChild(int slot, int index)
        {
            int[]? children = _children;
            if (children == null)
            {
                children = new int[ChildCount];
                for (int i = 0; i < ChildCount; i++)
                {
                    children[i] = -1;
                }

                _children = children;
            }

            children[slot] = index;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void AddKeyValue(TKey key, TValue value, int desiredInitialCapacity)
        {
            TKey[]? keys = _keys;
            TValue[]? values = _values;
            int entryCount = _entryCount;

			int newLength;
            if (keys == null)
            {
                newLength = desiredInitialCapacity < 4 ? 4 : desiredInitialCapacity;
                keys = new TKey[newLength];
                values = new TValue[newLength];
                _keys = keys;
                _values = values;
            }
            else if (entryCount == keys.Length)
            {
                newLength = keys.Length << 1;
                TKey[] newKeys = new TKey[newLength];
                TValue[] newValues = new TValue[newLength];
                new Span<TKey>(keys, 0, entryCount).CopyTo(newKeys);
                new Span<TValue>(values, 0, entryCount).CopyTo(newValues);
                keys = newKeys;
                values = newValues;
                _keys = newKeys;
                _values = newValues;
            }

            keys[entryCount] = key;
            values![entryCount] = value;
            _entryCount = entryCount + 1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public readonly int CopyValues(Span<TValue> destination)
        {
            TValue[]? values = _values;
            if (values == null)
            {
                return 0;
            }

            int entryCount = _entryCount;
            new Span<TValue>(values, 0, entryCount).CopyTo(destination);
            return entryCount;
        }
    }
}
