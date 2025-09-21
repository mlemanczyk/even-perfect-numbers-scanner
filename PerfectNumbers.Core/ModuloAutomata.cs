using System.Runtime.CompilerServices;
using UInt128 = System.UInt128;

namespace PerfectNumbers.Core;

/// <summary>
/// Tracks residues for a fixed-step arithmetic progression without divisions.
/// Optimized variant for the Mersenne scan q = 2*p*k + 1 that needs mod 10, 8, 3 and 5.
/// </summary>
public sealed class MersenneResidueAutomaton
{
    private readonly UInt128 _step;               // 2*p
    private UInt128 _currentQ;                     // current q value

    // Cached residues for current q
    public ulong Mod10R { get; private set; }
    public ulong Mod8R  { get; private set; }
    public ulong Mod3R  { get; private set; }
    public ulong Mod5R  { get; private set; }

    // Cached residues for step (2*p mod m)
    private readonly ulong _step10;
    private readonly ulong _step8;
    private readonly ulong _step3;
    private readonly ulong _step5;

    public MersenneResidueAutomaton(ulong exponent)
    {
        _step = (UInt128)exponent << 1; // 2 * p
        _currentQ = _step + 1UL;        // start at k = 1

        // init residues
        Mod10R = _currentQ.Mod10();
        Mod8R  = Mod8(_currentQ);
        Mod3R  = Mod3(_currentQ);
        Mod5R  = Mod5(_currentQ);

        _step10 = _step.Mod10().Mod10();
        _step8  = (ulong)(_step & 7UL); // % 8
        _step3  = Mod3(_step);
        _step5  = Mod5(_step);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public UInt128 CurrentQ() => _currentQ;

    /// <summary>
    /// Advance to next k (q += 2*p), updating residues branchlessly.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Next()
    {
        _currentQ += _step;

        ulong r10 = Mod10R + _step10;
        r10 -= (r10 >= 10UL) ? 10UL : 0UL;
        Mod10R = r10;

        ulong r8 = Mod8R + _step8;
        r8 &= 7UL; // % 8
        Mod8R = r8;

        ulong r3 = Mod3R + _step3;
        r3 -= (r3 >= 3UL) ? 3UL : 0UL;
        Mod3R = r3;

        ulong r5 = Mod5R + _step5;
        if (r5 >= 5UL)
        {
            r5 -= 5UL;
        }
        Mod5R = r5;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong Mod8(UInt128 value) => (ulong)value & 7UL;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong Mod3(UInt128 value)
    {
        // 2^64 ≡ 1 (mod 3)
        ulong high = (ulong)(value >> 64);
        ulong low = (ulong)value;
        ulong rem = (low % 3UL) + (high % 3UL);
        return rem >= 3UL ? rem - 3UL : rem;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong Mod5(UInt128 value)
    {
        // 2^64 ≡ 1 (mod 5)
        ulong high = (ulong)(value >> 64);
        ulong low = (ulong)value;
        ulong rem = (low % 5UL) + (high % 5UL);
        return rem >= 5UL ? rem - 5UL : rem;
    }
}

/// <summary>
/// Generic automaton for scanning integers ending with digit 7: n = 10*t + 7.
/// Maintains residues r_m without divisions while advancing by +10 each step.
/// </summary>
public sealed class Ending7Automaton
{
    private ulong _n; // current candidate n, ends with 7

    private readonly ulong[] _moduli;
    private readonly ulong[] _steps;   // 10 % m
    private readonly ulong[] _res;     // n % m
    private readonly bool[] _reachable;

    public Ending7Automaton(ulong start, params ulong[] moduli)
    {
        if (moduli == null || moduli.Length == 0)
        {
            _moduli = Array.Empty<ulong>();
            _steps = Array.Empty<ulong>();
            _res = Array.Empty<ulong>();
            _reachable = Array.Empty<bool>();
            _n = AlignTo7(start);
            return;
        }

        _n = AlignTo7(start);
        int len = moduli.Length;
        _moduli = new ulong[len];
        _steps = new ulong[len];
        _res = new ulong[len];
        _reachable = new bool[len];
        for (int i = 0; i < len; i++)
        {
            ulong m = moduli[i];
            _moduli[i] = m;
            ulong step = 10UL % m;
            _steps[i] = step;
            _res[i] = _n % m; // one-time division at construction
            // Zero is reachable iff gcd(step, m) | res
            ulong g = Gcd(step, m);
            _reachable[i] = (g == 0UL) ? false : (_res[i] % g == 0UL);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ulong Current() => _n;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool DivisibleBy(ulong modulus)
    {
        for (int i = 0; i < _moduli.Length; i++)
        {
            if (_moduli[i] == modulus)
            {
                return _reachable[i] && _res[i] == 0UL;
            }
        }

        return (_n % modulus) == 0UL;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Next()
    {
        _n += 10UL;
        for (int i = 0; i < _moduli.Length; i++)
        {
            if (!_reachable[i])
            {
                continue;
            }

            ulong m = _moduli[i];
            ulong r = _res[i] + _steps[i];
            if (r >= m)
            {
                r -= m;
            }
            _res[i] = r;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong AlignTo7(ulong start)
    {
        ulong r = start.Mod10();
		ulong add = (7UL + 10UL - r).Mod10();
        return start + add;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong Gcd(ulong a, ulong b)
    {
        while (b != 0UL)
        {
            ulong t = a % b;
            a = b;
            b = t;
        }
        return a;
    }
}
