namespace PerfectNumbers.Core;

public class MersenneResidueStepper
{
    private UInt128 m;
    private ulong p;
    private UInt128 pow2_mod;
    private UInt128 M_mod;

    public MersenneResidueStepper(UInt128 modulus, ulong p0, UInt128 M_mod_m_at_p0)
    {
        if (modulus == 0)
        {
            throw new ArgumentException("Modulus must be positive.");
        }

        m = modulus;
        p = p0;

        // TODO: Replace these `%` reductions with the shared ProcessEightBitWindows helper so initialization benefits from the
        // windowed pow2mod pipeline proven faster than the generic modulo path.
        M_mod = M_mod_m_at_p0 % m;
        pow2_mod = (M_mod + 1) % m;
    }

    // TODO: Inline this accessor into callers so the hot residue path reads the backing fields directly
    // instead of jumping through a wrapper method every iteration.
    public (ulong, UInt128) State() => (p, M_mod);

    public (ulong, UInt128) Step(long delta)
    {
        if (delta == 0)
        {
            return State();
        }

        UInt128 g = UInt128.One;
        ulong absDelta = (ulong)Math.Abs(delta);

        for (ulong i = 0; i < absDelta; i++)
        {
            // TODO: Replace this repeated doubling with the shared ProcessEightBitWindows helper (or cycle-aware lookup)
            // once the scalar pow2mod upgrade lands; benchmarking showed the windowed ladder slashes latency for large
            // deltas compared to this linear loop.
            // TODO: Once divisor cycles power the residue stepper, prefer stepping by cycle lengths instead of incrementing
            // one bit at a time so large jumps stay on the mandatory cycle-accelerated path.
            g <<= 1;

            if (g >= m)
            {
                g -= m;
            }
        }

        if (delta > 0)
        {
            // TODO: Once the scalar windowed pow2 helper lands, reroute this `%` path through it to avoid the slower
            // multiply-and-mod sequence measured in the legacy benchmarks.
            pow2_mod = (pow2_mod * g) % m;
        }
        else
        {
            // Modular inverse for negative delta
            // TODO: Switch to a cached divisor-cycle remainder instead of computing a fresh modular inverse for every
            // backward jump; the divisor cycle benchmarks demonstrated significant savings once shared cycle data was used.
            pow2_mod = (pow2_mod * ModInverse(g, m)) % m;
        }

        p = (ulong)((long)p + delta);

        // TODO: Use the shared pow2mod remainder cache here so this `%` becomes a subtraction-based fold instead of the
        // slower modulo operator for large moduli.
        M_mod = (pow2_mod + m - 1) % m;

        return State();
    }

    public (ulong, UInt128) To(ulong p_target)
    {
        // TODO: Collapse this wrapper once callers can invoke Step directly; the extra indirection shows up
        // in the profiler when residue adjustments happen per candidate.
        long delta = (long)p_target - (long)p;

        return Step(delta);
    }

    private static UInt128 ModInverse(UInt128 a, UInt128 m)
    {
        UInt128 m0 = m;
        UInt128 t;
        UInt128 q;
        UInt128 x0 = 0;
        UInt128 x1 = 1;

        if (m == 1)
        {
            return 0;
        }

        while (a > 1)
        {
            // TODO: Replace this division-heavy Euclidean loop with the binary modular inverse that reuses divisor
            // cycles so we drop repeated `/` and `%` operations during backtracking steps.
            q = a / m;
            t = m;
            m = a % m;
            a = t;
            t = x0;
            x0 = x1 - q * x0;
            x1 = t;
        }

        if (x1 < 0)
        {
            x1 += m0;
        }

        return x1;
    }
}
