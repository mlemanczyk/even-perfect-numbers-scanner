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

        // Inputs are already reduced in production; keep the old modulo guard commented for debugging reference.
        // M_mod = M_mod_m_at_p0 % m;
        // pow2_mod = (M_mod + 1) % m;
        M_mod = M_mod_m_at_p0;
        pow2_mod = M_mod + 1;
        if (pow2_mod >= m)
        {
            pow2_mod -= m;
        }
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

        ulong absDelta = (ulong)Math.Abs(delta);
        UInt128 g = absDelta.PowMod(m);

        if (delta > 0)
        {
            // Reuse the cached 2^p residue and only multiply by the adaptive pow2 step computed above.
            pow2_mod = (pow2_mod * g) % m;
        }
        else
        {
            // Modular inverse for negative delta
            // TODO: Switch to a cached divisor-cycle remainder instead of computing a fresh modular inverse for every
            // backward jump; the divisor cycle benchmarks demonstrated significant savings once shared cycle data was used.
            UInt128 inverse = ModInverse(g, m);
            pow2_mod = (pow2_mod * inverse) % m;
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
