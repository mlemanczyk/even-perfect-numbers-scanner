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
            throw new ArgumentException("Modulus must be positive.");
        m = modulus;
        p = p0;
        M_mod = M_mod_m_at_p0 % m;
        pow2_mod = (M_mod + 1) % m;
    }

    public (ulong, UInt128) State() => (p, M_mod);

    public (ulong, UInt128) Step(long delta)
    {
        if (delta == 0)
            return State();
        UInt128 g = UInt128.One;
        ulong absDelta = (ulong)Math.Abs(delta);
        for (ulong i = 0; i < absDelta; i++)
        {
            g <<= 1;
            if (g >= m)
                g -= m;
        }
        if (delta > 0)
            pow2_mod = (pow2_mod * g) % m;
        else
        {
            // Modular inverse for negative delta
            pow2_mod = (pow2_mod * ModInverse(g, m)) % m;
        }
        p = (ulong)((long)p + delta);
        M_mod = (pow2_mod + m - 1) % m;
        return State();
    }

    public (ulong, UInt128) To(ulong p_target)
    {
        long delta = (long)p_target - (long)p;
        return Step(delta);
    }

    private static UInt128 ModInverse(UInt128 a, UInt128 m)
    {
        UInt128 m0 = m, t, q;
        UInt128 x0 = 0, x1 = 1;
        if (m == 1) return 0;
        while (a > 1)
        {
            q = a / m;
            t = m;
            m = a % m; a = t;
            t = x0;
            x0 = x1 - q * x0;
            x1 = t;
        }
        if (x1 < 0) x1 += m0;
        return x1;
    }
}
