using System.Runtime.CompilerServices;
namespace Sdcb.SimdOnnx.Engine;

/// <summary>
/// Physical-layout algebra. A layout <c>L</c> lists, for each physical
/// (storage) dim i, the logical dim it holds: storage is row-major over
/// <c>[shape[L[0]], shape[L[1]], ...]</c>. Logical shapes never change —
/// fusion passes and perm reasoning stay in logical space; kernels address by
/// <see cref="Phys"/>. Size-1 dims carry no stride, so layouts that order the
/// non-1 dims identically are interchangeable (<see cref="Equivalent"/>).
/// </summary>
internal static class Layout
{
    /// <summary>Channels-last for rank-4 [n,c,h,w]: storage [n,h,w,c].</summary>
    public static readonly int[] Nhwc = [0, 2, 3, 1];

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static int[] Identity(int rank)
    {
        int[] l = new int[rank];
        for (int i = 0; i < rank; i++) l[i] = i;
        return l;
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static bool IsIdentity(int[] l)
    {
        for (int i = 0; i < l.Length; i++) if (l[i] != i) return false;
        return true;
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static int[] Phys(int[] shape, int[] l)
    {
        int[] p = new int[l.Length];
        for (int i = 0; i < l.Length; i++) p[i] = shape[l[i]];
        return p;
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static int[] Inverse(int[] p)
    {
        int[] inv = new int[p.Length];
        for (int i = 0; i < p.Length; i++) inv[p[i]] = i;
        return inv;
    }

    /// <summary>
    /// Layout of y = Transpose(x, perm) as a view of x's storage. y logical k
    /// is x logical perm[k], so x logical d is y logical perm⁻¹[d]:
    /// L'[i] = perm⁻¹[L[i]]. Always representable — every Transpose dissolves.
    /// </summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static int[] Transposed(int[] l, int[] perm)
    {
        int[] inv = Inverse(perm);
        int[] r = new int[l.Length];
        for (int i = 0; i < l.Length; i++) r[i] = inv[l[i]];
        return r;
    }

    /// <summary>Same storage order for <paramref name="shape"/> (size-1 dims ignored).</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static bool Equivalent(int[] shape, int[] a, int[] b)
    {
        // the non-1 dims in the same order (allocation-free: runs per plan node)
        int i = 0, j = 0;
        while (true)
        {
            while (i < a.Length && shape[a[i]] == 1) i++;
            while (j < b.Length && shape[b[j]] == 1) j++;
            if (i == a.Length || j == b.Length) return i == a.Length && j == b.Length;
            if (a[i++] != b[j++]) return false;
        }
    }

    /// <summary>Physical perm relaying storage from layout <paramref name="from"/> to
    /// <paramref name="to"/>: out phys dim i = in phys dim q[i].</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static int[] PhysPerm(int[] from, int[] to) =>
        to.Select(d => Array.IndexOf(from, d)).ToArray();

    /// <summary>
    /// Layout of a zero-copy reshape <paramref name="src"/>→<paramref name="dst"/>
    /// of storage laid out as <paramref name="l"/>, or null when the reshape
    /// needs data movement. The logical reshape factors into groups (a run of
    /// source dims ↔ a run of target dims with equal product); it is a view iff
    /// each group's non-1 source dims sit consecutively, in logical order, in
    /// storage. Target dims are then ordered by their group's storage position.
    /// </summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static int[]? Reshaped(int[] src, int[] l, int[] dst)
    {
        // allocation-free apart from the result (runs per alias resolution in every compile)
        Span<int> sd = stackalloc int[src.Length], td = stackalloc int[dst.Length];
        int ns = 0, nt = 0;
        for (int d = 0; d < src.Length; d++) if (src[d] != 1) sd[ns++] = d;
        for (int d = 0; d < dst.Length; d++) if (dst[d] != 1) td[nt++] = d;
        // storage rank of each non-1 source dim
        Span<int> rank = stackalloc int[src.Length];
        rank.Clear();
        int r = 0;
        foreach (int d in l) if (src[d] != 1) rank[d] = r++;

        Span<double> key = stackalloc double[dst.Length];
        key.Clear();
        int i = 0, j = 0;
        while (i < ns || j < nt)
        {
            if (i >= ns || j >= nt) return null;   // element count mismatch
            // a group: source dims sd[i0..i) <-> target dims td[j0..j) with equal products
            int i0 = i, j0 = j;
            long ps = src[sd[i++]], pt = dst[td[j++]];
            while (ps != pt)
            {
                if (ps < pt) { if (i >= ns) return null; ps *= src[sd[i++]]; }
                else { if (j >= nt) return null; pt *= dst[td[j++]]; }
            }
            for (int k = i0 + 1; k < i; k++)
                if (rank[sd[k]] != rank[sd[k - 1]] + 1) return null;
            int gt = j - j0;
            for (int k = 0; k < gt; k++)
                key[td[j0 + k]] = rank[sd[i0]] + k / (double)(gt + 1);
        }
        // size-1 target dims ride right behind their logical predecessor (or lead),
        // so an identity source yields an identity target
        double prev = -1;
        for (int e = 0; e < dst.Length; e++)
        {
            if (dst[e] != 1) prev = key[e];
            else key[e] = prev + 1e-3 * (e + 1) / (dst.Length + 1);
        }
        // order by key, then index (stable insertion sort; ranks are tiny)
        int[] o = new int[dst.Length];
        for (int e = 0; e < dst.Length; e++)
        {
            int p = e;
            while (p > 0 && key[o[p - 1]] > key[e]) { o[p] = o[p - 1]; p--; }
            o[p] = e;
        }
        return o;
    }

    public static string Str(int[]? l) => l is null ? "-" : "[" + string.Join(',', l) + "]";
}
