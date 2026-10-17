using System.Runtime.CompilerServices;
using Sdcb.SimdOnnx.Engine;

namespace Sdcb.SimdOnnx.Kernels;

/// <summary>
/// Generic strided gather (transpose / slice) and broadcast binary over a contiguous output: dims are
/// coalesced where every operand's strides allow, size-1 dims dropped, then the innermost run is a
/// block copy / vector op (stride 1), a broadcast scalar (stride 0) or a short gather, and the outer
/// rows are split over threads. Replaces the per-element div / mod index loops.
/// </summary>
internal static unsafe class Strided
{
    /// <summary>dst[i] = src[off + Σ idx_d · str_d] over <paramref name="shape"/> (row-major dst).</summary>
    public static void Gather(float* src, long off, long[] str, int[] shape, float* dst)
    {
        Coalesce(shape, [str], out int[] dims, out long[][] strides);
        long[] s = strides[0];
        int rank = dims.Length;
        long total = 1;
        foreach (int d in dims) total *= d;
        if (total == 0) return;
        int inner = dims[^1];
        long istr = s[^1];
        // a short innermost dim walked with a long stride (NCHW <-> NHWC with few channels): run the
        // next dim innermost instead, as a 2D transpose of 16-row tiles (dst writes stay sequential per tile)
        if (rank >= 2 && inner <= 8 && istr != 1)
        {
            int rows = dims[^2];
            long rstr = s[^2];
            long outer = total / ((long)inner * rows);
            int[] od = Head(dims, 2);
            long[] os = Head(s, 2);
            int tile = 16, tiles = (rows + tile - 1) / tile;
            long work = outer * tiles;
            ForChunks(work, total, (w0, w1) =>
            {
                for (long w = w0; w < w1; w++)
                {
                    long o = w / tiles;
                    int r0 = (int)(w % tiles) * tile, r1 = Math.Min(rows, r0 + tile);
                    long sb = off + Offset(o, od, os);
                    float* db = dst + o * rows * inner;
                    for (int c = 0; c < inner; c++)
                    {
                        float* sc = src + sb + c * istr;
                        for (int r = r0; r < r1; r++) db[(long)r * inner + c] = sc[r * rstr];
                    }
                }
            });
            return;
        }
        long nrows = total / inner;
        int[] odims = Head(dims, 1);
        long[] ostr = Head(s, 1);
        ForChunks(nrows, total, (r0, r1) =>
        {
            for (long r = r0; r < r1; r++)
            {
                float* sp = src + off + Offset(r, odims, ostr), dp = dst + r * inner;
                if (istr == 1) Buffer.MemoryCopy(sp, dp, (long)inner * 4, (long)inner * 4);
                else for (int i = 0; i < inner; i++) dp[i] = sp[i * istr];
            }
        });
    }

    /// <summary>d[i] = a[Σ idx·aStr] op b[Σ idx·bStr] over <paramref name="shape"/> (row-major d), then relu.</summary>
    public static void Binary(float* a, long[] aStr, float* b, long[] bStr, int[] shape, float* d, EwOp op, bool relu)
    {
        Coalesce(shape, [aStr, bStr], out int[] dims, out long[][] strides);
        long[] sa = strides[0], sb = strides[1];
        long total = 1;
        foreach (int x in dims) total *= x;
        if (total == 0) return;
        int inner = dims[^1];
        long ia = sa[^1], ib = sb[^1];
        long nrows = total / inner;
        int[] odims = Head(dims, 1);
        long[] oa = Head(sa, 1), ob = Head(sb, 1);
        ForChunks(nrows, total, (r0, r1) =>
        {
            for (long r = r0; r < r1; r++)
            {
                float* pa = a + Offset(r, odims, oa), pb = b + Offset(r, odims, ob), pd = d + r * inner;
                if (inner >= 16 && ia == 1 && ib == 1) Ew.BinarySerial(pa, pb, pd, inner, op, relu);
                else if (inner >= 16 && ia == 1 && ib == 0) Ew.ScalarSerial(pa, *pb, pd, inner, op, rev: false, relu);
                else if (inner >= 16 && ia == 0 && ib == 1) Ew.ScalarSerial(pb, *pa, pd, inner, op, rev: true, relu);
                else
                    for (int i = 0; i < inner; i++)
                    {
                        float x = pa[i * ia], y = pb[i * ib];
                        float v = op switch { EwOp.Add => x + y, EwOp.Sub => x - y, EwOp.Mul => x * y, _ => x / y };
                        pd[i] = relu && v < 0f ? 0f : v;
                    }
            }
        });
    }

    /// <summary>Row-major strides of <paramref name="shape"/>, 0 on size-1 dims (numpy broadcast) when right-aligned onto rank.</summary>
    public static long[] BroadcastStrides(int[] shape, int rank)
    {
        long[] s = new long[rank];
        long acc = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            s[i + rank - shape.Length] = shape[i] == 1 ? 0 : acc;
            acc *= shape[i];
        }
        return s;
    }

    // drops size-1 dims and merges dim i into i+1 when every operand steps through them as one
    private static void Coalesce(int[] shape, long[][] strs, out int[] dims, out long[][] strides)
    {
        var d = new List<int>(shape.Length);
        var s = new List<long>[strs.Length];
        for (int k = 0; k < strs.Length; k++) s[k] = new List<long>(shape.Length);
        for (int i = 0; i < shape.Length; i++)
        {
            if (shape[i] == 1) continue;
            if (d.Count > 0)
            {
                bool merge = true;
                for (int k = 0; k < strs.Length && merge; k++) merge = s[k][^1] == strs[k][i] * shape[i];
                if (merge)
                {
                    d[^1] *= shape[i];
                    for (int k = 0; k < strs.Length; k++) s[k][^1] = strs[k][i];
                    continue;
                }
            }
            d.Add(shape[i]);
            for (int k = 0; k < strs.Length; k++) s[k].Add(strs[k][i]);
        }
        if (d.Count == 0) { d.Add(1); foreach (var l in s) l.Add(0); }
        dims = [.. d];
        strides = new long[strs.Length][];
        for (int k = 0; k < strs.Length; k++) strides[k] = [.. s[k]];
    }

    // all but the last k elements (no range operator: netstandard2.0 lacks RuntimeHelpers.GetSubArray)
    private static T[] Head<T>(T[] a, int k)
    {
        var r = new T[a.Length - k];
        Array.Copy(a, r, r.Length);
        return r;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static long Offset(long idx, int[] dims, long[] str)
    {
        long o = 0;
        for (int i = dims.Length - 1; i >= 0; i--)
        {
            long q = idx / dims[i];
            o += (idx - q * dims[i]) * str[i];
            idx = q;
        }
        return o;
    }

    // [0, n) in contiguous chunks over the intra-op threads (serial for small work)
    private static void ForChunks(long n, long work, Action<long, long> body)
    {
        int threads = Par.Threads;
        if (work < 65536 || n < 2 || threads == 1) { body(0, n); return; }
        int shards = (int)Math.Min(threads * 4L, n);
        Par.For(0, shards, w => body(n * w / shards, n * (w + 1) / shards));
    }
}
