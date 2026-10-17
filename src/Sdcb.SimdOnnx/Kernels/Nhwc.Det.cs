using System.Runtime.Intrinsics;
using Sdcb.SimdOnnx.Engine;

using V4 = System.Runtime.Intrinsics.Vector128<float>;

namespace Sdcb.SimdOnnx.Kernels;

/// <summary>Packed <see cref="ExecOp.ConvTranspose"/> weights: one Gemm panel ([ic] x [oc]) per kernel tap.</summary>
internal sealed class ConvTPlan
{
    public required NhwcConvPlan[] Taps;
    public float[]? Dot;    // oc <= 4: [tap][oc][ic] rows for per-pixel dot products
    public float[] Bias = [];
}

/// <summary>
/// NHWC kernels for the PP-OCR det graph (SE blocks, nearest upsample, transposed-conv head).
/// Streaming loops use Vector128 (SSE / NEON / WASM SIMD128 alike — Vector&lt;float&gt; is not
/// accelerated on browser-wasm); the transposed conv reuses each tier's conv micro-kernel.
/// </summary>
internal static unsafe partial class Nhwc
{
    internal static bool V128 => Vector128.IsHardwareAccelerated;

    /// <summary>W [ic, oc, kh, kw] → per tap a [oc/16][ic][16] panel (bias folded into every tap).</summary>
    public static ConvTPlan PackConvT(ReadOnlySpan<float> w, int[] ws, ReadOnlySpan<float> bias)
    {
        int ic = ws[0], oc = ws[1], taps = ws[2] * ws[3];
        var plans = new NhwcConvPlan[taps];
        float[] b = new float[ic * oc];
        for (int t = 0; t < taps; t++)
        {
            for (int c = 0; c < ic; c++)
                for (int o = 0; o < oc; o++) b[c * oc + o] = w[(c * oc + o) * taps + t];
            plans[t] = PackGemm(b, ic, oc, false, bias);
        }
        if (oc > 4) return new ConvTPlan { Taps = plans };
        float[] dot = new float[taps * oc * ic], pb = new float[oc];
        for (int t = 0; t < taps; t++)
            for (int o = 0; o < oc; o++)
                for (int c = 0; c < ic; c++) dot[(t * oc + o) * ic + c] = w[(c * oc + o) * taps + t];
        if (!bias.IsEmpty) bias.CopyTo(pb);
        return new ConvTPlan { Taps = plans, Dot = dot, Bias = pb };
    }

    /// <summary>
    /// Transposed conv with stride == kernel and no padding: every input pixel owns a disjoint
    /// kh x kw output patch, so each tap is a plain [pixels, ic] x [ic, oc] product written with a
    /// pixel stride of kw*oc — the conv micro-kernel of the active tier with a strided output.
    /// </summary>
    public static void ConvTranspose(ReadOnlySpan<float> src, int ih, int iw, int ic,
        Span<float> dst, int oc, int kh, int kw, ConvTPlan kp, ActivationKind act)
    {
        if (kp.Dot is not null)
        {
            ConvTDot(src, ih, iw, ic, dst, oc, kh, kw, kp, act);
            return;
        }
        int taps = kh * kw, nb = kp.Taps[0].Blocks, ow = iw * kw, stride = kw * oc;
        bool avx = UseAvx, pk = !avx && UsePacked;
        int tr = pk ? PkTile : Tile;
        fixed (float* sp0 = src, dp0 = dst)
        {
            float* sp = sp0, dp = dp0;
            For(ih * taps * nb, (long)ih * iw * ic * oc * taps / 64, t =>
            {
                int y = t / (taps * nb), r = t % (taps * nb), tap = r / nb, blk = r % nb;
                int dy = tap / kw, dx = tap % kw;
                NhwcConvPlan p = kp.Taps[tap];
                int lanes = Math.Min(Ob, oc - blk * Ob);
                float** ptrs = stackalloc float*[Tile];
                float* part = stackalloc float[Tile * Ob];
                fixed (float* wp = p.W, bp = p.Bias)
                {
                    float* wb = wp + (long)blk * ic * Ob, bb = bp + blk * Ob;
                    float* orow = dp + ((long)(y * kh + dy) * ow + dx) * oc + blk * Ob;
                    for (int x0 = 0; x0 < iw; x0 += tr)
                    {
                        int rows = Math.Min(tr, iw - x0);
                        for (int i = 0; i < tr; i++) ptrs[i] = sp + ((long)y * iw + x0 + Math.Min(i, rows - 1)) * ic;
                        float* outp = orow + (long)x0 * stride;
                        if (avx) TileAvx(ptrs, rows, 1, ic, wb, bb, outp, null, stride, lanes, act);
                        else if (pk)
                        {
                            TilePkK(ptrs, PkTile, 1, ic, (long)ic * Ob, wb, bb, part, Ob);
                            StorePk(part, rows, outp, null, stride, lanes, act);
                        }
                        else TileVec(ptrs, rows, 1, ic, wb, bb, outp, null, stride, lanes, act);
                    }
                }
            });
        }
    }

    /// <summary>oc ≤ 4 transposed conv (the 1-channel prob head): one dot product over ic per output element.</summary>
    internal static void ConvTDot(ReadOnlySpan<float> src, int ih, int iw, int ic,
        Span<float> dst, int oc, int kh, int kw, ConvTPlan kp, ActivationKind act)
    {
        int taps = kh * kw, ow = iw * kw;
        bool v = V128;
        fixed (float* sp0 = src, dp0 = dst, wp0 = kp.Dot, bp0 = kp.Bias)
        {
            float* sp = sp0, dp = dp0, wp = wp0, bp = bp0;
            For(ih, (long)ih * iw * ic * oc * taps, y =>
            {
                for (int x = 0; x < iw; x++)
                {
                    float* xp = sp + ((long)y * iw + x) * ic;
                    for (int t = 0; t < taps; t++)
                    {
                        float* o = dp + ((long)(y * kh + t / kw) * ow + x * kw + t % kw) * oc;
                        for (int k = 0; k < oc; k++)
                        {
                            float* w = wp + (long)(t * oc + k) * ic;
                            int c = 0;
                            float s = bp[k];
                            if (v)
                            {
                                V4 acc = V4.Zero;
                                for (; c + 4 <= ic; c += 4) acc += Vector128.Load(xp + c) * Vector128.Load(w + c);
                                s += Vector128.Sum(acc);
                            }
                            for (; c < ic; c++) s += xp[c] * w[c];
                            o[k] = s;
                        }
                    }
                }
                if (act != ActivationKind.None) ActRow(dp + (long)y * kh * ow * oc, kh * ow * oc, act, UseAvx);
            });
        }
    }

    /// <summary>dst[c] = mean over all hw pixels of src[hw, c]; per-chunk partial sums, reduced in chunk order.</summary>
    public static void GlobalAvgPool(ReadOnlySpan<float> src, int hw, int c, Span<float> dst)
    {
        int chunks = (int)Math.Clamp((long)hw * c / 65536, 1, Par.Threads);
        float[] part = new float[chunks * c];
        bool v = V128;
        fixed (float* sp0 = src, pp0 = part)
        {
            float* sp = sp0, pp = pp0;
            For(chunks, (long)hw * c, k =>
            {
                int p0 = (int)((long)k * hw / chunks), p1 = (int)((long)(k + 1) * hw / chunks);
                float* acc = pp + (long)k * c;
                for (int px = p0; px < p1; px++)
                {
                    float* s = sp + (long)px * c;
                    int ch = 0;
                    if (v)
                        for (; ch + 4 <= c; ch += 4)
                            (Vector128.Load(acc + ch) + Vector128.Load(s + ch)).Store(acc + ch);
                    for (; ch < c; ch++) acc[ch] += s[ch];
                }
            });
        }
        float inv = 1f / hw;
        for (int ch = 0; ch < c; ch++)
        {
            float s = 0;
            for (int k = 0; k < chunks; k++) s += part[k * c + ch];
            dst[ch] = s * inv;
        }
    }

    /// <summary>Nearest resize, asymmetric coordinates, floor rounding: src = floor(dst / scale).</summary>
    public static void ResizeNearest(ReadOnlySpan<float> src, int ih, int iw, int c,
        Span<float> dst, int oh, int ow, float sy, float sx, ReadOnlySpan<float> add = default, bool addRelu = false)
    {
        if (!add.IsEmpty && add.Length != dst.Length)
            throw new NotSupportedException($"resize addend has {add.Length} elems, expected {dst.Length}.");
        int[] xs = new int[ow];
        for (int x = 0; x < ow; x++) xs[x] = Math.Min((int)MathF.Floor(x / sx), iw - 1);
        bool hasAdd = !add.IsEmpty, v = V128;
        fixed (float* sp0 = src, dp0 = dst, ap0 = add)
        {
            float* sp = sp0, dp = dp0, ap = ap0;
            For(oh, (long)oh * ow * c, y =>
            {
                int ys = Math.Min((int)MathF.Floor(y / sy), ih - 1);
                float* srow = sp + (long)ys * iw * c, orow = dp + (long)y * ow * c;
                if (!hasAdd)
                {
                    for (int x = 0; x < ow; x++) Copy(srow + (long)xs[x] * c, orow + (long)x * c, c);
                    return;
                }
                // FPN top-down merge: out = up(x) + [relu](lateral)
                float* arow = ap + (long)y * ow * c;
                for (int x = 0; x < ow; x++)
                {
                    float* s = srow + (long)xs[x] * c, a = arow + (long)x * c, o = orow + (long)x * c;
                    int ch = 0;
                    if (v)
                        for (; ch + 4 <= c; ch += 4)
                        {
                            V4 av = Vector128.Load(a + ch);
                            if (addRelu) av = Vector128.Max(av, V4.Zero);
                            (Vector128.Load(s + ch) + av).Store(o + ch);
                        }
                    for (; ch < c; ch++) o[ch] = s[ch] + (addRelu ? MathF.Max(a[ch], 0f) : a[ch]);
                }
            });
        }
    }

    /// <summary>
    /// DepthToSpace of one image, <paramref name="nhwc"/> storage or NCHW planes: output (c, y, x) reads input
    /// channel c·b² + (y % b)·b + x % b (CRD) or ((y % b)·b + x % b)·oc + c (DCR) at (y / b, x / b).
    /// </summary>
    public static void DepthToSpace(ReadOnlySpan<float> src, int ih, int iw, int ic, Span<float> dst, int b, bool crd, bool nhwc)
    {
        int oc = ic / (b * b), oh = ih * b, ow = iw * b;
        fixed (float* sp0 = src, dp0 = dst)
        {
            float* sp = sp0, dp = dp0;
            For(oh, (long)oh * ow * oc, y =>
            {
                int sy = y / b, dy = y % b;
                if (nhwc)
                {
                    float* orow = dp + (long)y * ow * oc;
                    for (int x = 0; x < ow; x++)
                    {
                        float* s = sp + ((long)sy * iw + x / b) * ic;
                        float* o = orow + (long)x * oc;
                        int sub = dy * b + x % b;
                        if (!crd) Copy(s + (long)sub * oc, o, oc);
                        else for (int c = 0; c < oc; c++) o[c] = s[c * b * b + sub];
                    }
                    return;
                }
                for (int c = 0; c < oc; c++)
                {
                    float* orow = dp + ((long)c * oh + y) * ow;
                    for (int x = 0; x < ow; x++)
                    {
                        int sub = dy * b + x % b, ch = crd ? c * b * b + sub : sub * oc + c;
                        orow[x] = sp[((long)ch * ih + sy) * iw + x / b];
                    }
                }
            });
        }
    }

    /// <summary>
    /// Squeeze-excitation gate of one image (<see cref="ExecOp.ChannelGate"/>): channel means of the NHWC
    /// image (<paramref name="nhwc"/>) or of its NCHW planes, then the two small fully connected layers.
    /// </summary>
    public static void ChannelGate(ReadOnlySpan<float> src, int hw, int c, bool nhwc, ReadOnlySpan<float> w1,
        ReadOnlySpan<float> b1, ReadOnlySpan<float> w2, ReadOnlySpan<float> b2, int r, ChannelGateP p, Span<float> dst)
    {
        float[] mean = new float[c], hid = new float[r];
        if (nhwc) GlobalAvgPool(src, hw, c, mean);
        else
            for (int ch = 0; ch < c; ch++)
            {
                float s = 0;
                foreach (float f in src.Slice(ch * hw, hw)) s += f;
                mean[ch] = s / hw;
            }
        for (int j = 0; j < r; j++)
        {
            float s = b1.IsEmpty ? 0f : b1[j];
            ReadOnlySpan<float> row = w1.Slice(j * c, c);
            for (int k = 0; k < c; k++) s += row[k] * mean[k];
            hid[j] = p.Act1 == ActivationKind.Relu ? MathF.Max(s, 0f) : s;
        }
        for (int ch = 0; ch < c; ch++)
        {
            float z = b2.IsEmpty ? 0f : b2[ch];
            ReadOnlySpan<float> row = w2.Slice(ch * r, r);
            for (int j = 0; j < r; j++) z += row[j] * hid[j];
            float g = p.Hard ? Math.Clamp(z * p.Alpha + p.Beta, 0f, 1f) : 1f / (1f + MathF.Exp(-z));
            dst[ch] = g + p.Offset;
        }
    }

    /// <summary>dst[r, :] = full[r, :] op vec[:] (or vec op full when <paramref name="rev"/>) — NHWC per-channel broadcast.</summary>
    public static void ChannelBroadcast(ReadOnlySpan<float> full, ReadOnlySpan<float> vec, Span<float> dst,
        int c, EwOp op, bool rev, ActivationKind act)
    {
        int rows = full.Length / c;
        bool relu = act == ActivationKind.Relu, v = V128;
        fixed (float* fp0 = full, vp0 = vec, dp0 = dst)
        {
            float* fp = fp0, vp = vp0, dp = dp0;
            int chunk = Math.Max(1, 16384 / c), tasks = (rows + chunk - 1) / chunk;
            For(tasks, (long)rows * c, t =>
            {
                for (int r = t * chunk, re = Math.Min(rows, r + chunk); r < re; r++)
                {
                    float* f = fp + (long)r * c, d = dp + (long)r * c;
                    int ch = 0;
                    if (v)
                        for (; ch + 4 <= c; ch += 4)
                        {
                            V4 a = Vector128.Load(f + ch), b = Vector128.Load(vp + ch);
                            if (rev) (a, b) = (b, a);
                            V4 s = op switch { EwOp.Add => a + b, EwOp.Sub => a - b, EwOp.Mul => a * b, _ => a / b };
                            if (relu) s = Vector128.Max(s, V4.Zero);
                            s.Store(d + ch);
                        }
                    for (; ch < c; ch++)
                    {
                        float a = f[ch], b = vp[ch];
                        if (rev) (a, b) = (b, a);
                        float s = op switch { EwOp.Add => a + b, EwOp.Sub => a - b, EwOp.Mul => a * b, _ => a / b };
                        d[ch] = relu ? MathF.Max(s, 0f) : s;
                    }
                }
            });
        }
    }

    /// <summary>Concat of arena tensors along a small inner block (NHWC channel concat), parallel over outer rows.</summary>
    public static void ConcatInner(float*[] ins, int[] counts, int outer, float* dst)
    {
        int total = counts.Sum();
        int chunk = Math.Max(1, 16384 / total), tasks = (outer + chunk - 1) / chunk;
        For(tasks, (long)outer * total, t =>
        {
            for (int o = t * chunk, oe = Math.Min(outer, o + chunk); o < oe; o++)
            {
                float* d = dst + (long)o * total;
                for (int i = 0; i < ins.Length; i++)
                {
                    Copy(ins[i] + (long)o * counts[i], d, counts[i]);
                    d += counts[i];
                }
            }
        });
    }

    /// <summary>dst = clamp(alpha·x + beta, 0, 1) (layout-agnostic).</summary>
    public static void HardSigmoid(ReadOnlySpan<float> src, Span<float> dst, float alpha, float beta)
    {
        int i = 0;
        if (V128)
        {
            V4 va = Vector128.Create(alpha), vb = Vector128.Create(beta), one = Vector128.Create(1f);
            fixed (float* s = src, d = dst)
                for (; i + 4 <= src.Length; i += 4)
                    Vector128.Min(Vector128.Max(Vector128.Load(s + i) * va + vb, V4.Zero), one).Store(d + i);
        }
        for (; i < src.Length; i++) dst[i] = Math.Clamp(src[i] * alpha + beta, 0f, 1f);
    }
}
