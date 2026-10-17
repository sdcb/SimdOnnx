using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using Sdcb.SimdOnnx.Engine;

namespace Sdcb.SimdOnnx.Kernels;

/// <summary>
/// Flat-float NCHW kernel library. Dispatch is a plain IsSupported chain:
/// AVX2+FMA hot paths, then the portable Vector&lt;float&gt; tier, else scalar
/// (the reference). Runtime knobs: DOTNET_EnableAVX2 / DOTNET_EnableAVX /
/// DOTNET_EnableHWIntrinsic.
/// </summary>
internal static unsafe partial class Ops
{
    // ======================== Conv ========================
    // src [1,IC,H,W]; w [OC,IC/g,kh,kw]; dst [1,OC,OH,OW]. N==1 assumed.
    // `add` is an optional same-shape residual addend, applied before the activation.
    public static void Conv(ReadOnlySpan<float> src, int[] inShape,
        ReadOnlySpan<float> w, int[] wShape, ReadOnlySpan<float> bias,
        Span<float> dst, int[] outShape, ConvP p, ReadOnlySpan<float> add = default)
    {
        if (!add.IsEmpty && add.Length != dst.Length)
            throw new NotSupportedException(
                $"conv residual addend has {add.Length} elems, expected {dst.Length}.");
        if (!add.IsEmpty)
        {
            // activation must run after the residual add: conv without act, then fused add+act
            var q = new ConvP
            {
                Pads = p.Pads, Strides = p.Strides, Dilations = p.Dilations, Kernel = p.Kernel,
                Group = p.Group, Act = ActivationKind.None, PadMode = p.PadMode,
                PadValue = p.PadValue, BiasValue = p.BiasValue, KernelPlan = p.KernelPlan,
            };
            Conv(src, inShape, w, wShape, bias, dst, outShape, q);
            AddInPlace(dst, add, p.Act);
            return;
        }
        int ic = inShape[1], ih = inShape[2], iw = inShape[3];
        int oc = wShape[0], kh = wShape[2], kw = wShape[3];
        int oh = outShape[2], ow = outShape[3];
        int sh = p.Strides[0], sw = p.Strides[1];
        int pt = p.Pads[0], pl = p.Pads[1], pb = p.Pads[2], pr = p.Pads[3];
        int group = p.Group;

        bool symmetricPad = pt == pb && pl == pr;
        if (Avx2.IsSupported && Fma.IsSupported)
        {
            if (group > 1 && group == ic)
            {
                Avx2Kernels.Depthwise(src, ic, ih, iw, w, bias, dst, oh, ow,
                    kh, kw, pt, pl, p.PadMode, p.PadValue, p.Act);
                return;
            }
            if (group == 1 && kh == 3 && kw == 3 && oc % 4 == 0 && symmetricPad)
            {
                if (sh == 1 && sw == 1)
                {
                    Avx2Kernels.Conv3x3S1(src, ic, ih, iw, w, bias, dst, oc, oh, ow,
                        pt, pl, p.PadMode, p.PadValue, p.Act);
                    return;
                }
                if (sh == 2 && sw == 2)
                {
                    Avx2Kernels.Conv3x3S2(src, ic, ih, iw, w, bias, dst, oc, oh, ow,
                        pt, pl, p.PadMode, p.PadValue, p.Act);
                    return;
                }
            }
            if (group == 1 && kh == 1 && kw == 1 && sh == 1 && sw == 1 && pt == 0 && pl == 0 && pb == 0 && pr == 0)
            {
                Avx2Kernels.Conv1x1(src, ic, ih * iw, w, bias, dst, oc, p.Act);
                return;
            }
            // unspecialized shape: fall through to the Vector<float> tier
        }
        else if (!Vector.IsHardwareAccelerated)
        {
            ConvScalar(src, ic, ih, iw, w, bias, dst, oc, oh, ow, kh, kw, sh, sw,
                pt, pl, group, p.PadMode, p.PadValue, p.Act);
            return;
        }

        // Vector<float> tier
        if (group > 1)
        {
            if (group != ic)   // grouped (non-depthwise) conv: scalar reference handles groups correctly
            {
                ConvScalar(src, ic, ih, iw, w, bias, dst, oc, oh, ow, kh, kw, sh, sw,
                    pt, pl, group, p.PadMode, p.PadValue, p.Act);
                return;
            }
            ConvDepthwiseVec(src, ic, ih, iw, w, bias, dst, oh, ow, kh, kw,
                pt, pl, p.PadMode, p.PadValue, p.Act);
            return;
        }
        if (kh == 1 && kw == 1 && sh == 1 && sw == 1 && pt == 0 && pl == 0)
        {
            Conv1x1Vec(src, ic, ih, iw, w, bias, dst, oc, oh, ow, p.Act);
            return;
        }
        ConvVecW(src, ic, ih, iw, w, bias, dst, oc, oh, ow, kh, kw, sh, sw,
            pt, pl, group, p.PadMode, p.PadValue, p.Act);
    }

    /// <summary>dst = act(dst + add) — residual epilogue for fused Conv+Add.</summary>
    private static void AddInPlace(Span<float> dst, ReadOnlySpan<float> add, ActivationKind act)
    {
        int n = dst.Length, i = 0, VL = Vector<float>.Count;
        var zero = Vector<float>.Zero;
        for (; i + VL <= n; i += VL)
        {
            var v = Simd.VectorLoad(dst, i) + Simd.VectorLoad(add, i);
            if (act == ActivationKind.Relu) v = Vector.Max(v, zero);
            Simd.VectorStore(dst, i, v);
        }
        for (; i < n; i++)
        {
            float v = dst[i] + add[i];
            dst[i] = act == ActivationKind.Relu ? MathF.Max(v, 0f) : v;
        }
        if (act != ActivationKind.None && act != ActivationKind.Relu)
            ApplyActVec(dst, act);
    }

    // ---- scalar reference conv (N==1, dense or grouped) ----
    private static void ConvScalar(ReadOnlySpan<float> src, int ic, int ih, int iw,
        ReadOnlySpan<float> w, ReadOnlySpan<float> bias,
        Span<float> dst, int oc, int oh, int ow, int kh, int kw,
        int sh, int sw, int pt, int pl, int group,
        PadMode padMode, float padValue, ActivationKind act)
    {
        int icg = ic / group, ocg = oc / group;
        for (int g = 0; g < group; g++)
        for (int o = g * ocg; o < (g + 1) * ocg; o++)
        for (int y = 0; y < oh; y++)
        for (int x = 0; x < ow; x++)
        {
            float acc = bias.IsEmpty ? 0f : bias[o];
            for (int c = g * icg; c < (g + 1) * icg; c++)
            for (int ky = 0; ky < kh; ky++)
            {
                int sy = y * sh + ky - pt;
                if ((uint)sy >= (uint)ih)
                {
                    if (padMode == PadMode.Reflect) sy = Reflect(sy, ih);
                    else continue;
                }
                for (int kx = 0; kx < kw; kx++)
                {
                    int sx = x * sw + kx - pl;
                    if ((uint)sx >= (uint)iw)
                    {
                        if (padMode == PadMode.Reflect) sx = Reflect(sx, iw);
                        else continue;
                    }
                    acc += src[(c * ih + sy) * iw + sx] * w[((o * icg + c - g * icg) * kh + ky) * kw + kx];
                }
            }
            dst[(o * oh + y) * ow + x] = ApplyAct(acc, act);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Reflect(int i, int dim)
    {
        // ONNX reflect (half-sample symmetric, no edge repeat): -1→1, dim→dim-2
        if (dim <= 1) return 0;   // reflect is undefined on a singleton axis; don't spin
        while (i < 0 || i >= dim) i = i < 0 ? -i : 2 * dim - 2 - i;
        return i;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ApplyAct(float v, ActivationKind act) =>
        act == ActivationKind.Relu ? MathF.Max(v, 0f) :
        act == ActivationKind.Sigmoid ? 1f / (1f + MathF.Exp(-v)) :
        act == ActivationKind.Gelu ? 0.5f * v * (1f + SimdMath.ErfScalar(v * 0.70710678f)) : v;

    // ---- vector-over-output-W conv: one output row vector per (oc,y) ----
    private static void ConvVecW(ReadOnlySpan<float> src, int ic, int ih, int iw,
        ReadOnlySpan<float> w, ReadOnlySpan<float> bias,
        Span<float> dst, int oc, int oh, int ow, int kh, int kw,
        int sh, int sw, int pt, int pl, int group,
        PadMode padMode, float padValue, ActivationKind act)
    {
        int icg = ic / group;
        int VL = Vector<float>.Count;
        // interior x-range where all taps in-bounds (no pad needed)
        for (int o = 0; o < oc; o++)
        {
            int g = o / (oc / group); // output channel -> group index
            for (int y = 0; y < oh; y++)
            {
                Span<float> drow = dst.Slice((o * oh + y) * ow, ow);
                if (!bias.IsEmpty) drow.Fill(bias[o]); else drow.Clear();
                for (int c = g * icg; c < (g + 1) * icg; c++)
                {
                    ReadOnlySpan<float> sbase = src.Slice(c * ih * iw, ih * iw);
                    ReadOnlySpan<float> wk = w.Slice((o * icg + (c - g * icg)) * kh * kw, kh * kw);
                    for (int ky = 0; ky < kh; ky++)
                    {
                        int sy = y * sh + ky - pt;
                        bool yOk = (uint)sy < (uint)ih;
                        int syR = yOk ? sy : (padMode == PadMode.Reflect ? Reflect(sy, ih) : -1);
                        for (int kx = 0; kx < kw; kx++)
                        {
                            float wv = wk[ky * kw + kx];
                            if (wv == 0f) continue;
                            if (sw == 1)
                            {
                                // contiguous reads: dst[x] += wv * src[sy][x + kx - pl]
                                int xBegin = Math.Max(0, pl - kx);
                                int xEnd = Math.Min(ow, iw + pl - kx);
                                if (yOk || padMode == PadMode.Reflect)
                                {
                                    VecAccMul(drow.Slice(xBegin, xEnd - xBegin),
                                        sbase.Slice(syR * iw + xBegin + kx - pl, xEnd - xBegin), wv);
                                }
                                else if (padValue != 0f)
                                {
                                    for (int x = xBegin; x < xEnd; x++) drow[x] += wv * padValue;
                                }
                                // pad regions (constant → add wv*padValue; reflect → gather)
                                for (int x = 0; x < xBegin; x++) drow[x] += wv * PadFetch(sbase, sy, x + kx - pl, ih, iw, padMode, padValue, syR, sy);
                                for (int x = xEnd; x < ow; x++) drow[x] += wv * PadFetch(sbase, sy, x + kx - pl, ih, iw, padMode, padValue, syR, sy);
                            }
                            else
                            {
                                for (int x = 0; x < ow; x++)
                                    drow[x] += wv * PadFetch(sbase, sy, x * sw + kx - pl, ih, iw, padMode, padValue, syR, sy);
                            }
                        }
                    }
                }
                ApplyActVec(drow, act);
            }
        }

        static float PadFetch(ReadOnlySpan<float> sbase, int sy, int sx, int ih, int iw,
            PadMode mode, float padValue, int syR, int syOrig)
        {
            int yy = syOrig, xx = sx;
            if ((uint)yy >= (uint)ih || (uint)xx >= (uint)iw)
            {
                if (mode == PadMode.Constant) return padValue;
                yy = Reflect(yy, ih); xx = Reflect(xx, iw);
            }
            return sbase[yy * iw + xx];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void VecAccMul(Span<float> dst, ReadOnlySpan<float> src, float w)
    {
        int i = 0, VL = Vector<float>.Count;
        var vw = new Vector<float>(w);
        for (; i + VL <= dst.Length; i += VL)
            Simd.VectorStore(dst, i, Simd.VectorAddMul(Simd.VectorLoad(dst, i), Simd.VectorLoad(src, i), vw));
        for (; i < dst.Length; i++) dst[i] += src[i] * w;
    }

    private static void ApplyActVec(Span<float> row, ActivationKind act)
    {
        if (act == ActivationKind.None) return;
        if (act == ActivationKind.Relu) { Relu(row, row); return; }
        if (act == ActivationKind.Gelu) { SimdMath.Gelu(row, row); return; }
        if (act == ActivationKind.Sigmoid) { SimdMath.Sigmoid(row, row); return; }
        for (int i = 0; i < row.Length; i++) row[i] = ApplyAct(row[i], act);
    }

    // ---- 1x1 conv = GEMM over pixels: dst[oc, HW] = W[oc, ic] * src[ic, HW] ----
    private static void Conv1x1Vec(ReadOnlySpan<float> src, int ic, int ih, int iw,
        ReadOnlySpan<float> w, ReadOnlySpan<float> bias,
        Span<float> dst, int oc, int oh, int ow, ActivationKind act)
    {
        int hw = ih * iw;
        int VL = Vector<float>.Count;
        for (int o = 0; o < oc; o++)
        {
            Span<float> drow = dst.Slice(o * hw, hw);
            float b = bias.IsEmpty ? 0f : bias[o];
            for (int i = 0; i < hw; i++) drow[i] = b;
            ReadOnlySpan<float> wo = w.Slice(o * ic, ic);
            for (int c = 0; c < ic; c++)
            {
                float wv = wo[c];
                if (wv == 0f) continue;
                VecAccMul(drow, src.Slice(c * hw, hw), wv);
            }
            ApplyActVec(drow, act);
        }
    }

    // ---- depthwise conv (group == C): dst[c,y,x] ----
    private static void ConvDepthwiseVec(ReadOnlySpan<float> src, int ic, int ih, int iw,
        ReadOnlySpan<float> w, ReadOnlySpan<float> bias,
        Span<float> dst, int oh, int ow, int kh, int kw,
        int pt, int pl, PadMode padMode, float padValue, ActivationKind act)
    {
        int VL = Vector<float>.Count;
        for (int c = 0; c < ic; c++)
        {
            ReadOnlySpan<float> sbase = src.Slice(c * ih * iw, ih * iw);
            ReadOnlySpan<float> wk = w.Slice(c * kh * kw, kh * kw);
            for (int y = 0; y < oh; y++)
            {
                Span<float> drow = dst.Slice((c * oh + y) * ow, ow);
                float b = bias.IsEmpty ? 0f : bias[c];
                for (int i = 0; i < ow; i++) drow[i] = b;
                for (int ky = 0; ky < kh; ky++)
                {
                    int sy = y + ky - pt;
                    for (int kx = 0; kx < kw; kx++)
                    {
                        float wv = wk[ky * kw + kx];
                        if (wv == 0f) continue;
                        int xBegin = Math.Max(0, pl - kx);
                        int xEnd = Math.Min(ow, iw + pl - kx);
                        if ((uint)sy < (uint)ih)
                        {
                            VecAccMul(drow.Slice(xBegin, xEnd - xBegin),
                                sbase.Slice(sy * iw + xBegin + kx - pl, xEnd - xBegin), wv);
                        }
                        for (int x = 0; x < xBegin; x++) drow[x] += wv * Fetch(sbase, sy, x + kx - pl, ih, iw, padMode, padValue);
                        for (int x = xEnd; x < ow; x++) drow[x] += wv * Fetch(sbase, sy, x + kx - pl, ih, iw, padMode, padValue);
                        if ((uint)sy >= (uint)ih && padMode == PadMode.Constant && padValue != 0f)
                            for (int x = xBegin; x < xEnd; x++) drow[x] += wv * padValue;
                        else if ((uint)sy >= (uint)ih && padMode == PadMode.Reflect)
                            for (int x = xBegin; x < xEnd; x++) drow[x] += wv * sbase[Reflect(sy, ih) * iw + x + kx - pl];
                    }
                }
                ApplyActVec(drow, act);
            }
        }

        static float Fetch(ReadOnlySpan<float> s, int y, int x, int ih, int iw, PadMode m, float pv)
        {
            if ((uint)y >= (uint)ih || (uint)x >= (uint)iw)
            {
                if (m == PadMode.Constant) return pv;
                y = Reflect(y, ih); x = Reflect(x, iw);
            }
            return s[y * iw + x];
        }
    }

    // ======================== MaxPool ========================
    public static void MaxPool(ReadOnlySpan<float> src, int[] inShape, Span<float> dst, int[] outShape,
        PoolP p)
    {
        int c = inShape[1], ih = inShape[2], iw = inShape[3];
        int oh = outShape[2], ow = outShape[3];
        int kh = p.Kernel[0], kw = p.Kernel[1], sh = p.Strides[0], sw = p.Strides[1];
        int pt = p.Pads[0], pl = p.Pads[1];
        if (Avx2.IsSupported && Fma.IsSupported &&
            kh == 2 && kw == 2 && sh == 2 && sw == 2 && pt == 0 && pl == 0)
        {
            Avx2Kernels.MaxPool2x2(src, c, ih, iw, dst, oh, ow, p.FuseRelu);
            return;
        }
        for (int ch = 0; ch < c; ch++)
        {
            ReadOnlySpan<float> s = src.Slice(ch * ih * iw, ih * iw);
            Span<float> d = dst.Slice(ch * oh * ow, oh * ow);
            for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow; x++)
            {
                float m = float.NegativeInfinity;
                for (int ky = 0; ky < kh; ky++)
                {
                    int sy = y * sh + ky - pt;
                    if ((uint)sy >= (uint)ih) continue;
                    for (int kx = 0; kx < kw; kx++)
                    {
                        int sx = x * sw + kx - pl;
                        if ((uint)sx >= (uint)iw) continue;
                        float v = s[sy * iw + sx];
                        if (v > m) m = v;
                    }
                }
                if (m == float.NegativeInfinity) m = 0f;
                d[y * ow + x] = p.FuseRelu && m < 0f ? 0f : m;
            }
        }
    }

    // ======================== elementwise ========================
    public static unsafe void Elementwise(ReadOnlySpan<float> a, int[] aShape,
        ReadOnlySpan<float> b, int[] bShape, Span<float> dst, int[] outShape, EwOp op,
        ActivationKind act = ActivationKind.None)
    {
        int n = 1;
        foreach (int d in outShape) n *= d;
        int oi = (int)op;
        bool relu = act == ActivationKind.Relu;
        bool aFull = Same(aShape, outShape), bFull = Same(bShape, outShape);
        int aN = Count(aShape), bN = Count(bShape), last = outShape.Length == 0 ? 1 : outShape[^1];
        fixed (float* pa = a, pb = b, pd = dst)
        {
            if (aFull && bFull) Ew.Binary(pa, pb, pd, n, op, relu);
            else if (aFull && bN == 1) Ew.Scalar(pa, pb[0], pd, n, op, rev: false, relu);
            else if (bFull && aN == 1) Ew.Scalar(pb, pa[0], pd, n, op, rev: true, relu);
            else if (last > 1 && aFull != bFull && Grouped(aFull ? bShape : aShape, outShape, last, out int groups))
            {
                // per-channel vector (bias / scale, or one per leading index: NHWC SE gate [N,1,1,C])
                float* full = aFull ? pa : pb, vec = aFull ? pb : pa;
                long rows = n / last / groups;
                for (int g = 0; g < groups; g++)
                    Ew.Channel(full + (long)g * rows * last, vec + (long)g * last, pd + (long)g * rows * last,
                        rows, last, op, rev: !aFull, relu);
            }
            else goto Strided;
        }
        if (act is not (ActivationKind.None or ActivationKind.Relu)) Activation(dst, dst, act);
        return;
    Strided:
        bool avx = Avx2.IsSupported && Fma.IsSupported && n >= 8192;
        // inner-block broadcast: a full-shape, b constant over trailing dims
        if (avx && aShape.SequenceEqual(outShape))
        {
            int[] bStr = BroadcastStrides(bShape, outShape.Length);
            int[] outStr = StridesOf(outShape);
            int inner = 1, rank = outShape.Length;
            while (rank > 1 && bStr[rank - 1] == 0)
            {
                inner *= outShape[rank - 1];
                rank--;
            }
            if (inner > 8 && n % inner == 0)
            {
                // EwInner walks the outer dims [0, rank) (its loop bound is rank - 1 of the argument)
                Avx2Kernels.EwInner(a, b, bStr, outShape, outStr,
                    rank + 1, inner, dst, n, oi, act);
                return;
            }
        }
        // general broadcast: coalesced strided rows over the threads
        int r = outShape.Length;
        fixed (float* pa = a, pb = b, pd = dst)
            Strided.Binary(pa, Strided.BroadcastStrides(aShape, r), pb, Strided.BroadcastStrides(bShape, r), outShape, pd, op,
                relu: act == ActivationKind.Relu);
        if (act is not (ActivationKind.None or ActivationKind.Relu)) Activation(dst, dst, act);
    }

    private static bool Same(int[] a, int[] b) => a.AsSpan().SequenceEqual(b);

    private static int Count(int[] s)
    {
        int n = 1;
        foreach (int d in s) n *= d;
        return n;
    }

    /// <summary>
    /// <paramref name="vs"/> (right-aligned onto <paramref name="outShape"/>) is a last-axis vector,
    /// possibly one per leading index: equal to the output on dims [0, k), 1 on [k, rank-1).
    /// </summary>
    private static bool Grouped(int[] vs, int[] outShape, int last, out int groups)
    {
        groups = 1;
        int rank = outShape.Length, off = rank - vs.Length, d = 0;
        int Dim(int i) => i - off >= 0 ? vs[i - off] : 1;
        if (Dim(rank - 1) != last) return false;
        for (; d < rank - 1 && Dim(d) == outShape[d] && Dim(d) != 1; d++) groups *= outShape[d];
        for (int i = d; i < rank - 1; i++) if (Dim(i) != 1) return false;
        return true;
    }

    private static bool IsTrailingVec(int[] s, int last) =>
        s.Length > 0 && s[^1] == last && s.Aggregate(1, (x, y) => x * y) == last;

    private static void VecBin(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> d, int n, int op)
    {
        int i = 0, VL = Vector<float>.Count;
        for (; i + VL <= n; i += VL)
        {
            var va = Simd.VectorLoad(a, i); var vb = Simd.VectorLoad(b, i);
            var r = op switch { 0 => va + vb, 1 => va - vb, 2 => va * vb, _ => va / vb };
            Simd.VectorStore(d, i, r);
        }
        for (; i < n; i++)
            d[i] = op switch { 0 => a[i] + b[i], 1 => a[i] - b[i], 2 => a[i] * b[i], _ => a[i] / b[i] };
    }

    private static void VecScalar(ReadOnlySpan<float> a, Span<float> d, int n, float s, int op)
    {
        int i = 0, VL = Vector<float>.Count;
        var vs = new Vector<float>(s);
        for (; i + VL <= n; i += VL)
        {
            var va = Simd.VectorLoad(a, i);
            var r = op switch { 0 => va + vs, 1 => va - vs, 2 => va * vs, _ => va / vs };
            Simd.VectorStore(d, i, r);
        }
        for (; i < n; i++)
            d[i] = op switch { 0 => a[i] + s, 1 => a[i] - s, 2 => a[i] * s, _ => a[i] / s };
    }

    private static void VecScalarR(ReadOnlySpan<float> b, Span<float> d, int n, float s, int op)
    {
        for (int i = 0; i < n; i++)
            d[i] = op == 1 ? s - b[i] : s / b[i];
    }


    private static int[] BroadcastStrides(int[] shape, int rank)
    {
        int[] s = new int[rank];
        int acc = 1;
        for (int i = shape.Length - 1; i >= 0; i--) { s[i + rank - shape.Length] = acc; acc *= shape[i]; }
        for (int i = 0; i < rank; i++)
        {
            int si = i - (rank - shape.Length);
            if (si < 0 || shape[si] == 1) s[i] = 0;
        }
        return s;
    }

    private static int[] StridesOf(int[] shape)
    {
        int[] s = new int[shape.Length];
        int acc = 1;
        for (int i = shape.Length - 1; i >= 0; i--) { s[i] = acc; acc *= shape[i]; }
        return s;
    }

    // ======================== unary ========================
    public static void Relu(ReadOnlySpan<float> src, Span<float> dst) => Unary(src, dst, ActivationKind.Relu);
    public static void Sigmoid(ReadOnlySpan<float> src, Span<float> dst) => Unary(src, dst, ActivationKind.Sigmoid);
    public static void Gelu(ReadOnlySpan<float> src, Span<float> dst) => Unary(src, dst, ActivationKind.Gelu);

    public static void Erf(ReadOnlySpan<float> src, Span<float> dst)
    {
        fixed (float* s = src, d = dst) Act.ApplyErf(s, d, src.Length);
    }

    private static void Unary(ReadOnlySpan<float> src, Span<float> dst, ActivationKind act)
    {
        fixed (float* s = src, d = dst) Act.Apply(s, d, src.Length, act);
    }

    // ======================== LayerNorm ========================
    // x [.., rows, C] normalized over last dim; w[C], b[C]
    public static void LayerNorm(ReadOnlySpan<float> x, int[] shape,
        ReadOnlySpan<float> w, ReadOnlySpan<float> b, Span<float> dst, float eps)
    {
        int c = shape[^1];
        int rows = (int)(x.Length / c);
        if (Avx2.IsSupported && Fma.IsSupported)
        {
            Avx2Kernels.LayerNorm(x, rows, c, w, b, dst, eps);
            return;
        }
        if (Nhwc.UseNeon)
        {
            NeonKernels.LayerNorm(x, rows, c, w, b, dst, eps);
            return;
        }
        if (Nhwc.UsePacked)
        {
            PackedKernels.LayerNorm(x, rows, c, w, b, dst, eps);
            return;
        }
        for (int r = 0; r < rows; r++)
        {
            ReadOnlySpan<float> xr = x.Slice(r * c, c);
            Span<float> dr = dst.Slice(r * c, c);
            float s = 0f, s2 = 0f;
            for (int i = 0; i < c; i++) { float v = xr[i]; s += v; s2 += v * v; }
            float mean = s / c;
            float inv = 1f / MathF.Sqrt(s2 / c - mean * mean + eps);
            for (int i = 0; i < c; i++) dr[i] = (xr[i] - mean) * inv * w[i] + b[i];
        }
    }

    // ======================== Gemm ========================
    // A [M,K] (last-2-dims of A), B [K,N] or [N,K] if transB, bias [N]; dst [M,N]
    public static void Gemm(ReadOnlySpan<float> a, int[] aShape, ReadOnlySpan<float> b,
        int[] bShape, ReadOnlySpan<float> bias, Span<float> dst, int[] outShape, GemmP p)
    {
        int m = 1;
        for (int i = 0; i < aShape.Length - 1; i++) m *= aShape[i];
        int k = aShape[^1];
        int n = outShape[^1];
        if (!p.TransB && Avx2.IsSupported && Fma.IsSupported)
            Avx2Kernels.GemmKN(a, m, k, b, n, bias, dst);
        else if (p.TransB)
            GemmBT(a, m, k, b, n, bias, dst);
        else
            GemmKN(a, m, k, b, n, bias, dst);
        switch (p.Act)
        {
            case ActivationKind.Relu: Relu(dst, dst); break;
            case ActivationKind.Gelu: SimdMath.Gelu(dst, dst); break;
            case ActivationKind.Sigmoid: SimdMath.Sigmoid(dst, dst); break;
        }
    }

    // B row-major [K,N]: dst[m,n] = Σ_k a[m,k]*b[k,n] — vector over N
    private static void GemmKN(ReadOnlySpan<float> a, int M, int K,
        ReadOnlySpan<float> b, int N, ReadOnlySpan<float> bias, Span<float> dst)
    {
        int VL = Vector<float>.Count;
        // tile M by 4 for B-row reuse
        int m0 = 0;
        for (; m0 + 4 <= M; m0 += 4)
        {
            ReadOnlySpan<float> a0 = a.Slice(m0 * K, K), a1 = a.Slice((m0 + 1) * K, K),
                a2 = a.Slice((m0 + 2) * K, K), a3 = a.Slice((m0 + 3) * K, K);
            for (int n0 = 0; n0 < N; n0 += VL)
            {
                int nw = Math.Min(VL, N - n0);
                if (nw < VL) break; // tail handled scalar below
                var acc0 = bias.IsEmpty ? Vector<float>.Zero : Simd.VectorLoad(bias, n0);
                var acc1 = acc0; var acc2 = acc0; var acc3 = acc0;
                for (int k = 0; k < K; k++)
                {
                    var bv = Simd.VectorLoad(b, k * N + n0);
                    acc0 = Simd.VectorAddMul(acc0, bv, a0[k]);
                    acc1 = Simd.VectorAddMul(acc1, bv, a1[k]);
                    acc2 = Simd.VectorAddMul(acc2, bv, a2[k]);
                    acc3 = Simd.VectorAddMul(acc3, bv, a3[k]);
                }
                Simd.VectorStore(dst, m0 * N + n0, acc0);
                Simd.VectorStore(dst, (m0 + 1) * N + n0, acc1);
                Simd.VectorStore(dst, (m0 + 2) * N + n0, acc2);
                Simd.VectorStore(dst, (m0 + 3) * N + n0, acc3);
            }
            // N tail
            for (int n0 = N - N % VL; n0 < N; n0++)
            {
                float s0 = bias.IsEmpty ? 0 : bias[n0], s1 = s0, s2 = s0, s3 = s0;
                for (int k = 0; k < K; k++)
                {
                    float bv = b[k * N + n0];
                    s0 += a0[k] * bv; s1 += a1[k] * bv; s2 += a2[k] * bv; s3 += a3[k] * bv;
                }
                dst[m0 * N + n0] = s0; dst[(m0 + 1) * N + n0] = s1;
                dst[(m0 + 2) * N + n0] = s2; dst[(m0 + 3) * N + n0] = s3;
            }
        }
        for (; m0 < M; m0++)
        {
            ReadOnlySpan<float> ar = a.Slice(m0 * K, K);
            for (int n0 = 0; n0 + VL <= N; n0 += VL)
            {
                var acc = bias.IsEmpty ? Vector<float>.Zero : Simd.VectorLoad(bias, n0);
                for (int k = 0; k < K; k++)
                    acc = Simd.VectorAddMul(acc, Simd.VectorLoad(b, k * N + n0), ar[k]);
                Simd.VectorStore(dst, m0 * N + n0, acc);
            }
            for (int n0 = N - N % VL; n0 < N; n0++)
            {
                float s = bias.IsEmpty ? 0 : bias[n0];
                for (int k = 0; k < K; k++) s += ar[k] * b[k * N + n0];
                dst[m0 * N + n0] = s;
            }
        }
    }

    // B row-major [N,K] (transB): dst[m,n] = Σ_k a[m,k]*b[n,k] — dot products
    private static void GemmBT(ReadOnlySpan<float> a, int M, int K,
        ReadOnlySpan<float> b, int N, ReadOnlySpan<float> bias, Span<float> dst)
    {
        for (int m = 0; m < M; m++)
        {
            ReadOnlySpan<float> ar = a.Slice(m * K, K);
            for (int n = 0; n < N; n++)
            {
                float s = bias.IsEmpty ? 0 : bias[n];
                ReadOnlySpan<float> br = b.Slice(n * K, K);
                int k = 0, VL = Vector<float>.Count;
                var acc = Vector<float>.Zero;
                for (; k + VL <= K; k += VL)
                    acc += Simd.VectorLoad(ar, k) * Simd.VectorLoad(br, k);
                for (int i = 0; i < VL; i++) s += acc[i];
                for (; k < K; k++) s += ar[k] * br[k];
                dst[m * N + n] = s;
            }
        }
    }

    // ======================== Resize ========================
    // bilinear, half_pixel, per-axis scale (NCHW; only spatial dims scale != 1)
    public static void ResizeBilinear(ReadOnlySpan<float> src, int[] inShape,
        Span<float> dst, int[] outShape, float[] scales,
        ActivationKind act = ActivationKind.None)
    {
        int c = inShape[1], ih = inShape[2], iw = inShape[3];
        int oh = outShape[2], ow = outShape[3];
        float sy = scales[2], sx = scales[3];
        // precompute row/col src indices + lerp weights
        int[] y0 = new int[oh], y1 = new int[oh];
        float[] wy = new float[oh];
        for (int y = 0; y < oh; y++)
        {
            float fy = (y + 0.5f) / sy - 0.5f;
            float fl = MathF.Floor(fy);
            y0[y] = Math.Clamp((int)fl, 0, ih - 1);
            y1[y] = Math.Clamp((int)fl + 1, 0, ih - 1);
            wy[y] = fy - fl;
        }
        int[] x0 = new int[ow], x1 = new int[ow];
        float[] wx = new float[ow];
        for (int x = 0; x < ow; x++)
        {
            float fx = (x + 0.5f) / sx - 0.5f;
            float fl = MathF.Floor(fx);
            x0[x] = Math.Clamp((int)fl, 0, iw - 1);
            x1[x] = Math.Clamp((int)fl + 1, 0, iw - 1);
            wx[x] = fx - fl;
        }
        if (Avx2.IsSupported && Fma.IsSupported && oh * ow >= 8192)
        {
            Avx2Kernels.Bilinear(src, c, ih, iw, dst, oh, ow, y0, y1, wy, x0, x1, wx, act);
            return;
        }
        for (int ch = 0; ch < c; ch++)
        {
            ReadOnlySpan<float> s = src.Slice(ch * ih * iw, ih * iw);
            Span<float> d = dst.Slice(ch * oh * ow, oh * ow);
            for (int y = 0; y < oh; y++)
            {
                ReadOnlySpan<float> r0 = s.Slice(y0[y] * iw, iw);
                ReadOnlySpan<float> r1 = s.Slice(y1[y] * iw, iw);
                float wY = wy[y];
                for (int x = 0; x < ow; x++)
                {
                    float wX = wx[x];
                    float top = r0[x0[x]] + (r0[x1[x]] - r0[x0[x]]) * wX;
                    float bot = r1[x0[x]] + (r1[x1[x]] - r1[x0[x]]) * wX;
                    float v = top + (bot - top) * wY;
                    d[y * ow + x] = act == ActivationKind.Relu ? MathF.Max(v, 0f) : v;
                }
            }
        }
    }

    // ======================== Pad ========================
    public static void Pad(ReadOnlySpan<float> src, int[] inShape, Span<float> dst,
        int[] outShape, PadP p)
    {
        int rank = inShape.Length;
        int n = 0, c = 0, ih = 0, iw = 0, oh = 0, ow = 0;
        if (rank == 4) { n = inShape[0]; c = inShape[1]; ih = inShape[2]; iw = inShape[3]; oh = outShape[2]; ow = outShape[3]; }
        else throw new NotSupportedException("Pad supports rank-4 only.");
        int pt = p.Pads[2], pl = p.Pads[3];
        // index maps
        int[] mapY = new int[oh], mapX = new int[ow];
        for (int y = 0; y < oh; y++)
        {
            int sy = y - pt;
            if ((uint)sy < (uint)ih) mapY[y] = sy;
            else if (p.Mode == PadMode.Reflect) mapY[y] = Reflect(sy, ih);
            else mapY[y] = -1;
        }
        for (int x = 0; x < ow; x++)
        {
            int sx = x - pl;
            if ((uint)sx < (uint)iw) mapX[x] = sx;
            else if (p.Mode == PadMode.Reflect) mapX[x] = Reflect(sx, iw);
            else mapX[x] = -1;
        }
        if (Avx2.IsSupported && Fma.IsSupported && oh * ow >= 4096)
        {
            Avx2Kernels.Pad4(src, n * c, ih, iw, dst, oh, ow, mapY, mapX, p.Value);
            return;
        }
        for (int nc = 0; nc < n * c; nc++)
        {
            ReadOnlySpan<float> s = src.Slice(nc * ih * iw, ih * iw);
            Span<float> d = dst.Slice(nc * oh * ow, oh * ow);
            for (int y = 0; y < oh; y++)
            {
                int my = mapY[y];
                Span<float> drow = d.Slice(y * ow, ow);
                if (my < 0) { drow.Fill(p.Value); continue; }
                ReadOnlySpan<float> srow = s.Slice(my * iw, iw);
                // x regions: [0,xl) pad, [xl,xr) copy, [xr,ow) pad
                int xl = 0; while (xl < ow && mapX[xl] < 0) xl++;
                int xr = ow; while (xr > xl && mapX[xr - 1] < 0) xr--;
                if (p.Mode == PadMode.Constant)
                {
                    for (int x = 0; x < xl; x++) drow[x] = p.Value;
                    for (int x = xr; x < ow; x++) drow[x] = p.Value;
                }
                else
                {
                    for (int x = 0; x < xl; x++) drow[x] = srow[mapX[x]];
                    for (int x = xr; x < ow; x++) drow[x] = srow[mapX[x]];
                }
                for (int x = xl; x < xr; x++) drow[x] = srow[mapX[x]];
            }
        }
    }

    // ======================== Transpose ========================
    public static void Transpose(ReadOnlySpan<float> src, int[] inShape, Span<float> dst,
        int[] outShape, int[] perm)
    {
        int rank = inShape.Length;
        // fast path: last-two-dim swap on any rank (most common here); leading dims must be identity
        bool isSwapLast2 = perm.Length == rank && perm[^1] == rank - 2 && perm[^2] == rank - 1;
        for (int i = 0; isSwapLast2 && i < rank - 2; i++) isSwapLast2 = perm[i] == i;
        if (isSwapLast2)
        {
            int a = inShape[rank - 2], b = inShape[rank - 1];
            int outer = 1;
            for (int i = 0; i < rank - 2; i++) outer *= inShape[i];
            if (Avx2.IsSupported && Fma.IsSupported && a * b >= 8192)
            {
                Avx2Kernels.Transpose2D(src, a, b, dst, copies: outer);
                return;
            }
            for (int o = 0; o < outer; o++)
                Transpose2D(src.Slice(o * a * b, a * b), dst.Slice(o * a * b, a * b), a, b);
            return;
        }
        // few channels (image tensors in and out of NHWC plans): the threaded strided gather beats the
        // single-threaded 2D transposes below
        bool fewC = rank == 4 && (perm is [0, 3, 1, 2] ? inShape[3] : perm is [0, 2, 3, 1] ? inShape[1] : int.MaxValue) <= 8;
        if (rank == 4 && !fewC && perm[0] == 0 && perm[1] == 3 && perm[2] == 1 && perm[3] == 2)
        {
            // [n,h,w,c] -> [n,c,h,w] == per image 2D transpose of [hw,c]->[c,hw]
            int hw = inShape[1] * inShape[2], cc = inShape[3], per = hw * cc;
            for (int b = 0; b < inShape[0]; b++)
            {
                if (Avx2.IsSupported && Fma.IsSupported && per >= 8192)
                    Avx2Kernels.Transpose2D(src.Slice(b * per, per), hw, cc, dst.Slice(b * per, per));
                else
                    Transpose2D(src.Slice(b * per, per), dst.Slice(b * per, per), hw, cc);
            }
            return;
        }
        if (rank == 4 && !fewC && perm[0] == 0 && perm[1] == 2 && perm[2] == 3 && perm[3] == 1)
        {
            // [n,c,h,w] -> [n,h,w,c] == per image 2D transpose of [c,hw]->[hw,c]
            int cc = inShape[1], hw = inShape[2] * inShape[3], per = hw * cc;
            for (int b = 0; b < inShape[0]; b++)
            {
                if (Avx2.IsSupported && Fma.IsSupported && per >= 8192)
                    Avx2Kernels.Transpose2D(src.Slice(b * per, per), cc, hw, dst.Slice(b * per, per));
                else
                    Transpose2D(src.Slice(b * per, per), dst.Slice(b * per, per), cc, hw);
            }
            return;
        }
        // generic permute: coalesced strided gather over the threads
        int[] inStr = StridesOf(inShape);
        long[] permutedInStr = new long[rank];
        for (int i = 0; i < rank; i++) permutedInStr[i] = inStr[perm[i]];
        fixed (float* sp = src, dp = dst) Strided.Gather(sp, 0, permutedInStr, outShape, dp);
    }

    // tiled [a,b] -> [b,a]
    private static void Transpose2D(ReadOnlySpan<float> src, Span<float> dst, int a, int b)
    {
        const int T = 16;
        for (int y0 = 0; y0 < a; y0 += T)
        for (int x0 = 0; x0 < b; x0 += T)
        {
            int ye = Math.Min(y0 + T, a), xe = Math.Min(x0 + T, b);
            for (int y = y0; y < ye; y++)
            for (int x = x0; x < xe; x++)
                dst[x * a + y] = src[y * b + x];
        }
    }

    // ======================== Slice ========================
    public static void Slice(ReadOnlySpan<float> src, int[] inShape, Span<float> dst,
        int[] outShape, SliceP p)
    {
        int rank = inShape.Length;
        int[] inStr = StridesOf(inShape);
        int[] start = new int[rank], step = Enumerable.Repeat(1, rank).ToArray();
        for (int i = 0; i < p.Axes.Length; i++) { start[p.Axes[i]] = p.Starts[i]; step[p.Axes[i]] = p.Steps[i]; }
        if (outShape.Aggregate(1, (a, b) => a * b) == 0) return;
        // coalesced strided gather over the threads
        long off = 0;
        long[] str = new long[rank];
        for (int d = 0; d < rank; d++) { off += (long)start[d] * inStr[d]; str[d] = (long)step[d] * inStr[d]; }
        fixed (float* sp = src, dp = dst) Strided.Gather(sp, off, str, outShape, dp);
    }

    // ======================== Concat ========================
    public static void Concat(FBuf[] ins, int[][] inShapes, Span<float> dst,
        int[] outShape, int axis)
    {
        int inner = 1;
        for (int d = axis + 1; d < outShape.Length; d++) inner *= outShape[d];
        int outer = 1;
        for (int d = 0; d < axis; d++) outer *= outShape[d];
        if (outer >= 1024 && ins.All(b => b.Data is null))
        {
            float*[] ptrs = new float*[ins.Length];
            for (int i = 0; i < ins.Length; i++) ptrs[i] = ins[i].Ptr;
            int[] counts = inShapes.Select(s => s[axis] * inner).ToArray();
            fixed (float* dp = dst) Nhwc.ConcatInner(ptrs, counts, outer, dp);
            return;
        }
        int dstOff = 0;
        for (int o = 0; o < outer; o++)
        {
            for (int i = 0; i < ins.Length; i++)
            {
                int count = inShapes[i][axis] * inner;
                if (count == 0) continue;
                ins[i].Span.Slice(o * count, count).CopyTo(dst.Slice(dstOff, count));
                dstOff += count;
            }
        }
    }


}
