using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Reflection;
using Sdcb.SimdOnnx.Engine;
using CK = Sdcb.SimdOnnx.Engine.ConstKind;

namespace Sdcb.SimdOnnx.Backends.Metal;

/// <summary>Binding slots resolved per-session: which underlying buffer.</summary>
public enum MtlSlot : int { Arena = 0, Consts = 1, Dummy = 2, Direct = 3 }

public readonly struct MtlBind
{
    public readonly MtlSlot Slot;
    internal readonly MtlBuffer? Buf;   // when Slot == Direct
    public readonly ulong Offset, Size;
    public MtlBind(MtlSlot s, ulong off, ulong size) { Slot = s; Offset = off; Size = size; }
    internal MtlBind(MtlBuffer b, ulong off, ulong size) { Slot = MtlSlot.Direct; Buf = b; Offset = off; Size = size; }
}

/// <summary>One recorded compute dispatch.</summary>
public sealed class MtlDispatch
{
    public required MtlPipeline Pipe;
    public required MtlBind[] Binds;
    public required byte[] Pc;
    public uint Gx, Gy = 1, Gz = 1;
    public string Name = "";
}

/// <summary>
/// Device-side const tensors (backend-packed weights) plus a 64B zero buffer, pooled per
/// (device, model): every <see cref="MtlGraphModel"/> of a model — any input size, any output
/// truncation — binds the same packed entries. Entries are keyed by source array and consumer
/// packing (content-compared when a plan re-materialized an array); a graph that needs entries the
/// pool lacks uploads only those, as one more chunk. Ref-counted per graph; every chunk is freed
/// when the last graph releases. Mirrors the Vulkan VkConstStore.
/// </summary>
internal sealed class MtlConstStore
{
    private static readonly Dictionary<(MtlDevice dev, object key), MtlConstStore> s_shared = new();

    private readonly MtlDevice _dev;
    private readonly (MtlDevice, object)? _key;
    private readonly List<Entry> _entries = new();
    private readonly Dictionary<(float[] src, int kind), int> _byRef = new(RefKey.Instance);
    private readonly List<MtlBuffer> _chunks = new();
    private int _refs = 1;
    internal readonly MtlBuffer Dummy;
    internal ulong Bytes { get; private set; }
    internal MtlBuffer Primary => _chunks.Count > 0 ? _chunks[0] : Dummy;

    private readonly record struct Entry(float[] Src, int Kind, int[] Shape, MtlBuffer Buf, ulong Off);

    private MtlConstStore(MtlDevice dev, (MtlDevice, object)? key)
    {
        _dev = dev;
        _key = key;
        unsafe
        {
            byte* z = stackalloc byte[64];
            new Span<byte>(z, 64).Clear();
            Dummy = dev.NewBuffer(z, 64);
        }
    }

    internal static MtlConstStore Acquire(MtlDevice dev, object? key)
    {
        if (key is null) return new MtlConstStore(dev, null);
        var k = (dev, key);
        lock (s_shared)
        {
            if (s_shared.TryGetValue(k, out MtlConstStore? s))
            {
                s._refs++;
                return s;
            }
            s = new MtlConstStore(dev, k);
            s_shared[k] = s;
            return s;
        }
    }

    /// <summary>Buffer + byte offset of every item, uploading the missing ones as a new chunk.</summary>
    internal (MtlBuffer buf, ulong off)[] Place(IReadOnlyList<(float[] src, int kind, int[] shape)> items,
        Func<int[], (ulong[] offsets, HostBytes host)> pack)
    {
        var placed = new (MtlBuffer buf, ulong off)[items.Count];
        lock (_entries)
        {
            var missing = new List<int>();
            for (int i = 0; i < items.Count; i++)
            {
                int e = Find(items[i].src, items[i].kind, items[i].shape);
                if (e >= 0) placed[i] = (_entries[e].Buf, _entries[e].Off);
                else missing.Add(i);
            }
            if (missing.Count == 0) return placed;
            var (offs, host) = pack(missing.ToArray());
            MtlBuffer chunk;
            using (host)
                unsafe { chunk = _dev.NewBuffer(host.Ptr, (nuint)host.Length); }
            _chunks.Add(chunk);
            Bytes += host.Length;
            for (int j = 0; j < missing.Count; j++)
            {
                var (src, kind, shape) = items[missing[j]];
                _byRef[(src, kind)] = _entries.Count;
                _entries.Add(new Entry(src, kind, shape, chunk, offs[j]));
                placed[missing[j]] = (chunk, offs[j]);
            }
        }
        return placed;
    }

    private int Find(float[] src, int kind, int[] shape)
    {
        if (_byRef.TryGetValue((src, kind), out int e) && _entries[e].Shape.AsSpan().SequenceEqual(shape)) return e;
        for (int i = 0; i < _entries.Count; i++)
        {
            Entry x = _entries[i];
            if (x.Kind == kind && x.Src.Length == src.Length && x.Shape.AsSpan().SequenceEqual(shape)
                && x.Src.AsSpan().SequenceEqual(src))
            {
                _byRef[(src, kind)] = i;
                return i;
            }
        }
        return -1;
    }

    internal void Release()
    {
        lock (s_shared)
        {
            if (--_refs > 0) return;
            if (_key is { } k && s_shared.TryGetValue(k, out MtlConstStore? cur) && cur == this)
                s_shared.Remove(k);
        }
        foreach (MtlBuffer c in _chunks) c.Dispose();
        Dummy.Dispose();
    }

    private sealed class RefKey : IEqualityComparer<(float[] src, int kind)>
    {
        internal static readonly RefKey Instance = new();
        public bool Equals((float[] src, int kind) a, (float[] src, int kind) b) => ReferenceEquals(a.src, b.src) && a.kind == b.kind;
        public int GetHashCode((float[] src, int kind) k) =>
            System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(k.src) * 31 + k.kind;
    }
}

/// <summary>
/// Per-(device, ExecPlan) half of the Metal backend: backend-private constant
/// packing/upload and the node list compiled to <see cref="MtlDispatch"/>
/// records. Sessions own the activation arena and resolve slot ids to live
/// buffers. Activations and packed weights are stored as fp16; all shader math
/// is fp32; conv bias and LayerNorm params stay fp32.
/// </summary>
public sealed class MtlGraphModel : IDisposable
{
    private readonly MtlDevice _dev;
    private readonly ExecPlan _plan;
    private int _refs = 1;  // the graph cache holds the first reference
    private readonly MtlConstStore _store;
    internal readonly MtlBuffer Consts;       // all const tensors, backend-packed (shared store)
    private readonly Dictionary<(int v, CK k), (MtlBuffer buf, ulong off)> _constOff = new();
    internal readonly MtlBuffer Dummy;        // 64B zero buffer for unused bindings
    internal readonly MtlDispatch[] Dispatches;
    internal readonly ulong ConstBytes;
    /// <summary>Activations + weights stored as float16 (fp32 math in every shader).</summary>
    internal const bool F16 = true;
    /// <summary>Bytes per stored activation element.</summary>
    public int Es => 2;


    /// <param name="constKey">Sharing key for the const store (the source model): graphs of the
    /// same model at other sizes reuse its uploaded weights. Null = private store.</param>
    internal MtlGraphModel(MtlDevice dev, ExecPlan plan, object? constKey = null)
    {
        _dev = dev;
        _plan = plan;
        if (!plan.Nhwc) throw new NotSupportedException("Metal backend requires a layout-planned (NHWC) plan.");

        // ---- consts: per-consumer packing ----
        var need = new List<(int v, CK k)>();
        void Need(int v, CK k)
        {
            if (v < 0 || plan.Values[v].Floats is null) return;
            if (!need.Contains((v, k))) need.Add((v, k));
        }
        foreach (ExecNode n in plan.Nodes)
        {
            if (n.Ext is { } ext)
            {
                foreach (var (v, k) in ext.GpuConsts(n)) Need(v, k);
                continue;
            }
            switch (n.P)
            {
                case ConvP cp:
                    Need(n.In[0], CK.Raw);
                    Need(n.In[1], cp.Group > 1 ? CK.DwW : UseMmConv(plan, n, cp) ? CK.ConvWT : CK.ConvW);
                    Need(cp.BiasValue, CK.F32);
                    Need(cp.AddValue, CK.Raw);
                    break;
                case GemmP gp:
                    Need(n.In[0], CK.Raw); Need(n.In[1], UseMmGemm(plan, n, gp) && !gp.TransB ? CK.GemmNK : CK.Raw);
                    Need(gp.BiasValue, CK.F32);
                    break;
                case LnP:
                    Need(n.In[0], CK.Raw); Need(n.In[1], CK.F32); Need(n.In[2], CK.F32);
                    break;
                case ChannelGateP gp:
                    Need(n.In[1], CK.F32); Need(n.In[2], CK.F32);
                    Need(gp.B1, CK.F32); Need(gp.B2, CK.F32);
                    break;
                case ConvTP tp:
                    Need(n.In[1], CK.ConvTW);
                    Need(tp.BiasValue, CK.F32);
                    break;
                default:
                    foreach (int iv in n.In) Need(iv, CK.Raw);
                    break;
            }
        }
        foreach (var (v, k) in need)
        {
            CValue cv = plan.Values[v];
            if (k == CK.Raw && cv.Layout is { } l && !Engine.Layout.IsIdentity(l))
                throw new NotSupportedException($"Metal: const '{cv.Name}' with non-identity layout");
        }
        var items = need.Select(x => (plan.Values[x.v].Floats!, x.k.Id, plan.Values[x.v].Shape)).ToArray();
        _store = MtlConstStore.Acquire(dev, constKey);
        var placed = _store.Place(items, miss => GpuPack.Build(plan, miss.Select(i => need[i]).ToList(), true));
        for (int i = 0; i < need.Count; i++) _constOff[need[i]] = placed[i];
        Consts = _store.Primary;
        Dummy = _store.Dummy;
        ConstBytes = _store.Bytes;

        var disps = new List<MtlDispatch>();
        NodeRanges = new (int start, int count)[plan.Nodes.Length];
        for (int i = 0; i < plan.Nodes.Length; i++)
        {
            int start = disps.Count;
            ExecNode n = plan.Nodes[i];
            // single-image kernels: one dispatch per image of a batch, binds offset to that image
            int nb = n.Op is ExecOp.Conv or ExecOp.MaxPool or ExecOp.ResizeBilinear or ExecOp.ResizeNearest
                or ExecOp.ConvTranspose or ExecOp.GlobalAvgPool && plan.Values[n.Out[0]].Shape is [var b, _, _, _] ? b : 1;
            if (nb > 1)
            {
                _images = nb;
                for (_image = 0; _image < nb; _image++) disps.AddRange(Emit(n));
                _images = 1;
            }
            else disps.AddRange(Emit(n));
            NodeRanges[i] = (start, disps.Count - start);
        }
        Dispatches = disps.ToArray();
    }



    /// <summary>conv_mm epilogue code (common.metal cact).</summary>
    private static uint MmAct(ActivationKind a) => a switch
    {
        ActivationKind.Relu => 1u, ActivationKind.Gelu => 2u, ActivationKind.HardSwish => 4u, ActivationKind.Swish => 5u,
        _ => 0u,
    };

    /// <summary>simdgroup-MMA conv eligibility (conv_mm): dense, output channels in
    /// 16 B quads, an epilogue it implements, M within the grid limit. cin%4!=0 uses
    /// the scalar-gather variant; constant pad must be zero.</summary>
    private static bool UseMmConv(ExecPlan plan, ExecNode n, ConvP cp)
    {
        if (cp.Group != 1 || cp.Pool is not null || cp.Dilations.Any(d => d != 1)) return false;
        if (cp.Act is ActivationKind.Sigmoid) return false;
        if (cp.PadMode == PadMode.Constant && cp.PadValue != 0f) return false;
        int[] so = plan.Values[n.Out[0]].PhysShape;   // [1,h,w,c]
        long m = (long)so[1] * so[2];
        return so[3] % 8 == 0 && m / 64 <= int.MaxValue;
    }


    /// <summary>MMA Gemm (a 1x1 conv over M "pixels" through conv_mm): const B,
    /// K/N in 16 B quads, relu/gelu/none epilogue.</summary>
    private static bool UseMmGemm(ExecPlan plan, ExecNode n, GemmP gp)
    {
        if (plan.Values[n.In[1]].Floats is null || plan.Values[n.In[1]].Shape.Length != 2) return false;
        // B packed per value: only when no other node reads it (its transB could differ)
        if (plan.Nodes.Count(m => m.In.Contains(n.In[1])) != 1) return false;
        if (gp.Act is ActivationKind.Sigmoid) return false;
        CValue a = plan.Values[n.In[0]];
        int k = a.Shape[^1], nn = plan.Values[n.Out[0]].Shape[^1];
        if (a.PhysShape[^1] != k) return false;
        long m = a.ElementCount / k;
        return k % 8 == 0 && nn % 8 == 0 && m / 64 <= int.MaxValue;
    }



    /// <summary>Dispatch index range per source ExecNode (for per-node debug runs).</summary>
    internal readonly (int start, int count)[] NodeRanges;

    // ---- value → bind resolution (byte offsets/sizes: all long math) ----
    private int _image, _images = 1;

    public MtlBind Bind(int valueId, CK? kind = null)
    {
        kind ??= CK.Raw;
        CValue v = _plan.Values[valueId];
        if (v.Buf >= 0 && _images > 1 && v.Shape is [var nb, _, _, _] && nb == _images)
        {
            // per-image emission (see the constructor): image _image of the batch
            ulong per = (ulong)(v.ElementCount / nb) * (ulong)Es;
            return new MtlBind(MtlSlot.Arena, (ulong)_plan.BufOffset[v.Buf] * (ulong)Es + (ulong)_image * per, per);
        }
        if (v.Buf >= 0)
            return new MtlBind(MtlSlot.Arena, (ulong)_plan.BufOffset[v.Buf] * (ulong)Es,
                (ulong)Math.Max(v.ElementCount, 1) * (ulong)Es);
        if (_constOff.TryGetValue((valueId, kind), out var c))
        {
            ulong es = !kind.Fp32 ? 2UL : 4UL;
            long elems = kind.Elems?.Invoke(v) ?? Math.Max(v.ElementCount, 1);
            return new MtlBind(c.buf, c.off, (ulong)elems * es);
        }
        if (v.Floats != null)
            throw new InvalidOperationException($"const value '{v.Name}' not packed as {kind}");
        throw new InvalidOperationException($"value '{v.Name}' has no GPU storage (ints must be folded)");
    }

    public static byte[] PcOf(params uint[] ints)
    {
        byte[] b = new byte[ints.Length * 4];
        Buffer.BlockCopy(ints, 0, b, 0, b.Length);
        return b;
    }

    public static byte[] PcOfF(uint[] ints, params float[] floats)
    {
        byte[] b = new byte[(ints.Length + floats.Length) * 4];
        Buffer.BlockCopy(ints, 0, b, 0, ints.Length * 4);
        Buffer.BlockCopy(floats, 0, b, ints.Length * 4, floats.Length * 4);
        return b;
    }

    public static uint[] U(params int[] v) { uint[] u = new uint[v.Length]; for (int i = 0; i < v.Length; i++) u[i] = (uint)v[i]; return u; }

    public static uint G(uint n) => (n + 255) / 256;

    /// <summary>(base, n, h, t, d) element offsets of a strided attention operand (see <see cref="AttentionP"/>).</summary>
    internal static uint[] StridedView(CValue v, int[] dims, int fix, int fixIndex)
    {
        int[] l = v.Layout ?? Engine.Layout.Identity(v.Shape.Length), ps = v.PhysShape;
        long Stride(int axis)
        {
            long s = 1;
            for (int j = ps.Length - 1; j >= 0 && l[j] != axis; j--) s *= ps[j];
            return s;
        }
        uint[] r = new uint[5];
        for (int j = 0; j < 4; j++) r[1 + j] = dims[j] < 0 ? 0u : (uint)Stride(dims[j]);
        r[0] = fix < 0 ? 0u : (uint)(fixIndex * Stride(fix));
        return r;
    }

    /// <summary>
    /// Broadcast strides (padded rank-4, storage order of the output) of operand
    /// <paramref name="a"/> onto output <paramref name="o"/>. Output physical dim i
    /// holds logical dim o.Layout[i]; the operand's logical dim is that minus the
    /// numpy right-alignment offset.
    /// </summary>
    private static int[] PhysBroadcastStrides(CValue a, CValue o)
    {
        int rank = o.Shape.Length, off = rank - a.Shape.Length;
        if (rank > 4) throw new NotSupportedException("Metal elementwise: rank > 4");
        int[] lo = o.Layout ?? Engine.Layout.Identity(rank);
        int[] la = a.Layout ?? Engine.Layout.Identity(a.Shape.Length);
        int[] ap = a.PhysShape;
        int[] aStr = new int[ap.Length];
        for (int i = ap.Length - 1, acc = 1; i >= 0; i--) { aStr[i] = acc; acc *= ap[i]; }
        int[] r4 = new int[4];
        for (int i = 0; i < rank; i++)
        {
            int d = lo[i] - off;
            r4[4 - rank + i] = d < 0 || a.Shape[d] == 1 ? 0 : aStr[Array.IndexOf(la, d)];
        }
        return r4;
    }

    private IEnumerable<MtlDispatch> Emit(ExecNode n)
    {
        CValue[] V = _plan.Values;
        // all addressing on storage-order shapes
        int[] shapeIn(int k) => V[n.In[k]].PhysShape;
        int[] shapeOut() => V[n.Out[0]].PhysShape;
        // spatial ops address [n,h,w,c] from the logical shape: NHWC-equivalent storage of a
        // 1x1-spatial value may keep an NCHW-ordered PhysShape (same bytes, different dims)
        int[] nhwc(int v) => V[v].Shape is [var nn, var cc, var hh, var ww] ? [nn, hh, ww, cc] : V[v].PhysShape;
        int[] nhwcIn(int k) => nhwc(n.In[k]);
        int[] nhwcOut() => nhwc(n.Out[0]);
        int nOut() => checked((int)V[n.Out[0]].ElementCount);

        switch (n.Op)
        {
            case ExecOp.Copy:
                yield return ND("unary", [Bind(n.In[0]), Bind(n.Out[0])],
                    PcOf((uint)nOut(), 4), G((uint)nOut()));
                break;

            case ExecOp.Relu or ExecOp.Sigmoid or ExecOp.Erf or ExecOp.Gelu or ExecOp.HardSwish or ExecOp.Swish:
            {
                uint mode = n.Op switch
                {
                    ExecOp.Relu => 0, ExecOp.Sigmoid => 1, ExecOp.Erf => 2, ExecOp.Gelu => 3,
                    ExecOp.HardSwish => 5, _ => 6,
                };
                yield return ND("unary", [Bind(n.In[0]), Bind(n.Out[0])],
                    PcOf((uint)nOut(), mode), G((uint)nOut()));
                break;
            }

            case ExecOp.Elementwise:
            {
                var p = (EwP)n.P!;
                CValue a = V[n.In[0]], bv = V[n.In[1]], o = V[n.Out[0]];
                int[] os = Pad4(shapeOut());
                int[] ostr = Strides(os);
                int[] astr = PhysBroadcastStrides(a, o);
                int[] bstr = PhysBroadcastStrides(bv, o);
                uint act = p.Act switch { ActivationKind.Relu => 1u, ActivationKind.Sigmoid => 2u, _ => 0u };
                // same-storage fast path: vec4 (n % 4 guard — no OOB tail writes)
                bool dense = astr.SequenceEqual(ostr) && bstr.SequenceEqual(ostr);
                if (nOut() % 4 == 0 && dense)
                {
                    yield return ND("ew2v", [Bind(n.In[0]), Bind(n.In[1]), Bind(n.Out[0])],
                        PcOf((uint)(nOut() / 4), (uint)p.Op, act), G((uint)(nOut() / 4)));
                    break;
                }
                yield return ND("ew2", [Bind(n.In[0]), Bind(n.In[1]), Bind(n.Out[0])],
                    PcOf(U(nOut(), ostr[0], ostr[1], ostr[2],
                        astr[0], astr[1], astr[2], astr[3],
                        bstr[0], bstr[1], bstr[2], bstr[3], (int)p.Op, (int)act)),
                    G((uint)nOut()));
                break;
            }

            case ExecOp.Conv:
            {
                var p = (ConvP)n.P!;
                RequireNhwc(n.In[0]); RequireNhwc(n.Out[0]);
                if (p.Pool is { } pool)
                {
                    // small-ic stem conv + 2x2/2 max pool (GraphCompiler.GpuSmallConv)
                    int[] ci = nhwcIn(0), co = nhwcOut(), cw = V[n.In[1]].Shape;
                    if (!GraphCompiler.GpuSmallConv(p, cw, ci[3]) || p.AddValue >= 0 || ci[1] != co[1] * 2 || ci[2] != co[2] * 2
                        || p.Act is not (ActivationKind.None or ActivationKind.Relu))
                        throw new NotSupportedException("Metal convpool: unsupported conv");
                    uint pf = (p.BiasValue >= 0 ? 1u : 0u) | (p.PadMode == PadMode.Reflect ? 2u : 0u)
                        | (p.Act == ActivationKind.Relu ? 4u : 0u) | (pool.FuseRelu ? 8u : 0u);
                    bool k3 = p.Kernel[0] == 3 && p.Kernel[1] == 3;
                    yield return ND(k3 ? "convpool_k3" : "convpool",
                        [Bind(n.In[0]), Bind(n.In[1], CK.ConvW), B(p.BiasValue, CK.F32), Bind(n.Out[0])],
                        PcOfF(U(ci[3], ci[1], ci[2], cw[0], co[1], co[2], p.Kernel[0], p.Kernel[1], p.Pads[0], p.Pads[1], (int)pf),
                            p.PadMode == PadMode.Constant ? p.PadValue : 0f),
                        G((uint)co[1] * (uint)co[2] * (uint)(cw[0] / 8)), name: "convpool");
                    break;
                }
                if (p.AddValue >= 0) RequireNhwc(p.AddValue);
                int[] si = nhwcIn(0), sw_ = V[n.In[1]].Shape, so = nhwcOut();   // [1,h,w,c]
                uint ih = (uint)si[1], iw = (uint)si[2], ic = (uint)si[3];
                uint oc = (uint)sw_[0], oh = (uint)so[1], ow = (uint)so[2];
                if (p.PadMode == PadMode.Constant && p.PadValue != 0f)
                    throw new NotSupportedException("Metal conv: nonzero constant pad");
                if (p.Dilations.Any(d => d != 1))
                    throw new NotSupportedException("Metal conv: dilation");
                if (p.Act == ActivationKind.Sigmoid)
                    throw new NotSupportedException("Metal conv: sigmoid epilogue");
                uint flags = (p.BiasValue >= 0 ? 1u : 0u)
                    | (p.PadMode == PadMode.Reflect ? 2u : 0u)
                    | (p.Act == ActivationKind.Relu ? 4u : 0u)
                    | (p.AddValue >= 0 ? 8u : 0u)
                    | (p.Act == ActivationKind.Gelu ? 16u : 0u)
                    | (p.Act == ActivationKind.HardSwish ? 32u : 0u)
                    | (p.Act == ActivationKind.Swish ? 64u : 0u);
                if (p.Group > 1)
                {
                    if (p.Group != ic || oc != ic) throw new NotSupportedException("grouped non-depthwise conv");
                    if (p.AddValue >= 0) throw new NotSupportedException("depthwise conv + residual");
                    bool v4 = ic % 4 == 0;
                    yield return ND(v4 ? "depthwise4" : "depthwise",
                        [Bind(n.In[0]), Bind(n.In[1], CK.DwW), B(p.BiasValue, CK.F32), Bind(n.Out[0])],
                        PcOfF(U((int)ic, (int)ih, (int)iw, (int)oh, (int)ow, p.Kernel[0], p.Kernel[1],
                            p.Strides[0], p.Strides[1], p.Pads[0], p.Pads[1], (int)(flags & ~8u)), p.PadValue),
                        G(ic * oh * ow / (v4 ? 4u : 1u)), name: "depthwise");
                    break;
                }
                if (UseMmConv(_plan, n, p))
                {
                    // BM = 64 rows (pixels); BN by oc divisibility; SCK when cin%4!=0
                    uint sck = ic % 4 != 0 ? 1u : 0u;
                    (string cm, uint bn) = oc % 128 == 0 ? ("conv_mm_128", 128u)
                        : oc % 64 == 0 ? ("conv_mm_64", 64u) : ("conv_mm_32", 32u);
                    cm += "_" + sck;
                    uint m = oh * ow, kr = (uint)(p.Kernel[0] * p.Kernel[1]) * ic, kd = (kr + 7u) & ~7u;
                    uint cflags = (p.BiasValue >= 0 ? 1u : 0u) | (p.AddValue >= 0 ? 2u : 0u)
                        | (p.PadMode == PadMode.Reflect ? 4u : 0u)
                        | (MmAct(p.Act) << 4);
                    yield return ND(cm,
                        [Bind(n.In[0]), Bind(n.In[1], CK.ConvWT), B(p.BiasValue, CK.F32), B(p.AddValue), Bind(n.Out[0])],
                        PcOf(m, oc, kd, cflags, ow, iw, ih, ic, (uint)p.Kernel[1],
                            (uint)p.Strides[0], (uint)p.Strides[1], (uint)p.Pads[0], (uint)p.Pads[1], kr),
                        (m + 63) / 64, (oc + bn - 1) / bn, name: "conv_mm");
                    break;
                }
                yield return ND("conv_tile",
                    [Bind(n.In[0]), Bind(n.In[1], CK.ConvW), B(p.BiasValue, CK.F32), B(p.AddValue), Bind(n.Out[0])],
                    PcOf(ic, ih, iw, oc, oh, ow, (uint)p.Kernel[0], (uint)p.Kernel[1],
                        (uint)p.Strides[0], (uint)p.Strides[1], (uint)p.Pads[0], (uint)p.Pads[1],
                        flags, 0),
                    (ow + 15) / 16, (oh + 7) / 8, (oc + 63) / 64, name: "conv");
                break;
            }

            case ExecOp.MaxPool:
            {
                var p = (PoolP)n.P!;
                RequireNhwc(n.In[0]); RequireNhwc(n.Out[0]);
                int[] si = nhwcIn(0), so = nhwcOut();
                bool v4 = si[3] % 4 == 0;
                yield return ND(v4 ? "maxpool4" : "maxpool", [Bind(n.In[0]), Bind(n.Out[0])],
                    PcOf((uint)si[3], (uint)si[1], (uint)si[2], (uint)so[1], (uint)so[2],
                        (uint)p.Kernel[0], (uint)p.Kernel[1], (uint)p.Strides[0], (uint)p.Strides[1],
                        (uint)p.Pads[0], (uint)p.Pads[1], p.FuseRelu ? 1u : 0u),
                    G((uint)so[1] * (uint)so[2] * (uint)so[3] / (v4 ? 4u : 1u)), name: "maxpool");
                break;
            }

            case ExecOp.Softmax:
            {
                CValue x = V[n.In[0]];
                uint len = (uint)x.Shape[^1];
                if ((uint)shapeIn(0)[^1] != len) throw new NotSupportedException("Metal Softmax: last axis not innermost");
                uint rows = (uint)(x.ElementCount / len);
                yield return ND("softmax", [Bind(n.In[0]), Bind(n.Out[0])], PcOf(rows, len), (rows + 7) / 8, name: "softmax");
                break;
            }

            case ExecOp.MatMul:
            {
                // batched [.., M, K] x [.., K, N] on logical (identity-layout) storage; B may broadcast
                CValue a = V[n.In[0]], b = V[n.In[1]], o = V[n.Out[0]];
                if (a.Layout is not null && !a.Layout.SequenceEqual(Engine.Layout.Identity(a.Shape.Length)) ||
                    b.Layout is not null && !b.Layout.SequenceEqual(Engine.Layout.Identity(b.Shape.Length)) ||
                    o.Layout is not null && !o.Layout.SequenceEqual(Engine.Layout.Identity(o.Shape.Length)))
                    throw new NotSupportedException("Metal MatMul: permuted layout");
                uint m = (uint)a.Shape[^2], k = (uint)a.Shape[^1], nn = (uint)b.Shape[^1];
                uint batch = (uint)(o.ElementCount / ((long)m * nn));
                long aBatch = a.ElementCount / ((long)m * k), bBatch = b.ElementCount / ((long)k * nn);
                if (aBatch != batch || (bBatch != batch && bBatch != 1))
                    throw new NotSupportedException("Metal MatMul: batch broadcast");
                yield return ND("gemm", [Bind(n.In[0]), Bind(n.In[1]), B(-1, CK.F32), Bind(n.Out[0])],
                    PcOf(m, nn, k, 0u, m * k, bBatch == 1 ? 0u : k * nn, m * nn),
                    (nn + 63) / 64, (m + 63) / 64, batch, name: "matmul");
                break;
            }

            case ExecOp.AvgPool:
            {
                var p = (PoolP)n.P!;
                RequireNhwc(n.In[0]); RequireNhwc(n.Out[0]);
                int[] si = nhwcIn(0), so = nhwcOut();
                bool v4 = si[3] % 4 == 0;
                yield return ND(v4 ? "avgpool4" : "avgpool", [Bind(n.In[0]), Bind(n.Out[0])],
                    PcOf((uint)si[3], (uint)si[1], (uint)si[2], (uint)so[1], (uint)so[2],
                        (uint)p.Kernel[0], (uint)p.Kernel[1], (uint)p.Strides[0], (uint)p.Strides[1],
                        (uint)p.Pads[0], (uint)p.Pads[1], p.CountIncludePad ? 1u : 0u, (uint)so[0]),
                    G((uint)nOut() / (v4 ? 4u : 1u)), name: "avgpool");
                break;
            }

            case ExecOp.LayerNorm:
            {
                var p = (LnP)n.P!;
                uint len = (uint)V[n.In[0]].Shape[^1];
                if ((uint)shapeIn(0)[^1] != len) throw new NotSupportedException("Metal LayerNorm: last dim not innermost");
                uint rows = (uint)(V[n.In[0]].ElementCount / len);
                yield return ND("layernorm",
                    [Bind(n.In[0]), Bind(n.In[1], CK.F32), Bind(n.In[2], CK.F32), Bind(n.Out[0])],
                    PcOfF([len, rows], p.Eps), (rows + 7) / 8, name: "layernorm");
                break;
            }

            case ExecOp.Gemm:
            {
                var p = (GemmP)n.P!;
                uint k = (uint)V[n.In[0]].Shape[^1];
                if ((uint)shapeIn(0)[^1] != k) throw new NotSupportedException("Metal Gemm: K not innermost");
                uint m = (uint)(V[n.In[0]].ElementCount / k);
                uint nn = (uint)V[n.Out[0]].Shape[^1];
                if (UseMmGemm(_plan, n, p))
                {
                    (string cm, uint bn) = nn % 128 == 0 ? ("conv_mm_128", 128u)
                        : nn % 64 == 0 ? ("conv_mm_64", 64u) : ("conv_mm_32", 32u);
                    cm += "_0";
                    uint cflags = (p.BiasValue >= 0 ? 1u : 0u)
                        | (MmAct(p.Act) << 4);
                    // [M,K] x [N,K]^T as a 1x1 conv over a 1 x M image
                    yield return ND(cm,
                        [Bind(n.In[0]), p.TransB ? Bind(n.In[1]) : Bind(n.In[1], CK.GemmNK), B(p.BiasValue, CK.F32), B(-1), Bind(n.Out[0])],
                        PcOf(m, nn, k, cflags, m, m, 1u, k, 1u, 1u, 1u, 0u, 0u, k),
                        (m + 63) / 64, (nn + bn - 1) / bn, name: "gemm_mm");
                    break;
                }
                uint gflags = (p.TransB ? 1u : 0u) | (p.BiasValue >= 0 ? 2u : 0u)
                    | (p.Act == ActivationKind.Relu ? 4u : 0u)
                    | (p.Act == ActivationKind.Gelu ? 8u : 0u)
                    | (p.Act == ActivationKind.HardSwish ? 16u : 0u)
                    | (p.Act == ActivationKind.Swish ? 32u : 0u);
                if (p.Act == ActivationKind.Sigmoid) throw new NotSupportedException("Metal Gemm: sigmoid epilogue");
                yield return ND("gemm",
                    [Bind(n.In[0]), Bind(n.In[1]), B(p.BiasValue, CK.F32), Bind(n.Out[0])],
                    PcOf(m, nn, k, gflags, 0u, 0u, 0u),
                    (nn + 63) / 64, (m + 63) / 64, name: "gemm");
                break;
            }

            case ExecOp.ResizeBilinear:
            {
                var p = (ResizeP)n.P!;
                RequireNhwc(n.In[0]); RequireNhwc(n.Out[0]);
                bool hasAdd = n.In.Length > 1;
                if (hasAdd) RequireNhwc(n.In[1]);
                int[] si = nhwcIn(0), so = nhwcOut();
                uint flags = (p.FuseRelu ? 1u : 0u) | (hasAdd ? 2u : 0u) | (p.AddRelu ? 4u : 0u);
                bool v4 = si[3] % 4 == 0;
                yield return ND(v4 ? "bilinear4" : "bilinear", [Bind(n.In[0]), hasAdd ? Bind(n.In[1]) : B(-1), Bind(n.Out[0])],
                    PcOfF([(uint)si[3], (uint)si[1], (uint)si[2], (uint)so[1], (uint)so[2], flags],
                        p.Scales[2], p.Scales[3]),
                    G((uint)so[1] * (uint)so[2] * (uint)so[3] / (v4 ? 4u : 1u)), name: "bilinear");
                break;
            }

            case ExecOp.Pad:
            {
                // generic rank-4 pad in storage order (pads permuted to physical dims)
                var p = (PadP)n.P!;
                CValue x = V[n.In[0]];
                if (x.Shape.Length != 4) throw new NotSupportedException("Metal pad: rank != 4");
                int[] l = x.Layout ?? Engine.Layout.Identity(4);
                int[] id4 = shapeIn(0), od4 = shapeOut();
                int[] pb4 = new int[4];
                for (int d = 0; d < 4; d++) pb4[d] = p.Pads[l[d]];
                yield return ND("pad", [Bind(n.In[0]), Bind(n.Out[0])],
                    PcOfF(U(id4[0], id4[1], id4[2], id4[3], od4[0], od4[1], od4[2], od4[3],
                        pb4[0], pb4[1], pb4[2], pb4[3], p.Mode == PadMode.Reflect ? 1 : 0),
                        p.Value),
                    G((uint)V[n.Out[0]].ElementCount));
                break;
            }

            case ExecOp.Transpose:
            {
                // only planner relayouts reach here; the perm is physical
                var p = (TransposeP)n.P!;
                if (shapeIn(0).Length > 4)
                {
                    int r = shapeIn(0).Length;
                    if (r > 6) throw new NotSupportedException($"Metal: transpose rank {r} > 6");
                    int[] is6 = new int[6], os6 = new int[6], p6 = [0, 1, 2, 3, 4, 5];
                    Array.Fill(is6, 1); Array.Fill(os6, 1);
                    shapeIn(0).CopyTo(is6, 6 - r); shapeOut().CopyTo(os6, 6 - r);
                    for (int i = 0; i < r; i++) p6[i + 6 - r] = p.Perm[i] + 6 - r;
                    int[] ost = new int[6], ist = new int[6];
                    ost[5] = ist[5] = 1;
                    for (int i = 4; i >= 0; i--) { ost[i] = ost[i + 1] * os6[i + 1]; ist[i] = ist[i + 1] * is6[i + 1]; }
                    yield return ND("transpose6", [Bind(n.In[0]), Bind(n.Out[0])],
                        PcOf(U(nOut(), ost[0], ost[1], ost[2], ost[3], ost[4],
                            ist[p6[0]], ist[p6[1]], ist[p6[2]], ist[p6[3]], ist[p6[4]], ist[p6[5]])),
                        G((uint)nOut()));
                    break;
                }
                int[] is4 = Pad4(shapeIn(0));
                int[] os4 = Pad4(shapeOut());
                int[] ostr = Strides(os4), istr = Strides(is4);
                int[] perm4 = PermuteTo4(p.Perm, shapeIn(0).Length);
                yield return ND("transpose", [Bind(n.In[0]), Bind(n.Out[0])],
                    PcOf(U(nOut(), ostr[0], ostr[1], ostr[2],
                        istr[perm4[0]], istr[perm4[1]], istr[perm4[2]], istr[perm4[3]])),
                    G((uint)nOut()));
                break;
            }

            case ExecOp.LeakyRelu or ExecOp.Tanh or ExecOp.Clip:
            {
                (uint mode, float a, float b) = n.P switch
                {
                    LeakyReluP lp => (0u, lp.Alpha, 0f),
                    ClipP cp => (2u, cp.Min, cp.Max),
                    _ => (1u, 0f, 0f),
                };
                yield return ND("unary2", [Bind(n.In[0]), Bind(n.Out[0])], PcOfF([(uint)nOut(), mode], a, b), G((uint)nOut()));
                break;
            }

            case ExecOp.DepthToSpace:
            {
                var p = (DepthToSpaceP)n.P!;
                RequireNhwc(n.In[0]); RequireNhwc(n.Out[0]);
                int[] si = nhwcIn(0), so = nhwcOut();
                yield return ND("d2s", [Bind(n.In[0]), Bind(n.Out[0])],
                    PcOf((uint)nOut(), (uint)so[3], (uint)so[1], (uint)so[2], (uint)si[1], (uint)si[2], (uint)p.Block, p.Crd ? 1u : 0u),
                    G((uint)nOut()), name: "d2s");
                break;
            }

            case ExecOp.Slice:
            {
                // SliceP.Axes are physical after the layout pass
                var p = (SliceP)n.P!;
                int[] is4 = Pad4(shapeIn(0)), os4 = Pad4(shapeOut());
                int[] ostr = Strides(os4), istr = Strides(is4);
                uint[] st = [0, 0, 0, 0], sp = [1, 1, 1, 1];
                int shift = 4 - shapeIn(0).Length;
                for (int i = 0; i < p.Axes.Length; i++)
                {
                    st[p.Axes[i] + shift] = (uint)p.Starts[i];
                    sp[p.Axes[i] + shift] = (uint)p.Steps[i];
                }
                yield return ND("slice", [Bind(n.In[0]), Bind(n.Out[0])],
                    PcOf(U(nOut(), ostr[0], ostr[1], ostr[2],
                        (int)st[0], (int)st[1], (int)st[2], (int)st[3],
                        (int)sp[0], (int)sp[1], (int)sp[2], (int)sp[3],
                        istr[0], istr[1], istr[2], istr[3])),
                    G((uint)nOut()));
                break;
            }

            case ExecOp.Concat:
            {
                // ConcatP.Axis is physical after the layout pass
                var p = (ConcatP)n.P!;
                int[] os = shapeOut();
                int axis = p.Axis;
                int inner = 1, dstAxis = os[axis];
                for (int d = axis + 1; d < os.Length; d++) inner *= os[d];
                int axisOff = 0;
                foreach (int iv in n.In)
                {
                    int[] ishape = V[iv].PhysShape;
                    int cnt = checked((int)V[iv].ElementCount);
                    if (cnt > 0)
                    {
                        yield return ND("concat_ax", [Bind(iv), Bind(n.Out[0])],
                            PcOf((uint)cnt, (uint)inner, (uint)ishape[axis], (uint)dstAxis, (uint)axisOff),
                            G((uint)cnt));
                    }
                    axisOff += ishape[axis];
                }
                break;
            }

            case ExecOp.HardSigmoid:
            {
                var p = (HardSigmoidP)n.P!;
                yield return ND("hsig", [Bind(n.In[0]), Bind(n.Out[0])],
                    PcOfF([(uint)nOut()], p.Alpha, p.Beta), G((uint)nOut()));
                break;
            }

            case ExecOp.GlobalAvgPool:
            {
                RequireNhwc(n.In[0]);
                int[] si = nhwcIn(0);
                uint c = (uint)si[3], hw = (uint)si[1] * (uint)si[2];
                bool v4 = c % 4 == 0;
                yield return ND(v4 ? "gap4" : "gap", [Bind(n.In[0]), Bind(n.Out[0])],
                    PcOfF([c, hw], 1f / hw), v4 ? (c / 4 + 15) / 16 : (c + 63) / 64, name: "gap");
                break;
            }

            case ExecOp.ConvTranspose:
            {
                var p = (ConvTP)n.P!;
                RequireNhwc(n.In[0]); RequireNhwc(n.Out[0]);
                int[] si = nhwcIn(0), so = nhwcOut();
                uint act = p.Act switch { ActivationKind.None => 0u, ActivationKind.Relu => 1u, ActivationKind.Sigmoid => 2u,
                    _ => throw new NotSupportedException($"Metal ConvTranspose: {p.Act}") };
                yield return ND("convt", [Bind(n.In[0]), Bind(n.In[1], CK.ConvTW), B(p.BiasValue, CK.F32), Bind(n.Out[0])],
                    PcOf((uint)si[3], (uint)si[1], (uint)si[2], (uint)so[3], (uint)p.Kernel[0], (uint)p.Kernel[1],
                        (p.BiasValue >= 0 ? 1u : 0u) | act << 1),
                    G((uint)nOut()));
                break;
            }

            case ExecOp.ResizeNearest:
            {
                var p = (ResizeNearestP)n.P!;
                RequireNhwc(n.In[0]); RequireNhwc(n.Out[0]);
                bool hasAdd = n.In.Length > 1;
                if (hasAdd) RequireNhwc(n.In[1]);
                int[] si = nhwcIn(0), so = nhwcOut();
                bool v4 = si[3] % 4 == 0;
                yield return ND(v4 ? "nearest4" : "nearest", [Bind(n.In[0]), hasAdd ? Bind(n.In[1]) : B(-1), Bind(n.Out[0])],
                    PcOfF([(uint)si[3], (uint)si[1], (uint)si[2], (uint)so[1], (uint)so[2],
                        (hasAdd ? 2u : 0u) | (p.AddRelu ? 4u : 0u)], p.Scales[2], p.Scales[3]),
                    G((uint)nOut() / (v4 ? 4u : 1u)), name: "nearest");
                break;
            }

            case ExecOp.ChannelGate:
            {
                var p = (ChannelGateP)n.P!;
                RequireNhwc(n.In[0]);
                int[] si = nhwcIn(0);
                uint hw = (uint)si[1] * (uint)si[2], c = (uint)si[3], r = (uint)V[n.In[1]].Shape[0];
                uint gflags = (p.B1 >= 0 ? 1u : 0u) | (p.B2 >= 0 ? 2u : 0u) | (p.Act1 == ActivationKind.Relu ? 4u : 0u) | (p.Hard ? 8u : 0u);
                yield return ND("se_gate",
                    [Bind(n.In[0]), Bind(n.In[1], CK.F32), B(p.B1, CK.F32), Bind(n.In[2], CK.F32), B(p.B2, CK.F32), Bind(n.Out[0])],
                    PcOfF([c, r, hw, gflags], 1f / hw, p.Alpha, p.Beta, p.Offset), (uint)si[0], name: "se_gate");
                break;
            }

            case ExecOp.Attention:
            {
                var p = (AttentionP)n.P!;
                CValue o = V[n.Out[0]];
                int rank = o.Shape.Length;
                int[] oDims = rank == 4 ? [0, 1, 2, 3] : [-1, 0, 1, 2];
                uint T = (uint)o.Shape[^2], DV = (uint)o.Shape[^1], H = (uint)o.Shape[^3], N = rank == 4 ? (uint)o.Shape[0] : 1u;
                uint S = (uint)V[n.In[1]].Shape[p.Dims[1][2]], D = (uint)V[n.In[0]].Shape[p.Dims[0][3]];
                var pc = new List<uint> { T, S, D, DV, H };
                for (int i = 0; i < 3; i++) pc.AddRange(StridedView(V[n.In[i]], p.Dims[i], p.FixAxis[i], p.FixIndex[i]));
                pc.AddRange(StridedView(o, oDims, -1, 0));
                yield return ND(D <= 32 && DV <= 32 ? "attn" : "attn_d64",
                    [Bind(n.In[0]), Bind(n.In[1]), Bind(n.In[2]), Bind(n.Out[0])],
                    PcOfF([.. pc], p.Scale), (T + 255) / 256, N * H, name: "attn");
                break;
            }

            case ExecOp.Ext:
                foreach (MtlDispatch d in n.Ext!.EmitMetal(this, n)) yield return d;
                break;

            default:
                throw new NotSupportedException($"Metal: ExecOp {n.Op} not implemented");
        }

        MtlDispatch ND(string shader, MtlBind[] binds, byte[] pc, uint gx, uint gy = 1, uint gz = 1, string? name = null) =>
            Dispatch(n, shader, binds, pc, gx, gy, gz);

        MtlBind B(int valueId, CK? kind = null) => BindOrDummy(valueId, kind);
    }

    /// <summary>A dispatch of kernel <paramref name="shader"/> from the MSL embedded in
    /// <paramref name="owner"/> (null = the core library).</summary>
    public MtlDispatch Dispatch(ExecNode n, string shader, MtlBind[] binds, byte[] pc, uint gx, uint gy = 1, uint gz = 1,
        Assembly? owner = null) => new()
    {
        Pipe = _dev.Pipe(shader, owner),
        Pc = pc, Binds = binds, Gx = gx, Gy = gy, Gz = gz,
        Name = $"{shader}({n.Name})",
    };

    /// <summary>Binding of a value, or the 64 B zero buffer for <paramref name="valueId"/> &lt; 0.</summary>
    public MtlBind BindOrDummy(int valueId, CK? kind = null) =>
        valueId >= 0 ? Bind(valueId, kind) : new MtlBind(MtlSlot.Dummy, 0, 64);

    internal MtlDevice Device => _dev;

    public void RequireNhwc(int v)
    {
        CValue cv = _plan.Values[v];
        if (cv.Shape.Length != 4 || !Engine.Layout.Equivalent(cv.Shape, cv.Layout ?? Engine.Layout.Identity(4), Engine.Layout.Nhwc))
            throw new NotSupportedException($"Metal: '{cv.Name}' expected NHWC storage, got {Engine.Layout.Str(cv.Layout)}");
    }

    // pad a shape to rank 4 with leading 1s
    private static int[] Pad4(int[] s)
    {
        if (s.Length == 4) return s;
        int[] r = [1, 1, 1, 1];
        s.CopyTo(r, 4 - s.Length);
        return r;
    }

    private static int[] Strides(int[] shape4)
    {
        int[] s = new int[4]; s[3] = 1;
        for (int i = 2; i >= 0; i--) s[i] = s[i + 1] * shape4[i + 1];
        return s;
    }

    // lift an original-rank perm into padded rank-4 (leading dims map to themselves)
    private static int[] PermuteTo4(int[] perm, int rank)
    {
        int shift = 4 - rank;
        int[] p4 = [0, 1, 2, 3];
        for (int i = 0; i < rank; i++) p4[i + shift] = perm[i] + shift;
        return p4;
    }

    /// <summary>Sessions hold a reference so cache eviction can't free live const/dummy buffers.</summary>
    public void Acquire()
    {
        if (Interlocked.Increment(ref _refs) <= 1)
            throw new ObjectDisposedException(nameof(MtlGraphModel), "const buffers already freed");
    }
    /// <summary>Debug/test: const buffer + byte offset of (value, packing-kind) for diagnostics.</summary>
    internal (MtlBuffer buf, ulong off, bool half) DebugConst(int valueId, CK kind)
    {
        return _constOff.TryGetValue((valueId, kind), out var c)
            ? (c.buf, c.off, !kind.Fp32)
            : (Dummy, ulong.MaxValue, false);
    }

    /// <summary>Acquire unless already freed (no resurrection; for racing cache readers).</summary>
    public bool TryAcquire()
    {
        while (true)
        {
            int r = Volatile.Read(ref _refs);
            if (r <= 0) return false;
            if (Interlocked.CompareExchange(ref _refs, r + 1, r) == r) return true;
        }
    }
    public void Release()
    {
        if (Interlocked.Decrement(ref _refs) == 0) _store.Release();
    }

    /// <summary>The plan this graph was compiled from (arena offsets, I/O values).</summary>
    public ExecPlan Plan => _plan;
    public void Dispose() => Release();
}
