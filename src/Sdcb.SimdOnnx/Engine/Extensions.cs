using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Reflection;
#if NET
using Sdcb.SimdOnnx.Backends.Metal;
using Sdcb.SimdOnnx.Backends.Vulkan;
#endif

namespace Sdcb.SimdOnnx.Engine;

/// <summary>
/// A consumer-defined executable operator (<see cref="ExecOp.Ext"/>). Graph passes create
/// nodes with <see cref="ExecNode.Ext"/> set; the core calls back for layout, CPU execution,
/// weight packing and GPU dispatch emission. Per-node parameters live in <see cref="ExecNode.P"/>.
/// A backend that does not implement the op must not be targeted by the fusing pass
/// (<see cref="PassContext.Target"/>): the core never falls back node by node.
/// </summary>
internal abstract class ExtOp
{
    public abstract string Name { get; }

    /// <summary>Value ids read through <see cref="ExecNode.P"/> (kept live by the arena planner).</summary>
    public virtual IEnumerable<int> ParamRefs(ExecNode n) => [];

    /// <summary>Layout pass: request input layouts and set the output's. Default: rank-4 NHWC in/out.</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public virtual void PlanLayout(LayoutPlanner lp, ExecNode n)
    {
        for (int k = 0; k < n.In.Length; k++)
            if (lp.Values[n.In[k]].IsActivation) lp.Need(n, k, Layout.Nhwc);
        foreach (int o in n.Out) lp.Values[o].Layout = Layout.Nhwc;
    }

    /// <summary>Packs CPU weights into <see cref="ExecNode.P"/> (idempotent; plans may be shared).</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public virtual void PackCpu(ExecPlan plan, ExecNode n, CpuKernels.Cache? cache) { }

    public abstract void RunCpu(InferenceSession s, ExecNode n);

    /// <summary>Consts the Vulkan dispatches bind, with the packing each needs.</summary>
    public virtual IEnumerable<(int value, ConstKind kind)> GpuConsts(ExecNode n) => [];
#if NET

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public virtual IEnumerable<VkDispatch> EmitVulkan(VkGraphModel g, ExecNode n) =>
        throw new NotSupportedException($"Vulkan: extension op {Name} not implemented");

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public virtual IEnumerable<MtlDispatch> EmitMetal(MtlGraphModel g, ExecNode n) =>
        throw new NotSupportedException($"Metal: extension op {Name} not implemented");
#endif
}

/// <summary>When a <see cref="GraphPass"/> runs.</summary>
internal enum PassStage
{
    /// <summary>Logical graph, after the built-in fusions, before layout planning.</summary>
    Logical,
    /// <summary>Physical graph (layouts assigned), after the built-in memory fusions, before arena planning.</summary>
    Physical,
}

/// <summary>A consumer-defined rewrite over the exec node list.</summary>
internal abstract class GraphPass
{
    public abstract PassStage Stage { get; }
    public abstract void Run(PassContext ctx);
}

/// <summary>Registered extension operators and passes; pass via <see cref="CompileOptions.Extensions"/>.</summary>
internal sealed class EngineExtensions
{
    private readonly List<GraphPass> _passes = [];
    public IReadOnlyList<GraphPass> Passes => _passes;
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public EngineExtensions Add(GraphPass pass) { _passes.Add(pass); return this; }
}

/// <summary>Graph view handed to <see cref="GraphPass.Run"/>.</summary>
internal sealed class PassContext(List<ExecNode> nodes, List<CValue> values, int outputValue, CompileOptions options)
{
    public List<ExecNode> Nodes { get; } = nodes;
    public List<CValue> Values { get; } = values;
    public int OutputValue { get; } = outputValue;
    public CompileOptions Options { get; } = options;
    public PlanTarget Target => Options.Target;

    public int Producer(int v) => Nodes.FindIndex(m => m.Out.Contains(v));

    /// <summary>Readers of v's buffer (aliases and param operands included); the graph output counts as infinite.</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public int Uses(int v)
    {
        int b = Values[v].Buf;
        if (b < 0 || v == OutputValue || Values[OutputValue].Buf == b) return int.MaxValue;
        int n = 0;
        foreach (ExecNode m in Nodes)
        {
            foreach (int iv in m.In) if (Values[iv].Buf == b) n++;
            foreach (int iv in m.ParamRefs()) if (Values[iv].Buf == b) n++;
        }
        return n;
    }

    public bool SameShape(int a, int b) => Values[a].Shape.AsSpan().SequenceEqual(Values[b].Shape);

    public bool IsNhwc4(int v) => Values[v].Shape.Length == 4 && Values[v].Layout is [0, 2, 3, 1];

    /// <summary>Rewrites every reader of <paramref name="oldValue"/> to read <paramref name="newValue"/>.</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public void Redirect(int oldValue, int newValue)
    {
        foreach (ExecNode m in Nodes)
            for (int i = 0; i < m.In.Length; i++)
                if (m.In[i] == oldValue) m.In[i] = newValue;
        if (Values[oldValue].Name.Length != 0 && Values[newValue].Name.Length == 0)
            Values[newValue].Name = Values[oldValue].Name;
    }
}

/// <summary>
/// How a const tensor is laid out on a GPU (decided by its consumer). Built-ins cover the core
/// shaders; extensions create their own with a <see cref="Pack"/> function. Identity is by reference.
/// </summary>
internal sealed class ConstKind(string name, bool fp32, Func<CValue, float[]>? pack = null, Func<CValue, long>? elems = null)
{
    /// <summary>Core packings write straight into the upload buffer (no packed copy per const).</summary>
    internal ConstKind(string name, GpuPack.Packer into, Func<CValue, long>? elems = null)
        : this(name, false, w => { float[] d = new float[elems?.Invoke(w) ?? w.Floats!.Length]; into(w, d); return d; }, elems)
        => PackInto = into;
    internal GpuPack.Packer? PackInto { get; }

    private static int s_next;
    /// <summary>Stable per-process id (const stores compare kind lists by it).</summary>
    public int Id { get; } = Interlocked.Increment(ref s_next);
    public string Name { get; } = name;
    /// <summary>Stored as fp32 even on fp16-storage graphs (bias, norm params).</summary>
    public bool Fp32 { get; } = fp32;
    public Func<CValue, float[]>? Pack { get; } = pack;
    /// <summary>Bound element count when it differs from the logical one (padded packings).</summary>
    public Func<CValue, long>? Elems { get; } = elems;
    public override string ToString() => Name;

    public static readonly ConstKind Raw = new("raw", false);
    public static readonly ConstKind F32 = new("f32", true);
    /// <summary>Conv [OC][IC][KH][KW] → [K][OC], k = (ky·KW + kx)·IC + ic (direct-conv shaders).</summary>
    public static readonly ConstKind ConvW = new("convw", GpuPack.Conv);
    /// <summary>Depthwise [C][1][KH][KW] → [tap][C].</summary>
    public static readonly ConstKind DwW = new("dww", GpuPack.Dw);
    /// <summary>Conv → [OC][Kpad] (k fastest, K padded to 8; matrix-core conv/GEMM shaders).</summary>
    public static readonly ConstKind ConvWT = new("convwt", GpuPack.ConvT, GpuPack.ConvTElems);
    /// <summary>Gemm B [K][N] → [N][K] (k fastest, matrix-core GEMM).</summary>
    public static readonly ConstKind GemmNK = new("gemmnk", GpuPack.GemmNK);
    /// <summary>ConvTranspose [IC][OC][KH][KW] → [tap][IC][OC].</summary>
    public static readonly ConstKind ConvTW = new("convtw", GpuPack.ConvTranspose);
    /// <summary>Conv → [OC padded to 128][K padded to 16], tap-major k (direct-load 8x16x16 GEMM).</summary>
    public static readonly ConstKind CmW = new("cmw", GpuPack.Cm, GpuPack.CmElems);
    /// <summary>Gemm B [K][N] → [N padded to 128][K padded to 16] (direct-load 8x16x16 GEMM).</summary>
    public static readonly ConstKind GemmCmW = new("gemmcmw", GpuPack.GemmCm, GpuPack.GemmCmElems);
}

/// <summary>Const packings shared by the Vulkan and Metal backends.</summary>
internal static class GpuPack
{
    /// <summary>Writes the packed form of <c>w</c> into <c>dst</c> (pre-zeroed, packed length).</summary>
    internal delegate void Packer(CValue w, Span<float> dst);

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static void Conv(CValue w, Span<float> dst)
    {
        int oc = w.Shape[0], ic = w.Shape[1], taps = w.Shape[2] * w.Shape[3];
        float[] src = w.Floats!;
        for (int o = 0; o < oc; o++)
            for (int c = 0; c < ic; c++)
                for (int t = 0; t < taps; t++)
                    dst[(t * ic + c) * oc + o] = src[(o * ic + c) * taps + t];
    }

    public static long ConvTElems(CValue w) => (long)w.Shape[0] * ((w.Shape[1] * w.Shape[2] * w.Shape[3] + 7) & ~7);

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static void ConvT(CValue w, Span<float> dst)
    {
        int oc = w.Shape[0], ic = w.Shape[1], taps = w.Shape[2] * w.Shape[3], k = (taps * ic + 7) & ~7;
        float[] src = w.Floats!;
        for (int o = 0; o < oc; o++)
            for (int c = 0; c < ic; c++)
                for (int t = 0; t < taps; t++)
                    dst[o * k + t * ic + c] = src[(o * ic + c) * taps + t];
    }

    internal static int Pad16(int k) => (k + 15) & ~15;
    internal static int Pad128(int n) => (n + 127) & ~127;

    public static long CmElems(CValue w) => (long)Pad128(w.Shape[0]) * Pad16(w.Shape[1] * w.Shape[2] * w.Shape[3]);

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static void Cm(CValue w, Span<float> dst)
    {
        int oc = w.Shape[0], ic = w.Shape[1], taps = w.Shape[2] * w.Shape[3], k = Pad16(taps * ic);
        float[] src = w.Floats!;
        for (int o = 0; o < oc; o++)
            for (int c = 0; c < ic; c++)
                for (int t = 0; t < taps; t++)
                    dst[o * k + t * ic + c] = src[(o * ic + c) * taps + t];
    }

    public static long GemmCmElems(CValue b) => (long)Pad128(b.Shape[1]) * Pad16(b.Shape[0]);

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static void GemmCm(CValue b, Span<float> dst)
    {
        int k = b.Shape[0], nn = b.Shape[1], kp = Pad16(k);
        float[] src = b.Floats!;
        for (int i = 0; i < k; i++)
            for (int j = 0; j < nn; j++) dst[j * kp + i] = src[i * nn + j];
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static void Dw(CValue w, Span<float> dst)
    {
        int c = w.Shape[0], taps = w.Shape[2] * w.Shape[3];
        float[] src = w.Floats!;
        for (int ch = 0; ch < c; ch++)
            for (int t = 0; t < taps; t++) dst[t * c + ch] = src[ch * taps + t];
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static void GemmNK(CValue b, Span<float> dst)
    {
        int k = b.Shape[0], nn = b.Shape[1];
        float[] src = b.Floats!;
        for (int i = 0; i < k; i++)
            for (int j = 0; j < nn; j++) dst[j * k + i] = src[i * nn + j];
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static void ConvTranspose(CValue w, Span<float> dst)
    {
        int ic = w.Shape[0], oc = w.Shape[1], taps = w.Shape[2] * w.Shape[3];
        float[] src = w.Floats!;
        for (int c = 0; c < ic; c++)
            for (int o = 0; o < oc; o++)
                for (int t = 0; t < taps; t++) dst[(t * ic + c) * oc + o] = src[(c * oc + o) * taps + t];
    }

    /// <summary>Packed host bytes + per-const offsets (64 B aligned; fp16 unless the kind is fp32), in
    /// native memory the caller frees after the upload: the packed weights never touch the GC heap
    /// (they are hundreds of MB of large-object churn per model otherwise).</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static unsafe (ulong[] offsets, HostBytes host) Build(ExecPlan plan, IReadOnlyList<(int v, ConstKind k)> need, bool f16)
    {
        ulong total = 0;
        var offs = new ulong[need.Count];
        var lens = new long[need.Count];
        for (int i = 0; i < need.Count; i++)
        {
            var (v, k) = need[i];
            CValue cv = plan.Values[v];
            lens[i] = k.PackInto is not null || k.Pack is null ? k.Elems?.Invoke(cv) ?? cv.Floats!.Length : -1;
            if (lens[i] < 0) continue;   // extension packing: sized below
            total = (total + 63) & ~63UL;
            offs[i] = total;
            total += (ulong)lens[i] * (f16 && !k.Fp32 ? 2UL : 4UL);
        }
        // extension packings return arrays: size them now (their arrays are the only managed copies)
        var ext = new float[need.Count][];
        for (int i = 0; i < need.Count; i++)
        {
            if (lens[i] >= 0) continue;
            var (v, k) = need[i];
            ext[i] = k.Pack!(plan.Values[v]);
            lens[i] = ext[i].Length;
            total = (total + 63) & ~63UL;
            offs[i] = total;
            total += (ulong)lens[i] * (f16 && !k.Fp32 ? 2UL : 4UL);
        }
        var host = new HostBytes(Math.Max((total + 63) & ~63UL, 64));
        float* scratch = null;
        long scratchLen = 0;
        try
        {
            for (int i = 0; i < need.Count; i++)
            {
                var (v, k) = need[i];
                CValue cv = plan.Values[v];
                long n = lens[i];
                bool half = f16 && !k.Fp32;
                ReadOnlySpan<float> data;
                if (ext[i] is { } e) data = e;
                else if (k.PackInto is { } into)
                {
                    if (!half)
                    {
                        // fp32 storage: pack in place
                        into(cv, new Span<float>(host.Ptr + offs[i], checked((int)n)));
                        continue;
                    }
                    if (scratchLen < n)
                    {
                        NativeMemory.Free(scratch);
                        scratch = (float*)NativeMemory.Alloc((nuint)(n * sizeof(float)));
                        scratchLen = n;
                    }
                    var tmp = new Span<float>(scratch, checked((int)n));
                    tmp.Clear();
                    into(cv, tmp);
                    data = tmp;
                }
                else data = cv.Floats!;
                if (half)
                {
#if NET
                    var dst = new Span<Half>(host.Ptr + offs[i], data.Length);
                    for (int j = 0; j < data.Length; j++) dst[j] = (Half)data[j];
#else
                    throw new PlatformNotSupportedException("fp16 GPU constants need net10.0.");
#endif
                }
                else data.CopyTo(new Span<float>(host.Ptr + offs[i], data.Length));
            }
        }
        catch
        {
            host.Dispose();
            throw;
        }
        finally { NativeMemory.Free(scratch); }
        return (offs, host);
    }
}

/// <summary>A zeroed native byte buffer (const uploads); free with <see cref="Dispose"/>.</summary>
internal sealed unsafe class HostBytes : IDisposable
{
    public HostBytes(ulong length)
    {
        Length = length;
        Ptr = (byte*)NativeMemory.AllocZeroed((nuint)length);
    }

    public byte* Ptr { get; private set; }
    public ulong Length { get; }

    public void Dispose()
    {
        NativeMemory.Free(Ptr);
        Ptr = null;
    }
}

/// <summary>Embedded GPU shader lookup across assemblies (the core and extension owners).</summary>
internal static class ShaderResources
{
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static Stream? Open(Assembly asm, string fileName)
    {
        string suffix = ".Shaders." + fileName;
        string? res = asm.GetManifestResourceNames().FirstOrDefault(n => n.EndsWith(suffix, StringComparison.Ordinal));
        return res is null ? null : asm.GetManifestResourceStream(res);
    }
}
