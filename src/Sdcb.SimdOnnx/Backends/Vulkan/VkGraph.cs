using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Reflection;
using Sdcb.SimdOnnx.Engine;
using CK = Sdcb.SimdOnnx.Engine.ConstKind;

namespace Sdcb.SimdOnnx.Backends.Vulkan;

/// <summary>Binding slots resolved per-session: which underlying buffer.</summary>
public enum VkSlot : int { Arena = 0, Consts = 1, Dummy = 2, StageIn = 3, StageOut = 4, Direct = 5 }

public readonly struct VkBind
{
    public readonly VkSlot Slot;
    internal readonly VkBuffer? Buf;   // when Slot == Direct
    public readonly ulong Offset, Size;
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public VkBind(VkSlot s, ulong off, ulong size) { Slot = s; Offset = off; Size = size; }
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal VkBind(VkBuffer b, ulong off, ulong size) { Slot = VkSlot.Direct; Buf = b; Offset = off; Size = size; }
}

/// <summary>One recorded compute dispatch.</summary>
public sealed class VkDispatch
{
    public required VkPipeline Pipe;
    public required VkBind[] Binds;
    public required byte[] Pc;
    public uint Gx, Gy = 1, Gz = 1;
    public string Name = "";
}

/// <summary>
/// Per-(device, ExecPlan) half of the Vulkan backend: backend-private constant
/// packing/upload and the node list compiled to <see cref="VkDispatch"/> records.
/// Sessions own the activation arena and resolve slot ids to live buffers.
///
/// Plans are layout-planned (NHWC, see <see cref="LayoutPlanner"/>); every
/// addressing decision reads <see cref="CValue.PhysShape"/>. With <see cref="F16"/>
/// activations and weights are stored as float16 (16-bit storage only — all
/// shader math, reductions included, is fp32); conv bias and LayerNorm params
/// stay fp32. Const bytes therefore differ per backend and per precision.
/// </summary>
public sealed class VkGraphModel : IDisposable
{
    private readonly VkDevice _dev;
    private readonly ExecPlan _plan;
    private int _refs = 1;  // the graph cache holds the first reference
    private readonly VkConstStore _store;
    internal readonly VkBuffer Consts;       // all const tensors, backend-packed (shared store)
    private readonly Dictionary<(int v, CK k), (VkBuffer buf, ulong off)> _constOff = new();
    internal readonly VkBuffer Dummy;        // 64B zero buffer for unused bindings
    internal readonly VkDispatch[] Dispatches;
    internal readonly ulong ConstBytes;
    /// <summary>Activations + weights stored as float16 (fp32 math in every shader).</summary>
    public readonly bool F16;
    /// <summary>Bytes per stored activation element.</summary>
    public int Es => F16 ? 2 : 4;
    /// <summary>Backend cache recency stamp.</summary>
    internal long LastUse;
    /// <summary>Arena bytes a session needs: the plan's activations, then this graph's scratch.</summary>
    internal ulong ArenaBytes => Math.Max(_scratchOff + _scratchBytes, (_tailBytes + 255) & ~255UL);
    private readonly ulong _scratchOff;
    private ulong _scratchBytes;


    /// <param name="constKey">Sharing key for the const store (the source model): graphs of the
    /// same model at other sizes reuse its uploaded weights. Null = private store.</param>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal VkGraphModel(VkDevice dev, ExecPlan plan, bool f16, object? constKey = null)
    {
        _dev = dev;
        _plan = plan;
        F16 = f16;
        if (!plan.Nhwc) throw new NotSupportedException("Vulkan backend requires a layout-planned (NHWC) plan.");

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
                    Need(n.In[1], cp.Group > 1 ? CK.DwW
                        : RouteM8Conv(dev, f16, plan, n, cp) is M8Route.Cm8 or M8Route.Im2col ? CK.CmW
                        : RouteM8Conv(dev, f16, plan, n, cp) is M8Route.Dot or M8Route.KDot ? CK.ConvW
                        : UseCoopConv(dev, f16, plan, n, cp) ? CK.ConvWT : CK.ConvW);
                    Need(cp.BiasValue, CK.F32);
                    Need(cp.AddValue, CK.Raw);
                    break;
                case GemmP gp:
                    Need(n.In[0], CK.Raw);
                    Need(n.In[1], RouteM8Gemm(dev, f16, plan, n, gp) switch
                    {
                        M8Route.Cm8 => CK.GemmCmW,
                        M8Route.Dot => CK.Raw,
                        _ => UseCoopGemm(dev, f16, plan, n, gp) && !gp.TransB ? CK.GemmNK : CK.Raw,
                    });
                    Need(gp.BiasValue, CK.F32);
                    break;
                case LnP:
                    Need(n.In[0], CK.Raw); Need(n.In[1], CK.F32); Need(n.In[2], CK.F32);
                    break;
                case ConvTP tp:
                    Need(n.In[1], CK.ConvTW);
                    Need(tp.BiasValue, CK.F32);
                    break;
                case ChannelGateP gp:
                    Need(n.In[1], CK.F32); Need(n.In[2], CK.F32);
                    Need(gp.B1, CK.F32); Need(gp.B2, CK.F32);
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
                throw new NotSupportedException($"Vulkan: const '{cv.Name}' with non-identity layout");
        }
        var items = need.Select(x => (plan.Values[x.v].Floats!, x.k.Id, plan.Values[x.v].Shape)).ToArray();
        _store = VkConstStore.Acquire(dev, constKey, f16);
        var placed = _store.Place(items, miss => GpuPack.Build(plan, miss.Select(i => need[i]).ToList(), f16));
        for (int i = 0; i < need.Count; i++) _constOff[need[i]] = placed[i];
        Consts = _store.Primary;
        Dummy = _store.Dummy;
        ConstBytes = _store.Bytes;

        _scratchOff = ((ulong)plan.ArenaFloats * (ulong)Es + 255) & ~255UL;
        var disps = new List<VkDispatch>();
        NodeRanges = new (int start, int count)[plan.Nodes.Length];
        for (int i = 0; i < plan.Nodes.Length; i++)
        {
            int start = disps.Count;
            Emit(plan.Nodes[i], disps);
            NodeRanges[i] = (start, disps.Count - start);
        }
        Dispatches = disps.ToArray();
    }



    /// <summary>Tensor-core conv eligibility (conv_cm.comp): fp16 storage on a coopmat device,
    /// dense, channel counts in 16 B quads, an epilogue it implements, M within the grid limit.</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static bool UseCoopConv(VkDevice dev, bool f16, ExecPlan plan, ExecNode n, ConvP cp)
    {
        if (!dev.CoopMat16 || dev.ConvDot || !f16 || cp.Group != 1 || cp.Pool is not null || cp.Dilations.Any(d => d != 1)) return false;
        // SG16 direct devices never take the staged conv_cm build (see RouteM8Conv)
        if (M8Direct(dev, f16, plan)) return false;
        // swish: a trailing in-place pass (an exp epilogue in conv_cm costs the GcdrNet staged convs ~0.6%)
        if (cp.Act is not (ActivationKind.None or ActivationKind.Relu or ActivationKind.Gelu or ActivationKind.HardSwish or ActivationKind.Swish)) return false;
        if (cp.PadMode == PadMode.Constant && cp.PadValue != 0f) return false;
        int[] si = plan.Values[n.In[0]].PhysShape, so = plan.Values[n.Out[0]].PhysShape;   // [1,h,w,c]
        long m = (long)so[0] * so[1] * so[2];
        if (cp.Kernel[0] == 1 && cp.Kernel[1] == 1 && Env.Flag("VK_CM1X1") == "0") return false;
        return so[3] % 8 == 0 && (m + 127) / 128 <= 65535;   // cin % 8 != 0 -> scalar-gather stem variant
    }


    /// <summary>Tensor-core Gemm (a 1x1 conv over M "pixels" through conv_cm.comp): fp16 storage
    /// on a coopmat device, const B, K/N in 16 B quads, relu/gelu/none epilogue.</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static bool UseCoopGemm(VkDevice dev, bool f16, ExecPlan plan, ExecNode n, GemmP gp)
    {
        if (!dev.CoopMat16 || dev.ConvDot || !f16 || plan.Values[n.In[1]].Floats is null || plan.Values[n.In[1]].Shape.Length != 2) return false;
        if (M8Direct(dev, f16, plan)) return false;
        // B packed per value: only when no other node reads it (its transB could differ)
        if (plan.Nodes.Count(m => m.In.Contains(n.In[1])) != 1) return false;
        if (gp.Act is ActivationKind.Sigmoid) return false;   // hardswish / swish: trailing unary pass
        CValue a = plan.Values[n.In[0]];
        int k = a.Shape[^1], nn = plan.Values[n.Out[0]].Shape[^1];
        if (a.PhysShape[^1] != k) return false;
        long m = a.ElementCount / k;
        return k % 8 == 0 && nn % 8 == 0 && (m + 127) / 128 <= 65535;
    }

    /// <summary>Kernel family for a dense conv / const Gemm on 8x16x16 (SG16, Intel Xe2) devices.</summary>
    internal enum M8Route { None, Dot, Cm8, KDot, Im2col }

    /// <summary>SG16 devices route dense convs to the direct kernels of the old OCR engine
    /// (pw_dot / gemm_cm8 / conv_kdot / im2col + gemm_cm8) instead of the shared-staged conv_cm
    /// SG16 build, which intermittently loses the device on the Arc B580 for the OCR models
    /// (CompileOptions.StagedCoopConv or GCDRNET_VK_SG16STAGED=1 keeps it). None = the generic shaders.</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static M8Route RouteM8Conv(VkDevice dev, bool f16, ExecPlan plan, ExecNode n, ConvP cp) =>
        Allowed(RouteM8ConvCore(dev, f16, plan, n, cp));

    // debug: GCDRNET_VK_M8=dot,cm8,kdot,im2col limits the direct routes (others take the generic shaders)
    private static readonly string? s_m8Routes = Environment.GetEnvironmentVariable("GCDRNET_VK_M8");
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static M8Route Allowed(M8Route r) =>
        r == M8Route.None || s_m8Routes is null || s_m8Routes.Split(',').Contains(r.ToString().ToLowerInvariant()) ? r : M8Route.None;

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static M8Route RouteM8ConvCore(VkDevice dev, bool f16, ExecPlan plan, ExecNode n, ConvP cp)
    {
        if (!M8Direct(dev, f16, plan))
            return SmallDotConv(dev, f16, plan, n, cp) ? M8Route.Dot : WideDotConv(dev, f16, plan, n, cp) ? M8Route.KDot : M8Route.None;
        if (cp.Group != 1 || cp.Pool is not null || cp.Dilations.Any(d => d != 1)) return M8Route.None;
        if (cp.PadMode != PadMode.Constant || cp.PadValue != 0f) return M8Route.None;
        if (cp.Act is not (ActivationKind.None or ActivationKind.Relu or ActivationKind.Gelu or ActivationKind.HardSwish or ActivationKind.Swish)) return M8Route.None;
        CValue x = plan.Values[n.In[0]];
        if (x.Shape.Length != 4) return M8Route.None;
        int ic = x.Shape[1], oc = plan.Values[n.In[1]].Shape[0];
        bool pointwise = cp.Kernel[0] == 1 && cp.Kernel[1] == 1 && cp.Strides.All(s => s == 1) && cp.Pads.All(q => q == 0);
        if (pointwise)
        {
            // small-K pointwise: the dot kernel outruns the tensor cores (old engine: cin <= 128)
            if (ic % 4 == 0 && oc % 4 == 0 && ic <= 128) return M8Route.Dot;
            return ic % 16 == 0 ? M8Route.Cm8 : M8Route.None;
        }
        if (ic % 4 != 0) return M8Route.None;
        int k = cp.Kernel[0] * cp.Kernel[1] * ic;
        if (oc % 4 == 0 && k <= 1024) return M8Route.KDot;
        // im2col matrix is one binding
        long rows = plan.Values[n.Out[0]].ElementCount / oc;
        return (ulong)((rows + 511) / 512 * 512 * GpuPack.Pad16(k) * 2L) <= dev.MaxStorageRange ? M8Route.Im2col : M8Route.None;
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static M8Route RouteM8Gemm(VkDevice dev, bool f16, ExecPlan plan, ExecNode n, GemmP gp) =>
        Allowed(RouteM8GemmCore(dev, f16, plan, n, gp));

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static M8Route RouteM8GemmCore(VkDevice dev, bool f16, ExecPlan plan, ExecNode n, GemmP gp)
    {
        bool direct = M8Direct(dev, f16, plan);
        if (!f16 || gp.TransB) return M8Route.None;
        CValue a = plan.Values[n.In[0]], b = plan.Values[n.In[1]];
        if (b.Floats is null || b.Shape.Length != 2 || a.Buf < 0) return M8Route.None;
        if (gp.Act is ActivationKind.Sigmoid) return M8Route.None;
        int k = a.Shape[^1], nn = b.Shape[1];
        if (a.PhysShape[^1] != k) return M8Route.None;
        if (k % 4 == 0 && nn % 4 == 0 && k <= 128
            && (direct || a.ElementCount / k * nn <= SmallDotLimit(dev, plan))) return M8Route.Dot;
        return direct && k % 16 == 0 ? M8Route.Cm8 : M8Route.None;
    }

    /// <summary>Largest M x N the small-K dot kernel takes on devices off the SG16 direct family
    /// (old OCR engine): 32-lane coopmat devices 2^18 (lite: 2^16, a quarter), no coopmat 2^16 —
    /// past that the tensor-core / tiled GEMM wins. Plans that opt into the staged coopmat build
    /// (<see cref="CompileOptions.StagedCoopConv"/>, GcdrNet) keep their routing. SIMDONNX_VK_SMALLDOT=0 disables.</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static long SmallDotLimit(VkDevice dev, ExecPlan plan) =>
        Env.Flag("VK_SMALLDOT") == "0" || plan.Options?.StagedCoopConv == true ? 0
            : dev.CoopMat16 && !dev.CoopLite ? 1L << 18 : 1L << 16;

    /// <summary>Small-K pointwise conv (cin &lt;= 128) through pw_dot on devices off the SG16
    /// direct family, within <see cref="SmallDotLimit"/>.</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static bool SmallDotConv(VkDevice dev, bool f16, ExecPlan plan, ExecNode n, ConvP cp)
    {
        if (!f16 || cp.Group != 1 || cp.Pool is not null || cp.Dilations.Any(d => d != 1)) return false;
        if (cp.Kernel[0] != 1 || cp.Kernel[1] != 1 || cp.Strides.Any(s => s != 1) || cp.Pads.Any(q => q != 0)) return false;
        if (cp.Act is not (ActivationKind.None or ActivationKind.Relu or ActivationKind.Gelu or ActivationKind.HardSwish or ActivationKind.Swish)) return false;
        CValue x = plan.Values[n.In[0]];
        if (x.Shape.Length != 4) return false;
        int ic = x.Shape[1], oc = plan.Values[n.In[1]].Shape[0];
        long m = plan.Values[n.Out[0]].ElementCount / oc;
        // wide-subgroup no-coopmat devices (Adreno): every 1x1 conv, the tiled GEMM is load-bound there
        if (dev.ConvDot && plan.Options?.StagedCoopConv != true && Env.Flag("VK_SMALLDOT") != "0")
            return ic % 4 == 0 && oc % 4 == 0;
        return ic % 4 == 0 && oc % 4 == 0 && ic <= 128 && m * oc <= SmallDotLimit(dev, plan);
    }

    /// <summary>Dense kxk convs on wide-subgroup no-coopmat devices (<see cref="VkDevice.ConvDot"/>,
    /// Adreno) that conv_dot does not take (other kernel sizes, batches, hardswish): conv_kdot, as the
    /// old OCR engine ran its convk_dot there instead of the shared-staged conv_tile.</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static bool WideDotConv(VkDevice dev, bool f16, ExecPlan plan, ExecNode n, ConvP cp)
    {
        if (!dev.ConvDot || !f16 || plan.Options?.StagedCoopConv == true || Env.Flag("VK_SMALLDOT") == "0") return false;
        if (cp.Group != 1 || cp.Pool is not null || cp.Dilations.Any(d => d != 1)) return false;
        if (cp.PadMode != PadMode.Constant || cp.PadValue != 0f) return false;
        if (cp.Act is not (ActivationKind.None or ActivationKind.Relu or ActivationKind.Gelu or ActivationKind.HardSwish or ActivationKind.Swish)) return false;
        CValue x = plan.Values[n.In[0]];
        if (x.Shape.Length != 4) return false;
        int ic = x.Shape[1], oc = plan.Values[n.In[1]].Shape[0], nb = plan.Values[n.Out[0]].Shape[0];
        if (ic % 4 != 0 || oc % 4 != 0) return false;
        bool k33 = cp.Kernel[0] == 3 && cp.Kernel[1] == 3 && cp.Strides[0] == cp.Strides[1] && cp.Strides[0] is 1 or 2;
        return !(k33 && nb == 1 && cp.Act is not (ActivationKind.HardSwish or ActivationKind.Swish));   // conv_dot's own case
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static bool M8Direct(VkDevice dev, bool f16, ExecPlan plan) =>
        f16 && dev.CoopM8 && plan.Options?.StagedCoopConv != true && Env.Flag("VK_SG16STAGED") != "1";

    /// <summary>gemm_cm8 tile (rows x columns) and pipeline by output width (old engine's CmTile).</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static (string shader, uint bm, uint bn) Cm8Tile(uint nn) =>
        nn <= 32 ? ("gemm_cm8_n32", 512u, 32u) : nn <= 64 ? ("gemm_cm8_n64", 256u, 64u) : ("gemm_cm8", 128u, 128u);

    /// <summary>Binding of an arena value widened to <paramref name="bytes"/> (tile reads past its
    /// end); the arena grows so the widened range stays inside it.</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private VkBind BindPadded(int valueId, ulong bytes)
    {
        VkBind b = Bind(valueId);
        if (b.Slot != VkSlot.Arena || bytes <= b.Size) return b;
        _tailBytes = Math.Max(_tailBytes, b.Offset + bytes);
        return new VkBind(VkSlot.Arena, b.Offset, bytes);
    }

    private ulong _tailBytes;



    /// <summary>Dispatch index range per source ExecNode (for per-node debug runs).</summary>
    internal readonly (int start, int count)[] NodeRanges;

    // ---- value → bind resolution (byte offsets/sizes: all long math) ----
    /// <summary>Transient arena space past the plan's values; one region, reused by every dispatch pair.</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public VkBind Scratch(ulong bytes)
    {
        _scratchBytes = Math.Max(_scratchBytes, bytes);
        return new VkBind(VkSlot.Arena, _scratchOff, bytes);
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public VkBind Bind(int valueId, CK? kind = null)
    {
        kind ??= CK.Raw;
        CValue v = _plan.Values[valueId];
        if (v.Buf >= 0)
            return new VkBind(VkSlot.Arena, (ulong)_plan.BufOffset[v.Buf] * (ulong)Es,
                (ulong)Math.Max(v.ElementCount, 1) * (ulong)Es);
        if (_constOff.TryGetValue((valueId, kind), out var c))
        {
            ulong es = F16 && !kind.Fp32 ? 2UL : 4UL;
            long elems = kind.Elems?.Invoke(v) ?? Math.Max(v.ElementCount, 1);
            return new VkBind(c.buf, c.off, (ulong)elems * es);
        }
        if (v.Floats != null)
            throw new InvalidOperationException($"const value '{v.Name}' not packed as {kind}");
        throw new InvalidOperationException($"value '{v.Name}' has no GPU storage (ints must be folded)");
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static byte[] PcOf(params uint[] ints)
    {
        byte[] b = new byte[ints.Length * 4];
        Buffer.BlockCopy(ints, 0, b, 0, b.Length);
        return b;
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static byte[] PcOfF(uint[] ints, params float[] floats)
    {
        byte[] b = new byte[(ints.Length + floats.Length) * 4];
        Buffer.BlockCopy(ints, 0, b, 0, ints.Length * 4);
        Buffer.BlockCopy(floats, 0, b, ints.Length * 4, floats.Length * 4);
        return b;
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static uint[] U(params int[] v) { uint[] u = new uint[v.Length]; for (int i = 0; i < v.Length; i++) u[i] = (uint)v[i]; return u; }

    public static uint G(uint n) => (n + 255) / 256;

    /// <summary>uints, one float, then more uints (push-constant blocks with a float in the middle).</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static byte[] PcOf3(uint[] ints, float f, uint[] tail)
    {
        byte[] b = new byte[(ints.Length + 1 + tail.Length) * 4];
        Buffer.BlockCopy(ints, 0, b, 0, ints.Length * 4);
        BitConverter.GetBytes(f).CopyTo(b, ints.Length * 4);
        Buffer.BlockCopy(tail, 0, b, (ints.Length + 1) * 4, tail.Length * 4);
        return b;
    }

    /// <summary>Elements between neighbours along logical axis <paramref name="axis"/> in the value's storage.</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static long LogicalStride(CValue v, int axis)
    {
        int[] l = v.Layout ?? Engine.Layout.Identity(v.Shape.Length), ps = v.PhysShape;
        long s = 1;
        for (int j = ps.Length - 1; j >= 0 && l[j] != axis; j--) s *= ps[j];
        return s;
    }

    /// <summary>Shaders without a batch index: the value must hold one image.</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private void RequireSingle(int valueId)
    {
        if (_plan.Values[valueId].PhysShape is { Length: 4 } s && s[0] != 1)
            throw new NotSupportedException($"Vulkan: '{_plan.Values[valueId].Name}' batch {s[0]} on a single-image shader");
    }

    /// <summary>
    /// Broadcast strides (padded rank-4, storage order of the output) of operand
    /// <paramref name="a"/> onto output <paramref name="o"/>. Output physical dim i
    /// holds logical dim o.Layout[i]; the operand's logical dim is that minus the
    /// numpy right-alignment offset.
    /// </summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static int[] PhysBroadcastStrides(CValue a, CValue o)
    {
        int rank = o.Shape.Length, off = rank - a.Shape.Length;
        if (rank > 4) throw new NotSupportedException("Vulkan elementwise: rank > 4");
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

    private string Sh(string name) => F16 ? name + "_h" : name;

    /// <summary>Epilogue code of the 3-bit act field (pw_dot / conv_kdot / gemm_cm8).</summary>
    private static uint EpiCode(ActivationKind a) => a switch
    {
        ActivationKind.Relu => 1u, ActivationKind.Gelu => 2u, ActivationKind.HardSwish => 3u, ActivationKind.Swish => 4u, _ => 0u,
    };

    /// <summary>conv_cm's act field: no swish (the caller adds a trailing pass).</summary>
    private static uint CmCode(ActivationKind a) => a == ActivationKind.Swish ? 0u : EpiCode(a);

    /// <summary>4-channel memory-bound kernels: the plan's choice, else the device default.</summary>
    private bool Vec4 => _plan.Options?.Vec4Eltwise ?? _dev.Vec4Eltwise;

    // a list, not an iterator: the body then JITs optimized (AggressiveOptimization does not reach iterator state machines)
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private void Emit(ExecNode n, List<VkDispatch> output)
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
        bool swishDone = false;   // the conv / gemm kernel applied a swish / hardswish epilogue itself

        switch (n.Op)
        {
            case ExecOp.Copy:
                output.Add(ND("unary", [Bind(n.In[0]), Bind(n.Out[0])],
                    PcOf((uint)nOut(), 4), G((uint)nOut())));
                break;

            case ExecOp.Relu or ExecOp.Sigmoid or ExecOp.Erf or ExecOp.Gelu:
            {
                uint mode = n.Op switch
                {
                    ExecOp.Relu => 0, ExecOp.Sigmoid => 1, ExecOp.Erf => 2, _ => 3,
                };
                output.Add(ND("unary", [Bind(n.In[0]), Bind(n.Out[0])],
                    PcOf((uint)nOut(), mode), G((uint)nOut())));
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
                // x op scalar const (and s + x, s - x, s · x): one affine pass a·x + b over the full operand
                int ak = -1;
                float sa = 0f, sb = 0f;
                for (int k = 0; k < 2 && ak < 0 && p.Act == ActivationKind.None; k++)
                {
                    if (V[n.In[1 - k]].Floats is not [var c] || V[n.In[k]].Buf < 0 || V[n.In[k]].ElementCount != nOut()
                        || !PhysBroadcastStrides(V[n.In[k]], o).SequenceEqual(ostr)) continue;
                    bool ok = true;
                    (sa, sb) = (p.Op, k) switch
                    {
                        (EwOp.Add, _) => (1f, c),
                        (EwOp.Mul, _) => (c, 0f),
                        (EwOp.Sub, 0) => (1f, -c),
                        (EwOp.Sub, _) => (-1f, c),
                        (EwOp.Div, 0) => (1f / c, 0f),
                        _ => (0f, 0f),
                    };
                    if (p.Op == EwOp.Div && k == 1) ok = false;   // c / x is not affine
                    if (ok) ak = k;
                }
                if (ak >= 0)
                {
                    (uint ux, uint uy) = G2((uint)nOut());
                    output.Add(ND("unary2", [Bind(n.In[ak]), Bind(n.Out[0])], PcOfF([(uint)nOut(), 3u], sa, sb), ux, uy));
                    break;
                }
                // same-storage fast path: vec4 (n % 4 guard — no OOB tail writes)
                bool dense = astr.SequenceEqual(ostr) && bstr.SequenceEqual(ostr);
                if (nOut() % 4 == 0 && dense)
                {
                    output.Add(ND("ew2v", [Bind(n.In[0]), Bind(n.In[1]), Bind(n.Out[0])],
                        PcOf((uint)(nOut() / 4), (uint)p.Op, act), G((uint)(nOut() / 4))));
                    break;
                }
                output.Add(ND("ew2", [Bind(n.In[0]), Bind(n.In[1]), Bind(n.Out[0])],
                    PcOf(U(nOut(), ostr[0], ostr[1], ostr[2],
                        astr[0], astr[1], astr[2], astr[3],
                        bstr[0], bstr[1], bstr[2], bstr[3], (int)p.Op, (int)act)),
                    G((uint)nOut())));
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
                    if (ci[0] != 1) throw new NotSupportedException("Vulkan convpool: batch > 1");
                    if (!GraphCompiler.GpuSmallConv(p, cw, ci[3]) || p.AddValue >= 0 || ci[1] != co[1] * 2 || ci[2] != co[2] * 2)
                        throw new NotSupportedException("Vulkan convpool: unsupported conv");
                    uint pf = (p.BiasValue >= 0 ? 1u : 0u) | (p.PadMode == PadMode.Reflect ? 2u : 0u)
                        | (p.Act == ActivationKind.Relu ? 4u : 0u) | (pool.FuseRelu ? 8u : 0u);
                    bool k3 = _dev.ConvPoolK3 && p.Kernel[0] == 3 && p.Kernel[1] == 3;
                    string cpName = k3 ? "convpool_k3" : _dev.ConvDot && F16 && cw[0] % 8 == 0 ? "convpool_pi" : "convpool";
                    output.Add(ND(cpName,
                        [Bind(n.In[0]), Bind(n.In[1], CK.ConvW), B(p.BiasValue, CK.F32), Bind(n.Out[0])],
                        PcOfF(U(ci[3], ci[1], ci[2], cw[0], co[1], co[2], p.Kernel[0], p.Kernel[1], p.Pads[0], p.Pads[1], (int)pf),
                            p.PadMode == PadMode.Constant ? p.PadValue : 0f),
                        G((uint)co[1] * (uint)co[2] * (uint)(cw[0] / 8)), name: "convpool"));
                    break;
                }
                if (p.AddValue >= 0) RequireNhwc(p.AddValue);
                int[] si = nhwcIn(0), sw_ = V[n.In[1]].Shape, so = nhwcOut();   // [1,h,w,c]
                uint ih = (uint)si[1], iw = (uint)si[2], ic = (uint)si[3];
                uint oc = (uint)sw_[0], oh = (uint)so[1], ow = (uint)so[2], nb = (uint)so[0];
                if (p.PadMode == PadMode.Constant && p.PadValue != 0f)
                    throw new NotSupportedException("Vulkan conv: nonzero constant pad");
                if (p.Dilations.Any(d => d != 1))
                    throw new NotSupportedException("Vulkan conv: dilation");
                if (p.Act == ActivationKind.Sigmoid)
                    throw new NotSupportedException("Vulkan conv: sigmoid epilogue");
                uint flags = (p.BiasValue >= 0 ? 1u : 0u)
                    | (p.PadMode == PadMode.Reflect ? 2u : 0u)
                    | (p.Act == ActivationKind.Relu ? 4u : 0u)
                    | (p.AddValue >= 0 ? 8u : 0u)
                    | (p.Act == ActivationKind.Gelu ? 16u : 0u)
                    | (p.Act == ActivationKind.HardSwish ? 32u : 0u);
                if (p.Act is not (ActivationKind.None or ActivationKind.Relu or ActivationKind.Gelu or ActivationKind.HardSwish or ActivationKind.Swish))
                    throw new NotSupportedException($"Vulkan conv: {p.Act} epilogue");   // swish: trailing pass below
                if (p.Group > 1)
                {
                    if (p.Group != ic || oc != ic) throw new NotSupportedException("grouped non-depthwise conv");
                    if (p.AddValue >= 0) throw new NotSupportedException("depthwise conv + residual");
                    bool v4 = Vec4 && ic % 4 == 0;
                    output.Add(ND(v4 ? "depthwise4" : "depthwise",
                        [Bind(n.In[0]), Bind(n.In[1], CK.DwW), B(p.BiasValue, CK.F32), Bind(n.Out[0])],
                        PcOf3(U((int)ic, (int)ih, (int)iw, (int)oh, (int)ow, p.Kernel[0], p.Kernel[1],
                            p.Strides[0], p.Strides[1], p.Pads[0], p.Pads[1], (int)(flags & ~8u)), p.PadValue, [nb]),
                        G(nb * ic * oh * ow / (v4 ? 4u : 1u)), name: "depthwise"));
                    break;
                }
                M8Route m8 = RouteM8Conv(_dev, F16, _plan, n, p);
                if (m8 != M8Route.None)
                {
                    swishDone = true;
                    uint m = nb * oh * ow, mImg = oh * ow;
                    uint m8flags = (p.BiasValue >= 0 ? 1u : 0u) | (p.AddValue >= 0 ? 2u : 0u)
                        | (EpiCode(p.Act) << 4);
                    if (m8 == M8Route.Dot)
                    {
                        (uint dx, uint dy) = G2((m + 3) / 4 * (oc / 4));
                        output.Add(ND("pw_dot",
                            [Bind(n.In[0]), Bind(n.In[1], CK.ConvW), B(p.BiasValue, CK.F32), B(p.AddValue), Bind(n.Out[0])],
                            PcOf(m, oc, ic, m8flags), dx, dy, name: "pw_dot"));
                        break;
                    }
                    if (m8 == M8Route.KDot)
                    {
                        (uint dx, uint dy) = G2((mImg + 3) / 4 * (oc / 4));
                        output.Add(ND("conv_kdot",
                            [Bind(n.In[0]), Bind(n.In[1], CK.ConvW), B(p.BiasValue, CK.F32), B(p.AddValue), Bind(n.Out[0])],
                            PcOf(mImg, oc, ow, iw, ih, ic / 4, (uint)(p.Kernel[0] * p.Kernel[1]), (uint)p.Kernel[1],
                                (uint)p.Strides[0], (uint)p.Strides[1], (uint)p.Pads[0], (uint)p.Pads[1], m8flags),
                            dx, dy, nb, name: "conv_kdot"));
                        break;
                    }
                    (string g8, uint bm, uint bn) = Cm8Tile(oc);
                    uint rowsPad = (m + bm - 1) / bm * bm;
                    VkBind a;
                    uint kp;
                    if (m8 == M8Route.Im2col)
                    {
                        uint kr = (uint)(p.Kernel[0] * p.Kernel[1]) * ic;
                        kp = (uint)GpuPack.Pad16((int)kr);
                        a = Scratch((ulong)rowsPad * kp * 2);
                        (uint ix, uint iy) = G2(mImg * (kp / 4));
                        output.Add(ND("im2col", [Bind(n.In[0]), Bind(n.In[0]), a],
                            PcOf(ow, oh, ic, (uint)p.Kernel[0], (uint)p.Kernel[1], (uint)p.Strides[0], (uint)p.Strides[1],
                                (uint)p.Pads[0], (uint)p.Pads[1], iw, ih, kr, kp, 0u),
                            ix, iy, nb, name: "im2col"));
                    }
                    else
                    {
                        kp = ic;
                        a = BindPadded(n.In[0], (ulong)rowsPad * kp * 2);
                    }
                    output.Add(ND(g8,
                        [a, Bind(n.In[1], CK.CmW), B(p.BiasValue, CK.F32), B(p.AddValue), Bind(n.Out[0])],
                        PcOf(m, oc, kp, m8flags), rowsPad / bm, (oc + bn - 1) / bn, name: "gemm_cm8"));
                    break;
                }
                if (UseCoopConv(_dev, F16, _plan, n, p))
                {
                    // BM = 128 rows (pixels) in every variant; BN by oc divisibility
                    // SG16 (Intel Xe2): the BN=32 build on 1x1 convs hangs the device intermittently (B580,
                    // driver 32.0.101.8135) — those take the BN=64 build with masked tail columns
                    bool no32 = _dev.CoopM8 && p.Kernel[0] == 1 && p.Kernel[1] == 1;
                    swishDone = p.Act != ActivationKind.Swish;
                    (string cm, uint bn) = ic % 8 != 0 ? ("conv_cm_n32s", 32u)
                        : oc % 128 == 0 ? ("conv_cm", 128u) : oc % 64 == 0 || no32 ? ("conv_cm_n64", 64u) : ("conv_cm_n32", 32u);
                    uint m = nb * oh * ow, kr = (uint)(p.Kernel[0] * p.Kernel[1]) * ic, kd = (kr + 7u) & ~7u;
                    uint cflags = (p.BiasValue >= 0 ? 1u : 0u) | (p.AddValue >= 0 ? 2u : 0u)
                        | (p.PadMode == PadMode.Reflect ? 4u : 0u)
                        | (CmCode(p.Act) << 4);
                    output.Add(ND(cm,
                        [Bind(n.In[0]), Bind(n.In[1], CK.ConvWT), B(p.BiasValue, CK.F32), B(p.AddValue), Bind(n.Out[0])],
                        PcOf(m, oc, kd, cflags, ow, iw, ih, ic, (uint)p.Kernel[1],
                            (uint)p.Strides[0], (uint)p.Strides[1], (uint)p.Pads[0], (uint)p.Pads[1], kr, oh),
                        (m + 127) / 128, (oc + bn - 1) / bn, name: "conv_cm"));
                    break;
                }
                string convShader =
                    p.Kernel[0] == 3 && p.Kernel[1] == 3 && p.Strides[0] == 1 && p.Strides[1] == 1 ? "conv_k33s1" :
                    p.Kernel[0] == 3 && p.Kernel[1] == 3 && p.Strides[0] == 2 && p.Strides[1] == 2 ? "conv_k33s2" :
                    p.Kernel[0] == 1 && p.Kernel[1] == 1 && p.Strides[0] == 1 && p.Strides[1] == 1 ? "conv_k11s1" :
                    "conv_gen";
                if (_dev.ConvDot && F16 && nb == 1 && (p.Act != ActivationKind.HardSwish) && convShader is "conv_k33s1" or "conv_k33s2" && ic % 4 == 0 && oc % 4 == 0)
                {
                    // Adreno 750 sweep (ms, 1600x1152 gc+dr): stride 1 at 4 px x 8 oc, stride 2 at 2 px x 8 oc
                    uint tm = p.Strides[1] == 1 ? 4u : 2u, tn = oc % 8 == 0 ? 2u : 1u;
                    uint threads = oc / (4 * tn) * ((ow + tm - 1) / tm) * oh;
                    uint wgs = (threads + 63) / 64, gx = Math.Min(wgs, 32768u);
                    output.Add(ND($"conv_dot_s{p.Strides[1]}_{tm}x{tn}",
                        [Bind(n.In[0]), Bind(n.In[1], CK.ConvW), B(p.BiasValue, CK.F32), B(p.AddValue), Bind(n.Out[0])],
                        PcOf(ic, ih, iw, oc, oh, ow, (uint)p.Kernel[0], (uint)p.Kernel[1],
                            (uint)p.Strides[0], (uint)p.Strides[1], (uint)p.Pads[0], (uint)p.Pads[1],
                            flags, 0),
                        gx, (wgs + gx - 1) / gx, name: "conv"));
                    break;
                }
                output.Add(ND(convShader,
                    [Bind(n.In[0]), Bind(n.In[1], CK.ConvW), B(p.BiasValue, CK.F32), B(p.AddValue), Bind(n.Out[0])],
                    PcOf(ic, ih, iw, oc, oh, ow, (uint)p.Kernel[0], (uint)p.Kernel[1],
                        (uint)p.Strides[0], (uint)p.Strides[1], (uint)p.Pads[0], (uint)p.Pads[1],
                        flags, nb),
                    (ow + 15) / 16, nb * ((oh + 7) / 8), (oc + 63) / 64, name: "conv"));
                break;
            }

            case ExecOp.MaxPool:
            {
                var p = (PoolP)n.P!;
                RequireNhwc(n.In[0]); RequireNhwc(n.Out[0]);
                int[] si = nhwcIn(0), so = nhwcOut();
                bool v4 = Vec4 && si[3] % 4 == 0;
                output.Add(ND(v4 ? "maxpool4" : "maxpool", [Bind(n.In[0]), Bind(n.Out[0])],
                    PcOf((uint)si[3], (uint)si[1], (uint)si[2], (uint)so[1], (uint)so[2],
                        (uint)p.Kernel[0], (uint)p.Kernel[1], (uint)p.Strides[0], (uint)p.Strides[1],
                        (uint)p.Pads[0], (uint)p.Pads[1], p.FuseRelu ? 1u : 0u, (uint)so[0]),
                    G((uint)so[0] * (uint)so[1] * (uint)so[2] * (uint)so[3] / (v4 ? 4u : 1u)), name: "maxpool"));
                break;
            }

            case ExecOp.LayerNorm:
            {
                var p = (LnP)n.P!;
                uint len = (uint)V[n.In[0]].Shape[^1];
                if ((uint)shapeIn(0)[^1] != len) throw new NotSupportedException("Vulkan LayerNorm: last dim not innermost");
                uint rows = (uint)(V[n.In[0]].ElementCount / len);
                output.Add(ND("layernorm",
                    [Bind(n.In[0]), Bind(n.In[1], CK.F32), Bind(n.In[2], CK.F32), Bind(n.Out[0])],
                    PcOfF([len], p.Eps), rows, name: "layernorm"));
                break;
            }

            case ExecOp.Gemm:
            {
                var p = (GemmP)n.P!;
                // hardswish / swish: the direct and tensor-core kernels apply them, the generic ones run a
                // separate in-place pass after the switch
                ActivationKind act = p.Act is ActivationKind.HardSwish or ActivationKind.Swish ? ActivationKind.None : p.Act;
                uint k = (uint)V[n.In[0]].Shape[^1];
                if ((uint)shapeIn(0)[^1] != k) throw new NotSupportedException("Vulkan Gemm: K not innermost");
                uint m = (uint)(V[n.In[0]].ElementCount / k);
                uint nn = (uint)V[n.Out[0]].Shape[^1];
                M8Route gm8 = RouteM8Gemm(_dev, F16, _plan, n, p);
                if (gm8 != M8Route.None)
                {
                    swishDone = true;
                    uint gf = (p.BiasValue >= 0 ? 1u : 0u) | (EpiCode(p.Act) << 4);
                    if (gm8 == M8Route.Dot)
                    {
                        (uint dx, uint dy) = G2((m + 3) / 4 * (nn / 4));
                        output.Add(ND("pw_dot", [Bind(n.In[0]), Bind(n.In[1]), B(p.BiasValue, CK.F32), B(-1), Bind(n.Out[0])],
                            PcOf(m, nn, k, gf), dx, dy, name: "gemm_dot8"));
                        break;
                    }
                    (string g8, uint bm, uint bn) = Cm8Tile(nn);
                    uint rowsPad = (m + bm - 1) / bm * bm;
                    output.Add(ND(g8,
                        [BindPadded(n.In[0], (ulong)rowsPad * k * 2), Bind(n.In[1], CK.GemmCmW), B(p.BiasValue, CK.F32), B(-1), Bind(n.Out[0])],
                        PcOf(m, nn, k, gf), rowsPad / bm, (nn + bn - 1) / bn, name: "gemm_cm8"));
                    break;
                }
                if (UseCoopGemm(_dev, F16, _plan, n, p))
                {
                    (string cm, uint bn) = nn % 128 == 0 ? ("conv_cm", 128u) : nn % 64 == 0 ? ("conv_cm_n64", 64u) : ("conv_cm_n32", 32u);
                    swishDone = p.Act != ActivationKind.Swish;
                    uint cflags = (p.BiasValue >= 0 ? 1u : 0u) | (CmCode(p.Act) << 4);
                    // [M,K] x [N,K]^T as a 1x1 conv over a 1 x M image.
                    // transB=1: B is already [N,K] row-major — bind it raw (GemmNK's pack
                    // assumes a [K,N] source and would transpose it the wrong way);
                    // transB=0: pack [K,N] into the [N,K] panel the shader expects.
                    output.Add(ND(cm,
                        [Bind(n.In[0]), p.TransB ? Bind(n.In[1]) : Bind(n.In[1], CK.GemmNK), B(p.BiasValue, CK.F32), B(-1), Bind(n.Out[0])],
                        PcOf(m, nn, k, cflags, m, m, 1u, k, 1u, 1u, 1u, 0u, 0u, k, 1u),
                        (m + 127) / 128, (nn + bn - 1) / bn, name: "gemm_cm"));
                    break;
                }
                uint gflags = (p.TransB ? 1u : 0u) | (p.BiasValue >= 0 ? 2u : 0u)
                    | (act == ActivationKind.Relu ? 4u : 0u)
                    | (act == ActivationKind.Gelu ? 8u : 0u);
                if (_dev.ConvDot && F16 && !p.TransB && k % 4 == 0 && nn % 8 == 0)
                {
                    // 4 rows x 8 columns per invocation
                    uint gth = nn / 8 * ((m + 3) / 4);
                    uint gwg = (gth + 63) / 64, ggx = Math.Min(gwg, 32768u);
                    output.Add(ND("gemm_dot",
                        [Bind(n.In[0]), Bind(n.In[1]), B(p.BiasValue, CK.F32), Bind(n.Out[0])],
                        PcOf(m, nn, k, gflags), ggx, (gwg + ggx - 1) / ggx, name: "gemm"));
                    break;
                }
                output.Add(ND("gemm",
                    [Bind(n.In[0]), Bind(n.In[1]), B(p.BiasValue, CK.F32), Bind(n.Out[0])],
                    PcOf(m, nn, k, gflags, 0u, 0u, 0u),
                    (nn + 63) / 64, (m + 63) / 64, name: "gemm"));
                break;
            }

            case ExecOp.ResizeBilinear:
            {
                var p = (ResizeP)n.P!;
                RequireNhwc(n.In[0]); RequireNhwc(n.Out[0]);
                RequireSingle(n.In[0]);
                bool hasAdd = n.In.Length > 1;
                if (hasAdd) RequireNhwc(n.In[1]);
                int[] si = nhwcIn(0), so = nhwcOut();
                uint flags = (p.FuseRelu ? 1u : 0u) | (hasAdd ? 2u : 0u) | (p.AddRelu ? 4u : 0u);
                bool v4 = Vec4 && si[3] % 4 == 0;
                output.Add(ND(v4 ? "bilinear4" : "bilinear", [Bind(n.In[0]), hasAdd ? Bind(n.In[1]) : B(-1), Bind(n.Out[0])],
                    PcOfF([(uint)si[3], (uint)si[1], (uint)si[2], (uint)so[1], (uint)so[2], flags],
                        p.Scales[2], p.Scales[3]),
                    G((uint)so[1] * (uint)so[2] * (uint)so[3] / (v4 ? 4u : 1u)), name: "bilinear"));
                break;
            }

            case ExecOp.Pad:
            {
                // generic rank-4 pad in storage order (pads permuted to physical dims)
                var p = (PadP)n.P!;
                CValue x = V[n.In[0]];
                if (x.Shape.Length != 4) throw new NotSupportedException("Vulkan pad: rank != 4");
                int[] l = x.Layout ?? Engine.Layout.Identity(4);
                int[] id4 = shapeIn(0), od4 = shapeOut();
                int[] pb4 = new int[4];
                for (int d = 0; d < 4; d++) pb4[d] = p.Pads[l[d]];
                output.Add(ND("pad", [Bind(n.In[0]), Bind(n.Out[0])],
                    PcOfF(U(id4[0], id4[1], id4[2], id4[3], od4[0], od4[1], od4[2], od4[3],
                        pb4[0], pb4[1], pb4[2], pb4[3], p.Mode == PadMode.Reflect ? 1 : 0),
                        p.Value),
                    G((uint)V[n.Out[0]].ElementCount)));
                break;
            }

            case ExecOp.Transpose:
            {
                // only planner relayouts reach here; the perm is physical
                var p = (TransposeP)n.P!;
                int[] is6 = PadR(shapeIn(0)), os6 = PadR(shapeOut());
                int[] ostr = StridesR(os6), istr = StridesR(is6);
                int[] perm6 = PermuteToR(p.Perm, shapeIn(0).Length);
                uint[] pc = new uint[12];
                pc[0] = (uint)nOut();
                for (int k = 0; k < 5; k++) pc[1 + k] = (uint)ostr[k];
                for (int k = 0; k < 6; k++) pc[6 + k] = (uint)istr[perm6[k]];
                (uint tx, uint ty) = G2((uint)nOut());
                output.Add(ND("transpose", [Bind(n.In[0]), Bind(n.Out[0])], PcOf(pc), tx, ty));
                break;
            }

            case ExecOp.Slice:
            {
                // SliceP.Axes are physical after the layout pass
                var p = (SliceP)n.P!;
                int[] is6 = PadR(shapeIn(0)), os6 = PadR(shapeOut());
                int[] ostr = StridesR(os6), istr = StridesR(is6);
                uint[] pc = new uint[24];
                pc[0] = (uint)nOut();
                for (int k = 0; k < 5; k++) pc[1 + k] = (uint)ostr[k];
                for (int k = 0; k < 6; k++) { pc[12 + k] = 1; pc[18 + k] = (uint)istr[k]; }
                int shift = MaxRank - shapeIn(0).Length;
                for (int i = 0; i < p.Axes.Length; i++)
                {
                    pc[6 + p.Axes[i] + shift] = (uint)p.Starts[i];
                    pc[12 + p.Axes[i] + shift] = (uint)p.Steps[i];
                }
                (uint tx, uint ty) = G2((uint)nOut());
                output.Add(ND("slice", [Bind(n.In[0]), Bind(n.Out[0])], PcOf(pc), tx, ty));
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
                        output.Add(ND("concat_ax", [Bind(iv), Bind(n.Out[0])],
                            PcOf((uint)cnt, (uint)inner, (uint)ishape[axis], (uint)dstAxis, (uint)axisOff),
                            G((uint)cnt)));
                    }
                    axisOff += ishape[axis];
                }
                break;
            }

            case ExecOp.HardSigmoid:
            {
                var p = (HardSigmoidP)n.P!;
                output.Add(ND("hsig", [Bind(n.In[0]), Bind(n.Out[0])],
                    PcOfF([(uint)nOut()], p.Alpha, p.Beta), G((uint)nOut())));
                break;
            }

            case ExecOp.GlobalAvgPool:
            {
                RequireNhwc(n.In[0]);
                int[] si = nhwcIn(0);
                uint hw = (uint)si[1] * (uint)si[2], c = (uint)si[3], cg = (c + 63) / 64;
                if (hw <= 512)
                {
                    output.Add(ND("gap", [Bind(n.In[0]), Bind(n.Out[0])], PcOfF([c, hw], 1f / hw), cg, (uint)si[0]));
                    break;
                }
                // large planes: one workgroup per image starves the GPU; reduce pixel chunks in parallel
                uint chunks = Math.Min((hw + 255) / 256, 1024u), per = (hw + chunks - 1) / chunks;
                chunks = (hw + per - 1) / per;
                VkBind part = Scratch((ulong)si[0] * chunks * c * 4);
                output.Add(ND("gap_part", [Bind(n.In[0]), part], PcOf(c, hw, per), cg, chunks, (uint)si[0]));
                output.Add(ND("gap_sum", [part, Bind(n.Out[0])], PcOfF([c, chunks], 1f / hw), cg, (uint)si[0]));
                break;
            }

            case ExecOp.ConvTranspose:
            {
                var p = (ConvTP)n.P!;
                RequireNhwc(n.In[0]); RequireNhwc(n.Out[0]);
                RequireSingle(n.In[0]);
                int[] si = nhwcIn(0), so = nhwcOut();
                uint act = p.Act switch { ActivationKind.None => 0u, ActivationKind.Relu => 1u, ActivationKind.Sigmoid => 2u,
                    _ => throw new NotSupportedException($"Vulkan ConvTranspose: {p.Act}") };
                output.Add(ND("convt", [Bind(n.In[0]), Bind(n.In[1], CK.ConvTW), B(p.BiasValue, CK.F32), Bind(n.Out[0])],
                    PcOf((uint)si[3], (uint)si[1], (uint)si[2], (uint)so[3], (uint)p.Kernel[0], (uint)p.Kernel[1],
                        (p.BiasValue >= 0 ? 1u : 0u) | act << 1),
                    G((uint)nOut())));
                break;
            }

            case ExecOp.ResizeNearest:
            {
                var p = (ResizeNearestP)n.P!;
                RequireNhwc(n.In[0]); RequireNhwc(n.Out[0]);
                RequireSingle(n.In[0]);
                bool hasAdd = n.In.Length > 1;
                if (hasAdd) RequireNhwc(n.In[1]);
                int[] si = nhwcIn(0), so = nhwcOut();
                bool v4 = Vec4 && si[3] % 4 == 0;
                output.Add(ND(v4 ? "nearest4" : "nearest", [Bind(n.In[0]), hasAdd ? Bind(n.In[1]) : B(-1), Bind(n.Out[0])],
                    PcOfF([(uint)si[3], (uint)si[1], (uint)si[2], (uint)so[1], (uint)so[2],
                        (hasAdd ? 2u : 0u) | (p.AddRelu ? 4u : 0u)], p.Scales[2], p.Scales[3]),
                    G((uint)nOut() / (v4 ? 4u : 1u)), name: "nearest"));
                break;
            }

            case ExecOp.ChannelGate:
            {
                // pass 1: gap_part's per-chunk channel sums; pass 2: means + both FC layers + gate, one workgroup per image
                var p = (ChannelGateP)n.P!;
                RequireNhwc(n.In[0]);
                int[] si = nhwcIn(0);
                uint hw = (uint)si[1] * (uint)si[2], c = (uint)si[3], r = (uint)V[n.In[1]].Shape[0];
                uint chunks = Math.Min((hw + 255) / 256, 1024u), per = (hw + chunks - 1) / chunks;
                chunks = (hw + per - 1) / per;
                VkBind part = Scratch((ulong)si[0] * chunks * c * 4);
                output.Add(ND("gap_part", [Bind(n.In[0]), part], PcOf(c, hw, per), (c + 63) / 64, chunks, (uint)si[0]));
                uint gflags = (p.B1 >= 0 ? 1u : 0u) | (p.B2 >= 0 ? 2u : 0u) | (p.Act1 == ActivationKind.Relu ? 4u : 0u) | (p.Hard ? 8u : 0u);
                output.Add(ND("se_gate",
                    [part, Bind(n.In[1], CK.F32), B(p.B1, CK.F32), Bind(n.In[2], CK.F32), B(p.B2, CK.F32), Bind(n.Out[0])],
                    PcOfF([c, r, chunks, gflags], 1f / hw, p.Alpha, p.Beta, p.Offset), (uint)si[0], name: "se_gate"));
                break;
            }

            case ExecOp.Attention:
            {
                var p = (AttentionP)n.P!;
                CValue o = V[n.Out[0]];
                int rank = o.Shape.Length;
                int[] oDims = rank == 4 ? [0, 1, 2, 3] : [-1, 0, 1, 2];
                uint[] View(int v, int[] dims, int fix, int fixIndex)
                {
                    uint[] r = new uint[5];
                    for (int j = 0; j < 4; j++) r[1 + j] = dims[j] < 0 ? 0u : (uint)LogicalStride(V[v], dims[j]);
                    r[0] = fix < 0 ? 0u : (uint)(fixIndex * LogicalStride(V[v], fix));
                    return r;
                }
                uint[] qv = View(n.In[0], p.Dims[0], p.FixAxis[0], p.FixIndex[0]);
                uint[] kv = View(n.In[1], p.Dims[1], p.FixAxis[1], p.FixIndex[1]);
                uint[] vv = View(n.In[2], p.Dims[2], p.FixAxis[2], p.FixIndex[2]);
                uint[] ov = View(n.Out[0], oDims, -1, 0);
                uint T = (uint)o.Shape[^2], DV = (uint)o.Shape[^1], H = (uint)o.Shape[^3], N = rank == 4 ? (uint)o.Shape[0] : 1u;
                uint S = (uint)V[n.In[1]].Shape[p.Dims[1][2]], D = (uint)V[n.In[0]].Shape[p.Dims[0][3]];
                var pc = new List<uint> { T, S, D, DV, H };
                pc.AddRange(qv); pc.AddRange(kv); pc.AddRange(vv); pc.AddRange(ov);
                output.Add(ND(D <= 32 && DV <= 32 ? "attn" : "attn_d64",
                    [Bind(n.In[0]), Bind(n.In[1]), Bind(n.In[2]), Bind(n.Out[0])],
                    PcOfF([.. pc], p.Scale), (T + 63) / 64, N * H, name: "attn"));
                break;
            }

            case ExecOp.LeakyRelu or ExecOp.Tanh or ExecOp.Clip:
            {
                (uint mode, float a, float b) = n.P switch
                {
                    LeakyReluP lp => (0u, lp.Alpha, 0f),
                    ClipP cp => (2u, cp.Min, cp.Max),
                    TanhP tp => (1u, tp.InScale, tp.OutScale),
                    _ => (1u, 1f, 1f),
                };
                (uint ux, uint uy) = G2((uint)nOut());
                output.Add(ND("unary2", [Bind(n.In[0]), Bind(n.Out[0])], PcOfF([(uint)nOut(), mode], a, b), ux, uy));
                break;
            }

            case ExecOp.DepthToSpace or ExecOp.SpaceToDepth:
            {
                var p = (DepthToSpaceP)n.P!;
                string d2sName = n.Op == ExecOp.DepthToSpace ? "d2s" : "s2d";
                RequireNhwc(n.In[0]); RequireNhwc(n.Out[0]);
                int[] si = nhwcIn(0), so = nhwcOut();
                (uint dx, uint dy) = G2((uint)nOut());
                output.Add(ND(d2sName, [Bind(n.In[0]), Bind(n.Out[0])],
                    PcOf((uint)nOut(), (uint)so[3], (uint)so[1], (uint)so[2], (uint)si[1], (uint)si[2], (uint)p.Block, p.Crd ? 1u : 0u),
                    dx, dy, name: d2sName));
                break;
            }

            case ExecOp.HardSwish or ExecOp.Swish:
                output.Add(ND("unary", [Bind(n.In[0]), Bind(n.Out[0])],
                    PcOf((uint)nOut(), n.Op == ExecOp.HardSwish ? 5u : 6u), G((uint)nOut())));
                break;

            case ExecOp.Softmax:
            {
                CValue x = V[n.In[0]];
                uint len = (uint)x.Shape[^1];
                if ((uint)shapeIn(0)[^1] != len) throw new NotSupportedException("Vulkan Softmax: last axis not innermost");
                uint rows = (uint)(x.ElementCount / len);
                uint gx = Math.Min(rows, 65535u);
                output.Add(ND("softmax", [Bind(n.In[0]), Bind(n.Out[0])], PcOf(rows, len),
                    gx, (rows + gx - 1) / gx, name: "softmax"));
                break;
            }

            case ExecOp.AvgPool:
            {
                var p = (PoolP)n.P!;
                RequireNhwc(n.In[0]); RequireNhwc(n.Out[0]);
                int[] si = nhwcIn(0), so = nhwcOut();
                output.Add(ND("avgpool", [Bind(n.In[0]), Bind(n.Out[0])],
                    PcOf((uint)si[3], (uint)si[1], (uint)si[2], (uint)so[1], (uint)so[2],
                        (uint)p.Kernel[0], (uint)p.Kernel[1], (uint)p.Strides[0], (uint)p.Strides[1],
                        (uint)p.Pads[0], (uint)p.Pads[1], p.CountIncludePad ? 1u : 0u, (uint)so[0]),
                    G((uint)nOut()), name: "avgpool"));
                break;
            }

            case ExecOp.MatMul:
            {
                // batched [.., M, K] x [.., K, N] on logical (identity-layout) storage; B may broadcast
                CValue a = V[n.In[0]], b = V[n.In[1]], o = V[n.Out[0]];
                if (a.Layout is not null && !a.Layout.SequenceEqual(Engine.Layout.Identity(a.Shape.Length)) ||
                    b.Layout is not null && !b.Layout.SequenceEqual(Engine.Layout.Identity(b.Shape.Length)) ||
                    o.Layout is not null && !o.Layout.SequenceEqual(Engine.Layout.Identity(o.Shape.Length)))
                    throw new NotSupportedException("Vulkan MatMul: permuted layout");
                uint m = (uint)a.Shape[^2], k = (uint)a.Shape[^1], nn = (uint)b.Shape[^1];
                uint batch = (uint)(o.ElementCount / ((long)m * nn));
                long aBatch = a.ElementCount / ((long)m * k), bBatch = b.ElementCount / ((long)k * nn);
                if (aBatch != batch || (bBatch != batch && bBatch != 1))
                    throw new NotSupportedException("Vulkan MatMul: batch broadcast");
                output.Add(ND("gemm", [Bind(n.In[0]), Bind(n.In[1]), B(-1, CK.F32), Bind(n.Out[0])],
                    PcOf(m, nn, k, 0u, m * k, bBatch == 1 ? 0u : k * nn, m * nn),
                    (nn + 63) / 64, (m + 63) / 64, batch, name: "matmul"));
                break;
            }

            case ExecOp.Ext:
                output.AddRange(n.Ext!.EmitVulkan(this, n));
                break;

            default:
                throw new NotSupportedException($"Vulkan: ExecOp {n.Op} not implemented");
        }
        // epilogues the chosen kernel lacks run in place on the output
        ActivationKind tailAct = n.P switch
        {
            GemmP { Act: ActivationKind.HardSwish or ActivationKind.Swish } g => g.Act,
            ConvP { Act: ActivationKind.Swish } => ActivationKind.Swish,
            _ => ActivationKind.None,
        };
        if (tailAct != ActivationKind.None && !swishDone)
            output.Add(ND("unary", [Bind(n.Out[0]), Bind(n.Out[0])],
                PcOf((uint)nOut(), tailAct == ActivationKind.HardSwish ? 5u : 6u), G((uint)nOut())));

        VkDispatch ND(string shader, VkBind[] binds, byte[] pc, uint gx, uint gy = 1, uint gz = 1, string? name = null) =>
            shader.StartsWith("conv_cm", StringComparison.Ordinal)
                ? new VkDispatch
                {
                    Pipe = _dev.GetCoopPipeline(shader, binds.Length, pc.Length),
                    Pc = pc, Binds = binds, Gx = gx, Gy = gy, Gz = gz, Name = $"{shader}({n.Name})",
                }
                : shader.StartsWith("gemm_cm8", StringComparison.Ordinal)
                ? new VkDispatch
                {
                    // 8x16x16 fragments: the lane mapping needs the subgroup pinned to 16
                    Pipe = _dev.GetPipeline(shader + "_h", binds.Length, pc.Length, 16u),
                    Pc = pc, Binds = binds, Gx = gx, Gy = gy, Gz = gz, Name = $"{shader}({n.Name})",
                }
                : Dispatch(n, shader, binds, pc, gx, gy, gz);

        VkBind B(int valueId, CK? kind = null) => BindOrDummy(valueId, kind);
    }

    /// <summary>A dispatch of <paramref name="shader"/> (fp16 variant on fp16 graphs) embedded in
    /// <paramref name="owner"/> (null = the core).</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public VkDispatch Dispatch(ExecNode n, string shader, VkBind[] binds, byte[] pc, uint gx, uint gy = 1, uint gz = 1,
        Assembly? owner = null) => new()
    {
        Pipe = _dev.GetPipeline(Sh(shader), binds.Length, pc.Length, owner: owner),
        Pc = pc, Binds = binds, Gx = gx, Gy = gy, Gz = gz,
        Name = $"{shader}({n.Name})",
    };

    /// <summary>Binding of a value, or the 64 B zero buffer for <paramref name="valueId"/> &lt; 0.</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public VkBind BindOrDummy(int valueId, CK? kind = null) =>
        valueId >= 0 ? Bind(valueId, kind) : new VkBind(VkSlot.Dummy, 0, 64);

    internal VkDevice Device => _dev;

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public void RequireNhwc(int v)
    {
        CValue cv = _plan.Values[v];
        if (cv.Shape.Length != 4 || !Engine.Layout.Equivalent(cv.Shape, cv.Layout ?? Engine.Layout.Identity(4), Engine.Layout.Nhwc))
            throw new NotSupportedException($"Vulkan: '{cv.Name}' expected NHWC storage, got {Engine.Layout.Str(cv.Layout)}");
    }

    // pad a shape to rank 4 with leading 1s
    private const int MaxRank = 6;

    /// <summary>Right-aligned rank-6 view (leading 1s) for the generic slice / transpose shaders.</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static int[] PadR(int[] s)
    {
        if (s.Length > MaxRank) throw new NotSupportedException($"Vulkan: rank {s.Length} > {MaxRank}");
        int[] r = [1, 1, 1, 1, 1, 1];
        s.CopyTo(r, MaxRank - s.Length);
        return r;
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static int[] StridesR(int[] shape6)
    {
        int[] s = new int[MaxRank]; s[MaxRank - 1] = 1;
        for (int i = MaxRank - 2; i >= 0; i--) s[i] = s[i + 1] * shape6[i + 1];
        return s;
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static int[] PermuteToR(int[] perm, int rank)
    {
        int shift = MaxRank - rank;
        int[] p6 = [0, 1, 2, 3, 4, 5];
        for (int i = 0; i < rank; i++) p6[i + shift] = perm[i] + shift;
        return p6;
    }

    /// <summary>256-thread groups for n items, split over x/y past the 65535 x-limit.</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static (uint x, uint y) G2(uint n)
    {
        uint g = G(n), gx = Math.Min(g, 65535u);
        if (gx == 0) return (0, 1); // empty tensor (e.g. a zero-length pad slice): no-op dispatch
        return (gx, (g + gx - 1) / gx);
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static int[] Pad4(int[] s)
    {
        if (s.Length == 4) return s;
        int[] r = [1, 1, 1, 1];
        s.CopyTo(r, 4 - s.Length);
        return r;
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static int[] Strides(int[] shape4)
    {
        int[] s = new int[4]; s[3] = 1;
        for (int i = 2; i >= 0; i--) s[i] = s[i + 1] * shape4[i + 1];
        return s;
    }

    // lift an original-rank perm into padded rank-4 (leading dims map to themselves)
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static int[] PermuteTo4(int[] perm, int rank)
    {
        int shift = 4 - rank;
        int[] p4 = [0, 1, 2, 3];
        for (int i = 0; i < rank; i++) p4[i + shift] = perm[i] + shift;
        return p4;
    }

    /// <summary>Sessions hold a reference so cache eviction can't free live const/dummy buffers.</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public void Acquire()
    {
        if (Interlocked.Increment(ref _refs) <= 1)
            throw new ObjectDisposedException(nameof(VkGraphModel), "const buffers already freed");
    }
    /// <summary>Acquire unless already freed (no resurrection; for racing cache readers).</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public bool TryAcquire()
    {
        while (true)
        {
            int r = Volatile.Read(ref _refs);
            if (r <= 0) return false;
            if (Interlocked.CompareExchange(ref _refs, r + 1, r) == r) return true;
        }
    }
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public void Release()
    {
        if (Interlocked.Decrement(ref _refs) == 0) _store.Release();
    }

    /// <summary>The plan this graph was compiled from (arena offsets, I/O values).</summary>
    public ExecPlan Plan => _plan;
    public void Dispose() => Release();
}
