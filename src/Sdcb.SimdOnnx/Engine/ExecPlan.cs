namespace Sdcb.SimdOnnx.Engine;

/// <summary>Executable op set — the lowered form the session runs.</summary>
public enum ExecOp
{
    Conv,            // NCHW conv, optional bias + fused activation
    MaxPool,         // NCHW pooling
    Elementwise,     // Add/Sub/Mul/Div with broadcast (incl. scalar const)
    Relu,
    Sigmoid,
    Erf,             // elementwise erf (normally fused into Gelu)
    Gelu,            // exact-erf GELU, elementwise
    LayerNorm,       // normalize over last dim
    Gemm,            // [.., M, K] x [K, N] + bias[N]
    ResizeBilinear,  // NCHW, half_pixel, per-axis scales
    Pad,             // constant / reflect on trailing dims
    Transpose,       // arbitrary permute (physical perm in layout-planned plans)
    Slice,           // strided subrange copy
    Concat,          // axis join
    Copy,            // plain copy (for Squeeze/Unsqueeze/identity leftovers)
    HardSigmoid,     // clamp(alpha·x + beta, 0, 1)
    GlobalAvgPool,   // mean over H,W (GlobalAveragePool / ReduceMean axes=[2,3] keepdims)
    ConvTranspose,   // stride == kernel, no pad (non-overlapping taps), optional bias + act
    ResizeNearest,   // nearest, asymmetric, floor
    MatMul,          // batched [.., M, K] x [.., K, N], both activations (const B lowers to Gemm)
    Softmax,         // over the last axis
    ReduceMean,      // mean over the last axis, keepdims
    Sqrt,
    AvgPool,         // NCHW average pooling (PoolP; count_include_pad = 0)
    HardSwish,       // x * clamp(x / 6 + 0.5, 0, 1) (CPU plans: HardSigmoid + Mul)
    Swish,           // x * sigmoid(x) (CPU plans: Sigmoid + Mul)
    Ext,             // consumer-registered operator (ExecNode.Ext)
    ChannelGate,     // squeeze-excitation gate: GAP -> 1x1 conv (+act) -> 1x1 conv -> (hard) sigmoid, [N,C,1,1]
    Attention,       // softmax(scale * q k^T) v over strided q / k / v views (GPU plans)
    LeakyRelu,       // x >= 0 ? x : alpha * x (LeakyReluP)
    Tanh,
    Clip,            // clamp(x, min, max) (ClipP)
    DepthToSpace,    // NCHW [N, C*b*b, H, W] -> [N, C, H*b, W*b] (DepthToSpaceP)
}

public enum ActivationKind { None, Relu, Sigmoid, Gelu, HardSwish, Swish }

public enum PadMode { Constant, Reflect }

public enum EwOp { Add, Sub, Mul, Div }

// ---- per-op parameter records (resolved at compile time) ----

public sealed class ConvP
{
    public required int[] Pads;      // [top, left, bottom, right] spatial only
    public required int[] Strides;   // [sh, sw]
    public required int[] Dilations; // [dh, dw]
    public required int[] Kernel;    // [kh, kw]
    public int Group = 1;
    public ActivationKind Act;
    public PadMode PadMode = PadMode.Constant;
    public float PadValue;           // constant-mode fill
    public int BiasValue = -1;       // value id of per-channel bias, -1 none
    public int AddValue = -1;        // value id of same-shape residual addend, -1 none
    public object? KernelPlan;       // packed weights / kernel choice, filled at plan build
    // NHWC plans: fused 2x2/2 max pool epilogue — Out[0] is the pooled tensor
    public PoolP? Pool;
}

public sealed class PoolP
{
    public required int[] Kernel;
    public required int[] Strides;
    public required int[] Pads;
    public bool FuseRelu;
    public bool CountIncludePad;     // AvgPool only
}

public sealed class EwP
{
    public required EwOp Op;
    public ActivationKind Act;
}

public sealed class LnP
{
    public float Eps;
}

public sealed class GemmP
{
    public bool TransB;             // B stored [N,K] when true (onnx transB=1)
    public int BiasValue = -1;
    public ActivationKind Act;
    public object? KernelPlan;      // backend-private packed B, filled at session build
}

public sealed class ResizeP
{
    public required float[] Scales; // per-dim, only [.., sh, sw] used
    public bool FuseRelu;           // relu on the resized value (before the optional add)
    // optional same-shape addend in In[1] (NHWC plans): out = [relu](resize(x)) + [relu](In[1])
    public bool AddRelu;            // relu on the addend
}

public sealed class HardSigmoidP
{
    public float Alpha, Beta;
}

public sealed class ConvTP
{
    public required int[] Kernel;    // [kh, kw] == strides
    public int BiasValue = -1;       // [oc] const (also folded from a trailing Add([1,oc,1,1] const))
    public ActivationKind Act;
    public object? KernelPlan;
}

public sealed class ResizeNearestP
{
    public required float[] Scales;
    // optional same-shape addend in In[1] (NHWC plans): out = resize(x) + [relu](In[1])
    public bool AddRelu;
}

public sealed class LeakyReluP
{
    public float Alpha = 0.01f;
}

public sealed class ClipP
{
    public float Min = float.NegativeInfinity, Max = float.PositiveInfinity;
}

public sealed class DepthToSpaceP
{
    public int Block;
    /// <summary>CRD: input channel c·b² + i·b + j feeds output (c, h·b + i, w·b + j); DCR: (i·b + j)·C + c.</summary>
    public bool Crd;
}

/// <summary>
/// Squeeze-excitation gate (In = [x, W1, W2]; W1 [R,C,1,1], W2 [C,R,1,1]):
/// s[n,c] = gate(b2[c] + Σ_r W2[c,r]·act1(b1[r] + Σ_k W1[r,k]·mean_hw(x[n,k]))) + Offset, out [N,C,1,1].
/// </summary>
public sealed class ChannelGateP
{
    public int B1 = -1, B2 = -1;    // [R] / [C] bias value ids, -1 none
    public ActivationKind Act1;     // None / Relu
    public bool Hard;               // clamp(Alpha·z + Beta, 0, 1), else sigmoid(z)
    public float Alpha, Beta;
    public float Offset;            // 1 when x·s + x folded into x·(s + 1)
}

/// <summary>
/// Scaled dot-product attention, out[n,h,t,:] = Σ_s softmax_s(Scale · q[n,h,t,:]·k[n,h,s,:]) v[n,h,s,:]
/// ([N,H,T,D] logical). In = [q, k, v] are views of earlier values: operand i reads In[i] at
/// logical index Dims[i] = the axes holding (n, h, t|s, d) (-1: size 1), with axis FixAxis[i]
/// (if >= 0) pinned to FixIndex[i]; any physical layout.
/// </summary>
public sealed class AttentionP
{
    public float Scale = 1f;
    public required int[][] Dims;
    public required int[] FixAxis;
    public required int[] FixIndex;
    /// <summary>Output logical->physical layout chosen at fusion (its consumer's transpose dissolves).</summary>
    public int[]? OutLayout;
}

public sealed class PadP
{
    public required int[] Pads;     // [2*rank] begin..end
    public PadMode Mode = PadMode.Constant;
    public float Value;
}

public sealed class TransposeP
{
    public required int[] Perm;
}

public sealed class SliceP
{
    public required int[] Starts;
    public required int[] Ends;
    public required int[] Axes;
    public required int[] Steps;
}

public sealed class ConcatP
{
    public int Axis;
}

/// <summary>A resolved tensor value: compile-time constant or activation.</summary>
public sealed class CValue
{
    public int[] Shape = [];
    public int DType;            // onnx dtype: 1 float, 7 int64
    public long[]? Ints;         // compile-time int content
    public float[]? Floats;      // compile-time float content (weights or small consts)
    public int Buf = -1;         // activation buffer id; >= 0 → runtime tensor
    public string Name = "";
    /// <summary>Physical dim order: physical dim i holds logical dim Layout[i]. null = identity.</summary>
    public int[]? Layout;
    /// <summary>Zero-copy view source (reshape/squeeze/unsqueeze/identity); -1 none.</summary>
    public int AliasOf = -1;
    public bool IsInt => Ints != null;
    public bool IsConst => Ints != null || Floats != null;
    public bool IsActivation => Buf >= 0;
    public long ElementCount => Shape.Aggregate(1L, (a, b) => a * b);
    /// <summary>Shape in storage order — what kernels address by.</summary>
    public int[] PhysShape => Layout is null ? Shape : Engine.Layout.Phys(Shape, Layout);
}

/// <summary>A buffer reference into either the activation arena or a const array.</summary>
internal readonly unsafe struct FBuf
{
    public readonly float[]? Data;
    public readonly float* Ptr;
    public readonly int Offset;
    public readonly int Length;
    public FBuf(float[] data, int offset, int length) { Data = data; Offset = offset; Length = length; }
    public FBuf(float* ptr, int length) { Ptr = ptr; Length = length; }
    public Span<float> Span => Data != null ? Data.AsSpan(Offset, Length) : new Span<float>(Ptr, Length);
}

/// <summary>One executable step: op + value ids.</summary>
public sealed class ExecNode
{
    public required ExecOp Op;
    public required int[] In;    // value ids
    public required int[] Out;
    public object? P;
    public ExtOp? Ext;           // set when Op == ExecOp.Ext
    public int SrcNode = -1;     // onnx node index (diagnostics)
    public string Name = "";

    /// <summary>Value ids read through op params rather than <see cref="In"/> (bias / residual operands).</summary>
    public IEnumerable<int> ParamRefs()
    {
        // core params without an iterator (compile passes call this per node); extensions as before
        if (Ext is null)
            return P switch
            {
                ConvP { BiasValue: >= 0, AddValue: >= 0 } c => [c.BiasValue, c.AddValue],
                ConvP { BiasValue: >= 0 } c => [c.BiasValue],
                ConvP { AddValue: >= 0 } c => [c.AddValue],
                GemmP { BiasValue: >= 0 } g => [g.BiasValue],
                ConvTP { BiasValue: >= 0 } t => [t.BiasValue],
                ChannelGateP { B1: >= 0, B2: >= 0 } s => [s.B1, s.B2],
                ChannelGateP { B1: >= 0 } s => [s.B1],
                ChannelGateP { B2: >= 0 } s => [s.B2],
                _ => Array.Empty<int>(),
            };
        return ParamRefsWithExt();
    }

    private IEnumerable<int> ParamRefsWithExt()
    {
        switch (P)
        {
            case ConvP c: if (c.BiasValue >= 0) yield return c.BiasValue; if (c.AddValue >= 0) yield return c.AddValue; break;
            case GemmP g: if (g.BiasValue >= 0) yield return g.BiasValue; break;
            case ConvTP t: if (t.BiasValue >= 0) yield return t.BiasValue; break;
        }
        if (Ext is not null)
            foreach (int v in Ext.ParamRefs(this)) yield return v;
    }
}

/// <summary>Compiled graph ready for a session: exec nodes + buffer plan.</summary>
public sealed class ExecPlan
{
    public required ExecNode[] Nodes;
    public required CValue[] Values;
    public required int[] BufSize;      // floats per activation buffer
    public required int[] BufOffset;    // arena offset per buffer
    public required int ArenaFloats;
    public required int InputValue;
    /// <summary>Every graph input, in graph order (InputValues[0] == InputValue); their buffers sit back to
    /// back in input order at the start of the input region.</summary>
    public int[] InputValues = [];
    public required int OutputValue;
    /// <summary>Rank-4 activations stored channels-last (layout pass ran).</summary>
    public bool Nhwc;
    /// <summary>Input/output stay in plan layout (no boundary relayout); else logical NCHW.</summary>
    public bool PhysicalIo;
    /// <summary>The options the plan was compiled with (backend kernel choices read them).</summary>
    public CompileOptions? Options;

    /// <summary>Graph inputs, in graph order (plans built before multi-input support: the one input).</summary>
    internal int[] Inputs => InputValues.Length > 0 ? InputValues : [InputValue];

    /// <summary>Arena span [first, end) in elements holding every input (input buffers are planned back to
    /// back, input 0 first) and each input's element offset into it.</summary>
    internal (long first, long count, long[] offsets) InputRegion()
    {
        int[] ins = Inputs;
        long first = BufOffset[Values[ins[0]].Buf], end = first;
        var offs = new long[ins.Length];
        for (int k = 0; k < ins.Length; k++)
        {
            long o = BufOffset[Values[ins[k]].Buf];
            if (o < first) throw new InvalidOperationException("graph inputs are not laid out in input order");
            offs[k] = o - first;
            end = Math.Max(end, o + Values[ins[k]].ElementCount);
        }
        return (first, end - first, offs);
    }
}

/// <summary>Backend a plan is compiled for (fusions are restricted to what it implements).</summary>
public enum PlanTarget { Cpu, Vulkan, Metal }

/// <summary>Plan compilation options.</summary>
/// <param name="MemFuse">memory fusions (conv+pool and extension memory passes; NHWC only).</param>
/// <param name="Target">backend the plan runs on.</param>
/// <param name="Extensions">consumer-registered operators and graph passes.</param>
/// <param name="OutputName">tensor to stop at instead of the graph output (nodes it does not need are dropped).</param>
public sealed record CompileOptions(bool Nhwc, bool PhysicalIo = false, bool MemFuse = true,
    PlanTarget Target = PlanTarget.Cpu, EngineExtensions? Extensions = null, string? OutputName = null,
    bool LogicalOutput = false, bool Fuse = true, bool StagedCoopConv = false, bool? Vec4Eltwise = null,
    IReadOnlyList<int[]?>? InputShapes = null)
{
    // InputShapes: shapes of the graph inputs after the first (null / missing: the model's static
    // dims). Compared by reference in backend graph caches: reuse one options instance per shape set.
    // Vec4Eltwise: GPU maxpool / bilinear / depthwise take 4 channels per invocation when c % 4 == 0.
    // null = the device default (VkDevice.Vec4Eltwise: integrated and Intel GPUs); true everywhere
    // matches the old OCR engine, which used the vec4 kernels on every device.
    // StagedCoopConv: on 8x16x16 (SG16, Intel Xe2) devices, take the shared-staged conv_cm build
    // for dense convs. Much faster on wide feature maps, but some workloads (the OCR models)
    // intermittently lose the device with it on the Arc B580, so the default is the direct-load
    // family (pw_dot / conv_kdot / gemm_cm8). Opt in only for workloads stress-tested with it.
    public bool Gpu => Target != PlanTarget.Cpu;
    /// <summary>SIMDONNX_NHWC=0 (or legacy GCDRNET_NHWC=0) forces the all-NCHW plan (A/B oracle).</summary>
    public static bool DefaultNhwc { get; } = Env.Flag("NHWC") != "0";
    public static CompileOptions Default => new(DefaultNhwc);
    /// <summary>SIMDONNX_NOFUSE_MEM=1 disables the CPU memory fusions.</summary>
    public static bool MemFusions { get; } = Env.Flag("NOFUSE_MEM") != "1";
}

/// <summary>Engine environment switches: SIMDONNX_X, falling back to the legacy GCDRNET_X.</summary>
internal static class Env
{
    // read once per name: knobs are set before the engine starts, and compiles query them per node
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, string?> s_flags = new();

    public static string? Flag(string name) => s_flags.GetOrAdd(name, static n =>
        Environment.GetEnvironmentVariable("SIMDONNX_" + n) ?? Environment.GetEnvironmentVariable("GCDRNET_" + n));
}
