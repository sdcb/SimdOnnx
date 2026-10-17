using System.Runtime.InteropServices;
using Sdcb.SimdOnnx.Kernels;
using Sdcb.SimdOnnx.Onnx;

namespace Sdcb.SimdOnnx.Engine;

/// <summary>
/// A reusable inference session over a compiled <see cref="ExecPlan"/>.
/// Not thread-safe; create one per thread or serialize externally.
/// </summary>
public sealed unsafe class InferenceSession : IDisposable
{
    private readonly ExecPlan _plan;
    // native so Dispose frees it deterministically: multi-GB LOH arenas of evicted
    // sessions otherwise pile up until a (rare, on big-RAM hosts) gen2 GC
    private nint _arena;
    private readonly long _arenaBytes;
    private readonly bool _ownsArena;
    private readonly int[] _offsets;
    private readonly long[] _nodeTicks;
    private readonly int _inputValue, _outputValue;

    public InferenceSession(CompiledModel model) : this(model.Plan) { }

    internal InferenceSession(ExecPlan plan) : this(plan, 0) { }

    /// <summary>
    /// Session over <paramref name="plan"/>. With <paramref name="arena"/> = 0 the session owns a
    /// fresh zeroed native arena; otherwise it borrows the caller's arena (at least
    /// ArenaFloats floats; contents need not be zero — every buffer is written before it is
    /// read), which outlives the session and is never freed by it.
    /// </summary>
    internal InferenceSession(ExecPlan plan, nint arena)
    {
        _plan = plan;
        _arenaBytes = (long)plan.ArenaFloats * sizeof(float);
        if (arena == 0)
        {
            // calloc: large blocks come straight from fresh OS pages (already zero, faulted in lazily
            // by the parallel kernels) instead of a single-threaded multi-GB memset up front
            _arena = (nint)NativeMemory.AllocZeroed((nuint)Math.Max(_arenaBytes, 64));
            _ownsArena = true;
            if (_arenaBytes > 0) GC.AddMemoryPressure(_arenaBytes);
        }
        else _arena = arena;
        _offsets = plan.BufOffset;
        _inputValue = plan.InputValue;
        _outputValue = plan.OutputValue;
        _nodeTicks = new long[plan.Nodes.Length];
        CpuKernels.Pack(plan, null);
    }
    /// <summary>Input tensor shape (e.g. [1,3,H,W]).</summary>
    public int[] InputShape => _plan.Values[_inputValue].Shape;
    /// <summary>Intra-op thread budget of <see cref="Run"/> (0 = process default).</summary>
    public int Threads { get; set; }
    internal ExecPlan Plan => _plan;
    public int[] OutputShape => _plan.Values[_outputValue].Shape;

    /// <summary>The writable input window inside the arena (NCHW float).</summary>
    public Span<float> Input => GetSpan(_inputValue);
    /// <summary>Number of graph inputs (<see cref="GetInput(int)"/>).</summary>
    public int InputCount => _plan.Inputs.Length;
    /// <summary>The writable window of graph input <paramref name="index"/> (graph order; 0 is <see cref="Input"/>).</summary>
    public Span<float> GetInput(int index) => GetSpan(_plan.Inputs[index]);
    /// <summary>Logical shape of graph input <paramref name="index"/>.</summary>
    public int[] GetInputShape(int index) => _plan.Values[_plan.Inputs[index]].Shape;
    /// <summary>Name of graph input <paramref name="index"/>.</summary>
    public string GetInputName(int index) => _plan.Values[_plan.Inputs[index]].Name;
    /// <summary>The output tensor after <see cref="Run"/>.</summary>
    public ReadOnlySpan<float> Output => GetSpan(_outputValue);

    public Span<float> GetSpan(int valueId)
    {
        CValue v = _plan.Values[valueId];
        if (v.Buf >= 0)
            return new Span<float>(Arena + _offsets[v.Buf], checked((int)v.ElementCount));
        if (v.Floats != null) return v.Floats;
        throw new InvalidOperationException($"Value '{v.Name}' has no float storage.");
    }

    public ReadOnlySpan<float> GetIn(int valueId)
    {
        if (valueId < 0) return ReadOnlySpan<float>.Empty;
        return GetSpan(valueId);
    }

    /// <summary>Runs the graph. Input must already be written to <see cref="Input"/>.</summary>
    public void Run()
    {
        using var par = Par.Use(Threads);
        ExecNode[] nodes = _plan.Nodes;
        CValue[] values = _plan.Values;
        bool prof = s_profile || EngineProfile.Enabled;
        for (int i = 0; i < nodes.Length; i++)
        {
            ExecNode n = nodes[i];
            long t0 = prof ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
            try { Exec(n); }
            catch (Exception ex)
            {
                string In(int v) => v < 0 ? "-" : $"{values[v].Name}:{string.Join('x', values[v].Shape)}";
                throw new InvalidOperationException(
                    $"node[{i}] {n.Op} '{n.Name}' in=[{string.Join(", ", n.In.Select(In))}] out=[{string.Join(", ", n.Out.Select(In))}]", ex);
            }
            if (prof)
            {
                long dt = System.Diagnostics.Stopwatch.GetTimestamp() - t0;
                _nodeTicks[i] += dt;
                if (EngineProfile.Enabled) EngineProfile.Add(n, dt);
            }
        }
    }

    private static readonly bool s_profile = Environment.GetEnvironmentVariable("GCDRNET_PROF") == "1";

    /// <summary>Per-op-type profile (ticks), populated when GCDRNET_PROF=1.</summary>
    internal Dictionary<ExecOp, TimeSpan> Profile()
    {
        var d = new Dictionary<ExecOp, TimeSpan>();
        for (int i = 0; i < _plan.Nodes.Length; i++)
        {
            ExecOp op = _plan.Nodes[i].Op;
            d.TryGetValue(op, out var cur);
            d[op] = cur + TimeSpan.FromSeconds((double)_nodeTicks[i] / System.Diagnostics.Stopwatch.Frequency);
        }
        return d;
    }

    /// <summary>Per-node profile (node index, op, name, ticks→time), populated when GCDRNET_PROF=1.</summary>
    internal IEnumerable<(int i, ExecNode n, TimeSpan t)> NodeProfile() =>
        _plan.Nodes.Select((n, i) => (i, n, TimeSpan.FromSeconds((double)_nodeTicks[i] / System.Diagnostics.Stopwatch.Frequency)));

    /// <summary>Clears the GCDRNET_PROF counters (e.g. after a warmup run).</summary>
    internal void ResetProfile() => Array.Clear(_nodeTicks);

    /// <summary>Debug/test: run a single node by index. Arena must be pre-initialized.</summary>
    internal void ExecNode(int i) => Exec(_plan.Nodes[i]);

    internal int[] ShapeOf(int valueId) => _plan.Values[valueId].Shape;

    /// <summary>Debug/test: read an activation value's arena region.</summary>
    internal ReadOnlySpan<float> ValueSpan(int valueId) => GetSpan(valueId);

    private void Exec(ExecNode n)
    {
        CValue[] values = _plan.Values;
        if (_plan.Nhwc && ExecNhwc(n)) return;
        switch (n.Op)
        {
            case ExecOp.Conv:
            {
                var p = (ConvP)n.P!;
                CValue x = values[n.In[0]], w = values[n.In[1]], o = values[n.Out[0]];
                ReadOnlySpan<float> bias = p.BiasValue >= 0 ? GetIn(p.BiasValue) : ReadOnlySpan<float>.Empty;
                ReadOnlySpan<float> add = p.AddValue >= 0 ? GetIn(p.AddValue) : ReadOnlySpan<float>.Empty;
                Ops.Conv(GetIn(n.In[0]), x.Shape, GetIn(n.In[1]), w.Shape, bias,
                    GetSpan(n.Out[0]), o.Shape, p, add);
                break;
            }
            case ExecOp.MaxPool:
            {
                var p = (PoolP)n.P!;
                Ops.MaxPool(GetIn(n.In[0]), values[n.In[0]].PhysShape,
                    GetSpan(n.Out[0]), values[n.Out[0]].PhysShape, p);
                break;
            }
            case ExecOp.Elementwise:
            {
                var p = (EwP)n.P!;
                Ops.Elementwise(GetIn(n.In[0]), PhysAligned(n.In[0], n.Out[0]),
                    GetIn(n.In[1]), PhysAligned(n.In[1], n.Out[0]),
                    GetSpan(n.Out[0]), values[n.Out[0]].PhysShape, p.Op, p.Act);
                break;
            }
            case ExecOp.Sqrt: Ops.Sqrt(GetIn(n.In[0]), GetSpan(n.Out[0])); break;
            case ExecOp.HardSwish: Ops.Activation(GetIn(n.In[0]), GetSpan(n.Out[0]), ActivationKind.HardSwish); break;
            case ExecOp.Swish: Ops.Activation(GetIn(n.In[0]), GetSpan(n.Out[0]), ActivationKind.Swish); break;
            case ExecOp.LeakyRelu or ExecOp.Tanh or ExecOp.Clip:
            {
                ReadOnlySpan<float> s = GetIn(n.In[0]);
                Span<float> d = GetSpan(n.Out[0]);
                fixed (float* sp = s, dp = d)
                {
                    if (n.P is LeakyReluP lp) Act.ApplyLeakyRelu(sp, dp, s.Length, lp.Alpha);
                    else if (n.P is ClipP cp) Act.ApplyClip(sp, dp, s.Length, cp.Min, cp.Max);
                    else Act.ApplyTanh(sp, dp, s.Length);
                }
                break;
            }
            case ExecOp.DepthToSpace:
            {
                // NCHW plans (the NHWC path is in ExecNhwc)
                var p = (DepthToSpaceP)n.P!;
                int[] i = values[n.In[0]].Shape;
                for (int b = 0; b < i[0]; b++)
                    Nhwc.DepthToSpace(Img(n.In[0], b, i[0]), i[2], i[3], i[1], ImgOut(n.Out[0], b, i[0]), p.Block, p.Crd, nhwc: false);
                break;
            }
            case ExecOp.Softmax:
            {
                int[] s = values[n.In[0]].Shape;
                Ops.Softmax(GetIn(n.In[0]), GetSpan(n.Out[0]), s[^1]);
                break;
            }
            case ExecOp.ReduceMean:
            {
                int[] s = values[n.In[0]].Shape;
                Ops.ReduceMeanLast(GetIn(n.In[0]), GetSpan(n.Out[0]), s[^1]);
                break;
            }
            case ExecOp.MatMul:
                Ops.MatMul(GetIn(n.In[0]), values[n.In[0]].Shape, GetIn(n.In[1]), values[n.In[1]].Shape,
                    GetSpan(n.Out[0]), values[n.Out[0]].Shape);
                break;
            case ExecOp.AvgPool:
                Ops.AvgPool(GetIn(n.In[0]), values[n.In[0]].PhysShape,
                    GetSpan(n.Out[0]), values[n.Out[0]].PhysShape, (PoolP)n.P!);
                break;
            case ExecOp.GlobalAvgPool:
                Ops.GlobalAvgPool(GetIn(n.In[0]), values[n.In[0]].PhysShape, GetSpan(n.Out[0]));
                break;
            case ExecOp.Relu: Ops.Relu(GetIn(n.In[0]), GetSpan(n.Out[0])); break;
            case ExecOp.Sigmoid: Ops.Sigmoid(GetIn(n.In[0]), GetSpan(n.Out[0])); break;
            case ExecOp.Erf: Ops.Erf(GetIn(n.In[0]), GetSpan(n.Out[0])); break;
            case ExecOp.Gelu: Ops.Gelu(GetIn(n.In[0]), GetSpan(n.Out[0])); break;
            case ExecOp.LayerNorm:
            {
                var p = (LnP)n.P!;
                Ops.LayerNorm(GetIn(n.In[0]), values[n.In[0]].PhysShape,
                    GetIn(n.In[1]), GetIn(n.In[2]), GetSpan(n.Out[0]), p.Eps);
                break;
            }
            case ExecOp.Gemm:
            {
                var p = (GemmP)n.P!;
                ReadOnlySpan<float> bias = p.BiasValue >= 0 ? GetIn(p.BiasValue) : ReadOnlySpan<float>.Empty;
                Ops.Gemm(GetIn(n.In[0]), values[n.In[0]].PhysShape,
                    GetIn(n.In[1]), values[n.In[1]].PhysShape, bias,
                    GetSpan(n.Out[0]), values[n.Out[0]].PhysShape, p);
                break;
            }
            case ExecOp.ResizeBilinear:
            {
                var p = (ResizeP)n.P!;
                Ops.ResizeBilinear(GetIn(n.In[0]), values[n.In[0]].PhysShape,
                    GetSpan(n.Out[0]), values[n.Out[0]].PhysShape, p.Scales,
                    p.FuseRelu ? ActivationKind.Relu : ActivationKind.None);
                break;
            }
            case ExecOp.Pad:
            {
                var p = (PadP)n.P!;
                Ops.Pad(GetIn(n.In[0]), values[n.In[0]].PhysShape,
                    GetSpan(n.Out[0]), values[n.Out[0]].PhysShape, p);
                break;
            }
            case ExecOp.Transpose:
            {
                var p = (TransposeP)n.P!;
                Ops.Transpose(GetIn(n.In[0]), values[n.In[0]].PhysShape,
                    GetSpan(n.Out[0]), values[n.Out[0]].PhysShape, p.Perm);
                break;
            }
            case ExecOp.Slice:
            {
                var p = (SliceP)n.P!;
                Ops.Slice(GetIn(n.In[0]), values[n.In[0]].PhysShape,
                    GetSpan(n.Out[0]), values[n.Out[0]].PhysShape, p);
                break;
            }
            case ExecOp.Concat:
            {
                var p = (ConcatP)n.P!;
                var ins = new FBuf[n.In.Length];
                var shapes = new int[n.In.Length][];
                for (int k = 0; k < n.In.Length; k++)
                {
                    CValue cv = values[n.In[k]];
                    ins[k] = cv.Buf >= 0
                        ? new FBuf(Arena + _offsets[cv.Buf], checked((int)cv.ElementCount))
                        : new FBuf(cv.Floats!, 0, checked((int)cv.ElementCount));
                    shapes[k] = cv.PhysShape;
                }
                Ops.Concat(ins, shapes, GetSpan(n.Out[0]), values[n.Out[0]].PhysShape, p.Axis);
                break;
            }
            case ExecOp.Copy:
                GetIn(n.In[0]).CopyTo(GetSpan(n.Out[0]));
                break;
            case ExecOp.Ext:
                n.Ext!.RunCpu(this, n);
                break;
            case ExecOp.HardSigmoid:
            {
                var p = (HardSigmoidP)n.P!;
                Nhwc.HardSigmoid(GetIn(n.In[0]), GetSpan(n.Out[0]), p.Alpha, p.Beta);
                break;
            }
            case ExecOp.ChannelGate:
            {
                var p = (ChannelGateP)n.P!;
                int[] s = values[n.In[0]].Shape;   // [N,C,H,W]; NHWC storage in layout-planned plans
                int nb = s[0], c = s[1], hw = s[2] * s[3], r = values[n.In[1]].Shape[0];
                ReadOnlySpan<float> x = GetIn(n.In[0]), w1 = GetIn(n.In[1]), w2 = GetIn(n.In[2]);
                ReadOnlySpan<float> b1 = GetIn(p.B1), b2 = GetIn(p.B2);
                Span<float> o = GetSpan(n.Out[0]);
                for (int b = 0; b < nb; b++)
                    Nhwc.ChannelGate(x.Slice(b * c * hw, c * hw), hw, c, _plan.Nhwc, w1, b1, w2, b2, r, p, o.Slice(b * c, c));
                break;
            }
            default:
                throw new NotSupportedException($"ExecOp {n.Op}");
        }
    }

    /// <summary>Channels-last kernels for the layout-sensitive ops; false = generic (storage-shape) path.
    /// Spatial kernels address one image; N &gt; 1 loops over the batch.</summary>
    private bool ExecNhwc(ExecNode n)
    {
        CValue[] values = _plan.Values;
        switch (n.Op)
        {
            case ExecOp.Conv:
            {
                var p = (ConvP)n.P!;
                int[] i = Nhwc4(values[n.In[0]]), o = Nhwc4(values[n.Out[0]]);
                int nb = i[0];
                if (p.Pool is not null)
                {
                    for (int b = 0; b < nb; b++)
                        Nhwc.ConvPool(Img(n.In[0], b, nb), i[1], i[2], i[3], ImgOut(n.Out[0], b, nb), o[1], o[2], o[3],
                            p, (NhwcConvPlan)p.KernelPlan!);
                    return true;
                }
                // one call for the whole batch: kernels shard batch × rows in a single parallel region
                ReadOnlySpan<float> addAll = p.AddValue >= 0 ? GetIn(p.AddValue) : ReadOnlySpan<float>.Empty;
                Nhwc.Conv(GetIn(n.In[0]), i[1], i[2], i[3], GetSpan(n.Out[0]), o[1], o[2], o[3],
                    p, (NhwcConvPlan)p.KernelPlan!, addAll, nb);
                return true;
            }
            case ExecOp.MaxPool:
            {
                int[] i = Nhwc4(values[n.In[0]]), o = Nhwc4(values[n.Out[0]]);
                for (int b = 0; b < i[0]; b++)
                    Nhwc.MaxPool(Img(n.In[0], b, i[0]), i[1], i[2], i[3], ImgOut(n.Out[0], b, i[0]), o[1], o[2], (PoolP)n.P!);
                return true;
            }
            case ExecOp.AvgPool:
            {
                int[] i = Nhwc4(values[n.In[0]]), o = Nhwc4(values[n.Out[0]]);
                for (int b = 0; b < i[0]; b++)
                    Nhwc.AvgPool(Img(n.In[0], b, i[0]), i[1], i[2], i[3], ImgOut(n.Out[0], b, i[0]), o[1], o[2], (PoolP)n.P!);
                return true;
            }
            case ExecOp.ResizeBilinear:
            {
                var p = (ResizeP)n.P!;
                int[] i = Nhwc4(values[n.In[0]]), o = Nhwc4(values[n.Out[0]]);
                for (int b = 0; b < i[0]; b++)
                    Nhwc.ResizeBilinear(Img(n.In[0], b, i[0]), i[1], i[2], i[3], ImgOut(n.Out[0], b, i[0]), o[1], o[2],
                        p.Scales[2], p.Scales[3], p.FuseRelu ? ActivationKind.Relu : ActivationKind.None,
                        n.In.Length > 1 ? Img(n.In[1], b, i[0]) : default, p.AddRelu);
                return true;
            }
            case ExecOp.Pad:
            {
                int[] i = Nhwc4(values[n.In[0]]), o = Nhwc4(values[n.Out[0]]);
                for (int b = 0; b < i[0]; b++)
                    Nhwc.Pad(Img(n.In[0], b, i[0]), i[1], i[2], i[3], ImgOut(n.Out[0], b, i[0]), o[1], o[2], (PadP)n.P!);
                return true;
            }
            case ExecOp.Gemm when ((GemmP)n.P!).KernelPlan is NhwcConvPlan gk:
            {
                var p = (GemmP)n.P!;
                int k = values[n.In[0]].Shape[^1], m = (int)(values[n.In[0]].ElementCount / k);
                Nhwc.Gemm(GetIn(n.In[0]), m, k, GetSpan(n.Out[0]), values[n.Out[0]].Shape[^1], gk, p.Act);
                return true;
            }
            case ExecOp.GlobalAvgPool:
            {
                int[] i = Nhwc4(values[n.In[0]]);
                for (int b = 0; b < i[0]; b++)
                    Nhwc.GlobalAvgPool(Img(n.In[0], b, i[0]), i[1] * i[2], i[3], ImgOut(n.Out[0], b, i[0]));
                return true;
            }
            case ExecOp.ConvTranspose:
            {
                var p = (ConvTP)n.P!;
                int[] i = Nhwc4(values[n.In[0]]), o = Nhwc4(values[n.Out[0]]);
                for (int b = 0; b < i[0]; b++)
                    Nhwc.ConvTranspose(Img(n.In[0], b, i[0]), i[1], i[2], i[3], ImgOut(n.Out[0], b, i[0]), o[3],
                        p.Kernel[0], p.Kernel[1], (ConvTPlan)p.KernelPlan!, p.Act);
                return true;
            }
            case ExecOp.DepthToSpace:
            {
                var p = (DepthToSpaceP)n.P!;
                int[] i = Nhwc4(values[n.In[0]]);
                for (int b = 0; b < i[0]; b++)
                    Nhwc.DepthToSpace(Img(n.In[0], b, i[0]), i[1], i[2], i[3], ImgOut(n.Out[0], b, i[0]), p.Block, p.Crd, nhwc: true);
                return true;
            }
            case ExecOp.ResizeNearest:
            {
                var p = (ResizeNearestP)n.P!;
                int[] i = Nhwc4(values[n.In[0]]), o = Nhwc4(values[n.Out[0]]);
                for (int b = 0; b < i[0]; b++)
                    Nhwc.ResizeNearest(Img(n.In[0], b, i[0]), i[1], i[2], i[3], ImgOut(n.Out[0], b, i[0]), o[1], o[2], p.Scales[2], p.Scales[3],
                        n.In.Length > 1 ? Img(n.In[1], b, i[0]) : default, p.AddRelu);
                return true;
            }
            default:
                return false;
        }
    }

    /// <summary>[N, H, W, C] of a rank-4 logical NCHW value. The layout pass guarantees NHWC-equivalent
    /// storage for NHWC kernels, but an equivalent layout (size-1 dims) keeps its own PhysShape order.</summary>
    private static int[] Nhwc4(CValue v) => v.Shape is [var nn, var c, var h, var w] ? [nn, h, w, c] : v.PhysShape;

    /// <summary>Image <paramref name="b"/> of an <paramref name="nb"/>-image batch tensor.</summary>
    private ReadOnlySpan<float> Img(int v, int b, int nb)
    {
        ReadOnlySpan<float> s = GetIn(v);
        if (nb == 1) return s;
        int per = s.Length / nb;
        return s.Slice(b * per, per);
    }

    private Span<float> ImgOut(int v, int b, int nb)
    {
        Span<float> s = GetSpan(v);
        if (nb == 1) return s;
        int per = s.Length / nb;
        return s.Slice(b * per, per);
    }

    /// <summary>
    /// Broadcast operand shape in the output's physical axis order. The layout pass stores an operand
    /// in the output layout restricted to its own dims, so its data can be addressed with this shape
    /// (a [C,1,1] const against an NHWC output becomes [1,1,1,C]).
    /// </summary>
    private int[] PhysAligned(int v, int outV)
    {
        CValue a = _plan.Values[v], o = _plan.Values[outV];
        if (o.Layout is not { } lo) return a.PhysShape;
        // also for same-shape operands: an equivalent layout (size-1 dims) has its own PhysShape order
        int off = lo.Length - a.Shape.Length;
        int[] p = new int[lo.Length];
        for (int i = 0; i < lo.Length; i++) p[i] = lo[i] - off >= 0 ? a.Shape[lo[i] - off] : 1;
        return p;
    }

    private float* Arena => _arena != 0 ? (float*)_arena : throw new ObjectDisposedException(nameof(InferenceSession));

    public void Dispose()
    {
        Free();
        GC.SuppressFinalize(this);
    }

    ~InferenceSession() => Free();

    private void Free()
    {
        nint p = Interlocked.Exchange(ref _arena, 0);
        if (p == 0 || !_ownsArena) return;
        NativeMemory.Free((void*)p);
        if (_arenaBytes > 0) GC.RemoveMemoryPressure(_arenaBytes);
    }
}
