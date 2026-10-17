using System.Runtime.CompilerServices;
namespace Sdcb.SimdOnnx.Engine;

/// <summary>
/// Physical layout pass (runs after fusion, before arena planning). Every
/// activation value gets a <see cref="CValue.Layout"/>; logical shapes are
/// untouched, so fusions and golden shapes stay in logical NCHW terms.
///
/// Rules (global channels-last; no runtime segment decisions — this model is a
/// homogeneous conv + token-MLP graph):
/// <list type="bullet">
/// <item>Conv / MaxPool / Resize / Pad / ChannelShift consume and produce
///   rank-4 NHWC (<see cref="Layout.Nhwc"/>).</item>
/// <item>LayerNorm / Gemm need their last logical dim innermost and keep the
///   input layout; elementwise / unary ops follow their full-shape input.</item>
/// <item>Every graph Transpose dissolves into a zero-copy view
///   (<see cref="Layout.Transposed"/>); reshape/squeeze aliases stay views when
///   <see cref="Layout.Reshaped"/> can express them. Layout bookkeeping follows
///   the Buf alias chain (<see cref="CValue.AliasOf"/>), not just ExecNodes —
///   aliases have no producer node.</item>
/// <item>A consumer whose requirement is not met gets an explicit relayout
///   (a physical Transpose node). On this model the only ones are the I/O
///   boundary relayouts when <c>physicalIo</c> is off.</item>
/// </list>
/// Op params that name axes are rewritten into physical terms (Slice axes,
/// Concat axis, relayout perms), so generic kernels address purely by
/// <see cref="CValue.PhysShape"/>.
/// </summary>
public sealed class LayoutPlanner
{
    private readonly List<ExecNode> _nodes;                      // planned output, in order
    private readonly List<CValue> _values;
    private readonly List<int> _bufSize;
    private readonly Dictionary<int, int[]> _perm = [];          // dissolved transpose → logical perm
    private readonly Dictionary<(int v, string l), int> _relayout = [];
    private bool _drop;                                          // the node being planned dissolved
    private int _dissolved;

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private LayoutPlanner(List<ExecNode> nodes, List<CValue> values, List<int> bufSize)
    {
        _nodes = nodes; _values = values; _bufSize = bufSize;
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static void Run(List<ExecNode> nodes, List<CValue> values, List<int> bufSize,
        int[] inputValues, ref int outputValue, bool physicalIn, bool physicalOut)
    {
        // streamed: relayouts land before the node being planned, dissolved transposes are dropped
        // (one pass, no Insert / RemoveAt shifting the list)
        ExecNode[] input = nodes.ToArray();
        nodes.Clear();
        var lp = new LayoutPlanner(nodes, values, bufSize);
        foreach (int inputValue in inputValues)
        {
            CValue inV = values[inputValue];
            inV.Layout = physicalIn && inV.Shape.Length == 4 ? Layout.Nhwc : Layout.Identity(inV.Shape.Length);
        }

        foreach (ExecNode n in input)
        {
            lp._drop = false;
            lp.Plan(n);
            if (!lp._drop) nodes.Add(n);
        }

        lp.L(outputValue);
        if (!physicalOut)
        {
            CValue ov = values[outputValue];
            int[] id = Layout.Identity(ov.Shape.Length);
            if (!Layout.Equivalent(ov.Shape, ov.Layout!, id))
                outputValue = lp.Relayout(outputValue, id);
        }
        // dead aliases (never consumed): identity layout so dumps never see null.
        // Resolving them could insert relayout nodes for values nobody reads.
        for (int v = 0; v < values.Count; v++)
            if (values[v].Buf >= 0 && values[v].Layout is null)
                values[v].Layout = Layout.Identity(values[v].Shape.Length);

        if (Env.Flag("DUMP") == "1")
            Console.Error.WriteLine($"[layout] dissolved {lp._dissolved} transposes, {lp._relayout.Count} relayouts");
    }

    /// <summary>Resolves (and caches) the layout of value <paramref name="v"/>, following aliases.</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public int[] L(int v)
    {
        CValue cv = _values[v];
        if (cv.Layout is not null) return cv.Layout;
        if (cv.Buf < 0) return cv.Layout = Layout.Identity(cv.Shape.Length);   // const
        if (cv.AliasOf < 0)
            throw new InvalidOperationException(
                $"layout: value '{cv.Name}' (v{v}) consumed before its producer was planned.");
        int s = cv.AliasOf;
        int[] ls = L(s);
        if (_perm.TryGetValue(v, out int[]? perm))
        {
            cv.Buf = _values[s].Buf;
            return cv.Layout = Layout.Transposed(ls, perm);
        }
        int[]? r = Layout.Reshaped(_values[s].Shape, ls, cv.Shape);
        if (r is null)
        {
            // the reshape needs data movement: materialize the source in
            // logical order, then the reshape is an identity-layout view of that
            int[] id = Layout.Identity(_values[s].Shape.Length);
            s = Relayout(s, id);
            cv.AliasOf = s;
            r = Layout.Reshaped(_values[s].Shape, id, cv.Shape)
                ?? throw new InvalidOperationException("identity reshape must be a view");
        }
        cv.Buf = _values[s].Buf;
        return cv.Layout = r;
    }

    /// <summary>Appends (before the current node) a physical copy of <paramref name="v"/> in layout <paramref name="target"/>.</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private int Relayout(int v, int[] target)
    {
        var key = (v, Layout.Str(target));
        if (_relayout.TryGetValue(key, out int done)) return done;
        CValue src = _values[v];
        int[] from = L(v);
        int buf = _bufSize.Count;
        _bufSize.Add(checked((int)src.ElementCount));
        int id = _values.Count;
        _values.Add(new CValue { Name = src.Name + "@relayout", Shape = src.Shape, DType = 1, Buf = buf, Layout = target });
        _nodes.Add(new ExecNode
        {
            Op = ExecOp.Transpose, In = [v], Out = [id],
            P = new TransposeP { Perm = Layout.PhysPerm(from, target) },
            Name = $"relayout{Layout.Str(from)}->{Layout.Str(target)}",
        });
        _relayout[key] = id;
        return id;
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public int Need(int v, int[] target) =>
        Layout.Equivalent(_values[v].Shape, L(v), target) ? v : Relayout(v, target);

    public void Need(ExecNode n, int k, int[] target) => n.In[k] = Need(n.In[k], target);

    public List<CValue> Values => _values;

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static bool LastInnermost(int[] shape, int[] l)
    {
        int r = shape.Length - 1;
        if (r < 0 || shape[r] == 1) return true;
        for (int i = l.Length - 1; i >= 0; i--)
            if (shape[l[i]] != 1) return l[i] == r;
        return true;
    }

    private void SetOut(ExecNode n, int[] l) => _values[n.Out[0]].Layout = l;

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private void Plan(ExecNode n)
    {
        switch (n.Op)
        {
            case ExecOp.Transpose:
            {
                // dissolve: y = T(x) becomes a view of x's storage
                int x = n.In[0], o = n.Out[0];
                L(x);
                _values[o].AliasOf = x;
                _values[o].Layout = null;
                _perm[o] = ((TransposeP)n.P!).Perm;
                L(o);
                _drop = true;
                _dissolved++;
                break;
            }
            case ExecOp.Conv or ExecOp.MaxPool or ExecOp.ResizeBilinear or ExecOp.Pad
                or ExecOp.GlobalAvgPool or ExecOp.ConvTranspose or ExecOp.ResizeNearest or ExecOp.AvgPool or ExecOp.DepthToSpace or ExecOp.SpaceToDepth:
            {
                if (_values[n.In[0]].Shape.Length != 4 || _values[n.Out[0]].Shape.Length != 4)
                    throw new NotSupportedException($"layout: {n.Op} '{n.Name}' is not rank-4.");
                if (n.P is PadP pp)
                {
                    int r = 4;
                    if (pp.Pads[0] != 0 || pp.Pads[1] != 0 || pp.Pads[r] != 0 || pp.Pads[r + 1] != 0)
                        throw new NotSupportedException($"layout: Pad '{n.Name}' pads N/C.");
                }
                if (n.P is ConvP cp)
                {
                    if (cp.Dilations.Any(d => d != 1))
                        throw new NotSupportedException($"layout: dilated Conv '{n.Name}'.");
                    if (cp.AddValue >= 0) cp.AddValue = Need(cp.AddValue, Layout.Nhwc);
                }
                Need(n, 0, Layout.Nhwc);
                if (n.Op is ExecOp.ResizeBilinear or ExecOp.ResizeNearest)
                    for (int k = 1; k < n.In.Length; k++) Need(n, k, Layout.Nhwc);   // fused addend
                SetOut(n, Layout.Nhwc);
                break;
            }
            case ExecOp.ChannelGate:
                Need(n, 0, Layout.Nhwc);
                SetOut(n, Layout.Nhwc);
                break;
            case ExecOp.Attention:
                // q / k / v are read through their strides, whatever their layout
                foreach (int v in n.In) L(v);
                SetOut(n, ((AttentionP)n.P!).OutLayout ?? Layout.Identity(_values[n.Out[0]].Shape.Length));
                break;
            case ExecOp.Ext:
                n.Ext!.PlanLayout(this, n);
                break;
            case ExecOp.LayerNorm or ExecOp.Gemm or ExecOp.Softmax or ExecOp.ReduceMean:
            {
                int x = n.In[0];
                if (!LastInnermost(_values[x].Shape, L(x)) || _values[x].Shape.Length != _values[n.Out[0]].Shape.Length)
                    Need(n, 0, Layout.Identity(_values[x].Shape.Length));
                SetOut(n, _values[x].Shape.Length == _values[n.Out[0]].Shape.Length
                    ? L(n.In[0]) : Layout.Identity(_values[n.Out[0]].Shape.Length));
                break;
            }
            case ExecOp.MatMul:
            {
                Need(n, 0, Layout.Identity(_values[n.In[0]].Shape.Length));
                Need(n, 1, Layout.Identity(_values[n.In[1]].Shape.Length));
                SetOut(n, Layout.Identity(_values[n.Out[0]].Shape.Length));
                break;
            }
            case ExecOp.Elementwise:
            {
                int[] os = _values[n.Out[0]].Shape;
                int rank = os.Length;
                // a full-shape operand's layout; with several, a channels-last one (an NCHW graph input mixed
                // with conv features then gets one cached relayout instead of every feature map going back)
                int ri = Array.FindIndex(n.In, v => _values[v].IsActivation && _values[v].Shape.SequenceEqual(os)
                    && rank == 4 && Layout.Equivalent(os, L(v), Layout.Nhwc));
                if (ri < 0) ri = Array.FindIndex(n.In, v => _values[v].IsActivation && _values[v].Shape.SequenceEqual(os));
                int[] lo = ri >= 0 ? L(n.In[ri]) : Layout.Identity(rank);
                for (int k = 0; k < n.In.Length; k++)
                {
                    CValue cv = _values[n.In[k]];
                    if (cv.ElementCount == 1) continue;
                    // operand storage must match lo restricted to its own dims
                    int off = rank - cv.Shape.Length;
                    int[] tv = lo.Where(d => d >= off).Select(d => d - off).ToArray();
                    Need(n, k, tv);
                }
                SetOut(n, lo);
                break;
            }
            case ExecOp.Relu or ExecOp.Sigmoid or ExecOp.Erf or ExecOp.Gelu or ExecOp.Copy or ExecOp.HardSigmoid or ExecOp.Sqrt
                or ExecOp.HardSwish or ExecOp.Swish or ExecOp.LeakyRelu or ExecOp.Tanh or ExecOp.Clip:
                SetOut(n, L(n.In[0]));
                break;
            case ExecOp.Slice:
            {
                int[] lx = L(n.In[0]);
                int rank = lx.Length;
                var sp = (SliceP)n.P!;
                for (int i = 0; i < sp.Axes.Length; i++)
                    sp.Axes[i] = Array.IndexOf(lx, sp.Axes[i] < 0 ? sp.Axes[i] + rank : sp.Axes[i]);
                SetOut(n, lx);
                break;
            }
            case ExecOp.Concat:
            {
                // channels-last when any rank-4 input already is (same reasoning as elementwise)
                int[] lo = L(n.In[0]);
                foreach (int v in n.In)
                    if (_values[v].Shape.Length == 4 && Layout.Equivalent(_values[v].Shape, L(v), Layout.Nhwc)) { lo = Layout.Nhwc; break; }
                Need(n, 0, lo);
                for (int k = 1; k < n.In.Length; k++) Need(n, k, lo);
                var cp = (ConcatP)n.P!;
                cp.Axis = Array.IndexOf(lo, cp.Axis);
                SetOut(n, lo);
                break;
            }
            default:
                throw new NotSupportedException($"layout: no rule for {n.Op} '{n.Name}'.");
        }
    }
}
