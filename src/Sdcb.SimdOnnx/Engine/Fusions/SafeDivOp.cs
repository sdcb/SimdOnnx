#if NET
using Sdcb.SimdOnnx.Backends.Metal;
using Sdcb.SimdOnnx.Backends.Vulkan;
#endif
using Sdcb.SimdOnnx.Kernels;

namespace Sdcb.SimdOnnx.Engine;

internal sealed class SafeDivP
{
    public float Eps, Lo, Hi;
    public override string ToString() => $"eps={Eps} lo={Lo} hi={Hi}";
}

/// <summary>out = clamp(a / max(b, eps), lo, hi), elementwise over same-shape operands. In = [a, b].
/// GPU shaders compute it in fp32, so the floor survives fp16 storage (eps may be subnormal there).</summary>
internal sealed class SafeDivOp : ExtOp
{
    public static readonly SafeDivOp Instance = new();
    public override string Name => "SafeDiv";

    public override void RunCpu(InferenceSession s, ExecNode n)
    {
        var p = (SafeDivP)n.P!;
        FusedKernels.SafeDiv(s.GetIn(n.In[0]), s.GetIn(n.In[1]), s.GetSpan(n.Out[0]), p.Eps, p.Lo, p.Hi);
    }
#if NET

    public override IEnumerable<VkDispatch> EmitVulkan(VkGraphModel g, ExecNode n)
    {
        var p = (SafeDivP)n.P!;
        uint count = (uint)g.Plan.Values[n.Out[0]].ElementCount;
        yield return g.Dispatch(n, "safediv", [g.Bind(n.In[0]), g.Bind(n.In[1]), g.Bind(n.Out[0])],
            VkGraphModel.PcOfF([count], p.Eps, p.Lo, p.Hi), VkGraphModel.G(count));
    }

    public override IEnumerable<MtlDispatch> EmitMetal(MtlGraphModel g, ExecNode n)
    {
        var p = (SafeDivP)n.P!;
        uint count = (uint)g.Plan.Values[n.Out[0]].ElementCount;
        yield return g.Dispatch(n, "safediv", [g.Bind(n.In[0]), g.Bind(n.In[1]), g.Bind(n.Out[0])],
            MtlGraphModel.PcOfF([count], p.Eps, p.Lo, p.Hi), MtlGraphModel.G(count));
    }
#endif
}

/// <summary>
/// Clip(Div(a, Clip(b, min = eps &gt; 0)), lo, hi) over same-shape operands becomes one
/// <see cref="SafeDivOp"/>: one pass instead of three, and no rounding of max(b, eps) to the
/// storage type in between. Rank-4 operands only in NHWC plans (they get channels-last).
/// </summary>
internal sealed class FuseSafeDivPass : GraphPass
{
    public static readonly FuseSafeDivPass Instance = new();
    public override PassStage Stage => PassStage.Logical;

    public override void Run(PassContext ctx)
    {
        List<ExecNode> nodes = ctx.Nodes;
        List<CValue> values = ctx.Values;
        for (int i = 0; i < nodes.Count; i++)
        {
            ExecNode outer = nodes[i];
            if (outer.Op != ExecOp.Clip || outer.P is not ClipP oc || outer.In.Length != 1) continue;
            int q = outer.In[0], di = ctx.Producer(q);
            if (di < 0 || ctx.Uses(q) != 1) continue;
            ExecNode div = nodes[di];
            if (div.Op != ExecOp.Elementwise || div.In.Length != 2 || div.P is not EwP { Op: EwOp.Div, Act: ActivationKind.None }) continue;
            int a = div.In[0], e = div.In[1], ci = ctx.Producer(e);
            if (ci < 0 || ctx.Uses(e) != 1) continue;
            ExecNode floor = nodes[ci];
            if (floor.Op != ExecOp.Clip || floor.P is not ClipP fc || floor.In.Length != 1
                || !(fc.Min > 0) || !float.IsPositiveInfinity(fc.Max)) continue;
            int b = floor.In[0];
            if (!ctx.SameShape(a, q) || !ctx.SameShape(b, q) || !ctx.SameShape(outer.Out[0], q)) continue;
            if (values[a].Buf < 0 || values[b].Buf < 0) continue;   // constants: not a runtime op
            if (ctx.Options.Nhwc && values[q].Shape.Length != 4) continue;
            var fused = new ExecNode
            {
                Op = ExecOp.Ext,
                Ext = SafeDivOp.Instance,
                In = [a, b],
                Out = outer.Out,
                P = new SafeDivP { Eps = fc.Min, Lo = oc.Min, Hi = oc.Max },
                SrcNode = outer.SrcNode,
                Name = outer.Name + ".safediv",
            };
            nodes[i] = fused;
            nodes.Remove(div);
            nodes.Remove(floor);
            i = -1;   // indices shifted; rescan
        }
    }
}
