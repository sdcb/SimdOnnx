using Sdcb.SimdOnnx.Onnx;

namespace Sdcb.SimdOnnx.Engine;

/// <summary>
/// The tail of a graph that ends in a last-axis Softmax: the logits tensor, and when the logits come
/// from a constant linear layer (MatMul / Gemm, optional bias Add), that layer. Lets a caller run the
/// graph up to the logits or up to the layer's input (<see cref="CpuModel.WithOutput"/>) and compute
/// the rest itself, e.g. a fused projection + argmax for CTC decoding.
/// </summary>
public sealed class SoftmaxHead
{
    internal SoftmaxHead(string logitsName, LinearProjection? projection)
    {
        LogitsName = logitsName;
        Projection = projection;
    }

    /// <summary>The Softmax input (empty when the compiler gave it no name).</summary>
    public string LogitsName { get; }

    /// <summary>The constant linear layer producing the logits, or null.</summary>
    public LinearProjection? Projection { get; }
}

/// <summary>logits = activation · Weights + Bias, with Weights stored row-major as [Inner, Columns].</summary>
public sealed class LinearProjection
{
    internal LinearProjection(string activationName, float[] weights, float[]? bias, int inner, int columns)
    {
        ActivationName = activationName;
        Weights = weights;
        Bias = bias;
        Inner = inner;
        Columns = columns;
    }

    /// <summary>The layer's input tensor (its last axis is <see cref="Inner"/>).</summary>
    public string ActivationName { get; }
    /// <summary>[Inner, Columns], row-major.</summary>
    public float[] Weights { get; }
    /// <summary>[Columns], or null.</summary>
    public float[]? Bias { get; }
    public int Inner { get; }
    public int Columns { get; }
}

/// <summary>Finds model heads on the compiled plan, so they are seen after the compiler's rewrites
/// (constant folding, fusions, alias views).</summary>
public static class ModelHeads
{
    /// <summary>The Softmax tail of <paramref name="model"/> compiled for <paramref name="probeShape"/>,
    /// or null when the graph output is not produced by a Softmax.</summary>
    public static SoftmaxHead? FindSoftmaxHead(OnnxModel model, ReadOnlySpan<int> probeShape, CompileOptions? options = null)
    {
        ExecPlan plan = GraphCompiler.Compile(model, probeShape, options ?? CompileOptions.Default);
        ExecNode? softmax = Producer(plan, plan.OutputValue);
        if (softmax is not { Op: ExecOp.Softmax }) return null;
        return new SoftmaxHead(plan.Values[softmax.In[0]].Name, TryLinear(plan, softmax));
    }

    private static ExecNode? Producer(ExecPlan plan, int value)
    {
        // zero-copy views (reshape / squeeze / fused tails) have no producer node of their own
        for (int hops = 0; value >= 0 && hops < 64; hops++, value = plan.Values[value].AliasOf)
            foreach (ExecNode n in plan.Nodes)
                if (n.Out.Length == 1 && n.Out[0] == value) return n;
        return null;
    }

    private static LinearProjection? TryLinear(ExecPlan plan, ExecNode softmax)
    {
        ExecNode? p = Producer(plan, softmax.In[0]);
        float[]? bias = null;
        if (p is { Op: ExecOp.Elementwise, P: EwP { Op: EwOp.Add, Act: ActivationKind.None } } && p.In.Length == 2)
        {
            int c = plan.Values[p.In[0]].Floats is not null ? 0 : plan.Values[p.In[1]].Floats is not null ? 1 : -1;
            if (c < 0) return null;
            bias = plan.Values[p.In[c]].Floats;
            p = Producer(plan, p.In[1 - c]);
            if (p is { P: GemmP { BiasValue: >= 0 } }) return null;
        }
        if (p is not { Op: ExecOp.Gemm, P: GemmP g } || g.Act != ActivationKind.None) return null;
        if (g.BiasValue >= 0)
        {
            if (bias is not null) return null;
            bias = plan.Values[g.BiasValue].Floats;
            if (bias is null) return null;
        }
        CValue a = plan.Values[p.In[0]], b = plan.Values[p.In[1]];
        if (b.Floats is not { } w || b.Shape.Length != 2 || a.Name.Length == 0 || a.Shape.Length < 2) return null;
        int inner = g.TransB ? b.Shape[1] : b.Shape[0], columns = g.TransB ? b.Shape[0] : b.Shape[1];
        if (a.Shape[^1] != inner || (bias is not null && bias.Length != columns)) return null;
        if (g.TransB)
        {
            float[] t = new float[w.Length];
            for (int n = 0; n < columns; n++)
                for (int k = 0; k < inner; k++) t[k * columns + n] = w[n * inner + k];
            w = t;
        }
        return new LinearProjection(a.Name, w, bias, inner, columns);
    }
}
