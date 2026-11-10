using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Onnx;
using Xunit;

namespace Sdcb.SimdOnnx.Tests;

/// <summary><see cref="OnnxModelBuilder"/>: two networks and their glue as one model.</summary>
public class ComposeTests
{
    /// <summary>GC-Net → x / max(shadow, 1e-6) clamped to [0, 1] → concat with x → DR-Net.</summary>
    internal static OnnxModel Gcdr(OnnxModel gc, OnnxModel dr)
    {
        var b = new OnnxModelBuilder(gc.Opset);
        string x = b.Input("x", 1, 3, -1, -1);
        string s = b.Inline(gc, "gc/", x)[0];
        string eps = b.Constant("eps", [1e-6f]), zero = b.Constant("zero", [0f]), one = b.Constant("one", [1f]);
        string safe = b.Node("Clip", [s, eps, ""]);
        string q = b.Node("Div", [x, safe]);
        string c = b.Node("Clip", [q, zero, one]);
        string x6 = b.Node("Concat", [x, c], null, ("axis", 1));
        b.Output(b.Inline(dr, "dr/", x6)[0]);
        return b.Build();
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Composed_MatchesTwoSessionsAndGlue(bool nhwc)
    {
        OnnxModel gc = OnnxModel.Load(EngineTests.GcPath), dr = OnnxModel.Load(EngineTests.DrPath);
        float[] x = Golden.F32("pad", "input");
        int[] shape = Golden.ShapeOf("pad", "input");
        int hw = shape[2] * shape[3];
        var opt = new CompileOptions(nhwc);

        float[] want;
        using (var g = new InferenceSession(GraphCompiler.Compile(gc, shape, opt)))
        using (var d = new InferenceSession(GraphCompiler.Compile(dr, [1, 6, shape[2], shape[3]], opt)))
        {
            x.CopyTo(g.Input);
            g.Run();
            float[] s = g.Output.ToArray();
            Span<float> x6 = d.Input;
            x.CopyTo(x6);
            for (int i = 0; i < 3 * hw; i++) x6[3 * hw + i] = Math.Clamp(x[i] / Math.Max(s[i], 1e-6f), 0f, 1f);
            d.Run();
            want = d.Output.ToArray();
        }

        OnnxModel m = Gcdr(gc, dr);
        ExecPlan p = GraphCompiler.Compile(m, shape, opt);
        using var one = new InferenceSession(p);
        x.CopyTo(one.Input);
        one.Run();
        Assert.Equal(want, one.Output.ToArray());
    }

    [Fact]
    public void Composed_SharesWeightArrays()
    {
        OnnxModel gc = OnnxModel.Load(EngineTests.GcPath), dr = OnnxModel.Load(EngineTests.DrPath);
        OnnxModel m = Gcdr(gc, dr);
        foreach (var (src, prefix) in new[] { (gc, "gc/"), (dr, "dr/") })
            foreach (KeyValuePair<string, OnnxTensorData> kv in src.Initializers)
                if (kv.Value.DataType == OnnxTensorData.TypeFloat)
                    Assert.Same(kv.Value.AsFloats(), m.Initializers[prefix + kv.Key].AsFloats());
    }

    [Fact]
    public void Builder_RejectsBadGraphs()
    {
        OnnxModel gc = OnnxModel.Load(EngineTests.GcPath);
        var b = new OnnxModelBuilder(gc.Opset + 1);
        string x = b.Input("x", 1, 3, -1, -1);
        Assert.Throws<NotSupportedException>(() => b.Inline(gc, "gc/", x));
        Assert.Throws<ArgumentException>(() => b.Input("x", 1));
        Assert.Throws<ArgumentException>(() => b.Node("Relu", ["nope"]));
        Assert.Throws<InvalidOperationException>(() => b.Build());
    }
}
