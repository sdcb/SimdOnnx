using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Onnx;
using Xunit;

namespace Sdcb.SimdOnnx.Tests;

/// <summary>
/// ONNX attributes the compiler used to ignore: each is now either implemented (checked against an
/// equivalent explicit graph) or refused with <see cref="NotSupportedException"/>.
/// </summary>
public class AttributeTests
{
    private static float[] Run(byte[] model, int[] shape, float[] input, bool nhwc = false)
    {
        using var s = new InferenceSession(GraphCompiler.Compile(OnnxModel.Load(model), shape, new CompileOptions(nhwc)));
        input.CopyTo(s.Input);
        s.Run();
        return s.Output.ToArray();
    }

    [Theory]
    [InlineData("SAME_LOWER", new[] { 1, 1, 0, 0 }, false)]
    [InlineData("SAME_UPPER", new[] { 0, 0, 1, 1 }, false)]
    [InlineData("VALID", new[] { 0, 0, 0, 0 }, false)]
    [InlineData("SAME_LOWER", new[] { 1, 1, 0, 0 }, true)]
    public void Conv_AutoPad_EqualsExplicitPads(string autoPad, int[] pads, bool nhwc)
    {
        var rng = new Random(3);
        float[] w = rng.Array(4 * 3 * 2 * 2), x = rng.Array(3 * 7 * 6);
        byte[] Model(params (string, object)[] attrs) => new OnnxBuilder().Input("x", 1, 3, 7, 6)
            .Init("W", w, 4, 3, 2, 2)
            .Node("Conv", ["x", "W"], ["y"], [("kernel_shape", new[] { 2, 2 }), .. attrs])
            .Output("y").Build();
        float[] auto = Run(Model(("auto_pad", autoPad)), [1, 3, 7, 6], x, nhwc);
        float[] expl = Run(Model(("pads", pads)), [1, 3, 7, 6], x, nhwc);
        Assert.Equal(expl, auto);
    }

    [Fact]
    public void MaxPool_SameLower_EqualsExplicitPads()
    {
        float[] x = new Random(5).Array(2 * 5 * 5);
        byte[] Model(params (string, object)[] attrs) => new OnnxBuilder().Input("x", 1, 2, 5, 5)
            .Node("MaxPool", ["x"], ["y"], [("kernel_shape", new[] { 2, 2 }), ("strides", new[] { 1, 1 }), .. attrs])
            .Output("y").Build();
        Assert.Equal(Run(Model(("pads", new[] { 1, 1, 0, 0 })), [1, 2, 5, 5], x),
            Run(Model(("auto_pad", "SAME_LOWER")), [1, 2, 5, 5], x));
    }

    [Fact]
    public void Gemm_Beta_ScalesBias()
    {
        var rng = new Random(7);
        float[] b = rng.Array(6 * 4), c = rng.Array(4), x = rng.Array(3 * 6);
        byte[] Model(float[] bias, params (string, object)[] attrs) => new OnnxBuilder().Input("x", 3, 6)
            .Init("B", b, 6, 4).Init("C", bias, 4)
            .Node("Gemm", ["x", "B", "C"], ["y"], attrs).Output("y").Build();
        float[] scaled = Run(Model(c, ("beta", 0.5f)), [3, 6], x);
        float[] manual = Run(Model(c.Select(v => v * 0.5f).ToArray()), [3, 6], x);
        Assert.Equal(manual, scaled);
    }

    [Fact]
    public void Gemm_TransA_IsRefused()
    {
        byte[] m = new OnnxBuilder().Input("x", 6, 3).Init("B", new float[24], 6, 4)
            .Node("Gemm", ["x", "B"], ["y"], ("transA", 1)).Output("y").Build();
        var ex = Assert.Throws<NotSupportedException>(() => GraphCompiler.Compile(OnnxModel.Load(m), [6, 3]));
        Assert.Contains("transA", ex.Message);
    }

    [Fact]
    public void Pad_Opset2Attributes_EqualOpset13Inputs()
    {
        float[] x = new Random(9).Array(1 * 2 * 3 * 4);
        int[] pads = [0, 0, 1, 2, 0, 0, 2, 1];
        byte[] old = new OnnxBuilder().Opset(2).Input("x", 1, 2, 3, 4)
            .Node("Pad", ["x"], ["y"], ("pads", pads), ("value", 1.5f)).Output("y").Build();
        byte[] cur = new OnnxBuilder().Input("x", 1, 2, 3, 4)
            .InitI64("p", pads.Select(v => (long)v).ToArray(), 8).Init("v", [1.5f])
            .Node("Pad", ["x", "p", "v"], ["y"]).Output("y").Build();
        Assert.Equal(Run(cur, [1, 2, 3, 4], x), Run(old, [1, 2, 3, 4], x));
    }

    [Fact]
    public void Split_Opset11SplitAttribute_IsHonoured()
    {
        float[] x = Enumerable.Range(0, 2 * 4).Select(i => (float)i).ToArray();
        byte[] m = new OnnxBuilder().Opset(11).Input("x", 2, 4)
            .Node("Split", ["x"], ["a", "b"], ("axis", 1), ("split", new[] { 1, 3 }))
            .Node("Relu", ["b"], ["y"]).Output("y").Build();
        Assert.Equal([1f, 2, 3, 5, 6, 7], Run(m, [2, 4], x));
    }

    [Fact]
    public void Softmax_Opset11_DefaultsToAxisOne()
    {
        // rank 2: axis 1 is the last axis, so it runs; rank 3: the flatten-at-axis-1 form is refused
        byte[] Model(params int[] shape) => new OnnxBuilder().Opset(11).Input("x", shape)
            .Node("Softmax", ["x"], ["y"]).Output("y").Build();
        float[] y = Run(Model(2, 3), [2, 3], [0, 0, 0, 1, 1, 1]);
        Assert.All(y, v => Assert.Equal(1f / 3, v, 6));
        Assert.Throws<NotSupportedException>(() => GraphCompiler.Compile(OnnxModel.Load(Model(2, 3, 4)), [2, 3, 4]));
    }

    [Fact]
    public void Resize_Linear_AlignCorners_IsRefused()
    {
        byte[] m = new OnnxBuilder().Input("x", 1, 1, 4, 4).Init("roi", [], 0).Init("s", [1f, 1f, 2f, 2f], 4)
            .Node("Resize", ["x", "roi", "s"], ["y"], ("mode", "linear"), ("coordinate_transformation_mode", "align_corners"))
            .Output("y").Build();
        var ex = Assert.Throws<NotSupportedException>(() => GraphCompiler.Compile(OnnxModel.Load(m), [1, 1, 4, 4]));
        Assert.Contains("align_corners", ex.Message);
    }

    // ---- bugs found by the ONNX node tests ----

    [Fact]
    public void MaxPool_DefaultStrides_AreOne()
    {
        // ONNX: strides default to 1 along each axis (they used to default to the kernel)
        float[] x = Enumerable.Range(0, 9).Select(i => (float)i).ToArray();
        byte[] m = new OnnxBuilder().Input("x", 1, 1, 3, 3)
            .Node("MaxPool", ["x"], ["y"], ("kernel_shape", new[] { 2, 2 })).Output("y").Build();
        Assert.Equal([4f, 5, 7, 8], Run(m, [1, 1, 3, 3], x));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Slice_NegativeAxis(bool nhwc)
    {
        float[] x = Enumerable.Range(0, 2 * 3 * 4).Select(i => (float)i).ToArray();
        byte[] m = new OnnxBuilder().Input("x", 2, 3, 4)
            .InitI64("s", [1], 1).InitI64("e", [2], 1).InitI64("a", [-1], 1)
            .Node("Slice", ["x", "s", "e", "a"], ["y"]).Output("y").Build();
        Assert.Equal([1f, 5, 9, 13, 17, 21], Run(m, [2, 3, 4], x, nhwc));
    }

    [Fact]
    public void Split_NumOutputs_Uneven_UsesCeilChunks()
    {
        // opset 18: 5 split 2 ways is [3, 2], not [2, 3]
        float[] x = [1, 2, 3, 4, 5];
        byte[] m = new OnnxBuilder().Opset(18).Input("x", 5)
            .Node("Split", ["x"], ["a", "b"], ("axis", 0), ("num_outputs", 2)).Output("a").Build();
        Assert.Equal([1f, 2, 3], Run(m, [5], x));
    }

    [Fact]
    public void Gemm_FullMatrixBias_Broadcasts()
    {
        var rng = new Random(11);
        float[] b = rng.Array(3 * 2), c = rng.Array(4 * 2), x = rng.Array(4 * 3);
        byte[] m = new OnnxBuilder().Input("x", 4, 3).Init("B", b, 3, 2).Init("C", c, 4, 2)
            .Node("Gemm", ["x", "B", "C"], ["y"]).Output("y").Build();
        float[] y = Run(m, [4, 3], x);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 2; j++)
            {
                float want = c[i * 2 + j];
                for (int k = 0; k < 3; k++) want += x[i * 3 + k] * b[k * 2 + j];
                Assert.Equal(want, y[i * 2 + j], 5);
            }
    }

    [Fact]
    public void Conv_RuntimeWeights_NhwcIsRefused()
    {
        // weights fed as a graph input: the NCHW kernel reads them at run time, the NHWC kernel needs
        // them packed up front (it used to pack an empty span and overrun it)
        byte[] m = new OnnxBuilder().Input("x", 1, 3, 5, 4).Input("W", 2, 3, 3, 3)
            .Node("Conv", ["x", "W"], ["y"], ("kernel_shape", new[] { 3, 3 })).Output("y").Build();
        ExecPlan plan = GraphCompiler.Compile(OnnxModel.Load(m), [1, 3, 5, 4], new CompileOptions(Nhwc: true));
        var ex = Assert.Throws<NotSupportedException>(() => new InferenceSession(plan));
        Assert.Contains("run-time weights", ex.Message);
    }
}
