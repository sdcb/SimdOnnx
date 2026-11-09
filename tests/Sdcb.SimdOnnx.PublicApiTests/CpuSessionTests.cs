using Sdcb.SimdOnnx.Backends.Metal;
using Sdcb.SimdOnnx.Backends.Vulkan;
using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Onnx;
using Sdcb.SimdOnnx.Tests;

namespace Sdcb.SimdOnnx.PublicApiTests;

/// <summary><see cref="CpuModel"/> / <see cref="CpuSession"/>, <see cref="ModelHeads"/>, <see cref="EngineProfiler"/>
/// and GPU device info, used the way a consumer would (public surface only).</summary>
public class CpuSessionTests
{
    private static readonly Random s_rng = new(17);

    // x -> 1x1 conv (8->16) "h" -> relu "r" -> 1x1 conv (16->4) "y"
    private static readonly float[] W1 = Rand(16 * 8), W2 = Rand(4 * 16);
    private static OnnxModel ConvModel() => OnnxModel.Load(new OnnxBuilder().Input("x", 1, 8, 6, 5)
        .Init("W1", W1, 16, 8, 1, 1).Init("W2", W2, 4, 16, 1, 1)
        .Node("Conv", ["x", "W1"], ["h"], ("kernel_shape", new[] { 1, 1 }))
        .Node("Relu", ["h"], ["r"])
        .Node("Conv", ["r", "W2"], ["y"], ("kernel_shape", new[] { 1, 1 }))
        .Output("y").Build());

    private static float[] Rand(int n)
    {
        lock (s_rng) return Enumerable.Range(0, n).Select(_ => (float)s_rng.NextDouble() * 2f - 1f).ToArray();
    }

    /// <summary>Reference: a separately compiled single-shape model.</summary>
    private static float[] Reference(OnnxModel m, int[] shape, float[] input, CompileOptions opt)
    {
        using InferenceSession s = CompiledModel.Compile(m, shape, opt).CreateSession();
        input.CopyTo(s.Input);
        s.Run();
        return s.Output.ToArray();
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Variants_RunOnTheInputWrittenOnce(bool nhwc)
    {
        OnnxModel m = ConvModel();
        var opt = new CompileOptions(nhwc);
        var full = new CpuModel(m, opt);
        CpuModel hidden = full.WithOutput("h");
        Assert.Same(hidden, full.WithOutput("h"));

        int[] shape = [1, 8, 6, 5];
        float[] x = Rand(8 * 6 * 5);
        using var s = new CpuSession(full);
        s.Reshape(shape);
        x.CopyTo(s.Input);
        float[] h = s.Run(hidden).ToArray();   // the variant first: the arena grows, the input must survive
        float[] y = s.Run().ToArray();
        float[] h2 = s.Run(hidden).ToArray();

        Assert.Equal(Reference(m, shape, x, opt with { OutputName = "h" }), h);
        Assert.Equal(Reference(m, shape, x, opt), y);
        Assert.Equal(h, h2);
        Assert.Equal([1, 16, 6, 5], s.OutputShape(hidden));
        Assert.Equal([1, 4, 6, 5], full.OutputShape(shape));
        Assert.True(s.ArenaBytes > x.Length * sizeof(float), "the plan arena should exceed the input");
    }

    [Fact]
    public void Reshape_WithPrepare_SizesTheArenaOnce()
    {
        OnnxModel m = ConvModel();
        var full = new CpuModel(m, new CompileOptions(Nhwc: true));
        CpuModel hidden = full.WithOutput("h");
        float[] x = Rand(8 * 6 * 5);
        using var s = new CpuSession(full);
        s.Reshape([1, 8, 6, 5], hidden);
        long arena = s.ArenaBytes;
        Assert.True(arena > x.Length * sizeof(float));
        x.CopyTo(s.Input);
        float[] h = s.Run(hidden).ToArray();
        Assert.Equal(arena, s.ArenaBytes);
        Assert.Equal(Reference(m, [1, 8, 6, 5], x, full.Options with { OutputName = "h" }), h);
    }

    [Fact]
    public void Reshape_AcrossShapes_MatchesPerShapeModels()
    {
        OnnxModel m = ConvModel();
        var opt = new CompileOptions(Nhwc: true);
        using var s = new CpuSession(new CpuModel(m, opt));
        foreach (int[] shape in new[] { new[] { 1, 8, 6, 5 }, [1, 8, 12, 9], [1, 8, 3, 4], [1, 8, 12, 9] })
        {
            float[] x = Rand(8 * shape[2] * shape[3]);
            s.Reshape(shape);
            x.CopyTo(s.Input);
            Assert.Equal(Reference(m, shape, x, opt), s.Run().ToArray());
        }
    }

    [Fact]
    public void Run_RejectsAVariantOfAnotherModel()
    {
        using var s = new CpuSession(new CpuModel(ConvModel()));
        s.Reshape([1, 8, 6, 5]);
        CpuModel other = new CpuModel(ConvModel()).WithOutput("h");
        Assert.Throws<ArgumentException>(() => s.Run(other));
        Assert.Throws<InvalidOperationException>(() => new CpuSession(new CpuModel(ConvModel())).Run());
    }

    // ---- ModelHeads ----

    [Fact]
    public void SoftmaxHead_MatMulAddSoftmax_FindsTheLinearLayer()
    {
        float[] w = Rand(8 * 6), b = Rand(6);
        OnnxModel m = OnnxModel.Load(new OnnxBuilder().Input("x", 2, 5, 8).Init("W", w, 8, 6).Init("B", b, 6)
            .Node("Relu", ["x"], ["a"]).Node("MatMul", ["a", "W"], ["mm"]).Node("Add", ["mm", "B"], ["logits"])
            .Node("Softmax", ["logits"], ["y"], ("axis", -1)).Output("y").Build());
        SoftmaxHead head = ModelHeads.FindSoftmaxHead(m, [2, 5, 8])!;
        Assert.NotNull(head);
        LinearProjection p = head.Projection!;
        Assert.Equal((8, 6), (p.Inner, p.Columns));
        Assert.Equal(w, p.Weights);
        Assert.Equal(b, p.Bias);

        // the caller runs up to the activation and applies the layer itself
        var model = new CpuModel(m);
        float[] x = Rand(2 * 5 * 8);
        using var s = new CpuSession(model);
        s.Reshape([2, 5, 8]);
        x.CopyTo(s.Input);
        float[] a = s.Run(model.WithOutput(p.ActivationName)).ToArray();
        float[] logits = s.Run(model.WithOutput(head.LogitsName)).ToArray();
        for (int r = 0; r < 10; r++)
            for (int c = 0; c < 6; c++)
            {
                float acc = p.Bias![c];
                for (int k = 0; k < 8; k++) acc += a[r * 8 + k] * p.Weights[k * 6 + c];
                Assert.Equal(logits[r * 6 + c], acc, 4);
            }
    }

    [Fact]
    public void SoftmaxHead_GemmTransB_WeightsAreInnerByColumns()
    {
        float[] bt = Rand(6 * 8), c = Rand(6);   // B stored [N, K]
        OnnxModel m = OnnxModel.Load(new OnnxBuilder().Input("x", 4, 8).Init("B", bt, 6, 8).Init("C", c, 6)
            .Node("Relu", ["x"], ["a"]).Node("Gemm", ["a", "B", "C"], ["logits"], ("transB", 1))
            .Node("Softmax", ["logits"], ["y"], ("axis", 1)).Output("y").Build());
        LinearProjection p = ModelHeads.FindSoftmaxHead(m, [4, 8])!.Projection!;
        for (int k = 0; k < 8; k++)
            for (int n = 0; n < 6; n++) Assert.Equal(bt[n * 8 + k], p.Weights[k * 6 + n]);
        Assert.Equal(c, p.Bias);
    }

    [Fact]
    public void SoftmaxHead_NoLinearLayer_OrNoSoftmax()
    {
        OnnxModel plain = OnnxModel.Load(new OnnxBuilder().Input("x", 4, 8)
            .Node("Relu", ["x"], ["a"]).Node("Softmax", ["a"], ["y"], ("axis", -1)).Output("y").Build());
        SoftmaxHead head = ModelHeads.FindSoftmaxHead(plain, [4, 8])!;
        Assert.Null(head.Projection);
        Assert.Equal("a", head.LogitsName);

        OnnxModel none = OnnxModel.Load(new OnnxBuilder().Input("x", 4, 8)
            .Node("Relu", ["x"], ["y"]).Output("y").Build());
        Assert.Null(ModelHeads.FindSoftmaxHead(none, [4, 8]));
    }

    // ---- profiler and device info ----

    [Fact]
    public void Profiler_CountsPlanOperatorsAndRecordedOnes()
    {
        bool was = EngineProfiler.Enabled;
        try
        {
            EngineProfiler.Enabled = true;
            using var s = new CpuSession(new CpuModel(ConvModel()));
            s.Reshape([1, 8, 6, 5]);
            s.Run();
            EngineProfiler.Record("ConsumerHead", 1234);
            Assert.True(EngineProfiler.Operators().TryGetValue("Conv", out var conv) && conv.Calls >= 2);
            Assert.True(EngineProfiler.Operators().ContainsKey("ConsumerHead"));
            Assert.NotEmpty(EngineProfiler.ConvClasses());
        }
        finally { EngineProfiler.Enabled = was; }
    }

    [SkippableFact]
    public void GpuDeviceInfo_IsFilled()
    {
        bool vk = VkBackend.TryGetDefault(out VkBackend? v), mtl = MetalBackend.TryGetDefault(out MetalBackend? mb);
        Skip.IfNot(vk || mtl, "no Vulkan or Metal device");
        var info = vk ? v!.Device : mb!.Device;
        Assert.False(string.IsNullOrEmpty(info.Name));
        Assert.InRange(info.SubgroupSizeMin, 1, info.SubgroupSizeMax);
        if (vk) Assert.False(v!.IsLost);
    }
}
