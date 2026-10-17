using Sdcb.SimdOnnx.Backends.Metal;
using Sdcb.SimdOnnx.Backends.Vulkan;
using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Onnx;

namespace Sdcb.SimdOnnx.Tests;

/// <summary>
/// Whole-graph GPU vs CPU checks of the ops and fused epilogues the OCR / enhancement models need
/// beyond plain conv (AveragePool, Softmax, batched MatMul, gated hardswish / swish after conv,
/// depthwise and Gemm), on every GPU backend present: Vulkan here, Metal on the macOS runners.
/// </summary>
public class GpuOpTests
{
    public static TheoryData<string> Backends() => new() { "vulkan", "metal" };

    [SkippableTheory]
    [MemberData(nameof(Backends))]
    public void AvgPool_MatchesCpu(string backend)
    {
        // c % 4 == 0 (vec4 kernel) and odd c, batch 2, padding with / without count_include_pad
        foreach ((int c, int k, int s, int pad, int inc) in new[] { (8, 2, 2, 0, 0), (6, 3, 1, 1, 0), (12, 3, 2, 1, 1), (5, 2, 1, 0, 0) })
        {
            byte[] m = new OnnxBuilder().Input("x", 2, c, 9, 14)
                .Node("AveragePool", ["x"], ["y"], ("kernel_shape", new[] { k, k }), ("strides", new[] { s, s }),
                    ("pads", new[] { pad, pad, pad, pad }), ("count_include_pad", inc))
                .Output("y").Build();
            Check(backend, OnnxModel.Load(m), [2, c, 9, 14], $"avgpool c{c} k{k} s{s} p{pad} inc{inc}");
        }
    }

    [SkippableTheory]
    [MemberData(nameof(Backends))]
    public void SoftmaxMatMul_MatchesCpu(string backend)
    {
        // attention-shaped: softmax(x · W) · V with W broadcast over the batch, V batched
        var rng = new Random(5);
        byte[] m = new OnnxBuilder().Input("x", 2, 3, 10, 8)
            .Init("W", Rand(rng, 8 * 37, 0.5f), 8, 37)
            .Init("V", Rand(rng, 2 * 3 * 37 * 12, 0.5f), 2, 3, 37, 12)
            .Node("MatMul", ["x", "W"], ["s"])
            .Node("Softmax", ["s"], ["a"], ("axis", -1))
            .Node("MatMul", ["a", "V"], ["y"])
            .Output("y").Build();
        Check(backend, OnnxModel.Load(m), [2, 3, 10, 8], "softmax / matmul");
    }

    [SkippableTheory]
    [MemberData(nameof(Backends))]
    public void GatedActs_MatchCpu(string backend)
    {
        foreach (bool hard in new[] { true, false })
        {
            var rng = new Random(hard ? 11 : 12);
            int c = 16, oc = 24;
            var b = new OnnxBuilder().Input("x", 1, c, 12, 18)
                .Init("W1", Rand(rng, oc * c * 9, 0.3f), oc, c, 3, 3).Init("B1", Rand(rng, oc, 0.5f), oc)
                .Init("W2", Rand(rng, oc * 9, 0.3f), oc, 1, 3, 3).Init("B2", Rand(rng, oc, 0.5f), oc)
                .Init("W3", Rand(rng, oc * 32, 0.3f), oc, 32).Init("B3", Rand(rng, 32, 0.5f), 32)
                .Node("Conv", ["x", "W1", "B1"], ["c1"], ("kernel_shape", new[] { 3, 3 }), ("pads", new[] { 1, 1, 1, 1 }));
            Act(b, "c1", "a1", hard);
            b.Node("Conv", ["a1", "W2", "B2"], ["c2"], ("kernel_shape", new[] { 3, 3 }), ("pads", new[] { 1, 1, 1, 1 }), ("group", oc));
            Act(b, "c2", "a2", hard);
            b.Node("Transpose", ["a2"], ["t"], ("perm", new[] { 0, 2, 3, 1 }))
                .Node("MatMul", ["t", "W3"], ["g"])
                .Node("Add", ["g", "B3"], ["gb"]);
            Act(b, "gb", "y", hard);
            OnnxModel m = OnnxModel.Load(b.Output("y").Build());
            ActivationKind want = hard ? ActivationKind.HardSwish : ActivationKind.Swish;
            // CPU plans fold the gates into the conv / gemm epilogues
            ExecPlan plan = GraphCompiler.Compile(m, [1, c, 12, 18]);
            Assert.Equal(2, plan.Nodes.Count(x => x.P is ConvP cp && cp.Act == want));
            Assert.Contains(plan.Nodes, x => x.P is GemmP gp && gp.Act == want);
            Check(backend, m, [1, c, 12, 18], hard ? "hardswish" : "swish");
        }
    }

    [SkippableTheory]
    [MemberData(nameof(Backends))]
    public void OneByOneSpatial_MatchesCpu(string backend)
    {
        // a 1x1-spatial value may keep an NCHW-ordered physical shape: spatial ops must read
        // [n,h,w,c] from the logical shape (the depthwise conv saw ic = 1 on Metal)
        var rng = new Random(3);
        int c = 24;
        byte[] m = new OnnxBuilder().Input("x", 1, c, 6, 10)
            .Init("Wd", Rand(rng, c, 0.5f), c, 1, 1, 1).Init("Bd", Rand(rng, c, 0.5f), c)
            .Init("Wp", Rand(rng, 16 * c, 0.3f), 16, c, 1, 1)
            .Node("GlobalAveragePool", ["x"], ["g"])
            .Node("Conv", ["g", "Wd", "Bd"], ["d"], ("kernel_shape", new[] { 1, 1 }), ("group", c))
            .Node("Relu", ["d"], ["r"])
            .Node("Conv", ["r", "Wp"], ["y"], ("kernel_shape", new[] { 1, 1 }))
            .Output("y").Build();
        Check(backend, OnnxModel.Load(m), [1, c, 6, 10], "1x1 spatial");
    }

    [SkippableTheory]
    [MemberData(nameof(Backends))]
    public void SqueezeExcite_FusesAndMatchesCpu(string backend)
    {
        // PP-LCNet SE (hard gate, x * s) and RSE-FPN (x + x * s), batch 2, a no-op reshape between the
        // two FC convs as paddle2onnx exports them; odd channel count for the scalar paths
        foreach ((int c, bool hard, bool residual) in new[] { (24, true, true), (20, false, false), (13, true, false) })
        {
            var rng = new Random(c);
            int r = Math.Max(c / 4, 2);
            var b = new OnnxBuilder().Input("x", 2, c, 10, 14)
                .Init("W0", Rand(rng, c * c * 9, 0.3f), c, c, 3, 3)
                .Init("W1", Rand(rng, r * c, 0.5f), r, c, 1, 1).Init("B1", Rand(rng, r, 0.5f), r)
                .Init("W2", Rand(rng, c * r, 0.5f), c, r, 1, 1).Init("B2", Rand(rng, c, 0.5f), c)
                .InitI64("shp", [2, r, 1, 1], 4)
                .Node("Conv", ["x", "W0"], ["x0"], ("kernel_shape", new[] { 3, 3 }), ("pads", new[] { 1, 1, 1, 1 }))
                .Node("GlobalAveragePool", ["x0"], ["g"])
                .Node("Conv", ["g", "W1", "B1"], ["h0"], ("kernel_shape", new[] { 1, 1 }))
                .Node("Relu", ["h0"], ["h1"])
                .Node("Reshape", ["h1", "shp"], ["h"])
                .Node("Conv", ["h", "W2", "B2"], ["z"], ("kernel_shape", new[] { 1, 1 }));
            if (hard) b.Node("HardSigmoid", ["z"], ["s"], ("alpha", 0.2f), ("beta", 0.5f));
            else b.Node("Sigmoid", ["z"], ["s"]);
            b.Node("Mul", ["x0", "s"], [residual ? "m" : "y"]);
            if (residual) b.Node("Add", ["x0", "m"], ["y"]);
            OnnxModel m = OnnxModel.Load(b.Output("y").Build());
            int[] shape = [2, c, 10, 14];
            ExecPlan plan = GraphCompiler.Compile(m, shape);
            Assert.Single(plan.Nodes, x => x.Op == ExecOp.ChannelGate);
            Assert.DoesNotContain(plan.Nodes, x => x.Op == ExecOp.GlobalAvgPool);
            if (residual) Assert.DoesNotContain(plan.Nodes, x => x.P is EwP { Op: EwOp.Add });
            CheckCpuFusion(m, shape, $"se c{c}");
            Check(backend, m, shape, $"se c{c} hard={hard} residual={residual}");
        }
    }

    [SkippableTheory]
    [MemberData(nameof(Backends))]
    public void NearestUpsampleAdd_FusesAndMatchesCpu(string backend)
    {
        // FPN top-down merge: up2(low) + [relu](lateral); c % 4 == 0 (vec4 kernels) and odd c
        foreach ((int c, bool relu) in new[] { (16, false), (6, true) })
        {
            var rng = new Random(c + 1);
            var b = new OnnxBuilder().Input("x", 1, c, 12, 10)
                .Init("W", Rand(rng, c * c, 0.5f), c, c, 1, 1)
                .Init("sc", [1f, 1f, 2f, 2f], 4)
                .Node("MaxPool", ["x"], ["p"], ("kernel_shape", new[] { 2, 2 }), ("strides", new[] { 2, 2 }))
                .Node("Conv", ["p", "W"], ["low"], ("kernel_shape", new[] { 1, 1 }))
                .Node("Resize", ["low", "", "sc"], ["up"], ("mode", "nearest"),
                    ("coordinate_transformation_mode", "asymmetric"), ("nearest_mode", "floor"));
            string lat = "x";
            if (relu) { b.Node("Relu", ["x"], ["xr"]); lat = "xr"; }
            b.Node("Add", ["up", lat], ["y"]);
            OnnxModel m = OnnxModel.Load(b.Output("y").Build());
            int[] shape = [1, c, 12, 10];
            ExecPlan plan = GraphCompiler.Compile(m, shape);
            Assert.Contains(plan.Nodes, x => x.Op == ExecOp.ResizeNearest && x.In.Length == 2);
            Assert.DoesNotContain(plan.Nodes, x => x.Op == ExecOp.Elementwise);
            CheckCpuFusion(m, shape, $"nearest+add c{c}");
            Check(backend, m, shape, $"nearest+add c{c} relu={relu}");
        }
    }

    [SkippableTheory]
    [MemberData(nameof(Backends))]
    public void Attention_PackedQkv_FusesAndMatchesCpu(string backend)
    {
        // SVTR / ViT block: qkv = x W + b -> [n,T,3,H,D] -> [3,n,H,T,D], q / k / v single-index slices,
        // q * scale, q kᵀ, softmax, · v, heads back next to D; head sizes 12 (attn) and 40 (attn_d64)
        foreach ((int t, int h, int d) in new[] { (37, 4, 12), (70, 2, 40) })
        {
            var rng = new Random(t);
            int c = h * d;
            var b = new OnnxBuilder().Input("x", 1, t, c)
                .Init("Wqkv", Rand(rng, c * 3 * c, 0.4f), c, 3 * c).Init("Bqkv", Rand(rng, 3 * c, 0.2f), 3 * c)
                .Init("Wo", Rand(rng, c * c, 0.3f), c, c)
                .Init("scale", [1f / MathF.Sqrt(d)])
                .InitI64("s5", [1, t, 3, h, d], 5).InitI64("s4", [1, h, t, d], 4).InitI64("s3", [1, t, c], 3)
                .Node("MatMul", ["x", "Wqkv"], ["qkv0"]).Node("Add", ["qkv0", "Bqkv"], ["qkv1"])
                .Node("Reshape", ["qkv1", "s5"], ["qkv5"])
                .Node("Transpose", ["qkv5"], ["qkvt"], ("perm", new[] { 2, 0, 3, 1, 4 }));
            foreach ((string name, int i) in new[] { ("q", 0), ("k", 1), ("v", 2) })
                b.Node("Slice", ["qkvt"], [name + "5"], ("starts", new[] { i }), ("ends", new[] { i + 1 }), ("axes", new[] { 0 }))
                    .Node("Reshape", [name + "5", "s4"], [name]);
            b.Node("Mul", ["q", "scale"], ["qs"])
                .Node("Transpose", ["k"], ["kt"], ("perm", new[] { 0, 1, 3, 2 }))
                .Node("MatMul", ["qs", "kt"], ["sc"])
                .Node("Softmax", ["sc"], ["pr"], ("axis", -1))
                .Node("MatMul", ["pr", "v"], ["ctx"])
                .Node("Transpose", ["ctx"], ["ctxt"], ("perm", new[] { 0, 2, 1, 3 }))
                .Node("Reshape", ["ctxt", "s3"], ["ctx3"])
                .Node("MatMul", ["ctx3", "Wo"], ["y"]);
            OnnxModel m = OnnxModel.Load(b.Output("y").Build());
            int[] shape = [1, t, c];
            foreach (PlanTarget target in new[] { PlanTarget.Vulkan, PlanTarget.Metal })
            {
                ExecPlan gp = GraphCompiler.Compile(m, shape, new CompileOptions(Nhwc: true, Target: target));
                Assert.Single(gp.Nodes, x => x.Op == ExecOp.Attention);
                Assert.DoesNotContain(gp.Nodes, x => x.Op is ExecOp.Slice or ExecOp.Softmax or ExecOp.Transpose or ExecOp.MatMul);
            }
            Check(backend, m, shape, $"attention t{t} h{h} d{d}");
        }
    }

    [SkippableTheory]
    [MemberData(nameof(Backends))]
    public void Attention_Rank3_MatchesCpu(string backend)
    {
        // separate q / k / v projections, [B,T,D] without heads, scale as a Div after q kᵀ
        var rng = new Random(9);
        int t = 29, c = 24;
        byte[] model = new OnnxBuilder().Input("x", 2, t, c)
            .Init("Wq", Rand(rng, c * c, 0.4f), c, c).Init("Wk", Rand(rng, c * c, 0.4f), c, c).Init("Wv", Rand(rng, c * c, 0.4f), c, c)
            .Init("sq", [MathF.Sqrt(c)])
            .Node("MatMul", ["x", "Wq"], ["q"]).Node("MatMul", ["x", "Wk"], ["k"]).Node("MatMul", ["x", "Wv"], ["v"])
            .Node("Transpose", ["k"], ["kt"], ("perm", new[] { 0, 2, 1 }))
            .Node("MatMul", ["q", "kt"], ["s"]).Node("Div", ["s", "sq"], ["sd"])
            .Node("Softmax", ["sd"], ["p"], ("axis", -1))
            .Node("MatMul", ["p", "v"], ["y"])
            .Output("y").Build();
        OnnxModel m = OnnxModel.Load(model);
        ExecPlan gp = GraphCompiler.Compile(m, [2, t, c], new CompileOptions(Nhwc: true, Target: PlanTarget.Vulkan));
        Assert.Single(gp.Nodes, x => x.Op == ExecOp.Attention);
        Check(backend, m, [2, t, c], "attention rank 3");
    }

    [SkippableTheory]
    [MemberData(nameof(Backends))]
    public void GemmTransB_MatchesCpu(string backend)
    {
        // B stored [N,K] (transB = 1) on the tensor-core paths binds raw (issue #1); a Cast chain in
        // the shape math folds (issue #2)
        var rng = new Random(21);
        int m = 256, k = 128, n = 64;
        byte[] model = new OnnxBuilder().Input("x", m, k)
            .Init("B", Rand(rng, n * k, 0.3f), n, k).Init("C", Rand(rng, n, 0.5f), n)
            .Node("Shape", ["x"], ["s"]).Node("Cast", ["s"], ["s32"], ("to", 6)).Node("Cast", ["s32"], ["s64"], ("to", 7))
            .Node("Reshape", ["x", "s64"], ["xr"])
            .Node("Gemm", ["xr", "B", "C"], ["y"], ("transB", 1))
            .Output("y").Build();
        Check(backend, OnnxModel.Load(model), [m, k], "gemm transB=1");
    }

    [SkippableTheory]
    [MemberData(nameof(Backends))]
    public void DocGainOps_TwoInputs_MatchCpu(string backend)
    {
        // the ops a torch-exported image-enhancement net (DocGainNet) brings: two graph inputs (image +
        // fixed-size thumbnail), LeakyRelu, 3·tanh(x/3), Clip, DepthToSpace (CRD and DCR), a pixel
        // unshuffle through rank-6 reshape / transpose, ReduceMean(keepdims = 0) -> Gemm(transB) -> broadcast add
        var rng = new Random(31);
        int c = 8;
        byte[] model = new OnnxBuilder().Input("x", 1, c, 10, 12).Input("t", 1, 4, 6, 6)
            .Init("W1", Rand(rng, 16 * c * 9, 0.3f), 16, c, 3, 3).Init("B1", Rand(rng, 16, 0.3f), 16)
            .Init("W2", Rand(rng, 64 * 16, 0.3f), 64, 16, 1, 1)
            .Init("Wg", Rand(rng, 32 * 4, 0.5f), 32, 4).Init("Bg", Rand(rng, 32, 0.5f), 32)
            .Init("third", [1f / 3f]).Init("three", [3f]).Init("lo", [-0.5f]).Init("hi", [0.7f]).Init("half", [0.25f])
            .InitI64("ax3", [3], 1).InitI64("ax23", [2, 3], 2).InitI64("r1", [0, 0, -1, 2, 0], 5)
            .InitI64("r2", [0, 0, 0, 0, -1, 2], 6).InitI64("r3", [0, -1, 1, 1, 0, 0], 6)
            .Node("Conv", ["x", "W1", "B1"], ["c1"], ("kernel_shape", new[] { 3, 3 }), ("pads", new[] { 1, 1, 1, 1 }))
            .Node("LeakyRelu", ["c1"], ["a1"], ("alpha", 0.2f))
            .Node("Mul", ["a1", "third"], ["m1"]).Node("Tanh", ["m1"], ["th"]).Node("Mul", ["th", "three"], ["m2"])
            .Node("Conv", ["m2", "W2"], ["c2"], ("kernel_shape", new[] { 1, 1 }))
            .Node("DepthToSpace", ["c2"], ["d1"], ("blocksize", 2), ("mode", "CRD"))     // [1,16,20,24]
            .Node("DepthToSpace", ["d1"], ["d2"], ("blocksize", 2), ("mode", "DCR"))     // [1,4,40,48]
            .Node("Unsqueeze", ["d2", "ax3"], ["u"])                                     // [1,4,40,1,48]
            .Node("Reshape", ["u", "r1"], ["s1"]).Node("Reshape", ["s1", "r2"], ["s2"])  // [1,4,20,2,24,2]
            .Node("Transpose", ["s2"], ["st"], ("perm", new[] { 0, 1, 3, 5, 2, 4 }))
            .Node("Reshape", ["st", "r3"], ["s3"]).Node("Squeeze", ["s3", "ax23"], ["un"]) // [1,16,20,24]
            .Node("ReduceMean", ["t"], ["g"], ("axes", new[] { 2, 3 }), ("keepdims", 0))
            .Node("Gemm", ["g", "Wg", "Bg"], ["gg"], ("transB", 1))                       // [1,32]
            .Node("Unsqueeze", ["gg", "ax23"], ["g4"])                                   // [1,32,1,1]
            .Node("Slice", ["g4"], ["g16"], ("starts", new[] { 0 }), ("ends", new[] { 16 }), ("axes", new[] { 1 }))
            .Node("Add", ["un", "g16"], ["sum"])
            .Node("Sub", ["half", "sum"], ["sum2"])                                      // scalar - x: GPU affine pass
            .Node("Clip", ["sum2", "lo", "hi"], ["y"])
            .Output("y").Build();
        OnnxModel m = OnnxModel.Load(model);
        int[][] shapes = [[1, c, 10, 12], [1, 4, 6, 6]];
        var opt = new CompileOptions(Nhwc: true);
        ExecPlan plan = GraphCompiler.Compile(m, shapes[0], opt);
        // 3·tanh(x/3) folds into one scaled Tanh; the rank-6 pixel unshuffle into a SpaceToDepth
        Assert.Contains(plan.Nodes, x => x.Op == ExecOp.Tanh && x.P is TanhP { InScale: > 0.33f and < 0.34f, OutScale: 3f });
        Assert.Contains(plan.Nodes, x => x.Op == ExecOp.SpaceToDepth && x.P is DepthToSpaceP { Block: 2, Crd: true });
        Assert.DoesNotContain(plan.Nodes, x => x.Op == ExecOp.Transpose && plan.Values[x.In[0]].Shape.Length > 4);
        using var cpu = new InferenceSession(plan);
        Assert.Equal(2, cpu.InputCount);
        Assert.Equal(shapes[1], cpu.GetInputShape(1));   // static dims from the model
        var r = new Random(4);
        float[][] ins = [Rand(r, 8 * 120, 2f), Rand(r, 4 * 36, 2f)];
        for (int k = 0; k < 2; k++) ins[k].CopyTo(cpu.GetInput(k));
        cpu.Run();
        float[] want = cpu.Output.ToArray();
        // the unfused NCHW plan agrees (layouts, relayouts and the new kernels' NCHW paths)
        using (var plain = new InferenceSession(GraphCompiler.Compile(m, shapes[0], new CompileOptions(Nhwc: false, Fuse: false))))
        {
            for (int k = 0; k < 2; k++) ins[k].CopyTo(plain.GetInput(k));
            plain.Run();
            float[] p = plain.Output.ToArray();
            for (int i = 0; i < want.Length; i++) Assert.True(Math.Abs(p[i] - want[i]) <= 1e-4f, $"nchw [{i}] {p[i]} vs {want[i]}");
        }
        float[]? got = null;
        if (backend == "vulkan")
        {
            VkBackend be;
            try { be = VkBackend.Default; _ = be.DeviceName; }
            catch { Skip.If(true, "no vulkan device"); return; }
            using VkInferenceSession s = be.NewSession(m, shapes[0], opt);
            Assert.Equal(2, s.InputCount);
            for (int k = 0; k < 2; k++) ins[k].CopyTo(s.GetInput(k));
            s.Run();
            got = s.Output.ToArray();
        }
        else
        {
            Skip.IfNot(OperatingSystem.IsMacOS(), "no metal device");
            MetalBackend mb;
            try { mb = MetalBackend.Default; _ = mb.DeviceName; }
            catch { Skip.If(true, "no metal device"); return; }
            using MtlInferenceSession s = mb.NewSession(m, shapes[0], opt);
            for (int k = 0; k < 2; k++) ins[k].CopyTo(s.GetInput(k));
            s.Run();
            got = s.Output.ToArray();
        }
        float tol = 0.01f * want.Max(Math.Abs) + 0.02f;
        for (int i = 0; i < want.Length; i++)
            Assert.True(Math.Abs(got[i] - want[i]) <= tol, $"{backend} [{i}] gpu {got[i]} vs cpu {want[i]}");
    }

    /// <summary>CPU fused plan against the unfused one (fusions must not change the math beyond rounding).</summary>
    private static void CheckCpuFusion(OnnxModel m, int[] shape, string what)
    {
        using var fused = new InferenceSession(GraphCompiler.Compile(m, shape));
        using var plain = new InferenceSession(GraphCompiler.Compile(m, shape, new CompileOptions(Nhwc: true, Fuse: false)));
        float[] input = Rand(new Random(shape.Sum() + 1), fused.Input.Length, 2f);
        input.CopyTo(fused.Input);
        input.CopyTo(plain.Input);
        fused.Run();
        plain.Run();
        float[] want = plain.Output.ToArray(), got = fused.Output.ToArray();
        float tol = 1e-4f * want.Max(Math.Abs) + 1e-4f;
        for (int i = 0; i < want.Length; i++)
            Assert.True(Math.Abs(want[i] - got[i]) <= tol, $"cpu {what}: [{i}] fused {got[i]} vs plain {want[i]}");
    }

    // x * HardSigmoid(x; 1/6, 0.5) (hardswish) or x * Sigmoid(x) (swish)
    private static void Act(OnnxBuilder b, string x, string y, bool hard)
    {
        if (hard) b.Node("HardSigmoid", [x], [x + "_g"], ("alpha", 1f / 6f), ("beta", 0.5f));
        else b.Node("Sigmoid", [x], [x + "_g"]);
        b.Node("Mul", [x, x + "_g"], [y]);
    }

    private static void Check(string backend, OnnxModel m, int[] shape, string what)
    {
        Func<float[], float[]>? gpu = Gpu(backend, m, shape);
        Skip.If(gpu is null, $"no {backend} device");
        using var cpu = new InferenceSession(GraphCompiler.Compile(m, shape));
        float[] input = Rand(new Random(shape.Sum()), cpu.Input.Length, 2f);
        input.CopyTo(cpu.Input);
        cpu.Run();
        float[] want = cpu.Output.ToArray();
        float[] got = gpu!(input);
        Assert.Equal(want.Length, got.Length);
        float tol = 0.01f * want.Max(Math.Abs) + 0.02f;   // fp16 storage
        float maxd = 0;
        int at = 0;
        for (int i = 0; i < got.Length; i++)
        {
            float d = Math.Abs(got[i] - want[i]);
            if (!(d <= maxd)) { maxd = d; at = i; }
        }
        Assert.True(maxd <= tol, $"{backend} {what}: max |gpu-cpu| {maxd} (tol {tol}) at {at}: gpu={got[at]} cpu={want[at]}");
    }

    private static Func<float[], float[]>? Gpu(string backend, OnnxModel m, int[] shape)
    {
        if (backend == "vulkan")
        {
            VkBackend be;
            try { be = VkBackend.Default; _ = be.DeviceName; }
            catch { return null; }
            return input =>
            {
                using VkInferenceSession s = be.NewSession(m, shape);
                input.CopyTo(s.Input);
                s.Run();
                return s.Output.ToArray();
            };
        }
        if (!OperatingSystem.IsMacOS()) return null;
        MetalBackend mb;
        try { mb = MetalBackend.Default; _ = mb.DeviceName; }
        catch { return null; }
        return input =>
        {
            using MtlInferenceSession s = mb.NewSession(m, shape);
            input.CopyTo(s.Input);
            s.Run();
            return s.Output.ToArray();
        };
    }

    private static float[] Rand(Random rng, int n, float scale) =>
        Enumerable.Range(0, n).Select(_ => (float)(rng.NextDouble() * 2 - 1) * scale).ToArray();
}
