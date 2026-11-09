using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Kernels;
using Sdcb.SimdOnnx.Onnx;
using Xunit;

namespace Sdcb.SimdOnnx.Tests;

/// <summary>Plan-level fusion assertions, NOFUSE cross-check, kernel unit checks.</summary>
public class EngineTests
{
    internal static string GcPath => Path.Combine(Golden.ModelDir, "gcdrnet_gcnet_sim.onnx");
    internal static string DrPath => Path.Combine(Golden.ModelDir, "gcdrnet_drnet_sim.onnx");

    [Fact]
    public void Fusion_Conv_Gelu_ThroughAlias()
    {
        // the gelu decomposition commutes before its producer transpose, then
        // fuses into the 1x1 conv via the reshape Buf-alias — the plan must
        // contain no Gelu node and the conv must carry the epilogue
        ExecPlan gc = GraphCompiler.Compile(OnnxModel.Load(GcPath), [1, 3, 96, 128]);
        Assert.DoesNotContain(gc.Nodes, n => n.Op == ExecOp.Gelu);
        Assert.True(gc.Nodes.Count(n => n.Op == ExecOp.Conv &&
                ((ConvP)n.P!).Act == ActivationKind.Gelu) >= 2,
            "gc plan lost the Conv+Gelu fusion");
    }

    [Theory]
    [InlineData(false, 2)]   // logical NCHW I/O: one boundary relayout in, one out
    [InlineData(true, 0)]    // physical I/O (fused pipelines): no transposes at all
    public void Nhwc_TransposesDissolve(bool physicalIo, int expected)
    {
        // all 32 model transposes per net ([0,2,1], [0,3,1,2]) must dissolve into
        // zero-copy layout views; acceptance bound is <= 8
        foreach (var (path, c) in new[] { (GcPath, 3), (DrPath, 6) })
        {
            ExecPlan p = GraphCompiler.Compile(OnnxModel.Load(path), [1, c, 96, 128],
                new CompileOptions(Nhwc: true, PhysicalIo: physicalIo));
            int n = p.Nodes.Count(m => m.Op == ExecOp.Transpose);
            Assert.True(n <= 8, $"{Path.GetFileName(path)}: {n} transposes");
            Assert.Equal(expected, n);
        }
    }

    [Fact]
    public void Nhwc_MatchesNchw()
    {
        // layout-planned CPU plan vs the legacy all-NCHW plan, same input
        foreach (var (path, tag, name) in new[] { (GcPath, "pad", "input"), (DrPath, "pad", "dr_in") })
        {
            float[] input = Golden.F32(tag, name);
            int[] shape = Golden.ShapeOf(tag, name);
            OnnxModel m = OnnxModel.Load(path);
            float[] Run(bool nhwc)
            {
                using var s = new InferenceSession(GraphCompiler.Compile(m, shape, new CompileOptions(nhwc)));
                input.CopyTo(s.Input); s.Run();
                return s.Output.ToArray();
            }
            var (max, rmse, _) = Golden.Diff(Run(true), Run(false));
            Assert.True(max < 1e-5f, $"{Path.GetFileName(path)} nhwc vs nchw max diff {max} rmse {rmse}");
        }
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ReleaseInputs_ReusesTheInputBuffer_SameOutput(bool nhwc)
    {
        // x -> conv 8->32 -> relu -> conv 32->8: the input is dead after the first conv, so with
        // ReleaseInputs the second conv's output may take its place; the outputs are the same bits
        var rng = new Random(11);
        byte[] bytes = new OnnxBuilder().Input("x", 1, 8, 24, 20)
            .Init("W1", rng.Array(32 * 8), 32, 8, 1, 1).Init("W2", rng.Array(8 * 32), 8, 32, 1, 1)
            .Node("Conv", ["x", "W1"], ["c1"], ("kernel_shape", new[] { 1, 1 }))
            .Node("Relu", ["c1"], ["r1"])
            .Node("Conv", ["r1", "W2"], ["y"], ("kernel_shape", new[] { 1, 1 }))
            .Output("y").Build();
        OnnxModel m = OnnxModel.Load(bytes);
        int[] shape = [1, 8, 24, 20];
        float[] input = rng.Array(8 * 24 * 20);
        (float[] output, int arena) Run(bool release)
        {
            ExecPlan plan = GraphCompiler.Compile(m, shape, new CompileOptions(nhwc) { ReleaseInputs = release });
            using var s = new InferenceSession(plan);
            input.CopyTo(s.Input); s.Run();
            return (s.Output.ToArray(), plan.ArenaFloats);
        }
        var (kept, keptArena) = Run(false);
        var (released, releasedArena) = Run(true);
        Assert.Equal(kept, released);
        Assert.True(releasedArena < keptArena, $"arena {releasedArena} floats with ReleaseInputs, {keptArena} without");
    }

    [Fact]
    public void NoFuse_MatchesFused()
    {
        float[] input = Golden.F32("small", "input");
        int[] shape = Golden.ShapeOf("small", "input");
        OnnxModel m = OnnxModel.Load(GcPath);

        float[] Run(bool fuse)
        {
            using var s = new InferenceSession(GraphCompiler.Compile(m, shape, CompileOptions.Default with { Fuse = fuse }));
            input.CopyTo(s.Input); s.Run();
            return s.Output.ToArray();
        }

        var (max, rmse, _) = Golden.Diff(Run(fuse: true), Run(fuse: false));
        Assert.True(max < 1e-5f, $"fused vs unfused max diff {max} rmse {rmse}");
    }

    [Fact]
    public void Transpose_NonIdentityLeadingPerm_Correct()
    {
        // perm [1,0,3,2] must NOT take the last-two-dim fast path (would be wrong);
        // it should fall through to the generic permute and still be correct
        var rng = new Random(5);
        int d0 = 2, d1 = 3, d2 = 4, d3 = 5;
        float[] src = rng.Array(d0 * d1 * d2 * d3);
        float[] got = new float[src.Length];
        Ops.Transpose(src, [d0, d1, d2, d3], got, [d1, d0, d3, d2], [1, 0, 3, 2]);
        for (int i0 = 0; i0 < d0; i0++)
            for (int i1 = 0; i1 < d1; i1++)
                for (int i2 = 0; i2 < d2; i2++)
                    for (int i3 = 0; i3 < d3; i3++)
                        Assert.Equal(
                            src[((i0 * d1 + i1) * d2 + i2) * d3 + i3],
                            got[((i1 * d0 + i0) * d3 + i3) * d2 + i2]);
    }

    [Fact]
    public void Conv_GroupedNonDepthwise_MatchesReference()
    {
        // group=2 with ic=4 (not depthwise) routes to the scalar path — check vs manual
        var rng = new Random(11);
        int ic = 4, oc = 4, ih = 5, iw = 6, g = 2;
        float[] x = rng.Array(ic * ih * iw);
        float[] w = rng.Array(oc * (ic / g) * 9);
        float[] dst = new float[oc * ih * iw];
        var p = new ConvP { Pads = [1, 1, 1, 1], Strides = [1, 1], Dilations = [1, 1], Kernel = [3, 3], Group = g };
        Ops.Conv(x, [1, ic, ih, iw], w, [oc, ic / g, 3, 3], ReadOnlySpan<float>.Empty,
            dst, [1, oc, ih, iw], p);

        for (int o = 0; o < oc; o++)
            for (int y = 0; y < ih; y++)
                for (int xx = 0; xx < iw; xx++)
                {
                    float acc = 0;
                    int gi = o / (oc / g) * (ic / g);
                    for (int c = 0; c < ic / g; c++)
                        for (int ky = 0; ky < 3; ky++)
                            for (int kx = 0; kx < 3; kx++)
                            {
                                int sy = y + ky - 1, sx = xx + kx - 1;
                                if ((uint)sy < (uint)ih && (uint)sx < (uint)iw)
                                    acc += x[(gi + c) * ih * iw + sy * iw + sx] *
                                           w[((o * (ic / g) + c) * 3 + ky) * 3 + kx];
                            }
                    Assert.Equal(acc, dst[o * ih * iw + y * iw + xx], 5);
                }
    }

    [Fact]
    public void Conv_ResidualAdd_MatchesManual()
    {
        // unit-level check of the Conv+Add epilogue (the fused path has no model coverage)
        var rng = new Random(7);
        int ic = 4, oc = 8, ih = 9, iw = 11;
        float[] x = rng.Array(ic * ih * iw);
        float[] w = rng.Array(oc * ic * 9);
        float[] b = rng.Array(oc);
        float[] res = rng.Array(oc * ih * iw);
        float[] want = new float[oc * ih * iw], got = new float[oc * ih * iw];

        var p = new ConvP { Pads = [1, 1, 1, 1], Strides = [1, 1], Dilations = [1, 1], Kernel = [3, 3] };
        Ops.Conv(x, [1, ic, ih, iw], w, [oc, ic, 3, 3], b, want, [1, oc, ih, iw], p);
        for (int i = 0; i < want.Length; i++) want[i] += res[i];

        Ops.Conv(x, [1, ic, ih, iw], w, [oc, ic, 3, 3], b, got, [1, oc, ih, iw], p, res);
        var (max, _, _) = Golden.Diff(want, got);
        Assert.True(max < 1e-4f, $"conv+add max diff {max}");
    }

    [SkippableTheory]
    [InlineData(16, 16, 9, 11, 0, 0f, false, 0)]   // constant pad
    [InlineData(24, 20, 13, 6, 1, 0f, true, 1)]    // reflect, residual, relu, oc lane tail, odd tiles
    [InlineData(32, 48, 4, 4, 0, 0.5f, false, 3)]  // nonzero pad value, gelu
    [InlineData(64, 16, 17, 23, 1, 0f, true, 0)]
    public void Conv_Winograd_MatchesDirect(int ic, int oc, int h, int w, int mode, float padValue,
        bool residual, int act)
    {
        var rng = new Random(ic * 31 + oc);
        float[] x = rng.Array(h * w * ic);           // NHWC
        float[] wt = rng.Array(oc * ic * 9);
        float[] b = rng.Array(oc);
        float[] res = residual ? rng.Array(h * w * oc) : [];
        var p = new ConvP
        {
            Pads = [1, 1, 1, 1], Strides = [1, 1], Dilations = [1, 1], Kernel = [3, 3],
            PadMode = (PadMode)mode, PadValue = padValue, Act = (ActivationKind)act,
        };
        Skip.IfNot(Nhwc.WinogradEligible(p, [oc, ic, 3, 3], ic), "Winograd disabled or ISA tier without it");
        NhwcConvPlan direct = Nhwc.Pack(wt, [oc, ic, 3, 3], b, 1, ic);
        NhwcConvPlan wino = Nhwc.Pack(wt, [oc, ic, 3, 3], b, 1, ic);
        wino.WinoU = Nhwc.PackWinograd(wt, oc, ic);
        float[] want = new float[h * w * oc], got = new float[h * w * oc];
        Nhwc.Conv(x, h, w, ic, want, h, w, oc, p, direct, res);
        Nhwc.Conv(x, h, w, ic, got, h, w, oc, p, wino, res);
        var (max, _, _) = Golden.Diff(want, got);
        float scale = want.Max(MathF.Abs);
        Assert.True(max < 1e-4f * scale, $"winograd max diff {max} (|out| max {scale})");
    }
}

internal static class RngExt
{
    public static float[] Array(this Random r, int n)
        => Enumerable.Range(0, n).Select(_ => (float)r.NextDouble() * 2f - 1f).ToArray();
}
