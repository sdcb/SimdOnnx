using Sdcb.SimdOnnx.Backends.Vulkan;
using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Onnx;

namespace Sdcb.SimdOnnx.Tests;

/// <summary>
/// Whole-graph Vulkan vs CPU checks of single conv / Gemm models, one per kernel route: the
/// SG16 direct family (pw_dot, conv_kdot, im2col + gemm_cm8, gemm_cm8 tiles) on Intel Xe2, the
/// default coopmat / generic shaders elsewhere. Covers batch > 1, odd channel counts and tile tails.
/// </summary>
public class VkRouteTests
{
    private static bool VkOk()
    {
        try { _ = VkBackend.Default.Dev.DeviceName; return true; }
        catch { return false; }
    }

    [SkippableTheory]
    // n, ic, oc, k, stride, pad, h, w, relu, expected SG16 route (prefix of the dispatch name)
    [InlineData(1, 32, 64, 1, 1, 0, 9, 13, true, "pw_dot")]
    [InlineData(3, 64, 24, 1, 1, 0, 5, 7, false, "pw_dot")]          // batch, narrow N
    [InlineData(1, 256, 96, 1, 1, 0, 11, 17, true, "gemm_cm8")]      // 128-wide tile, N tail
    [InlineData(2, 160, 48, 1, 1, 0, 7, 9, false, "gemm_cm8_n64")]   // batch, M tail
    [InlineData(1, 144, 30, 1, 1, 0, 6, 6, true, "gemm_cm8_n32")]    // N % 4 != 0
    [InlineData(1, 16, 32, 3, 1, 1, 12, 20, true, "conv_kdot")]
    [InlineData(2, 24, 48, 3, 2, 1, 13, 15, false, "conv_kdot")]     // batch, stride 2, odd sizes
    [InlineData(1, 8, 16, 5, 1, 2, 9, 11, true, "conv_kdot")]
    [InlineData(1, 128, 64, 3, 1, 1, 10, 14, true, "im2col")]        // K = 1152
    [InlineData(2, 128, 36, 3, 2, 1, 9, 11, false, "im2col")]        // batch, stride 2, N tail
    [InlineData(1, 3, 16, 3, 2, 1, 32, 48, true, "conv_k33s2")]      // stem: ic % 4 != 0 -> generic
    public void Conv_MatchesCpu(int n, int ic, int oc, int k, int stride, int pad, int h, int w, bool relu, string route)
    {
        Skip.IfNot(VkOk(), "no Vulkan device");
        var rng = new Random(ic * 131 + oc * 7 + k);
        var b = new OnnxBuilder().Input("x", n, ic, h, w)
            .Init("W", Rand(rng, oc * ic * k * k, 0.3f), oc, ic, k, k)
            .Init("B", Rand(rng, oc, 0.5f), oc)
            .Node("Conv", ["x", "W", "B"], [relu ? "c" : "y"],
                ("kernel_shape", new[] { k, k }), ("pads", new[] { pad, pad, pad, pad }), ("strides", new[] { stride, stride }));
        if (relu) b.Node("Relu", ["c"], ["y"]);
        Check(OnnxModel.Load(b.Output("y").Build()), [n, ic, h, w], route, rng);
    }

    [SkippableTheory]
    // t rows, k, n, expected SG16 route
    [InlineData(40, 64, 96, "gemm_dot8")]
    [InlineData(80, 256, 200, "gemm_cm8")]
    [InlineData(33, 160, 64, "gemm_cm8_n64")]
    [InlineData(7, 96, 120, "gemm_dot8")]
    public void Gemm_MatchesCpu(int t, int k, int nn, string route)
    {
        Skip.IfNot(VkOk(), "no Vulkan device");
        var rng = new Random(t * 31 + k);
        byte[] model = new OnnxBuilder().Input("x", 1, t, k)
            .Init("W", Rand(rng, k * nn, 0.2f), k, nn)
            .Init("B", Rand(rng, nn, 0.5f), nn)
            .Node("MatMul", ["x", "W"], ["m"])
            .Node("Add", ["m", "B"], ["y"])
            .Output("y").Build();
        Check(OnnxModel.Load(model), [1, t, k], route == "gemm_dot8" ? "pw_dot" : route, rng);
    }

    [SkippableTheory]
    // the wide-subgroup no-coopmat routing (Adreno) emulated on this device:
    // n, ic, oc, k, stride, pad, hardswish, expected dispatch prefix
    [InlineData(1, 64, 96, 1, 1, 0, false, "pw_dot")]       // every 1x1 (any K, any size)
    [InlineData(2, 256, 64, 1, 1, 0, true, "pw_dot")]
    [InlineData(1, 32, 48, 3, 1, 1, false, "conv_dot_s1")]  // conv_dot's own case
    [InlineData(1, 32, 48, 3, 2, 1, false, "conv_dot_s2")]
    [InlineData(2, 32, 48, 3, 1, 1, false, "conv_kdot")]    // batched
    [InlineData(1, 32, 48, 3, 1, 1, true, "conv_kdot")]     // hardswish epilogue
    [InlineData(1, 16, 24, 5, 2, 2, false, "conv_kdot")]    // other kernel sizes
    [InlineData(1, 8, 32, 7, 1, 3, true, "conv_kdot")]
    public void WideDot_MatchesCpu(int n, int ic, int oc, int k, int stride, int pad, bool hswish, string route)
    {
        VkBackend? be = s_wideDot.Value;
        Skip.If(be is null || !be.F16, "needs a Vulkan device with fp16 storage");
        var rng = new Random(ic * 17 + oc + k);
        var b = new OnnxBuilder().Input("x", n, ic, 11, 19)
            .Init("W", Rand(rng, oc * ic * k * k, 0.3f), oc, ic, k, k)
            .Init("B", Rand(rng, oc, 0.5f), oc)
            .Node("Conv", ["x", "W", "B"], [hswish ? "c" : "y"],
                ("kernel_shape", new[] { k, k }), ("pads", new[] { pad, pad, pad, pad }), ("strides", new[] { stride, stride }));
        if (hswish)
            b.Node("HardSigmoid", ["c"], ["g"], ("alpha", 1f / 6f), ("beta", 0.5f)).Node("Mul", ["c", "g"], ["y"]);
        OnnxModel m = OnnxModel.Load(b.Output("y").Build());
        int[] shape = [n, ic, 11, 19];
        Assert.Contains(be!.GraphFor(m, shape).Dispatches, d => d.Name.StartsWith(route, StringComparison.Ordinal));
        Check(m, shape, route, rng, be);
    }

    // a private device with the wide-subgroup flags (no coopmat, ConvDot): flipping them on the
    // shared default device would re-route graphs other tests compile concurrently
    private static readonly Lazy<VkBackend?> s_wideDot = new(() =>
    {
        try
        {
            var be = new VkBackend();
            be.Dev.CoopMat16 = false;
            be.Dev.CoopM8 = false;
            be.Dev.ConvDot = true;
            be.Dev.NcDirect = false;
            return be;
        }
        catch { return null; }
    });

    [SkippableTheory]
    // Intel no-coopmat routing (VkDevice.NcDirect, Xe-LP iGPUs; needs the 8-lane pin, so real devices only):
    // n, ic, oc, k, stride, pad, h, w, epilogue (0 none, 1 relu, 2 swish), expected dispatch prefix
    [InlineData(1, 64, 96, 1, 1, 0, 40, 60, 1, "pw_dot(")]        // cin <= 96: pw_dot at any size
    [InlineData(1, 128, 96, 1, 1, 0, 9, 13, 1, "pw_dot(")]        // cin 128 within 2^16 outputs
    [InlineData(1, 128, 96, 1, 1, 0, 40, 60, 1, "gemm_nc_s(")]    // cin 128 past 2^16 outputs
    [InlineData(2, 160, 48, 1, 1, 0, 7, 9, 0, "gemm_nc_s(")]      // batch, M tail, N < tile
    [InlineData(1, 144, 30, 1, 1, 0, 6, 6, 1, "gemm_nc_s(")]      // N % 4 != 0
    [InlineData(1, 256, 72, 1, 1, 0, 11, 17, 2, "gemm_nc(")]      // swish: the staged epilogue build
    [InlineData(1, 16, 32, 3, 1, 1, 12, 20, 1, "conv_kdot(")]
    [InlineData(2, 24, 48, 3, 2, 1, 13, 15, 0, "conv_kdot(")]     // batch, stride 2, odd sizes
    [InlineData(1, 128, 64, 3, 1, 1, 10, 14, 1, "gemm_nc_s(")]    // K = 1152: im2col + gemm_nc
    [InlineData(2, 128, 36, 3, 2, 1, 9, 11, 2, "gemm_nc(")]       // batch, stride 2, N tail
    public void NcDirect_MatchesCpu(int n, int ic, int oc, int k, int stride, int pad, int h, int w, int epi, string route)
    {
        Skip.IfNot(VkOk() && VkBackend.Default.Dev.NcDirect && VkBackend.Default.F16, "no NcDirect device");
        var rng = new Random(ic * 29 + oc * 3 + k + epi);
        var b = new OnnxBuilder().Input("x", n, ic, h, w)
            .Init("W", Rand(rng, oc * ic * k * k, 0.3f), oc, ic, k, k)
            .Init("B", Rand(rng, oc, 0.5f), oc)
            .Node("Conv", ["x", "W", "B"], [epi == 0 ? "y" : "c"],
                ("kernel_shape", new[] { k, k }), ("pads", new[] { pad, pad, pad, pad }), ("strides", new[] { stride, stride }));
        if (epi == 1) b.Node("Relu", ["c"], ["y"]);
        if (epi == 2) b.Node("Sigmoid", ["c"], ["g"]).Node("Mul", ["c", "g"], ["y"]);
        OnnxModel m = OnnxModel.Load(b.Output("y").Build());
        int[] shape = [n, ic, h, w];
        Assert.Contains(VkBackend.Default.GraphFor(m, shape).Dispatches, d => d.Name.StartsWith(route, StringComparison.Ordinal));
        Check(m, shape, route, rng);
    }

    [SkippableTheory]
    // t rows, k, n, expected dispatch prefix on NcDirect devices
    [InlineData(40, 64, 96, "pw_dot(")]
    [InlineData(80, 256, 200, "gemm_nc_s(")]
    [InlineData(33, 160, 64, "gemm_nc_s(")]
    public void NcDirectGemm_MatchesCpu(int t, int k, int nn, string route)
    {
        Skip.IfNot(VkOk() && VkBackend.Default.Dev.NcDirect && VkBackend.Default.F16, "no NcDirect device");
        var rng = new Random(t * 37 + k);
        OnnxModel m = OnnxModel.Load(new OnnxBuilder().Input("x", 1, t, k)
            .Init("W", Rand(rng, k * nn, 0.2f), k, nn)
            .Init("B", Rand(rng, nn, 0.5f), nn)
            .Node("MatMul", ["x", "W"], ["mm"])
            .Node("Add", ["mm", "B"], ["y"])
            .Output("y").Build());
        int[] shape = [1, t, k];
        Assert.Contains(VkBackend.Default.GraphFor(m, shape).Dispatches, d => d.Name.StartsWith(route, StringComparison.Ordinal));
        Check(m, shape, route, rng);
    }

    private static void Check(OnnxModel m, int[] shape, string route, Random rng, VkBackend? backend = null)
    {
        using var cpu = new InferenceSession(GraphCompiler.Compile(m, shape));
        float[] input = Rand(rng, cpu.Input.Length, 2f);
        input.CopyTo(cpu.Input);
        cpu.Run();
        float[] want = cpu.Output.ToArray();

        VkBackend be = backend ?? VkBackend.Default;
        using var gpu = be.NewSession(m, shape);
        input.CopyTo(gpu.Input);
        gpu.Run();
        float[] got = gpu.Output.ToArray();

        Assert.Equal(want.Length, got.Length);
        float scale = want.Max(Math.Abs);
        float tol = 0.01f * scale + 0.02f;   // fp16 storage of inputs / weights / outputs
        float maxd = 0;
        int at = 0;
        for (int i = 0; i < got.Length; i++)
        {
            float d = Math.Abs(got[i] - want[i]);
            if (d > maxd) { maxd = d; at = i; }
        }
        Assert.True(maxd <= tol, $"{route}: max |gpu-cpu| {maxd} (tol {tol}) at {at}: gpu={got[at]} cpu={want[at]}");
        // numbers first: they still hold when a knob moves the routing
        if (backend is null && be.Dev.CoopM8 && be.F16)
        {
            VkGraphModel g = be.GraphFor(m, shape);
            Assert.Contains(g.Dispatches, d => d.Name.StartsWith(route + "(", StringComparison.Ordinal)
                || d.Name.StartsWith(route + "_", StringComparison.Ordinal) && route == "im2col");
        }
    }

    private static float[] Rand(Random rng, int n, float scale) =>
        Enumerable.Range(0, n).Select(_ => (float)(rng.NextDouble() * 2 - 1) * scale).ToArray();
}
