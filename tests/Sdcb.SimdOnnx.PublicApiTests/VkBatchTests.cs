using Sdcb.SimdOnnx.Backends.Vulkan;
using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Onnx;
using Sdcb.SimdOnnx.Tests;

namespace Sdcb.SimdOnnx.PublicApiTests;

/// <summary>Vulkan sessions on a shared <see cref="VkBufferPool"/>, async Submit / Wait, shape queries and
/// <see cref="VkBatch"/> (public surface only). Skipped without a Vulkan device.</summary>
public class VkBatchTests
{
    private static readonly Random s_rng = new(23);
    private static readonly float[] W2 = Rand(4 * 16);

    private static float[] Rand(int n)
    {
        lock (s_rng) return Enumerable.Range(0, n).Select(_ => (float)s_rng.NextDouble() * 2f - 1f).ToArray();
    }

    // x -> 3x3 conv (8->16) "h" -> relu -> 1x1 conv (16->4) "y"
    private static readonly OnnxModel s_model = OnnxModel.Load(new OnnxBuilder().Input("x", 1, 8, 6, 5)
        .Init("W1", Rand(16 * 8 * 9), 16, 8, 3, 3).Init("W2", W2, 4, 16, 1, 1)
        .Node("Conv", ["x", "W1"], ["h"], ("kernel_shape", new[] { 3, 3 }), ("pads", new[] { 1, 1, 1, 1 }))
        .Node("Relu", ["h"], ["r"])
        .Node("Conv", ["r", "W2"], ["y"], ("kernel_shape", new[] { 1, 1 }))
        .Output("y").Build());

    private static VkBackend Backend()
    {
        Skip.IfNot(VkBackend.TryGetDefault(out VkBackend? b), "no Vulkan device");
        return b!;
    }

    private static float[] RunOwn(VkBackend b, int[] shape, float[] x)
    {
        using VkInferenceSession s = b.NewSession(s_model, shape, VkBackend.PlanOptions);
        x.CopyTo(s.Input);
        s.Run();
        return s.Output.ToArray();
    }

    private static void Close(float[] want, ReadOnlySpan<float> got)
    {
        Assert.Equal(want.Length, got.Length);
        for (int i = 0; i < want.Length; i++) Assert.True(Math.Abs(want[i] - got[i]) <= 1e-5f, $"[{i}] {want[i]} vs {got[i]}");
    }

    [SkippableFact]
    public void OutputShape_WithoutASession()
    {
        VkBackend b = Backend();
        Assert.Equal([1, 4, 12, 9], b.OutputShape(s_model, [1, 8, 12, 9], VkBackend.PlanOptions));
        Assert.Equal([1, 16, 12, 9], b.OutputShape(s_model, [1, 8, 12, 9], VkBackend.PlanOptions with { OutputName = "h" }));
    }

    [SkippableFact]
    public void Pool_GrowingMakesEarlierSessionsStale_SubmitWaitMatchesRun()
    {
        VkBackend b = Backend();
        using VkBufferPool pool = b.CreateBufferPool();
        int[] small = [1, 8, 6, 5], large = [1, 8, 24, 20];
        float[] xs = Rand(8 * 6 * 5), xl = Rand(8 * 24 * 20);

        using VkInferenceSession s1 = b.NewSession(s_model, small, VkBackend.PlanOptions, pool);
        xs.CopyTo(s1.Input);
        s1.Submit();
        s1.Wait();
        Close(RunOwn(b, small, xs), s1.Output);
        Assert.False(s1.IsStale);

        using VkInferenceSession s2 = b.NewSession(s_model, large, VkBackend.PlanOptions, pool);
        Assert.True(s1.IsStale, "a larger session should regrow the pool");
        Assert.Throws<InvalidOperationException>(() => s1.Run());
        xl.CopyTo(s2.Input);
        s2.Run();
        Close(RunOwn(b, large, xl), s2.Output);

        // a session made after the growth fits in the grown buffers: nothing goes stale
        using VkInferenceSession s3 = b.NewSession(s_model, small, VkBackend.PlanOptions, pool);
        Assert.False(s2.IsStale);
        xs.CopyTo(s3.Input);
        s3.Run();
        Close(RunOwn(b, small, xs), s3.Output);
    }

    [SkippableFact]
    public void Batch_UnitsInWaves_MatchPerShapeSessions()
    {
        VkBackend b = Backend();
        int[][] shapes = [[1, 8, 6, 5], [1, 8, 10, 7], [1, 8, 3, 12], [1, 8, 6, 5]];
        float[][] xs = [.. shapes.Select(s => Rand(8 * s[2] * s[3]))];
        using VkBatch batch = b.CreateBatch();
        for (int round = 0; round < 2; round++)   // reuse after Clear
        {
            batch.Clear();
            for (int u = 0; u < shapes.Length; u++)
            {
                CompileOptions opt = u == 3 ? VkBackend.PlanOptions with { OutputName = "h" } : VkBackend.PlanOptions;
                Assert.Equal(u, batch.Add(s_model, shapes[u], opt));
                if (u == 1) batch.EndWave();
            }
            Assert.Equal(4, batch.Count);
            Assert.Equal(2, batch.Waves);
            for (int u = 0; u < shapes.Length; u++) xs[u].CopyTo(batch.Input(u));
            batch.Submit();
            batch.Wait(0);
            Assert.Equal(0, batch.WaveOf(1));
            Close(RunOwn(b, shapes[0], xs[0]), batch.Output(0));
            Close(RunOwn(b, shapes[1], xs[1]), batch.Output(1));
            batch.WaitAll();
            Close(RunOwn(b, shapes[2], xs[2]), batch.Output(2));
            Assert.Equal([1, 16, 6, 5], batch.OutputShape(3));
            using VkInferenceSession hs = b.NewSession(s_model, shapes[3], VkBackend.PlanOptions with { OutputName = "h" });
            xs[3].CopyTo(hs.Input);
            hs.Run();
            Close(hs.Output.ToArray(), batch.Output(3));
        }
    }
}
