using Sdcb.SimdOnnx.Backends.Vulkan;
using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Onnx;

namespace Sdcb.SimdOnnx.Tests.Api;

/// <summary>
/// A third-party extension written against the public surface only (this assembly has no
/// InternalsVisibleTo): a logical graph pass swaps every Relu for a custom operator with its own
/// CPU kernel and Vulkan emission, and the results must match the built-in Relu.
/// </summary>
public class ExtensionTests
{
    /// <summary>HardSigmoid (clamp(a·x + b, 0, 1)) — on the CPU by hand, on Vulkan through the core "hsig" shader.</summary>
    private sealed class MyHardSigmoid : ExtOp
    {
        public static readonly MyHardSigmoid Instance = new();
        public int CpuRuns, VulkanEmits;

        public override string Name => "MyHardSigmoid";

        public override void PlanLayout(LayoutPlanner lp, ExecNode n)
        {
            // elementwise: run in whatever layout the input already has
            lp.Values[n.Out[0]].Layout = lp.Values[n.In[0]].Layout;
        }

        public override void RunCpu(InferenceSession s, ExecNode n)
        {
            ReadOnlySpan<float> x = s.GetIn(n.In[0]);
            Span<float> y = s.GetSpan(n.Out[0]);
            var p = (HardSigmoidP)n.P!;
            for (int i = 0; i < y.Length; i++) y[i] = Math.Clamp(p.Alpha * x[i] + p.Beta, 0f, 1f);
            Interlocked.Increment(ref CpuRuns);
        }

        public override IEnumerable<VkDispatch> EmitVulkan(VkGraphModel g, ExecNode n)
        {
            Interlocked.Increment(ref VulkanEmits);
            var p = (HardSigmoidP)n.P!;
            uint count = (uint)g.Plan.Values[n.Out[0]].ElementCount;
            // a core shader (owner null: the engine's own SPIR-V); extensions pass their assembly for their own
            yield return g.Dispatch(n, "hsig", [g.Bind(n.In[0]), g.Bind(n.Out[0])], VkGraphModel.PcOfF([count], p.Alpha, p.Beta), VkGraphModel.G(count));
        }
    }

    private sealed class SwapHardSigmoid : GraphPass
    {
        public override PassStage Stage => PassStage.Logical;

        public override void Run(PassContext ctx)
        {
            for (int i = 0; i < ctx.Nodes.Count; i++)
            {
                ExecNode n = ctx.Nodes[i];
                if (n.Op != ExecOp.HardSigmoid) continue;
                ctx.Nodes[i] = new ExecNode { Op = ExecOp.Ext, Ext = MyHardSigmoid.Instance, In = n.In, Out = n.Out, P = n.P, Name = n.Name };
            }
        }
    }

    private static OnnxModel Model()
    {
        var rng = new Random(7);
        float[] R(int n) => Enumerable.Range(0, n).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        // conv -> mul -> hardsigmoid -> conv
        byte[] m = new OnnxBuilder().Input("x", 1, 8, 12, 16)
            .Init("W1", R(16 * 8 * 9), 16, 8, 3, 3).Init("B1", R(16), 16)
            .Init("K", R(16), 1, 16, 1, 1)
            .Init("W2", R(8 * 16), 8, 16, 1, 1)
            .Node("Conv", ["x", "W1", "B1"], ["c1"], ("kernel_shape", new[] { 3, 3 }), ("pads", new[] { 1, 1, 1, 1 }))
            .Node("Mul", ["c1", "K"], ["m"])
            .Node("HardSigmoid", ["m"], ["r"], ("alpha", 0.25f), ("beta", 0.4f))
            .Node("Conv", ["r", "W2"], ["y"], ("kernel_shape", new[] { 1, 1 }))
            .Output("y").Build();
        return OnnxModel.Load(m);
    }

    private static float[] Run(InferenceSession s, float[] input)
    {
        input.CopyTo(s.Input);
        s.Run();
        return s.Output.ToArray();
    }

    [Fact]
    public void Cpu_ExtensionMatchesBuiltIn()
    {
        OnnxModel m = Model();
        int[] shape = [1, 8, 12, 16];
        float[] input = Enumerable.Range(0, 8 * 12 * 16).Select(i => MathF.Sin(i * 0.37f)).ToArray();
        float[] want = Run(CompiledModel.Compile(m, shape).CreateSession(), input);

        var opt = CompileOptions.Default with { Extensions = new EngineExtensions().Add(new SwapHardSigmoid()) };
        CompiledModel ext = CompiledModel.Compile(m, shape, opt);
        int before = MyHardSigmoid.Instance.CpuRuns;
        float[] got = Run(ext.CreateSession(), input);
        Assert.True(MyHardSigmoid.Instance.CpuRuns > before, "the extension op did not run");
        for (int i = 0; i < want.Length; i++) Assert.Equal(want[i], got[i], 5);
    }

    [SkippableFact]
    public void Vulkan_ExtensionMatchesBuiltIn()
    {
        VkBackend be;
        try { be = VkBackend.Default; _ = be.DeviceName; }
        catch (Exception ex) when (ex is not SkipException) { throw new SkipException("no Vulkan device"); }
        OnnxModel m = Model();
        int[] shape = [1, 8, 12, 16];
        float[] input = Enumerable.Range(0, 8 * 12 * 16).Select(i => MathF.Sin(i * 0.37f)).ToArray();
        int emits = MyHardSigmoid.Instance.VulkanEmits;
        using VkInferenceSession plain = be.NewSession(m, shape, VkBackend.PlanOptions);
        using VkInferenceSession ext = be.NewSession(m, shape,
            VkBackend.PlanOptions with { Extensions = new EngineExtensions().Add(new SwapHardSigmoid()) });
        input.CopyTo(plain.Input); plain.Run();
        input.CopyTo(ext.Input); ext.Run();
        Assert.True(MyHardSigmoid.Instance.VulkanEmits > emits, "the extension op was not emitted");
        float[] a = plain.Output.ToArray(), b = ext.Output.ToArray();
        for (int i = 0; i < a.Length; i++) Assert.True(MathF.Abs(a[i] - b[i]) < 1e-3f, $"{i}: {a[i]} vs {b[i]}");
    }
}
