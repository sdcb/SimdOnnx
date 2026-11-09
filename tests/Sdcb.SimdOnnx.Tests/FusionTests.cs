using System.Text;
using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Onnx;
using Xunit;

namespace Sdcb.SimdOnnx.Tests;

/// <summary>The built-in pattern fusions (<see cref="CompileOptions.FuseUpHead"/>,
/// <see cref="CompileOptions.FuseChannelShift"/>): where they fire, that they match the unfused
/// plan, and that they leave the PP-OCR plans untouched.</summary>
public class FusionTests
{
    private static readonly CompileOptions NoBuiltIns = new(Nhwc: true) { FuseUpHead = false, FuseChannelShift = false };

    private static int Count(ExecPlan p, string ext) => p.Nodes.Count(n => n.Ext?.Name == ext);

    [Theory]
    [InlineData(PlanTarget.Cpu)]
    [InlineData(PlanTarget.Vulkan)]
    public void GcdrNet_FusionsFire_AndToggleOff(PlanTarget target)
    {
        foreach (var (path, c) in new[] { (EngineTests.GcPath, 3), (EngineTests.DrPath, 6) })
        {
            OnnxModel m = OnnxModel.Load(path);
            ExecPlan on = GraphCompiler.Compile(m, [1, c, 96, 128], new CompileOptions(Nhwc: true, PhysicalIo: true, Target: target));
            ExecPlan off = GraphCompiler.Compile(m, [1, c, 96, 128],
                NoBuiltIns with { PhysicalIo = true, Target = target });
            string name = Path.GetFileName(path);
            Assert.True(Count(on, "UpHead") >= 1, $"{name}: no UpHead");
            Assert.True(Count(on, "ChannelShift") >= 1, $"{name}: no ChannelShift");
            Assert.Equal(0, Count(off, "UpHead") + Count(off, "ChannelShift"));
            Assert.True(on.Nodes.Length < off.Nodes.Length);
        }
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void GcdrNet_FusedMatchesUnfused(bool nhwc)
    {
        // NCHW plans get ChannelShift only (UpHead is NHWC-only)
        foreach (var (path, tag, name) in new[] { (EngineTests.GcPath, "pad", "input"), (EngineTests.DrPath, "pad", "dr_in") })
        {
            float[] input = Golden.F32(tag, name);
            int[] shape = Golden.ShapeOf(tag, name);
            OnnxModel m = OnnxModel.Load(path);
            float[] Run(CompileOptions opt)
            {
                using var s = new InferenceSession(GraphCompiler.Compile(m, shape, opt));
                input.CopyTo(s.Input);
                s.Run();
                return s.Output.ToArray();
            }
            var (max, rmse, _) = Golden.Diff(Run(new CompileOptions(nhwc)), Run(NoBuiltIns with { Nhwc = nhwc }));
            Assert.True(max < 1e-5f, $"{Path.GetFileName(path)} nhwc={nhwc}: fused vs unfused max {max} rmse {rmse}");
        }
    }

    [Fact]
    public void ChannelShift_CountsPerNet()
    {
        // if a model export change breaks the shift pattern, this fails loudly
        ExecPlan dr = GraphCompiler.Compile(OnnxModel.Load(EngineTests.DrPath), [1, 6, 96, 128]);
        Assert.Equal(8, Count(dr, "ChannelShift"));
        ExecPlan gc = GraphCompiler.Compile(OnnxModel.Load(EngineTests.GcPath), [1, 3, 96, 128]);
        Assert.True(Count(gc, "ChannelShift") >= 4, "gc plan lost its ChannelShift fusion");
    }

    [Fact]
    public void ChannelShift_OffsetBeyondWidth_Zeros()
    {
        // |off| >= w must produce a fully zeroed group, not a negative Slice
        float[] src = Enumerable.Repeat(1f, 2 * 4 * 5).ToArray();   // c=2,h=4,w=5
        float[] dst = Enumerable.Repeat(-1f, src.Length).ToArray();
        var p = new ShiftP { Axis = 3, Offsets = [5], GroupEnds = [2] };
        Kernels.FusedKernels.ChannelShiftNchw(src, [1, 2, 4, 5], dst, [1, 2, 4, 5], p);
        Assert.All(dst, v => Assert.Equal(0f, v));
    }

    [Fact]
    public void MemFusions_BitIdentical_AndShrinkArena()
    {
        // conv+maxpool and UpHead skip recompute must be present in CPU NHWC plans, must
        // not change a single bit of the output, and must shrink the arena
        foreach (var (path, c) in new[] { (EngineTests.GcPath, 3), (EngineTests.DrPath, 6) })
        {
            OnnxModel m = OnnxModel.Load(path);
            int[] shape = [1, c, 96, 128];
            var rnd = new Random(7);
            float[] input = Enumerable.Range(0, c * 96 * 128).Select(_ => (float)rnd.NextDouble()).ToArray();
            (float[] y, ExecPlan p) Run(bool fuse)
            {
                ExecPlan p = GraphCompiler.Compile(m, shape, new CompileOptions(Nhwc: true, PhysicalIo: true, MemFuse: fuse));
                using var s = new InferenceSession(p);
                input.CopyTo(s.Input); s.Run();
                return (s.Output.ToArray(), p);
            }
            var (fy, fp) = Run(true);
            var (uy, up) = Run(false);
            string name = Path.GetFileName(path);
            Assert.Equal(3, fp.Nodes.Count(n => n.P is ConvP { Pool: not null }));
            Assert.DoesNotContain(fp.Nodes, n => n.Op == ExecOp.MaxPool);
            Assert.Contains(fp.Nodes, n => n.P is UpHeadP { SkipConv: not null });
            Assert.True(fp.ArenaFloats * 2 < up.ArenaFloats, $"{name}: arena {fp.ArenaFloats} vs unfused {up.ArenaFloats}");
            Assert.True(fy.AsSpan().SequenceEqual(uy), $"{name}: fused output differs");
        }
    }

    [Fact]
    public void MemFusions_GpuPlans_StemOnly()
    {
        // Vulkan plans get the stem conv+pool and the UpHead recompute (small-ic shaders) only
        foreach (var (path, c) in new[] { (EngineTests.GcPath, 3), (EngineTests.DrPath, 6) })
        {
            ExecPlan p = GraphCompiler.Compile(OnnxModel.Load(path), [1, c, 96, 128],
                new CompileOptions(Nhwc: true, PhysicalIo: true, Target: PlanTarget.Vulkan));
            Assert.Single(p.Nodes, n => n.P is ConvP { Pool: not null });
            Assert.Equal(2, p.Nodes.Count(n => n.Op == ExecOp.MaxPool));
            Assert.Contains(p.Nodes, n => n.P is UpHeadP { SkipConv: not null, SkipW: >= 0 });
        }
    }

    /// <summary>SimdPaddleOCR's checkout (sibling repo locally, ./SimdPaddleOCR in CI) or SIMDONNX_PADDLE_MODELS.</summary>
    private static string? PaddleModelDir()
    {
        if (Environment.GetEnvironmentVariable("SIMDONNX_PADDLE_MODELS") is { Length: > 0 } env) return env;
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d != null; d = d.Parent)
        {
            string cand = Path.Combine(d.FullName, "SimdPaddleOCR", "models");
            if (File.Exists(Path.Combine(cand, "det.onnx"))) return cand;
        }
        return null;
    }

    /// <summary>Everything a backend reads from a plan: nodes, their operands, layouts and arena offsets.</summary>
    private static string Signature(ExecPlan p)
    {
        var sb = new StringBuilder().Append("arena ").Append(p.ArenaFloats).Append('\n');
        string V(int v) => v < 0 ? "-" : $"v{v}/b{p.Values[v].Buf}@{(p.Values[v].Buf >= 0 ? p.BufOffset[p.Values[v].Buf] : -1)}[{string.Join('x', p.Values[v].Shape)}]{(p.Values[v].Layout is { } l ? string.Join(',', l) : "")}";
        foreach (ExecNode n in p.Nodes)
            sb.Append(n.Op).Append(' ').Append(n.Ext?.Name).Append(' ')
              .Append(string.Join(' ', n.In.Select(V))).Append(" -> ").Append(string.Join(' ', n.Out.Select(V)))
              .Append(' ').Append(n.P?.GetType().Name).Append('\n');
        return sb.ToString();
    }

    [SkippableTheory]
    [InlineData(PlanTarget.Cpu)]
    [InlineData(PlanTarget.Vulkan)]
    public void PaddleOcr_PlansUnchangedByBuiltInFusions(PlanTarget target)
    {
        string? dir = PaddleModelDir();
        Skip.If(dir is null, "SimdPaddleOCR models not found (set SIMDONNX_PADDLE_MODELS)");
        string[] files = Directory.GetFiles(dir!, "*.onnx");
        Assert.NotEmpty(files);
        foreach (string f in files)
        {
            string n = Path.GetFileNameWithoutExtension(f);
            int[] shape = n.Contains("det") ? [1, 3, 960, 736] : n.Contains("cls") ? [1, 3, 48, 192] : [1, 3, 48, 320];
            OnnxModel m = OnnxModel.Load(f);
            var opt = new CompileOptions(Nhwc: true, Target: target);
            string on = Signature(GraphCompiler.Compile(m, shape, opt));
            string off = Signature(GraphCompiler.Compile(m, shape, opt with { FuseUpHead = false, FuseChannelShift = false }));
            Assert.True(on == off, $"{n}: the built-in fusions changed the {target} plan");
        }
    }
}
