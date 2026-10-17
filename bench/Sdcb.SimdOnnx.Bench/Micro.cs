using System.Diagnostics;
using System.Runtime.Intrinsics;
using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Kernels;

namespace Sdcb.SimdOnnx.Bench;

/// <summary>Kernel micro-benchmarks: <c>--micro ew|dw|conv ...</c> (single thread).</summary>
internal static class Micro
{
    public static void Run(string[] args)
    {
        switch (args[0])
        {
            case "ew":
            {
                int[] s = [1, 6, 80, 96];
                int n = s.Aggregate(1, (a, b) => a * b);
                float[] a = Rand(n, 1), b = Rand(n, 2), d = new float[n];
                Time("ew same-shape add 46080", () => Ops.Elementwise(a, s, b, s, d, s, EwOp.Add));
                int[] cs = [1, 1, 1, 96];
                float[] c = Rand(96, 3);
                Time("ew channel mul 46080", () => Ops.Elementwise(a, s, c, cs, d, s, EwOp.Mul));
                foreach (ActivationKind k in Enum.GetValues<ActivationKind>())
                    if (k != ActivationKind.None)
                        Time($"act {k} 46080", () => Ops.Activation(a, d, k));
                Time("relu 46080", () => Ops.Relu(a, d));
                                float[] big = Rand(40 * 6906, 5), outb = new float[big.Length];
                Time("exp128 276k", () => Exp128(big, outb), 200);
                Time("exp256 276k", () => Exp256(big, outb), 200);
                Time("softmax(noalloc) 40x6906", () => Ops.Softmax(big, outb, 6906), 200);
                break;
            }
            case "calib":
            {
                // dependent multiply chain: 1e9 × imul (3-cycle latency) → GHz estimate
                long x = 1;
                long t0 = Stopwatch.GetTimestamp();
                for (int i = 0; i < 1_000_000_000; i++) x = x * 3;
                double s = Stopwatch.GetElapsedTime(t0).TotalSeconds;
                Console.WriteLine($"calib: {s:F3}s → ~{3.0 / s:F2} GHz effective ({x & 1})");
                break;
            }
            case "dwk":
                unsafe
                {
                    int c = 64, taps = 25;
                    float* src = (float*)System.Runtime.InteropServices.NativeMemory.AlignedAlloc(64 * 1024 * 4, 64);
                    float* w = (float*)System.Runtime.InteropServices.NativeMemory.AlignedAlloc((nuint)(taps * c * 4), 64);
                    float* b = (float*)System.Runtime.InteropServices.NativeMemory.AlignedAlloc((nuint)(c * 4), 64);
                    float* o = (float*)System.Runtime.InteropServices.NativeMemory.AlignedAlloc((nuint)(4 * c * 4), 64);
                    for (int i = 0; i < 64 * 1024; i++) src[i] = (i % 7) * 0.1f;
                    for (int i = 0; i < taps * c; i++) w[i] = (i % 5) * 0.1f;
                    for (int i = 0; i < c; i++) b[i] = 0.5f;
                    float** ptrs = stackalloc float*[taps * 4];
                    for (int p = 0; p < 4; p++)
                        for (int t = 0; t < taps; t++) ptrs[p * taps + t] = src + ((t / 5) * 80 + p + t % 5) * c;
                    const int N = 200000;
                    for (int i = 0; i < 1000; i++) Nhwc.DwBlockAvx(ptrs, taps, c, w, b, o, null, 4, ActivationKind.None);
                    long t0 = Stopwatch.GetTimestamp();
                    for (int i = 0; i < N; i++) Nhwc.DwBlockAvx(ptrs, taps, c, w, b, o, null, 4, ActivationKind.None);
                    double ns = Stopwatch.GetElapsedTime(t0).TotalMilliseconds * 1e6 / N;
                    Console.WriteLine($"DwBlockAvx 4px x 64ch x 25 taps: {ns:F1} ns/call  ({ns * 4.4 / (taps * 4 * 64 / 8.0):F2} cycles per vector FMA @4.4GHz)");
                }
                break;
            case "compile":
            {
                // per-shape plan compile cost after the first (packs cached): micro compile <model.onnx> CxHxW
                var model = Sdcb.SimdOnnx.Onnx.OnnxModel.Load(args[1]);
                int[] chw = args[2].Split('x').Select(int.Parse).ToArray();
                var cm = new CpuModel(model, new CompileOptions(Nhwc: true, PhysicalIo: true, LogicalOutput: true));
                foreach (int n in new[] { 8, 16, 24, 32, 40, 1, 2, 3 })
                {
                    long al = GC.GetAllocatedBytesForCurrentThread();
                    long t0 = Stopwatch.GetTimestamp();
                    ExecPlan plan = cm.Plan(args.Length > 3 ? [1, chw[0], chw[1] + 32 * n, chw[2]] : [n, .. chw]);
                    Console.Write($"alloc {(GC.GetAllocatedBytesForCurrentThread() - al) / 1024.0,8:F0}KB  ");
                    double ms = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
                    t0 = Stopwatch.GetTimestamp();
                    using (var s = new InferenceSession(plan)) { }
                    Console.WriteLine($"N={n,-3} plan {ms,7:F2} ms   session {Stopwatch.GetElapsedTime(t0).TotalMilliseconds,6:F2} ms  nodes={plan.Nodes.Length}");
                }
                break;
            }
            case "vkcompile":
            {
                // Vulkan-target plan compile cost per new shape: micro vkcompile <model.onnx> NxCxHxW [count]
                var model = Sdcb.SimdOnnx.Onnx.OnnxModel.Load(args[1]);
                int[] s0 = args[2].Split('x').Select(int.Parse).ToArray();
                int count = args.Length > 3 ? int.Parse(args[3]) : 200;
                var opt = new CompileOptions(Nhwc: true, PhysicalIo: true, LogicalOutput: true, Target: PlanTarget.Vulkan,
                    OutputName: args.Length > 4 ? args[4] : null);
                GraphCompiler.Compile(model, s0, opt);
                long t0 = Stopwatch.GetTimestamp();
                for (int i = 0; i < count; i++) GraphCompiler.Compile(model, [s0[0], s0[1], s0[2], s0[3] + 32 * (i % 40)], opt);
                Console.WriteLine($"compile avg {Stopwatch.GetElapsedTime(t0).TotalMilliseconds / count:F3} ms");
                long al = GC.GetTotalAllocatedBytes(true);
                for (int i = 0; i < 20; i++) GraphCompiler.Compile(model, [s0[0], s0[1], s0[2], s0[3] + 32 * (i % 40)], opt);
                Console.WriteLine($"alloc/compile {(GC.GetTotalAllocatedBytes(true) - al) / 20 / 1024.0:F0} KB");
                if (Sdcb.SimdOnnx.Backends.Vulkan.VkBackend.Default is var vb && vb.GraphFor(model, s0, opt with { PhysicalIo = false, LogicalOutput = false }) is not null)
                {
                    al = GC.GetTotalAllocatedBytes(true);
                    t0 = Stopwatch.GetTimestamp();
                    for (int i = 0; i < 20; i++) vb.GraphFor(model, [s0[0], s0[1], s0[2] + 32, s0[3] + 32 * (i % 40)], opt with { PhysicalIo = false, LogicalOutput = false });
                    Console.WriteLine($"vkgraph (compile+emit) avg {Stopwatch.GetElapsedTime(t0).TotalMilliseconds / 20:F3} ms, alloc {(GC.GetTotalAllocatedBytes(true) - al) / 20 / 1024.0:F0} KB");
                }
                break;
            }
            case "planhash":
            {
                // fingerprint of compiled plans (nodes, params, layouts, arena), to prove compiler
                // refactors plan-neutral: micro planhash <model.onnx> NxCxHxW[,NxCxHxW...]
                var model = Sdcb.SimdOnnx.Onnx.OnnxModel.Load(args[1]);
                foreach (string sh in args[2].Split(','))
                {
                    int[] s0 = sh.Split('x').Select(int.Parse).ToArray();
                    foreach (PlanTarget target in new[] { PlanTarget.Cpu, PlanTarget.Vulkan, PlanTarget.Metal })
                        foreach (bool phys in new[] { false, true })
                        {
                            var opt = new CompileOptions(Nhwc: true, PhysicalIo: phys, LogicalOutput: phys, Target: target);
                            ExecPlan p = GraphCompiler.Compile(model, s0, opt);
                            var sb = new System.Text.StringBuilder();
                            string A(int[]? a) => a is null ? "-" : string.Join(",", a);
                            foreach (ExecNode n in p.Nodes)
                            {
                                sb.Append(n.Op).Append('|').Append(A(n.In)).Append('|').Append(A(n.Out)).Append('|');
                                if (n.P is not null)
                                    foreach (var f in n.P.GetType().GetFields())
                                    {
                                        object? v = f.GetValue(n.P);
                                        sb.Append(f.Name).Append('=').Append(v switch
                                        {
                                            int[] ia => A(ia), float[] fa => string.Join(",", fa), long[] la => string.Join(",", la),
                                            PoolP pp => $"pool({A(pp.Kernel)};{A(pp.Strides)};{A(pp.Pads)};{pp.FuseRelu})",
                                            _ => v?.ToString(),
                                        }).Append(';');
                                    }
                                sb.Append(';');
                            }
                            foreach (CValue v in p.Values)
                                sb.Append(A(v.Shape)).Append('|').Append(v.Buf).Append('|').Append(A(v.Layout)).Append('|').Append(v.AliasOf)
                                    .Append('|').Append(v.Floats?.Length ?? -1).Append('|').Append(v.Floats is { Length: > 0 } fl ? fl.Sum() : 0).Append(';');
                            sb.Append(A(p.BufSize)).Append('|').Append(A(p.BufOffset)).Append('|').Append(p.ArenaFloats)
                                .Append('|').Append(p.InputValue).Append('|').Append(p.OutputValue);
                            byte[] h = System.Security.Cryptography.SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(sb.ToString()));
                            Console.WriteLine($"{sh} {target} phys={phys} nodes={p.Nodes.Length} {Convert.ToHexString(h)[..16]}");
                        }
                }
                break;
            }
            case "multi":
            {
                // multi-input model against a reference output (tools/ort_ref_multi.py):
                // micro multi <model.onnx> <shape0;shape1;..> <in0.f32,in1.f32,..> <ref.f32> <cpu,vulkan,metal> [iters] [threads]
                var model = Sdcb.SimdOnnx.Onnx.OnnxModel.Load(args[1]);
                int[][] shapes = args[2].Split(';').Select(s => s.Split('x').Select(int.Parse).ToArray()).ToArray();
                float[][] ins = args[3].Split(',').Select(f => System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(f)).ToArray()).ToArray();
                float[] want = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(args[4])).ToArray();
                int iters = args.Length > 6 ? int.Parse(args[6]) : 10;
                int threads = args.Length > 7 ? int.Parse(args[7]) : Environment.ProcessorCount;
                var opt = new CompileOptions(Nhwc: true, InputShapes: shapes.Skip(1).ToArray());
                foreach (string engine in args[5].Split(','))
                {
                    var sw = Stopwatch.StartNew();
                    (Func<int, Span<float>> input, Action run, Func<float[]> output, IDisposable s) = engine switch
                    {
                        "cpu" => Cpu(),
                        "vulkan" => Vk(),
                        "metal" => Mtl(),
                        _ => throw new ArgumentException(engine),
                    };
                    double setup = sw.Elapsed.TotalMilliseconds;
                    for (int k = 0; k < ins.Length; k++) ins[k].CopyTo(input(k));
                    run();
                    float[] got = output();
                    var times = new List<double>();
                    for (int i = 0; i < iters; i++)
                    {
                        for (int k = 0; k < ins.Length; k++) ins[k].CopyTo(input(k));
                        long t0 = Stopwatch.GetTimestamp();
                        run();
                        times.Add(Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
                    }
                    double maxd = 0, sum = 0, se = 0;
                    for (int i = 0; i < want.Length; i++)
                    {
                        double d = Math.Abs(got[i] - want[i]);
                        maxd = Math.Max(maxd, d); sum += d; se += d * d;
                    }
                    double mse = se / want.Length, psnr = 10 * Math.Log10(1.0 / Math.Max(mse, 1e-20));
                    times.Sort();
                    Console.WriteLine($"{engine,-7} setup {setup,7:F1} ms  median {times[times.Count / 2],8:F2} ms  min {times[0],8:F2} ms  " +
                        $"max|d| {maxd:E2} mean|d| {sum / want.Length:E2} psnr-vs-ref {psnr:F1} dB  peakWS {Process.GetCurrentProcess().PeakWorkingSet64 / 1048576} MB");
                    s.Dispose();
                }
                break;

                (Func<int, Span<float>>, Action, Func<float[]>, IDisposable) Cpu()
                {
                    var s = new InferenceSession(GraphCompiler.Compile(model, shapes[0], opt)) { Threads = threads };
                    return (k => s.GetInput(k), s.Run, () => s.Output.ToArray(), s);
                }
                (Func<int, Span<float>>, Action, Func<float[]>, IDisposable) Vk()
                {
                    var s = Sdcb.SimdOnnx.Backends.Vulkan.VkBackend.Default.NewSession(model, shapes[0], opt);
                    return (k => s.GetInput(k), s.Run, () => s.Output.ToArray(), s);
                }
                (Func<int, Span<float>>, Action, Func<float[]>, IDisposable) Mtl()
                {
                    var s = Sdcb.SimdOnnx.Backends.Metal.MetalBackend.Default.NewSession(model, shapes[0], opt);
                    return (k => s.GetInput(k), s.Run, () => s.Output.ToArray(), s);
                }
            }
            case "runalloc":
            {
                // managed bytes allocated per CPU run at a fixed shape: micro runalloc <model.onnx> NxCxHxW [threads]
                var model = Sdcb.SimdOnnx.Onnx.OnnxModel.Load(args[1]);
                int[] s0 = args[2].Split('x').Select(int.Parse).ToArray();
                var cm = new CpuModel(model, new CompileOptions(Nhwc: true, PhysicalIo: true, LogicalOutput: true));
                using var s = new InferenceSession(cm.Plan(s0));
                s.Threads = args.Length > 3 ? int.Parse(args[3]) : 4;
                for (int i = 0; i < 3; i++) s.Run();
                long al = GC.GetTotalAllocatedBytes(true);
                long t0 = Stopwatch.GetTimestamp();
                for (int i = 0; i < 20; i++) s.Run();
                Console.WriteLine($"run {Stopwatch.GetElapsedTime(t0).TotalMilliseconds / 20:F2} ms, alloc {(GC.GetTotalAllocatedBytes(true) - al) / 20 / 1024.0:F0} KB per run, {s.Plan.Nodes.Length} nodes");
                break;
            }
            case "vkemit":
            {
                // Vulkan graph emission cost alone (plan compiled once): micro vkemit <model.onnx> NxCxHxW [count] [output]
                var model = Sdcb.SimdOnnx.Onnx.OnnxModel.Load(args[1]);
                int[] s0 = args[2].Split('x').Select(int.Parse).ToArray();
                int count = args.Length > 3 ? int.Parse(args[3]) : 200;
                var be = Sdcb.SimdOnnx.Backends.Vulkan.VkBackend.Default;
                var opt = Sdcb.SimdOnnx.Backends.Vulkan.VkBackend.PlanOptions with { Target = PlanTarget.Vulkan, OutputName = args.Length > 4 ? args[4] : null };
                ExecPlan plan = GraphCompiler.Compile(model, s0, opt);
                // one live graph keeps the model's const pool (as the backend cache does)
                var keep = new Sdcb.SimdOnnx.Backends.Vulkan.VkGraphModel(be.Dev, plan, be.F16, constKey: model);
                long al = GC.GetTotalAllocatedBytes(true);
                long t0 = Stopwatch.GetTimestamp();
                for (int i = 0; i < count; i++)
                    new Sdcb.SimdOnnx.Backends.Vulkan.VkGraphModel(be.Dev, plan, be.F16, constKey: model).Release();
                Console.WriteLine($"emit avg {Stopwatch.GetElapsedTime(t0).TotalMilliseconds / count:F3} ms, alloc {(GC.GetTotalAllocatedBytes(true) - al) / count / 1024.0:F0} KB, {plan.Nodes.Length} nodes");
                keep.Release();
                break;
            }
            case "plansize":
            {
                // retained bytes per cached plan: micro plansize <model.onnx> NxCxHxW
                var model = Sdcb.SimdOnnx.Onnx.OnnxModel.Load(args[1]);
                int[] s0 = args[2].Split('x').Select(int.Parse).ToArray();
                var popt = new CompileOptions(Nhwc: true, PhysicalIo: true, LogicalOutput: true, Target: Environment.GetEnvironmentVariable("PLAN_GPU") == "1" ? PlanTarget.Vulkan : PlanTarget.Cpu);
                var cm = new CpuModel(model, popt);
                cm.Plan(s0);
                long m0 = GC.GetTotalMemory(true);
                var keep = new List<ExecPlan>();
                int step = args.Length > 3 ? int.Parse(args[3]) : 8;
                for (int i = 1; i <= 16; i++) keep.Add(cm.Plan([s0[0], s0[1], s0[2], s0[3] + step * i]));
                long m1 = GC.GetTotalMemory(true);
                Console.WriteLine($"retained per plan: {(m1 - m0) / 16 / 1024.0:F1} KB");
                ExecPlan p = keep[^1];
                long floats = 0;
                var seen = new HashSet<float[]>(ReferenceEqualityComparer.Instance);
                foreach (CValue v in p.Values)
                    if (v.Floats is { } f && seen.Add(f)) floats += f.Length;
                Console.WriteLine($"values={p.Values.Length} nodes={p.Nodes.Length} distinct const floats={floats * 4 / 1024.0:F1} KB");
                foreach (CValue v in p.Values.Where(v => v.Floats is { Length: > 4096 }).OrderByDescending(v => v.Floats!.Length).Take(8))
                    Console.WriteLine($"  {v.Name,-40} [{string.Join('x', v.Shape)}] {v.Floats!.Length * 4 / 1024.0:F1} KB shared={keep[0].Values.Any(u => ReferenceEquals(u.Floats, v.Floats))}");
                break;
            }
            case "packsize":
            {
                // packed-weight bytes vs raw weights: micro packsize <model.onnx> NxCxHxW
                var model = Sdcb.SimdOnnx.Onnx.OnnxModel.Load(args[1]);
                int[] s0 = args[2].Split('x').Select(int.Parse).ToArray();
                long gc0 = GC.GetTotalMemory(true);
                var cm = new CpuModel(model, new CompileOptions(Nhwc: true, PhysicalIo: true, LogicalOutput: true));
                ExecPlan p = cm.Plan(s0);
                long gc1 = GC.GetTotalMemory(true);
                long w = 0, u = 0, raw = 0;
                foreach (ExecNode n in p.Nodes)
                    if (n.P is ConvP { KernelPlan: NhwcConvPlan kp })
                    {
                        w += kp.W.Length; u += kp.WinoU?.Length ?? 0;
                        raw += p.Values[n.In[1]].Floats?.Length ?? 0;
                    }
                Console.WriteLine($"conv raw={raw * 4 / 1048576.0:F2}MB packedW={w * 4 / 1048576.0:F2}MB winoU={u * 4 / 1048576.0:F2}MB  model+plan managed={(gc1 - gc0) / 1048576.0:F2}MB");
                GC.KeepAlive(cm);
                break;
            }
            case "dw":
            {
                foreach ((int h, int w, int c, int k, int s) in new[] { (5, 80, 64, 5, 1), (40, 80, 8, 3, 1), (20, 80, 16, 3, 1), (6, 80, 48, 3, 1) })
                {
                    int pad = k / 2, oh = (h + 2 * pad - k) / s + 1, ow = (w + 2 * pad - k) / s + 1;
                    float[] src = Rand(h * w * c, 1), wt = Rand(c * k * k, 2), bias = Rand(c, 3), dst = new float[oh * ow * c];
                    var p = new ConvP { Pads = [pad, pad, pad, pad], Strides = [s, s], Dilations = [1, 1], Kernel = [k, k], Group = c, Act = Enum.Parse<ActivationKind>(Environment.GetEnvironmentVariable("MICRO_ACT") ?? "HardSwish") };
                    NhwcConvPlan kp = Nhwc.Pack(wt, [c, 1, k, k], bias, c, c);
                    double macs = (double)oh * ow * c * k * k;
                    Time($"dw{k}x{k}s{s} {h}x{w}x{c} ({macs / 1e6:F2}M MAC)", () => Nhwc.Conv(src, h, w, c, dst, oh, ow, c, p, kp, default));
                }
                break;
            }
        }
    }

    private static unsafe void Exp128(float[] s, float[] d)
    {
        fixed (float* ps = s, pd = d)
            for (int i = 0; i + 4 <= s.Length; i += 4)
                SimdMath.ExpCore(System.Runtime.Intrinsics.Vector128.Load(ps + i)).Store(pd + i);
    }

    private static unsafe void Exp256(float[] s, float[] d)
    {
        fixed (float* ps = s, pd = d)
            for (int i = 0; i + 8 <= s.Length; i += 8)
                SimdMath.ExpCore(System.Runtime.Intrinsics.Vector256.Load(ps + i)).Store(pd + i);
    }

    internal static float[] Rand(int n, int seed)
    {
        var r = new Random(seed);
        return Enumerable.Range(0, n).Select(_ => (float)r.NextDouble() * 2 - 1).ToArray();
    }

    internal static void Time(string name, Action a, int iters = 2000)
    {
        for (int i = 0; i < 50; i++) a();
        long t0 = Stopwatch.GetTimestamp();
        for (int i = 0; i < iters; i++) a();
        Console.WriteLine($"{name,-40} {Stopwatch.GetElapsedTime(t0).TotalMilliseconds * 1000 / iters,8:F2} us");
    }
}
