using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Onnx;

namespace Sdcb.SimdOnnx.Backends.Metal;

/// <summary>
/// Metal backend root: the device, the per-(model, shape) graph cache, and
/// factories for inference sessions and fused enhancer pipelines. Thread-safe;
/// dispose with the process or when the backend is torn down.
/// </summary>
public sealed class MetalBackend : IDisposable
{
    internal readonly MtlDevice Dev;
    private readonly Dictionary<(OnnxModel m, string shape, CompileOptions opt), MtlGraphModel> _graphs = new();
    private readonly object _gate = new();

    private static readonly Lazy<MetalBackend> s_default = new(() => new MetalBackend());
    public static MetalBackend Default => s_default.Value;

    /// <summary>The device's capabilities.</summary>
    public GpuDeviceInfo Device { get; }

    /// <summary><see cref="Default"/>, or false off macOS or when no Metal device exists.</summary>
    public static bool TryGetDefault([System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out MetalBackend? backend)
    {
        try { backend = OperatingSystem.IsMacOS() ? Default : null; }
        catch (Exception ex) when (ex is not OutOfMemoryException) { backend = null; }
        return backend is not null;
    }

    public static CompileOptions PlanOptions => new(Nhwc: true, PhysicalIo: false, Target: PlanTarget.Metal);
    public static CompileOptions PipeOptions => new(Nhwc: true, PhysicalIo: true, Target: PlanTarget.Metal);

    internal MetalBackend()
    {
        Dev = MtlDevice.Probe() ?? throw new PlatformNotSupportedException("Metal: no device");
        // Apple GPUs: fp16 storage, 32-lane simdgroups, unified memory
        Device = new GpuDeviceInfo(Dev.Name, 0, Fp16Storage: true, CooperativeMatrixFp16: false,
            SubgroupSizeMin: 32, SubgroupSizeMax: 32, UnifiedMemory: true, DeviceLocalBytes: 0);
        if (Environment.GetEnvironmentVariable("GCDRNET_MTL_VERBOSE") == "1")
            Console.Error.WriteLine($"[metal] device: {Dev.Name}");
    }

    /// <summary>New inference session for a model and input shape (non-physical I/O plan).</summary>
    public MtlInferenceSession NewSession(OnnxModel m, int[] shape) => NewSession(m, shape, (EngineExtensions?)null);

    internal MtlInferenceSession NewSession(OnnxModel m, int[] shape, EngineExtensions? ext)
    {
        MtlGraphModel g = AcquireGraph(m, shape, physIo: false, ext);   // caller ref: later evictions can't free it under us
        try { return new MtlInferenceSession(Dev, g.Plan, g); }
        finally { g.Release(); }
    }

    /// <summary>The cached graph for (model, shape); acquires a reference for the caller.</summary>
    internal MtlGraphModel GraphFor(OnnxModel m, int[] shape, bool physIo = false, EngineExtensions? ext = null) =>
        GraphFor(m, shape, (physIo ? PipeOptions : PlanOptions) with { Extensions = ext });

    /// <summary>The cached graph for (model, shape, options); acquires a reference for the caller.</summary>
    internal MtlGraphModel GraphFor(OnnxModel m, int[] shape, CompileOptions opt)
    {
        opt = opt with { Target = PlanTarget.Metal, Nhwc = true };
        var key = (m, string.Join('x', shape), opt);
        lock (_gate)
        {
            if (_graphs.TryGetValue(key, out var gg))
            {
                gg.Acquire();
                return gg;
            }
        }
        ExecPlan plan = GraphCompiler.Compile(m, shape, opt);
        var g = new MtlGraphModel(Dev, plan, constKey: m);
        lock (_gate)
        {
            if (_graphs.TryGetValue(key, out var won)) { g.Release(); won.Acquire(); return won; }
            _graphs[key] = g;
            // bound the cache: graphs share the model's const store, so eviction only
            // frees the small per-size plan+dispatch list
            while (_graphs.Count > MaxGraphs)
            {
                var victim = _graphs.Keys.First(k => !k.Equals(key));
                _graphs.Remove(victim, out var old);
                old?.Release();
            }
            g.Acquire();
            return g;
        }
    }
    private const int MaxGraphs = 32;

    /// <summary>Alias: acquires the graph for the caller (caller must Release).</summary>
    internal MtlGraphModel AcquireGraph(OnnxModel m, int[] shape, bool physIo, EngineExtensions? ext = null)
        => GraphFor(m, shape, physIo, ext);

    /// <summary>Session over the graph for <paramref name="shape"/> under explicit plan options.</summary>
    /// <summary>The device's name.</summary>
    public string DeviceName => Dev.Name;

    /// <summary>A session over the graph of <paramref name="m"/> at <paramref name="shape"/> under explicit options.</summary>
    public MtlInferenceSession NewSession(OnnxModel m, int[] shape, CompileOptions opt)
    {
        MtlGraphModel g = GraphFor(m, shape, opt);
        try { return new MtlInferenceSession(Dev, g.Plan, g); }
        finally { g.Release(); }
    }

    public void Dispose() => Dev.Dispose();
}
