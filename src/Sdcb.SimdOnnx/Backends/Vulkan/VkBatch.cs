using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Onnx;

namespace Sdcb.SimdOnnx.Backends.Vulkan;

/// <summary>
/// Device buffers (activation arena, input and output staging) shared by sessions that run one at a
/// time, e.g. one worker's sessions over many input shapes: each session then costs only its command
/// buffer. The buffers grow to the largest session; growing makes the earlier sessions stale
/// (<see cref="VkInferenceSession.IsStale"/>). Not thread-safe. Create with
/// <see cref="VkBackend.CreateBufferPool"/>.
/// </summary>
public sealed class VkBufferPool : IDisposable
{
    internal VkBufferPool(VkBackend backend)
    {
        Backend = backend;
        Buffers = new VkSessionBuffers(backend.Dev);
    }

    /// <summary>The backend whose device owns the buffers.</summary>
    public VkBackend Backend { get; }
    internal VkSessionBuffers Buffers { get; }

    public void Dispose() => Buffers.Dispose();
}

/// <summary>
/// Many graph runs (units) of possibly different shapes and models in one submission: inputs are
/// written straight into mapped staging, the units' dispatches are recorded level-interleaved so
/// independent units overlap on the device, and the units are split into waves so the host can
/// consume the first wave's outputs while later waves still run. Not thread-safe. Create with
/// <see cref="VkBackend.CreateBatch"/>; reuse across calls (<see cref="Clear"/>).
/// </summary>
/// <remarks>Preview API: the wave semantics may still change.</remarks>
public sealed class VkBatch : IDisposable
{
    private readonly VkUnitRunner _runner;

    internal VkBatch(VkBackend backend)
    {
        Backend = backend;
        _runner = new VkUnitRunner(backend.Dev);
    }

    /// <summary>The backend this batch records on. After a device loss it is dead: create a new batch
    /// on <see cref="VkBackend.Default"/>.</summary>
    public VkBackend Backend { get; }

    /// <summary>Units added since the last <see cref="Clear"/>.</summary>
    public int Count => _runner.Count;

    /// <summary>Waves of the current units (the last, open one included once units were added).</summary>
    public int Waves => _runner.Waves + (OpenUnits ? 1 : 0);

    // units after the last EndWave form the final wave (Submit closes it)
    private bool OpenUnits => _runner.Count > (_runner.Waves == 0 ? 0 : _runner.WaveEnd(_runner.Waves - 1));

    /// <summary>Drops the previous run's units (it must have been waited for).</summary>
    public void Clear() => _runner.Clear();

    /// <summary>Adds a run of <paramref name="model"/> at <paramref name="shape"/> (graphs are cached by
    /// the backend, as for sessions) and returns its index.</summary>
    public int Add(OnnxModel model, int[] shape, CompileOptions options)
    {
        VkGraphModel g = Backend.AcquireGraph(model, shape, options);
        try { return _runner.Add(g); }
        finally { g.Release(); }
    }

    /// <summary>Ends the current wave: the units added since the previous one.</summary>
    public void EndWave() => _runner.Cut();

    /// <summary>The wave unit <paramref name="unit"/> belongs to.</summary>
    public int WaveOf(int unit)
    {
        if ((uint)unit >= (uint)_runner.Count) throw new ArgumentOutOfRangeException(nameof(unit));
        return OpenUnits && unit >= (_runner.Waves == 0 ? 0 : _runner.WaveEnd(_runner.Waves - 1)) ? _runner.Waves : _runner.WaveOf(unit);
    }

    /// <summary>The unit's input: the mapped fp32 staging itself (write before <see cref="Submit"/>).</summary>
    public Span<float> Input(int unit) => _runner.Input(unit);

    /// <summary>The unit's output shape.</summary>
    public int[] OutputShape(int unit) => _runner.OutputShape(unit);

    /// <summary>The unit's output once its wave was waited for (valid until the next run).</summary>
    public ReadOnlySpan<float> Output(int unit) => _runner.Output(unit);

    /// <summary>Records and submits every wave.</summary>
    public void Submit() => _runner.Submit();

    /// <summary>Waits for <paramref name="wave"/> and every earlier one.</summary>
    public void Wait(int wave) => _runner.Wait(wave);

    /// <summary>Waits for every wave.</summary>
    public void WaitAll() => _runner.WaitAll();

    public void Dispose() => _runner.Dispose();
}
