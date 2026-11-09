using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Sdcb.SimdOnnx.Onnx;

namespace Sdcb.SimdOnnx.Engine;

/// <summary>
/// CPU plans of one model for varying input shapes: each shape is compiled once (bounded LRU) and
/// every plan shares the packed weights, so a new shape costs only the graph lowering. Output
/// variants (<see cref="WithOutput"/>) are the same graph truncated at another tensor and share
/// those packed weights too. Thread-safe; share one instance between sessions.
/// </summary>
public sealed class CpuModel
{
    private readonly Dictionary<string, (Lazy<ExecPlan> plan, long used)> _plans = new(StringComparer.Ordinal);
    private readonly Dictionary<string, CpuModel> _variants = new(StringComparer.Ordinal);
    private readonly CpuKernels.Cache _packs;
    private readonly int _capacity;
    private long _clock;

    /// <param name="model">The model.</param>
    /// <param name="options">Compile options (default <see cref="CompileOptions.Default"/>).</param>
    /// <param name="capacity">Input shapes kept compiled (least recently used are dropped).</param>
    public CpuModel(OnnxModel model, CompileOptions? options = null, int capacity = 64)
        : this(model, options, capacity, packs: null)
    {
    }

    /// <param name="packs">Packed-weight cache to share with other CPU models of the same graph
    /// (e.g. a head-truncated variant); null = private.</param>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal CpuModel(OnnxModel model, CompileOptions? options, int capacity, CpuKernels.Cache? packs)
    {
        Model = model ?? throw new ArgumentNullException(nameof(model));
        Options = options ?? CompileOptions.Default;
        _capacity = Math.Max(1, capacity);
        _packs = packs ?? new CpuKernels.Cache();
        Root = this;
    }

    /// <summary>The model.</summary>
    public OnnxModel Model { get; }
    /// <summary>The compile options of every plan.</summary>
    public CompileOptions Options { get; }
    internal CpuKernels.Cache Packs => _packs;
    /// <summary>The model <see cref="WithOutput"/> was called on (this one if it is not a variant).</summary>
    internal CpuModel Root { get; private set; }

    /// <summary>
    /// The same graph with its output at tensor <paramref name="tensorName"/> (e.g. the logits before a
    /// final Softmax). Shares this model's packed weights, and a <see cref="CpuSession"/> of this model
    /// can run it on the input it already holds. Repeated calls return the same instance.
    /// </summary>
    public CpuModel WithOutput(string tensorName)
    {
        if (string.IsNullOrEmpty(tensorName)) throw new ArgumentException("Tensor name is empty.", nameof(tensorName));
        lock (_variants)
        {
            if (_variants.TryGetValue(tensorName, out CpuModel? v)) return v;
            v = new CpuModel(Model, Options with { OutputName = tensorName }, _capacity, _packs) { Root = Root };
            _variants[tensorName] = v;
            return v;
        }
    }

    /// <summary>The output shape for <paramref name="inputShape"/> (compiles that shape if needed).</summary>
    public int[] OutputShape(ReadOnlySpan<int> inputShape)
    {
        ExecPlan p = Plan(inputShape);
        return (int[])p.Values[p.OutputValue].Shape.Clone();
    }

    /// <summary>The packed plan for <paramref name="shape"/> (compiled on first use).</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal ExecPlan Plan(ReadOnlySpan<int> shape)
    {
        string key = string.Join(",", shape.ToArray());
        Lazy<ExecPlan> lazy;
        lock (_plans)
        {
            if (_plans.TryGetValue(key, out var e))
            {
                _plans[key] = (e.plan, ++_clock);
                lazy = e.plan;
            }
            else
            {
                // one compile per shape: concurrent callers of a new shape wait for the same plan
                int[] sh = shape.ToArray();
                lazy = new Lazy<ExecPlan>(() =>
                {
                    ExecPlan p = GraphCompiler.Compile(Model, sh, Options);
                    CpuKernels.Pack(p, _packs);
                    return p;
                }, LazyThreadSafetyMode.ExecutionAndPublication);
                _plans[key] = (lazy, ++_clock);
                if (_plans.Count > _capacity)
                {
                    string oldest = _plans.Where(kv => kv.Key != key).OrderBy(kv => kv.Value.used).First().Key;
                    _plans.Remove(oldest);
                }
            }
        }
        try { return lazy.Value; }
        catch
        {
            lock (_plans)
                if (_plans.TryGetValue(key, out var cur) && cur.plan == lazy) _plans.Remove(key);
            throw;
        }
    }
}

/// <summary>
/// A reshapable CPU session over a <see cref="CpuModel"/>: one grow-only native arena reused by the
/// plan of whatever input shape is current, and by the model's output variants
/// (<see cref="CpuModel.WithOutput"/>). Graph input 0 sits at the start of the arena in every plan,
/// so <see cref="Input"/> is written once and any variant can run on it. Not thread-safe: one run at
/// a time; use one session per worker.
/// </summary>
/// <remarks>With <see cref="CompileOptions.ReleaseInputs"/> a run may overwrite the input: write
/// <see cref="Input"/> again before every <see cref="Run"/>.</remarks>
public sealed unsafe class CpuSession : IDisposable
{
    private readonly CpuModel _model;
    private (CpuModel Model, InferenceSession Session)[] _bound = [];
    private int[] _shape = [];
    private int _volume;
    private nint _arena;
    private long _arenaBytes;
    private bool _disposed;

    public CpuSession(CpuModel model) => _model = model ?? throw new ArgumentNullException(nameof(model));

    /// <summary>The model this session runs (and whose variants it can run).</summary>
    public CpuModel Model => _model;

    /// <summary>Intra-op thread budget of each run (0 = process default).</summary>
    public int Threads { get; set; }

    /// <summary>Current input shape (empty before <see cref="Reshape"/>).</summary>
    public ReadOnlySpan<int> InputShape => _shape;

    /// <summary>Bytes of the native arena (grows, never shrinks, until disposed).</summary>
    public long ArenaBytes => _arenaBytes;

    /// <summary>Graph input 0 of the current shape, laid out as the model's options say (logical NCHW,
    /// or channels-last with <see cref="CompileOptions.PhysicalIo"/>).</summary>
    public Span<float> Input => _arena == 0 ? [] : new Span<float>((float*)_arena, _volume);

    /// <summary>Binds <paramref name="inputShape"/> (no-op when already bound). The input buffer keeps
    /// what was written to it when the arena has to grow later.</summary>
    public void Reshape(ReadOnlySpan<int> inputShape) => Reshape(inputShape, prepare: null);

    /// <summary>Binds <paramref name="inputShape"/> and compiles <paramref name="prepare"/>'s plan for it
    /// (a variant, or <see cref="Model"/>), sizing the arena once for that plan instead of for the
    /// input first and again on the first run.</summary>
    public void Reshape(ReadOnlySpan<int> inputShape, CpuModel? prepare)
    {
        ThrowIfDisposed();
        if (_arena != 0 && inputShape.SequenceEqual(_shape))
        {
            if (prepare is not null) Bind(prepare);
            return;
        }
        long volume = 1;
        foreach (int d in inputShape)
        {
            if (d <= 0) throw new ArgumentException("Input dims must be positive.", nameof(inputShape));
            volume = checked(volume * d);
        }
        CpuModel? m = prepare;
        if (m is not null && !ReferenceEquals(m.Root, _model.Root))
            throw new ArgumentException("The variant is not a WithOutput sibling of this session's model.", nameof(prepare));
        ExecPlan? plan = m?.Plan(inputShape);   // compiled first: one arena size covers input and plan
        _shape = inputShape.ToArray();
        _volume = checked((int)volume);
        _bound = [];
        long need = Math.Max(Math.Max(plan?.ArenaFloats ?? 0, volume) * sizeof(float), 64);
        if (need > _arenaBytes) Grow(need);
        if (m is not null) Bind(m);
    }

    /// <summary>Runs <paramref name="variant"/> (default: <see cref="Model"/>) on <see cref="Input"/> and
    /// returns its output, valid until the next run or reshape.</summary>
    public ReadOnlySpan<float> Run(CpuModel? variant = null)
    {
        InferenceSession s = Bind(variant);
        s.Threads = Threads;
        s.Run();
        return s.Output;
    }

    /// <summary>Output shape of <paramref name="variant"/> (default: <see cref="Model"/>) for the current
    /// input shape; compiles and binds the plan, so a following <see cref="Run"/> starts at once.</summary>
    public int[] OutputShape(CpuModel? variant = null) => (int[])Bind(variant).OutputShape.Clone();

    private InferenceSession Bind(CpuModel? variant)
    {
        ThrowIfDisposed();
        if (_arena == 0) throw new InvalidOperationException("Call Reshape first.");
        CpuModel m = variant ?? _model;
        if (!ReferenceEquals(m.Root, _model.Root))
            throw new ArgumentException("The variant is not a WithOutput sibling of this session's model.", nameof(variant));
        foreach (var (bm, bs) in _bound)
            if (ReferenceEquals(bm, m)) return bs;
        ExecPlan plan = m.Plan(_shape);
        // every plan places graph input 0 at offset 0, which is what lets variants share the input
        if (plan.BufOffset[plan.Values[plan.InputValue].Buf] != 0)
            throw new InvalidOperationException("Plan input is not at arena offset 0.");
        long need = Math.Max(Math.Max((long)plan.ArenaFloats, _volume) * sizeof(float), 64);
        if (need > _arenaBytes) Grow(need);
        var s = new InferenceSession(plan, _arena);
        _bound = [.. _bound, (m, s)];
        return s;
    }

    // keeps the (possibly already written) input; sessions bound to the old arena are dropped
    private void Grow(long bytes)
    {
        nint next = (nint)NativeMemory.AllocZeroed((nuint)bytes);
        if (_arena != 0)
        {
            Buffer.MemoryCopy((void*)_arena, (void*)next, bytes, Math.Min(_arenaBytes, (long)_volume * sizeof(float)));
            NativeMemory.Free((void*)_arena);
            GC.RemoveMemoryPressure(_arenaBytes);
        }
        _arena = next;
        _arenaBytes = bytes;
        GC.AddMemoryPressure(bytes);
        _bound = [];
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CpuSession));
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _bound = [];
        if (_arena != 0)
        {
            NativeMemory.Free((void*)_arena);
            GC.RemoveMemoryPressure(_arenaBytes);
            _arena = 0;
        }
    }
}
