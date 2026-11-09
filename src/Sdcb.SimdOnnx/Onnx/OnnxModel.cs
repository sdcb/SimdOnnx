namespace Sdcb.SimdOnnx.Onnx;

/// <summary>Validated ONNX model: graph nodes, initializers, IO metadata.</summary>
public sealed class OnnxModel
{
    private OnnxModel(int opset, OnnxGraphData graph, Dictionary<string, OnnxTensorData> initializers)
    {
        Opset = opset;
        Graph = graph;
        Initializers = initializers;
    }

    public int Opset { get; }
    internal OnnxGraphData Graph { get; }
    internal Dictionary<string, OnnxTensorData> Initializers { get; }

    private IReadOnlyList<OnnxNodeInfo>? _nodes;

    /// <summary>The graph's nodes in topological order, after the load-time rewrites (Constant nodes
    /// become initializers, Conv + BatchNormalization folds). Read-only.</summary>
    public IReadOnlyList<OnnxNodeInfo> Nodes => _nodes ??= Graph.Nodes
        .Select(n => new OnnxNodeInfo(n.OpType, n.Name, n.Inputs.ToArray(), n.Outputs.ToArray())).ToArray();

    /// <summary>Dims of initializer <paramref name="name"/> (a constant tensor such as conv weights).</summary>
    public bool TryGetInitializerShape(string name, [System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out int[]? dims)
    {
        dims = Initializers.TryGetValue(name, out OnnxTensorData? t) ? t.DimsArray() : null;
        return dims is not null;
    }

    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, float[]> _derived = new(StringComparer.Ordinal);

    /// <summary>
    /// The model's canonical instance of a constant derived at compile time (folded BatchNorm,
    /// merged biases, cast shape values): every per-shape compile gets the same array for the same
    /// contents, so packed-weight caches and GPU const pools match it by reference.
    /// </summary>
    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, (float[] A, float[]? B, float[] Result)> _derivedFrom = new(StringComparer.Ordinal);

    /// <summary>
    /// A constant computed from source arrays <paramref name="a"/> / <paramref name="b"/>, built once
    /// per distinct pair: initializers are the same arrays in every compile, so a later compile gets
    /// the earlier result without recomputing or comparing it; sources that are rebuilt per shape
    /// (folded shape-dependent constants) are new arrays and miss.
    /// </summary>
    internal float[] Derive(string key, float[] a, float[]? b, Func<float[]> make)
    {
        if (_derivedFrom.TryGetValue(key, out var hit) && ReferenceEquals(hit.A, a) && ReferenceEquals(hit.B, b)) return hit.Result;
        float[] r = Intern(key, make());
        _derivedFrom[key] = (a, b, r);
        return r;
    }

    internal float[] Intern(string key, float[] values)
    {
        float[] cur = _derived.GetOrAdd(key, values);
        if (ReferenceEquals(cur, values) || cur.AsSpan().SequenceEqual(values)) return cur;
        _derived[key] = values;   // shape-dependent contents: the latest wins
        return values;
    }

    public static OnnxModel Load(string path)
    {
        using FileStream fs = File.OpenRead(path);
        return Load(fs);
    }

    public static OnnxModel Load(Stream source) => Build(OnnxProtoReader.Parse(source));

    /// <summary>Parses the model straight from <paramref name="source"/> (pinned for the duration
    /// of the call, no intermediate copy); weights are decoded into model-owned arrays, so the
    /// memory may be released afterwards.</summary>
    public static unsafe OnnxModel Load(ReadOnlySpan<byte> source)
    {
        fixed (byte* p = source)
        {
            using var ums = new UnmanagedMemoryStream(p, source.Length);
            return Load(ums);
        }
    }

    private static OnnxModel Build(OnnxModelData parsed)
    {
        OnnxGraphData g = parsed.Graph!;
        int opset = checked((int)(parsed.Opsets.FirstOrDefault(
            static x => x.Domain.Length == 0 || x.Domain == "ai.onnx")?.Version ?? 0));
        Dictionary<string, OnnxTensorData> init = new(g.Initializers.Count);
        foreach (OnnxTensorData t in g.Initializers) init[t.Name] = t;
        OnnxRewrites.Apply(g, init);
        // rewrites add folded weights; drop the initializers no node reads any more
        HashSet<string> used = new(g.Nodes.SelectMany(n => n.Inputs), StringComparer.Ordinal);
        g.Initializers.Clear();
        g.Initializers.AddRange(init.Values.Where(t => used.Contains(t.Name)));
        init = g.Initializers.ToDictionary(t => t.Name, StringComparer.Ordinal);
        return new OnnxModel(opset, g, init);
    }
}
