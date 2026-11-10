namespace Sdcb.SimdOnnx.Onnx;

/// <summary>
/// Builds an in-memory <see cref="OnnxModel"/> out of loaded models and glue nodes, e.g. two networks
/// that run back to back with a few elementwise ops between them become one model: one plan, one
/// session and one arena, where the first network's activations are reused by the second.
/// Inlined models share their weight arrays with the source model (nothing is copied).
/// </summary>
/// <remarks>Preview API. Every inlined model must use <see cref="Opset"/> (ONNX operator semantics
/// depend on it); the glue nodes are read under it too.</remarks>
public sealed class OnnxModelBuilder
{
    private readonly OnnxGraphData _g = new();
    private readonly HashSet<string> _names = new(StringComparer.Ordinal);
    private int _auto;

    /// <param name="opset">The default-domain opset of the built model.</param>
    public OnnxModelBuilder(int opset)
    {
        if (opset < 1) throw new ArgumentOutOfRangeException(nameof(opset));
        Opset = opset;
    }

    /// <summary>The default-domain opset of the built model.</summary>
    public int Opset { get; }

    private string Claim(string name)
    {
        if (string.IsNullOrEmpty(name)) throw new ArgumentException("Tensor names must not be empty.", nameof(name));
        if (!_names.Add(name)) throw new ArgumentException($"Tensor '{name}' is already defined.", nameof(name));
        return name;
    }

    private void Require(string name, string what)
    {
        if (!_names.Contains(name)) throw new ArgumentException($"{what} '{name}' is not defined in this builder.");
    }

    /// <summary>Adds a float graph input (the first one added is the model's main input). A dim of -1 is
    /// dynamic (given at compile time).</summary>
    /// <returns><paramref name="name"/>.</returns>
    public string Input(string name, params int[] dims)
    {
        if (dims is null) throw new ArgumentNullException(nameof(dims));
        _g.Inputs.Add(new OnnxValueInfoData { Name = Claim(name), ElementType = OnnxTensorData.TypeFloat, Shape = (int[])dims.Clone() });
        return name;
    }

    /// <summary>Adds a float constant (initializer).</summary>
    /// <returns><paramref name="name"/>.</returns>
    public string Constant(string name, float[] values, params int[] dims)
    {
        if (values is null) throw new ArgumentNullException(nameof(values));
        long n = 1;
        foreach (int d in dims) n *= d;
        if (n != values.Length) throw new ArgumentException($"Constant '{name}': {values.Length} values for dims [{string.Join(",", dims)}].", nameof(values));
        var t = new OnnxTensorData { Name = Claim(name), DataType = OnnxTensorData.TypeFloat };
        foreach (int d in dims) t.Dims.Add(d);
        t.SetFloats((float[])values.Clone());
        _g.Initializers.Add(t);
        return name;
    }

    /// <summary>
    /// Copies every node and weight of <paramref name="model"/> in, with each tensor name prefixed by
    /// <paramref name="prefix"/>; the model's graph inputs are bound, in order, to <paramref name="inputs"/>.
    /// </summary>
    /// <returns>The (prefixed) names of the model's graph outputs.</returns>
    public IReadOnlyList<string> Inline(OnnxModel model, string prefix, params string[] inputs)
    {
        if (model is null) throw new ArgumentNullException(nameof(model));
        if (prefix is null) throw new ArgumentNullException(nameof(prefix));
        if (model.Opset != Opset)
            throw new NotSupportedException($"Inline: the model uses opset {model.Opset}, the builder opset {Opset}.");
        OnnxGraphData src = model.Graph;
        var graphInputs = src.Inputs.Where(i => !model.Initializers.ContainsKey(i.Name)).ToList();
        if (graphInputs.Count != inputs.Length)
            throw new ArgumentException($"Inline: the model has {graphInputs.Count} graph inputs, {inputs.Length} given.", nameof(inputs));
        var map = new Dictionary<string, string>(StringComparer.Ordinal);
        for (int i = 0; i < inputs.Length; i++)
        {
            Require(inputs[i], "Input");
            map[graphInputs[i].Name] = inputs[i];
        }
        string Map(string n)
        {
            if (n.Length == 0) return n;   // an omitted optional input
            if (map.TryGetValue(n, out string? m)) return m;
            return map[n] = Claim(prefix + n);
        }
        // weights: same decoded arrays under the new name
        foreach (OnnxTensorData t in src.Initializers) _g.Initializers.Add(t.Renamed(Map(t.Name)));
        foreach (OnnxNodeData n in src.Nodes)
        {
            var c = new OnnxNodeData { Name = n.Name.Length == 0 ? "" : prefix + n.Name, OpType = n.OpType, Domain = n.Domain };
            foreach (string i in n.Inputs)
            {
                if (i.Length != 0 && !map.ContainsKey(i)) throw new InvalidDataException($"Inline: '{i}' is read before it is produced.");
                c.Inputs.Add(Map(i));
            }
            foreach (string o in n.Outputs) c.Outputs.Add(Map(o));
            c.Attributes.AddRange(n.Attributes);
            _g.Nodes.Add(c);
        }
        return src.Outputs.Select(o => map[o.Name]).ToArray();
    }

    /// <summary>
    /// Adds one node with a single output. Attribute values may be <see cref="int"/>, <see cref="long"/>,
    /// <see cref="float"/>, <see cref="string"/>, <c>int[]</c>, <c>long[]</c> or <c>float[]</c>.
    /// Pass <c>""</c> for an omitted optional input.
    /// </summary>
    /// <returns>The output name: <paramref name="output"/>, or a generated one when null.</returns>
    public string Node(string opType, string[] inputs, string? output = null, params (string Name, object Value)[] attributes)
    {
        if (string.IsNullOrEmpty(opType)) throw new ArgumentException("opType must not be empty.", nameof(opType));
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));
        foreach (string i in inputs)
            if (i.Length != 0) Require(i, "Input");
        string outName = output ?? NextName(opType);
        var n = new OnnxNodeData { Name = outName, OpType = opType };
        n.Inputs.AddRange(inputs);
        n.Outputs.Add(Claim(outName));
        foreach (var (name, value) in attributes) n.Attributes.Add(Attribute(name, value));
        _g.Nodes.Add(n);
        return outName;
    }

    private string NextName(string op)
    {
        string n;
        do n = $"{op}_{_auto++}"; while (_names.Contains(n));
        return n;
    }

    private static OnnxAttributeData Attribute(string name, object value)
    {
        var a = new OnnxAttributeData { Name = name };
        switch (value)
        {
            case int i: a.Type = 2; a.Int = i; break;
            case long l: a.Type = 2; a.Int = l; break;
            case float f: a.Type = 1; a.Float = f; break;
            case string s: a.Type = 3; a.String = System.Text.Encoding.UTF8.GetBytes(s); break;
            case int[] ii: a.Type = 7; a.Ints.AddRange(ii.Select(x => (long)x)); break;
            case long[] ll: a.Type = 7; a.Ints.AddRange(ll); break;
            case float[] ff: a.Type = 6; a.Floats.AddRange(ff); break;
            default: throw new ArgumentException($"Attribute '{name}': unsupported value type {value?.GetType().Name ?? "null"}.", nameof(value));
        }
        return a;
    }

    /// <summary>Marks <paramref name="name"/> as the graph output (the engine runs single-output graphs).</summary>
    public void Output(string name)
    {
        Require(name, "Output");
        if (_g.Outputs.Count != 0) throw new InvalidOperationException("The graph output is already set.");
        _g.Outputs.Add(new OnnxValueInfoData { Name = name, ElementType = OnnxTensorData.TypeFloat });
    }

    /// <summary>The model. The builder may not be used afterwards.</summary>
    public OnnxModel Build()
    {
        if (_g.Inputs.Count == 0) throw new InvalidOperationException("The graph has no input.");
        if (_g.Outputs.Count == 0) throw new InvalidOperationException("The graph has no output.");
        return OnnxModel.FromGraph(Opset, _g);
    }
}
