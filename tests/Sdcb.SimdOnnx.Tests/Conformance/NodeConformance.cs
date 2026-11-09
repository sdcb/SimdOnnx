using System.Text;
using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Onnx;

namespace Sdcb.SimdOnnx.Tests.Conformance;

/// <summary>
/// Runs one ONNX backend node test (onnx/backend/test/data/node/&lt;name&gt;: model.onnx plus
/// test_data_set_k/input_i.pb, output_j.pb) on the CPU and classifies the result.
/// <para>Graph inputs that are float tensors are fed at run time. Integer inputs are baked into the
/// model as initializers: the engine evaluates integer tensors at compile time, so a shape / index
/// argument is only usable as a constant. If the run-time form does not compile, every input after
/// the first is baked too (status <c>pass-const</c>), which still exercises the kernel on input 0.</para>
/// <para>Each graph output is compiled separately with <see cref="CompileOptions.OutputName"/>.
/// Comparison uses ONNX's backend-test tolerance, |a - b| &lt;= atol + rtol·|b| (rtol 1e-3, atol 1e-7).</para>
/// </summary>
internal static class NodeConformance
{
    internal const string Tag = "v1.22.0";
    private const float Rtol = 1e-3f, Atol = 1e-7f;

    /// <summary>Folder of the fetched node tests (tools/fetch-onnx-tests.*), or null when absent.</summary>
    internal static string? DataDir()
    {
        if (Environment.GetEnvironmentVariable("SIMDONNX_ONNX_NODE_TESTS") is { Length: > 0 } env)
            return Directory.Exists(env) ? env : null;
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d is not null; d = d.Parent)
            if (File.Exists(Path.Combine(d.FullName, "Sdcb.SimdOnnx.slnx")))
            {
                string p = Path.Combine(d.FullName, "artifacts", "cache", $"onnx-node-{Tag}");
                return File.Exists(Path.Combine(p, ".complete")) ? p : null;
            }
        return null;
    }

    /// <summary>The manifest committed next to this file (tests/…/Conformance/node-expected.tsv).</summary>
    internal static string ManifestPath()
    {
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d is not null; d = d.Parent)
            if (File.Exists(Path.Combine(d.FullName, "Sdcb.SimdOnnx.slnx")))
                return Path.Combine(d.FullName, "tests", "Sdcb.SimdOnnx.Tests", "Conformance", "node-expected.tsv");
        throw new DirectoryNotFoundException("repository root not found");
    }

    /// <summary>One status per test: "pass", "pass-const", "mismatch", "unsupported\t…", "skip\t…" or "error\t…".</summary>
    internal static string Classify(string testDir, bool nhwc)
    {
        try { return Run(testDir, nhwc); }
        catch (Exception ex) { return Status(ex); }
    }

    /// <summary>Like <see cref="Classify"/> but compiles the run-time form only and lets exceptions escape (debugging).</summary>
    internal static void Trace(string testDir, bool nhwc)
    {
        string set = Path.Combine(testDir, "test_data_set_0");
        byte[] modelBytes = File.ReadAllBytes(Path.Combine(testDir, "model.onnx"));
        OnnxModel probe = OnnxModel.Load(modelBytes);
        int nIn = probe.Graph.Inputs.Count(i => !probe.Initializers.ContainsKey(i.Name));
        var inputs = Enumerable.Range(0, nIn).Select(i => ReadPb(Path.Combine(set, $"input_{i}.pb"))).ToList();
        var outputs = new List<OnnxTensorData>();
        for (int j = 0; File.Exists(Path.Combine(set, $"output_{j}.pb")); j++) outputs.Add(ReadPb(Path.Combine(set, $"output_{j}.pb")));
        Compare(modelBytes, inputs, outputs, nhwc, bakeAll: false, out _);
    }

    private static string Run(string testDir, bool nhwc)
    {
        string set = Path.Combine(testDir, "test_data_set_0");
        if (!Directory.Exists(set)) return "skip\tno test data";
        byte[] modelBytes = File.ReadAllBytes(Path.Combine(testDir, "model.onnx"));
        OnnxModel probe = OnnxModel.Load(modelBytes);
        int nIn = probe.Graph.Inputs.Count(i => !probe.Initializers.ContainsKey(i.Name));
        var inputs = new List<OnnxTensorData>();
        for (int i = 0; i < nIn; i++) inputs.Add(ReadPb(Path.Combine(set, $"input_{i}.pb")));
        var outputs = new List<OnnxTensorData>();
        for (int j = 0; File.Exists(Path.Combine(set, $"output_{j}.pb")); j++) outputs.Add(ReadPb(Path.Combine(set, $"output_{j}.pb")));
        if (outputs.Count == 0) return "skip\tno outputs";
        if (outputs.FirstOrDefault(o => o.DataType != OnnxTensorData.TypeFloat) is { } nf)
            return $"skip\toutput dtype {nf.DataType}";
        if (inputs.FirstOrDefault(t => t.DataType is not (OnnxTensorData.TypeFloat or 6 or 7)) is { } bad)
            return $"skip\tinput dtype {bad.DataType}";
        if (inputs.Count == 0 || inputs[0].DataType != OnnxTensorData.TypeFloat)
            return "skip\tfirst input is not float";

        Exception? first = null;
        foreach (bool bakeAll in new[] { false, true })
        {
            if (bakeAll && inputs.Count(t => t.DataType == OnnxTensorData.TypeFloat) <= 1) break;
            try
            {
                if (!Compare(modelBytes, inputs, outputs, nhwc, bakeAll, out string? why)) return "mismatch\t" + why;
                return bakeAll ? "pass-const" : "pass";
            }
            catch (Exception ex) when (ex is NotSupportedException or InvalidDataException or ArgumentException
                                          or InvalidOperationException)
            {
                first ??= ex;
            }
        }
        return Status(first!);
    }

    private static bool Compare(byte[] modelBytes, List<OnnxTensorData> inputs, List<OnnxTensorData> outputs,
        bool nhwc, bool bakeAll, out string? why)
    {
        why = null;
        OnnxModel m = OnnxModel.Load(modelBytes);   // fresh: baking mutates the graph
        var graphInputs = m.Graph.Inputs.Where(i => !m.Initializers.ContainsKey(i.Name)).ToList();
        var runtime = new List<(OnnxTensorData data, string name)>();
        for (int i = 0; i < inputs.Count; i++)
        {
            bool bake = inputs[i].DataType != OnnxTensorData.TypeFloat || (bakeAll && i > 0);
            if (!bake) { runtime.Add((inputs[i], graphInputs[i].Name)); continue; }
            OnnxTensorData t = inputs[i];
            t.Name = graphInputs[i].Name;
            m.Graph.Initializers.Add(t);
            m.Initializers[t.Name] = t;
        }
        if (runtime.Count == 0) throw new NotSupportedException("no float graph input");
        int[] shape0 = runtime[0].data.DimsArray();
        var shapes = runtime.Skip(1).Select(r => (int[]?)r.data.DimsArray()).ToList();
        var outNames = m.Graph.Outputs.Select(o => o.Name).ToList();
        for (int j = 0; j < outputs.Count; j++)
        {
            var opt = new CompileOptions(nhwc) { InputShapes = shapes, OutputName = outNames.Count > 1 ? outNames[j] : null };
            ExecPlan plan;
            try { plan = GraphCompiler.Compile(m, shape0, opt); }
            catch (ArgumentException ex) when (j > 0 && ex.Message.Contains("is not produced"))
            {
                throw new NotSupportedException($"secondary graph output '{outNames[j]}' is not produced");
            }
            using var s = new InferenceSession(plan);
            for (int i = 0; i < runtime.Count; i++) runtime[i].data.AsFloats().CopyTo(s.GetInput(i));
            s.Run();
            float[] got = s.Output.ToArray(), want = outputs[j].AsFloats();
            int[] gotShape = s.OutputShape, wantShape = outputs[j].DimsArray();
            if (got.Length != want.Length)
            {
                why = $"output {j}: {got.Length} elements, expected {want.Length}";
                return false;
            }
            if (!gotShape.SequenceEqual(wantShape) && !(gotShape.Length == 0 && want.Length == 1))
            {
                why = $"output {j}: shape [{string.Join(",", gotShape)}], expected [{string.Join(",", wantShape)}]";
                return false;
            }
            for (int k = 0; k < got.Length; k++)
            {
                float a = got[k], b = want[k];
                if (float.IsNaN(b) ? !float.IsNaN(a) : !(Math.Abs(a - b) <= Atol + Rtol * Math.Abs(b)) && a != b)
                {
                    why = $"output {j}";
                    return false;
                }
            }
        }
        return true;
    }

    private static OnnxTensorData ReadPb(string path)
    {
        using FileStream f = File.OpenRead(path);
        return OnnxProtoReader.ParseTensor(f);
    }

    private static string Status(Exception ex)
    {
        while (ex is InvalidOperationException { InnerException: { } inner } && ex.Message.StartsWith("node[", StringComparison.Ordinal)) ex = inner;
        string kind = ex is NotSupportedException ? "unsupported" : ex is InvalidDataException or ArgumentException ? "invalid" : "error";
        string msg = ex.Message.Replace('\t', ' ').Replace('\r', ' ').Replace('\n', ' ');
        if (msg.Length > 120) msg = msg[..120];
        return $"{kind}\t{(kind == "error" ? ex.GetType().Name + ": " : "")}{msg}";
    }


    /// <summary>The single op of a node test's model, or "(graph)" for multi-node (function-expanded) tests.</summary>
    internal static string OpOf(string testDir)
    {
        try
        {
            OnnxModel m = OnnxModel.Load(File.ReadAllBytes(Path.Combine(testDir, "model.onnx")));
            var ops = m.Graph.Nodes.Select(n => n.OpType).Distinct().ToList();
            return ops.Count == 1 ? ops[0] : "(graph)";
        }
        catch (Exception) { return "(unreadable)"; }
    }

    internal readonly record struct Row(string Op, string Nchw, string Nhwc);

    /// <summary>Manifest rows by test name.</summary>
    internal static SortedDictionary<string, Row> ReadManifest(string path)
    {
        var d = new SortedDictionary<string, Row>(StringComparer.Ordinal);
        if (!File.Exists(path)) return d;
        foreach (string line in File.ReadAllLines(path))
        {
            if (line.Length == 0 || line[0] == '#') continue;
            string[] c = line.Split('\t');
            // name, op, nchw-status, nchw-detail, nhwc-status, nhwc-detail
            d[c[0]] = new Row(c[1], Join(c, 2), Join(c, 4));
        }
        return d;
        static string Join(string[] c, int i) => c.Length > i + 1 && c[i + 1].Length > 0 ? c[i] + "\t" + c[i + 1] : c.Length > i ? c[i] : "";
    }

    internal static void WriteManifest(string path, SortedDictionary<string, Row> rows)
    {
        var sb = new StringBuilder();
        sb.Append("# ONNX backend node tests (onnx ").Append(Tag).Append(") on the CPU, NCHW and NHWC plans.\n");
        sb.Append("# Regenerate: SIMDONNX_WRITE_MANIFEST=1 dotnet test tests/Sdcb.SimdOnnx.Tests -f net10.0 --filter FullyQualifiedName~NodeTests\n");
        sb.Append("# Edits are reviewed like goldens: a change here is a change in what the engine supports.\n");
        sb.Append("# name\top\tnchw\tnchw-detail\tnhwc\tnhwc-detail\n");
        foreach (var kv in rows)
            sb.Append(kv.Key).Append('\t').Append(kv.Value.Op).Append('\t').Append(Cols(kv.Value.Nchw)).Append('\t').Append(Cols(kv.Value.Nhwc)).Append('\n');
        File.WriteAllText(path, sb.ToString());
        static string Cols(string s) => s.Contains('\t') ? s : s + "\t";
    }

    /// <summary>docs/operators.md, generated from the manifest (so CI can check it without the test data).</summary>
    internal static string OperatorsMarkdown(SortedDictionary<string, Row> rows)
    {
        static bool Pass(string s) => s.StartsWith("pass", StringComparison.Ordinal);
        static string Kind(string s) => s.Split('\t')[0];
        var sb = new StringBuilder();
        sb.Append("# Operator coverage\n\n");
        sb.Append("Generated from [`tests/Sdcb.SimdOnnx.Tests/Conformance/node-expected.tsv`](../tests/Sdcb.SimdOnnx.Tests/Conformance/node-expected.tsv), ");
        sb.Append("the ONNX backend node tests of onnx ").Append(Tag).Append(" run on the CPU (`NodeTests`). Do not edit by hand: ");
        sb.Append("`SIMDONNX_WRITE_MANIFEST=1 dotnet test tests/Sdcb.SimdOnnx.Tests -f net10.0 --filter FullyQualifiedName~NodeTests` rewrites both files.\n\n");
        sb.Append("How to read it:\n\n");
        sb.Append("- A test **passes** when every float output is within ONNX's backend-test tolerance (|a − b| ≤ 1e-7 + 1e-3·|b|). ");
        sb.Append("Integer inputs (shapes, indices) are baked in as constants, because the engine evaluates integer tensors at compile time; ");
        sb.Append("`pass-const` means the test also needed its other float inputs (e.g. weights, Clip bounds) as constants.\n");
        sb.Append("- **Skipped** tests have non-float outputs or inputs the engine does not take (bool, string, fp16, int8, sequences, …). ");
        sb.Append("They are not counted in the per-op totals below.\n");
        sb.Append("- NCHW and NHWC are the two CPU plan layouts (`CompileOptions.Nhwc`).\n\n");

        int total = rows.Count, skip = rows.Values.Count(r => Kind(r.Nchw) == "skip");
        sb.Append("| | tests | pass (NCHW) | pass (NHWC) |\n|---|---:|---:|---:|\n");
        sb.Append($"| all | {total - skip} | {rows.Values.Count(r => Pass(r.Nchw))} | {rows.Values.Count(r => Pass(r.Nhwc))} |\n");
        sb.Append($"| skipped (dtype) | {skip} | | |\n\n");

        sb.Append("## Single-op tests\n\n");
        sb.Append("| op | tests | pass NCHW | pass NHWC | most common reason for the rest |\n|---|---:|---:|---:|---|\n");
        foreach (var g in rows.Values.Where(r => r.Op != "(graph)" && Kind(r.Nchw) != "skip")
                     .GroupBy(r => r.Op).OrderBy(g => g.Key, StringComparer.Ordinal))
        {
            int n = g.Count(), a = g.Count(r => Pass(r.Nchw)), b = g.Count(r => Pass(r.Nhwc));
            string reason = g.Where(r => !Pass(r.Nchw)).Select(r => r.Nchw.Replace('\t', ' ').Replace("|", "\\|"))
                .GroupBy(s => s).OrderByDescending(x => x.Count()).ThenBy(x => x.Key, StringComparer.Ordinal)
                .Select(x => x.Key).FirstOrDefault() ?? "";
            if (reason.Length > 100) reason = reason[..100] + "…";
            sb.Append($"| {g.Key} | {n} | {a} | {b} | {reason} |\n");
        }
        var graph = rows.Values.Where(r => r.Op == "(graph)" && Kind(r.Nchw) != "skip").ToList();
        sb.Append("\n## Function-expanded and multi-node tests\n\n");
        sb.Append($"{graph.Count} tests; {graph.Count(r => Pass(r.Nchw))} pass on NCHW, {graph.Count(r => Pass(r.Nhwc))} on NHWC.\n");
        return sb.ToString();
    }

    internal static string OperatorsDocPath()
    {
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d is not null; d = d.Parent)
            if (File.Exists(Path.Combine(d.FullName, "Sdcb.SimdOnnx.slnx")))
                return Path.Combine(d.FullName, "docs", "operators.md");
        throw new DirectoryNotFoundException("repository root not found");
    }
}
