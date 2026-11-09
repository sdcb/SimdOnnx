using System.Collections.Concurrent;
using Xunit;
using Row = Sdcb.SimdOnnx.Tests.Conformance.NodeConformance.Row;

namespace Sdcb.SimdOnnx.Tests.Conformance;

/// <summary>
/// The ONNX backend node tests against the committed manifest (node-expected.tsv): every test's
/// status (pass / mismatch / unsupported / …) must equal the recorded one, so a regression fails and
/// so does a newly passing test until the manifest is updated. Needs tools/fetch-onnx-tests.*;
/// skips without the data unless SIMDONNX_REQUIRE_CONFORMANCE=1. docs/operators.md is generated
/// from the manifest and checked without the data.
/// </summary>
public class NodeTests
{
    /// <summary>Debug helper: SIMDONNX_TRACE_NODE_TEST=&lt;name&gt;[:nhwc] runs one test with exceptions escaping.</summary>
    [SkippableFact]
    public void Trace()
    {
        string? spec = Environment.GetEnvironmentVariable("SIMDONNX_TRACE_NODE_TEST");
        Skip.If(spec is null || NodeConformance.DataDir() is null, "set SIMDONNX_TRACE_NODE_TEST=<name>[:nhwc]");
        string[] p = spec!.Split(':');
        NodeConformance.Trace(Path.Combine(NodeConformance.DataDir()!, p[0]), p.Length > 1 && p[1] == "nhwc");
    }

    [SkippableFact]
    public void Cpu_MatchesManifest()
    {
        string? dir = NodeConformance.DataDir();
        if (dir is null && Environment.GetEnvironmentVariable("SIMDONNX_REQUIRE_CONFORMANCE") == "1")
            throw new InvalidOperationException("ONNX node tests not fetched: run tools/fetch-onnx-tests.ps1 or .sh");
        Skip.If(dir is null, "ONNX node tests not fetched (tools/fetch-onnx-tests.ps1 / .sh)");

        string[] tests = Directory.GetDirectories(dir!).OrderBy(d => d, StringComparer.Ordinal).ToArray();
        var actual = new ConcurrentDictionary<string, Row>(StringComparer.Ordinal);
        Parallel.ForEach(tests, new ParallelOptions { MaxDegreeOfParallelism = Math.Max(1, Environment.ProcessorCount / 2) },
            t => actual[Path.GetFileName(t)] = new Row(NodeConformance.OpOf(t),
                NodeConformance.Classify(t, nhwc: false), NodeConformance.Classify(t, nhwc: true)));
        var rows = new SortedDictionary<string, Row>(actual, StringComparer.Ordinal);

        string manifest = NodeConformance.ManifestPath();
        if (Environment.GetEnvironmentVariable("SIMDONNX_WRITE_MANIFEST") == "1")
        {
            NodeConformance.WriteManifest(manifest, rows);
            File.WriteAllText(NodeConformance.OperatorsDocPath(), NodeConformance.OperatorsMarkdown(rows));
            return;
        }
        var expected = NodeConformance.ReadManifest(manifest);
        var diffs = new List<string>();
        foreach (var kv in rows)
        {
            Row got = kv.Value;
            if (!expected.TryGetValue(kv.Key, out Row want)) diffs.Add($"{kv.Key}: not in manifest, got [{got.Nchw}] [{got.Nhwc}]");
            else if (want != got) diffs.Add($"{kv.Key}: expected [{want.Nchw}] [{want.Nhwc}], got [{got.Nchw}] [{got.Nhwc}]");
        }
        foreach (string name in expected.Keys.Where(k => !rows.ContainsKey(k)))
            diffs.Add($"{name}: in manifest but not in the test data");
        Assert.True(diffs.Count == 0,
            $"{diffs.Count} node tests differ from {Path.GetFileName(manifest)} (SIMDONNX_WRITE_MANIFEST=1 regenerates it):\n"
            + string.Join("\n", diffs.Take(60)));
    }

    [Fact]
    public void OperatorsDoc_MatchesManifest()
    {
        var rows = NodeConformance.ReadManifest(NodeConformance.ManifestPath());
        Assert.NotEmpty(rows);
        string want = NodeConformance.OperatorsMarkdown(rows);
        string got = File.ReadAllText(NodeConformance.OperatorsDocPath()).Replace("\r\n", "\n");
        Assert.True(want == got, "docs/operators.md is stale: SIMDONNX_WRITE_MANIFEST=1 regenerates it");
    }
}
