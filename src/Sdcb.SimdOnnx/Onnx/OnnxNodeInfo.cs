namespace Sdcb.SimdOnnx.Onnx;

/// <summary>One graph node of an <see cref="OnnxModel"/>: op type, name and tensor names (an empty
/// input name is an omitted optional input).</summary>
public sealed record OnnxNodeInfo(string OpType, string Name, IReadOnlyList<string> Inputs, IReadOnlyList<string> Outputs);
