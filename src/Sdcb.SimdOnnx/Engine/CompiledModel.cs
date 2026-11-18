using Sdcb.SimdOnnx.Onnx;

namespace Sdcb.SimdOnnx.Engine;

/// <summary>
/// An <see cref="OnnxModel"/> specialized to a concrete input shape: all
/// dynamic-shape plumbing folded, ops lowered to the exec set, buffers
/// planned. Immutable; share across sessions freely.
/// </summary>
internal sealed class CompiledModel
{
    internal ExecPlan Plan { get; }

    internal CompiledModel(ExecPlan plan) => Plan = plan;

    public static CompiledModel Compile(OnnxModel model, ReadOnlySpan<int> inputShape) =>
        new(GraphCompiler.Compile(model, inputShape));

    public static CompiledModel Compile(OnnxModel model, ReadOnlySpan<int> inputShape, CompileOptions options) =>
        new(GraphCompiler.Compile(model, inputShape, options));

    public InferenceSession CreateSession() => new(Plan);

    public int NodeCount => Plan.Nodes.Length;
    public int ArenaFloats => Plan.ArenaFloats;
}
