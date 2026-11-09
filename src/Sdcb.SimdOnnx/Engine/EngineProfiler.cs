namespace Sdcb.SimdOnnx.Engine;

/// <summary>
/// Process-wide CPU operator profile for benchmarks: milliseconds and calls per operator and per
/// convolution class, summed over every session while <see cref="Enabled"/>. Also on with
/// SIMDONNX_PROF=1. Costs two timestamps per node while enabled.
/// </summary>
public static class EngineProfiler
{
    /// <summary>Collect timings (process-wide).</summary>
    public static bool Enabled
    {
        get => EngineProfile.Enabled;
        set => EngineProfile.Enabled = value;
    }

    /// <summary>Clears the collected timings.</summary>
    public static void Reset() => EngineProfile.Reset();

    /// <summary>Adds time spent by a consumer-side operator (e.g. a head computed outside the plan),
    /// in <see cref="System.Diagnostics.Stopwatch"/> ticks.</summary>
    public static void Record(string name, long stopwatchTicks) => EngineProfile.AddOp(name, stopwatchTicks);

    /// <summary>Milliseconds and calls per operator (ExecOp or extension op name).</summary>
    public static IReadOnlyDictionary<string, (double Ms, long Calls)> Operators() => EngineProfile.Operators();

    /// <summary>Milliseconds and calls per convolution class (e.g. Conv1x1, Depthwise3x3, Stride2Conv3x3).</summary>
    public static IReadOnlyDictionary<string, (double Ms, long Calls)> ConvClasses() => EngineProfile.ConvClasses();
}
