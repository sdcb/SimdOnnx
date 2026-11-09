using System.Runtime.CompilerServices;

// SimdGcdrNet still builds against engine internals (fused GPU pipeline, kernel helpers) and pins an
// exact Sdcb.SimdOnnx version, so internal changes cannot skew at runtime. SimdPaddleOCR uses the
// public API only (docs/decisions.md).
[assembly: InternalsVisibleTo("Sdcb.SimdGcdrNet")]
[assembly: InternalsVisibleTo("Sdcb.SimdOnnx.Tests")]
[assembly: InternalsVisibleTo("Sdcb.SimdOnnx.Bench")]
// consumers' own test / bench / device-runner assemblies drive the engine directly
[assembly: InternalsVisibleTo("Sdcb.SimdGcdrNet.Tests")]
[assembly: InternalsVisibleTo("Sdcb.SimdGcdrNet.Benchmark")]
[assembly: InternalsVisibleTo("Sdcb.SimdGcdrNet.AndroidBench")]
[assembly: InternalsVisibleTo("Sdcb.SimdGcdrNet.WasmBench")]
[assembly: InternalsVisibleTo("Sdcb.SimdPaddleOCR.Tests")]
[assembly: InternalsVisibleTo("Sdcb.SimdPaddleOCR.UnitTests")]
[assembly: InternalsVisibleTo("Sdcb.SimdPaddleOCR.GpuBench")]
[assembly: InternalsVisibleTo("Sdcb.SimdPaddleOCR.AndroidBench")]
[assembly: InternalsVisibleTo("Sdcb.SimdPaddleOCR.WasmBench")]
