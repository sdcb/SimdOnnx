using System.Runtime.CompilerServices;

// Shipped libraries (SimdPaddleOCR, SimdGcdrNet) use the public API only (docs/decisions.md); only
// test, benchmark and device-runner assemblies see the internals.
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
