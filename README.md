# Sdcb.SimdOnnx [![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

<img src="assets/icon.svg" width="96" align="right" alt="icon" />

Pure C# inference engine for a **subset** of ONNX. It is the shared engine behind
[SimdPaddleOCR](https://github.com/sdcb/SimdPaddleOCR) and
[SimdGcdrNet](https://github.com/sdcb/Sdcb.GcdrNet).

> Not affiliated with Microsoft ONNX Runtime. Only the operators those models need are
> implemented; unsupported graphs fail at compile time with the offending node named.

- **Graph compiler**: shape-specialized lowering, fusion (conv + activation / residual / pool,
  GELU, transposed-conv bias, …), physical layout planning (NHWC, zero-copy transposes and
  reshapes) and a static, lifetime-packed memory arena.
- **CPU**: hand-written kernels for AVX-512, AVX2+FMA, AArch64 NEON, WASM SIMD128,
  `Vector<T>` and scalar, selected at runtime.
- **GPU** (net10.0): Vulkan (fp16 storage, cooperative-matrix conv/GEMM where available) and
  Metal (macOS); no native binaries, everything goes through P/Invoke.
- **Extensible**: consumers register model-specific fused operators and GPU kernels without
  touching the core.

Targets `net10.0` and `netstandard2.0` (CPU only, `Vector<T>` and scalar kernels).

## Status

A source library, not a NuGet package. Every type is `internal`: consumers take this repository as a
git submodule and compile the engine into their own assembly, so each consumer package carries its own
copy and applications can use several of them without a shared version to agree on
([docs/decisions.md](docs/decisions.md), 2026-11-18).

To embed it, import the source props from a project that targets `net10.0` and/or `netstandard2.0`
and allows unsafe blocks:

```xml
<Import Project="external/SimdOnnx/src/Sdcb.SimdOnnx/Sdcb.SimdOnnx.Sources.props" />
```

It adds the engine's `.cs` files (with the per-target exclusions), the Vulkan / Metal shaders as
embedded resources under their usual names, and the netstandard2.0 polyfill packages. The consumer's
own `InternalsVisibleTo` grants then cover its test and bench assemblies.

## Performance

Before/after numbers for both consumers (interleaved A/B on the same machine) and the open
performance work are in [docs/perf.md](docs/perf.md); trade-offs are logged in
[docs/decisions.md](docs/decisions.md).

## Device verification

Apple Silicon (Metal), arm64 Linux/Android (NEON) and mobile Vulkan were ported best effort
and are checked by CI for correctness only. `bench/Sdcb.SimdOnnx.Bench/run.sh` runs the
correctness + performance matrix on a device; see its header for the expected output.

## License

Apache-2.0
