# Decisions

Trade-offs taken while the maintainer was away. Each entry: what, why, how to revisit.

## 2026-10-13 — Core seeded from the GcdrNet engine

The GcdrNet `ExecPlan` design (shape-specialized lowering → fusion → layout → arena, one IR for
CPU / Vulkan / Metal) is the trunk; SimdPaddleOCR's strengths are ported into it.
Spike (`Sdcb.GcdrNet` branch `spike/simdcore-det`) showed PP-OCR det on this engine at CPU parity
(multi-thread), lower peak memory, Vulkan correct.

## 2026-10-13 — First-party consumers use InternalsVisibleTo

SimdPaddleOCR and SimdGcdrNet register extension ops and drive fused GPU pipelines through engine
internals (`ExecPlan`, `VkGraphModel`, …). Designing a stable public extension API up front would
freeze types that are still moving. Instead the core grants `InternalsVisibleTo` to both, and they
reference an exact version (`[x.y.z]`) so a mismatched internal API cannot load.
Revisit when a third-party extension is needed.

## 2026-10-13 — GCDR glue stays in SimdGcdrNet

`bgr2nhwc` / `div6` / `nhwc2bgr` shaders, `Vk/MtlGcdrPipeline` and `PixelOps` are GCDR-specific and
were not copied into the core.

## 2026-10-13 — netstandard2.0: polyfills, not a separate kernel set

The ns2.0 build compiles the same sources. `Compatibility/Intrinsics.cs` stubs the x86/Arm/Wasm
intrinsic types with `IsSupported == false`, so the `Vector<T>` and scalar tiers run;
`Compatibility/Polyfills.cs` supplies the missing BCL members (C# 14 extension members).
Static-abstract interface kernels (`IOp`, `IAct`) became instance interfaces called on
`default(T)` (same codegen on net10). GPU backends are net10-only. Friend test assemblies that
consume both builds reference the engine with an alias (`simdonnx_ns20`) to avoid duplicate
polyfill types.

## 2026-10-13 — Intel Arc B580: avoid the 1x1 cooperative-matrix BN=32 variant

Intermittent `VK_ERROR_DEVICE_LOST` on the B580 (subgroup 16 coopmat path) also reproduces on
the old GcdrNet main. It correlates with the BN=32 1x1 coopmat conv; on subgroup-16 devices 1x1
convs now always use the BN=64 variant (`conv_cm_n64`). Stress runs went from flaky to clean
(det/rec 8/8, GcdrNet 5/5). `GCDRNET_VK_CM1X1=0` disables coopmat 1x1 entirely if it recurs.
A rare DeviceLost in GcdrNet's size-switching Vulkan test (~1 in 10 full-suite runs) remains.
*Update 2026-10-14:* most of that flakiness was test interference, see below.

## 2026-10-13 — Ship with Paddle Vulkan slower than the old hand-fused backend

*Update 2026-10-14:* the gap was 4–9% in one interleaved run (noisy: 0.97–1.43 across
runs), memory +15–23% instead of +100%. *Update 2026-10-15:* a re-run does not confirm the
4–9%: three clean interleaved rounds give tiny +19–33%, small +36–50%, medium +7–13%, and 2 of
13 new runs lost the device (re-creation then failed with `ErrorIncompatibleDriver`, CPU
fallback). See `docs/perf.md`. The remaining gap is per-new-shape setup, not kernels.

CPU met the "not slower" bar everywhere (see `docs/perf.md`); Paddle Vulkan did not (tiny +40%,
small +20%, medium +5%), and Vulkan peak working set is higher. Rather than port the old
PP-OCR-specific fused shaders into the generic engine under time pressure, the PR ships the
generic path (correct, accuracy equal or better) and lists the gap as the first follow-up.
Revisit: fusion work in `docs/perf.md` "Known gaps"; do not merge the Paddle PR to main until the
maintainer accepts or closes the gap.

## 2026-10-13 — Engine-internal tests/benches move to SimdOnnx

Paddle's kernel tests for removed kernels (`NhwcKernelTests`, conv parts of
`KernelCorrectnessTests`, `GpuSessionShapeTests`, `InferenceSessionRunContractTests`), the
`GpuBench` project and the AndroidBench GPU probes (caps, gemm, layer probe, profiling) tested
engine internals that no longer live in Paddle. They were removed; equivalent coverage lives in
`Sdcb.SimdOnnx.Tests` and `bench/Sdcb.SimdOnnx.Bench` (`gcdr-vkdiff`, `--micro`). Paddle keeps
its own tests for what it still owns (CTC/MatMul argmax head, preprocessing, pipeline).

## 2026-10-13 — Pre-release packages on the ProGet test feed

Consumers pin `Sdcb.SimdOnnx` exactly and restore it from the ProGet test feed through a
`nuget.config` with package-source mapping (only `Sdcb.SimdOnnx` comes from ProGet). Before
merging to main: publish the engine to nuget.org and delete the ProGet source.

Push with `--no-symbols`: the test feed accepts overwrites, and `dotnet nuget push` sends the
`.snupkg` next to the `.nupkg` to the same endpoint, which replaced 0.1.0-preview.4 with its symbols
package (pdbs only, no assemblies) on the first upload.

## 2026-10-14 — GPU consts pooled per model, not per identical const list

The const store used to be shared only between graphs whose const lists matched exactly; the
CTC/logits graphs (truncated outputs) of every rec shape therefore re-packed and re-uploaded the
whole model (~170 MB of allocations each, medium rec), which was the "Vulkan memory doubled".
Now each (device, model, precision) has one pool of packed entries keyed by source array +
packing kind (content-compared on a reference miss); a graph uploads only missing entries as a
new chunk, and const binds address the chunk buffer directly (`VkSlot.Direct`). Metal mirrors it.

## 2026-10-14 — Device loss: recover instead of failing forever

`VK_ERROR_DEVICE_LOST` raises `VkDeviceLostException` and marks the device lost;
`VkBackend.Default` then creates a new device on next use. Paddle finishes the failing call on the
CPU and goes back to the GPU (permanent CPU fallback only after 3 losses per session); GcdrNet
retries the image once on a fresh pipeline pool. Objects of the lost device are abandoned, not
destroyed. Tests for this are opt-in (`SIMDONNX_TEST_DEVICELOSS=1`) because they disturb every
other Vulkan test in the process.

## 2026-10-14 — B580 flakiness: what was found

- GcdrNet's Vulkan tests failed ~1 in 4–10 full runs ("size-switch differs from a fresh
  worker", earlier also DeviceLost). Two tests changed process-wide state while others ran in
  parallel: `NoFuse_MatchesFused` set `GCDRNET_NOFUSE` (other tests then compiled unfused graphs)
  and the ConvDot route test flipped `ConvDot` on the shared device. Fixed with
  `CompileOptions.Fuse` and a private device for the ConvDot test: 0 failures in 25 + 10 runs.
- Validation layers (core, thread safety) on a 4-worker Paddle Vulkan run: clean. GPU-assisted
  validation on the GcdrNet Vulkan tests: no out-of-bounds access; it reports shared-memory races in
  `conv_cm` between `coopMatStore` and the epilogue loads of the same subgroup, separated by
  `subgroupMemoryBarrierShared(); subgroupBarrier()` — taken as false positives of GPU-AV's race
  model for cooperative-matrix ops, not changed.
- Paddle Vulkan e2e stress (tiny, 4 workers): 0 crashes in 46 runs; an access violation inside the driver's
  `vkQueueSubmit` was seen 3 times in one cluster (while VS Code was installing an update) and
  once a run silently fell back to the CPU; neither reproduced afterwards. Fallbacks now log one
  line to stderr. `GCDRNET_VK_QUEUE=gfx` (graphics queue instead of async compute) was tried:
  no stability difference, 2x slower, so it stays a debug knob.

## 2026-10-15 — B580 device loss: the staged SG16 conv, not the runtime

Reproduced with a stress loop that counts CPU fallbacks (`.rerun`-style: 15–25 x tiny+small,
4 workers): about 1 run in 7 lost the device (also with 1 worker), the old Paddle engine 0 in 24.
No TDR event in the system log; GPU-assisted validation over 100 images found no out-of-bounds
access, only the known coopmat shared-memory race reports. Disabling the SG16 coopmat path
(`GCDRNET_VK_SG16=0`) removed every loss (0 / 35) at 1.5x the time; a full-workgroup barrier in
the staged epilogue did not help (2 / 21); disabling only the 1x1 coopmat did not either (2 / 23).

Fix: on 8x16x16 (SG16) devices dense convs and const Gemms take the old OCR engine's direct
family instead of the shared-staged `conv_cm` SG16 build — `pw_dot` (1x1 / Gemm, K <= 128),
`gemm_cm8` (direct-load 8x16x16 GEMM, K % 16 == 0), `conv_kdot` (kxk, K <= 1024) and
`im2col` + `gemm_cm8`. Paddle: 0 losses in 50 runs, accuracy unchanged. GcdrNet is 2x slower
on that family (its wide 3x3 convs need the staged tensor-core conv) and was 0 / 36 corpus runs
with the staged build, so its fused pipeline opts back in (`CompileOptions.StagedCoopConv`).
No shape rule separates the two workloads: staged for every batch-1 conv still lost the device
once in 50 Paddle runs; staged only for M >= 65536 was clean but costs GcdrNet 30–100%.
Revisit with a newer Intel driver (tested 32.0.101.8135).

Side findings: GcdrNet stacked its gc/dr arenas by `ExecPlan.ArenaFloats`, ignoring the graph
scratch (now used by im2col) — it uses `VkGraphModel.ArenaBytes` now. Conv/Pool/Resize/Ew + Relu
fusion lost the graph output value when the Relu was the last node.

## 2026-10-15 — Per-shape compile / emission: optimized JIT, not tier-0

GcdrNet's mixed-size corpus on Vulkan was 3–5% slower than the old engine although every graph
ran as fast. `DOTNET_TieredCompilation=0` closed the gap: the compiler, layout planner, graph
emission and const pool run a few dozen times per process, so their tier-0 code (no inlining,
iterator state machines) was most of a new size's cost. They carry
`MethodImplOptions.AggressiveOptimization` now and `VkGraphModel.Emit` fills a list instead of
yielding (corpus 37.3 → 35.6 ms vs 36.3 old). Graphs are built once per (model, shape, options)
key: a racing warm-up and the caller share one build; `VkBackend.Prefetch` builds on
below-normal-priority threads.

## 2026-10-15 — Metal: parity with the Vulkan op set, checked on the macOS runners

The Paddle macOS smoke fell back to the CPU ("ExecOp AvgPool not implemented"): the Metal
backend had been ported from GcdrNet's needs and lacked AveragePool, Softmax, batched MatMul and
HardSwish / Swish. Worse, its conv / gemm epilogues ignored hardswish and swish, so a plan
carrying them (an extension pass, or now the GPU gated-act fusion) would have produced wrong
values silently; the Vulkan generic `gemm` had the same hole (it now runs those two as a trailing
in-place unary). `GpuOpTests` runs whole graphs on every GPU backend present — Metal on the
`macos-26` (M1) runners — so a missing or wrong Metal kernel fails CI instead of falling back.

## 2026-10-16 — Paddle's "slower CPU stages" in Vulkan mode: preprocessing and per-shape compiles

The 2026-10-15 open item resolved into two causes. (1) Paddle's own resize + normalize was slower
in that process than the same arithmetic in `Sdcb.SimdOnnx.Imaging.ImageTensor` (one fused
3-channel write per pixel, ArrayPool scratch instead of per-session workspaces): det preprocessing
2.2 → 1.2 ms, rec 1.8 → 1.0 ms per image, bit-identical output. (2) Every first-seen input shape
compiles a plan and a Vulkan graph on the prefetch threads; that CPU work and its allocations
(~1 MB per graph, GCs pause every thread) overlapped the crop / det post-processing. Compile time
and allocations were halved (no RemoveAt / Insert in passes, no LINQ closures per node,
allocation-free layout helpers, bias folds cached per source arrays) with plans proven identical
(`bench --micro planhash`); with every shape compiled ahead, small runs 0.98x the old engine and
every CPU stage matches it. Rec-width bucketing would remove most compiles but costs accuracy
(rejected earlier), so first-seen shapes still cost ~1.2 ms of background CPU each.

## 2026-10-16 — GPU constants are packed outside the GC heap

Packing a model's weights for the GPU (tap-major, coopmat tiles, fp16) produced a float[] per
const plus a byte[] per upload chunk — 200 MB of large-object garbage for Paddle medium, which
the GC kept committed (+130 MB working set). Core packings now write into a reused native scratch
and the upload bytes live in native memory freed after the copy; extensions keep the
`Func<CValue, float[]>` packing API. Medium peak 1195 → 976 MB (old engine 956).

## 2026-10-17 — SE, attention and FPN merges are engine fusions, not Paddle extensions

The old OCR engine had hand-fused Vulkan kernels for squeeze-excitation (`se_fused`), attention
(`attn`) and the FPN top-down merge (`resize4add`, `catresize`). They are not OCR-specific — every
MobileNetV3 / PP-LCNet / EfficientNet has SE blocks, every transformer has attention, every FPN /
U-Net decoder has upsample + add — so they became structural fusions in `GraphCompiler` with
kernels on every backend instead of SimdPaddleOCR extensions:

- `ExecOp.ChannelGate`: GlobalAvgPool → 1x1 conv (+relu) → 1x1 conv → HardSigmoid / Sigmoid,
  matched through the zero-copy reshapes paddle2onnx puts between the convs (reader counts are per
  buffer, so a read through any view counts). `x + x·s` (RSE-FPN) folds into `x·(s + 1)`. Vulkan:
  `gap_part` + `se_gate` (2 dispatches instead of 5–6), Metal `se_gate`, CPU a scalar FC.
- `ExecOp.Attention`: `MatMul(Softmax([scale·] MatMul(q, kᵀ)), v)`, q / k / v followed back through
  transposes, a packed-qkv single-index slice, squeeze views and scalar muls, read in place through
  strides (`AttentionP`); the consumer's `[0,2,1,3]` transpose dissolves into the output layout. GPU
  plans only (CPU keeps the MatMul / Softmax kernels, which are already cache-friendly there).
  Head size up to 64; online softmax with k / v tiles in shared memory. On the B580 it is a wash
  for SVTR (T = 40, 2 blocks): the win is 20 fewer dispatches and no relayout copies.
- `ResizeNearest` takes an addend like bilinear (`FuseResizeAdd`): one pass per FPN level.
- Swish (`x·sigmoid(x)`) fuses on GPU plans too. `pw_dot` / `conv_kdot` / `gemm_cm8` apply it in the
  epilogue; `conv_cm` does not (an exp branch there cost GcdrNet's staged convs ~0.6%), so those
  convs and the generic kernels run a trailing in-place pass.

Not moved: GcdrNet's `UpHead` (resize → add → 1x1 conv, its full-resolution intermediate never
stored) stays a GcdrNet extension until a second model needs it; its AVX-512 / NEON / WASM kernels
are tuned to that network. Paddle's DB post-processing stays OCR code.

## 2026-10-17 — Multi-input graphs and the first non-consumer model (DocGainNet)

Graph inputs after the first take their shape from `CompileOptions.InputShapes` or the model's static
dims; the arena planner puts every input buffer first, back to back in input order, so the GPU
sessions keep staging one region (no per-input upload path) and `GetInput(i)` is a sub-span of it.
LeakyRelu / Tanh / Clip / DepthToSpace landed as plain ops with their own parameter records rather than
`ActivationKind` values: the CPU conv epilogues take an `ActivationKind` through ~70 signatures and
carry no parameter, so a fused LeakyRelu needs a parameterized activation first — worth doing when a
second model leans on it. The SpaceToDepth recognizer verifies the reshape / transpose chain's element
map (every output axis swept plus 4k pseudo-random points) instead of matching exporter-specific
reshape constants. A customer PR with the same four ops for CPU + Vulkan (sdcb/SimdOnnx#6) was closed
in favour of this superset (Metal, DCR, multi-input).
