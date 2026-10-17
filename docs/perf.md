# Performance: consumers before / after the SimdOnnx switch

Machine: AMD Ryzen 9 5950X (16C/32T, AVX2, no AVX-512), Intel Arc B580 (Vulkan), Windows, .NET 10.
Every row is an **interleaved** A/B: the old (main) and new (feature/simdonnx) binaries run back
to back per case, alternating which goes first. The machine's absolute speed drifted by up to 35%
between sessions, so only compare numbers that come from the same table.

Reproduce: `tools/ab-paddle.ps1` (raw JSON in `docs/ab/paddle/{old,new}`), `tools/ab-gcdr.ps1`.

Last run 2026-10-15: SimdOnnx `fb775f4` (consumers built with `UseLocalSimdOnnx=true`);
new = Paddle `feature/simdonnx` `ec9645d`, GcdrNet `feature/simdonnx` `a2a0bc9` + `main` 1.0.0
merged; old = Paddle `main` `33ff8a8`, GcdrNet `main` `2b8fdfe`.

## SimdPaddleOCR (e2e over the 100-image dataset, median ms per image)

`ratio` = new / old (< 1 is faster). Stage columns are old/new mean ms. `exact` = exactly
recognised lines (old/new).

| case | old | new | ratio | det | cls | rec | peak WS MB | exact |
|---|---:|---:|---:|---|---|---|---|---|
| tiny cpu 1w | 84.5 | 82.4 | **0.98** | 22.6/20.5 | 10.9/12.1 | 42.8/41.4 | 501/533 | 742/742 |
| tiny cpu 4w | 50.5 | 46.6 | **0.92** | 23.8/22.2 | 13.0/17.5 | 60.8/60.9 | 520/563 | 742/742 |
| tiny cpu 16w | 47.3 | 46.5 | **0.98** | 23.4/21.8 | 14.3/21.0 | 112.2/97.5 | 539/596 | 742/742 |
| small cpu 1w | 298.5 | 230.9 | **0.77** | 80.1/36.6 | 12.8/12.9 | 182.2/161.5 | 603/666 | 950/950 |
| small cpu 4w | 202.5 | 141.6 | **0.70** | 91.1/46.2 | 23.3/22.8 | 311.5/279.7 | 650/760 | 950/950 |
| small cpu 16w | 214.2 | 158.6 | **0.74** | 98.9/48.2 | 33.9/24.9 | 640.5/578.9 | 719/872 | 950/950 |
| medium cpu 1w | 1084.7 | 1094.6 | 1.01 | 218.3/201.5 | 11.0/14.7 | 827.4/845.2 | 1105/1190 | 1004/1004 |
| medium cpu 4w | 532.1 | 526.3 | **0.99** | 226.3/207.7 | 22.2/25.8 | 1051.5/1061.7 | 1172/1446 | 1004/1004 |
| medium cpu 16w | 652.3 | 654.9 | 1.00 | 231.5/213.2 | 19.4/55.9 | 2368.7/2483.8 | 1269/1776 | 1004/1004 |

Vulkan rows of the first run lost the device (since fixed, see decisions). Current Vulkan
(SimdOnnx `be3ef34`, Paddle with `Sdcb.SimdOnnx.Imaging` preprocessing), interleaved, 3–5 rounds,
4 workers (`.rerun/alt.sh`; this machine's old-engine tiny median alone wanders 17–24 ms, so
tiny / small ratios move by ±5% between sessions):

| model | ratio | peak WS MB (old/new) | exact |
|---|---:|---|---|
| tiny | 0.94–0.98 | 581/599 | 740/741 |
| small | 1.00 (0.98 with every shape warm) | 661/715 | 950/955 |
| medium | 0.99 | 956/981 | 1003/1003 |

What closed it (2026-10-16): preprocessing through `ImageTensor` (det 2.2 → 1.2 ms, rec 1.8 → 1.0 ms
per image, same bits); per-shape compile about half the time and allocations; GPU const packing
outside the GC heap (medium peak 1195 → 976 MB). With all shapes compiled ahead (`--warmup 100`)
small runs 0.98x with every stage at parity or faster: what is left at `--warmup 2` is first-seen
shape compiles overlapping the CPU stages.

- **CPU**: at parity or faster in every configuration (small det is ~2x faster). Accuracy identical.
  Memory: per-key packing and plans (one build shared by concurrent workers) cut medium 16w peak
  1776 → 1388 MB, still above old (1262); medium 4w 1280 vs 1172, small 16w 812 vs 717. The live
  heap after a full GC is ~90 MB above old: same reachable weights, but twice the allocation churn
  per run (956 vs 510 MB) and per-shape statics.
- **Vulkan**: device losses fixed (0 in 50 stress runs, was ~1 in 7). At parity or faster on all
  three models; accuracy equal or better. Peak working set +3% (tiny, medium) / +8% (small): the
  managed heap and GC commit are now at or below old, the rest is native (about 16 MB fixed plus
  8 MB per worker; not the const pool's memory type, the session staging slack or the cached
  command buffers — each ruled out by experiment).
- **Metal**: the macOS runners now run the Paddle smoke on Metal (device engaged, detected-line
  counts equal to the CPU) and `GpuOpTests` on M1; not timed against the old engine.

## SimdGcdrNet (`Sdcb.SimdGcdrNet.Benchmark`, e2e ms)

| case | old | new | ratio | peak WS MB |
|---|---:|---:|---:|---|
| cpu 1600x1152 t=1 | 1091.0 | 893.3 | **0.82** | 307/304 |
| cpu 1600x1152 t=4 | 323.7 | 292.2 | **0.90** | 308/312 |
| cpu 1600x1152 t=16 | 208.1 | 163.3 | **0.78** | 341/344 |
| vulkan 1600x1152 | 16.6 | 16.5 | 0.99 | 175/176 |
| cpu 3200x2304 t=16 | 676.7 | 614.7 | **0.91** | 905/904 |
| vulkan 3200x2304 | 65.1 | 65.0 | 1.00 | 205/208 |

CPU 10–22% faster, Vulkan at parity, memory unchanged; matches the previous run.

Mixed-size corpus (every image a different size, so per-shape compile counts), interleaved:
Vulkan 35.6 vs 36.3 ms per image, peak 1192 vs 1249 MB; CPU 349 vs 415 ms, peak 1145 vs 1150 MB.
PSNR goldens identical.

## Known gaps / next steps

1. **Paddle Vulkan native memory** +8% peak on small (+3% tiny / medium): unattributed native
   growth during the run (~16 MB + ~8 MB per worker).
2. **Paddle CPU memory** ~+9–13% peak: allocation churn per run, per-shape caches.
3. **Devices**: RTX 3080 Ti and Snapdragon 8 Gen 3 measured on the real hardware
   (`docs/devices/`); Radeon 880M and UHD 770 pending. The 880M direct-load `sg32d` GEMM is not ported.
   Squeeze-excitation, attention and nearest-upsample + add are engine fusions now (2026-10-17).
   Open on the 3080 Ti: GcdrNet Vulkan e2e 1600x1152 8.9 → 9.8 ms (3200x2304 1.02x) although its
   GPU kernel time is 34% lower (14.3 → 9.4 ms) — host-side cost (staging memory type, upload /
   download path), to probe there with `SIMDONNX_VK_STAGE_WC=0`; Paddle medium Vulkan 1.03x.
4. **Device re-creation after a loss** still fails with `ErrorIncompatibleDriver` on the B580
   (no longer triggered).
5. **Metal performance** unmeasured against the old engines (correctness checked in CI).

## DocGainNet (2026-10-17, new model: no old engine to compare with)

`DocGainNet/release/docgainnet.onnx` (sibling repo), image 1x3x1152x1600 + thumb 1x3x512x512, a RealDAE
photo; `bench --micro multi` against onnxruntime 1.31 CPU (`tools/ort_ref_multi.py`).

| engine | median ms | min ms | vs ORT fp32 |
|---|---:|---:|---|
| onnxruntime CPU, 16 threads | 154.0 | 147.5 | — |
| SimdOnnx CPU, 16 threads | 115.8 | 96.1 | max \|d\| 2.6e-6 |
| SimdOnnx Vulkan (B580, fp16 storage) | 14.45 | 14.41 | PSNR 68.7 dB |

The first working version ran the CPU in 540 ms: the graph's full-resolution tail (a per-pixel 3x4
colour affine on channel slices, a pixel unshuffle, mixed NCHW input / NHWC features) went through
single-threaded per-element index loops. Generic fixes, no model-specific kernel: coalesced, threaded
strided gather / broadcast (`Kernels/Strided.cs`) behind Transpose, Slice and broadcast Elementwise;
the layout planner prefers channels-last for mixed-layout elementwise / Concat (one cached relayout
of the NCHW input); `x·s -> Tanh -> ·t` folds into one scaled Tanh; scalar-const elementwise is one
affine pass on the GPUs; a reshape / rank-6 transpose / reshape pixel unshuffle is recognised as
SpaceToDepth. Paddle / GcdrNet plans are unchanged (planhash) and their CPU / Vulkan runs equal.
Not done: LeakyRelu (23 nodes, ~7 ms CPU / 0.5 ms GPU) still runs as its own pass after each conv.

## Engine fusions for SE / attention / FPN merge (2026-10-17)

Paddle, interleaved 3 rounds against the old engine, SimdOnnx with the fusions (consumers at
`UseLocalSimdOnnx=true`):

| case | old | new | ratio | det graph | rec graph | peak WS MB | exact |
|---|---:|---:|---:|---|---|---|---|
| tiny vulkan 4w | 21.7 | 21.0 | **0.97** | 6.09/3.98 | 6.38/5.83 | 584/605 | 740/743 |
| small vulkan 4w | 33.1 | 32.2 | **0.97** | 7.60/5.88 | 15.97/14.66 | 665/715 | 950/956 |
| medium vulkan 4w | 55.8 | 52.4 | **0.94** | 21.49/18.52 | 27.08/24.81 | 958/994 | 1003/1003 |
| tiny cpu 4w | 64.5 | 58.5 | **0.91** | 29.3/26.6 | 79.6/76.5 | 518/556 | 742/742 |
| small cpu 1w | 407.5 | 317.5 | **0.78** | 96.9/49.7 | 274.4/238.2 | 603/658 | 950/950 |
| small cpu 4w | 202.4 | 136.5 | **0.67** | 96.9/46.2 | 320.4/276.4 | 652/735 | 950/950 |
| medium cpu 4w | 546.0 | 529.0 | **0.97** | 230.2/207.8 | 1086.0/1085.7 | 1175/1282 | 1004/1004 |

Graph-only on the B580 (det 960x736, fusions on / off): det 3.11 → 2.86 ms, small det 5.20 → 4.79,
medium det 19.66 → 19.33; 162 → 112 dispatches. GcdrNet plans are unchanged (planhash) and its
Vulkan e2e stays at 16.5 ms.

## What closed most of the Vulkan gap (2026-10-14)

| change | effect |
|---|---|
| per-model const pool (entries shared across shapes and output truncations) | medium peak WS 1946 → 1175 MB; no more ~170 MB weight re-pack + upload per CTC/logits graph |
| two-pass GAP (`gap_part`/`gap_sum`, chunked over pixels) | det GPU time 3.9 → 2.7 ms (GAP was a third of it) |
| zero-copy staging (`Input`/`Output` are the mapped staging buffers) | det 736x960 5.6 → 3.8 ms wall |
| LRU graph cache 96 → 256 | rec graphs 154 compiles → 88 (no thrash) |
| Paddle: compile device graphs at `Reshape`/`ReshapeMany` on the thread pool | det/rec setup overlaps preprocessing |
| Paddle: CTC head (MatMul+ArgMax) on the thread pool in growing batches | rec CPU head 1879 → 1412 ms per 100 images (medium) |
| GPU-timestamp profiler (`GCDRNET_PROF=1`, VK_QUERY_TYPE_TIMESTAMP was 1 instead of 2) | per-dispatch device times instead of submit round trips |
