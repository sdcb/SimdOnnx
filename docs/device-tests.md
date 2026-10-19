# Real-device validation (old consumer engines vs SimdOnnx)

Procedure for an agent re-testing SimdOnnx on one of the owner's machines. The bar
(see README / docs/decisions.md): on every device each consumer must be **as fast or faster, with
the same or lower peak memory and the same or better accuracy** than its old engine.

## Common procedure

1. **Layout**: the three repos side by side in one parent directory (names matter, the build
   resolves `..\Sdcb.SimdOnnx`):
   `Sdcb.SimdOnnx` (branch `main`), `SimdPaddleOCR` (branch `feature/simdonnx`),
   `Sdcb.GcdrNet` (branch `main`). Clone what is missing (`github.com/sdcb/SimdOnnx`,
   `github.com/sdcb/SimdPaddleOCR`, `github.com/sdcb/SimdGcdrNet`), `git pull` what exists.
2. **Baselines (old engines)** as detached worktrees under `Sdcb.SimdOnnx/.baseline-bin/`
   (git-ignored): `paddle-main` = SimdPaddleOCR `origin/main`, `gcdr-main` = Sdcb.GcdrNet `2b8fdfe`
   (last main before the SimdOnnx switch). Never junction / symlink anything into a worktree and
   then `git worktree remove --force` it: git follows the link and deletes its target.
3. **New side** builds against the engine source: `-p:UseLocalSimdOnnx=true` on both consumers, so
   the result reflects SimdOnnx `main`, not the pinned package.
4. **Data**: Paddle `dataset/` (git-ignored, 100 fixed synthetic images):
   `dotnet run --project test/Sdcb.SimdPaddleOCR.TestData -c Release -- --out dataset`; GcdrNet
   corpus: `tools/fetch-fixture.ps1`.
5. **Correctness first** (on the real device, GPU tests are skipped in CI):
   `dotnet test tests/Sdcb.SimdOnnx.Tests -c Release -f net10.0` in SimdOnnx, the consumers' test
   projects on the new side. Note every skipped GPU test and why (`no coopmat device` is expected on
   devices without cooperative matrices).
6. **Device breadcrumb**: record the `[vk] device=...` line printed with `GCDRNET_VK_VERBOSE=1`
   (vendor, driver, subgroup size, `coopmat` / `m8` / `lite` / `vec4` / `dot` / `ncd` / `fullSg`); it decides the
   routing (`src/Sdcb.SimdOnnx/Backends/Vulkan/VkDevice.cs`, `VkGraph.cs`).
7. **A/B, interleaved**: old and new back to back per case, alternating which goes first, at least
   3 rounds (6 when ratios are within 3%), medians. Machine idle, high-performance power plan, AC
   power on laptops, one warm-up run discarded.
   - Paddle: `tools/ab-paddle.ps1` (pass `-Old` / `-New` / `-Dataset` for this machine; `-Configs`
     covers `sharp:1`, `sharp:4`, `sharp:16`, `vulkan:4` by default) over tiny / small / medium.
     Compare total median, det / cls / rec stage means, peak working set, exact lines.
   - GcdrNet: `tools/ab-gcdr.ps1` (CPU t=1/4/16 and Vulkan at 1600x1152 and 3200x2304; ms and peak).
8. **Routing probes** when the new side is slower on Vulkan (one variable at a time, same A/B):
   `SIMDONNX_VK_SMALLDOT=0` (no small-K / wide-subgroup `pw_dot` / `conv_kdot` routes),
   `GCDRNET_VK_VEC4=0` (no vec4 elementwise; Paddle forces vec4 on through `CompileOptions`),
   `SIMDONNX_VK_F16=0` (fp32 storage), `SIMDONNX_VK_CM1X1=0`, `SIMDONNX_VK_SG16STAGED=1`,
   `SIMDONNX_VK_STAGE_WC=0`, `GCDRNET_VK_NCDIRECT=0` (Intel no-coopmat iGPUs: back to `conv_tile` / `gemm`\r
   instead of `pw_dot` / `conv_kdot` / `gemm_nc`). Device details: `GCDRNET_VK_VERBOSE=1` (the breadcrumb of step 6),
   `SIMDONNX_STATS=1` (graph-cache misses and compile cost at exit). Per-shader GPU time: the GcdrNet
   benchmark's `--vkprof <h> <w>` (`VKPROF_NODES=1` adds the top dispatches); CPU operator profile:
   `SIMDONNX_PROF=1`. For CPU: the ISA switches `DOTNET_EnableAVX512=0` / `DOTNET_EnableAVX2=0`
   apply to both sides.
9. **Report**: write `docs/devices/<machine>.md` in SimdOnnx (English): hardware, OS, driver,
   .NET version, commits of all five trees, breadcrumb, tables (old / new / ratio / peak MB /
   accuracy), skipped tests, probe results, conclusion against the bar.
10. **Fixes**: allowed when a gap is found, but only device-gated (by `VkDevice` capabilities or
    vendor) so the other devices keep their routes, with a test, and re-measured on this device.
    Work on a branch `device/<machine>` in SimdOnnx, push it and open a PR to `main`; do not push to
    `main` directly (four machines test in parallel), do not publish NuGet packages. Model-specific
    kernels belong in the consumer repos through the extension API, not in the engine.

## Android (phone over adb)

- Hosts: SimdPaddleOCR `test/Sdcb.SimdPaddleOCR.AndroidBench` (`run.ps1` builds / installs /
  pushes models and dataset / runs / pulls `bench-out/android/`; `ab.ps1` alternates backends with a
  cool-down and prints median, last-50 median, peak, thermal, accuracy), Sdcb.GcdrNet
  `bench/Sdcb.SimdGcdrNet.AndroidBench/run.ps1`. Both need the Android SDK, a JDK 21 and the
  `android` workload (`-Adb` / `-AndroidSdk` / `-Jdk` or `ADB` / `ANDROID_HOME` / `JAVA_HOME`).
- Old and new APKs share a package id: build both once (old from the baseline worktree, new from the
  feature branch with `-p:UseLocalSimdOnnx=true`), keep the two signed APKs, and alternate
  `adb install -r` per round. Thermals dominate on a phone: 60 s+ cool-down, record the thermal
  status, discard rounds that throttled, compare last-50 medians as well as all-image medians.
- In the Vulkan route the Adreno 750 is a `dot=True` device (no coopmat, subgroup 64): 1x1 convs take
  `pw_dot`, other dense convs `conv_kdot`.

## Known gaps to look at

- Ported from the old engines but only emulated on the B580: small-K `pw_dot` (NVIDIA / AMD),
  Adreno `pw_dot` / `conv_kdot`, vec4 memory-bound kernels for Paddle on every device.
- Not ported: the Radeon 880M direct-load `sg32d` GEMM; SimdPaddleOCR's fused attention, SE,
  catresize, resize4add, convt2s2, sigmoid_out kernels; `conv_dw4t`; global queue priority.
- Paddle peak memory is +3–8% on Vulkan, +9–13% on CPU (B580 / 5950X) — check it on each device.
- AVX-512 tier: bit-identical to AVX2 in CI (GitHub x64 runners have AVX-512), speed never measured
  on a real AVX-512 desktop / laptop.
