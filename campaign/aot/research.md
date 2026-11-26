# A1 (5i) Research Ledger: Ahead-of-Time Compiled Inkling 2 Mini on Apple M5 Max

**Lane:** `lane/inkling-metal-aot` (`eb18fe252`)
**Target checkpoint:** `~/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd` (plus `~/models/Inkling-2-Mini-DFlash2@4f9088ea66c2`)
**Host / Toolchain:** Apple M5 Max (40 GPU cores, 48 GiB), macOS `27.2` (`26B123`), Xcode `27.2` (`27B52`), Metal `32023.922` (`metalfe-32023.922.4`, toolchain `27.2.47.0`), Zig `0.17.1`

## Executive Verdict

- **Installed Metal 3/4 AOT capability is complete on this host:** Offline `metal` + `metallib` + `applegpu-nt` natively target `applegpu_g17s (Apple M5 Max)`, while the macOS `27.2` SDK exposes `MTLBinaryArchive`, `MTLIndirectCommandBuffer`, `MTLResidencySet`, and the full `MTL4Compiler` / `MTL4Archive` / `MTL4PipelineDataSetSerializer` / `MTL4ArgumentTable` stack.
- **No safe hardware grid-wide barrier exists on Apple Silicon:** MSL `threadgroup_barrier(..., thread_scope_device)` sets *memory visibility* only; it does not synchronize across threadgroups. Persistent megakernels that spin-wait globally risk forward-progress deadlock under occupancy/preemption limits unless bounded by `recommendedPersistentThreadgroupsPerGridForThreadsPerThreadgroup`, which is far too small to hide DRAM latency on `3.872 GB` of active per-token weights and grids ranging from `1` to `9,216` threadgroups.
- **Phase B (static schedule with today's kernels) alone will not clear G3 (`>=1.15x` plain decode over current `main`):** V4 already merged (`d129b60f8..eb18fe252`), eliminating `424` of `1,302` 1-row dispatches (`14.16–15.52 ms/tok` in `roundtable`, `13.33 ms` rested 1K GPU time, `69.3–72.5 tok/s` served) and SC1 proved unchained host GPU-idle is only `2.7–3.3%` (`0.98–1.30 ms/round`). Removing `1,302->878` dispatch host overhead and early CB commits leaves only `~0.1–0.5 ms` (`~1–4%`) in Phase B and `~0.1–0.4 ms` (`~1–3%`) from bit-exact Phase C repacking.
- **Clearing G3 (`>=1.15x`, saving `>=1.74–2.02 ms/tok`) without arithmetic changes is tight and requires Phase D + Phase E together:** Against the `7.819 ms` (1K) / `8.012 ms` (4K) active-weights+KV bytes floor at `521 GB/s`, post-V4 decode still spends `~5.5–6.7 ms` above bytes across `878` dispatches (`144` controller dispatches + `49` `i2p_rms`/`rms3` + `48` `i2_sig_mul` + `96` `router`/`routepick` + `96` `split`/`combine` + `95` `mix` + per-dispatch barrier drain + host wait / CPU `argmax`). Realistic B+C+D+E ceiling is **`~1.75–2.75 ms` (`1.15x–1.25x`)** — achievable if and only if Phase D fuses controllers, `routepick`, `i2_sig_mul`, and tail passes without register spills, and Phase E eliminates host per-token sync/argmax round-trips.
- **G5 (`2,000 tok/s` prefill, `200 tok/s` decode):** Unseen prefill at 1K–8K (`2,004–2,067 tok/s` plain, `1,939–1,994 tok/s` with drafter in SC2) can clear `2,000 tok/s` with Phase F selective drafter tap capture + AOT warmup elimination; `200 tok/s` single-stream decode (`<=5.0 ms/committed tok`) is impossible in plain decode (`7.433 ms` weight-read floor caps plain at `<=134.5 tok/s`) and remains gated under DFlash2 by routed NVFP4 experts at 8–16 rows (`19.18–31.64 ms`, `1.6–2.0x` byte floor) plus drafter acceptance rates.

---

## 1. Target Hardware & Installed Metal 3/4 Toolchain Reality

### 1.1 Hardware and OS/Toolchain Pins (`campaign/host.json`)

Every hardware and toolchain metric below is directly recorded in `campaign/host.json` (L1–47) and verified via `xcrun --sdk macosx`:

| Property | Measured / Reported Value | Primary Source |
| --- | --- | --- |
| CPU | Apple M5 Max, 18 physical cores (`6` Super + `12` Performance) | `campaign/host.json` L3–8 |
| System RAM (`hw.memsize`) | `51,539,607,552` B (`48.00 GiB`) | `campaign/host.json` L9 |
| GPU cores & Metal tier | `40` GPU cores, `"Metal 4"` | `campaign/host.json` L12–13 |
| Recommended Max Working Set | `40,200,896,512` B (`37.44 GiB`) | `campaign/host.json` L37 |
| Max Buffer Length | `30,150,672,384` B (`28.08 GiB`) | `campaign/host.json` L39 |
| Unified Memory | `true` | `campaign/host.json` L38 |
| GPU Families | `apple7`, `apple8`, `apple9`, `apple10`, `metal3`, `metal4` all `true` | `campaign/host.json` L40–45 |
| OS & Xcode | macOS `27.2` (`26B123`), Xcode `27.2` (`27B52`) | `campaign/host.json` L16–23 |
| Metal Compiler | `Apple metal version 32023.922 (metalfe-32023.922.4)`, target `air64-apple-darwin27.2.0` | `campaign/host.json` L27–28, `metal --version` |
| GPU Native Architecture | `applegpu_g17s (Apple M5 Max)` (present in `applegpu-nt -available-archs macos 27.2`) | `metal-arch --default --name`, `applegpu-nt` |
| Zig Compiler | `0.17.1` | `campaign/host.json` L31 |
| Effective DRAM Streaming Floor | `521 GB/s` (used as campaign floor in `run.py` / `inkling-v4-fixed.md`; `i2a_dense` `lm_head` hits `875.4 µs` for `489.7 MB` -> `~559 GB/s` peak effective) | `inkling-v4-fixed-runs/run.py` L448, `inkling-v2-arith.md` L219 |

*(Note: Hardware limits not printed in `campaign/host.json` — such as exact per-core threadgroup SRAM capacity or hardware SLC size — must be queried at runtime via `MTLDevice.maxThreadgroupMemoryLength` / `pipeline.maxTotalThreadsPerThreadgroup`, never assumed. In-tree kernels allocate up to `32 * 32 * sizeof(half4) * 2 = 16,384` B in `i2_attn_split` / `i2_batch_split`, `18,432` B in `i2_dmv`, and `9,216` B+ in `i2_mxfp8` / `i2e_routed` without spilling.)*

### 1.2 Verified MacOSX27.2 SDK AOT & Execution Capabilities

Inspected directly in `/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.2.sdk/System/Library/Frameworks/Metal.framework/Headers/`:

1. **Offline `.metallib` & `MTLBinaryArchive` (`MTLLibrary.h`, `MTLBinaryArchive.h`, `MTLDevice.h`):**
   - `metal` (`air64-apple-darwin27.2.0`) -> `metallib` compiles `.metal` sources offline; `MTLDevice.newLibraryWithURL:error:` / `newLibraryWithData:error:` are already wrapped in `zig/src/metal/library.zig` (`Library.fromFile`, `Library.fromBytes` L57–73).
   - `MTLBinaryArchiveDescriptor` + `newBinaryArchiveWithDescriptor:error:` (`MTLDevice.h` L1248) + `addComputePipelineFunctionsWithDescriptor:error:` + `serializeToURL:error:` (`MTLBinaryArchive.h`) cache compiled `applegpu_g17s` machine code (PSOs) on disk or load pre-serialized archives without runtime AIR -> GPU ISA JIT compilation.
   - `MTLFunctionConstantValues` (`MTLFunctionConstantValues.h`) supports compile-time/PSO-time specialization constants (`[[function_constant(N)]]`) so offline `metallib` functions can specialize shapes (`K=2304`, `H_Q=24`, `E=128`, `WINDOW=1024`, `FULL=0/1`) cleanly.
2. **Metal 3 Compute Indirect Command Buffers (`MTLIndirectCommandBuffer.h`, `MTLIndirectCommandEncoder.h`, `MTLDevice.h`):**
   - `newIndirectCommandBufferWithDescriptor:maxCommandCount:options:` (`MTLDevice.h` L996).
   - `MTLIndirectCommandBufferDescriptor`: supports `MTLIndirectCommandTypeConcurrentDispatch` (`1<<5`) and `MTLIndirectCommandTypeConcurrentDispatchThreads` (`1<<6`), `inheritPipelineState`, `inheritBuffers`, `maxKernelBufferBindCount` (default `31`), `maxKernelThreadgroupMemoryBindCount`.
   - `MTLIndirectComputeCommand`: exposes `setComputePipelineState:`, `setKernelBuffer:offset:atIndex:`, `concurrentDispatchThreadgroups:threadsPerThreadgroup:`, `concurrentDispatchThreads:threadsPerThreadgroup:`, `setBarrier`, `clearBarrier`, `setThreadgroupMemoryLength:atIndex:`.
   - **Critical SDK Constraint #1 (No `setBytes:` on ICBs):** `MTLIndirectComputeCommand` has **no** `setBytes:length:atIndex:` method (verified absent in `MTLIndirectCommandEncoder.h`). Today `prefill.zig` and `group.zig` pass small structs (`setValue`) and per-row pointer tables (`setBytes`) inline on nearly every dispatch (`878`+ times/step). Moving to ICBs requires placing all dispatch parameter structs and row tables in a device-shared constant/parameter arena buffer and binding them by `(buffer, offset)`.
   - **Critical SDK Constraint #2 (Kernel buffer binding limit = 31):** All 16 `.metal` files in `zig/src/families/inkling2_loop/kernels/` have `max_buffer_idx <= 20` (`affine.metal`, `rows.metal`), safely within the default `31`-slot ICB and `MTL4ArgumentTable` limit.
   - **TensorFold Wrapper Gap (`zig/src/metal/library.zig` L119, L137):** `Pipeline.init(..., indirect = false)` sets `setSupportIndirectCommandBuffers: false` on every Inkling pipeline today, so ICB execution requires switching `indirect = true` (wrapped in `zig/src/metal/icb.zig` L13–71).
3. **Residency Management & `heavy.py` Double-Charge Hazard (`MTLResidencySet.h`, `zig/src/metal/residency.zig`, `campaign/heavy.py`, `campaign/ledger.md`):**
   - `MTLResidencySetDescriptor` + `newResidencySetWithDescriptor:error:` (`MTLDevice.h` L1328) keeps allocations resident per queue without per-command-buffer `useResource:` validation overhead. Currently DFlash2 uses `ResidencySet` (`qwen27/dflash/runtime_backend.zig` L94), while target `inkling2_loop` (`model.zig`, `state.zig`, `batch.zig`) holds **no residency set** and issues dynamic `useResource:` calls (`prefill.zig` L1047, L1148, L1179; `group.zig` L203, L280, L391, L417, L468).
   - **Memory Accounting Hazard (`campaign/heavy.py` L65–68, L87–96; `campaign/ledger.md` B2):** `heavy.py`'s `charge()` computes `sum(max(phys_footprint, resident_size)) + max(0, wired_growth - footprint)`. Metal wires clean file-backed `mmap` pages (`newBufferWithBytesNoCopy`) into GPU memory (`wired_growth`) outside `phys_footprint`, while faulted mapped pages also inflate `resident_size`, causing double-counting toward the `28 GB` cap (e.g., `llama.cpp` mmap stopped at `35.8 GB` charged with `0.6 GB` footprint). An AOT sidecar must prevent double-charging (e.g., streaming `pread` into `MTLResourceStorageModeShared` / `private` buffers + `F_NOCACHE`, or unmapping/evicting clean file-backed pages immediately, keeping peak charged RSS well below `26 GB`).
4. **Metal 4 (`MTL4*.h`, `MTLComputePipeline.h` macOS 27.0+):**
   - `MTLDevice.h` L1385–1519 exposes `newCommandAllocator`, `newMTL4CommandQueue`, `newCommandBuffer`, `newArgumentTableWithDescriptor:error:` (`MTL4ArgumentTableDescriptor.maxBufferBindCount <= 31`, `setAddress:atIndex:`), `newCompilerWithDescriptor:error:`, `newArchiveWithURL:error:`, `newPipelineDataSetSerializerWithDescriptor:` (`MTL4Archive.h`, `MTL4PipelineDataSetSerializer.h` for pre-compiled GPU binary datasets).
   - `MTL4ComputeCommandEncoder.h`: `setArgumentTable:`, `dispatchThreadgroupsWithIndirectBuffer:threadsPerThreadgroup:` (zero-CPU-roundtrip GPU-driven dispatch grid sizing!), `executeCommandsInBuffer:`.
   - `MTL4CommandEncoder.h`: fine-grained stage barriers (`barrierAfterQueueStages:beforeStages:visibilityOptions:`, `barrierAfterEncoderStages:beforeEncoderStages:visibilityOptions:` with `MTL4VisibilityOptionDevice = 1 << 0`).
   - **TensorFold Metal 4 Status (`zig/src/metal/metal4.zig` L11–12; `zig/src/families/inkling2_loop/run.zig` L36):** `tf_metal4` is compiled out of `tensorfold-native` (`metal4.zig` enabled only in `run.zig` for `TF_METAL4_TIMELINE` stamping), and `Model.load` opens a Metal 3 serial queue (`device.queue()`, `model.zig` L179).
   - **macOS 27 Persistent Kernel Hints (`MTLComputePipeline.h` L153–155, L298–299; `MTL4ComputePipeline.h` L63–65):** SDK adds `optimizeForPersistentKernel`, `forwardProgressUsage` (`MTLForwardProgressUsageAutomatic/Weak/SIMDGroupParallel`), `contentionRelief`, and `recommendedPersistentThreadgroupsPerGridForThreadsPerThreadgroup:`.

---

## 2. Prior-Art & Architectural Analysis for AOT Compiled GPU LLM Inference

| Technique / Prior Art | What Transfers to Inkling 2 Mini on M5 Max | What Fails or Breaks Under A1 / Hardware Constraints |
| --- | --- | --- |
| **Offline `.metallib` + `MTLBinaryArchive` / `MTL4Archive`** | Directly eliminates all `~25` runtime `MTLDevice.newLibraryWithSource` calls in `inkling2_loop` (and duplicate `PrefillScratch.init` + DFlash2 compiles) and pre-compiles `applegpu_g17s` binaries for **G2**. Zero arithmetic impact. | Does not change runtime kernel latency once pipelines are warm; first-touch weight/cache paging stall (`0.75–1.81 s` in P5c) still occurs unless buffers are faulted/resident at load. |
| **Metal Compute ICBs (`MTLIndirectCommandBuffer`) & `MTL4ArgumentTable`** | Pre-encodes the fixed 48-slot static dispatch graph (`878` -> `~350` dispatches after Phase D) into reusable command templates, replacing `objc_msgSend` dispatch loops and `useResource:` via `MTLResidencySet` + `MTL4ArgumentTable.setAddress`. Dynamic context length (`attnPartition` splits `1..16` and per-row offsets) updates cleanly via a GPU-written parameter arena + indirect dispatch grid (`dispatchThreadgroupsWithIndirectBuffer`). | Does **not** bypass `setBarrier` GPU pipeline drains between dependent dispatches (`cb.compute(.serial)` / `barrierAfterEncoderStages`), and `MTLIndirectComputeCommand` lacks `setBytes:`. `maxKernelBufferBindCount=31` is satisfied by all existing kernels (`<=20` bindings), provided fused Phase D kernels stay `<=31` bindings. |
| **MLX Compiled Graphs (`mx.compile`) & `megakernel-zoo` (`95e315c`)** | Confirms that static graph capture + fusing small elementwise ops + grouping verify+draft under one submit removes host round-trips (`inkling-sc1-chain.md` L67). | `megakernel-zoo`'s `16.4 ms` / `28.5 ms` expert numbers rely on changing arithmetic (`float4` partial sums + horizontal add, FP16 activations). Under TensorFold's bit-exact Decision 18 lane chains, `inkling-v3-experts.md` (L299–315) measured its runs-of-4 layout **1.23–1.51x slower** at 8–16 rows (reading `54.1` vs `44.6` expert weights/visit at 16 rows). SC1 proved host round-trips in TensorFold are already only `2.7–3.3%` of a round. |
| **llama.cpp Metal / ExecuTorch / TVM / IREE Metal** | llama.cpp's single prebuilt `.metallib`, single-CB encoding, and static workspace allocation pattern transfers directly (`inkling-b2-baselines.md`, `ledger.md` I0). | Generic MLIR/TVM/IREE Metal lowering cannot reproduce Apple10 cooperative `matmul2d<execution_simdgroups<4>>` tensor-unit tiles or exploit M5 Max's sequential FMA-chain `simdgroup_multiply_accumulate` property (`inkling-v2-arith.md` L22–35) needed to match bit-exact 1-row semantics across multi-row tiles. Keeping handwritten Apple10 Metal shaders compiled AOT outperforms generic compiler codegen. |
| **CUDA Graphs vs Persistent Megakernels (Global Barriers)** | Graph replay semantics transfer cleanly to Metal ICBs + `MTL4CommandBuffer` + `MTLSharedEvent` bounded loops (Phase E) without cross-TG spinlocks. | **Hardware deadlock & residency mismatch:** Verified in MSL `<metal_compute>` (`L65–84`) & `<metal_types>` (`L214–219`): `threadgroup_barrier(..., thread_scope_device)` lowers to `__metal_wg_barrier` (intra-TG execution barrier + device memory fence; **NOT** a grid-wide execution barrier). Spin-waiting on device atomics across threadgroups deadlocks whenever `grid_TGs > concurrent_resident_TGs` or macOS WindowServer/GPU preemption evicts a TG. Even with macOS 27's `recommendedPersistentThreadgroupsPerGridForThreadsPerThreadgroup`, a persistent megakernel faces two structural disqualifiers: (1) each token reads `3.872 GB` of weights (far exceeding on-chip cache/SRAM residency, so cross-layer fusion cannot keep weights on chip), and (2) dispatch widths vary from `1` TG (`i2p_routepick`) to `9,216` TGs (`i2p_gu`) and `6,252` TGs (`lm_head`), so capping the grid to resident persistent TGs under-utilizes memory-latency hiding on `lm_head` and MoE. **Decision:** forbid global cross-TG spin barriers; use intra-TG / encoder barriers only. |

---

## 3. Quantitative Ceiling & Feasibility Model for A1 (`G3` and `G5`)

### 3.1 Baseline 1-Row Plain Decode Budget (`main` post-V4) & Bytes Floor

Measured ground truth from `inkling-v4-fixed.md` (L33–52, L174–179, L247–269), `inkling-sc2-scorecard.md` (L78–89), and exact checkpoint header accounting (Section 1.2 of `inventory.md`):

- **Current `main` plain 1-row decode baseline:**
  - `phases-lane-a` rested GPU time (`inkling-v4-fixed.md` L258, L268): **`13.33 ms`** at 1K context (`main` pre-V4 `14.81`), **`14.40 ms`** at 4K context (`main` pre-V4 `15.72`).
  - `roundtable` 511-token decode (`inkling-v4-fixed.md` L174–179): `14.16 ms` (`code-1024`, `70.6 tok/s`), `14.66 ms` (`prose-1024`, `68.2 tok/s`), `15.24 ms` (`structured-1024`, `65.6 tok/s`), `15.48 ms` (`code-4096`, `64.6 tok/s`), `15.52 ms` (`prose-4096`, `64.4 tok/s`), `15.51 ms` (`structured-4096`, `64.5 tok/s`); geomean `~15.09 ms` (`~66.3 tok/s`).
  - Served plain cold decode (`inkling-sc2-scorecard.md` L78–89): **`69.3–72.5 tok/s`** (`13.79–14.43 ms/tok`).
- **Hardware Bytes Floor (`521 GB/s`, `run.py` / `model-0000*.safetensors` JSON headers):**
  - Exact 1-row active model weights (48 slots, 8 distinct experts/slot): **`3,872,476,976` B (`3.872 GB`) = `7.433 ms`**.
  - Plus 1K KV read (`40 * 1024 + 8 * 1024 = 49,152` pos * `4,096` B = `201.3 MB`) = **`0.386 ms`**, total weight+KV floor **`7.819 ms`** (`~7.87 ms` with ~`27.5 MB` activation traffic). At 4K (`302.0 MB` KV = `0.580 ms`), total weight+KV floor **`8.012 ms`**; at 16K (`704.6 MB` KV = `1.352 ms`), **`8.785 ms`**; at 32K (`1,241.5 MB` KV = `2.383 ms`), **`9.816 ms`**.

**V4 1K Class Attribution (`phases-lane-a` 1-row grouped / batch step, `inkling-v4-fixed.md` L249–268 & `phase_bytes` at `521 GB/s`):**

| Kernel Class | Current `main` (V4) Dispatches (1r) | V4 `alone` (ms) | V4 `in place` (ms) | Bytes Floor (`521 GB/s`) | Overhead (`in place - floor`) | Dominant Cost Mechanism |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `depth mixes & norms` (`i2pr_batch_mix*`, `i2p_rms*`) | `144` | `1.04` | `1.26` | `0.019` | `+1.24 ms` | `1`-TG launches (`1` row) x `95` slot mixes + `49` norm barriers (`~8–10 µs`/dispatch) |
| `controllers` (V4 final keeps 3-kernel 1r) | `144` | `0.91`* | `1.01`* | `0.436` | `+0.57 ms` | `24` bank slots x (`attn3` + `mlp1`) x `3` dispatches, small `8`-TG grids on `down` |
| `k/v conv, clips & store` (`i2_batch_qkv`) | `48` | `0.26` | `0.42` | `0.017` | `+0.40 ms` | `48` barriers + internal TG barriers; already `144` dispatches fused by V4 |
| `gate, convs & residuals` (`i2_sig_mul`, `conv`, `scatter`) | `152` | `0.40` | `0.54` | `0.014` | `+0.53 ms` | **`48` separate `i2_sig_mul` dispatches** + `96` `i2p_batch_conv` + `8` `i2p_scatter` |
| `router & pick` (`i2a_float` + `i2p_routepick`) | `96` | `0.40` | `0.56` | `0.109` | `+0.45 ms` | `48` x `i2p_routepick` (`1` warp / `32` threads/TG) serialized after `48` `i2a_float` |
| `affine 4-bit experts` (layers 20–23) | `8` | `0.30` | `0.34` | `0.275` | `+0.06 ms` | Already bandwidth-bound (`4` x up + `4` x down; `combine` folded into `conv`) |
| `split attention & combine` (`1K` ctx) | `96` | `1.22` | `1.73` | `0.386` | `+1.34 ms` | `48` x (`split` + DRAM `PART` + barrier + `24`-warp `i2_batch_combine`); `1.98 ms` at 4K |
| `projections + o-proj + NVFP4 experts + head` | `186` | `7.31` | `8.06` | `6.601` | `+1.46 ms` | `48` `o_proj` (`+0.23`), `44` `i2p_gu` (`+0.22`), `44` `i2e_down1` (`+0.25`), `head` (`2` dispatches) |
| `edges` (`embed`, `copy`, `pass_copy`) | `4` | — | — | `~0.00` | `~0.05 ms` | Minor copy/embed launch |
| **Total 1-row step (1K)** | **`878`** | *(sum `12.02`)* | **`13.33` batch (`13.70` grp1)** | **`~7.87`** | **`~5.46 ms` above floor** | **Note:** `phases` batch GPU `13.33 ms`; `roundtable` wall `14.16–15.24 ms` includes host wait/argmax (`~0.8 ms`) |

*\*Final V4 (`0394b8734`) kept main's 3-kernel 1-row controllers (`0.91 / 1.01 ms` in `phases-main-a`), avoiding the `0.16–0.18 ms` regression measured in `phases-lane-a` (`1.09 / 1.21 ms`; `inkling-v4-fixed.md` L277–279).*

### 3.2 Realistic Per-Phase Recovery Ceilings (Phases B–F)

All ceilings assume **Decision 18 bit-exactness** (no FP order/precision change) unless explicitly marked:

| Phase | Scope & Mechanism | Realistic 1-Row Plain Recovery (ms/tok) | Realistic % Over `main` (V4) | Cumulative Plain Decode (`14.5 ms` ref) | Why / Dominant Bound |
| --- | --- | --- | --- | --- | --- |
| **Baseline (A)** | Current `main` (V4 merged) control + instrumentation | `0.00 ms` | `0%` | `13.33 ms` GPU (`14.16–15.52 ms` wall) | Measured baseline (`878` dispatches, `2` CBs, CPU `argmax`) |
| **Phase B** | Static schedule (ICB / Metal 4 argument table) with **today's kernels** | **`0.10–0.50 ms`** | **`~1–4%`** | `~13.8–14.4 ms` wall | SC1 measured total unchained host GPU-idle = `2.7–3.3%` (`0.98–1.30 ms/round`, of which V4 early commit already hides `~0.4 ms`). Remaining CB commit latency + Metal 3 driver dispatch overhead across `878` serial dispatches caps Phase B well under `5%`. |
| **Phase C** | Static memory layout + **bit-exact weight repacking** (`weights.aot`, strip `636.5 MB` vision, pre-transpose 144 controllers, interleave `(scale,bias)` as `bf16x2`, pre-divide NVFP4 `global/den`, `MTLResidencySet`) | **`0.10–0.40 ms`** *(+`~1.5–2.5 s` load / `-0.75 GB` RAM)* | **`~1–3%`** incremental (`~2–6%` cum. B+C) | `~13.5–14.2 ms` wall | `lm_head` (`875 µs`) and `attn_proj` (`40.6 µs`) already run at `521–559 GB/s` (`dflash-kernels`); DRAM bandwidth already saturates. Eliminating scattered scale/bias TLB & buffer overhead helps mostly `o_proj`/`NVFP4`/`controllers`, plus huge **G2/G4** wins. |
| **Phase D** | **Generated Metal fusion** (Decision 18 bit-exact, no global barriers): (1) `i2_batch_combine + i2_sig_mul` (`-48` dispatches), (2) `router + routepick` (`-48`), (3) 1r `controllers` + post-RMS folds (`-96..144`), (4) `mlp_post` conv + next-slot `mix_norm` / `scatter` folds (`-40..80`), (5) specialized `k`-constant unrolled `o_proj`/`gu`/`down1` & barrier pruning | **`1.25–1.90 ms`** | **`~9–14%`** incremental (`~11–18%` cum. B+C+D) | **`~11.9–13.0 ms`** wall | Cuts dispatches from `878 -> ~400–480`, removing `~400+` inter-dispatch GPU drains (`~1–1.5 µs` each) + redundant DRAM scratch traffic (`attn[3072]`, `router[128]`, `n0..n2`, `normed[64]`, `act`). Tightest risk: register spills / low-TG under-occupancy (V4 `controls` failed at `1r` for `8`-TG `down`). |
| **Phase E** | **GPU-resident bounded token loop** (`i2v_argmax` for plain decode + GPU token-id feedback + `MTL4` / ICB multi-step execution + `MTLSharedEvent` streaming/cancel) | **`0.25–0.45 ms`** | **`~2–3.5%`** incremental (**`~13–21%` cum. B+C+D+E**) | **`~11.4–12.7 ms` wall (`~79–88 tok/s`)** | Removes per-token `800 KB` `logits` coherent read + CPU `200,058`-elem `argmax` loop + host wakeup & CB commit latency (`~0.25–0.45 ms/tok` gap between GPU `13.33 ms` and wall `14.16+ ms`). |
| **Phase F** | **Prefill + 8/16-row verify + DFlash2 reuse** (static prefill AOT, selective drafter tap capture/absorb `[P7]`, AOT DFlash2 precompiled `.metallib` + packed `q4` weights) | Plain decode `0 ms` (target: verify `-2..4 ms`, draft `-0.5..1.5 ms`, prefill `+3..8%`) | N/A on plain decode | Prefill `>=2,000 tok/s` (`1K–8K`); Spec `~120–245 tok/s` | Removes the `~3%` drafter tap-absorb prefill penalty (`SC2`), cuts DFlash2 `3.2–4.9 ms` draft GPU latency, and trims verify fixed dispatch overhead; **does NOT solve NVFP4 8–16r `19–32 ms` expert excess** without arithmetic/layout breakthroughs beyond V4. |

### 3.3 Explicit Assessment of G3 and G5

1. **Is G3 (`>=1.15x` plain decode geomean over current `main`) achievable without arithmetic changes?**
   - **Conditional YES, with narrow headroom:** `1.15x` over V4 (`13.33 ms` GPU / `14.16–15.52 ms` `roundtable` / `69.3–72.5 tok/s` served) requires saving **`>=13.04%` (`>=1.74 ms` on `13.33 ms` GPU, `>=1.85–2.02 ms` on `14.2–15.5 ms` wall)**.
   - **Phase B + Phase C alone will fail G3** (combined ceiling `~0.2–0.9 ms` = `~1.5–6.5%`), because `attn_proj`/`head` already saturate DRAM bandwidth and V4 already eliminated `424` dispatches + `~0.4 ms` encode idle.
   - **G3 rests squarely on Phase D + Phase E clearing `>=1.85 ms/tok` across the remaining bottlenecks:**
     - **Controllers + post-norms (`144` ctl + `48` bank `rms3`/`rms` = `192` of `878` dispatches, `~1.26 ms` vs `0.44 ms` floor):** largest un-fused 1-row dispatch concentration left in V4 (`0394b8734` kept main's `144` controller dispatches).
     - **Attention split/combine (`96` dispatches, `1.73 ms` at 1K, `1.98 ms` at 4K) + `i2_sig_mul` (`48` dispatches):** folding `i2_sig_mul` into `i2_batch_combine` (or `o_proj`) removes `48` barriers; specializing `combine` + reducing `split` barrier overhead recovers `~0.35–0.6 ms`. Notice at `32K` context (`2.383 ms` KV floor vs `0.386 ms` at 1K), dispatch overhead shrinks in share while KV bandwidth dominates, so Phase D/E must preserve `split` occupancy across `1..16` splits to avoid any long-context regression (G3 explicitly checks long context).
     - **Routed NVFP4 experts (`44` `i2p_gu` + `44` `i2e_down1`, `4.21 ms` in place vs `3.027 ms` floor) + `router/pick` (`96` dispatches, `0.56 ms` vs `0.109 ms` floor):** fusing `router + routepick` (`-48` dispatches) + specialized `K=2304/1152` `gu`/`down1` tails targets `~0.5–0.8 ms`.
2. **Is G5 (`2,000 tok/s` prefill, `200 tok/s` single-stream decode) achievable?**
   - **Prefill (`2,000 tok/s`):** Plain prefill already hits `2,004 / 2,067 / 2,011 tok/s` at 1K/4K/8K in SC2 (`2,063 / 2,130 / 2,072 tok/s` in PH2). Closing the `~3%` default-drafter gap via P7 selective tap capture/absorb + AOT staging clears `2,000 tok/s` at 1K–8K. Reaching `2,000 tok/s` at `16K–31.7K` (`1,885` / `1,766` today) additionally requires attention/relative-bias bandwidth optimization (P4 `16K` attention bucket `48.7 ms` vs `17.6 ms` at `1K` per 512 rows).
   - **Decode (`200 tok/s` single stream = `<=5.0 ms/committed tok`):** Physically impossible via 1-row plain decode (`7.433 ms` active weight floor -> `134.5 tok/s` theoretical zero-overhead max). Requires DFlash2 speculation; under SC2 (`3.3–7.8` committed tokens/round), non-code prompts (`~3.4–4.2` tokens/round) need **`17–21 ms` rounds**, whereas today's verify spends **`19.18–31.64 ms` in 8–16-row NVFP4 experts alone**. Without arithmetic changes (rejected/separate mode under Decision 18/roadmap) or higher multi-token acceptance, AOT Phase F improves speculative tok/s noticeably on code/medium prompts but cannot reach `200 tok/s` uniformly across all prompts.

### 3.4 Falsification & Kill Criteria by Phase

Per `docs/roadmap.md` L215–216 ("Three failed variants of one idea park it. A negative verdict is a valid outcome"):

| Phase | Hypothesis Under Test | Quantitative Go Criterion | Kill / Park Criterion (Falsification) |
| --- | --- | --- | --- |
| **A (Control)** | Baseline attribution & bytes model matches `main` | Reproduces V4 `phases` / `roundtable` within noise & passes Q4 suite | Any unexplained `>3%` drift from Q4/`SC2` baseline before AOT edits |
| **B (Static Schedule)** | ICB / Metal 4 argument table with existing kernels cuts host/driver overhead | `G1` bit-exact pass + `>0%` (`~1–4%`) plain decode gain + zero encode stalls | **Park** if static schedule is **slower** than 2-CB early commit on `batchStep`, or requires `>1.5x` constant buffer bloat; confirm host bound (`<4%`) and proceed to C/D |
| **C (Memory & Repack)** | Stripping vision, pre-transposing controllers, `(scale,bias)` packing & residency improves load/RAM/decode | **`G2`** (`<=50%` cold load / zero runtime MSL compile), **`G4`** (`<=18.5 GB` plain ready under `heavy.py`, no double-charge), `>=0%` decode speed | **Kill/revert** any layout repack that exceeds `26 GB` load limit, trips `heavy.py` `28 GB` cap via mmap double-charge, breaks Decision 18 bitwise parity, or slows `lm_head`/`attn_proj` bandwidth |
| **D (Metal Fusion)** | Fusing controllers, `routepick`, `i2_sig_mul`, `conv+mix` cuts `>=1.25 ms/tok` bit-exactly | Bit-exact on `batch_test` + `dflash-widths hashes` (228/228), cumulative B+C+D `>=1.11x` plain decode | **Kill any individual fusion** after 3 failed variants if it loses to unfused at 1r/4K (`>0%` regression from register spill/low TG occupancy, e.g. V4 `controls`), fails Decision 18 parity, or causes `16K/32K` regression |
| **E (GPU Loop)** | GPU `argmax` + bounded multi-token loop saves `>=0.25 ms/tok` cleanly | **Clears G3:** geomean plain decode `>=1.15x` over `main` (95% CI excluding `1.0`), `32K` non-regressing, cancellation latency `<=1` burst | **Kill GPU loop** if G3 already clears without loop complexity, or if cancellation lag / watchdog timeout / `MTLSharedEvent` poll adds tail latency or breaks `Host` stop/EOS semantics |
| **F (Prefill & Spec)** | AOT + P7 selective taps + AOT DFlash2 pushes prefill `>=2,000 tok/s` (1K–8K) & boosts spec decode | Prefill `>=2,000 tok/s` 1K–8K with drafter on; speculative geomean `>=1.03x` over `main` + `100%` greedy parity (`Q4`) | **Park** any drafter/verify fusion that regresses 4K acceptance/ms, fails `Q4` task parity, or breaches `26 GB` load / `28 GB` cap with DFlash2 loaded |
