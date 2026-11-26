# A1 Phase D: Generated Bit-Exact Metal Kernel Fusion and Microarchitecture Study

Phase D profiles the 14 kernel classes of the 48-slot decode graph on M5 Max (`applegpu_g17s`) and tests targeted bit-exact Metal kernels in `zig/src/families/inkling2_loop/aot/aot_fused.metal`. Every kept variant preserves 0-bit logit parity against `main` (`98a7c1028`) and beats the unfused baseline in balanced 4-round interleaved A/B pairs (`base/cand`, `cand/base`, `base/cand`, `cand/base`) on a warm GPU.

## 1. Per-Class GPU Breakdown (`bundle-c --opt=0`, `ctx-1024`, 12 reps)

Measured via `tf-inkling2-mini-aot profile` (isolated class replay per command buffer, 48 slots):

| Kernel Class | Calls / Step | Isolated GPU (ms) | Share of Sum | Notes |
| --- | ---: | ---: | ---: | --- |
| `attn_proj_48` | 48 | 2.024 | 18.7% | Affine-8 Q/K/V/R/Gate/Sink projections (`bfloat2 sb`) |
| `gu_nv_44` | 44 | 2.004 | 18.5% | NVFP4 routed expert gate+up (top-8 of 128, `inter=1152`) |
| `controllers_24` | 24×8 | 1.245 | 11.5% | Cross-pass norm + down + apply + rms (passes 2 & 3) |
| `lm_head_1` | 1 | 0.859 | 7.9% | `200,058 x 2,304` affine-8 (`~570 GB/s`, already bandwidth-bound) |
| `mix_norms_all` | 96 | 0.823 | 7.6% | Depth-mix + RMSNorm kernels |
| `o_proj_48` | 48 | 0.806 | 7.4% | Affine-8 `o_proj` (`2304 x 3072`) |
| `split_48` | 48 | 0.718 | 6.6% | Split-K attention (40 sliding 1K + 8 full) |
| `down1_nv_44` | 44 | 0.708 | 6.5% | NVFP4 routed expert down (`2304 x 1152`) |
| `combine_sig_48` | 96 | 0.544 | 5.0% | Attention split combine + `sig_mul` |
| `router_pick_48` | 96 | 0.362 | 3.3% | Float router MV + top-8 pick |
| `qkv_48` | 48 | 0.234 | 2.2% | K/V conv, Q/K clip, rel-bias, KV store |
| `aff4_exp_4` | 8 | 0.220 | 2.0% | Affine-4 experts (`layers 20..23`) |
| `mlp_post_48` | 48 | 0.176 | 1.6% | Post-MLP conv + residual |
| `attn_post_48` | 48 | 0.114 | 1.1% | Post-attention conv + residual |
| **Isolated Sum** | **878** | **10.837** | **100%** | Full-step GPU is `13.225 ms` (`+2.39 ms` serial barrier/cache drain across 878 dependent dispatches) |

## 2. Retained Fusions (`--opt=7`, `878 -> 782` Dispatches, `-96` Dispatches/Step)

Interleaved 4-round A/B ablations on `bundle-c` (`ctx-1024`, warm GPU, `memory_pressure` `80–83%` free):

| Opt Flag | Kernel(s) | Dispatches | Interleaved Δ GPU (ms) | Bit-Exact vs `--opt=0` | Mechanism |
| --- | --- | ---: | ---: | --- | --- |
| `D1` (bit 0) | `i2_aot_combine_sig_mul` | `878 -> 830` (`-48`) | **-0.043 to -0.091 ms** | **true** | Fuses attention split combine with `gate` sigmoid multiply; keeps exact `pv[16][4]` register-prefetched accumulation order. |
| `D2` (bit 1) | `i2_aot_gu_nv` / `i2_aot_gu_nv_packed` + `i2_aot_down1_nv` | `878 -> 878` | **-0.186 to -0.418 ms** | **true** | 8 SIMDgroups/TG (256 threads) with threadgroup-SRAM `lut[16]` + `scales[256]` tables, pre-divided `global_scale`, and `ulong2(wg, wu)` + `uchar2(sg, su)` coalesced stream on `bundle-c2`. |
| `D3` (bit 2) | `i2_aot_mix2_inv` + `i2_aot_mix_inv` + `i2_aot_ctl_down3` + `i2_aot_ctl_down1` + `i2_aot_ctl_apply{3,1}_gf2` | `878 -> 830` (`-48`) | **-0.451 to -0.461 ms** | **true** | Emits 16B `inv` RMS scalars from depth-mix to eliminate `i2_controls_norm` (`-24`) and `i2p_rms_pair` (`-24`); 6-way unrolled `float4` loads in `ctl_down3`/`ctl_down1` (2 SGs/TG = 96 TGs for full M5 GPU saturation, `-0.24 ms` from unrolling alone); `float2(g, f)` interleaved apply on `bundle-c2`. |
| **`D1+D2+D3` (`--opt=7`)** | All retained fusions | **`878 -> 782` (`-96`)** | **-0.671 to -0.778 ms** | **true** | Combined interleaved GPU drop: `13.685 -> 13.014 ms` (and `13.930 -> 13.152 ms` on warmer run); rested `bundle-c2` reaches `12.918 ms` GPU (`13.150 ms` wall). |

### Experimental Long-Context Kernel (`D9`, bit 8)

- `i2_aot_qkv_perm` + `i2_aot_split_perm`: writes KV cache head-major with `half4`-permuted head dimension (`(tid & 31) * 4 + (tid >> 5)`) so `i2_aot_split_perm` loads contiguous `half4` vectors with zero bank conflicts.
- Verified **0 differing bits** against `main` (`logits-c-opt263.bin` vs `logits-main.bin`). Neutral at 1K (`~0.00 ms`), deferred to Phase F / autotuning for 16K and 31.7K evaluation.

## 3. Parked Variants (Empirical M5 Max Microarchitecture Findings)

Every variant below was implemented, benchmarked in balanced interleaved A/B pairs on `ctx-1024`, and rejected/removed from `aot_fused.metal` (respecting the 3-variant parking rule):

| Variant | Idea Tested | Δ GPU (ms) | Bit-Exact | Root Cause on M5 Max |
| --- | --- | ---: | --- | --- |
| `D2-alu` | Compute E2M1/E4M3 in ALU instead of LUT | `+3.160 ms` | true | ALU bit-manipulation chain far slower than TG-SRAM LUT lookup. |
| `D2-const` | `constant float*` LUT instead of TG SRAM | `+0.090 ms` | true | Constant cache misses vs 1-cycle TG SRAM `lut[16]`/`scales[256]`. |
| `D15` | `simd_shuffle(e2m1_reg, nibble)` register LUT | `+4.280 ms` | false | `simd_shuffle` throughput bottleneck + divergent lane values; massive regression. |
| `D12-2col` | 2 output cols/SG in `gu_nv` (reuse `x` loads) | `+0.156 ms` | true | Register pressure cut SIMD occupancy below memory-latency hiding threshold. |
| `D12-k4` | 4-way K-hoisted `ag[4], au[4]` in `gu_nv` | `+0.456 ms` | true | Live `xv[16]` + 8 `ulong` registers spilled to thread memory. |
| `D13` | 2 output rows/SG in `dense8_sb` / `batch_proj8_sb` | `+0.041 ms` | true | Halved threadgroup count and increased register pressure vs 1 col/SG. |
| `D11` | 4 cols/SG in `ctl_apply` | `+0.307 ms` | true | Reduced parallelism across M5's 40 GPU cores (72 TGs vs 288 TGs). |
| `D10` | Streaming `combine_sig` / specialized `S==2` path | `+0.010 ms` | true | At 1K all 48 layers use `S=16` splits; `pv[16][4]` prefetch beats streaming. |
| `D4` | Fused `conv_scatter` (`batch_conv + scatter`) | `+0.151 ms` | true | Extra register/branch overhead on all 48 slots outweighed 8 eliminated scatters. |
| `D4-r` / `D14` | Fused `router + routepick` (1-TG / 32-SG) | `+0.014 to +0.034 ms` | true | Serializing 128-row router MV into 1 TG underutilized the 40-core GPU vs 16-TG `float_mv`. |
| `D5` / `D6` | Vectorized / half-memory `split` attention | `+0.370 to +0.856 ms` | true | Broke `i2_batch_split`'s tuned `R=32` tile and `half` TG packing. |
| `D7` / `D8` | Fused `attn_conv + mix` / `mlp_conv + mix` | `+0.095 to +0.471 ms` | true | Single-TG `mix` serialized the `2304`-wide parallel `batch_conv` pass. |

## Takeaways

1. **Occupancy beats arithmetic reuse on M5 Max decode:** 1 output column per SIMDgroup with low live registers (`<= 32 regs`) and TG-SRAM LUTs consistently outperformed 2-col/4-col tiling and K-hoisting.
2. **Deep unrolling wins when register pressure is low:** 6-way `float4` unrolling in `i2_aot_ctl_down3`/`ctl_down1` (where only 1 accumulator + 12 `float4` loads are live) alone delivered `-0.24 ms`.
3. **Hard floor of bit-exact Phase D:** Combining `D1+D2+D3` with Phase C2 repacking recovers **`~0.75 ms` GPU** (`-96` dispatches/step, `13.225 -> 12.918 ms` rested, `13.685 -> 13.014 ms` warm). The remaining `+2.1 ms` above isolated class sum sits in cross-dispatch barrier drain across the 782 data-dependent dispatches of Inkling 2 Mini's 48-slot recurrent graph, where single-TG bottlenecks (`mix`, `rms`, `routepick`) prevent multi-layer megakernel fusion without changing arithmetic or dropping occupancy.
