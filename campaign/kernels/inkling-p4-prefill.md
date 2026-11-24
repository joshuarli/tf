# P4: Inkling 2 Mini oQ4e served prefill toward 2,000 tok/s

Item P4 of Decision 17. Lane `lane/p4-prefill`, head `665006b84`, against main's code at `3881b3d8c` (main's later
commits change no code). M5 Max, oQ4e `3eea042751cd`, greedy, served chunk 1,024 rows, context 32,768,
`INKLING_LOAD_LIMIT_GB=26`. Raw runs are in [`inkling-p4-prefill-runs/`](inkling-p4-prefill-runs/).

**Verdict.** The default path meets 2,000 tok/s at no length, on main or on the lane. Every default-on change keeps
the prefill bytes, and the oMLX gate on the lane gives D2's numbers exactly. Against main, the lane cuts the GPU span
per 512 rows to 0.81-0.94× in paired split timelines, and the everything-else bucket alone falls by about half.
Served throughput rises 1-12% at 4K-31.7K in both paired trials. At 1K the two pairs disagree (0.89× and 1.02×), so
1K is inconclusive.

The opt-in `INKLING_PREFILL_HALF=both` changes bytes and stays off by default. It served 2,048 tok/s at 4K and 2,059
at 8K in one valid trial, 1.18-1.36× the lane default bracketing it, and it passes the oMLX gate. But it agrees with
the exact path's top-1 on 97.49% of the rows whose margin is at least 0.1, short of the roadmap's 99.5%, so it cannot
become the default as it stands. Quality dev with it scores 7/11, as the exact path does: it loses one case and
gains one.

The 1K and held-out 1K prompts are slow because of the served host's fork-point chunking, not the kernels. With
prefix reuse off the lane serves 1K at 1,693 tok/s, held-out as fast as dev, and the first request at 985 tok/s.
That host change (`zig/src/native/inkling2_loop_host.zig`, outside P4's files) is the next prefill lever. The parked
scratch commit `f0ff849e6` (here `d9e41207a`) stays: it cut the median first request from 2.89 to 2.45 s, inside a
1.2-3.7 s spread across trials of one binary.

| Prompt | main (2 trials) | lane default | lane / main (2 pairs) | lane `both` (opt-in) | 2,000 met |
| --- | ---: | ---: | ---: | ---: | --- |
| first request (1,258 tokens) | 368-430 | 342-480 | 1.05, 1.19 | 1,024 | no |
| 1K (six prompts) | 1,076-1,155 | 989-1,263 | 0.89, 1.02 | 1,679 | no |
| 4K | 1,432-1,438 | 1,454-1,591 | 1.01, 1.06 | 2,048 | default no; `both` yes, one trial |
| 8K | 1,462-1,485 | 1,540-1,625 | 1.04, 1.10 | 2,059 | default no; `both` yes, one trial |
| 16K | 1,381-1,437 | 1,461-1,568 | 1.02, 1.09 | 1,850 | no |
| ~31.7K | 1,290-1,315 | 1,395-1,503 | 1.08, 1.12 | 1,715 | no |

Served client tok/s, medians of each trial's prompts at a length, one fresh server a trial. The lane column spans
its four default trials (`t3-head-a`, `t3-head-b`, `t4-default-a`, `t4-default-b`); the ratios are per-request
lane/main over the two interleaved pairs of session t3.

## Measurement conditions

Owner work on the host ran through the whole window: a bctl qualification smoke test from another session (krun
VMs, screen grabs through WindowServer at about 30% CPU and 10% GPU, later a release build), fseventsd near 100%
CPU for stretches, an XProtect scan, and memory pressure that dropped free pages to about 140 MB with the compressor
at up to 534,672 pages/s. Every trial ran under the gates (see Method). Trials that tripped one were repeated and are
kept as invalid in `inkling-p4-prefill-runs/invalid/`:

- profiles: `t3-1k-1024-B1` try 1 (6.95 cores busy) and `t3-4k-1024-F2` try 1 (compressor 7,073/s)
- runner bench: `t3-1024-B2`, `t3-16384-B2` and `t3-31744-B1` try 1 (compressor 12,540, 64,102, 7,647/s)
- served curves: `t3f-4-with` tries 1 and 2 (gate; compiler observed), `t3f-6-without` try 1 (gate),
  `t4-inputs-a` try 1 (post-run CPU), `t4-both-a` try 1 (compiler observed)

One trial passed the gates but ran about 50% slow in every bucket, the relative-bias kernel included:
`t3-1k-128-F1`. It is discarded. The 128-row comparison uses the quieter session-t2 pair instead.

Served curves also record the host's state right after the server is ready (serve_curve's `valid`). Eight
default-path trials of lane builds carry a compressor flag there: `t3-head-a`, `t3-head-b`, `t4-default-a`,
`t4-default-b` and `t3f-1`, `t3f-3`, `t3f-4`, `t3f-6`. No main trial and no variant trial does. The flag also came up
on both no-scratch controls (`t3f-3`, `t3f-6`), so the scratch commit is not its cause. Since outside load was
present, results are given as paired ratios from interleaved trials, and as ranges, not as single rates.

## Method

- Builds: each side exported from its commit (`git archive`) and built with `zig build native tf-inkling2-loop-run
  -Dcpu=apple_m1`. No build or compile ran inside a timed window; the gate refuses one.
- Every timed trial ran under `campaign/heavy.py` (28 GB cap) and the timing gates. Before it:
  `gpu-gate-ready -1` (GPU and compressor quiet) and `perf-gate-ready` (at most 4 cores busy). During it: a watch
  for `zig`, `clang` or `metal` processes. After it: `perf-gate-ready -1`.
- Served curve: `comparators/serve_curve.py tensorfold` with one fresh server a trial (prompt cache 0), each side
  interleaved as main, lane, lane, main. A trial sends a first request of 1,258 tokens, then unseen prompts: six at
  1K (dev and held-out code, prose and structured) and two each at 4K, 8K, 16K and 31,744.
- Split timelines: `tf-inkling2-loop-run ... profile-prefill CHUNK` with `TF_INKLING_PREFILL_PROFILE_BASE=0` and
  `TF_METAL4_TIMELINE=<chunks a pass>`, after eight warm chunks. Each is reduced by `inkling_profile.py timeline
  --split` and bucketed as in P3c, in ms of GPU time per 512 rows. Identical trials of one binary differ by 5-15% in
  every bucket (the GPU's clock state), so every table gives both trials of each side.
- Runner wall time: `tf-inkling2-loop-run ... bench 1 0 1024 PASSES` runs fresh-cache prefill passes of 1,024-row
  chunks in one process; pass 1 is the process's first prefill.

## What changed

Every default-on change keeps the prefill bytes. The fused-prefill test (`prefill_staged_test.zig`) runs oQ4e
prefill twice, with the new paths on and off (`PrefillScratch.fuse`). It compares the logits and every cache buffer
(keys, values and the four convolution rings) at chunk patterns of 137/163/9, 9/137/163 and 163/137/5/1/3 rows. A
planted fault in each new kernel made it fail.

| Commit | Change | Why the bytes hold |
| --- | --- | --- |
| `d595a6d7a` | The attention projections that read one input run as one staged GEMM: in_proj on local layers; q, k and v on full layers. Each source is dequantized once and each input split once; the o projection's split applies the attention gate | each 64-column tile keeps its operands, tile and order |
| `d9e41207a` | The prefill scratch's pages are committed at setup (`f0ff849e6` reimplemented on the current buffers) | no arithmetic |
| `aea9d6660` | One kernel a row does the depth-mix gather, the mix and the RMS norm after it | same thread map, SIMD sums and partial order |
| `203a1ab84` | The bank messages' gather and four-query mix use the same kernel | as above |
| `e2bf963a5` | One kernel clips q, r and k and stores the k and RMS-normalized v halves | a lane holds what the old threadgroup's active simdgroups held; the idle ones added exact zeros |
| `05af1197c` | A chunk commits every sixth slot, so the GPU starts while the host encodes | no arithmetic |
| `a40ecdf89` | The ring store launches only the rows that write, the KV store copies four halves a thread, and the post-attention and post-MLP convolutions add into the residual | same bytes moved; same taps, order and rounding |
| `5e4eb9249` | One kernel runs a loop controller's gain and forget products (f32 MPP, 16 x 32 tiles) and writes the update | same MPP op and operands for each product, same update expression |
| `3482ec282` | Opt-in `INKLING_PREFILL_HALF` (`experts`, `inputs`, `both`), off by default; other values refused since `c9ffa68f7` | the default is unchanged |
| `8a4679663` | `zig build test-inkling2-loop-filtered -Dinkling-filter=TEXT` | build only |

`e2880656c` merges main's V2 decode tiles. `1e0f68e2a` and `665006b84` remove the five changes under "Tried and
dropped".

## Correctness

All at `665006b84` unless named.

- Fused-prefill byte test (`zig build test-inkling2-loop-filtered "-Dinkling-filter=oQ4e fused prefill"`): passes
  with the default and with `INKLING_PREFILL_HALF=experts`. With `inputs` or `both` the fused path rounds the staged
  inputs and the unfused path does not, so the test does not apply to those.
- Suite (`python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu,inkling-omlx-oracle`, exit 0, no
  skips; `suite/`): inkling-host 25/25 tests, inkling-gpu 165/165 tests in 17/17 steps. The oMLX oracle gate passes
  with top-1 2,863/3,015 (94.96%), mean KL 0.0194994 and row-weighted 0.0639952. Those are D2's numbers (0.0195), so
  mean KL does not move.
- Quality dev on oQ4e (`quality/`): the lane and main score 7/11 each (code 2/3, reasoning 3/3, retrieval 2/2,
  structured 0/3; `retrieval-t05` excluded by the context budget). All 11 scored replies are byte-equal between the
  two (`quality/lane-vs-main.md`). The suite marks the workload FAIL and make exits 1 on both sides because cases fail
  (structured output unsupported, `code-c02` truncated at 8,192 tokens); that is main's behaviour, not a regression.
  The lane run's record notes GPU contention from another lane's tests (device utilization 97%); its replies equal
  main's anyway.
- Quality dev with `INKLING_PREFILL_HALF=both` (`quality/half-both-vs-lane.md`) scores 7/11 (code 3/3, reasoning
  3/3, retrieval 1/2, structured 0/3) against the exact lane's 7/11, with six replies byte-equal. It loses
  `retrieval-t01` and gains `code-c02`. The lost case thinks to the 8,192-token cap and writes no content, where the
  exact path answers `[[QX-7419]]` after 5,345 tokens. The first 1,000 characters of reasoning, all that the record
  keeps, are identical on both, so the reply diverges later in a long trace: a length failure, not a wrong
  retrieval. `code-c02` hits the same cap on the exact path and finishes with passing tests under `both`. One lost
  case is within the rule's limit of one; the top-1 rule above is what `both` fails.
- `zig build native -Dcpu=apple_m1`: exit 0; `tensorfold-native` sha256 `5cf29037fe18...`.
- `tools/zig/lean_new.py` against main: 256 problems, 256 on the base, 0 new.

## Profile attribution

Split timelines of main (B) and the lane (F), in the order B1, F1, F2, B2, ms of GPU time per 512 rows
(`profiles/t3-*`). The lane/main ratio of the summed span is F1/B1, F2/B2: 0.878 and 0.878 at 1K, 0.944 and 0.856
at 4K, 0.906 and 0.811 at 16K. B2 at 16K ran in a slow clock state (relative bias 7.5 against 6.4).

1K prompt, 1,024-row chunk:

| Bucket | B1 | B2 | F1 | F2 |
| --- | ---: | ---: | ---: | ---: |
| Routed NVFP4 matrices | 95.5 | 101.8 | 92.4 | 97.9 |
| Dense projections (staging, split, GEMM) | 68.5 | 73.2 | 61.3 | 64.7 |
| f32 projections (router, controllers) | 8.2 | 8.3 | 9.1 | 9.6 |
| Grouped low-occupancy experts | 30.7 | 32.9 | 30.6 | 31.9 |
| Attention | 17.0 | 18.1 | 17.6 | 18.2 |
| Affine-4 experts | 11.7 | 10.9 | 10.6 | 10.5 |
| Relative bias | 5.3 | 5.5 | 5.5 | 5.6 |
| Everything else | 48.3 | 49.1 | 24.0 | 25.3 |
| Summed span | 286.2 | 300.7 | 251.4 | 264.0 |

4K prompt, 1,024-row chunks:

| Bucket | B1 | B2 | F1 | F2 |
| --- | ---: | ---: | ---: | ---: |
| Routed NVFP4 matrices | 103.3 | 95.8 | 106.7 | 89.9 |
| Dense projections (staging, split, GEMM) | 77.2 | 71.2 | 72.2 | 60.4 |
| f32 projections (router, controllers) | 8.8 | 8.1 | 10.9 | 9.1 |
| Grouped low-occupancy experts | 37.4 | 34.7 | 39.9 | 32.8 |
| Attention | 28.3 | 26.3 | 30.1 | 25.1 |
| Affine-4 experts | 11.9 | 10.9 | 12.5 | 10.8 |
| Relative bias | 5.9 | 5.6 | 6.2 | 5.6 |
| Everything else | 50.3 | 47.5 | 27.1 | 23.8 |
| Summed span | 324.2 | 301.0 | 305.9 | 257.7 |

16K prompt, 2,048-row chunks:

| Bucket | B1 | B2 | F1 | F2 |
| --- | ---: | ---: | ---: | ---: |
| Routed NVFP4 matrices | 107.2 | 116.4 | 105.3 | 104.8 |
| Dense projections (staging, split, GEMM) | 77.7 | 88.9 | 72.9 | 72.8 |
| f32 projections (router, controllers) | 8.9 | 10.0 | 10.8 | 11.0 |
| Grouped low-occupancy experts | 15.7 | 17.4 | 15.9 | 16.1 |
| Attention | 47.7 | 54.8 | 48.7 | 48.6 |
| Affine-4 experts | 11.5 | 12.2 | 11.3 | 11.3 |
| Relative bias | 6.4 | 7.5 | 6.9 | 6.8 |
| Everything else | 56.2 | 62.8 | 29.0 | 29.2 |
| Summed span | 332.1 | 371.1 | 301.0 | 300.9 |

The 128-row chunk that every served prompt runs (1K prompt in 128-row chunks; session t2, main against
`c9ffa68f7`, which still had the paired expert and 64-row attention tiles): 866.6 ms per 512 rows on main against
811.5 on the lane. Grouped low-occupancy experts take 444.7 of main's 866.6. Dense projections fall from 130.2 to 98.2,
and everything else from 53.8 to 34.1.

Where the everything-else bucket went (1K, ms per 512 rows, both trials a side):

| Work | main kernels | main | lane kernels | lane |
| --- | --- | ---: | --- | ---: |
| q, r, k clips and the k/v half stores | `i2p_clip_scaled`, `i2p_clip`, `i2_rms_kv_store`, `i2p_kv_store` | 9.1-9.7 | `i2pr_clips`, `i2pr_kv_store` | 1.9-2.0 |
| depth-mix gather, mixes and the norm after them | `i2p_gather_rms`, `i2p_mix`, `i2p_mix4` | 13.4-13.8 | `i2pr_mix_norm` | 6.6-7.0 |
| convolutions and residual adds | `i2p_conv`, `i2p_conv_store`, `i2_add` | 9.0-9.2 | `i2p_conv`, `i2pr_conv_add`, `i2pr_ring_store` | 5.8-6.3 |
| attention gate multiply | `i2_sig_mul` | 1.4-1.8 | folded into the o projection's split | 0 |
| controller update | `i2p_ctl` | 4.0-4.1 | folded into `i2af_gate_ctl` (f32 bucket) | 0 |

The controller fusion moves its cost into the f32 bucket, which is why that bucket grows. At 16K, main's
`i2af_gemm` and `i2p_ctl` take 54.6 ms a 2,048-row chunk against 41.3 for the lane's `i2af_gemm` and
`i2af_gate_ctl` (session t2), 24% less.

The routed NVFP4 matrices (85-116 ms per 512 rows), the dense projections (60-89) and attention at long context
(15-18 at 1K, 46-55 at 16K) remain the large buckets. The lane removed launch, staging and memory traffic around
them; their inner loops are main's.

## Engine against served

Runner wall time of fresh-cache 1,024-row-chunk prefill, the best warm pass, ms (`bench/t3-*`; order B1, F1, F2,
B2 at each length):

| Tokens | B1 | F1 | F2 | B2 | F1/B1 | F2/B2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 511 | 543 | 598 | 603 | 1.06 | 0.99 |
| 4,096 | 3,189 | 2,389 | 2,430 | 2,507 | 0.75 | 0.97 |
| 16,384 | 13,143 | 10,115 | 10,302 | 10,641 | 0.77 | 0.97 |
| 31,744 | 22,226 | 21,286 | 21,316 | 22,835 | 0.96 | 0.93 |

B1 ran slow at 4K and 16K (its 16K warm pass took longer than its cold one), so the F2/B2 pair is the cleaner one:
0.93-0.99 at 4K and above, with 1K inconclusive. The engine alone, one 1,024-row chunk at a time, reaches 1,670-2,000
tok/s at 1K and 1,600-1,700 at 4K. The served path loses the difference to the chunk schedule below.

## Holdout 1K prompts and the served tail chunks

The served host splits a fresh prompt at fork points on a 128-token grid that starts 256 tokens before the prompt's
end (prefix reuse on, fork window 256, interval 128), and prefills the last token alone. So after its bulk, every
prompt runs a 128-row chunk, a tail chunk of 0-127 rows and a 1-row chunk (`chunks.py`, a copy of the host's loop):

| Prompt tokens | Chunks |
| ---: | --- |
| 1,027 (dev code) | 896, 128, 2, 1 |
| 1,032 (dev prose) | 896, 128, 7, 1 |
| 1,033 (dev structured) | 896, 128, 8, 1 |
| 1,019 (held-out code) | 768, 128, 122, 1 |
| 1,014 (held-out prose) | 768, 128, 117, 1 |
| 1,003 (held-out structured) | 768, 128, 106, 1 |
| 1,258 (first request) | 1,024, 128, 105, 1 |

The held-out 1K prompts are slow because their lengths fall just under the next grid point. Each one carries a
106-122-row tail chunk where the dev prompts carry 2-8 rows. A 128-row chunk costs 217 ms on main and 203 on the
lane (866.6 and 811.5 ms per 512 rows), half of it in the grouped f32 experts: 128 rows over 128 experts top-8 leave
most experts below the tensor-unit thresholds (14 rows gate/up, 8 down). So a tail of about 110 rows adds about
0.15-0.2 s to a 1K prompt.

Content does not explain the gap. On the GPU, one 1,024-row chunk of each corpus file's first 1,024 tokens costs no
more for held-out than for dev (main, ms per 512 rows, `profiles/t2-src-*`):

| Source | Summed span | Grouped experts |
| --- | ---: | ---: |
| dev code | 276.7 | 30.2 |
| dev prose | 285.3 | 32.4 |
| dev structured | 308.4 | 46.7 |
| held-out code | 264.9 | 28.7 |
| held-out prose | 289.5 | 36.8 |
| held-out structured | 289.1 | 33.9 |

With prefix reuse off (`INKLING_PREFIX_REUSE=0`, one trial, `t3v-noreuse-a`) the host sends 1,024-row chunks and a
remainder. The lane then serves the six 1K prompts in 0.55-0.66 s (1,576-1,871 tok/s, median 1,693; dev 1,691,
held-out 1,695), against 0.84-1.02 s with reuse on in the adjacent lane trial.

## First request and the scratch commit (`f0ff849e6`)

The first request (1,258 tokens; chunks 1,024, 128, 105, 1) took 2.45-3.68 s of server prefill in 10 of the 12
default-path trials, main and lane alike, 10.38 s in one under heavy compressor activity, and 1.33 s in one. The
scratch commit's A/B ran the lane with and without `try s.commit(device)` (`t3f-*`, order with, without, without,
with, with, without):

| Trial | first request (s) | conditions |
| --- | ---: | --- |
| with 1 | 2.454 | compressor flag after load |
| without 2 | 2.654 | valid |
| without 3 | 2.888 | compressor flag |
| with 4 | 2.646 | compressor flag (tries 1-2 tripped the gate) |
| with 5 | 1.331 | valid |
| without 6 | 10.375 | compressor at 534,672 pages/s after load |

Every with-trial is at or below the without-trials' median. The medians are 2.45 s with and 2.89 s without, and the
one valid pair is 1.33 against 2.65 s. Verdict: keep `d9e41207a`, the reimplementation of `f0ff849e6`. It moves
about 0.6 GB of scratch page faults to setup, which costs no measurable startup time against the 4-28 s spread in
model load. Its gain is real but smaller than the spread the host's memory state puts on the same request. In the
runner, a process's first 1,024-token prefill costs 266-540 ms more than its warm passes on both sides.

What the rest of the first request's cost is remains open. In every trial with `INKLING_PREFILL_HALF` set, or with
prefix reuse off, it took 1.18-1.84 s (six trials). The two default trials that bracket the t4 variants took 3.68
and 2.62 s. Prefix reuse off runs the default kernels and was as fast as the variants, so the kernels alone do not
explain the gap. This window could not isolate the cause.

## Tried and dropped

Each of these kept the bytes (fused-prefill test) and was dropped on quiet split timelines.

| Commit | Change | Evidence | Removed in |
| --- | --- | --- | --- |
| `fb6190c93` | grouped f32 experts: eight columns of one expert share rows staged in threadgroup memory | grouped bucket 445.3 -> 739.6 ms per 512 rows at 128-row chunks and 30.1 -> 52.1 at 1K (`t2-1k-128-E1`, `t2-1k-1024-E1` against the head without it) | `1e0f68e2a` |
| `55bda59bc` | 16-row tensor-unit tiles for experts of up to 16 rows | routed NVFP4 134.9 -> 147.8 at 128-row chunks (affine-4 32.0 -> 27.6), net slower | `1e0f68e2a` |
| `e828da920` | double-buffered expert tile weights | -5% to +3% per kernel, no consistent sign | `665006b84` |
| `787bf80bb` | 64-row attention tiles | +7.2% at 1K, +7.5% at 4K, +0.7% at 16K | `665006b84` |
| `528c4ac52` | opt-in 128-row staged GEMM tiles (`INKLING_PREFILL_TILE`, `c9ffa68f7`) | +5.1% at 1K, +2.2% at 4K, +1.2% at 16K | `665006b84` |

Kernel time a chunk (ms), each shape against the kernel it replaces, in adjacent runs of one binary (`t2-*-H2` had
the paired and 64-row shapes on; `t2-*-X1` had them off and the 128-row GEMM on):

| Prompt/chunk | Shape | Replaced kernel | Shape's kernel | Change |
| --- | --- | ---: | ---: | ---: |
| 1K/1024 | 64-row attention | 34.2 | 36.6 | +7.2% |
| 1K/1024 | paired gate/up | 114.2 | 117.2 | +2.7% |
| 1K/1024 | paired down | 66.0 | 64.6 | -2.2% |
| 1K/1024 | 128-row GEMM | 75.1 | 78.9 | +5.1% |
| 4K/1024 | 64-row attention | 59.2 | 63.7 | +7.5% |
| 4K/1024 | paired gate/up | 137.2 | 139.1 | +1.3% |
| 4K/1024 | paired down | 76.2 | 74.8 | -1.9% |
| 4K/1024 | 128-row GEMM | 94.5 | 96.5 | +2.2% |
| 16K/2048 | 64-row attention | 189.6 | 190.9 | +0.7% |
| 16K/2048 | paired gate/up | 262.6 | 256.1 | -2.5% |
| 16K/2048 | paired down | 153.4 | 145.8 | -5.0% |
| 16K/2048 | 128-row GEMM | 194.2 | 196.5 | +1.2% |

## Opt-in variants

`INKLING_PREFILL_HALF` changes bytes, so it stays off. `experts` sends every routed expert to the tensor-unit
matrices (thresholds 1, half activations), which takes the grouped f32 experts out of small chunks. `inputs` rounds
the staged dense projections' inputs to one half instead of a hi and a lo half, so each staged GEMM runs once
instead of twice. `both` does both.

Served curves (tok/s; `t3v-*` in session t3, after the lane's default trials; `t4-*` interleaved default, inputs,
both, default):

| Trial | first | 1K | 4K | 8K | 16K | 31.7K | conditions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `t3v-experts-a` | 681 | 935 | 1,229 | 1,412 | 1,530 | 1,456 | valid |
| `t3v-both-a` | 915 | 1,350 | 1,802 | 1,917 | 1,729 | 1,678 | valid |
| `t4-default-a` | 342 | 1,167 | 1,497 | 1,553 | 1,478 | 1,395 | compressor flag after load |
| `t4-inputs-a` | 982 | 1,245 | 1,697 | 1,763 | 1,616 | 1,584 | valid |
| `t4-both-a` | 1,024 | 1,679 | 2,048 | 2,059 | 1,850 | 1,715 | valid |
| `t4-default-b` | 480 | 1,263 | 1,591 | 1,625 | 1,568 | 1,503 | compressor flag after load |
| `t4-both-a` try 1 | 1,065 | 1,650 | 1,971 | 1,900 | 1,717 | 1,693 | invalid: compiler observed |

Per-request ratios against the mean of the two bracketing default trials at 1K / 4K / 8K / 16K / 31.7K: `inputs`
1.01 / 1.10 / 1.11 / 1.06 / 1.09, and `both` 1.36 / 1.33 / 1.30 / 1.21 / 1.18. `both` against the adjacent lane
default in t3 (`t3-head-b`): 1.26 / 1.19 / 1.19 / 1.15 / 1.16. `both` reached 2,000 tok/s at 4K and 8K in one of its
two valid trials.

Unresolved: `experts` alone was slower than the lane default at 1K-8K in its one trial (t3). Yet `both` beats
`inputs` alone by 8-35%, which the two halves' separate effects do not add up to. Settling it needs an interleaved
`experts` trial.

Correctness of each variant. The oMLX gate compares against the oracle (prefill 512, the 16 oracle histories, 3,015
rows). The agreement columns compare the lane's exact path with the variant (`agreement/*-vs-exact.md`), which is the
roadmap's rule for a changed arithmetic: mean KL within 1.10× the baseline's, and top-1 agreement of at least 99.5%
where the reference's top-2 margin is at least 0.1.

| Variant | gate top-1 | gate mean KL | gate | top-1 against exact | at margin ≥ 0.1 | KL against exact |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| exact (default) | 2,863 (94.96%) | 0.0194994 | pass | - | - | - |
| `experts` | 2,872 (95.26%) | 0.0185822 | pass | 2,899 (96.15%) | 2,795/2,874 (97.25%) | 0.00736 |
| `inputs` | 2,857 (94.76%) | 0.0191293 | pass | 2,900 (96.19%) | 2,797/2,874 (97.32%) | 0.00759 |
| `both` | 2,856 (94.73%) | 0.0199132 | pass | 2,899 (96.15%) | 2,802/2,874 (97.49%) | 0.00606 |

Gate bounds: top-1 at least 94.392%, mean KL at most 0.0292627, row-weighted mean KL at most 0.0962956. Each
variant's mean KL to the oracle is within 1.10× the exact path's (0.02145). None meets the top-1 rule: at margin ≥ 0.1
each agrees with the exact path on 97.25-97.49% of rows, against the 99.5% required. So under the roadmap's rule no
half variant can become the default as it stands, `both` included, whatever quality dev shows. Quality dev with
`both` is in Correctness.

## The next prefill lever

The host's fork-point chunking (`zig/src/native/inkling2_loop_host.zig` and the prefix code; outside P4's files,
unchanged here). With prefix reuse on, every fresh prompt pays a 128-row chunk, a 0-127-row tail chunk and a 1-row
chunk. Those run at about 600 tok/s, because 128 rows leave most experts on the grouped f32 path. With reuse off, the
lane served 1K at 1,693 tok/s (1.58× the adjacent reuse-on trial), held-out as fast as dev, and the first request at
985 tok/s.

The same reuse-off trial was slower at 8K-31.7K (1,498 / 1,361 / 1,091 against 1,610 / 1,508 / 1,451). That is one
trial with no cause found, so the long-prompt side needs a second look before the host changes. Candidate changes
include fork points measured back from the prompt's end, so the tail falls on the chunk grid, or snapshots taken
from inside a full chunk; either is a host and prefix-format question for its owner.

Inside P4's files, the large buckets are still the routed NVFP4 matrices, the dense projections and attention at
long context. `INKLING_PREFILL_HALF=both` is the only measured route to 2,000 tok/s on this path. It is changed
arithmetic, so the decision is a qualification item.

## Reproduction

Environment for every command: `PATH="/opt/homebrew/bin:$PATH" TMPDIR="$PWD/.work/tmp"
ZIG_GLOBAL_CACHE_DIR="$PWD/.work/zig-global-cache" INKLING_LOAD_LIMIT_GB=26` and
`MODEL=~/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd`, run from the repository root.

Builds, one tree a side. No compile may run while a trial does:

```sh
git archive 3881b3d8c | tar -x -C .work/tmp/main-src        # the lane: git archive 665006b84
(cd .work/tmp/main-src && zig build native tf-inkling2-loop-run -Dcpu=apple_m1 --prefix "$PWD/../main-build")
# first-request control: the lane's tree with the line `try s.commit(device);` deleted from prefill.zig
```

The gates around each trial. A trial that trips one is repeated and kept as invalid:

```sh
GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60 gpu-gate-ready -1 && PERF_GATE_MAX_BUSY_CORES=4 perf-gate-ready
# the trial, under python3 campaign/heavy.py --, with no zig, clang or metal process alive while it runs
PERF_GATE_MAX_BUSY_CORES=4 perf-gate-ready -1
```

Served curve, one fresh server a trial, sides interleaved main, lane, lane, main. Set `INKLING_PREFILL_HALF=both`
or `INKLING_PREFIX_REUSE=0` in the environment for the variants:

```sh
cd campaign && uv run --project . python comparators/serve_curve.py tensorfold --model-dir "$MODEL" \
  --binary <build>/native/bin/tensorfold-native --out <trial>.jsonl [--lengths 1024]
python3 campaign/kernels/inkling-p4-prefill-runs/curvesum.py <trial>.jsonl...
```

Split timelines, runner wall time and the host's chunk schedule:

```sh
TF_INKLING_PREFILL_PROFILE_BASE=0 TF_METAL4_TIMELINE=<chunks a pass> \
  <build>/bin/tf-inkling2-loop-run "$MODEL" <ids.json> profile-prefill <chunk> > <log> 2>&1
python3 campaign/kernels/inkling_profile.py timeline --split --log <log> > <table>
python3 campaign/kernels/inkling-p4-prefill-runs/buckets.py <chunk rows> <table>... [--kernels]
<build>/bin/tf-inkling2-loop-run "$MODEL" <ids.json> bench 1 0 1024 <passes>
python3 campaign/kernels/inkling-p4-prefill-runs/chunks.py <prompt tokens>...
```

The token fixtures are leading tokens of the corpus files, from `tf-inkling2-loop-run <tokenizer.json> <text>
tokenize <out>`: `campaign/prompts/dev/code-32768.txt` cut to 1,024, 4,096, 16,384 and 31,744 tokens, and the
first 1,024 tokens of each dev and held-out file.

Correctness, and the variants against the exact path:

```sh
python3 campaign/heavy.py -- sh -c 'TF_INKLING2_LOOP_OQ4E_DIR="$MODEL" zig build test-inkling2-loop-filtered \
  -Dcpu=apple_m1 "-Dinkling-filter=oQ4e fused prefill"'
python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu,inkling-omlx-oracle
python3 -I campaign/suite/suite.py run --names inkling-quality-dev    # INKLING_PREFILL_HALF=both for the variant
make -C campaign quality-inkling2-mini-oq4e SET=dev BASELINE_WT="$PWD" BIN=<main build>/native/bin/tensorfold-native
cd campaign && uv run --project ref python -I -B parity/inkling2-mini/omlx_gate.py compare \
  --ref <exact capture dir> --cand <variant capture dir> --report <out>.md
```

The variant captures come from `make -C campaign -f parity/inkling2-mini/parity.mk metal-logits RUNNER_MODE=prefill
PREFILL_CHUNK=512 RUNNER=<runner> INKLING_DIR="$MODEL" METAL_LOGITS=<dir>`, with `INKLING_PREFILL_HALF` set.
