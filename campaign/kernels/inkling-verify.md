# Inkling 2 Mini copy verification

Status: runner parity, rollback, and served copy activation passed.

## Verification cost

On Apple M5 Max, the `verifybench` runner prefills the 1,139-token prose fixture, then measures
all-logits prefill calls. It does not include snapshot, rollback, proposal, or serving costs.
The checkpoint was `Inkling-2-Mini@ed40e7da625f`, with the runner built from base `23793959f`
and the measurement mode in this branch. The 28 GB `heavy.py` cap and GPU lock were active.
The host used the busy-host limits of 35% GPU time and utilization. Both performance and GPU
admission checks passed before and after each process.

| Rows | Median ms | Range ms | Samples |
| ---: | ---: | ---: | ---: |
| 2 | 123.360 | 122.683–132.644 | 7 |
| 4 | 130.805 | 129.957–139.488 | 7 |
| 8 | 152.552 | 151.440–152.766 | 7 |
| 16 | 153.428 | 152.541–154.005 | 7 |

The two-row run used a separate process. The remaining widths alternated order in one process.
Context length increased as the calls completed. These numbers do not establish a served speedup.
The native binary grew from 20,956,704 to 20,974,384 bytes on this build. The rollback test's
peak charged memory was 22.2 GB under the 28 GB cap.

## Rollback

The real-checkpoint GPU test starts at position 1,023, verifies sixteen rows, keeps two, and
processes five continuation rows. The continuation logits matched the accepted-prefix reference
bit for bit. The test crosses the 1,024-row local KV window and the four-row convolution rings.

## Greedy runner

The runner generated 512 tokens for each development prompt with EOS disabled. Copy and
one-row prefill output IDs matched for all three prompts. These are single gated pairs,
not a repeated served benchmark. The runner times generation after prompt prefill.

| Prompt | Decode tok/s | Plain prefill tok/s | Copy tok/s | Rounds | Accepted / drafted | Acceptance | Tokens per round |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code-1024 | 38.99 | 8.18 | 14.70 | 183 | 328 / 1,071 | 30.6% | 2.79 |
| prose-1024 | 39.13 | 8.19 | 17.93 | 187 | 324 / 596 | 54.4% | 2.73 |
| structured-1024 | 39.09 | 8.20 | 55.32 | 59 | 452 / 482 | 93.8% | 8.66 |

Rates and tokens per round count the 511 tokens after the initial prompt prediction. The first
copy attempt failed at the 16-row boundary and was discarded. Checked row-count arithmetic
passed the shorter reproduction and the three full comparisons. The memory watchdog reported
a peak charge of 21.0 GB for the full comparison process.

The decode-kernel runner generated the same 512 IDs on each prompt, starting from the same
chunked prefill state. Its single gated generation runs took 13.106 seconds for code,
13.059 seconds for prose and 13.071 seconds for structured. There were no token differences
to classify as near-ties on these runs. The copy path is opt-in because its rate varies by
prompt and the code and prose runner rates were below decode in these single samples.

With `INKLING_COPY_DRAFTS=1`, a served 128-token structured request reported 75 drafts,
20 accepted, and 107 rounds. The same request with `--no-drafts` reported zero drafts
and 127 rounds. Both completed. Server shutdown reported allocator leaks in both the
drafts-off check and the base server's existing benchmark logs.

## Few-row prefill

The few-row dense kernel shares each bf16 weight load across up to eight f32 activation rows.
For one or two rows, the per-slot expert kernels avoid sorting and grouped dispatches.
Larger chunks retain the existing dense and grouped expert paths.

The family test ran with the checkpoint under `heavy.py` and exited 0. It compared the dense
projection shapes against sampled host FP64 values for 1, 2, 4, 8 and 16 rows, and tested
both direct expert formats and projections against FP64 for 1 and 2 rows. The checkpoint
rollback test reported a maximum logit difference of zero. Peak charged memory was 21.3 GB.
The prefill parity commands for chunk sizes 4 and 512 each exited 2 with 14 histories within
contract, 2 explained divergences and 0 unexplained. Reference divergence diagnostics exceeded the 28 GB watchdog cap at 28.6 GB for chunk 4
and 28.5 GB for chunk 512. Candidate inference did not exceed the cap.

The split Metal 4 timeline for two rows reported 38.4 ms summed GPU span: 13.8 ms in the
few-row dense kernel, 5.0 ms in expert gate and up, 3.5 ms in expert down, and 4.9 ms in
route selection. At eight rows the dense-only timeline reported 91.4 ms, including 37.5 ms
in dense and 33.8 ms in grouped experts. At sixteen rows it reported 140.4 ms, including
54.2 ms in dense GEMMs and 58.1 ms in grouped experts. Timeline instrumentation is separate
from the timing measurements below.

### Busy-host measurements

On Apple M5 Max and macOS 27.2, the `verifybench` runs used the 1,139-token prose fixture
and the same checkpoint. Five baseline and candidate processes alternated. Each process
interleaved five calls per width. The table gives the median and range of the five process
medians in milliseconds. The supplied earlier baseline medians were 123.4, 130.8, 152.6,
and 153.4 ms for 2, 4, 8, and 16 rows respectively.

| Rows | Baseline median (range) ms | Candidate median (range) ms |
| ---: | ---: | ---: |
| 2 | 122.906 (122.501–123.725) | 46.357 (46.163–46.416) |
| 4 | 129.514 (129.192–129.751) | 70.345 (70.269–70.584) |
| 8 | 151.947 (151.138–152.282) | 105.549 (105.360–105.681) |
| 16 | 152.617 (152.421–152.939) | 152.972 (152.921–153.297) |

The greedy runner generated 512 tokens with EOS disabled. Five baseline and candidate
processes alternated for each prompt and mode. Rates count the 511 tokens after the initial
prompt prediction. Values are median tokens per second, with the five-sample range in
parentheses. Candidate copy and prefill output IDs matched in every sample.

| Prompt | Mode | Baseline tok/s (range) | Candidate tok/s (range) |
| --- | --- | ---: | ---: |
| code-1024 | decode | 39.01 (35.29–39.20) | 39.02 (35.41–39.18) |
| code-1024 | prefill | 8.18 (8.17–8.18) | 24.08 (24.06–24.11) |
| code-1024 | copy | 14.71 (13.77–14.72) | 23.38 (20.97–23.75) |
| prose-1024 | decode | 39.30 (39.22–39.33) | 39.29 (39.18–39.37) |
| prose-1024 | prefill | 8.20 (8.19–8.22) | 24.39 (24.35–24.42) |
| prose-1024 | copy | 17.94 (17.94–17.96) | 33.95 (33.93–33.96) |
| structured-1024 | decode | 39.37 (39.28–40.16) | 39.33 (39.17–39.37) |
| structured-1024 | prefill | 8.20 (8.20–8.21) | 24.37 (24.27–24.38) |
| structured-1024 | copy | 55.38 (55.32–55.42) | 71.76 (71.62–73.16) |

Copy used 183 rounds with 328 of 1,071 drafts accepted for code, 187 rounds with 324 of
596 accepted for prose, and 59 rounds with 452 of 482 accepted for structured. The runner
excludes prompt prefill from generation timing. These are busy-host runner measurements, not
served throughput. Copy still trails decode for code and prose.

Each model run used `python3 campaign/heavy.py --` with the 28 GB cap. Before and immediately
after each measurement, `perf-gate-ready` and `gpu-gate-ready -1` checked admission with
`GPU_GATE_MAX_GPU_PCT=35` and `GPU_GATE_MAX_UTIL_PCT=35`. Runs that failed a post-check were
discarded. The inputs were the development code, prose, and structured prompts, tokenized
to 1,334, 1,139, and 1,109 IDs. The baseline and candidate runner binaries were 1,586,760 and
1,587,128 bytes. The measured verifybench command used `verifybench 2 5`; generation used
`greedy-copy`, `greedy-prefill`, or `greedy-decode` with 512 steps.

## Hybrid rounds

Rounds without a copy draft now run the decode kernels (one `position`, ~25 ms) instead of a one-row prefill pass
(~41 ms), and a round verifies at most `hybrid_drafts` = 7 drafts (8 rows, ~106 ms after the few-row kernels). The
runner's `greedy-hybrid` mode does the same; `TF_INKLING_HYBRID_DRAFTS` overrides the cap for measurement only.
Single gated runs on main b5c778934 (512 greedy tokens; decode for comparison 12.8-13.1 s); ids equal to
`greedy-decode` on all three prompts at cap 7:

| Cap | code-1024 s | prose-1024 s | structured-1024 s |
| ---: | ---: | ---: | ---: |
| 1 | 15.76 | 13.26 | 12.47 |
| 2 | 15.88 | 13.52 | 11.41 |
| 3 | 16.30 | 12.22 | 9.77 |
| 4 | 15.97 | 13.14 | 9.26 |
| 7 | 18.02 | 12.02 | 8.11 |

Even at cap 1, where a 2-row verify (46 ms) is cheaper than two decode steps and 76% of drafts are accepted, code
is slower than decode: each verification round also allocates a snapshot buffer, blits the convolution rings and
the touched local KV rows in a separate command buffer, reads all rows' logits back for a CPU argmax, and on a
partial accept re-runs the accepted prefix. Those per-round costs (~12 ms) are the next target; copy drafts stay
opt-in until code is no slower than decode.

## Round instrumentation

`roundbench` measures snapshot allocation and save, target forward, CPU logits read and argmax,
restore, and accepted-prefix replay. It restores the initial cache before each width and alternates
width order. The benchmark accepts half the rows, rounded down. A timeline run accepts zero rows
to keep replay kernels out of the target-forward profile. Hybrid generation also reports round
costs. Hybrid throughput comparisons are not yet measured.

One busy-host baseline process ran five rounds per width in alternating order from the same
1,139-token prose cache state. The table gives median milliseconds for each component and the
range of total round time. The round has no drafter; accepted rows are half the width rounded
down. CPU argmax includes access to the shared logits buffer. Host time is the median of the
total less the measured components; independently rounded component medians need not sum to it.
The forward includes the vocabulary head. Kernel attribution remains to be measured separately.

| Rows | Accepted | Save ms | Forward ms | Read and argmax ms | Restore ms | Replay ms | Host ms | Total ms (range) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0.371 | 39.344 | 0.319 | 0.271 | 0 | 0.007 | 40.318 (40.263–40.601) |
| 2 | 1 | 0.400 | 44.955 | 0.657 | 0.305 | 39.353 | 0.021 | 85.670 (85.588–86.071) |
| 4 | 2 | 0.467 | 68.402 | 1.311 | 0.357 | 44.936 | 0.022 | 115.566 (115.465–116.145) |
| 8 | 4 | 0.600 | 102.735 | 2.708 | 0.913 | 68.461 | 0.022 | 175.543 (175.343–176.343) |
| 16 | 8 | 0.939 | 150.172 | 5.492 | 1.202 | 102.757 | 0.025 | 260.610 (260.060–262.502) |

The runner used `roundbench 5` under `heavy.py`, with `perf-gate-ready` and
`gpu-gate-ready -1` before the run and their post-checks immediately after it. The busy-host
GPU limits were 35% for time and utilization. This is one process, not an interleaved
baseline/candidate comparison. Peak charged memory was 1.4 GB as reported by `heavy.py`;
the resident checkpoint pages were already warm and this charge is not the model's footprint.

Split Metal 4 profiles used the same prose cache state, five timestamped rounds after eight
warm rounds, and accepted zero rows so replay did not enter the profile. Profiling splits
command buffers, so these GPU spans are not the unsplit wall times above. Dense includes the
vocabulary head, which is also shown separately.

| Rows | Summed GPU span ms | Dense ms | Expert gate/up and down ms | Attention ms | Route selection ms | Vocabulary head ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 41.699 | 11.110 | 4.587 | 11.174 | 4.920 | 1.580 |
| 2 | 47.154 | 13.096 | 7.984 | 11.309 | 4.900 | 1.677 |
| 4 | 71.075 | 20.841 | 21.355 | 12.630 | 4.893 | 3.042 |
| 8 | 108.880 | 37.877 | 33.533 | 18.640 | 5.001 | 5.846 |
| 16 | 154.792 | 53.584 | 60.232 | 18.091 | 4.894 | 2.181 |

The profiler logs passed the same admission checks as the baseline. An eight-row attempt
and a sixteen-row attempt failed post-checks and were discarded.

### Bounded dense row loops

The small-row dense kernel now bounds each row loop at sixteen, with an early exit at the
actual row count. This exposes the bound to the Metal compiler while retaining the source's f32
accumulation order. Keep: the eight-row total improved outside the observed process-median
spread. Five baseline and candidate processes alternated order. Each interleaved five rounds
per width from the same restored prose cache state. The table reports the median of the five
process medians, and the range of process medians for total time.

| Rows | Save ms | Forward ms | Read and argmax ms | Restore ms | Replay ms | Host ms | Candidate total ms (range) | Baseline total ms (range) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.370 | 39.638 | 0.332 | 0.249 | 0 | 0.006 | 40.599 (40.415–40.708) | 40.442 (40.286–40.618) |
| 2 | 0.393 | 44.015 | 0.657 | 0.275 | 39.606 | 0.020 | 85.057 (84.759–85.134) | 85.791 (85.735–86.140) |
| 4 | 0.496 | 63.045 | 1.323 | 0.364 | 44.070 | 0.022 | 109.310 (109.236–109.344) | 115.981 (115.879–116.623) |
| 8 | 0.638 | 91.051 | 2.655 | 1.138 | 63.041 | 0.023 | 158.502 (158.249–158.812) | 176.233 (176.011–176.355) |
| 16 | 0.891 | 150.932 | 5.317 | 1.424 | 90.998 | 0.022 | 249.700 (248.972–250.042) | 261.741 (261.142–262.042) |

Each run passed CPU and busy-host GPU checks before and after execution under `heavy.py`.
Peak charged memory across the pairs was 21.0 GB. Both runner binaries were 1,604,552 bytes.
The family test exited 0 with byte-identical rollback continuation. End-to-end hybrid
throughput for the candidate remains pending.

Candidate split profiles use the same method as the baseline profiles above. The five
timestamped rounds per width ran in one locked process, with admission checks around each
model run. Peak charged memory was 24.0 GB. Dense includes the separately listed head.

| Rows | Summed GPU span ms | Dense ms | Expert gate/up and down ms | Attention ms | Route selection ms | Vocabulary head ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 42.037 | 11.392 | 4.670 | 11.218 | 4.913 | 1.594 |
| 2 | 46.563 | 12.377 | 8.058 | 11.392 | 4.904 | 1.601 |
| 4 | 65.857 | 15.616 | 21.383 | 12.664 | 4.906 | 1.721 |
| 8 | 95.433 | 25.509 | 33.026 | 18.433 | 4.900 | 3.089 |
| 16 | 154.982 | 53.668 | 60.186 | 18.234 | 4.894 | 2.178 |

### Per-slot experts through eight rows

The per-slot expert path now handles up to eight rows. Keep: five alternating process pairs
showed lower four- and eight-row round time than the bounded-dense version. The table uses
the same component and range statistics as the preceding comparison.

| Rows | Save ms | Forward ms | Read and argmax ms | Restore ms | Replay ms | Host ms | Candidate total ms (range) | Baseline total ms (range) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.345 | 39.529 | 0.331 | 0.245 | 0 | 0.006 | 40.497 (40.434–40.587) | 40.563 (40.373–40.585) |
| 2 | 0.387 | 43.943 | 0.648 | 0.278 | 39.542 | 0.020 | 84.815 (84.740–84.888) | 84.886 (84.858–84.897) |
| 4 | 0.451 | 55.667 | 1.327 | 0.340 | 43.964 | 0.019 | 101.795 (101.441–102.149) | 109.067 (108.829–109.207) |
| 8 | 0.598 | 86.076 | 2.656 | 1.145 | 55.714 | 0.021 | 146.286 (146.041–146.822) | 158.307 (157.848–158.494) |
| 16 | 0.856 | 150.450 | 5.313 | 1.343 | 86.281 | 0.020 | 244.604 (243.707–245.399) | 249.221 (249.015–250.064) |

The family test passed under `heavy.py`, with 21.6 GB peak charged memory. The per-slot
expert test checked both checkpoint formats and projections at one, two, four, and eight rows
against FP64 bounds. Rollback continuation remained byte-identical. Both runner binaries
were 1,604,552 bytes. The end-to-end comparison remains pending.

The split profile used five timestamped rounds after eight warm rounds per width. All
five model runs passed the admission checks under one `heavy.py` lock, with a 20.8 GB peak
charge. Dense includes the separately listed head.

| Rows | Summed GPU span ms | Dense ms | Expert gate/up and down ms | Attention ms | Route selection ms | Vocabulary head ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 41.985 | 11.386 | 4.681 | 11.190 | 4.896 | 1.596 |
| 2 | 46.541 | 12.408 | 8.059 | 11.362 | 4.909 | 1.601 |
| 4 | 58.404 | 15.579 | 15.206 | 12.693 | 4.913 | 1.721 |
| 8 | 89.106 | 25.498 | 29.929 | 18.335 | 4.892 | 3.089 |
| 16 | 154.503 | 53.365 | 60.172 | 18.159 | 4.889 | 2.185 |

After both retained changes, the prefill parity commands at chunks 4 and 512 each exited 2
with 14 histories within contract, two explained divergences, and zero unexplained. The
candidate inference runs stayed below the 28 GB `heavy.py` cap.

### Hybrid against ordinary decode

Five pairs per workload alternated `greedy-hybrid` and `greedy-decode`, using chunked
prefill and the same prompt IDs. Each emitted 512 tokens with EOS disabled. Timing starts
after the prompt prediction, so throughput counts the 511 subsequently generated tokens.
Each process passed CPU and busy-host GPU admission checks before and after execution under
`heavy.py`. Hybrid uses the mixed target: verified rows use prefill kernels and undrafted
rows use decode kernels. Output IDs matched the decode control in every pair. This does not
establish universal greedy equivalence.

| Workload | Decode tok/s median (range) | Hybrid tok/s median (range) | Hybrid seconds median |
| --- | ---: | ---: | ---: |
| code-1024 | 39.03 (35.99–39.99) | 33.83 (30.39–34.07) | 15.106 |
| prose-1024 | 40.08 (40.06–40.20) | 47.72 (47.64–47.75) | 10.708 |
| structured-1024 | 40.19 (40.17–40.22) | 76.58 (76.45–76.72) | 6.673 |

Hybrid effective throughput is committed tokens divided by draft, snapshot, verification,
restore, replay, ordinary decode, and remaining host time. The code median includes 8.742 s
in verification forward, 3.392 s in replay, 0.274 s in logits read and argmax, 0.079 s in
save, and 0.076 s in restore. The original receipt did not separate undrafted decode from
remaining host time. Code remains below the decode control.

| Workload | Draft ms | Save s | Verify forward s | Read and argmax s | Restore s | Replay s | Undrafted decode and host s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code-1024 | 0.270 | 0.079 | 8.742 | 0.274 | 0.076 | 3.392 | 2.552 |
| prose-1024 | 0.257 | 0.054 | 5.388 | 0.171 | 0.031 | 1.486 | 3.578 |
| structured-1024 | 0.171 | 0.074 | 6.090 | 0.183 | 0.005 | 0.293 | 0.028 |

Each component is the median across processes. The last column is the median per-process
residual after subtracting the measured components from generation time; rounded medians
do not necessarily sum to the median total. The runner now separately measures undrafted
decode for subsequent comparisons.

The family test passed with `TF_INKLING2_LOOP_DIR` set to the checkpoint and
`python3 campaign/heavy.py -- zig build test-inkling2-loop -Dcpu=apple_m1`. It compared
repeated predictions after snapshot reset and continuation logits byte for byte across the local
window and convolution ring boundaries. Peak charged memory was 22.2 GB.

Prefill parity at chunk sizes 4 and 512 each returned exit 2 with 14 histories within contract,
two explained divergences, and zero unexplained. The 512 comparison's runner binary was rebuilt
during the trace diagnostics. The rebuild changed runner instrumentation, not prefill arithmetic,
but the receipt does not pin a single binary for the entire comparison. One chunk-4 candidate
run observed memory pressure level 2 during execution. Neither result establishes throughput.
