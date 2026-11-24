# Inkling oQ4e cold prefill

Lane P3c, base `4c2af5d2a`, M5 Max 48 GiB, Inkling 2 Mini oQ4e `3eea042751cd`, greedy. The routed experts, tiled attention and staged projections of chunks over eight rows now run on the M5 tensor units, prefill chunks may exceed the local window, and the served chunk is 1,024 rows. The change is integrated through `997eb064a`. With weights already resident, prefill runs 2.9× the base at 1K, 3.3× at 4K and 3.1× at 8K. A process's first prefill gains 2.0× at 4K; at 1K, two of three pairs show no gain because one-time cost dominates. Decode and batch timing are unchanged. Served 2- and 4-agent aggregate throughput rose 5–41%. Quality dev is 7/11 against the base's 6/11, with one new deterministic reasoning loop on `code-c02`; Pi is 7/8 on both.

## P3c: tensor-unit prefill

Base revision `4c2af5d2a`. Every measurement below used the oQ4e checkpoint `3eea042751cd`, greedy decoding, the M5 Max (48 GiB, GPU generation 17) and `campaign/heavy.py` with `INKLING_LOAD_LIMIT_GB=26`. The owner dropped the fidelity requirement for this lane, so no saved-capture comparison was run.

### Where the expert time went

The routed NVFP4 matrices (`i2p_expert_matrix_wide`) ran 8×8 simdgroup MMAs on the shader cores. A standalone Swift + Metal microbenchmark (`.work/tmp/mb/bench.swift`, compiled from the family's own `prefill.metal` and `prefill_expert_wide.metal`) timed the kernel on synthetic routing with 128 experts, top-8, skewed occupancy, NVFP4 weights and the served shapes (gate/up K 2,304 to 2,304 interleaved columns; down K 1,152 to 2,304). Rates count only experts above the matrix thresholds.

| Tokens | Projection | Shipped wide kernel | Tensor-unit 64×64, f32 rows | Tensor-unit 64×64, half rows |
| ---: | --- | ---: | ---: | ---: |
| 512 | gate/up | 7.1 TFLOP/s | 12.3 | 23.9 |
| 512 | down | 7.0 | 11.9 | 22.4 |
| 1,024 | gate/up | 7.9 | 17.6 | 34.9 |
| 1,024 | down | 7.8 | 15.6 | 32.5 |
| 2,048 | gate/up | 8.7 | 19.1 | 39.0 |
| 2,048 | down | 8.2 | 14.7 | 35.2 |

The shipped kernel stays at 7–9 TFLOP/s while its rows per expert quadruple, so the brief's diagnosis holds only in part. Weight decode is indeed not amortized (each 32-row tile decodes its weights again), but amortizing it alone does not help: the base profile at chunk 1,024 spent 776 ms on the kernel for 1,024 rows, 388 ms per 512 against 375 at chunk 512. What bounds it is the shader-core 8×8 MMA rate. On the tensor units (`matmul2d` with `execution_simdgroup`, 16×32×16, cooperative fragments in the M5 operand layout) the same 64×64 tile with each weight tile decoded once roughly doubles the rate when it converts f32 rows itself and quadruples it when the rows are rounded to half once beforehand, the rounding the half path already applies. Larger per-simdgroup tiles (32×64, 64×32, 64×64), 128-row tiles, 8-simdgroup threadgroups and a 32-row tile were each slower at every occupancy measured (14–35 TFLOP/s at 2,048 tokens, against 39.0 for gate/up and 35.6 for down with the chosen tile).

### Where the dense time went

`i2as_gemm` multiplies f32 activations by the staged half weights through MPP `matmul2d`. A second microbenchmark (`.work/tmp/mb2/gemm.swift`, served shapes q/gate 3,072×2,304, k/v 1,024×2,304, o 2,304×3,072) showed that the f32 operand, not the tile, sets its rate:

| Rows | Shape (N×K) | f32×half, shipped 16×32 tile | f32×half, best other tile | hi/lo half pair, 64×64 | half×half, 64×64 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 512 | 3,072×2,304 | 14.6 TFLOP/s | 14.8 | 20.0 | 41.8 |
| 512 | 2,304×3,072 | 13.1 | 13.8 | 26.1 | 36.9 |
| 2,048 | 3,072×2,304 | 15.1 | 15.6 | 28.7 | 61.7 |
| 2,048 | 1,024×2,304 | 15.0 | 15.3 | 27.0 | 59.7 |
| 2,048 | 2,304×3,072 | 14.8 | 15.7 | 26.8 | 58.8 |

Every f32×half tile shape produced the same bytes and stayed at or below about 15.7 TFLOP/s. (The first kernel timed in each process ran slow; the table uses only kernels timed after another one.) Splitting each f32 activation into `hi = half(x)` and `lo = half(x − hi)` and running two accumulating half passes doubles the rate; its outputs differ from the f32-input product by at most 5.8e-5 at magnitude 16 (about 2^-18 relative), against 3.3e-3 for rounding the activations to half once. The split keeps about 22 bits of each input (less for |x| below 2^-3, where `lo` becomes subnormal); the cost is that inputs above 65,504 now count as a half overflow and refill the prompt in f32, as the other half paths do. That run's CPU post-check failed (another process at 239% CPU); the kernels are timed by GPU timestamps.

### What changed

Six commits on `lane/p3c-prefill`, each on top of the last:

| Commit | Change |
| --- | --- |
| `ec9b5b990` | `bench <steps> [warmup] [chunk] [prefills]`: the prompt prefilled `prefills` times from an empty cache, each pass reported. |
| `36b5c7c44` | Routed NVFP4 and affine-4 expert matrices on the tensor units (64×64 tiles of 16×32×16 half MMAs, f32 sums), rows rounded to half once, half activations from gate/up to down; grouped f32 kernels unchanged below the matrix thresholds. |
| `2662b573f` | Order-safe local-ring KV store, chunks longer than the window, logits sized to the rows a caller needs, `default_chunk` 512 → 1,024. |
| `37998dc7b` | Relative bias a row a threadgroup, bit-identical to the per-output kernel. |
| `582a54164` | Tiled half attention on the tensor units, one KV head's three query heads a threadgroup. |
| `997eb064a` | Staged affine projections as hi/lo half pairs on the tensor units instead of f32 × half. |

All of it is behind `PrefillScratch.half` and `PrefillScratch.units` and the existing M5 checks; chunks of eight rows or fewer (decode, batch, verify) and devices without tensor units take the kernels they took before.

### Split timeline

GPU time per chunk from `profile-prefill` with `TF_INKLING_PREFILL_PROFILE_BASE=0` (each slot its own segment; the 25 largest kernels a segment are kept), after eight warm chunks. Per-512 figures divide the chunk's time by its rows / 512. The candidate columns are `.work/tmp/cand-b`, the code of `997eb064a`; the base 1K column is `4c2af5d2a`; the base 4K column is P3's profile of `b9bdf4e55`, because the 4K profile of `4c2af5d2a` was cancelled with the wind-down. Logs: `.work/tmp/prof/{base-1k-512,candb-1024-1024,candb-4096-2048}.log`.

| Kernel | Base, 1K, chunk 512 | Candidate, 1K, chunk 1,024 (per 512) | Base, 4K, chunk 512 (P3) | Candidate, 4K, chunk 2,048 (per 512) |
| --- | ---: | ---: | ---: | ---: |
| Routed NVFP4 matrices | 375.3 | 98.4 | 380.7 | 100.5 |
| Dense projections (staging, split, GEMM) | 103.5 | 70.7 | 110.3 | 75.1 |
| Grouped low-occupancy experts | 84.3 | 33.8 | 83.5 | 14.2 |
| Attention | 53.5 | 17.6 | 93.9 | 27.7 |
| Affine-4 experts | 59.9 | 11.4 | 61.4 | 10.4 |
| Relative bias | 21.1 | 5.4 | 21.8 | 6.0 |
| Everything else | 59.2 | 57.0 | 65.4 | 62.1 |
| **Summed span** | **756.9** | **294.3** | **817.2** | **296.0** |

### First pass and later passes

`bench` times the prompt once right after the model loads, so its first pass also pays one-time costs: the first command buffers that use the 17 GB weight buffer the loader filled, the first touch of the prefill scratch, and each pipeline's first dispatch. A second pass from an empty cache in the same process (`bench 32 4 CHUNK 2`, the option added in `ec9b5b990`) measures the same cold-cache prefill without them. In the screen of all six commits (`.work/tmp/screen3.out`, one run each):

| Run | First pass ms | Third pass ms | First-pass excess |
| --- | ---: | ---: | ---: |
| Base, 1K, chunk 512 (rejected attempt, post-check busy) | 1,819.9 | 1,502.8 | 317 |
| Base, 1K, chunk 512 | 3,670.1 | 1,501.8 | 2,168 |
| Candidate, 1K, chunk 1,024 | 2,099.0 | 519.6 | 1,579 |
| Candidate, 1K, chunk 512 | 1,037.6 | 680.8 | 357 |
| Base, 4K, chunk 512 | 6,935.1 | 7,662.9 | −728 |
| Candidate, 4K, chunk 2,048 | 3,875.8 | 2,374.2 | 1,502 |
| Candidate, 4K, chunk 1,024 | 2,555.4 | 2,401.1 | 154 |
| Candidate, 4K, chunk 4,096 | 2,334.4 | 2,084.0 | 250 |

The same base binary's first-pass excess ranged from 0.3 to 2.2 s between runs, while its later passes agreed to a millisecond, so the excess comes from the host's state (other lanes' memory and GPU use) more than from the code. Long prompts reverse it: three back-to-back 7-second base passes ran slower each time (heat), so 4K first passes are the cooler measurement. The 1K target sits inside this noise: at 1,050 tok/s the whole 1,024-token prefill must take 975 ms, less than the largest excess observed.

### Timing

The campaign wound down before the three-pair interleaved matrix ran, so the timing evidence is the gated single screens below: each run went through `heavy.py`, the GPU gate, the 4-core CPU gate, the compiler monitor and the CPU post-check, and exits 75 and 76 were retried, not counted. Rates are tok/s for the whole prompt; "first" is the `bench 32 4 CHUNK` line, "third" the third pass from an empty cache in the same process (`bench 32 4 CHUNK 3`).

Screen of the expert and chunk code that became `36b5c7c44` and `2662b573f`, before the default chunk changed (`.work/tmp/screen1.out`, base `4c2af5d2a`, first pass only):

| Prompt | Base, chunk 512 | Candidate, chunk 512 | Candidate, chunk 1,024 | Candidate, chunk 4,096 |
| --- | ---: | ---: | ---: | ---: |
| 1K | 453.7 | 546.3 | 419.1 | – |
| 4K | 612.1 | 980.9 | 1,175.0 (invalid: compiler overlap) | 796.8 |

Screen of all six commits (`.work/tmp/screen3.out`, candidate `.work/tmp/cand-b`, the code of `997eb064a` with the chunk given on the command line; base `.work/tmp/base-r`, `4c2af5d2a` plus the bench option):

| Prompt | Run | First | Third | Decode ms a step after it |
| --- | --- | ---: | ---: | ---: |
| 1K | base, chunk 512 | 279.0 | 681.9 | 20.029 |
| 1K | candidate, chunk 1,024 | 487.9 | 1,970.7 | 19.960 |
| 1K | candidate, chunk 512 | 986.9 | 1,504.2 | 20.157 |
| 4K | base, chunk 512 | 590.6 | 534.5 | 21.665 |
| 4K | candidate, chunk 1,024 | 1,602.9 | 1,705.9 | 21.728 |
| 4K | candidate, chunk 2,048 | 1,056.8 | 1,725.2 | 22.836 |
| 4K | candidate, chunk 4,096 | 1,754.7 | 1,965.5 | 22.204 |

At 4K the candidate clears 1,050 tok/s in every pass at every chunk tried (2.7× the base's first pass at the served chunk of 1,024). At 1K its later passes run 2.9× the base's (1,971 against 682), but its first passes (488 and 987) are inside the one-time cost described above, and this table holds one run a cell. The decode medians after prefill decode different tokens on each side (the candidate's prefill rounds differently, so its argmax sequence and expert routing differ) and are not a like-for-like decode comparison; decode and batch kernels are unchanged by these commits (rows ≤ 8 take the same code), but `batchbench` pairs were not run.

### Interleaved matrix

Every run used the gated wrapper from [the f16 KV timing procedure](inkling-f16-kv.md#timing-procedure) with `PERF_GATE_MAX_BUSY_CORES=4`; gate refusals were retried and never counted. The base is `4c2af5d2a` plus the `bench` option from `ec9b5b990`, at chunk 512 (`bench 32 4 512 2`). The candidate is `997eb064a` at chunk 1,024 (`bench 32 4 1024 2`). Pair order was base/candidate, candidate/base, base/candidate. "First" is the process's first prefill and "repeat" is the second, from an empty cache with the weights resident. An identical-build control at 4K differed by 2.7% on first passes and 2.4% on repeats.

| Prefill, tok/s | Base | Candidate | Gain per pair |
| --- | --- | --- | --- |
| 1K first | 256.9, 446.3, 534.0 | 432.2, 429.3, 524.9 | 1.68, 0.96, 0.98 |
| 1K repeat | 692.4, 692.8, 694.2 | 1,994.9, 1,995.5, 1,995.1 | 2.88, 2.88, 2.87 |
| 4K first | 594.6, 587.2, 563.7 | 1,521.2, 1,034.0, 992.3 | 2.56, 1.76, 1.76 |
| 4K repeat | 620.4, 566.2, 542.4 | 1,911.0, 1,829.4, 1,913.2 | 3.08, 3.23, 3.53 |
| 8K first (one pair) | 523.0 | 1,328.2 | 2.54 |
| 8K repeat (one pair) | 535.8 | 1,651.3 | 3.08 |

Batch decode on the 2,500-ID fixture is unchanged: `batchbench 1 5` medians were 19.916, 19.659, 19.670 ms on the base and 19.803, 19.759, 19.697 on the candidate, and `batchbench 4 5` were 39.823, 39.938, 40.042 and 39.848, 40.310, 39.614. Peak charged memory was at most 20.4 GB on the base and 19.3 GB on the candidate.

The scratch-commit work in progress (`f0ff849e6`) is not integrated. Against `997eb064a` at 1K over two pairs, its first pass ran 1,113.8 and 1,187.6 tok/s against 491.2 and 1,075.3, and its repeat pass was equal. Two pairs with that spread do not establish a gain, and the commit has not run the test suite.

### Served sweeps

The sweeps ran `campaign/fx/sweep-subagents.sh` at 4,096 context on oQ4e, in the order base 2, candidate 2, candidate 4, base 4, one gated run a cell with no identical-build control. The base is main `8e07fdb6f`; the candidate is that tree with this lane merged. Both use the corrected reasoning-replay clients from `69651393c`, which the lane's own base predates, so the sweep must run from a merged tree.

| Agents | Repetition | Base tok/s | Candidate tok/s | Gain | Base peak GB | Candidate peak GB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 36.18 | 50.52 | 1.40 | 20.91 | 20.59 |
| 2 | 2 | 64.62 | 67.86 | 1.05 | 20.91 | 20.59 |
| 4 | 1 | 40.65 | 57.48 | 1.41 | 21.92 | 21.60 |
| 4 | 2 | 54.83 | 58.74 | 1.07 | 21.92 | 21.60 |

Concurrent replies equalled their solo and serial replies at 1, 2 and 4 streams on both builds. The two builds generate different tokens (552 and 540 at two agents, 1,113 and 1,087 at four), so these rates are not identical work.

Mean time to first token at four agents fell from 2,577 to 1,510 ms on repetition 1 and rose from 1,099 to 1,209 ms on repetition 2, where turn 2 reaches about 1,255 ms for all four agents against 516–902 ms on the base. Server-side service time still fell for every prefill size (for 90–119 uncached tokens, the median went from 436 to 310 ms), so the rise is in how requests overlap. Its cause is not traced.

### Fidelity and quality

These compare the merged tree with the pinned oMLX oracle over 3,015 positions. Fidelity is reported and does not gate.

| Capture | Top-1 | Mean KL | Max KL |
| --- | ---: | ---: | ---: |
| Base, chunk 512 | 2,861 (94.89%) | 0.01951 | 8.01 |
| Candidate, chunk 512 | 2,863 (94.96%) | 0.01950 | 14.25 |
| Candidate, served chunk 1,024 | 2,858 (94.79%) | 0.01917 | 9.70 |

On the oQ4e quality dev set the base scores 6/11 and the candidate 7/11. The candidate gains `retrieval-t01` and `retrieval-t02`, which the base runs to the 8,192-token limit, and loses `code-c02`. The three structured cases are refused on both builds and `retrieval-t05` exceeds the context cap. Neither build reaches the 8-case bar set for the fast profile, and `quality-qualify` refuses both runs because of the refused and excluded cases. Pi scores 7/8 on both builds with no parse errors; `read-and-answer` fails on both.

The lost case is a repeatable model loop, not a crash. On `code-c02` the candidate's reasoning first differs from the base's 127 characters in, and two fresh servers then produced byte-identical output that ends in a 37-character cycle covering 10,323 of its 14,986 characters until the token limit. Decode kernels are unchanged, so the difference enters through prefill rounding.

```sh
make -C campaign quality-inkling2-mini-oq4e SET=dev BASELINE_WT="$TREE" BIN="$TREE/zig-out/native/bin/tensorfold-native"
make -C campaign pi-harness-inkling2-mini-oq4e BIN="$TREE/zig-out/native/bin/tensorfold-native"
```

### Verification on `997eb064a`

```sh
export PATH="/opt/homebrew/bin:$PATH" TMPDIR="$PWD/.work/tmp" ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache" INKLING_LOAD_LIMIT_GB=26
python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu   # exit 0: inkling-host 25/25 tests, inkling-gpu 97/97 tests, no skips
zig build native -Dcpu=apple_m1 -j1                                        # exit 0, 3 min 11 s
python3 tools/zig/lean_check.py > head.txt; python3 tools/zig/lean_new.py base.txt head.txt   # 247 problems, 247 on the base, 0 new
git diff --check 4c2af5d2a..997eb064a                                      # clean
```

The suite's GPU workload peaked at 23.2 GB charged under the 28 GB cap. Each intermediate commit's runner also built (`zig build tf-inkling2-loop-run -Dcpu=apple_m1 -j1` from `git archive` of the commit). Every new test runs and none skips on this M5; on devices without tensor units the new kernel tests skip and the shipped kernels run.

The work in progress on top, committing the prefill scratch with one blit fill at setup so the first chunk does not take its page faults, is unverified: it builds and was never timed or tested. It is the `wip:` commit after this receipt.

### Correctness

New tests, in `units_test.zig`, `units_attention_test.zig`, `prefill_test.zig` and `affine_test.zig`:

- Routed NVFP4 experts on the tensor units against fp64 with half-rounded rows and exact half weights, at experts of 0, 7, 13, 14, 64, 65 and 130 rows (both matrix thresholds and every tile edge), column tails, sorted-row permutations and `topk = 3`; worst gate/up error 1.56e-2 on activations up to 49 and worst down error 1.34e-5, inside `2^-11 |v| + 16 γ Σ|terms|` and `γ Σ|terms|`. The same test checks that the grouped gate/up kernel writes exactly `half(act)` for experts of 8–13 rows, leaves experts under 8 rows unwritten, and that `i2p_half_rows` rounds exactly and counts two planted values past 65,504.
- Affine-4 experts the same way (weights `half(fma(q, scale, bias))`), worst errors 9.6e-4 and 2.7e-6.
- Tensor-unit attention and `i2p_attn_half` against fp64 over a 48-row ringed local window whose keys straddle the chunk start and a global layer with log scaling, 45- and 37-row chunks; worst errors 1.57e-4 and 1.74e-4 for the new kernel against 1.59e-4 and 1.74e-4 for the shipped one, bound `2 (2^-11 + δs) max|v| + 1e-5` with δs each row's f32 score error.
- The hi/lo staged projection against fp64 of the f32-input product, with strided rows and the logit cap: worst error 0.011 of a bound made of each input's exact representation error plus f32 accumulation over 2k products; a planted 70,000 is counted.
- The row-staged relative bias equals the per-output kernel byte for byte at the served and tiny shapes.
- Chunks longer than the window: on the tiny fixture (window 16) chunks of 17, 24 and 48 rows leave logits, routes, keys, values and all convolution rings bit-identical to 12-row chunks on the f32 path; on oQ4e a 1,100-row chunk equals 512-row chunks bit for bit on the f32 path, logits and the next decode step, and both paths keep per-token decode's argmax (half path 0.237 max logit difference from 512-row chunks).
- A 600-token oQ4e prefill with every tensor-unit kernel against the shader-core kernels: same argmax, 0.221 max logit difference at scale 22.8.

The previous tests pass unchanged, including the decode-consistency tests: chunks of 1, 4 and 5 against per-token decode on the tiny fixture (zero difference), the oQ4e 64-token f32 prefill against decode (zero difference; its half path is 0.234 from decode with the same argmax), batch widths 1, 2 and 4 against decode (zero), rollback, fork and verification.

A first version of the long-chunk tests compared against per-token decode at 1.5e-5 and failed: on the tiny fixture any chunk over eight rows takes row-tiled kernels whose sums are not in decode's order (keys within one f16 ulp), and on oQ4e decode attention splits windows longer than 64 positions. Neither difference involves the window, so the tests compare long chunks with window-bounded chunks instead, which the f32 path must match bit for bit.

### What remains

At chunk 2,048 on the 4K prompt the candidate's 1,184 ms a chunk splits as: routed NVFP4 matrices 34% (`i2p_expert_units_up` 21.8%, `_down` 12.3%), staged projections 25% (`i2as_gemm_split` 22.3%, `i2as_split` 2.4%, `i2as_prepare` 0.8%), attention 9.4%, grouped low-occupancy experts 4.8%, affine experts 3.5%, f32-weight projections 2.9%, relative bias 2.0%, and 18% in small per-row kernels (`i2p_clip_scaled`, `i2p_gather_rms`, `i2p_conv`, `i2p_mix`, `i2p_ctl`, `i2p_combine`, …), which spill out of cache at long chunks and cost slightly more a row than at 512.

- The staged projections run at about 20 TFLOP/s in the model against 27–31 in isolation: six projections on a local layer's one input each split and multiply separately, and the 24- and 384-column ones fill 64-column tiles poorly. Splitting each input once and running the layer's projections as one staged GEMM would recover part of the gap. Rounding the inputs to half once would roughly double the rate again (59–62 TFLOP/s) at the cost of the inputs' precision.
- At chunk 1,024 the grouped f32 gate/up kernel still spends 26 ms per 512 rows on experts with fewer than 14 rows (the matrix threshold kept from the half path); routing them to the tensor units would change those rows' rounding.
- The first `bench` pass after load carries 0.15–2.2 s of host-dependent one-time cost. Committing the prefill scratch at setup (the `wip:` commit, built but never timed) and making the weight buffer resident at load are the in-process levers; the second belongs to the model loader, outside this lane. The prepared matrix ends with two pairs that time the first.
- The family's kernels, old and new, are still not registered with the kernel contract runner (noted in [Inkling prefill](inkling-prefill.md)).

### Reproduction

Builds (the base gets only the bench option, from an export of `4c2af5d2a`):

```sh
export PATH="/opt/homebrew/bin:$PATH" TMPDIR="$PWD/.work/tmp" ZIG_GLOBAL_CACHE_DIR="$HOME/repos/tf/.work/zig-global-cache"
zig build tf-inkling2-loop-run -Dcpu=apple_m1 -j1 --prefix "$PWD/.work/tmp/cand-final"
mkdir -p .work/tmp/src-base-r && git archive 4c2af5d2a | tar -x -C .work/tmp/src-base-r
git show ec9b5b990:zig/src/families/inkling2_loop/run.zig > .work/tmp/src-base-r/zig/src/families/inkling2_loop/run.zig
(cd .work/tmp/src-base-r && zig build tf-inkling2-loop-run -Dcpu=apple_m1 -j1 --prefix "$OLDPWD/.work/tmp/base-r")
zig build native -Dcpu=apple_m1 -j1
```

The 4,096-, 8,192- and 2,500-ID fixtures are the first IDs of `campaign/prompts/dev/code-4096.txt` (5,265 tokens) and `code-32768.txt` (43,088 tokens) tokenized by the runner, as in [the f16 KV timing procedure](inkling-f16-kv.md#timing-procedure). Every timed run used `python3 campaign/heavy.py -- .work/tmp/timed-run.sh MARKER RUNNER MODEL IDS ARGS` with that procedure's wrapper and `PERF_GATE_MAX_BUSY_CORES=4`; exits 75 and 76 were retried, never counted. Prefill rows used `bench 32 4 CHUNK 2` (base chunk 512, candidate 1,024), batch rows `batchbench K 5`.

Profiles:

```sh
TF_INKLING_PREFILL_PROFILE_BASE=0 TF_METAL4_TIMELINE=4 python3 campaign/heavy.py -- .work/tmp/timed-run.sh .work/tmp/prof.marker \
  "$PWD/.work/tmp/cand-final/bin/tf-inkling2-loop-run" "$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" \
  .work/tmp/code-4096.json profile-prefill 1024 > .work/tmp/prof/final-4096-1024.log 2>&1
python3 campaign/kernels/inkling_profile.py timeline --split --log .work/tmp/prof/final-4096-1024.log
```

Microbenchmarks: `.work/tmp/mb/bench.swift` and `.work/tmp/mb2/gemm.swift` (`swiftc -O`), run under `heavy.py` and the same wrapper; the expert source is `prefill.metal` + `prefill_expert_wide.metal` with the half-library defines prepended, plus the candidate kernels.

## Earlier P3 record at `b9bdf4e55`

P3 profiled admitted revision `b9bdf4e55f3b64855ce52a6359efb15463e5ae14`. The checkpoint is Inkling 2 Mini oQ4e revision `3eea042751cd`. The host is an M5 Max with 48 GiB unified memory. No kernel change has passed screening. The 1,050 tok/s target is not established.

### Baseline profile

The Metal 4 timeline reports mean GPU time per 512-row chunk after eight warm chunks. The 1K prompt has 1,024 tokens and the 4K prompt has 4,096 tokens. Both retained profiles exited 0 under the 28 GB guard, GPU and CPU admission checks, and the immediate CPU post-check. Each segment retains only its 25 largest kernel entries. Summed spans were 761.433 ms and 817.188 ms respectively.

| Kernel | 1K ms | 4K ms |
| --- | ---: | ---: |
| `i2p_expert_matrix_wide` | 372.109 | 380.721 |
| `i2as_gemm` | 96.717 | 100.457 |
| `i2p_gu_grouped` | 64.473 | 63.028 |
| `i2p_attn_half` | 53.952 | 93.930 |
| `i2a_half_group_up` | 40.431 | 42.020 |
| `i2p_relbias` | 20.880 | 21.831 |
| `i2p_down_grouped` | 19.014 | 20.470 |
| `i2a_half_group_down` | 18.609 | 19.407 |
| `i2af_gemm` | 13.227 | 14.022 |
| `i2as_prepare` | 9.679 | 9.884 |

The timeline does not include routed row counts, transferred bytes, or FLOP counts by kernel. Achieved TFLOP/s and GB/s cannot be derived reliably from these records. The earlier host measurements report a 56 TFLOP/s fp16/bf16 GEMM rate in [Inkling prefill](inkling-prefill.md) and a 521 GB/s read ceiling in [FP4](fp4.md). Neither is a measured rate for these dispatches.

Reproduce after building the runner and tokenizing the 4K fixture as in [the f16 KV timing procedure](inkling-f16-kv.md):

```sh
export PATH="/opt/homebrew/bin:$PATH" TMPDIR="$PWD/.work/tmp" INKLING_LOAD_LIMIT_GB=26
TF_METAL4_TIMELINE=2 python3 campaign/heavy.py -- sh -c 'GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60 gpu-gate-ready -1 || exit 75; perf-gate-ready || exit 75; "$@"; result=$?; perf-gate-ready -1 || exit 75; exit "$result"' sh \
  "$RUNNER" "$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" campaign/prompts/inkling/code-1024.json profile-prefill 512 > "$LOG" 2>&1
python3 campaign/kernels/inkling_profile.py timeline --split --log "$LOG"
```

For 4K use the tokenized 4,096-ID fixture and set `TF_METAL4_TIMELINE=8`.

### Rejected screens

The 64-row expert tile increased the 1K expert matrix span from 372.109 ms to 417.916 ms. A 128-column tile measured 383.324 ms. A cooperative TensorOps implementation that decoded weights inside each tile measured 1,353.219 ms. Each screen passed the admission and post-check gates, but each was a single profiled run. None was retained.

Changing the gate/up matrix threshold from 14 to 8 reduced grouped gate/up time from 64.473 ms to 27.668 ms and increased expert matrix time from 372.109 ms to 402.728 ms. Three interleaved throughput pairs did not establish an improvement. At 1K, base samples were 589.3, 577.2, and 548.9 tok/s and candidate samples were 613.1, 594.8, and 583.1 tok/s. At 4K, base samples were 518.2, 633.8, and 624.5 tok/s and candidate samples were 629.0, 504.6, and 604.6 tok/s. The ranges overlap substantially. The threshold change was discarded.

### Correctness

The restored source passed `zig build native -Dcpu=apple_m1 -j1` with exit 0. `python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu` exited 0: 35/35 host tests and 82/82 GPU tests passed, with no skips. The GPU log contains `failed command` labels beside successful test summaries, despite the 9/9 build-step and 82/82 test pass summary. Peak charged memory was 22.3 GB of 28 GB.

The [saved-capture procedure](inkling-f16-kv.md#saved-capture-fidelity) produced 2,763 matches in 3,015 positions, 91.6418% top-1 agreement, mean KL 0.0388930435, and zero non-finite logits. The streamed comparator exited 0. The cross-checkpoint f32 parity command exited 2 under its stricter contract after producing all 16 captures, as described in that procedure. No float32 oracle model was loaded.

### Identical-source prefill comparison

Both runners were built separately from the admitted source after rejecting the experiments. The retained cells used `bench 32 4 512`, the corrected Inkling-tokenized fixtures, the 28 GB guard, GPU and CPU admission, the immediate CPU post-check, and a one-second compiler monitor. Pair order was base/candidate, candidate/base, base/candidate. Exit 75 and compiler-overlap attempts were discarded and retried.

| Prompt | Base tok/s | Candidate tok/s | Base median | Candidate median |
| --- | --- | --- | ---: | ---: |
| 1K | 316.1, 193.6, 305.0 | 255.6, 488.0, 280.7 | 305.0 | 280.7 |
| 4K | 516.4, 483.5, 626.3 | 446.0, 475.8, 631.2 | 516.4 | 475.8 |

The source is identical on both sides. The large spread and sustained page-ins make these samples unsuitable for attributing a performance difference to code. One 4K candidate run reported 2,818,491 page-ins while the guard included time waiting for the shared GPU lock. Earlier screening pairs on the same source also varied widely. The samples do not establish the target or a regression.

The per-cell invocation followed [the f16 KV timing procedure](inkling-f16-kv.md#timing-procedure), using its `timed-run.sh` compiler monitor:

```sh
python3 campaign/heavy.py -- "$PWD/.work/tmp/timed-run.sh" "$MARKER" \
  "$RUNNER" "$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" "$IDS" bench 32 4 512
```

For batch, the same paired procedure used the corrected 2,500-ID fixture and `batchbench K 5`. Values are the median milliseconds per step reported by each invocation.

| Width | Base ms | Candidate ms | Base median | Candidate median |
| --- | --- | --- | ---: | ---: |
| 1 | 21.611, 21.785, 21.662 | 21.684, 21.789, 21.755 | 21.662 | 21.755 |
| 4 | 44.257, 42.288, 41.983 | 42.404, 41.825, 44.507 | 42.288 | 42.404 |

The candidate differences are within the observed spread. Single-stream decode medians from the 1K prefill runs were 20.584 ms for base and 20.588 ms for candidate. At 4K they were 22.408 ms and 24.092 ms, within the candidate's 22.455–24.579 ms range. These are comparisons of the same source under a variable host, not evidence of an optimization.

### Two-agent sweep

The gated two-agent sweep exited 0. Its two repetitions produced 44.354 and 64.286 aggregate tok/s for 744 output tokens each. The first repetition had one cold request and seven requests with cached tokens. The second reused cached tokens in all eight requests. Both sampled peak charges were 20,899,623,656 bytes. The guard reported 20.9 GB peak charged of the 28 GB cap. The server drained cleanly and its completed requests reported zero accepted or drafted tokens.

The first attempt exited 1 because the Python client wrapped an admission deferral in a `CalledProcessError` reporting exit 75. It was discarded and retried. The retained attempt passed the script's prefix and request checks and the additional CPU post-check.

```sh
mkdir -p .work/tmp/gates
cat > .work/tmp/gates/perf-gate-ready <<'SH'
#!/bin/sh
if [ "$#" -eq 0 ]; then
    GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60 /Users/josh/.local/bin/gpu-gate-ready -1 || exit "$?"
fi
exec /Users/josh/.local/bin/perf-gate-ready "$@"
SH
chmod +x .work/tmp/gates/perf-gate-ready
PATH="$PWD/.work/tmp/gates:/opt/homebrew/bin:$PATH" TMPDIR="$PWD/.work/tmp" INKLING_LOAD_LIMIT_GB=26 \
INKLING_MODEL="$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" sh campaign/fx/sweep-subagents.sh 2 4096
perf-gate-ready -1
```
