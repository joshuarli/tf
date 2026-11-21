# M5 Max native inference campaign: Flint, then Inkling 2 Mini

## Status

- **Goal:** a faster, leaner native Zig/Metal TensorFold on this M5 Max. Flint comes first, then a qualified Inkling 2 Mini port (see **Acceptance**).
- **Upstream frozen at:** `fa13df5dd1fdd23ada775510a464c9aa03a39e09` (TensorFold `main`, 2026-11-21). The campaign branch is `campaign/m5`.
- **Last accepted revision:** `7ed909515` (C1 merge).
- **Accepted:** C0 (fetch, host and build-baseline exit 0; second fetch downloads nothing); F1 (`--no-drafts` already disables copies and the drafter: `no_drafts_test.zig` in `test-flint-draft-host`; `make -C campaign smoke MODEL=flint MODE=ar` gives 512 tokens, 0 drafted, 1.00 tokens per decode round, about 169 tok/s); C1 (`make -C campaign bench MODEL=flint MODE=ar CELLS=1024 PAIRS=1` plus a byte-identical second `scorecard`; baseline ar at 1,024 tokens decodes at 153 tok/s with a 22.5 GiB peak); Q1 (`make -C campaign quality MODEL=flint SET=dev`: 12/12 on the baseline; 48 cases; libtfgrammar built from xgrammar v0.2.8 `97787376`); H1 (`make -C campaign pi-harness MODEL=flint`: 8/8 completed, 8/8 checkers passed, 0 tool-call parse errors); I0 (`make -C campaign ref-check MODEL=inkling2-mini`: the float32 reference matches llama.cpp `40b7a546` top-1 on every confident position of 4 histories, max logit error 1.4e-4, mean KL 3e-11; recipe at `docs/recipes/inkling-2-mini.md`). **Red:** none beyond **Baseline**. **Parked:** none.
- **In flight** (bridge workers; lane `.work/wt/<lane>`, branch `lane/<lane>`):
  - F2 `f2-parity`, `usrw_4bb1c566-c653-43f6-8d38-cab9288ef301`;
- **Next:** F3 after F2; I1 after I0.
- **Resume:** run `worker_status` on each in-flight session, then review and merge each lane whose acceptance passes. `.work/wt/c0-ground` holds an earlier C0 worker's leftover worktree. That worker has exited, and its branch is merged except commit `91a5a06a1`, which was superseded and not adopted.

## Charter

- Execute until **Acceptance** is met or every remaining item is parked behind a hard stop. Never stop after writing a plan. Before ending a session, rewrite **Status**, including any lane paths and the first command to resume. Resume by reading **Status**, checking that `campaign/host.json` still matches the machine, and taking the next item.
- Decide and keep going. Record a decision in **Decisions** only when a later slice depends on it.
- The execution grant: everything listed under **Execution boundary**. Park any item that needs more than that.
- Hard stops, each of which parks the item:
  - an action that is irreversible or acts outside this repo, `~/models` and `.work/`;
  - an effect that **Execution boundary** forbids;
  - a loop that cannot check anything.
  
  Pushing is always forbidden.
- Failing items: find the cause before retrying (`root-cause`). After three attempts at the same item, or three failed variants of one optimization idea, park the item. Park it on its branch with the cause and a retry condition, then go to the next ready item.
- Fix egregious problems in their own commit when you meet them. Egregious means any of:
  - a crash or hang over 60 s;
  - resident memory growing more than 5% across 10 identical requests;
  - a warm no-op `zig build native` over 60 s;
  - a campaign script over 2× its recorded time with unchanged inputs.
  
  Record anything smaller in the ledger and leave it.
- Parity before optimization. Product performance is measured and logged from F3 on, but it gates nothing until phase 4.
- **Lanes:** parallel for the disjoint sets named in **Sequence**, serial otherwise. Each lane works in `.work/wt/<lane>` on branch `lane/<id>`, following `orchestrate`'s worktree lifecycle; never `/tmp`. Set `TMPDIR=$PWD/.work/tmp`. The parent integrates into `campaign/m5` and runs the parent steering checkpoint (`orchestrate` skill, `references/campaigns.md`, section "Parent steering checkpoint") after each accepted slice.
- **GPU lock:** take it for any timed run, any process that loads a full model, and the Metal device suite. Use `lockf ~/repos/tensorfold-m5/.work/gpu.lock <command>`. No compiles inside a timed window. Stop only processes the campaign started.
- **Commits:** Conventional Commits, scoped to the workstream (`campaign`, `flint`, `inkling2-loop`, `metal`). Put the item ID in the body. End every commit with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Code comments never cite this doc.
- **TensorFold rules:** follow `docs/recipes/SKILL.md`, `docs/kernel-contract.md` (for GPU work) and `docs/recipes/adding-a-zig-family.md` (for I-items). Skip their ATC, PR and publication steps; see **Execution boundary**.

## Decisions

The owner made the first two decisions on 2026-11-21:

1. **Owner: layout.** One repo at `~/repos/tensorfold-m5`, a TensorFold clone with `upstream` as a fetch-only remote (push URL `DISABLED`). Campaign tooling and records live in `campaign/`. The original draft lived at `~/d/infer/plan.md`; this file supersedes it.
2. **Owner: no push.** Commits stay local and the owner pushes.
3. **Owner: build inputs through `laputa-buildlib`.** Every downloaded input is pinned in `campaign/sources.lock.json` (schema 1, see the [build repository contract](https://github.com/laputa-systems/stuff/blob/main/docs/build-repos.md)) and fetched through `buildlib.locks.fetch_input`. Downloaded inputs include weights, comparator sources and the xgrammar sources. Pin buildlib in `campaign/pyproject.toml` as follows and commit `campaign/uv.lock`:
   - version `laputa-buildlib==0.1.0`;
   - index `https://pypi.17166969.xyz/current/simple/`;
   - wheel sha256 `edd102a1713c903f93e19b10036983d123dd76fa1b04a0e5f3c5070f00a95383`.
   
   `campaign/bootstrap.lock.json` pins the Zig 0.17.1 macOS aarch64 tarball. The executor uses the installed Zig after checking it matches that tarball's version. Python 3.15.1 is installed.
   - **Owner, 2026-11-21: Zig 0.17.1.** The owner upgraded the installed Zig and pinned 0.17.1. `build.zig` requires an exact `.zig-version` match, so `campaign/m5` sets `.zig-version` and `minimum_zig_version` to 0.17.1. The frozen baseline is `fa13df5d` with only those two lines overridden (`make -C campaign baseline-toolchain` applies the override and refuses any other diff), so the baseline and every candidate share one compiler.
4. **Owner: the harness is tested with Pi.** Coding-agent serving checks drive the local server with `pi` 1.5.0 (H1). Use a campaign-local `PI_CODING_AGENT_DIR=campaign/pi/agent` with `PI_OFFLINE=1`, `PI_TELEMETRY=0`, `--no-extensions --no-mcp --no-skills -nc --no-session --mode json`. Never read or write `~/.pi`.
5. **Owner: Inkling 2 Mini is the Inkling target.** That is `thinkingmachines/Inkling-2-Mini` at `ed40e7da625f594fc9c4990f5f66be24873c22d7`, `model_type: inkling2_loop`. Upstream's `zig/src/families/inkling2/` is Inkling 2 **Small**: CUDA/EXL3 only, about 394 GB, and not served. That makes Mini a new Metal family. Reuse the Small package's config, template and tokenizer code only where its tests show the contracts match.
6. **Prescribed: base and baseline.** TensorFold stays the base: it is Apache-2.0, it serves Flint natively on Metal (#1177, merged), and it has in-tree NVFP4/FP8 readers and Metal kernels (`zig/src/core/nvfp4.zig`, `zig/src/core/quant/fp8.zig`, `zig/kernels/metal/drama3/nvfp4.metal`). The frozen Flint baseline is a build of `fa13df5d`. The Inkling baseline is the first qualified port (I3), labelled "first native port", never "upstream". Reverse only if a licensed, buildable alternative is shown to save a phase.
7. **Prescribed: weights.** Weights go in `~/models/<repo-name>@<rev12>/`, pinned and verified by sha256:
   - Flint: `RecursiveSI/flint-int4@3b93f9053b530488405ae9fca51d4900d6ce21f0`, 22.6 GB.
   - Flint drafter: `incoai/Flint-DFlash2@39b0a7f7a12a14461b9eac8d90a9fcec24168889`, 0.8 GB.
   - Inkling 2 Mini: 20.2 GB, mixed NVFP4/MXFP8. It includes `mtp.safetensors`, its native draft head.
   - Comparator weights, fetched only when their item runs.

   The fetch total stays under 120 GB; 621 GiB is free. Never delete files you did not create. Only one full model is resident at a time.
8. **Prescribed: memory.** The machine has 48 GiB of RAM. C0 records the Metal device's `recommendedMaxWorkingSetSize` (W).
   - Every benchmark server sets `FLINT_LOAD_LIMIT_GB=floor(0.9·W/1e9)` (or the Inkling equivalent) and an explicit `--prompt-cache-gib`.
   - Prompt cache is 0 for cold-prefill cells and 2 otherwise.
   - Speculation may add at most 4 GiB of draft residency, which is min(4 GiB, 10% of RAM).
   - A run that sees memory pressure or thermal throttling (`pmset -g therm`, `memory_pressure`) is recorded as an invalid trial and the workload is reduced. Never change system settings.
9. **Prescribed: references.**
   - **Flint:** the oracle is `rec_mlx` (`recursive-oss/rec` v0.1.0 at `e53663b2`), which is where TensorFold's kernels match rounding. It runs in a `.work/venv-rec` uv environment as a development tool only.
   - **Inkling 2 Mini:** the semantic reference is llama.cpp `src/models/inkling2-loop.cpp` plus `conversion/inkling2.py`, pinned at the commit I0 records (master was `40b7a546defe1f60edd1338c719552ffc58c89a0` on 2026-11-21). The vLLM recipe and the model card's linked code are cross-checks. The numerical oracle is `campaign/ref/inkling2_loop_ref.py`: a CPU float32 reference that streams one layer at a time and decodes the checkpoint's exact NVFP4/MXFP8 values. It needs no 105 GB float32 load. `trl-internal-testing/tiny-Inkling2LoopForConditionalGeneration@5e761216759d5d040c09f458d5f077b9aaa5ac78` is the tiny unit fixture.
   - **Delegated (I0):** if llama.cpp converts the official checkpoint to GGUF without requantizing, use it as a second teacher-forced oracle and as the matched comparator. Otherwise it goes only in the "best useful" table. Tie-break: the float32 reference wins.
10. **Prescribed: quality rules.** These are fixed now and never relaxed later.
    - Unchanged-arithmetic transformations must be bit-identical to the frozen baseline on logits (`tf-flint-run exact` or the Inkling equivalent).
    - Reordering changes are judged on the F2/I2 teacher-forced corpus:
      - The candidate's mean KL to the oracle must stay within 1.10× the baseline's.
      - Top-1 agreement with the baseline must be at least 99.5% on positions where the baseline's top-2 margin is at least 0.1 logit.
      - Every disagreeing position must be traced to its first divergent tensor before acceptance.
    - Q1 quality suite: no more than one case lost against the baseline, and every lost case is examined. Lossy requantization and KV compression are separate experimental configurations, never promoted as speedups.
    - Deterministic operator, state and protocol fixtures allow zero new failures.
11. **Prescribed: performance gates.** These apply in phase 4 only.
    - **Promotion:** a geometric-mean gain of at least 3%, with a paired 95% CI above 1.0.
    - **No regressions:** no primary-cell decode or prefill regression above 5%, and peak-memory growth no more than max(2%, 128 MiB).
    - **Memory route:** a change may instead qualify with at least 5% lower peak memory and no latency regression above 2%.
    - **Pairing:** at least 6 balanced baseline/candidate process pairs across two sessions, expanding to 12 only to settle a borderline result; otherwise the result is "inconclusive".
    - **Dispatch:** workload-specific dispatch is allowed, and the final dispatching build is validated on every primary cell. A gain below 3% is kept only with a stated simplification or cumulative benefit.
12. **Prescribed: targets.** These are goals, not gates. Unmet targets are reported, never hidden.

    | Milestone | Target |
    | --- | --- |
    | Flint runtime | ≥20% geometric-mean true-autoregressive decode gain over the frozen `fa13df5d` build |
    | Competitive | ≥10% geometric-mean gain over the strongest reproducible comparator on the same suite |
    | Stretch | ≥30% runtime gain, or a substantial measured memory reduction at equal speed and quality |
    | Inkling | Qualified native text inference, then the same optimization process against I3's baseline |

13. **Prescribed: speculation.** Advertise greedy-only speculation first, labelled as such. A sampled mode needs the Leviathan et al. acceptance/rejection correction, or the model's native method. Its proof is the item's own enumerable-distribution tests and argument, as §7 of the source draft required (see **Sequence** F5/I5). Every mode keeps its no-drafter path and automatic fallback, and fallback cost is included in timings.
14. **Delegated: optimization order.** Inside phase 4, the executor ranks hypotheses by measured cost share and keeps the ranked table in `campaign/ledger.md`. Candidates come from §8 A–E of the source draft:
    - A: Exo #2533 fusions;
    - B: M5 TensorOps/MPP matrix paths and quantized layouts;
    - C: attention and state;
    - D: larger fusion or persistent execution, only where dispatch cost is significant;
    - E: model-specific speculation.
    
    Bound: one experiment per item, smallest decisive change first. Tie-break: lower correctness risk.
15. **Prescribed: Exo's M5 exclusions.** Exo PR #2533 (head `c4685725bf6a71fb424e668744744499d44a4fea`, open) gates its fused kernels to M1/M3/M4. The reason it gives is "until verified on that hardware". That is missing validation, not a known numerical difference. Enable a port on M5 only after a bit-exact check against the unfused path on this machine. Exo PR #2530 (open) targets Inkling-**Small**, not Mini; its ideas transfer only where the Mini graph has the same operator.

## Baseline

[baseline.md](baseline.md) holds the frozen binary's sha256, the loop timings and every known red test. In brief, for `fa13df5d` built with Zig 0.17.1:

- `zig build native -Dcpu=apple_m1 -Dversion=1.0.0`: 239.3 s cold and 4.0 s warm no-op;
- `zig build test-flint-draft-host`: 17.7 s;
- `zig build test test-golden -Dcpu=apple_m1`: 547.6 s under the GPU lock, with 7 known red steps (the memcpy link check plus three tests in two modules each).

The benchmark load limit is `FLINT_LOAD_LIMIT_GB=36`, from W = 40,200,896,512 bytes.

Host facts measured on 2026-11-21:
- Apple M5 Max: 6 performance and 12 efficiency CPU cores, a 40-core GPU with Metal 4, and 48 GiB of RAM.
- macOS 27.2 (26B123), Xcode 27.2 (27B52), Metal toolchain 27.2.47.0.
- Zig 0.17.1 (owner upgrade; upstream `.zig-version` was 0.17.0). Python 3.15.1, uv, and Pi 1.5.0.
- 621 GiB of free disk.

## Sequence

The phases run in order. Within a phase, items whose owned files are disjoint run in parallel. Every item ends on its acceptance command plus the narrowest affected suite, run on the integrated `campaign/m5`. The full `zig build test test-golden` runs at milestones M1 (after F3), M2 (after I3) and the close.

**Disjoint ownership sets:**
- `campaign/` (C, Q and H items, plus the F2 and I2 harness files): one writer at a time.
- Flint: `zig/src/families/flint/`, `zig/src/native/flint.zig`, `zig/kernels/metal/flint/`, `zig/tests/flint/`, `zig/tests/contract/metal_flint*`.
- Inkling 2 Mini: `zig/src/families/inkling2_loop/`, `zig/src/native/inkling2_loop.zig`, `zig/kernels/metal/inkling2_loop/`, `zig/build/inkling2_loop.zig`, `zig/tests/inkling2_loop/`, `campaign/ref/`.
- Shared: `build.zig`, `zig/build/metal_*.zig`, the native registry, `zig/src/core/`, `zig/src/server/` and `zig/kernels/metal/core/`. One registration edit is folded into the item that needs it, and the parent merges these. Any other shared-core change is its own serial item. While it is open, it blocks the other workstream from touching core.

GPU-timed and full-model steps serialize on the GPU lock even when their code lanes run in parallel.

### P0: ground

- **C0. Ground** (S, no prerequisites). Owns:
  - `campaign/pyproject.toml`, `campaign/uv.lock`, `campaign/sources.lock.json`, `campaign/bootstrap.lock.json`;
  - `campaign/fetch.py`, `campaign/host.py`, `campaign/host.json`, `campaign/Makefile`;
  - `campaign/baseline.md` and `campaign/README.md`.

  It delivers:
  - `make -C campaign fetch [NAMES=...]`, an idempotent fetch with a hash check. It skips verified files, and small Hugging Face files are pinned by the sha256 of their first fetch at the pinned revision.
  - `make -C campaign host`, which writes `host.json`: sysctl, `system_profiler`, `sw_vers`, Xcode, Metal toolchain, Zig, and the Metal device name, family support and `recommendedMaxWorkingSetSize` through a `swift` probe.
  - `make -C campaign build-baseline`, which builds `fa13df5d` in `.work/wt/baseline` and records the sha256 of `tensorfold-native`.
  - **Baseline** timings and known red.

  **Acceptance:** `make -C campaign fetch NAMES="flint-int4 inkling-2-mini" && make -C campaign host && make -C campaign build-baseline` exits 0, the second `fetch` run completes with no downloads, and `baseline.md` lists each timing and every red test.

### P1: Flint parity and harness

- **F1. True autoregressive switch and first inference** (S, after C0). Owns the Flint set.
  - Prove whether `--no-drafts` disables Flint's copy drafts as well as `--drafter`. Make the per-request lane counters show 0 proposals and exactly 1.00 tokens per round.
  - If it does not, add a verified switch and a test that fails when copies run.
  - Serve one greedy request.
  
  **Acceptance:** `zig build test-flint-draft-host`, plus a new Flint test for the switch; under the lock, `make -C campaign smoke MODEL=flint MODE=ar` returns 512 tokens with zero proposals.
- **C1. Benchmark harness** (M, after C0; parallel with F1). Owns `campaign/bench/`, `campaign/prompts/` and `campaign/scorecard.py`.
  - Build on `tensorfold benchmark` and `tools/bench_*.py`; do not write a new framework.
  - **Corpus:** fixed prompts covering code, prose/reasoning and structured output, at actual token lengths 1,024, 4,096 and 32,768, built from natural varied text with no repeated filler. Add a separate held-out set. Hash every prompt.
  - **Modes:** `ar` (`--no-drafts`), `copy`, `drafter` and `combined`.
  - **Fixed length:** `ignore_eos: true` with 512 output tokens on every engine. Natural terminations are reported separately.
  - **Records:**
    - startup and load time, cold prefill and warm requests, and prefix-cache hits, with cached and uncached token counts;
    - TTFT, prefill rate, decode rate and total latency;
    - peak and steady footprint;
    - acceptance and committed tokens per verification.
  - **Decode interval:** from the end of prefill to the last committed token, using synchronized wall time.
  - Raw trials go to `campaign/runs/<date>/<run>.jsonl`, tracked in git, including failures. Binary, weight and config hashes go in each record. `scorecard.py` regenerates `campaign/scorecard.md` from `runs/` alone.
  - `--publish` and `--upload-url` are never passed.
  
  **Acceptance:** `make -C campaign bench MODEL=flint MODE=ar CELLS=1024 PAIRS=1 && make -C campaign scorecard`. A second `scorecard` run must produce an identical file.
- **F2. Flint reference parity** (M, after F1). Owns `campaign/parity/flint/`.
  - Run teacher-forced logits from `tf-flint-run` against `rec_mlx` on 16 held-out histories of at most 4,096 tokens.
  - Report the full distribution of error: absolute and relative error, top-1 agreement, logit margins and KL.
  - Trace the first divergent tensor for any outlier. Check tokenizer, masks, quantization and config before calling a difference "floating-point noise".
  - Write the baseline numbers into **Baseline**.
  
  **Acceptance:** `make -C campaign parity MODEL=flint` exits 0 and writes `campaign/parity/flint/report.md`.
- **Q1. Quality suite** (M, after C1). Owns `campaign/quality/`.
  - 48 cases, drawn from public or local evaluations where practical:
    - 16 executable code tasks, scored by tests;
    - 12 structured-output or tool-call cases, which need `zig build grammar` with xgrammar v0.2.8 pinned in the lock;
    - 12 reasoning cases;
    - 8 context-retrieval cases at 4K and 32K.
  - A development subset of 12.
  
  **Acceptance:** `make -C campaign quality MODEL=flint SET=dev` scores the frozen baseline and records it.
- **H1. Pi harness** (S, after C1). Owns `campaign/pi/`.
  - `campaign/pi/agent/models.json` defines a local OpenAI-compatible provider at `http://127.0.0.1:8888/v1`.
  - 8 coding tasks run under `sandbox-exec -f campaign/pi/sandbox.sb`. Writes are allowed only in `.work/pi/<task>` and `.work/tmp`; network only to localhost.
  - Record task pass/fail, tool-call parse errors, turns and per-turn latency.
  
  **Acceptance:** `make -C campaign pi-harness MODEL=flint` gives 8/8 runs completing, and 0 tool-call parse errors on the baseline.
- **F3. Freeze the Flint baseline matrix** (M, after F1, C1 and F2). Owns `campaign/runs/`.
  - Run the primary matrix, 3 prompts × 3 lengths, on the frozen build in `ar` and `copy` modes with 6 pairs.
  - **Comparators:** `rec` v0.1.0, Exo `main` and the #2533 head, and mlx-lm where it serves Flint. Pin each in the lock. An engine that cannot run here is marked unavailable.
  - Rach's engine and codegraff's nanoattention get a 30-minute check for public, licensed, runnable source. Code under the "Other" licence is benchmarked only if its licence permits, and never copied.
  - Record omitted cells and the reason for each.
  
  **Acceptance:** `make -C campaign scorecard` shows both tables, matched and best-useful, for every runnable engine. **Milestone M1:** full suite.

### P2: Inkling 2 Mini parity

I-items own the Inkling set. They may start code work parallel to F-items, and use the GPU only under the lock.

- **I0. Prior-art map and reference** (M, after C0). Write `docs/recipes/inkling-2-mini.md` from the pinned llama.cpp graph, the checkpoint index, `hf_quant_config.json` and the chat template. It must cover:
  - the 24 layers, with `local_layer_ids` and a 1024-token sliding window;
  - the core loop from layer 6 to 18 with 3 passes and `gate_rank` 64; each pass keeps its own KV and convolution state, and later passes take different Q/K/V inputs;
  - the attention-residual blocks;
  - `d_rel`/`rel_extent` relative positions and log scaling;
  - the qk and rel L2 clips;
  - 128 experts with 8 active per token, gate bias, `route_scale` and softcaps;
  - the final logit softcap of 30, and the 201,024 vocabulary padded from 200,058;
  - the per-tensor NVFP4/MXFP8 groups and the MTP head taps.
  
  Then build `campaign/ref/inkling2_loop_ref.py` and check it against the tiny fixture's transformers output, if that output exists, or against llama.cpp.
  
  **Acceptance:** `make -C campaign ref-check MODEL=inkling2-mini` reproduces llama.cpp's teacher-forced top-1 on 4 short histories and states the error it finds.
- **I1. Family loader and refusals** (M, after I0). This covers config, tensor map, tokenizer and template, NVFP4/MXFP8 geometry checks, registration, and a clear refusal of image parts and `--vision`. Vision is out of scope.
  
  **Acceptance:** `zig build test-inkling2-loop-host` passes, including tests that the refusals fire.
- **I2. Forward pass on Metal, teacher-forced** (L, after I1). Reuse the shared NVFP4/FP8 kernels and the Flint/GLM attention pieces where their contracts match; write new kernels only where they do not. State stays explicit for each pass.
  
  **Acceptance:** `make -C campaign parity MODEL=inkling2-mini` meets Decision 10 against the reference on 16 histories.
- **I3. Serving and baseline** (M, after I2, C1, Q1 and H1). Cover chunked versus full prefill, cached continuation versus a fresh request, EOS, cancellation, context limits, reset after errors, reasoning template, special tokens and tool-call syntax. Then freeze the Inkling baseline matrix. Comparators: llama.cpp (Metal, ggml-org GGUF), MLX (`mlx-community/Inkling-2-Mini-4bit` through mlx-vlm), Exo and Ollama where they run.
  
  **Acceptance:** `zig build test-inkling2-loop`, `make -C campaign quality MODEL=inkling2-mini SET=full`, `make -C campaign pi-harness MODEL=inkling2-mini` and `make -C campaign scorecard`. **Milestone M2:** full suite.

### P3: hardening and speculation

- **F5 and I5. Greedy speculation** (M each; F5 after F3, I5 after I3). F5 is Flint DFlash2 plus copies. I5 is the Inkling MTP head. Test:
  - zero, partial and full acceptance;
  - EOS and the output limit inside a block;
  - rollback of every KV, convolution and per-pass state;
  - resumption after fallback;
  - block versus stepwise verification masks.
  
  Speculative output must equal the qualified target's greedy output on the same histories. Then add sampled mode: enumerable distributions, zero-probability cases, top-k/top-p, temperature, repetition penalty, and grammar constraints. Tune draft length from total measured cost.
  
  **Acceptance:** each model's speculation tests plus `make -C campaign bench MODE=drafter` within the speculation memory envelope.
- **K1. Contract write-up:** record the contract changes listed under **Execution boundary**, and update `docs/supported-models.md` and the recipes.

### P4: optimization

- **O-items** (each S–M). Each is created from the ranked hypothesis table after profiling: Metal System Trace and capture outside timed windows, short and long context. Profiling attributes time to experts and projections, attention, routing, norms and residuals, quantization, dispatch, synchronization, sampling and memory movement, together with bandwidth estimates.
  
  Each O-item follows the source draft's §10 loop: bottleneck, hypothesis, smallest change with a fallback, focused correctness checks, a short paired screen, then the full matrix and held-out prompts for winners. Commit winners. Revert losers but keep their ledger entry and runs. Re-profile and pick the next.
  
  Flint O-items and Inkling O-items run in parallel lanes, with the GPU serialized.
  
  **Plateau:** stop a model's optimization after re-profiling and at least eight substantive failed candidates across more than one hypothesis family, or sooner if a measured hard bound (for example, decode at about 90% of measured weight-read bandwidth) establishes the limit.
- **S2. Secondary investigations:** 128K context and 2 or 4 concurrent streams, reported as aggregate throughput, separate from single-user speed.

### P5: close

- **Z1.** Fetch upstream once and record what changed, without rebasing; import upstream fixes only as separately recorded commits. Use `dead-code-archaeology` to delete campaign scaffolding and kernels made dead by a promoted replacement; prove deadness by search plus the build. Run `ultrareview` and fix its P1 and P2 findings. Write `campaign/report.md`:
  - the Mac, models and quantization, and lengths and modes;
  - raw rates with uncertainty, memory peaks and baseline commits;
  - task quality and its limits;
  - local reproduction kept separate from external claims;
  - regressions, unsupported features, targets met or missed, and the narrowest claim the data supports.

## Acceptance

1. **Native engine.** `zig build native -Dcpu=apple_m1` builds. Flint and Inkling 2 Mini serve text. Qualified defaults are on, with opt-in experimental modes and fallbacks. `zig build test test-golden` adds no red beyond what **Baseline** lists.
2. **Correctness.** Run `make -C campaign parity MODEL={flint,inkling2-mini}` and `make -C campaign quality MODEL=… SET=full`; both must meet Decision 10.
   - Speculation tests pass, including state rollback.
   - **Failure condition:** any advertised mode whose output differs from its target-distribution contract, or a free-run text comparison standing in for teacher-forced evidence.
3. **Harness.** `make -C campaign bench …` and `make -C campaign scorecard` regenerate both tables from `campaign/runs/`. `make -C campaign pi-harness` passes for both models.
4. **Provenance.** `make -C campaign fetch` verifies every pin. `campaign/sources.lock.json` and `host.json` record source SHAs, PR heads, compilers, model revisions, tokenizer and config hashes, quantization metadata and licences. `NOTICE` and `THIRD_PARTY_NOTICES.md` cover any imported code.
5. **Ledger and report.** `campaign/ledger.md` lists every experiment, including failures. `campaign/report.md` states each target in Decision 12 as met or not met, with its numbers.
6. **Docs.** `campaign/README.md` says how to build, fetch weights, run, benchmark, verify and resume.

## Needs environment

- M1–M4 compatibility of new kernels: an offline compile only, run with `zig build dev-tools`. Running them needs older Macs; the command is the same `make -C campaign parity` on such a machine.
- Linux and CUDA host CI (`zig build -Dfixed-cpu test-host`) on a Linux host, if the owner wants upstream-style evidence.
- Comparators that need CUDA, such as vLLM and SGLang: unavailable here, not defeated.

## Execution boundary and non-goals

**Authorized:**
- edits and local commits on `campaign/m5` and `lane/*` branches in this repo;
- `git fetch upstream`;
- downloads pinned through the lock: public weights, comparator sources and Python development oracles in `.work/` uv environments;
- writes under `~/models/`;
- local servers on 127.0.0.1;
- GPU runs and profiling under the lock;
- adding development-only Python tools in `campaign/`.

The shipping path stays native Zig/Metal with its existing C/Objective-C glue. A new Zig or native runtime dependency must:
- keep the stripped `tensorfold-native` size growth at or below 2%;
- add no decode-path cost;
- be recorded in the lock with its licence.

**Forbidden:**
- pushing, PRs, issues or comments upstream;
- ATC `claim`, `start`, `pulse`, `land` or `--public`;
- `tensorfold benchmark --publish` and `--upload-url`;
- renting compute or buying services;
- changing system-wide security, memory or power settings;
- writing to `~/.pi`, `~/.cache` or other global caches, beyond reading the Hugging Face cache;
- deleting files the campaign did not create;
- Python, MLX or another framework in the shipping path;
- a language rewrite.

**Contract changes for the owner** (K1 keeps this list current):
- the new `inkling2_loop` family and its CLI and capability surface;
- any new switch from F1;
- any default changed by a promoted O-item.

**Non-goals, with reasons:**
- distributed inference, CUDA optimization, training and new UI: all outside the single-Mac scope;
- Inkling vision input: text and coding use come first, and images are refused with a clear message;
- the variable-loop research checkpoint (`TML-d24-varloop-research`): a different configuration;
- compatibility work for older Macs: see **Needs environment**;
- repository-wide mutation, coverage or formal-proof work: only the changed inference path is tested.

## Reference

These references were verified on 2026-11-21:

- **TensorFold.** Apache-2.0. Flint recipe at `docs/recipes/flint.md`. Flint native entry at `zig/src/native/flint.zig`. `--no-drafts` defined at `zig/src/server/cli.zig:88`. `ignore_eos` honoured at `zig/src/server/chat_generation.zig:82`. `tf-flint-run` registered at `zig/build/metal_targets.zig:283`. The Inkling 2 Small host package is wired at `build.zig:123`.
- **Upstream receipts** in `docs/recipes/flint.md` come from an M5 Ultra with 256 GiB. They are not a baseline for this machine.
- **`buildlib.locks`:** `load_sources` and `fetch_input` (URL order, digest and size checks, atomic publish).
- **Hugging Face pins:**
  - `Inkling-2-Mini` `model-00001-of-00001.safetensors`: sha256 `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1`, 19,526,019,168 bytes.
  - `mtp.safetensors`: sha256 `7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2`, 635,801,416 bytes.
  - `tokenizer.json`: sha256 `7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6`.
