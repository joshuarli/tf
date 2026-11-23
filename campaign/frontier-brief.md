# TensorFold: advance the Inkling 2 Mini inference frontier

Continue the optimization campaign in joshuarli/tf. Establish how current TensorFold compares with the strongest reproducible Inkling 2 Mini implementations, then improve the native Zig/Metal engine on the campaign machine. Optimize complete coding-agent work: cold and warm turn latency, sustained generation, concurrent throughput, memory, and useful task completion.

Treat this as a research-and-implementation campaign. Use the source map below as a starting inventory, discover newer work, and turn the best supported ideas into measured experiments. “SOTA” means an explicitly stated workload, hardware, memory budget, quality level, and date. It is not a universal ranking.

Research snapshot: 23 November 2026. Refresh every moving branch and PR before using it.

## 1. Recover the current ground truth and preserve campaign scope

Read the current AGENTS instructions, campaign/plan.md, campaign/shootout.md, campaign/kernels ledgers, current source, pending branches, and latest owner steering. Resolve historical statements by chronology and actual code. Preserve the current delegation and integration policy.

The target is Inkling 2 Mini, architecture inkling2_loop. Distinguish it from Inkling 2 Small, earlier Inkling models, research checkpoints, and modified/abliterated variants. Other models supply possible mechanisms, not interchangeable benchmark results.

The known campaign host is M5 Max, 40 GPU cores, 48 GiB. Reconfirm OS, Xcode/Metal, compiler, build options and power/thermal state. Preserve the current 32 GB campaign budget, 28 GB charged-run ceiling, 26 GB load limit, and heavy.py accounting unless later owner instructions supersede them. Keep benchmark context within 32,768 total tokens including reserved output. Research 128K/256K/1M scalability and report projections separately; do not run those cells without an authorized cap change.

The shipped implementation remains native Zig/Metal. Python, MLX and other engines may serve as development references and comparators. Preserve existing boundaries on system settings, global installations, external writes and paid compute.

Two mandatory starting facts:

- TF commit [69651393](https://github.com/joshuarli/tf/commit/69651393c736d3fa263b6959514fb897e9082207) corrects reasoning replay in both agent workload clients. Earlier sweeps dropped reasoning_content and used an ignored enable_thinking setting. Remeasure both sides on the corrected workload.
- Read [the existing DSpark receipt](https://github.com/joshuarli/tf/blob/main/campaign/kernels/inkling-dspark.md) and parked lane/ds1-dspark-drafter before proposing a new implementation. Its measured draft cost invalidated earlier optimistic estimates.

Audit which ideas are implemented, pending, rejected, superseded, or genuinely untested. Existing f16 KV, prefix reuse/store, batching, controller fusion and TensorOps work are not new research. Revisit a failed experiment only with a concrete changed hypothesis, workload, precision contract or implementation.

Maintain the distinction between state/kernel consistency, fidelity to the reference, and speculation against its chosen target. Under the latest owner steering, fidelity is reporting-only: run the saved-capture comparison at final integration, and do not revive older top-1/KL thresholds as lane or integration gates. Consistency, greedy-equality where required, and applicable task-quality requirements remain in force.

## 2. Expand and maintain the source map

Search GitHub commits, open and merged PRs, review comments, issues, releases, forks, experimental branches, benchmark scripts, raw receipts and negative-result ledgers. Search X/Twitter for author announcements, replies, corrections, linked branches and independent reproductions. Follow every material social claim to primary code or measurements where possible.

Use model aliases, architecture names and mechanism terms: Inkling-2-Mini, inkling2_loop, DSpark, Markov, MTP, NVFP4, MXFP8, relative bias, sink attention, Metal, NAX, TensorOps, speculative verification, expert routing, KV, prefix and agent serving. Search both the last 24–72 hours and the release window. Trace dependencies beyond commits mentioning the model.

For each lead record date checked, exact revision, status, source/license, hardware, model/quantization, actual execution path, workload, metric definition, reproducibility, correctness evidence and applicability to TF. Deduplicate forks. Distinguish author claims, reproduced results, source inspection, hypotheses, and inaccessible evidence. Do not claim exhaustive X coverage if search or post retrieval is incomplete.

### Required starting sources

| Area | Starting points and questions |
|---|---|
| Official model and conversions | [Official model](https://huggingface.co/thinkingmachines/Inkling-2-Mini), [model card](https://thinkingmachines.ai/model-card/inkling-2-mini/), [Unsloth GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF), [DSpark conversion notes](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/blob/main/dspark/README.md), [Xylolabs](https://huggingface.co/Xylolabs/Inkling-2-Mini-GGUF). Verify tensor formats, scales, tokenizer, template, drafter schema and checkpoint identity. |
| llama.cpp upstream | [Current upstream](https://github.com/ggml-org/llama.cpp). Refresh #32825 native Metal NVFP4, #32740 compact-bias contract, #32808 Metal indexed-bias attention, #32752/#32883 chunking, #32855 depth-mix operand order, #32824 CONCAT, #32218 DSpark and #32773 speculation costs. Inspect final diffs and dependencies. |
| llama.cpp supporting work | #32240 separate global/SWA KV types, #32777 loop-aware placement, #30196 output synchronization, #32076 MoE cache, #30685 host-input copies. Audit tokenizer/template/scale/cache fixes #32799, #32763, #32704, #32807, #32828 and #32833. Establish backend and architecture eligibility before proposing flags. |
| llama.cpp forks | [Koosh0610](https://github.com/Koosh0610/llama.cpp-inkling), its [technical write-up](https://koosh0610.github.io/llama.cpp-inkling/), and [realugbun/inkling2-speed](https://github.com/realugbun/llama.cpp/tree/inkling2-speed). Compare their changes with current upstream; isolate what remains novel. |
| MLX ecosystem | [MLX](https://github.com/ml-explore/mlx), [mlx-lm](https://github.com/ml-explore/mlx-lm), [MLX-VLM](https://github.com/Blaizzy/mlx-vlm), [oMLX](https://github.com/jundot/omlx). Refresh model branches and kernel changes, including mlx-lm #2086 and MLX #4712/#4781/#4796. Inspect oMLX #5505, #5508, #5528, #5531, #5559 and #5560. |
| New Mac comparators | [mlx-serve](https://github.com/ddalcu/mlx-serve), especially [drafter global-context fix f501857](https://github.com/ddalcu/mlx-serve/commit/f50185721650f6d8045a550844c70a87f341f49b); [MTPLX](https://github.com/youssofal/MTPLX), [2.13.2 notes](https://mtplx.ai/releases/2.13.2/), and [Nov 23 comparison](https://www.mtplx.com/compare/mlx-serve-vs-omlx-vs-mtplx/). Audit sampler paths, exactness and raw benchmark methodology. |
| Native Metal research | [nanoattention](https://github.com/justrach/nanoattention), [DankAttn](https://github.com/Dankpaws/DankAttn), [erisinky](https://github.com/Eriskii/erisinky), [gall](https://github.com/ltrgoddard/gall), [tur](https://github.com/danylokravchenko/tur), [rustling](https://github.com/onatm/rustling), [rink](https://github.com/adapt/rink), [saetta](https://github.com/antonellof/saetta). Identify actual Mini support, serving capability and numerical tradeoffs. |
| Quantization and profiling | [lookatmyquant](https://github.com/ShinMegamiBoson/lookatmyquant), [deeflect/inkling-m4-metal](https://github.com/deeflect/inkling-m4-metal), [mps-moe-perf-notes](https://github.com/ArthurZucker/mps-moe-perf-notes). Separate weight compression, KV compression, activation precision and changed reasoning policies. |
| CUDA/ROCm/Vulkan prior art | [TokenSpeed](https://github.com/lightseekorg/tokenspeed), [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [kern-inkling-2mini](https://github.com/pegainfer-project/kern-inkling-2mini), [NInfer](https://github.com/iamwavecut/ninfer-all), [ninfer-3090](https://github.com/ashalliants/ninfer-3090), [Strata](https://github.com/Niko1221/Strata), [capy-flash](https://github.com/ValerioDolci/capy-flash). Extract mechanisms, including exact hardware guards; do not transfer their tok/s or percentage gains to Metal. |
| Scheduling, offload and broader discovery | [Resin](https://github.com/ekzhang/resin), [Exo](https://github.com/exo-explore/exo), [TideLM](https://github.com/raspy135/tideLM), [Pulsar](https://github.com/loopai-hq/pulsar), [awesome-inkling](https://github.com/deeflect/awesome-inkling). Follow references to additional projects. Larger-model, other-model and distributed work stays transfer research until Mini applicability is established. |

This is a discovery inventory, not a mandate to install every engine. Prioritize sources with relevant code and reproducible evidence. Verify licenses before porting and preserve required notices.

## 3. Establish comparisons that answer different questions

Keep three result tables:

1. **Controlled engine comparison:** matched checkpoint/tensor values where formats permit, tokenizer, rendered input, reasoning policy, precision settings, KV policy, sampler, speculation mode and workload. Disclose every remaining arithmetic difference.
2. **Best practical configuration:** each engine tuned within the same machine/memory budget and a declared task-quality standard. Different quants or drafting strategies are allowed but identified.
3. **Experimental speed–quality frontier:** explicitly approximate changes, with their quality and memory costs. Never merge these results into the controlled table.

Same file names or “official weights” labels are insufficient. Inspect tensor hashes/values, scale layouts, conversion steps and included vision/drafter weights. Preserve distinct official-format and oQ4e results.

Benchmark current upstream llama.cpp separately from selected PR/fork stacks. Verify GPU placement and eliminate unintended CPU fallback. Refresh oMLX, mlx-serve, MTPLX and nanoattention before treating an older version as competitive. Record exact commands and SHAs.

Use identical token arrays or record and explain tokenizer differences. Fix chat templates, reasoning_effort, hidden reasoning, tool schemas, stop IDs, EOS handling and output budgets. Correctly replay reasoning and tool calls.

Measure cold load separately from first request, unseen prefill, warm repeated prompt, continuation, prefix fork and mixed multi-turn work. Warm up on unrelated inputs when measuring cold prefill. Let engines use appropriate chunk sizes and record those choices.

Start with 1K/4K cells; extend winners to 16K and applicable cells under the total-token cap. Exercise 1/2/4/8 concurrent agents, including staggered arrivals and unequal histories. State client count and resident slot count separately. Include long generations and copy-heavy edits alongside novel code, prose, reasoning and structured/tool outputs.

Report:

- Client and server TTFT; first engine token, first reasoning token and first visible answer token separately.
- Actual evaluated input tokens and reused tokens; prefill latency and throughput.
- Committed output tokens over the full decode interval, per-agent and aggregate rates; distinguish reasoning, visible answer and tool arguments.
- Complete turn/task time, queue delay, latency tails, errors, retries and completed work.
- Charged memory, resident/file-backed weights, scratch, target/draft KV, snapshots and prefix stores.
- Speculative proposals, acceptance, committed tokens per round and the full cost of each round.
- Quality, output/schema correctness, thermal/contention conditions and uncertainty.

Do not count proposed or replayed tokens as useful output. Do not call a configured context capacity a populated-context result. Do not compare a fastest 32-token stretch with a full-answer average. Use both fixed replay workloads for kernel isolation and live agent tasks for practical value.

## 4. Research avenues to cover and triage

### A. Model semantics and loop-aware execution

Audit 24 physical layers versus 48 logical visits, repeated expert traffic, per-pass KV and convolution states, local/global attention, relative-position bias and sinks, depth-residual banks, normalization and loop controllers. Preserve raw DSpark taps and frozen snapshot ownership.

Investigate reuse of immutable normalized states, graph order, depth-mix operand orientation, shared reads and direct writes into the next consumer. Check aliasing and buffer lifetime. Account for prior TF failures before retrying cached normalization. Do not share loop states merely because physical weights are shared.

### B. Expert execution and packed weights

Profile real routed-row distributions and expert fan-out at single-token decode, few-row verification, concurrency and prefill. Investigate expert grouping/worklists, gather/scatter elimination, route sorting, occupancy-dependent dispatch, gate/up/down tiles, shared activation loads, fused activation/reduction and register-level dequantization.

Compare native NVFP4/MXFP8, affine and other permitted layouts. Evaluate scale traffic, vectorization, split-K, tensor operands and precision by shape. Check whether staging weights costs more bandwidth than it saves. Inspect the actual Metal scale-plane contract: native NVFP4 and MXFP8 scale formats are not interchangeable. Treat single-row and multirow results separately.

### C. Attention and positional bias

Investigate compact/indexed relative bias inside attention, sink handling, FlashAttention/FlashDecoding, GQA sharing, split-KV, query tiling, local-window bounds and paged/ring layouts. Measure crossover points by populated context and row count.

Audit masking, absolute positions, clipping/softcaps, summation order and row-width consistency. Compare dense-mask, mask-chunk and query-chunk approaches with both peak memory and latency. Metal scheduling may execute nominally separate chunks concurrently, extending allocation lifetimes. Smaller intermediates and fewer launches are hypotheses, not speed guarantees.

### D. Prefill and intermediate lifetime

Investigate larger/adaptive chunks, aggregate expert routes, tile occupancy, scratch packing by live interval, phase aliasing, direct controller outputs, and backward row-liveness that skips outputs never consumed. Preserve every state needed by future loops, convolution, DSpark, caching and later turns.

Benchmark at actual prefix depth, with cold caches and real token distributions. Distinguish terminal-row-only head work from skipping required intermediate rows. Recheck nanoattention’s corrected methodology and rejected variants before relying on early headline gains.

### E. GPU scheduling and host overhead

Measure CPU encoding, command-buffer submission, synchronization, resource barriers, dependency depth, allocations, first-touch residency, page faults, queue gaps and CPU/GPU overlap. Use paired traces and wall timings; do not subtract unrelated profiling runs.

Evaluate reusable buffers/arguments, pipeline caches, shape buckets, indirect dispatch, concurrent independent work, reduced readbacks and safe command scheduling. Investigate persistent kernels only where dependencies and platform guarantees support them. Check register pressure, occupancy and pipeline-specific threadgroup limits. Removing dispatches alone is not a success criterion.

### F. Few-row verification

Treat verification as a separate workload. Compare decode-style row-exact kernels with batched GEMM/attention for widths around all dispatch thresholds. Preserve the chosen target’s arithmetic where greedy equality is required.

Use fixed real token/route traces as well as live drafts. Price padding, expert fan-out, head computation, graph capture/rebuild, compilation and additional scratch. Evaluate fixed and bucketed widths before adaptive widths; include the cost and memory of keeping multiple graphs.

### G. Native DSpark and alternative proposals

Audit the existing TF branch first. Confirm runtime/converter/drafter compatibility, ctx_mix normalization, Markov tensors, target taps, embedding/head sharing, vocabulary padding and local/global drafter attention. The trained eight-slot block includes an anchor; distinguish verification width from up to seven newly proposed tokens.

Investigate lower-cost drafter precision, efficient head/Markov computation, context updates, full-history attention, shared target features and batched drafts. Benchmark no drafter, loaded-but-paused drafter, and active drafting.

Compare fixed proposal counts 0–7 with adaptive policies based on measured accepted tokens per millisecond, context, concurrency and expert fan-out. Include warmup, draft, verification, rejection, rollback, state-copy and scheduler costs. Use hysteresis/backoff only if they win.

Explore prompt lookup, context copies, suffix/ngram proposals, hybrid copy-plus-DSpark, and carefully bounded trees/beam proposals. Longer copy windows need not obey the trained drafter’s block size, but must satisfy target verification and state semantics. External drafters, EAGLE/DFlash-style methods, self-speculation and distillation are conditional research: establish model support, training needs and feasible resources first.

### H. GPU sampling and exact speculative semantics

Profile GPU-to-CPU logits transfer, vocabulary reduction, softmax, top-k/top-p/min-p, temperature, penalties, logit bias, RNG and grammar work. Investigate GPU-resident selection and certified filtering without unnecessary full sorts.

For sampled speculation, establish the actual proposal and target distributions, acceptance/rejection correction, bonus tokens and history-dependent controls per speculative row. Include earlier proposals when computing that row’s penalties or grammar state.

Separate greedy equality, preservation of a specified target distribution, and seed-identical outputs. These are different claims. Audit fallback behavior for unsupported controls, uncertain thresholds and numerical edges; do not restart already reached sampling decisions in a way that conditions them on later events. Keep a correct fallback and measure when real clients trigger it.

### I. KV, convolution and speculative state

f16 KV is already implemented. Explore global/local policies independently; key/value asymmetry; FP8/int8 and lower-bit methods; scales and outliers; fused dequantization; paging; bounded sliding rings; and state reserved only for history that can be read again.

Treat generic KV-quantization support as unproven for Inkling-specific attention until the actual kernels accept it. Audit per-pass ownership, absolute positions, prefix boundaries, convolution history, rejected tokens and target/drafter rollback together.

Measure prefix sharing/copy-on-write, selective snapshots, immutable shared prefixes and private suffixes. Include draft state and graph buffers in admission. Research 128K/256K scalability analytically under the existing run cap; distinguish projected allocation from measured speed, stability and retrieval quality.

### J. Serving, concurrency and cache policy

Investigate continuous batching, ragged histories, prefill/decode interleaving, decode latency protection, admission, fair scheduling, cancellation and session eviction. Compare synchronized synthetic agents with realistic staggered work.

Measure longest-prefix lookup, cross-session forks, prompt-state checkpoints, cache eviction, disk persistence and target-plus-draft restore atomicity. Optimize useful reuse under the corrected reasoning history. Price cache storage and copying against saved prefill, and ensure all mutable state is restored consistently.

Choose speculation per session or batch when warranted. A single-stream gain may lose under concurrency because verification increases expert fan-out or occupies the GPU longer.

### K. Agent protocol and application costs

Profile rendering/tokenization, reasoning replay, streaming, tool argument parsing, structured decoding, stop handling, response assembly and network flushing. Include long source-file tool arguments and parallel calls.

Look for quadratic rescanning, whole-history copying, repeated tokenization, blocked event loops and unnecessary synchronization. Inspect vLLM’s recent Inkling parser/state work, including #69293/#69297/#51141, #69351/#69347 and #69444/#69181, as mechanism leads. Check final status and applicable model/backend.

Measure completed tool round-trips and task success. Preserve exact arguments, escaping, Unicode, newlines, call IDs and finish reasons. Keep first visible answer latency separate from hidden reasoning generation.

### L. Precision and the quality–speed frontier

Explore selective activation precision, accumulator choices, per-layer/per-operation sensitivity, mixed-bit weights, importance-calibrated quantization, head/embedding compression, drafter quantization and compressed KV. Separate each intervention so its benefit and loss can be understood.

Report task quality, coding checks, tool/schema success, retrieval, refusals, looping/runaways, non-finite values and representative logit/greedy comparisons. Include calibration/holdout separation and relevant long histories within the cap.

Expert pruning/skipping, sparse attention, stale-state reuse, reduced loops, early exits and reduced reasoning budgets change model behavior. Keep them explicitly experimental and separately evaluated. A variable-loop research checkpoint is a different comparator. Do not silently apply fewer loops to the fixed Mini checkpoint or label modified models equivalent.

### M. Hardware limits, offload and other architectures

Build a measured traffic/compute model for actual kernels: physical and repeated weights, KV, activations, scales, head, controllers and temporary buffers. Compare attainable bandwidth and tensor throughput by shape with measured stage time. Use Amdahl-style end-to-end estimates, not multiplication of isolated speedups.

Study CUDA/ROCm/Vulkan implementations for transferable scheduling, worklists, paging and sampling ideas. Inspect architecture guards before assuming a generic optimization executes for Inkling.

Expert caching/offload, CPU/GPU overlap, tensor/expert/context parallelism and multi-device execution are conditional future avenues. Mini fits the present machine, so justify their relevance before introducing complexity or new hardware. Vision preprocessing, encoder/projection caching and multimodal prefixes are similarly conditional on expanding the current text-agent scope.

## 5. Execute in an evidence-driven order

1. Freeze the corrected workload and current source/weight pins. Produce fresh TF and primary comparator baselines, including speculation on/off where supported.
2. Measure the integrated TF build, including already landed post-shootout changes. Profile single-stream attention, multi-stream expert work, prefill experts and actual served latency. Choose priorities from current measurements.
3. Build a compact research ledger: mechanism, source/revision, applicability, existing TF status, expected benefit, implementation cost, correctness risk and next discriminating experiment. Mark each avenue tested, active, rejected with evidence, deferred with reason, or unsupported.
4. Select the highest-value bounded experiments. Parallelize independent source research and coding within the existing delegation policy; serialize GPU timing and keep one resident engine at a time.
5. Follow the current lightweight progression: one gated paired screen; three interleaved pairs for promising candidates; integrated workload and quality checks for retained changes. Use identical-build controls or additional pairs when a concrete noise concern requires them. Reuse existing harnesses and receipts.
6. Keep, revise or revert based on complete-request benefit, memory and applicable correctness/quality requirements. A local kernel win, higher acceptance, lower dispatch count or smaller scratch allocation does not establish an end-to-end win.
7. Refresh the frontier at integration milestones. Continue when new evidence changes the best next experiment. Avoid endless surveying before implementing a strong lead or repeatedly testing an unchanged failed idea.

Retain the owner’s target of at least **1.10× oMLX aggregate throughput in every same-weights 1/2/4/8-agent cell, on both repetitions**, recalculating thresholds from the corrected workload. If TF uses MTP, remeasure oMLX with MTP on and retain the off comparison. Treat previously suggested absolute rates such as 2,000 prefill tok/s or 80 decode tok/s as aspirations unless current hardware evidence supports them. Establish the broader competitive frontier beyond one old oMLX baseline.

## 6. Required deliverables

Produce an updated source/coverage map, reproducible current scorecards, a ranked experiment ledger, integrated improvements with concise receipts, and a remaining-opportunities list.

The final comparison must answer:

- Where does TF lead or trail current llama.cpp upstream and the best relevant experimental stack?
- Where does it stand against current oMLX, mlx-serve, MTPLX and other credible native Metal implementations?
- Which results isolate engine performance, and which reflect different precision, weights, speculation or output behavior?
- What wins on cold turns, warm turns, long generations and 1/2/4/8 agents under the same memory limit?
- Which proposed optimizations failed, why, and what evidence would justify retrying?
- What is the highest-value next experiment, and which claims remain unverified?

Support performance claims with exact revisions, settings and receipts. State inaccessible X posts, missing raw data and untested transfer assumptions explicitly. Do not call TF SOTA or the research exhaustive on the strength of marketing peaks, a short greedy match, theoretical bandwidth, or an incomplete search.

**Begin with the current-state audit and corrected baseline, then run the first high-value experiment. Do not stop at a literature review.**
