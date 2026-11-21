
## Mission

Execute this campaign, including implementation, measurement, commits, and a working deliverable. Do not stop after writing a plan.

I own an Apple M5 machine. Discover its exact chip variant, GPU configuration, memory, macOS version, and development tools. Do not assume M5 Max, M5 Ultra, or any particular RAM capacity.

Build a fast, lightweight, native Zig/Metal inference engine by extending a working open-source implementation. TensorFold is the default base. Start with Flint to establish the first qualified implementation and performance improvements; then implement and optimize Inkling 2 Mini, the original motivation for this project. Incorporate useful ideas from Exo and other primary implementations where measurements justify them.

The objectives are correct inference, lower single-user latency, higher sustained generation throughput, and efficient memory use on this exact Mac. Keep the shipping execution path native. Python, MLX, and other frameworks are acceptable development oracles and benchmark competitors. Small existing C/Objective-C interoperability layers are acceptable. Preserve useful upstream functionality and avoid an unrelated language rewrite.

This is a single-Mac campaign. Distributed inference, CUDA optimization, new model training, a new UI, and compatibility campaigns for older Macs are outside the initial scope. Preserve existing fallbacks where practical.

## 1. Work autonomously and preserve progress

Inspect the current directory, git status, remotes, AGENTS.md, contributor guidance, and applicable local skills before editing. Preserve unrelated changes. Use an isolated worktree when necessary.

If no project checkout is supplied, create a local TensorFold fork in ./tensorfold-m5, retaining the original repository as upstream. Use a campaign branch. If a writable user-owned fork is already configured, commit and push qualified changes there. Otherwise retain local commits and continue; remote setup must not block implementation. Do not push to the source project's upstream.

You may make routine architectural decisions, refactor relevant code, add focused tooling, obtain public dependencies and model weights that fit the resource budget, run local experiments, and commit without asking for approval at every step. Keep attribution and third-party notices. Use existing credentials only for their intended services.

Do not open upstream PRs, contact maintainers, publish releases, rent compute, buy services, or change system-wide security or memory settings. Those are not required for this campaign. Missing optional credentials or profiling permissions should remove that optional dependency, not stop all work.

Resolve ordinary ambiguity from source and measurements. For a genuine blocker, finish independent useful work, record the exact obstacle and next command, and report it. Do not claim access to a Mac, a successful benchmark, or background continuation that the environment does not actually provide.

Maintain concise, durable campaign state in the repository. At every milestone, summarize measured changes, remaining uncertainty, and the next experiment. Before a session or context ends, checkpoint commits, experiment status, artifact locations, and exact resumption commands.

## 2. Verify the source landscape once, then freeze a baseline

Read current primary sources and inspect actual code. Treat earlier chat descriptions and social-media speed claims as leads, not ground truth.

Start with:

- [TensorFold](https://github.com/ashhart/TensorFold), including its current build instructions, contributor guidance, supported models, and license notices.
- [TensorFold Flint recipe](https://github.com/ashhart/TensorFold/blob/main/docs/recipes/flint.md) and [Flint implementation PR #1177](https://github.com/ashhart/TensorFold/pull/1177).
- [Exo](https://github.com/exo-explore/exo) and [Flint fusion PR #2533](https://github.com/exo-explore/exo/pull/2533). Inspect current merge status and the exact relevant head.
- [Exo PR #2530](https://github.com/exo-explore/exo/pull/2530), if accessible, and current Inkling support. Verify which model each optimization actually targets.
- [Recursive rec](https://github.com/recursive-oss/rec), including any accessible reference implementation and supported hardware requirements.
- [Rach's announcement](https://x.com/rachpradhan/status/2123991407484658012) and the nanoattention integration mentioned in [Codegraff's changelog](https://github.com/justrach/codegraff/blob/main/CHANGELOG.md). Follow authoritative links to determine whether the actual engine now has public, licensed source.
- [Inkling 2 Mini release](https://thinkingmachines.ai/news/inkling-2-mini/), its linked model repository and reference code, and [vLLM's implementation notes](https://vllm.ai/blog/2026-11-20-inkling2-mini).
- [Apple's M5 GPU programming talk](https://developer.apple.com/videos/play/tech-talks/111432/), [Metal Performance Primitives guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf), and relevant installed SDK documentation.

Time-box the initial repository search to about 30 minutes. An inaccessible URL does not prove that a repository is absent. Do not stall waiting for Rach's implementation. Change the default base only if a licensed, buildable alternative demonstrably saves work or improves the technical starting point.

Important leads to recheck:

- Exo's Flint fusion PR has explicit M5 exclusions for several optimized paths pending hardware validation. Determine the reason for each restriction; simply removing the checks is not validation.
- TensorFold's Flint recipe distinguishes an external drafter from copy drafting. Its published no-external-head benchmark is not necessarily a no-speculation benchmark.
- Existing upstream checks do not automatically establish cross-engine numerical parity or downstream task quality.

Record full source SHAs, PR heads, compiler versions, dependency locks, model revisions, tokenizer/config hashes, quantization metadata, and source licenses. Preserve separate frozen baseline and candidate builds. Revisit upstream at milestone boundaries, not after every experiment. Record improvements imported from upstream separately from changes developed here.

## 3. Discover the machine and establish resource limits

Before downloading large weights, inventory the actual host and Metal device. Record CPU/GPU identity, RAM, available disk, macOS build, Xcode/SDK, Zig, Metal language support, device capabilities, and power/thermal conditions where observable.

Check capabilities through supported APIs and compile/run probes. Do not infer every feature from the marketing name. Distinguish GPU tensor acceleration from the separate Apple Neural Engine.

Estimate packed weight residency, loader/conversion peaks, scratch buffers, KV and other recurrent state, optional draft state, and representative context lengths. Respect the GPU's recommended working set and leave practical headroom for macOS and existing applications. Do not blindly sum overlapping unified-memory counters.

Use cached weights where possible. Avoid duplicate multi-gigabyte downloads and retain sufficient disk headroom; never delete unrelated user files or global caches to make an experiment fit. Set explicit server/cache limits rather than inheriting large defaults.

If Flint or Inkling cannot fit safely, use small fixtures for kernel development and harness validation, clearly mark full-model measurements blocked, and finish all feasible work. A smaller model is not a substitute result for either target.

Run GPU benchmarks exclusively, with a shared lock across all agents. Keep compilation and other heavy jobs outside timed windows. Stop only campaign-owned processes. Respond to thermal or memory pressure by reducing workload or recording an invalid trial, not by altering the user's system settings.

## 4. Define success before tuning

Freeze the feasible primary matrix after hardware discovery. The default is batch-one, one active stream, with actual prompt lengths of 1,024, 4,096, and 32,768 tokens and 512 emitted output tokens. Use several fixed prompts covering code, prose/reasoning, and structured output. Record omitted cells and why.

Context means actual tokenized input length, not the configured maximum capacity. Include natural, varied material; repeated filler must not dominate a suite that benefits from copy drafting. Keep a separate held-out prompt set for final confirmation.

Treat 128K context and two/four concurrent streams as secondary investigations when memory permits. Report aggregate serving throughput separately from single-user speed.

Maintain two comparison tables:

1. Matched implementations: identical target checkpoint, quantized values and scales, tokenizer, model semantics, activation/KV policy, prompt tokens, and decoding mode. Lossless repacking is allowed.
2. Best useful configurations: each engine's strongest qualified configuration, with precision, speculation, memory, and quality differences visible. This table is not evidence of a pure kernel speedup.

Include frozen upstream TensorFold, the candidate, Exo, and the strongest other runnable model-compatible comparator. Include rec and Rach's engine when actual support and availability permit. A competitor unavailable on this Mac is unavailable, not defeated.

Targets:

| Milestone | Target |
| --- | --- |
| Qualified foundation | Working native implementation, trustworthy correctness evidence, reproducible baseline |
| Flint runtime improvement | At least 20% geometric-mean no-speculation decode improvement over frozen upstream TensorFold |
| Competitive configuration | At least 10% geometric-mean improvement over the strongest reproducible comparator on the same primary suite |
| Stretch | At least 30% runtime improvement, or a substantial measured memory reduction without sacrificing speed or quality |
| Inkling delivery | Qualified native text inference, then the same controlled optimization/comparison process |

These are goals, not predicted gains or permission to move the baseline. If upstream lacks Inkling support, freeze the first independently qualified native port as its baseline and label it accordingly. Do not describe that comparison as a win over upstream Inkling support.

For an ordinary performance candidate, default promotion gates are at least 3% geometric-mean gain, a paired 95% confidence interval above 1.0, no primary-cell decode or prefill regression exceeding 5%, and peak-memory growth no greater than max(2%, 128 MiB). A useful memory optimization may instead qualify through at least 5% lower peak memory with no material latency regression.

Allow workload-specific dispatch to retain the faster implementation in each supported regime. Validate the final dispatching product across all primary cells. Retain improvements below 3% only with a clear cumulative or simplification benefit; do not promote measurement noise as an isolated speed win.

Speculation has a separate memory envelope: start with at most min(4 GiB, 10% of physical RAM) of incremental draft-related residency, while preserving total working-set headroom. Record all overhead and retain the no-drafter mode.

Fix quality/numerical acceptance rules before optimization. Investigate failures rather than weakening those rules to obtain a result.

## 5. Build the smallest useful benchmark system

Reuse upstream tools. Add missing measurements and adapters rather than inventing a benchmark framework or workflow language.

Expose repeatable entry points for building, smoke tests, kernel correctness, model correctness, benchmarking, comparison, reporting, and resume. Derive actual commands from the checked-out tree; do not copy undocumented flags from this prompt.

Make acceleration modes explicit:

- True autoregressive generation, with every draft and speculative mechanism disabled.
- Copy-draft speculation only.
- External learned drafter only, where supported.
- Combined speculation, where supported.

If upstream cannot actually disable its default accelerator, add a verified switch or instrumentation before calling any result autoregressive.

Use the same explicit benchmark-only EOS/stop policy across engines for fixed-length measurements. Report naturally terminated requests separately; never pad outputs or compare early completions as if they emitted 512 tokens.

Record process startup and model loading separately; separate cold prompt processing, warm model requests, and prefix-cache hits. Disable prefix reuse for cold-prefill comparisons. Report cached and uncached token counts.

Measure synchronized wall time, not just GPU enqueue duration. Define the decode interval consistently: after prefill completes and before generation/sampling begins, through completion of the final committed output. Document how the first token is handled. Count only emitted target tokens. Include drafting, verification, rejected proposals, rollback, sampling, and host overhead in accelerated timings.

Report TTFT, prefill duration/rate, generation duration/rate, total request latency, output token counts, loading/steady-state peak footprint, GPU allocations, and available pressure indicators. Report draft acceptance and committed tokens per verification separately. Useful code-task latency should include all generated reasoning tokens.

Use short A/B screens for development. For shortlisted candidates, use at least six balanced or randomized baseline/candidate process pairs across two separate run sessions. Warm both fairly. Estimate uncertainty from independent paired runs and prompts, not correlated individual token timings. Expand to twelve pairs only to resolve a concrete borderline result; otherwise mark it inconclusive.

After tuning, repeat the chosen configuration independently and on held-out prompts. Disable GPU captures and validation instrumentation during timed measurements. Retain all raw trials, including failures and predeclared environmental exclusions. Never discard a slow trial merely because it weakens the result.

Keep hashes of binaries, weights, prompts, configurations, raw outputs, and measurement artifacts. A report must be regenerable from retained data.

## 6. Establish correctness before optimizing

Use three levels of evidence:

1. Independent small reference implementations for changed operators.
2. Teacher-forced intermediate activations and logits against a qualified model reference on identical token histories.
3. Held-out generation, coding/task checks, and serving behavior appropriate to actual use.

Do not make TensorFold's existing output the sole oracle. Qualify it against the official/reference implementation or another independently checked implementation. Diagnose tokenizer, quantization, masks, precision, and model configuration mismatches before blaming floating-point noise.

For numerical differences, record absolute/relative error where meaningful, top-token agreement, logit margins, and likelihood/KL measurements on common histories. Trace the first materially divergent tensor. Do not compare only two free-running texts after their histories have diverged.

Require exact equivalence when the transformation promises unchanged arithmetic. For legitimate floating-point reordering, document justified tolerances and evaluate near-tie behavior and held-out quality. Do not demand universal bitwise identity across unrelated implementations, or assume every differing token is a harmless tie.

Require zero new failures in deterministic operator, state-management, and protocol regression fixtures. For changes that alter numerical results, predeclare likelihood and task-quality acceptance criteria; passing a few fluent generations is insufficient.

Create a compact held-out quality suite using existing public/local evaluations where practical: executable code tasks, structured output/tool calls, reasoning, and context retrieval. Start with roughly 32–64 meaningful cases, run the relevant subset during development, and reserve the full suite for model bring-up and finalists. If evidence is too weak to support a quality claim, report that limitation and keep the altered mode experimental.

Target edge cases relevant to the changed code: real model shapes and dispatch boundaries, alignment/tails, quantization groups and scales, reduction precision, tied router scores, window boundaries, masks, position handling, and exceptional values under the operator's contract.

Verify full versus chunked prefill, cached continuation versus fresh requests, one-step versus block verification, state reuse, EOS, cancellation, context limits, and reset after errors. Preserve reasoning templates, special tokens, tool-call syntax, and supported constrained decoding.

For Flint, derive attention, normalization, routing, shared-expert, and positional rules from the pinned implementation. A superficially similar MoE kernel is not automatically interchangeable.

For Inkling, audit the actual looped graph. Preserve fixed pass count, message inputs, per-pass KV/convolution state, and required precision. Shared weights do not imply shared recurrent state. Later-pass Q/K/V inputs may differ, so stacking their projections as if the inputs were identical is invalid. Preserve the reference's pass initialization, merge arithmetic, and draft hidden-state tap locations. A variable-loop research checkpoint is a separate model configuration.

Focus tests on real failure modes. Avoid repository-wide mutation, coverage, or formal-proof campaigns unrelated to the changed inference path.

## 7. Make speculative decoding correct

Get the no-speculation target qualified first. Greedy-only speculation is an acceptable first milestone if it is explicitly advertised as such.

For greedy generation, compare speculative output to the qualified target on the same histories. Test zero, partial, and full draft acceptance; EOS/output limits inside blocks; rejected-block rollback; position counters; all KV/recurrent state; and resumption after fallback.

For sampled generation, use an algorithm with a valid target-distribution argument, such as the methods described in [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html), or the model's correctly implemented native method. Verify the exact acceptance/rejection correction, support assumptions, and interaction with logits processors.

Test small enumerable distributions, zero-probability cases, top-k/top-p, temperature, repetition penalties, and grammar/tool constraints. Statistical tests support the argument; they do not replace it. Identical random seeds do not require identical strings across algorithms consuming different random draws.

Check block-verification masks and numerical behavior against stepwise target execution. Verify complete state rollback, including any convolution or loop-specific state. Only advertise modes for which the implemented verifier and sampler satisfy the stated contract.

For Flint, inspect the current DFlash2 implementation and any copy-draft path. For Inkling, inspect the model-native draft mechanism and its compatible checkpoint. Do not assume that one model's drafter can accelerate another.

Tune draft length and enablement from total measured cost, not acceptance rate alone. Retain automatic fallback when speculation loses, has insufficient memory, encounters unsupported settings, or fails. Include fallback costs in measurements.

## 8. Profile and choose experiments from evidence

Before each major wave, capture representative short- and long-context execution. Attribute time to expert/projection work, attention, routing, normalization/residuals, quantization, dispatch, synchronization, sampling, and memory movement.

Measure the actual model shapes and batch regimes. Estimate relevant weight/cache traffic and achievable bandwidth rather than deriving speed from total parameter count alone. Track host overhead independently. A Zig implementation is not inherently faster merely because a competing implementation uses Python to schedule GPU work.

Maintain a ranked list of hypotheses with expected benefit, correctness risk, implementation cost, and the smallest decisive experiment. Prioritize the largest measured costs.

Study these candidates where applicable:

### A. Exo-derived local fusion

Map Exo PR #2533 and current Inkling optimizations onto the native engine's existing graph. Identify what TensorFold already does before porting anything.

Candidate operations include norms, residual-plus-norm, Q/K normalization with positional work, expert combine, stacked projections, and activation/down-projection fusion. Port the mathematical operation plus its dtype, layout, reduction, and dispatch constraints. MLX kernel wrappers are not a portable runtime interface.

For M5 exclusions, determine whether they arise from missing validation, different numeric behavior, changed kernel selection, or unsupported features. Validate on actual hardware and enable only the supported cases.

### B. M5-specific matrix and memory paths

Compare existing shader/GEMV kernels with supported TensorOps/MPP paths using real projection, expert, prefill, and verification shapes. Matrix acceleration may help larger work more than skinny batch-one decode; measure both.

Investigate direct quantized execution, useful packing layouts, shared-weight reuse, reusable workspaces, fewer temporary tensors, and elimination of unnecessary host/device synchronization. Preserve quantization semantics. Report repacking time, on-disk expansion, load peaks, and resident expansion.

Treat lossy requantization and KV compression as separate configurations requiring quality evaluation, not ordinary implementation speedups.

### C. Attention and state management

Optimize actual sliding/full attention regimes and supported cache layouts. Bound temporary memory, avoid redundant cache copies, and retain masks, sinks, offsets, and state semantics.

If changing command submission or adopting Metal 4 queues, audit dependencies, buffer lifetimes, aliasing, and synchronization using [Apple's resource-synchronization documentation](https://developer.apple.com/documentation/metal/resource-synchronization).

### D. Larger fusion and persistent execution

Attempt broader fusion only if profiling identifies significant dispatch or intermediate-memory cost. Measure occupancy, register pressure, code size, compilation time, and sustained behavior.

Document synchronization scope explicitly. A threadgroup barrier does not provide arbitrary grid-wide synchronization. Do not introduce a persistent-kernel deadlock or depend on unspecified scheduling to eliminate launches. Keep a simple fallback and reject complexity without an end-to-end benefit.

### E. Model-specific speculation

After target correctness, optimize verification batching, draft integration, acceptance-dependent policy, and state restoration. Measure these separately from the underlying runtime gains.

## 9. Sequence the model work

First build the pinned upstream engine and run a small smoke model if needed. Bring up Flint's qualified native path on this M5, freeze the baseline, profile it, and measure the three most promising improvement hypotheses. Do not spend the entire campaign polishing benchmark infrastructure.

If Flint is unavailable or cannot fit but Inkling is feasible, proceed directly to Inkling. Flint is the preferred bootstrap, not a dependency of all later work.

Then begin Inkling bring-up; reaching the Flint percentage target is not a prerequisite. Parallel source analysis for Inkling can happen earlier without competing for GPU measurements.

If current upstream already supports Inkling, reuse and qualify it. Otherwise add the narrowest correct family implementation using the official model definition and existing reusable primitives. Keep mixed weight/activation formats and all model state explicit. Initially target text inference and coding-agent use; clearly state if vision input is outside the delivered scope.

Establish a working reference-compatible port before aggressive fusion or native drafting. If full-model validation is blocked, finish independent operator/loader work and record the precise gap. Do not present unverified model support as complete.

Reuse shared improvements across both models only where their contracts match. Preserve useful upstream architecture instead of proliferating special cases.

## 10. Run a bounded, resumable optimization loop

For each experiment:

1. Record the measured bottleneck, hypothesis, baseline commit, proposed change, expected benefit, and correctness risk.
2. Implement the smallest useful change with a fallback when appropriate.
3. Run focused correctness checks.
4. Run a short paired performance screen.
5. For credible winners, run the full relevant promotion matrix and held-out checks.
6. Commit qualified improvements; revert failed implementations while preserving the measurements and explanation.
7. Reprofile and choose the next experiment.

After three failed variants of the same idea, reassess the hypothesis before trying another. Do not brute-force endless compiler flags or hard-code benchmark prompts.

Use parallel agents for independent code investigation, operator work, correctness review, and benchmark analysis. Give them bounded ownership and integrate through coherent commits. Serialize GPU work and keep a single owner of the benchmark manifest.

Keep one compact campaign directory containing the plan/status, source-and-model lock manifest, experiment ledger, benchmark configuration, and generated scorecard. Store large weights and traces outside git; retain their hashes and retrieval paths. Avoid dozens of overlapping planning documents.

Resume by checking repository state, reading the latest checkpoint, verifying required artifacts and hardware identity, then taking the next unfinished high-value task. Do not repeat completed research or rebuild unchanged evidence without a reason.

Continue until the deliverables and primary milestones are met, a genuine blocker prevents further useful work, or measurements show a plateau. Stretch goals are optional, and unavailable comparators cannot block completion. A plateau normally requires re-profiling, investigation of the largest remaining cost categories, and roughly eight substantive unsuccessful candidates across more than one hypothesis family. Conclude sooner when a measured hard resource bound or exhausted credible hypotheses already establishes the limit. Record what was tried and why further work has low expected value.

An unmet target is a valid reported outcome. Never fabricate a win, change quality standards, silently shrink the workload, or run indefinitely just to obtain a favorable number.

## 11. Deliver a usable result

The campaign must leave:

- A buildable native engine with tested model support, qualified defaults, clear opt-in experimental modes, and a fallback path.
- Focused correctness tests and a reproducible benchmark/comparison harness.
- Exact source/model/toolchain provenance and preserved license notices.
- Raw measurement artifacts, an automatically regenerated scorecard, and an experiment ledger including negative results.
- Documented commands to build, obtain compatible weights, run locally, benchmark, verify, and resume.
- Clean commits and a concise final report explaining what changed, what improved, what regressed, what remains unsupported, and which targets were met.

The final report must name the actual Mac, model and quantization, context/output lengths, mode, raw rates, uncertainty, memory peaks, and the baseline commits. Separate local reproduction from external claims. Report task-quality evidence and its limits.

Do not claim fastest-on-Mac or a victory over Rach from incomparable chips, weights, or workloads. Make the narrow claim the measurements support.

**Start now: inspect the workspace and hardware, freeze the source baseline and resource budget, build the engine, run the first qualified inference and baseline measurement, and continue into the highest-value implementation experiment without waiting for another prompt.**