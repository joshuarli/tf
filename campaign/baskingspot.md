# Baskingspot reuse assessment

Reviewed on 2026-11-24 against TensorFold
[`1c49c52390ab9bdbeea7519259d148225f06391b`](https://github.com/joshuarli/tf/commit/1c49c52390ab9bdbeea7519259d148225f06391b).
This is a source review, not a compatibility or performance receipt. No dependency was added or benchmark run.
The [roadmap](../docs/roadmap.md) remains the work queue; this page records selection criteria and source pins.

## Priority and integration boundary

The current SC1 receipt identifies a GPU-bound tree round: 35–41 ms, with 1.0–1.3 ms of host-induced GPU idle time.
That evidence keeps V4's verifier work and P6's prefill work ahead of host-library experiments.
See the [reviewed status](https://github.com/joshuarli/tf/blob/1c49c52390ab9bdbeea7519259d148225f06391b/docs/roadmap.md#status).
CPU utility microbenchmarks cannot establish an improvement in committed decode throughput.

TensorFold pins Zig 0.17.1. A package's claim to support 0.17 is a lead, not proof for this exact compiler,
target, libc, optimization mode or linkage. Validate those together before adoption. New runtime dependencies
retain the roadmap's limits: at most 2% stripped binary growth, no decode-path cost, and pinned source/licence.
Keep the existing campaign harness, GPU lock, memory limits and correctness gates.

## Reviewed revisions

| Project | Pinned source | Decision |
| --- | --- | --- |
| ziglint | [4e6aa5f1a741](https://github.com/baskingspot/ziglint/tree/4e6aa5f1a74151a4ecce8098f88bfa6b1cf40988) | Conditional development-tool pilot. |
| testingx | [572fdeeab652](https://github.com/baskingspot/testingx/tree/572fdeeab652857ba0af03ed2f50ce117a0c66d5) | Borrow benchmark discipline; preserve current test runners. |
| aho-corasick | [55ea8818f9d0](https://github.com/baskingspot/aho-corasick/tree/55ea8818f9d0e186e35ee08ac7ba1854aca8f6dc) | Bounded added-token matcher comparison if profiling supports it. |
| utf | [810b70233d97](https://github.com/baskingspot/utf/tree/810b70233d9725929dfffd7b49a100384a652994) | Validation-only candidate if measured relevant. |
| text | [e6ac19bd3a74](https://github.com/baskingspot/text/tree/e6ac19bd3a74dc411d2263d794539bc6604bd331) | Reference for conformance; tokenizer semantics prevent a direct swap. |
| fastmem | [7a722f67bff0](https://github.com/baskingspot/fastmem/tree/7a722f67bff0af8be0dfc2294da3ab6283adadf8) | No adoption for the current Mac campaign. |
| tracy | [95d718fbaa8a](https://github.com/baskingspot/tracy/tree/95d718fbaa8a250e58d7f401ae5da0790470893a) | Optional CPU attribution only when an unresolved question needs it. |

## ziglint: validate the gate before enforcing it

The AST linter supplies useful diagnostics without running `zig fmt` or autofixing code. Its reviewed CI
uses Zig 0.16 and its manifest declares a 0.16.0 minimum. A prebuilt executable successfully parsing some
0.17.1 files does not establish full syntax or standard-library resolution support.
Read its [README](https://github.com/baskingspot/ziglint/blob/4e6aa5f1a74151a4ecce8098f88bfa6b1cf40988/README.md)
and [build](https://github.com/baskingspot/ziglint/blob/4e6aa5f1a74151a4ecce8098f88bfa6b1cf40988/build.zig).

Start with a pinned executable and an explicit, narrow rule set on owned source. Exercise known-good and
known-bad Zig 0.17.1 fixtures, standard-library lookup, relative imports, syntax errors and error exits.
Check that source paths exist and contain files: a missing path or invalid configuration can otherwise report
a diagnostic without a failing exit. Treat setup warnings as failure in a gating wrapper; do not trust exit zero
alone. Named package imports and comptime-created APIs remain outside the linter's full view.

Keep the first real-tree pass advisory, inspect false positives, then enforce the reviewed scope without adding
an unbounded legacy cleanup. Suppress individual justified findings with a reason. Preserve `make lint`'s
compile/link and kernel-contract checks, the existing lean checker and the no-formatter policy documented in
[Zig conformance](../docs/zig-conformance.md). Do not enable the whole upstream house style by default.

An advisory scan of `zig/`, `tools/zig/` and `build.zig` used released ziglint v0.9.0,
[`281f71a6c2ea`](https://github.com/baskingspot/ziglint/tree/281f71a6c2ea3d1e88c35ffe471813da249d6bc0),
with `Z002,Z003,Z007,Z012,Z013,Z015,Z016,Z025,Z026`. It completed in 13.4 seconds with 1,772 findings,
including 914 `Z026` empty catches, and a missing-compiler warning. This is triage evidence, not a verified
0.17.1 gate; many hits are deliberate cleanup or syntactic heuristics. The newer main pin has an additional
rule absent from this release. Missing input and malformed configuration were both observed to exit zero.

One inspected `Z026` finding has a concrete contract: [disk_namespace.zig](../zig/src/sessions/disk_namespace.zig)
promises owner-only prompt-state directories but suppresses `setPermissions` failure in `prepare`.
Add a failing permission-change fixture, check lock/cleanup behavior, and make the success/error contract
explicit. This is an error-handling finding, not evidence of an exploit. Do not turn it into blanket catch removal.

## Added-token matching: the strongest runtime lead

The current [tokenizer trie](../zig/src/core/tokenizer/tokenizer.zig) restarts traversal at each input byte;
`Trie.nextCounted` has no failure links. A runtime Aho-Corasick automaton can avoid repeated prefix traversal.
This is a hypothesis about host work, with no measured speedup on TensorFold yet.

Use [Automaton](https://github.com/baskingspot/aho-corasick/blob/55ea8818f9d0e186e35ee08ac7ba1854aca8f6dc/src/Automaton.zig),
not the comptime `Matcher`: checkpoint vocabularies arrive at runtime. Begin with its compact representation;
the optional dense DFA trades memory for simpler transitions. Preserve these contracts before timing:

| Contract | TensorFold today | Adapter requirement |
| --- | --- | --- |
| Match selection | Leftmost-longest. | Select the matching policy and preserve token IDs. |
| Empty patterns | Ignored; an empty trie is valid. | Handle empty sets outside the automaton, which rejects them. |
| Duplicate spellings | First inserted ID wins. | Deduplicate in the same order; normalized spellings can collide. |
| Pattern lengths | No explicit u16 limit in the trie. | The automaton rejects lengths over 65,535 bytes; preserve a fallback. |
| Token boundaries | Raw/normalized paths, `lstrip`/`rstrip`, request masks. | Preserve byte offsets, strip behavior and masked-span refusals. |

Reuse [tokenizer parity](../tools/zig/tokenizer_parity.py),
[tokenizer tests](../zig/src/core/tokenizer/tokenizer_test.zig) and
[masked fixtures](../zig/src/core/tokenizer/masked_test.zig). Add only missing discriminator cases:
overlapping/shared prefixes, duplicate normalized spellings, empty inputs, long patterns and masked boundaries.
Compare real coding prompts and common-prefix stress cases, construction time, retained allocations and served
TTFT. `Automaton.memoryUsage()` excludes arena construction scratch retained until `deinit`; measure actual
allocator use or process memory too. Retain the current implementation unless the complete tradeoff wins.

## testingx: benchmark mechanics, without a runner migration

[testingx](https://github.com/baskingspot/testingx/blob/572fdeeab652857ba0af03ed2f50ce117a0c66d5/README.md)
offers benchmark loops, result-retention helpers, byte/custom metrics, repeated samples and benchfmt/JSONL output.
Those are useful conventions for a small matcher comparison. Reuse existing harnesses first; add testingx only
as a pinned development dependency if it removes more maintenance than it adds.

Its benchmark helper defaults to ReleaseFast. TensorFold's
[stream-scan benchmark](../zig/build/stream_scan_bench.zig) deliberately defaults to ReleaseSafe, matching the
server module. Build both comparison arms with the same shipping mode and CPU target. Keep setup/reset costs
explicit, consume the computed result, and report repeated measurements. A benchmark-only win is not served TTFT.

The existing [macOS allocator runner](../zig/build/test_allocator.zig) patches the pinned compiler runner to
disable allocation stack traces while preserving test/leak reporting; it has threaded-allocation and planted-leak
checks. Other runners enforce fixture-specific diagnostics. Preserve those contracts and the campaign's rejection
of skipped checkpoint tests. testingx's simple runner does not provide the built-in fuzz protocol; keep the
compiler runner for fuzzing. Allocator metrics observe only the allocator being measured, not GPU memory or RSS.

## utf and text: preserve tokenizer semantics

[utf](https://github.com/baskingspot/utf/blob/810b70233d9725929dfffd7b49a100384a652994/README.md)
has aarch64 NEON validation/transcoding and supports Zig 0.16/0.17. If validation is material in a host profile,
compare validation alone against `std.unicode.utf8ValidateSlice`, including malformed/truncated inputs and tails.
Its APIs consume whole buffers; they do not provide incremental validation. Its base64 decoder always tolerates
ASCII whitespace, so it cannot replace a strict decoder without preserving that caller's rejection behavior.

[text](https://github.com/baskingspot/text/blob/e6ac19bd3a74dc411d2263d794539bc6604bd331/README.md)
is explicitly incubating. More importantly, TensorFold intentionally combines Unicode 16 character categories
with Unicode 9 normalization for HF tokenizer parity: see the
[table generator](../tools/zig/unicode_tables.py) and [normalization fixtures](../zig/src/core/tokenizer/unicode_nfkc_test.zig).
Updating to a newer UCD is a semantic change that can change token IDs. Borrow generation/conformance practices;
do not swap in newer normalization or full case-folding rules as a performance cleanup.

## fastmem and tracy: target-specific limits

[fastmem's dispatch](https://github.com/baskingspot/fastmem/blob/7a722f67bff0af8be0dfc2294da3ab6283adadf8/src/root.zig)
selects its tuned aarch64 kernels only for ELF; macOS gets the generic fallback. The export layer also requires
ELF and LLVM. Its EC2/glibc comparisons do not establish a Mac win. TensorFold already has a
[libSystem memcpy check](../zig/build/metal_server.zig) and a documented compiler-rt linkage failure under 0.17.1.
Investigate that existing issue before considering replacements. Linux reuse can be reconsidered with a relevant
CPU profile and same-target comparison; it does not belong on the active Mac critical path.

[tracy](https://github.com/baskingspot/tracy/blob/95d718fbaa8a250e58d7f401ae5da0790470893a/README.md)
provides CPU zones/plots and an allocator wrapper, but its Zig API does not expose GPU zones. Enabled builds
compile the vendored C++ client. Use it only for an unresolved host attribution question, local-only with deliberate
callstack/allocation settings; confirm disabled builds do not link the client or add hot-path work. Keep Metal
profiling for GPU spans and qualify any optimization using uninstrumented wall time. Existing SC1 evidence gives
no reason to prioritize a new profiler integration over verifier and prefill work.
