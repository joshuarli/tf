# Inkling 2 Mini: native and MLX-VLM

Busy-host comparison on Apple M5 Max, 48 GiB unified memory, using the checkpoint
`Inkling-2-Mini@ed40e7da625f`. Both engines read the same JSON token IDs and
generated 16 greedy tokens without drafts. Each cell has three interleaved pairs
in fresh processes. The pairs ran in MLX, native; native, MLX; MLX, native order
for 512 tokens and the reverse order for 1,024 tokens.

| Prompt IDs | Engine | TTFT seconds | Prefill tokens/s | Decode tokens/s | Peak charged GB | First differing generated token index |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| [512](../prompts/inkling/code-512.json) | native | 2.278 (2.118–3.778) | 224.8 (135.5–241.7) | 40.47 (38.94–40.55) | 20.4 | 9 |
| 512 | MLX-VLM | 0.864 (0.842–7.687) | 592.4 (66.6–608.4) | 35.08 (34.93–35.09) | 25.5 | 9 |
| [1,024](../prompts/inkling/code-1024.json) | native | 4.103 (4.103–5.646) | 249.5 (181.4–249.6) | 39.58 (39.08–39.59) | 20.4 | none in 16 |
| 1,024 | MLX-VLM | 3.776 (1.508–3.932) | 271.2 (260.4–679.1) | 32.70 (32.53–33.10) | 24.9 | none in 16 |

Timing columns show median and range of three runs. Peak charged memory is the
largest `heavy.py` peak across the three. Token indices are zero-based. Each
engine produced the same token sequence in its three runs. The 512-token
outputs diverge at index 9: native produced 175114 and MLX-VLM produced 196956.
The 1,024-token outputs match for all 16 tokens. These checks do not establish
model fidelity beyond these two prompts and 16-token continuations.

The timing comparison is inconclusive as a general speed claim. The host was
busy, prefill varied widely, the 512-token continuations differ, and the
engines use different execution paths. The 512-token first MLX run includes a
much slower prefill than the following two. No limiter was profiled.

## Measurement

Native uses chunked prefill of 128 tokens, then 15 sequential decode steps.
TTFT starts just before prefill and ends after host argmax of the first token.
Decode rate is 15 divided by elapsed time through the final argmax.
[The native runner](../comparators/mlx-vlm/native.zig) loads the model before
timing and uses the same family calls as `tf-inkling2-loop-run`.

[The MLX runner](../comparators/mlx-vlm/compare.py) uses the pinned
`mlx_vlm.generate.ar.generate_step`, chunk size 128 and temperature zero.
It times from the first generator advance through the first yielded token,
then the next 15 yields. The generator synchronizes the first token with
`mx.eval` and calls `.item()` for every yielded token. It also schedules the
next decode before yielding each token, so these boundaries include pipelining
and are not identical to the native synchronous boundaries. Both timings omit
model load and tokenizer work. Neither measures served request latency.

Native source revision was `cacfa9403e23c6badb3172b275a40c92ec1c97c4` and the
measured native binary SHA-256 was
`148fe3f9c09d7fe7ef4b9f29f7c9ca80e71c80f50cb6d2f8e5ab1d0820f95878`.
The native binary was built with Zig 0.17.1, `-Dcpu=apple_m1`, and the
repository's `-Ofast` tool configuration. MLX-VLM source was
`f1618d6d52be4ef25d3fe65270f7528a59c55df3`, MLX was 0.32.4, MLX-VLM 0.8.0,
and Transformers 5.24.0. The environment is locked in
[uv.lock](../comparators/mlx-vlm/uv.lock).

Every accepted run used `heavy.py` with its 28 GB cap and ran between
`gpu-gate-ready -1` checks under the same lock. GPU and utilization limits were
35%; the memory pressure, swap, and compressor limits were unchanged. A
`perf-gate-ready` check preceded each timed run and `perf-gate-ready -1`
followed it. The GPU postcheck ignored the wrapper shell PID. Host pressure was
level 1 before each heavy command. Some accepted commands observed pressure
level 2 during the run but passed the postcheck. The raw receipt includes
successful runs and all rejected attempts, including the first mistakenly
selected native build helper. Failed gate checks were excluded.

[Raw run records](../runs/inkling-mlxvlm-pairs.jsonl) contain the exact commands,
exit codes, output IDs and memory receipts. The first pair used the same command
as the remaining pairs, invoked directly. Subsequent pairs used
[run_pairs.py](../comparators/mlx-vlm/run_pairs.py). Rebuild the native runner
outside a measurement window with `PATH=/opt/homebrew/bin:$PATH
TMPDIR=/Users/josh/repos/tensorfold-m5/.work/tmp
ZIG_GLOBAL_CACHE_DIR=/Users/josh/repos/tensorfold-m5/.work/zig-global-cache
python3 campaign/comparators/mlx-vlm/build_native.py`. The regenerated binary
hash differed from the measured binary; the measured hash above identifies the
actual artifact used for all six native runs.
