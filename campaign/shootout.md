# Inkling 2 Mini on the M5 Max: TensorFold against oMLX and nanoattention

**Status: final, 2026-11-23.** TensorFold is main at `6285f2f56`, which includes every lane through H6 (f16 KV).

**Superseded workload.** These sweeps used the agent clients from before `69651393c`, which dropped `reasoning_content` between turns. The GO decision stands, but roadmap Decision 16 requires remeasuring both engines on the corrected clients. Later numbers do not pair with these. Current next steps are in roadmap **Status**.

## Verdict

**Keep building TensorFold.** TensorFold and oMLX serve the same oQ4e weights on the same multi-turn agent workload. On the warm repetition, TensorFold reaches 0.91–1.08 of oMLX's aggregate throughput at 1, 2, 4 and 8 agents. Its accuracy is the same. At 8 agents it serves eight true concurrent sessions in 23.9 GB, while oMLX needed to drop to four slots to stay under the cap. The decision rule asked for within about 15%, and every cell meets it.

## Question

Is TensorFold worth continuing to build for concurrent coding agents on an M5 Max, compared with oMLX 0.8.3? The
comparison must use the same weights and the same workload.

**Decision rule.** Keep building TensorFold when it comes within about 15% of oMLX's aggregate throughput on the
same weights and workload, with equal or better accuracy and memory.

## Setup

- **Weights.** Both engines serve oMLX's `Inkling-2-Mini-oQ4e-mtp@3eea042751cd` checkpoint, with MTP off. It
  stores the experts as NVFP4 in layers 0–19 and affine 4-bit in layers 20–23, and attention, embed and head as
  affine 8-bit.
- **Workload.** Each agent runs `campaign/fx/sweep-subagents.sh`: four turns over a 2.5K-token, 115-function
  system prompt. Decoding is greedy, with a 256-token output cap. Each cell has two repetitions in run order. Repetition 1 starts cold and repetition 2 reuses the server's prompt cache. Every run has gates before and after.
- **Host.** M5 Max with 48 GiB. Runs use `heavy.py` with its 28 GB charge cap and `INKLING_LOAD_LIMIT_GB=26`.
  oMLX runs with `--memory-guard-gb 26` and four slots.
- **Same work.** Both engines produce the same number of output tokens per cell: at 1 agent, 369 for TensorFold and about 371 for oMLX; at 4 agents, 1,491 and about 1,466.

## Served agents, same weights

Aggregate output tokens per second, repetitions 1 / 2. oMLX numbers are from
[inkling-omlx-concurrency.md](kernels/inkling-omlx-concurrency.md). TensorFold's 1- and 2-agent cells ran on main at `6285f2f56`. The 4- and 8-agent cells come from the H6 lane on identical source ([inkling-f16-kv.md](kernels/inkling-f16-kv.md)).

| Agents | TensorFold | oMLX 0.8.3 | Ratio, rep 1 | Ratio, rep 2 | TensorFold peak charged |
|---:|---:|---:|---:|---:|---:|
| 1 | 26.4 / 43.3 | 26.0 / 45.2 | 1.01 | 0.96 | 19.6 GB |
| 2 | 41.8 / 64.3 | 54.5 / 59.4 | 0.77 | 1.08 | 20.9 GB |
| 4 | 57.4 / 70.3 | 72.7 / 77.4 | 0.79 | 0.91 | 21.9 GB |
| 8 | 62.1 / 66.3 | 70.1 / 70.0 (four slots) | 0.89 | 0.95 | 23.9 GB |

- **Repetition 1 is cold-start prefill.** TensorFold prefills at about 620 tok/s and oMLX at about 930. At 2 agents, the second agent's first turn waited 5.6 s in the queue behind the first agent's prefill.
- **Correctness under concurrency.** At 1, 2, 4 and 8 streams, every concurrent reply equalled the same request run alone, for both code and chat.
- **oMLX at 8 agents.** With its default eight slots, oMLX went past the 28 GB cap and was stopped. Its 8-agent figure is eight clients sharing four slots.

The progression at 4 agents, on repetition 2, was:

| Lanes landed | Rep 2 tok/s |
|---|---:|
| H5 (fast oQ4e kernels) | 39.7 |
| P1 (prefix store) | 47.6 |
| B1g (batched attention) | 57.2 |
| B1h (fused batched decode) | 60.3 |
| P2 (prefill on TensorOps) | 67.6 |
| H6 (f16 KV) | 70.3 |

## Time to first token

Client-side TTFT is not comparable between the two engines on this workload. The Inkling chat template has no `enable_thinking` switch, so the model always reasons; the client's `enable_thinking: false` only changes what the server shows. TensorFold withholds reasoning text and streams only the answer. Both engines generate the same tokens, but oMLX's 0.23–0.84 s client TTFT implies it streams the reasoning.

A direct probe on a cached 2K prompt measured TensorFold's server at 0.043 s to first token. Its first streamed reasoning chunk arrived at 0.085 s. With thinking marked off, the first answer chunk arrived at 1.13 s, after the hidden reasoning ended. On the agent workload, TensorFold's server-side TTFT for cached turns is 0.04–0.5 s at every agent count.

## Raw engine rates on oQ4e

| Metric | TensorFold | oMLX (estimated from served logs) |
|---|---:|---:|
| Prefill, 1K and 4K | about 576–627 tok/s | about 930 tok/s |
| Served decode, one stream | 46.9 tok/s | about 55 tok/s |
| Batched step, 1 row | 21.6 ms | — |
| Batched step, 4 rows, 2.5K cache | 44.7 ms | — |

## nanoattention

On the native checkpoint (`ed40e7da625f`), engine-only and unserved, nanoattention prefilled about 1.5× faster
than TensorFold before H3 to H5. Its decode was about 3% faster. See
[inkling-nanoattention.md](kernels/inkling-nanoattention.md); its 16K cells and oracle comparison were not
completed. nanoattention has no concurrent server, so it does not enter the agent comparison. TensorFold's later prefill and decode lanes have since passed both of those figures on oQ4e.

## Accuracy

These figures compare against saved f32 oracle captures: 16 histories and 3,015 scored positions.

| Weights and engine | Top-1 agreement | Mean KL |
|---|---:|---:|
| oQ4e, TensorFold first-cut kernels (H3) | 91.64% | 0.0394 |
| oQ4e, TensorFold fast kernels (H5) | 92.04% | 0.0393 |
| oQ4e, TensorFold final, f16 KV and half prefill (H6) | 91.64% | 0.0389 |

The first 45 greedy IDs on the 1K code prompt are the same for TensorFold and oMLX. The roughly 8% top-1 loss is
the cost of oMLX's 4-bit weights, not of either engine. f16 KV and half-precision prefill cost nothing measurable
on top of it.

## Memory at long context

Per session, TensorFold's KV cache stores 8 global layers' keys and values per token. Local layers keep a
1,024-row ring each. The figures below are cache allocations only.

| KV precision | Global bytes per token | Fixed local rings per session | 128K session | 256K session |
|---|---:|---:|---:|---:|
| f32 (before H6) | 64 KiB | 335 MB | 8.9 GB | 17.5 GB |
| f16 (main) | 32 KiB | 168 MB | 4.5 GB | 8.8 GB |
| 8-bit (not built) | 16 KiB | 84 MB | 2.2 GB | 4.4 GB |

The oQ4e weights hold about 16 GB, which leaves about 10 GB of KV under the 26 GB load limit. With f16 on main, two 128K sessions fit, or one 256K session with little margin. 8-bit KV would allow a 256K session with room for a second. No 128K or 256K run was made under the owner's 32 GB cap, so these are allocation projections, not measurements.

## Next steps, ranked

1. **Cold prefill.** It is the largest remaining gap: 620 against about 930 tok/s, which shows up as the 0.77–0.79 repetition 1 ratios at 2 and 4 agents. The next candidates are NVFP4 expert tiles staged directly for TensorOps, and chunked prefill interleaved with decode for new agents joining a running batch.
2. **8-bit KV.** It makes 256K practical with a second session alongside. f16 KV landed with zero fidelity cost, and the same store-once, read-everywhere rounding applies.
3. **Thinking-off requests.** The model still spends about 50 hidden reasoning tokens per turn. Rendering an empty thinking block when the client asks for no thinking would cut that work on both engines. Check that the model honours it before relying on it.
4. **Long-context measurement.** Run a 32K–64K agent cell under the cap to confirm the allocation projections and attention cost per token.
