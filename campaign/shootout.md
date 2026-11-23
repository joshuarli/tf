# Inkling 2 Mini on the M5 Max: TensorFold against oMLX and nanoattention

**Status: draft.** The served comparison below is the state after the prefix store (P1). The decode-speed (H4b),
admission-interleaving (B1f) and prefill lanes are still open. The final table replaces these numbers after they
land.

## Question

Is TensorFold worth continuing to build for concurrent coding agents on an M5 Max, compared with oMLX 0.8.3?
The comparison must use the same weights and the same workload.

**Decision rule.** Keep building TensorFold when it comes within about 15% of oMLX's aggregate throughput on the
same weights and workload, with equal or better accuracy and memory.

## Setup

- **Weights.** Both engines serve oMLX's `Inkling-2-Mini-oQ4e-mtp@3eea042751cd` checkpoint, with MTP off. It
  stores the experts as NVFP4 in layers 0–19 and affine 4-bit in layers 20–23, and attention, embed and head as
  affine 8-bit.
- **Workload.** Each agent runs `campaign/fx/sweep-subagents.sh`: four turns over a 2.5K-token, 115-function
  system prompt. Decoding is greedy, with a 256-token output cap. Each cell has two repetitions in run order, and
  each has gates before and after.
- **Host.** M5 Max with 48 GiB. Runs use `heavy.py` with its 28 GB charge cap and `INKLING_LOAD_LIMIT_GB=26`.
  oMLX runs with `--memory-guard-gb 26` and four slots.

## Served agents, same weights

Aggregate output tokens per second, repetitions 1 and 2. oMLX numbers are from
[inkling-omlx-concurrency.md](kernels/inkling-omlx-concurrency.md).

| Agents | TensorFold (after P1) | oMLX 0.8.3 | TensorFold / oMLX, rep 2 |
|---:|---:|---:|---:|
| 1 | 20.8 / 36.0 | 26.0 / 45.2 | 0.80 |
| 2 | 30.5 / 42.2 | 54.5 / 59.4 | 0.71 |
| 4 | 39.0 / 47.6 | 72.7 / 77.4 | 0.62 |
| 8 | not run | 70.1 / 70.0 | — |

The TensorFold 2-agent row comes from the P1 receipt, from before the store fix that removed the double insert.
Every TensorFold 4-agent concurrent reply equals the same request run alone.

The prefix store brought a large gain on warm repetitions: at 1 agent, TensorFold went from 20.2 to 36.0 tok/s.

## Where the remaining time goes

These raw engine rates on oQ4e come from the gated `tf-inkling2-loop-run` and `batchbench` runs in
[inkling-oq4e-fast.md](kernels/inkling-oq4e-fast.md).

| Metric | TensorFold | oMLX (estimated from served logs) |
|---|---:|---:|
| Prefill, 1K | 329 tok/s | about 930 tok/s |
| Single-stream decode, 1K | 41.8 tok/s | about 55 tok/s |
| Batched step, 4 rows | 69.0 tok/s | — |
| Batched step, 8 rows | 76.6 tok/s | — |

At 4 agents, TensorFold's served rate (47.6) is well below its own 4-row batch rate (69.0). It is not admission: interleaving prefill with decode (B1f) changed nothing, because the agents run in lockstep. The cause is attention over context in the batched step. At about 2.6K context, each row costs about 1.9 ms per 1K tokens, three times single-stream decode, so a 4-row step takes 74 ms against 55 ms on short prompts.

## nanoattention

On the native checkpoint (`ed40e7da625f`), engine-only and unserved, nanoattention prefilled about 1.5× faster
than TensorFold before H3 to H5. Its decode was about 3% faster. See
[inkling-nanoattention.md](kernels/inkling-nanoattention.md); its 16K cells and oracle comparison were not
completed. nanoattention has no concurrent server, so it does not enter the agent comparison.

## Accuracy

These figures compare against saved f32 oracle captures: 16 histories and 3,015 scored positions.

| Weights and engine | Top-1 agreement | Mean KL |
|---|---:|---:|
| oQ4e, TensorFold first-cut kernels (H3) | 91.64% | 0.0394 |
| oQ4e, TensorFold fast kernels (H5) | 92.04% | 0.0393 |

The first 45 greedy IDs on the 1K code prompt are the same for TensorFold and oMLX. The roughly 8% top-1 loss is
the cost of oMLX's 4-bit weights, not of either engine.

## Memory at long context

Per session, TensorFold's KV cache stores 8 global layers' keys and values per token. Local layers keep a
1,024-row ring each.

| KV precision | Global bytes per token | Fixed local rings per session | 128K session | 256K session |
|---|---:|---:|---:|---:|
| f32 (today) | 64 KiB | 335 MB | 8.9 GB | 17.5 GB |
| f16 (H4b) | 32 KiB | 168 MB | 4.5 GB | 8.8 GB |

The oQ4e weights hold about 16 GB, which leaves about 10 GB of KV under the 26 GB load limit:

- **f32 KV.** One 128K session fits. A 256K session does not.
- **f16 KV.** Two 128K sessions fit, or one 256K session with little margin.
- **8-bit KV.** It would halve f16 again, to 16 KiB per token, and allow 256K with room for a second session.

## Recommendation

Pending the final table. Today, TensorFold is at 62–80% of oMLX on the same weights. Every remaining gap has a
measured cause and a design in progress:

1. Decode (H4b).
2. Prefill-stall admission (B1f).
3. Raw prefill (4-bit weights on TensorOps).
4. 8-bit KV for 256K contexts.
