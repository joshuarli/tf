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
