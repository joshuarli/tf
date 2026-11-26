# A1 Phase E: Bounded, Cancellable GPU-Resident Decode Loop

Phase E replaces per-token host `argmax` and host-synchronized step loops with a 2-stage parallel GPU `argmax` and a bounded, host-cancellable GPU-resident token burst loop (`Runtime.generateBurst` + `Runtime.requestCancel`).

## Architecture

1. **Fast 2-stage parallel GPU `argmax` (`i2_aot_argmax_p1` + `i2_aot_argmax_p2`):**
   - Replaces `main`'s single-threadgroup `i2v_argmax` (`1 x 1024` threads, which cost `+0.35 ms` in Phase B) with a 128-threadgroup (`32,768` threads) Stage-1 reduction over the `200,058` logits followed by a 1-threadgroup (`128` threads) Stage-2 reduction.
   - Deterministic lowest-index tie-breaking (`v > bv || (v == bv && i < bi)`), matching `cpuArgmax` bit-for-bit.
   - Stage 2 writes the selected token directly into device-visible `ids[0]` (feeding the next token's `i2_aot_embed_sb` without host intervention) and appends it to `token_ring[step_idx]`.
2. **Bounded, cancellable GPU-resident burst (`Runtime.generateBurst`):**
   - Enqueues `B` decode steps (`1 <= B <= 64`, default `B = 8`) in back-to-back command buffers with a single host wait at the end of the burst (`syncs = 1` per `B` tokens; `4` syncs for 32 tokens vs `32` syncs per-token).
   - Shared `LoopCtrl` control block (`{ cancelled, stopped, produced, pad }`) in coherent shared memory:
     - **Natural EOS stop:** when `argmax_p2` selects `stop_token` (`200006`), it records the token, increments `ctrl->produced`, sets `ctrl->stopped = 1`, and halts further writes to `ids`/`token_ring`/`produced`.
     - **Host cancellation:** calling `Runtime.requestCancel()` from another host thread writes `ctrl->cancelled = 1` atomically into shared memory; the next completing step observes `cancelled != 0`, sets `ctrl->stopped = 1`, and stops emitting tokens. Cancel latency is strictly bounded by the burst window (`<= B * ~12.8 ms`, ~102 ms at `B = 8`).

## Verification

### 1. Functional Parity, EOS Stop, and Host Cancellation (`cancel-test`)

Executed on `bundle-c2 --opt=7` under `campaign/heavy.py`:
- **`burst16_parity=true`:** two consecutive `B = 8` GPU-resident bursts (`16` tokens) matched 16 sequential `stepDirect` CPU-argmax tokens identically (`b1 = 12.09 ms/tok`, `b2 = 12.05 ms/tok` at short 32-token context).
- **`eos_stop_ok=true (n=2)`:** setting `stop_token` to an emitted token halted output at exact first occurrence (`produced = 2`, `stopped = true`, prefix bit-identical).
- **`host_cancel_ok=true (n=2/16 in 193.0 ms)`:** a concurrent host thread calling `requestCancel()` at `30 ms` stopped token emission cleanly at `n = 2` tokens with bit-identical prefix.

### 2. End-to-End 1K Decode (`ctx-1024`, 32 Measured Tokens, `bundle-c2 --opt=7`)

Measured on a quiet host (`memory_pressure` `81%` free, worst pressure `1`, `+0 MB` swap, `heavy.py`):

| Config | Syncs / 32 tok | Dispatches / tok | Wall p50 (ms) | GPU p50 (ms) | Host Gap (`wall - gpu`) | Plain tok/s | `first_sha` | `tok_sha` | Peak Charged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| `main` Phase A (`batch_step`, no argmax) | 32 | 878 | 13.449 | 13.181 | 0.268 ms | 74.35 | — | — | 18.6–19.9 GB |
| `bundle-c2 --opt=7 --burst=0` (CPU argmax) | 32 | 782 | 13.150 | 12.918 | 0.232 ms | 76.05 | `2bfe11565dcd` | `8534f71c2459` | 16.0 GB |
| `bundle-c2 --opt=7 --burst=8` (GPU loop) | **4** | 784 | **12.814** | **12.759** | **0.055 ms** | **78.04** | `2bfe11565dcd` | `8534f71c2459` | **16.0 GB** |

### 3. G1 oMLX Oracle Verification on Standalone AOT Runner (`bundle-c2 --opt=7`)

Executed `tf-inkling2-mini-aot bundle-c2 histories campaign/parity/inkling2-mini/histories.json ... --opt=7` under `campaign/heavy.py` (`40.87 s` for all 16 histories / `3,015` positions, `1.58 s` verified load, `17.2 GB` peak charged, worst pressure `1`, `+0 MB` swap), then scored against `.work/parity/inkling2-mini/omlx-ref` via `omlx_gate.py compare` and checked against `omlx_gate.json` thresholds:

| Metric | AOT `bundle-c2 --opt=7` | `omlx_gate.json` Threshold | Verdict |
| --- | ---: | ---: | --- |
| Positions (`16` histories) | `3,015` | `3,015` | **PASS** |
| Non-finite logits | `0` | `0` | **PASS** |
| Top-1 agreement | `2,865 / 3,015` (`95.02%`) | `>= 94.392%` | **PASS** |
| Confident top-1 (`margin >= 0.1`) | `2,755 / 2,872` (`95.93%`) | reported | `95.93%` |
| Mean KL (mean of history means) | `0.0208875` | `<= 0.0292627` | **PASS** |
| Row-weighted mean KL | `0.0740774` | `<= 0.0962956` | **PASS** |
| Max KL (reported, not gated) | `14.7466` (`cells_code:561`) | reported | — |

## Takeaways

1. **Host/sync gap is essentially eliminated:** `--burst=8` shrinks the wall-to-GPU gap from `0.268 ms` (`main`) and `0.232 ms` (`--burst=0`) down to **`0.055 ms/token`** (`0.43%`), while reducing GPU time by a further `-0.159 ms` (eliminating inter-step GPU idle/wakeup bubbles and replacing CPU argmax with the ~`0.02 ms` 2-stage parallel GPU argmax).
2. **Cumulative AOT plain-decode gain at 1K:** `12.814 ms` wall (`78.04 tok/s`) vs `main` `batch_step` `13.449 ms` (`+5.0%` over `batch_step` without argmax; `~+8–10%` over full served decode `13.85–14.1 ms` with argmax), with `100%` bit-identical token hashes (`8534f71c2459`), `1.58–1.71 s` verified load, and `16.0 GB` peak memory.
3. **G3 (`>= 1.15x`) gap check:** Reaching `1.15x` over `main` `batch_step` (`13.45 ms`) requires `<= 11.70 ms/tok`, leaving a `~1.11 ms` gap under unchanged bit-exact arithmetic. Formal 6-pair G3 evaluation across V4 prompts and 1K–31.7K context follows Phase F.
