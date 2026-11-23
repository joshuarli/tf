# oMLX concurrent agents under a 26 GB memory guard

oMLX 0.8.3 loaded `Inkling-2-Mini-oQ4e-mtp@3eea042751cd` offline. With the default eight server slots, the 26 GB oMLX guard did not keep the eight-agent workload under the `heavy.py` 28 GB charge limit. With four server slots, the multi-turn sweep completed at all client levels and its post-run timing gate passed. The standard benchmark completed, but its post-run timing gate failed, so its timings are discarded.

The multi-turn workload uses four turns, a 115-function system prompt, greedy decoding, and a 256-token limit. The server uses four slots, `--memory-guard-gb 26`, and MTP off. TTFT is client time to first visible token, including any queue delay. Each cell is two repetitions, in run order. Peak charged GB uses the `heavy.py` accounting method sampled once per second.

| Agents | Wall seconds, reps 1 and 2 | Aggregate output tokens/s, reps 1 and 2 | Median per-agent tokens/s, reps 1 and 2 | Median TTFT seconds, reps 1 and 2 | Peak charged GB, reps 1 and 2 |
|---:|---:|---:|---:|---:|---:|
| 1 | 14.28 / 8.14 | 25.98 / 45.19 | 25.98 / 45.19 | 0.44 / 0.23 | 21.08 / 18.68 |
| 2 | 13.50 / 12.39 | 54.51 / 59.38 | 27.30 / 29.73 | 0.64 / 0.28 | 20.49 / 20.14 |
| 4 | 20.16 / 18.94 | 72.72 / 77.39 | 18.33 / 19.45 | 1.15 / 0.84 | 25.70 / 24.63 |
| 8 | 42.72 / 42.81 | 70.13 / 70.01 | 9.61 / 9.34 | 5.90 / 5.85 | 25.37 / 24.28 |

The server reported cached prompt tokens for 119 of 120 requests. The first request was the only zero. The eight-agent cells are eight clients served by four concurrent slots. The server log did not report queue durations, so TTFT cannot be split into queue and prefill. No guard pressure event or model eviction was logged during the four-slot sweep.

In the eight-slot guarded attempt, oMLX requested cache reclamation twice at soft pressure, deferred enforcement while pooled Metal buffers drained, and returned to normal pressure after each. At the next hard-pressure event it requested reclamation again, but `heavy.py` had already recorded 28.9 GB charged and stopped the server. The client then reported connection refused. The log did not establish that the guard deferred request admission or evicted the model before the stop.

The four-slot standard benchmark used `--levels 1,2,4,8 --tokens 256 --reps 2 --temperatures 0 --alone --serial`. It reported no failed requests. Its post-run `perf-gate-ready -1` exited 75 with 3.65 busy CPU cores against a limit of 2. The throughput numbers are discarded. The server did not provide token hashes, so the benchmark could not decide token equality. The table compares the output-text hashes recorded by the client; a match does not prove token equality. All responses reached the output limit.

| Prompt | Agents | Failed requests | Output-text matches against alone | Token equality |
|---|---:|---:|---:|---|
| code | 1 | 0 | 2 / 2 | unverified |
| code | 2 | 0 | 0 / 4 | unverified |
| code | 4 | 0 | 0 / 8 | unverified |
| code | 8 | 0 | 0 / 16 | unverified |
| chat | 1 | 0 | 2 / 2 | unverified |
| chat | 2 | 0 | 0 / 4 | unverified |
| chat | 4 | 0 | 2 / 8 | unverified |
| chat | 8 | 0 | 4 / 16 | unverified |

The four-slot run began with 44% system-wide memory free and 16,081.69 MB swap used. A snapshot after the failed final gate showed 73% free and 8,971.50 MB swap used. The second snapshot was taken after server shutdown, not at the exact measurement end. These are host-wide values.

## Earlier unguarded partial runs

Each cell below is one repetition from an unguarded run that stopped at eight agents. No post-run timing gate was taken, so these numbers are unqualified. The first run automatically enabled Lightning MTP. The second explicitly disabled it. `omlx serve --help` has no MTP flag; the isolated per-model settings file controls it.

| MTP | Agents | Wall seconds | Aggregate output tokens/s | Median per-agent tokens/s | Median TTFT seconds | Peak charged GB |
|---|---:|---:|---:|---:|---:|---:|
| off | 1 | 12.33 | 30.08 | 30.08 | 0.45 | 19.91 |
| off | 2 | 13.40 | 54.93 | 27.50 | 0.64 | 20.03 |
| off | 4 | 20.08 | 73.21 | 18.36 | 0.91 | 24.11 |
| on | 1 | 12.54 | 29.59 | 29.59 | 0.56 | 22.77 |
| on | 2 | 14.10 | 52.19 | 26.14 | 0.63 | 20.76 |
| on | 4 | 19.45 | 75.36 | 18.94 | 1.00 | 24.01 |

The MTP-off run stopped at 29.1 GB charged. The MTP-on run stopped at 28.9 GB. A guarded MTP-on arm was not run because the MTP-off standard benchmark did not pass its final timing gate.

## Reproduction

From the repository root, with the pinned model and oMLX environment already installed, run:

```sh
make -C campaign omlx-tests
sh campaign/comparators/omlx/sweep-concurrency.sh
```

The script uses the primary checkout's oMLX environment and cache directories, `HF_HUB_OFFLINE=1`, `INKLING_LOAD_LIMIT_GB=26`, both GPU admission limits at 60%, and the default `heavy.py` 28 GB cap. It uses an isolated settings directory under `.work/tmp/omlx-concurrency-guarded-four` with `mtp_enabled` false. It starts `omlx serve` with `--memory-guard-gb 26 --max-concurrent-requests 4`, waits for `/v1/models`, and runs both clients with timing gates. The copied multi-turn client retains the original workload and reads cached-token counts from the streaming usage response because the original requires TensorFold-specific log records.

The default eight-slot guarded attempt used the same script with `--max-concurrent-requests 4` omitted and a separate isolated settings directory. Its partial multi-turn results were 23.83, 56.48, and 71.60 aggregate tokens/s at one, two, and four agents. Those single repetitions have no post-run gate and are unqualified.
