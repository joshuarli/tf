# Inkling 2 Mini Pi prompt-prefix overlap

Pi template reconstruction did not defeat append-only matching in this run: all 21 consecutive requests across eight tasks exactly extended the preceding prompt and all emitted token IDs. This observation does not establish numerical equivalence between retained state and a fresh batched prefill.

The server used baseline `a1862ad552db19779c280729f1d3c7f127001115` with temporary host instrumentation, `--no-drafts --temperature 0 --context 32768`, and the Inkling 2 Mini checkpoint. The corrected main harness at `13f49d8a218910d4f125edc220caa2c01ad4169a` set Pi’s context to 32,768 and output limit to 8,192. The first capture attempt was blocked by a 32,768-token output request against a 32,768-token context.

The harness ran eight tasks serially under `heavy.py` with the 28 GB cap. Memory pressure was level 1 when the harness started and level 2 while the server waited for the GPU lock. All eight tasks completed, seven checkers passed, and no parse errors were reported. `read-and-answer` failed its checker. This run is an overlap measurement, not a latency qualification.

Temporary host logging captured the IDs of each rendered prompt and every emitted token, including the final stop token. The server’s log lines sometimes interrupted a token dump; the analysis removed those log lines before parsing the complete braces. Every parsed prompt and emitted length matched the corresponding Pi turn’s reported input and output usage. No mismatches remained, so there are no mismatch windows to decode.

The table compares each prompt with the preceding request in the same task. Previous-prompt LCP is the number of matching initial token IDs. Previous prompt + emitted is the length tested by full append. Skippable counts the preceding prompt plus all but the last emitted token, because the no-drafts decode loop stops before consuming that last token. It is an eligibility estimate for a single resident state, not a measured cache hit.

| Task | Turn | Prompt tokens | Previous-prompt LCP | Previous prompt + emitted | Full append | Skippable tokens |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| add-function | 1 | 1780 | — | — | — | 0 |
| add-function | 2 | 2158 | 1780 | 1845 | yes | 1844 |
| add-function | 3 | 2583 | 2158 | 2366 | yes | 2365 |
| add-function | 4 | 3073 | 2583 | 2930 | yes | 2929 |
| bugfix-dedupe | 1 | 1756 | — | — | — | 0 |
| bugfix-dedupe | 2 | 1867 | 1756 | 1801 | yes | 1800 |
| bugfix-dedupe | 3 | 2034 | 1867 | 1938 | yes | 1937 |
| bugfix-dedupe | 4 | 2240 | 2034 | 2175 | yes | 2174 |
| bugfix-intervals | 1 | 1776 | — | — | — | 0 |
| bugfix-intervals | 2 | 2140 | 1776 | 1824 | yes | 1823 |
| bugfix-intervals | 3 | 2314 | 2140 | 2308 | yes | 2307 |
| bugfix-window | 1 | 1860 | — | — | — | 0 |
| bugfix-window | 2 | 2179 | 1860 | 1908 | yes | 1907 |
| bugfix-window | 3 | 2679 | 2179 | 2499 | yes | 2498 |
| multifile-config | 1 | 1776 | — | — | — | 0 |
| multifile-config | 2 | 2073 | 1776 | 1835 | yes | 1834 |
| multifile-config | 3 | 2421 | 2073 | 2307 | yes | 2306 |
| read-and-answer | 1 | 1827 | — | — | — | 0 |
| read-and-answer | 2 | 2282 | 1827 | 1868 | yes | 1867 |
| read-and-answer | 3 | 2720 | 2282 | 2680 | yes | 2679 |
| read-and-answer | 4 | 2833 | 2720 | 2799 | yes | 2798 |
| refactor-preserve | 1 | 1791 | — | — | — | 0 |
| refactor-preserve | 2 | 2063 | 1791 | 1835 | yes | 1834 |
| refactor-preserve | 3 | 2306 | 2063 | 2096 | yes | 2095 |
| refactor-preserve | 4 | 3112 | 2306 | 2882 | yes | 2881 |
| shell-inspect | 1 | 1781 | — | — | — | 0 |
| shell-inspect | 2 | 2047 | 1781 | 1829 | yes | 1828 |
| shell-inspect | 3 | 2162 | 2047 | 2148 | yes | 2147 |
| shell-inspect | 4 | 2276 | 2162 | 2262 | yes | 2261 |

Across all 29 requests, 46,114 of 63,909 prompt tokens (72.16%) meet this eligibility estimate. The first request of each task starts cold. Restricting the denominator to the 21 continuation requests gives 46,114 of 49,562 tokens (93.04%). No cache was implemented or exercised.

The harness result is `campaign/pi/results/c1-overlap-20261122T140000Z.json` in the primary checkout. The result is untracked; the detailed token dumps and task event files are disposable local measurement artifacts.

To repeat the measurement, log `r.prompt` and `job.emitted.items` around `Host.serve`, build the native server, and run the corrected harness with `--model inkling2-mini --pi-model inkling2-mini --bin <binary>`. Run the model through the harness's `heavy.py` guard and compare the token arrays in task order. Remove the instrumentation after capture.

## Retained-state check

The replay used the first two `add-function` requests from the C1 capture at `.work/tmp/pi-server-c1-overlap-20261122T140000Z.log` in the primary checkout. The first prompt has 1,780 IDs; 64 of 65 emitted IDs were consumed by decode; the second prompt has 2,158 IDs and begins with the first prompt and all 65 emitted IDs. The replay input is `.work/wt/c2-prefix-cache/.work/tmp/c1-turn-pair.txt`. It preserves the 512-token first-prompt chunk boundaries, the 64 decode positions, and the suffix chunk boundary.

With one model loaded under `campaign/heavy.py`, the retained-state logits and a fresh-cache replay of the same schedule had `max_abs=0` and zero bit mismatches. Against a fresh full prefill of the second prompt, `max_abs=0.7173862457275391`, KL(retained || full) was `2.617496669352239e-13`, and both greedy tokens were `200001`. The full-prefill comparison characterizes different kernel schedules; it is not the retained-state equality check. The heavy receipt reported peak charge 21.4 GB under the 28 GB cap and worst host memory pressure level 1.

`zig build test-inkling2-loop-host -Dcpu=apple_m1 --summary all` passed 22 tests, including the one prefix-decision test. The broader `test-native` failed twice in the existing `draft_sources_metal_test` GLM test with a trap, after 169 passes and three skips; that unrelated test is outside this lane's writable scope.

## Pi paired runs

The same ReleaseFast native binary ran the eight Pi tasks serially with `INKLING_PREFIX_REUSE=0`, then `1`, then `0`. The host was busy: `perf-gate-ready` exited 75, reporting 5.56–6.19 CPU cores used over its three one-second samples. These are busy-host observations, not a qualified performance gate. The host was an Apple M5 Max with 48 GiB RAM; the model was Inkling 2 Mini at revision `ed40e7da625f`, with context 32,768, greedy decoding and drafts off. The server ran under `heavy.py` with its 28 GB charge cap and `INKLING_LOAD_LIMIT_GB=26`.

The full-run receipts are `campaign/pi/results/c2-prefix-off-20261122.json`, `campaign/pi/results/c2-prefix-on-20261122.json`, and `campaign/pi/results/c2-prefix-off2-20261122.json` in the primary checkout. Their server logs are `.work/tmp/pi-server-c2-prefix-{off,on,off2}-20261122.log` there. Each invocation used separate scratch task workspaces; a runtime output redirect kept the harness's tracked latest-result pointer unchanged. The first off run completed in 329.60 s and the on run in 188.90 s, including startup and lock wait; each completed 8/8 tasks, passed 7/8 checkers and had zero parse errors. In both, `read-and-answer` was the checker failure. The lifetime physical-footprint peak for the on server was 3,797,533,800 bytes; this process footprint does not include all file-backed Metal-wired model weights. The separate replay's 21.4 GB heavy charge above includes host wired growth.

Per-turn prefill is the server's measured prefill duration, and TTFT is its logged first-token time. Task wall and checker values repeat for each turn. Output paths and tool results can alter later prompt lengths, so rows are observations of each run, not identical-prompt timing samples.

| Mode | Task | Turn | Prompt tokens | Reused tokens | Prefill s | TTFT s | Task wall s | Checker |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| off | add-function | 1 | 1787 | 0 | 8.199874 | 8.201 | 61.22 | true |
| off | add-function | 2 | 2165 | 0 | 5.714739 | 5.717 | 61.22 | true |
| off | add-function | 3 | 2558 | 0 | 7.017451 | 7.019 | 61.22 | true |
| off | add-function | 4 | 2837 | 0 | 8.347863 | 8.349 | 61.22 | true |
| off | add-function | 5 | 3214 | 0 | 9.430810 | 9.432 | 61.22 | true |
| off | bugfix-dedupe | 1 | 1763 | 0 | 5.244678 | 5.247 | 32.64 | true |
| off | bugfix-dedupe | 2 | 1889 | 0 | 5.341312 | 5.342 | 32.64 | true |
| off | bugfix-dedupe | 3 | 2036 | 0 | 5.692555 | 5.693 | 32.64 | true |
| off | bugfix-dedupe | 4 | 2242 | 0 | 6.255194 | 6.256 | 32.64 | true |
| off | bugfix-intervals | 1 | 1783 | 0 | 4.915300 | 4.916 | 26.36 | true |
| off | bugfix-intervals | 2 | 2147 | 0 | 5.897542 | 5.898 | 26.36 | true |
| off | bugfix-intervals | 3 | 2317 | 0 | 6.431667 | 6.433 | 26.36 | true |
| off | bugfix-window | 1 | 1867 | 0 | 5.128703 | 5.130 | 32.29 | true |
| off | bugfix-window | 2 | 2186 | 0 | 6.321399 | 6.322 | 32.29 | true |
| off | bugfix-window | 3 | 2663 | 0 | 7.669664 | 7.670 | 32.29 | true |
| off | multifile-config | 1 | 1783 | 0 | 4.799010 | 4.800 | 30.82 | true |
| off | multifile-config | 2 | 2080 | 0 | 6.074084 | 6.075 | 30.82 | true |
| off | multifile-config | 3 | 2428 | 0 | 8.088057 | 8.089 | 30.82 | true |
| off | read-and-answer | 1 | 1834 | 0 | 5.929185 | 5.931 | 55.8 | false |
| off | read-and-answer | 2 | 2288 | 0 | 7.005282 | 7.007 | 55.8 | false |
| off | read-and-answer | 3 | 2827 | 0 | 9.648048 | 9.654 | 55.8 | false |
| off | read-and-answer | 4 | 2921 | 0 | 9.707912 | 9.710 | 55.8 | false |
| off | refactor-preserve | 1 | 1798 | 0 | 6.289490 | 6.292 | 49.83 | true |
| off | refactor-preserve | 2 | 2096 | 0 | 6.876218 | 6.878 | 49.83 | true |
| off | refactor-preserve | 3 | 2339 | 0 | 7.452069 | 7.453 | 49.83 | true |
| off | refactor-preserve | 4 | 2744 | 0 | 8.734055 | 8.736 | 49.83 | true |
| off | shell-inspect | 1 | 1788 | 0 | 6.312151 | 6.314 | 35.91 | true |
| off | shell-inspect | 2 | 2054 | 0 | 6.552318 | 6.554 | 35.91 | true |
| off | shell-inspect | 3 | 2162 | 0 | 6.530139 | 6.531 | 35.91 | true |
| off | shell-inspect | 4 | 2373 | 0 | 6.763259 | 6.764 | 35.91 | true |
| on | add-function | 1 | 1787 | 0 | 4.728246 | 4.730 | 32.76 | true |
| on | add-function | 2 | 2165 | 1851 | 1.089055 | 1.090 | 32.76 | true |
| on | add-function | 3 | 2486 | 2358 | 0.660031 | 0.662 | 32.76 | true |
| on | add-function | 4 | 2692 | 2685 | 0.215041 | 0.217 | 32.76 | true |
| on | add-function | 5 | 3127 | 2983 | 0.677439 | 0.680 | 32.76 | true |
| on | bugfix-dedupe | 1 | 1763 | 0 | 5.470072 | 5.473 | 16.51 | true |
| on | bugfix-dedupe | 2 | 1900 | 1833 | 0.328709 | 0.330 | 16.51 | true |
| on | bugfix-dedupe | 3 | 2056 | 1959 | 0.418742 | 0.420 | 16.51 | true |
| on | bugfix-dedupe | 4 | 2262 | 2196 | 0.342241 | 0.343 | 16.51 | true |
| on | bugfix-intervals | 1 | 1783 | 0 | 4.687311 | 4.689 | 14.67 | true |
| on | bugfix-intervals | 2 | 2147 | 1830 | 0.949535 | 0.950 | 14.67 | true |
| on | bugfix-intervals | 3 | 2317 | 2310 | 0.116372 | 0.117 | 14.67 | true |
| on | bugfix-window | 1 | 1867 | 0 | 5.479675 | 5.481 | 21.27 | true |
| on | bugfix-window | 2 | 2186 | 1914 | 0.881083 | 0.882 | 21.27 | true |
| on | bugfix-window | 3 | 2735 | 2554 | 0.650344 | 0.651 | 21.27 | true |
| on | multifile-config | 1 | 1783 | 0 | 5.056897 | 5.058 | 17.19 | true |
| on | multifile-config | 2 | 2079 | 1840 | 0.751020 | 0.752 | 17.19 | true |
| on | multifile-config | 3 | 2427 | 2312 | 0.474116 | 0.475 | 17.19 | true |
| on | read-and-answer | 1 | 1834 | 0 | 5.045799 | 5.047 | 26.22 | false |
| on | read-and-answer | 2 | 2288 | 1873 | 1.163325 | 1.164 | 26.22 | false |
| on | read-and-answer | 3 | 2741 | 2722 | 0.187068 | 0.188 | 26.22 | false |
| on | read-and-answer | 4 | 2835 | 2823 | 0.174068 | 0.175 | 26.22 | false |
| on | refactor-preserve | 1 | 1798 | 0 | 4.809089 | 4.810 | 30.58 | true |
| on | refactor-preserve | 2 | 2096 | 1867 | 0.693083 | 0.694 | 30.58 | true |
| on | refactor-preserve | 3 | 2339 | 2128 | 0.650858 | 0.652 | 30.58 | true |
| on | refactor-preserve | 4 | 3195 | 2965 | 0.829690 | 0.831 | 30.58 | true |
| on | shell-inspect | 1 | 1788 | 0 | 5.044217 | 5.046 | 15.67 | true |
| on | shell-inspect | 2 | 2054 | 1835 | 0.665242 | 0.666 | 15.67 | true |
| on | shell-inspect | 3 | 2185 | 2170 | 0.190529 | 0.191 | 15.67 | true |
| on | shell-inspect | 4 | 2257 | 2234 | 0.188529 | 0.189 | 15.67 | true |
| off2 | add-function | 1 | 1788 | 0 | 7.567670 | 7.569 | 45.82 | true |
| off2 | add-function | 2 | 2166 | 0 | 5.717149 | 5.718 | 45.82 | true |
| off2 | add-function | 3 | 2493 | 0 | 6.988119 | 6.989 | 45.82 | true |
| off2 | add-function | 4 | 2661 | 0 | 7.380550 | 7.381 | 45.82 | true |
| off2 | bugfix-dedupe | 1 | 1764 | 0 | 4.589998 | 4.592 | 31.19 | true |
| off2 | bugfix-dedupe | 2 | 1901 | 0 | 5.239536 | 5.240 | 31.19 | true |
| off2 | bugfix-dedupe | 3 | 2057 | 0 | 5.628279 | 5.629 | 31.19 | true |
| off2 | bugfix-dedupe | 4 | 2263 | 0 | 5.830872 | 5.832 | 31.19 | true |
| off2 | bugfix-intervals | 1 | 1784 | 0 | 4.610011 | 4.612 | 24.61 | true |
| off2 | bugfix-intervals | 2 | 2148 | 0 | 5.325841 | 5.327 | 24.61 | true |
| off2 | bugfix-intervals | 3 | 2318 | 0 | 5.829535 | 5.830 | 24.61 | true |
| off2 | bugfix-window | 1 | 1868 | 0 | 5.164840 | 5.166 | 33.01 | true |
| off2 | bugfix-window | 2 | 2187 | 0 | 6.812226 | 6.813 | 33.01 | true |
| off2 | bugfix-window | 3 | 2687 | 0 | 7.538261 | 7.539 | 33.01 | true |
| off2 | multifile-config | 1 | 1784 | 0 | 4.755498 | 4.757 | 28.09 | true |
| off2 | multifile-config | 2 | 2074 | 0 | 5.828088 | 5.829 | 28.09 | true |
| off2 | multifile-config | 3 | 2422 | 0 | 6.559581 | 6.560 | 28.09 | true |
| off2 | read-and-answer | 1 | 1835 | 0 | 4.709271 | 4.710 | 44.95 | false |
| off2 | read-and-answer | 2 | 2289 | 0 | 5.926155 | 5.927 | 44.95 | false |
| off2 | read-and-answer | 3 | 2741 | 0 | 6.887538 | 6.888 | 44.95 | false |
| off2 | read-and-answer | 4 | 2835 | 0 | 7.759264 | 7.760 | 44.95 | false |
| off2 | refactor-preserve | 1 | 1799 | 0 | 4.645743 | 4.647 | 52.52 | true |
| off2 | refactor-preserve | 2 | 2097 | 0 | 5.624673 | 5.625 | 52.52 | true |
| off2 | refactor-preserve | 3 | 2340 | 0 | 7.086321 | 7.088 | 52.52 | true |
| off2 | refactor-preserve | 4 | 3210 | 0 | 10.488144 | 10.494 | 52.52 | true |
| off2 | shell-inspect | 1 | 1789 | 0 | 5.986517 | 5.989 | 39.44 | true |
| off2 | shell-inspect | 2 | 2055 | 0 | 6.781796 | 6.785 | 39.44 | true |
| off2 | shell-inspect | 3 | 2165 | 0 | 6.965104 | 6.967 | 39.44 | true |
| off2 | shell-inspect | 4 | 2311 | 0 | 7.132780 | 7.135 | 39.44 | true |

The second off control completed in 324.65 s, with 8/8 tasks complete, 7/8 checkers passed, and no parse errors; `read-and-answer` failed its checker. Its peak lifetime process footprint was 3,807,544,376 bytes, and the on run's was 3,797,533,800 bytes. The sampled backend peak was 3,772,793,912 bytes off and 3,772,875,880 bytes on. The first off run's memory samples were not exported by the harness, so its peak is unavailable. Memory samples for the paired off2/on runs are in the lane's `.work/tmp/c2-{off2,on}-memory.jsonl`; the `/memory` process lifetime peak covers the process from startup even when sampling starts after launch.

The replay and Pi runs used no copy drafts. Cancellation, refusal, capacity exhaustion and copy-draft retained-state behavior were not exercised end to end by these measurements.

The repository lean check reported 249 existing findings on the pinned base and 249 on the change, with zero new findings. `metal.zig` was already over 600 lines at the base; this lane added only the permitted test import there. The host source is 600 lines after removing its three unreferenced sizing helpers; the serving adapter uses the family cache sizing implementation.

The final rebuild followed removal of the unused sizing helpers. Its Mach-O `__TEXT,__text` bytes compare equal to the binary used in all three Pi runs; their executable file hashes differ. The recorded Pi measurements therefore use the same machine code as the final rebuild.
