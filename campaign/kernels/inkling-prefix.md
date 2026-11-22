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
