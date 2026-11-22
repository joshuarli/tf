# Inkling 2 Mini prompt-prefix capture

The baseline capture on 2026-11-22 did not reach the native host. The Pi model template requests 32,768 output tokens, and the campaign server uses a 32,768-token context. Every nonempty prompt therefore exceeds the server's context budget before inference starts.

The run used the baseline at `a1862ad552db19779c280729f1d3c7f127001115`, with temporary host instrumentation to print the rendered prompt and emitted token IDs. The server used `--no-drafts --temperature 0 --context 32768`. The harness ran all eight tasks serially under `heavy.py` with its 28 GB cap. The Pi task workspaces were isolated from the shared campaign workspaces.

| Task | Prompt tokens | Requests reaching the host | Checker |
| --- | ---: | ---: | --- |
| add-function | 1780 | 0 | fail |
| bugfix-dedupe | 1756 | 0 | fail |
| bugfix-intervals | 1776 | 0 | fail |
| bugfix-window | 1860 | 0 | fail |
| multifile-config | 1776 | 0 | fail |
| read-and-answer | 1827 | 0 | fail |
| refactor-preserve | 1791 | 0 | fail |
| shell-inspect | 1781 | 0 | fail |

Each task recorded one Pi turn ending in a context-length error. The harness returned exit 0 and reported eight completed tasks because Pi itself exited successfully. All eight checkers failed. No prompt-prefix overlaps were measured, and this run does not establish whether Pi reconstructs an append-only template.

The previous result `campaign/pi/results/inkling2-mini-20261122T020629Z.json` used a server without the 32,768-token context flag. Its prompt token IDs are not recorded, so its turn counts cannot establish prefix overlap.

The model template at `campaign/pi/agent/models.json` sets `maxTokens` to 32,768. `campaign/pi/run.py` copies that value into the Pi configuration. A meaningful run with the campaign context needs a smaller output-token limit in the model configuration or harness. Both files are outside this lane's permitted edits. The temporary host instrumentation was removed.

The host was busy according to `perf-gate-ready`, which exited 75. The failed run is not latency evidence. No cache implementation, cache correctness tests, or performance comparison was attempted.
