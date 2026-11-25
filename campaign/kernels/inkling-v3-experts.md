# Inkling 2 Mini grouped verify kernels (V3)

**Verdict.** The grouped verify past eight rows is 6–9% cheaper rested (9 to 16 rows, 1K and 4K) and 12% cheaper in
rounds at 16 rows (60.8 against 69.1 ms). The 9–16-row dense and projection tiles take one weight tile against both
batch tiles (0.67–0.84× DF2b's two-tile kernels at 9 to 12 rows) and routed experts run in runs of at most eight
members (0.89–0.97× main past eight rows). Up to eight rows the verify is unchanged: 0.98–1.02× rested and 0–2% cheaper
in rounds. Down's two- and three-member runs gain (0.92× at 2 rows), but no expert layout tried beat V2's 8x8 tiles for
experts of four to eight live rows, megakernel-zoo's runs of four included (1.23–1.51× at 8 to 16 rows under tf's
arithmetic). Experts with 4–8 live rows still cost about 1.5× their bytes (gate/up 1.51×, down 1.54× at 8 rows), the
lever V2 named, left open. The multi-row controllers and two new attention orders were measured slower or equal and
were dropped. The plan's fixed cost table, refitted from in-round costs, steps 4.6 ms at nine rows instead of 10.5.
Over six interleaved round pairs in two sessions, speculative decode's geometric mean is 1.022 with a 95% interval of
0.986–1.059: inconclusive under Decision 11. Per prompt, code-1024 gains 1.09 (187 tok/s median, 5.34 ms a committed
token against Decision 17's 5.0) and the other five 0.98–1.03. Every verified row equals its one-row step, every reply
equals plain greedy, and the replies' token hashes equal main's. The one-row step is unchanged: it runs main's kernels
(rested 0.98–0.99×, plain decode 0.99–1.00×). Next: the experts' 1.5× excess at 4–8 live rows and the ~13 ms that
every width pays in kernels `prefill.zig` dispatches (mixes, norms, convolutions, router, MLP controller), outside
this lane's files.

**Base.** Main is `53a3b8894` (DF2b's merge). Main's runner is `git archive 53a3b8894` with one measurement
change: `dflash-widths` accepts up to 16 cells (its four 8-cell arrays become 16), so both sides run the same nine
cells in one process. For the kernels-alone comparison main's runner also takes this lane's `kbench.zig` (same call,
a warm-up, widths 1 to 16 and a byte check of each served output); no kernel changes.


## Round pairs

Each process runs DF1's six prompts (`prompts.json`, unchanged) with the oQ4e target and the q4 drafter: 512 greedy
tokens with EOS ignored, plain and speculative from the same prefill state, both builds with tree rounds (`--drafts
tree`), each with its own plan table (main: DF2b's; V3: the refit below). Twelve processes in two back-to-back
sessions, main first in pairs a, c and e and V3 first in b, d and f. Spec tok/s counts the 511 tokens after
prefill's first. Before is main, after is V3; tok/s is the median over the six processes, range in brackets (`run.py
pairs`):

| Prompt | Committed/round before | after | Spec tok/s before | after | After/before per pair | Geomean |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| code-1024 | 7.51 | 7.63 | 164.7 (146.1–184.6) | 187.4 (160.5–193.3) | 0.90, 1.10, 1.24, 1.22, 1.08, 1.04 | 1.09 |
| prose-1024 | 4.52 | 4.48 | 109.5 (108.3–111.7) | 111.0 (109.6–119.0) | 1.00, 1.01, 1.03, 1.02, 1.01, 1.06 | 1.02 |
| structured-1024 | 4.26 | 4.29 | 104.1 (97.8–107.1) | 105.9 (104.2–110.7) | 0.98, 1.02, 1.06, 0.99, 1.09, 1.02 | 1.03 |
| code-4096 | 2.95 | 2.94 | 76.2 (73.8–78.6) | 77.3 (71.1–81.1) | 0.99, 1.03, 1.10, 0.91, 1.03, 1.00 | 1.01 |
| prose-4096 | 2.92 | 2.92 | 78.7 (74.4–81.1) | 78.7 (73.8–82.2) | 1.04, 0.96, 1.04, 0.94, 1.05, 0.98 | 1.00 |
| structured-4096 | 3.73 | 3.62 | 93.3 (88.6–94.0) | 91.8 (86.4–95.0) | 0.94, 1.03, 1.01, 0.93, 0.98, 1.00 | 0.98 |

all prompt-pairs: geometric mean 1.022, 95% interval 0.999–1.046 (n = 36)
0.976, 1.025, 1.077, 0.999, 1.041, 1.018: each process pair's geometric mean over prompts
process pairs: geometric mean 1.022, 95% interval 0.986–1.059 (n = 6)

**Inconclusive.** The interval over process pairs includes 1.0, so Decision 11's promotion rule (a gain of at least 3%
with the interval above 1.0) is not met. Session 1 (pairs a–c) gives 1.025 (0.907–1.159), session 2 (d–f) 1.019
(0.969–1.072): session 2's interval is the narrower.

**An outside soak.** At about 15:47 (its processes' elapsed time) another repository's soak run started on the host,
two `nanos-krun` processes at about 110% CPU each (`bctl/.work/tmp/h1-soak-*`) with `fseventsd` at about 100% and
host swap at 8.4 of 9.2 GiB. It belongs to the owner; the campaign may not pause it. The gates refused or rejected
runs around it, and each run's attempts are recorded in its `NAME.host.json`:

| Run | Attempts (exit codes) | Timed window |
| --- | --- | --- |
| main-a | 2 (75, 0) | 15:21:16–15:23:32 |
| lane-a | 1 | 15:23:34–15:25:57 |
| lane-b | 1 | 15:25:59–15:28:14 |
| main-b | 1 | 15:28:17–15:30:28 |
| main-c | 1 | 15:30:29–15:32:43 |
| lane-c | 2 (75, 0) | 15:35:33–15:37:42 |
| lane-d | 8 (75 ×7, 0) | 15:44:04–15:46:26 |
| main-d | 2 (75, 0) | 15:49:26–15:51:47 |
| main-e | 1 | 15:51:49–15:54:10 |
| lane-e | 1 | 15:54:13–15:56:31 |
| lane-f | 1 | 15:56:34–15:58:50 |
| main-f | 1 | 15:58:51–16:01:09 |

Pair d straddles the soak's start (lane-d before it, main-d after): its ratio is 0.999 over the prompts (code-1024
1.22, prose-1024 1.02, structured-1024 0.99, code-4096 0.91, prose-4096 0.94, structured-4096 0.93). Without it the
five pairs give 1.027 (0.982–1.074); pairs e and f, both inside the soak, give 1.029 (0.893–1.186). The gates check
the host only before and after a run, so a run inside the soak can still be perturbed during it.

### Where the time goes

Median over the six processes of each process's mean, range in brackets (`run.py roundtable`). **Verify** holds the
grouped forward, the GPU argmax, the acceptance walk and the commit in one command buffer.

V3:

| Prompt | Equal | Rounds | Committed/round | Round mean ms | Round p90 ms | Draft | Verify | Host | Spec tok/s | Plain ms/token | Plain tok/s | Spec/plain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code-1024 | yes | 67 | 7.63 | 40.7 (39.5–47.5) | 56.2 (54.6–71.0) | 2.96 | 37.8 | 0.02 | 187.4 (160.5–193.3) | 16.25 (15.90–17.50) | 61.5 (57.1–62.9) | 3.04 (2.68–3.07) |
| prose-1024 | yes | 114 | 4.48 | 40.4 (37.7–40.9) | 46.1 (44.3–48.9) | 4.63 | 35.7 | 0.02 | 111.0 (109.6–119.0) | 16.32 (16.24–17.36) | 61.3 (57.6–61.6) | 1.84 (1.80–1.93) |
| structured-1024 | yes | 119 | 4.29 | 40.5 (38.8–41.2) | 46.4 (44.6–47.4) | 4.55 | 36.0 | 0.01 | 105.9 (104.2–110.7) | 16.79 (16.41–17.27) | 59.6 (57.9–60.9) | 1.80 (1.74–1.82) |
| code-4096 | yes | 174 | 2.94 | 38.0 (36.2–41.3) | 46.1 (43.9–50.7) | 4.77 | 33.2 | 0.01 | 77.3 (71.1–81.1) | 17.06 (16.35–17.36) | 58.6 (57.6–61.2) | 1.32 (1.22–1.35) |
| prose-4096 | yes | 175 | 2.92 | 37.1 (35.5–39.6) | 46.0 (43.6–49.8) | 4.79 | 32.3 | 0.01 | 78.7 (73.8–82.2) | 17.05 (16.78–17.63) | 58.7 (56.7–59.6) | 1.35 (1.30–1.38) |
| structured-4096 | yes | 141 | 3.62 | 39.5 (38.2–41.9) | 48.3 (46.1–51.2) | 4.78 | 34.7 | 0.01 | 91.8 (86.4–95.0) | 17.01 (16.51–17.61) | 58.8 (56.8–60.6) | 1.55 (1.48–1.60) |

Main:

| Prompt | Equal | Rounds | Committed/round | Round mean ms | Round p90 ms | Draft | Verify | Host | Spec tok/s | Plain ms/token | Plain tok/s | Spec/plain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code-1024 | yes | 68 | 7.51 | 45.7 (40.7–51.4) | 68.4 (59.1–82.1) | 3.06 | 42.7 | 0.02 | 164.7 (146.1–184.6) | 16.25 (15.85–16.49) | 61.5 (60.6–63.1) | 2.68 (2.41–2.94) |
| prose-1024 | yes | 113 | 4.52 | 41.3 (40.5–41.8) | 46.1 (45.6–51.4) | 4.73 | 36.5 | 0.02 | 109.5 (108.3–111.7) | 16.29 (16.22–16.59) | 61.4 (60.3–61.6) | 1.80 (1.78–1.81) |
| structured-1024 | yes | 120 | 4.26 | 40.9 (39.8–43.5) | 46.3 (44.7–49.0) | 4.57 | 36.3 | 0.01 | 104.1 (97.8–107.1) | 16.88 (16.69–17.14) | 59.3 (58.4–59.9) | 1.76 (1.68–1.79) |
| code-4096 | yes | 173 | 2.95 | 38.8 (37.6–40.0) | 46.4 (44.8–47.8) | 4.86 | 33.9 | 0.01 | 76.2 (73.8–78.6) | 17.07 (16.85–17.22) | 58.6 (58.1–59.3) | 1.30 (1.27–1.32) |
| prose-4096 | yes | 175 | 2.92 | 37.1 (36.0–39.2) | 45.2 (43.8–47.6) | 4.74 | 32.4 | 0.01 | 78.7 (74.4–81.1) | 17.16 (16.94–17.32) | 58.3 (57.7–59.0) | 1.34 (1.29–1.38) |
| structured-4096 | yes | 137 | 3.73 | 40.0 (39.7–42.1) | 47.6 (46.8–50.0) | 4.77 | 35.2 | 0.01 | 93.3 (88.6–94.0) | 17.21 (17.11–17.56) | 58.1 (56.9–58.5) | 1.61 (1.52–1.63) |

`run.py same` finds every V3 round's tree, the round counts and the output hashes identical in all six V3 processes,
and the six output hashes (`e694b52ab532`, `03832d4821d9`, `cffafa0bf13f`, `bcec689a428e`, `46e6c4217d25`,
`a850b54dc998`) equal main's. With the refitted table V3 verifies different trees from main (67 against 68 rounds on
code-1024, 141 against 137 on structured-4096), and it now picks 11, 13 and 15 rows in a few rounds.

### Verify by rows in rounds

All twelve processes' tree rounds by verified rows (`run.py costs`): the verify phase's mean ms and the tokens
committed.

| Rows | Main rounds | Main verify ms | V3 rounds | V3 verify ms | V3/main | Main committed | V3 committed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 36 | 18.39 | 48 | 18.37 | 1.00 | 1.00 | 1.00 |
| 2 | 336 | 22.99 | 426 | 22.51 | 0.98 | 1.79 | 1.75 |
| 3 | 624 | 27.10 | 582 | 26.71 | 0.99 | 2.16 | 2.24 |
| 4 | 282 | 30.02 | 156 | 29.84 | 0.99 | 2.55 | 2.73 |
| 5 | 768 | 32.76 | 1224 | 32.61 | 1.00 | 2.98 | 2.89 |
| 6 | 630 | 35.40 | 570 | 35.36 | 1.00 | 3.40 | 3.61 |
| 7 | 696 | 37.99 | 72 | 37.98 | 1.00 | 3.96 | 6.00 |
| 8 | 1182 | 40.67 | 1464 | 40.24 | 0.99 | 5.28 | 4.99 |
| 11 | 0 | – | 12 | 49.39 | – | – | 9.50 |
| 13 | 0 | – | 12 | 52.49 | – | – | 13.00 |
| 15 | 0 | – | 6 | 59.85 | – | – | 8.00 |
| 16 | 162 | 69.07 | 168 | 60.82 | 0.88 | 14.00 | 13.29 |

In rounds V3's verify is 0–2% cheaper at one to eight rows and 12% cheaper at 16 (the rounds that verify 16 rows are
lookup chains with the drafter skipped). The drafter ran in 4,554 of V3's rounds at 4.77 ms mean (main 4.81).

## Verify cost by width

`dflash-widths`, nine repetitions of every width on every prompt, the grouped rows each prompt's greedy continuation;
the four processes of each table ran main, V3, V3, main. A cell is the mean over prompts and processes of each
prompt's p50 (range in brackets; `run.py widthcompare`), forward and GPU argmax. Every grouped width's rows were
bit-equal to their one-row steps in every process.

**Rested** (60 s idle inside the lock before each process; `widths-main-a`, `-b`, `widths-lane-a`, `-b`):

| Context | Path | Width | Before p50 | After p50 | Change | After GPU p50 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1,024 | batch step | 1 | 17.3 (15.9–18.5) | 17.0 (15.9–17.8) | 0.98× | 16.4 |
| 1,024 | grouped | 1 | 17.6 (16.2–18.7) | 17.6 (16.3–19.0) | 1.00× | 16.8 |
| 1,024 | grouped | 2 | 21.4 (20.1–22.5) | 21.4 (20.0–22.5) | 1.00× | 20.5 |
| 1,024 | grouped | 4 | 27.3 (25.8–28.8) | 27.6 (26.0–29.3) | 1.01× | 26.8 |
| 1,024 | grouped | 8 | 36.1 (34.6–37.7) | 36.8 (34.8–38.9) | 1.02× | 36.0 |
| 1,024 | grouped | 9 | 46.1 (43.7–48.6) | 42.2 (39.6–45.0) | 0.92× | 41.4 |
| 1,024 | grouped | 10 | 49.0 (46.3–51.7) | 44.8 (42.0–48.1) | 0.91× | 43.9 |
| 1,024 | grouped | 12 | 53.6 (50.5–56.8) | 49.7 (46.0–53.7) | 0.93× | 48.8 |
| 1,024 | grouped | 16 | 61.9 (58.0–66.6) | 58.4 (53.9–63.3) | 0.94× | 57.5 |
| 4,096 | batch step | 1 | 19.3 (18.7–19.7) | 19.0 (18.3–19.5) | 0.99× | 18.4 |
| 4,096 | grouped | 1 | 19.4 (19.0–20.0) | 19.2 (18.4–19.8) | 0.99× | 18.4 |
| 4,096 | grouped | 2 | 23.5 (23.0–24.0) | 23.0 (22.4–23.3) | 0.98× | 22.2 |
| 4,096 | grouped | 4 | 30.5 (30.2–30.7) | 30.6 (29.9–31.2) | 1.00× | 29.8 |
| 4,096 | grouped | 8 | 40.9 (40.2–41.9) | 41.2 (39.9–42.7) | 1.01× | 40.4 |
| 4,096 | grouped | 9 | 52.5 (51.1–53.4) | 47.9 (46.8–49.5) | 0.91× | 47.0 |
| 4,096 | grouped | 10 | 56.0 (54.4–57.5) | 50.8 (49.5–51.8) | 0.91× | 49.9 |
| 4,096 | grouped | 12 | 62.2 (59.9–64.1) | 56.7 (54.8–58.1) | 0.91× | 55.8 |
| 4,096 | grouped | 16 | 72.6 (69.7–75.8) | 67.4 (64.4–70.6) | 0.93× | 66.5 |

**Sustained** (each prompt after 20 s of continuous eight-row verifies; `widths-main-s`, `-s2`, `widths-lane-s`,
`-s2`):

| Context | Path | Width | Before p50 | After p50 | Change | After GPU p50 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1,024 | batch step | 1 | 19.0 (18.1–19.9) | 19.2 (18.0–20.9) | 1.01× | 18.6 |
| 1,024 | grouped | 1 | 19.5 (18.8–20.2) | 19.5 (18.3–21.1) | 1.00× | 18.6 |
| 1,024 | grouped | 2 | 23.7 (22.4–24.9) | 23.7 (21.3–26.0) | 1.00× | 22.8 |
| 1,024 | grouped | 4 | 30.6 (28.5–32.9) | 31.7 (28.9–34.8) | 1.04× | 30.7 |
| 1,024 | grouped | 8 | 40.7 (37.3–44.0) | 42.6 (38.2–47.3) | 1.05× | 41.6 |
| 1,024 | grouped | 9 | 52.8 (49.2–57.8) | 49.1 (44.0–53.7) | 0.93× | 48.1 |
| 1,024 | grouped | 10 | 55.9 (52.0–61.1) | 52.3 (45.7–58.0) | 0.94× | 51.3 |
| 1,024 | grouped | 12 | 61.3 (56.8–67.8) | 58.2 (50.3–64.6) | 0.95× | 57.2 |
| 1,024 | grouped | 16 | 70.9 (63.5–79.7) | 69.3 (61.0–77.2) | 0.98× | 68.3 |
| 4,096 | batch step | 1 | 20.2 (18.1–21.9) | 19.2 (17.4–21.7) | 0.95× | 18.5 |
| 4,096 | grouped | 1 | 20.4 (18.4–22.8) | 19.5 (17.5–21.9) | 0.95× | 18.6 |
| 4,096 | grouped | 2 | 25.4 (23.1–28.7) | 23.6 (21.9–25.1) | 0.93× | 22.7 |
| 4,096 | grouped | 4 | 33.0 (29.6–37.3) | 31.9 (29.7–33.9) | 0.97× | 31.0 |
| 4,096 | grouped | 8 | 44.5 (39.8–50.5) | 42.8 (39.4–44.9) | 0.96× | 41.9 |
| 4,096 | grouped | 9 | 58.0 (50.7–66.2) | 49.8 (47.1–52.0) | 0.86× | 48.8 |
| 4,096 | grouped | 10 | 61.7 (53.8–71.2) | 52.9 (48.3–55.7) | 0.86× | 51.9 |
| 4,096 | grouped | 12 | 67.7 (58.2–78.3) | 58.7 (52.7–62.2) | 0.87× | 57.7 |
| 4,096 | grouped | 16 | 79.2 (68.9–92.9) | 70.2 (63.1–74.5) | 0.89× | 69.2 |

Rested, the one-row step is 0.98× at 1K and 0.99× at 4K, widths one to eight 0.98–1.02×, and 9 to 16 rows
0.91–0.94×. Sustained, each side loads itself with its own eight-row verifies, and the spreads are wider. At 1K
widths four and eight read 1.04× and 1.05×, inside both sides' ranges (8 rows: main 37.3–44.0, V3 38.2–47.3 ms). At
4K every width reads 0.86–0.97×.

## Kernels alone: main against V3

The served kernels alone on the recorded routes, both builds with the same `kbench.zig` (`kernels-cmp-main-a`,
`kernels-cmp-lane-a`, `-b`, `kernels-cmp-main-b`, run in that order), µs a call, the mean of each side's two processes;
every served output was byte-equal to the one-row kernel's. The bytes floor is each call's weights once at 521 GB/s
(codes, scales and biases; 4.48 MB for each distinct routed expert); the one-row head beats it, so 521 GB/s is not a
ceiling.

| Rows | Kind | One-row kernel µs (main / V3) | Main served µs | V3 served µs | V3/main | Bytes floor µs | V3/floor |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | head | 883.5 / 882.9 | 885.8 | 888.0 | 1.00 | 940 | 0.94 |
| 1 | oproj | 17.6 / 17.6 | 17.6 | 17.5 | 1.00 | 14 | 1.21 |
| 1 | proj | 40.9 / 42.2 | 40.7 | 40.5 | 0.99 | 40 | 1.00 |
| 1 | gate_up | 52.8 / 51.5 | 52.8 | 52.5 | 1.00 | 46 | 1.15 |
| 1 | down | 36.7 / 35.7 | 29.0 | 29.4 | 1.01 | 23 | 1.28 |
| 2 | head | 1508.5 / 1509.7 | 1018.6 | 1023.7 | 1.00 | 940 | 1.09 |
| 2 | oproj | 25.9 / 25.9 | 21.0 | 21.0 | 1.00 | 14 | 1.45 |
| 2 | proj | 50.5 / 50.1 | 49.2 | 49.6 | 1.01 | 40 | 1.23 |
| 2 | gate_up | 91.5 / 92.8 | 89.7 | 87.1 | 0.97 | 76 | 1.15 |
| 2 | down | 66.7 / 66.6 | 54.4 | 50.1 | 0.92 | 38 | 1.32 |
| 4 | head | 2990.4 / 2994.2 | 1120.6 | 1120.2 | 1.00 | 940 | 1.19 |
| 4 | oproj | 47.3 / 47.3 | 21.0 | 21.1 | 1.00 | 14 | 1.46 |
| 4 | proj | 75.0 / 74.1 | 55.7 | 56.9 | 1.02 | 40 | 1.41 |
| 4 | gate_up | 180.3 / 177.1 | 143.0 | 151.0 | 1.06 | 116 | 1.30 |
| 4 | down | 132.3 / 129.8 | 83.8 | 81.1 | 0.97 | 58 | 1.40 |
| 8 | head | 6525.7 / 6763.6 | 1253.8 | 1295.7 | 1.03 | 940 | 1.38 |
| 8 | oproj | 110.6 / 106.8 | 27.1 | 25.9 | 0.96 | 14 | 1.79 |
| 8 | proj | 176.4 / 172.0 | 64.7 | 60.7 | 0.94 | 40 | 1.50 |
| 8 | gate_up | 415.7 / 425.2 | 252.8 | 265.9 | 1.05 | 176 | 1.51 |
| 8 | down | 303.2 / 317.1 | 138.5 | 135.7 | 0.98 | 88 | 1.54 |
| 9 | head | 7803.4 / 8028.7 | 2697.8 | 2163.1 | 0.80 | 940 | 2.30 |
| 9 | oproj | 104.9 / 104.3 | 43.0 | 31.0 | 0.72 | 14 | 2.14 |
| 9 | proj | 162.7 / 164.6 | 117.1 | 77.9 | 0.67 | 40 | 1.93 |
| 9 | gate_up | 395.5 / 395.3 | 262.1 | 252.1 | 0.96 | 188 | 1.34 |
| 9 | down | 298.3 / 298.5 | 149.2 | 140.7 | 0.94 | 94 | 1.50 |
| 10 | head | 8599.4 / 8937.7 | 2595.8 | 2070.4 | 0.80 | 940 | 2.20 |
| 10 | oproj | 117.5 / 117.3 | 43.4 | 31.2 | 0.72 | 14 | 2.16 |
| 10 | proj | 182.7 / 184.0 | 117.3 | 79.5 | 0.68 | 40 | 1.97 |
| 10 | gate_up | 438.1 / 441.4 | 283.9 | 276.2 | 0.97 | 200 | 1.38 |
| 10 | down | 331.6 / 332.2 | 163.7 | 147.3 | 0.90 | 100 | 1.47 |
| 12 | head | 10914.1 / 10652.9 | 2681.6 | 2265.2 | 0.84 | 940 | 2.41 |
| 12 | oproj | 139.9 / 140.3 | 43.7 | 31.7 | 0.73 | 14 | 2.20 |
| 12 | proj | 217.7 / 216.7 | 116.5 | 83.4 | 0.72 | 40 | 2.06 |
| 12 | gate_up | 524.5 / 525.3 | 322.9 | 310.8 | 0.96 | 213 | 1.46 |
| 12 | down | 408.8 / 410.2 | 186.9 | 166.7 | 0.89 | 107 | 1.56 |
| 16 | head | 15118.4 / 15232.6 | 3011.2 | 2952.7 | 0.98 | 940 | 3.14 |
| 16 | oproj | 184.0 / 182.9 | 44.1 | 33.7 | 0.76 | 14 | 2.33 |
| 16 | proj | 292.4 / 295.5 | 116.7 | 81.5 | 0.70 | 40 | 2.02 |
| 16 | gate_up | 707.5 / 717.3 | 392.9 | 380.8 | 0.97 | 240 | 1.59 |
| 16 | down | 584.9 / 598.3 | 233.1 | 214.1 | 0.92 | 120 | 1.79 |

The one-row kernels agree between the sides within 4%, so the processes ran in comparable states. At one row every
served kernel is main's (1.00–1.01×). Down's two- and three-member runs make it 0.92× at 2 rows and 0.89–0.94× past
eight; gate/up past eight rows is 0.96–0.97×. The one-tile dense kernels take the head to 0.80–0.84× at 9 to 12 rows
(0.98× at 16), the o-projection to 0.72–0.76× and the fused projections to 0.67–0.72×. Gate/up at 4 and 6 rows reads
4–6% slower than main's in both V3 processes (151 against 143 µs at 4 rows), although it runs main's scalar code and
V2's tiles. In rounds the verify at those widths is not slower (29.8 against 30.0 ms at 4 rows); the cause is not
traced (Open problems).

## Kernels alone, on real routes

Each `dflash-kernels` process times every projection kind alone at widths 1 to 16 on DF2b's recorded routes of
code-1024's greedy continuation (`picks16-code-1024.json`: 48 visits of 16 rows), 24 calls a command buffer with the
layer rotating, five interleaved rounds, median µs a call. Every candidate's output was checked byte for byte against
the one-row kernel on the checkpoint's weights in the same process. Ratios below are to the kernel served in that
process at the same width: candidates of one process share its state, while sustained load slows every kernel as a
long sweep goes on (the unchanged one-row kernels ran 15–30% slower at 8 to 16 rows in the long sweeps than in main's
short run), so only same-process ratios rank candidates. Sweeps `a` and `b` ran the first candidate set
(`285624262`), `c` and `d` the second (`2ccad3140`), each pair back to back.

Against the served kernel of the same sweep (below 1 is faster):

| Kind | Kernel | 1 rows | 2 rows | 4 rows | 8 rows | 9 rows | 12 rows | 16 rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gate_up | runs of 8, scalar to 4 members (a/b) | – | 0.99 | 0.99 | 1.01 | – | – | – |
| gate_up | runs of 4, one column a simdgroup (a/b) | – | 0.96 | 1.05 | 1.23 | – | – | – |
| gate_up | runs of 4, one column, 8 simdgroups (a/b) | – | 0.97 | 1.06 | 1.27 | – | – | – |
| gate_up | runs of 4, one column, 16-row kernel (a/b) | – | 0.98 | 1.07 | 1.30 | 1.32 | 1.41 | 1.51 |
| gate_up | two slots a threadgroup (a/b) | – | 1.16 | 1.14 | 1.14 | 1.10 | 1.09 | 1.09 |
| down | down padded to tiles from 2 members (a/b) | 1.01 | 1.13 | 1.08 | 1.08 | – | – | – |
| down | runs of 4, two rows a simdgroup (a/b) | 0.99 | 1.09 | 1.14 | 1.51 | 1.44 | 1.45 | 1.56 |
| down | runs of 4, four rows a simdgroup (a/b) | 1.02 | 1.03 | 1.04 | 1.14 | 1.02 | 1.06 | 1.14 |

| Kind | Kernel | 1 rows | 2 rows | 4 rows | 8 rows | 9 rows | 12 rows | 16 rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gate_up | main's scalar paths, runs of 8 (c/d) | – | 1.02 | 0.98 | 0.98 | – | – | – |
| gate_up | main's scalar paths, 16-row kernel (c/d) | – | 0.97 | 0.96 | 1.01 | 0.99 | 1.03 | 1.06 |
| down | main's paths: one member scalar, more tiles (c/d) | 0.91 | 1.14 | 1.18 | 1.05 | – | – | – |
| down | main's one-member path, new 2-3 (c/d) | 0.92 | 1.03 | 1.00 | 0.95 | – | – | – |
| head | one weight tile, both batch tiles (c/d) | – | – | – | – | 0.75 | 0.83 | 0.93 |
| head | two threadgroups, one 8-row half each (c/d) | – | – | – | – | 0.90 | 0.91 | 0.90 |
| oproj | one weight tile, both batch tiles (c/d) | – | – | – | – | 0.69 | 0.72 | 0.75 |
| proj | one weight tile, both batch tiles (c/d) | – | – | – | – | 0.72 | 0.68 | 0.73 |
| proj | two threadgroups, one 8-row half each (c/d) | – | – | – | – | 0.86 | 0.81 | 0.82 |

- **Experts, runs of eight with tiles stay.** Every routed-expert layout tried lands within a few percent of V2's at
  2 to 8 rows, and the ones that split an expert's members more finely are slower past four rows (next section).
- **Down for two or three members.** Main pads them to an 8-row tile; decoding each group once for the run is faster
  (`main's paths` row: main's down is 1.14× and 1.18× the new one at 2 and 4 rows), so the new path serves them. For
  one member, main's one-row path is 0.91–0.92× the new scalar path at one row, so a one-row step keeps it.
- **Gate/up up to eight rows** keeps main's scalar code, within 2% of the new path (1.02× at 2 rows, 0.98× at 4 and
  8). Past eight rows the new path is faster (main's code 1.06× at 16 rows).
- **Dense and projections past eight rows.** One weight tile against both batch tiles is 0.68–0.93× DF2b's kernel
  (two weight tiles, each decode shared by both batch tiles); two threadgroups each taking one 8-row half is
  0.81–0.91×. So sharing the decode was not what made 16 rows cost twice 8: the likely cost is register pressure,
  since two weight tiles and two batch tiles hold 16 float2 accumulators a lane and one weight tile holds 8 (not
  measured directly).

## megakernel-zoo's expert layout (external comparison)

megakernel-zoo (Apache-2.0, `95e315c`) serves Inkling 2 Mini on this M5 Max. Its `expert_layout()` (`mk_target.py`)
groups a pass's (row, expert) pairs into slots of at most four pairs of one expert, branches on each slot's live pair
count, gives each simdgroup one gate/up row pair (eight accumulators) and gives down two rows a simdgroup below nine
query rows and four above. Its source records experts at 17.9 → 16.4 ms at 8 rows and 31.3 → 28.5 at 16; those are
the author's figures, not reproduced here. Its arithmetic is not tf's: each 16-weight group is summed as four float4
partial sums and a horizontal add, on 16-bit activations, and its one-row step uses the same order. Decision 18 keeps
tf's sequential lane chains and f32 activations, so only the layout transfers; the sweeps ran it under tf's chains.

- Runs of at most four, one gate/up column a simdgroup: 0.96–0.98× at 2 rows, 1.05–1.07× at 4, 1.23–1.30× at 8 and
  1.32–1.51× at 9 to 16. Down with two rows a simdgroup: 1.44–1.56× past eight rows; four rows a simdgroup:
  1.02–1.14×.
- Why it does not carry over: tf's per-weight work is fixed by its chains (a decode and one fma per member), so a
  slot of four saves no arithmetic over a run of eight, while it reads an expert's weights once per slot. On the
  recorded routes, runs of four read 54.1 experts' weights a visit at 16 rows against 44.6 for runs of eight (41.8
  distinct); at 8 rows 33.8 against 30.7.

| Width | Slots | Distinct experts a visit | 1 member | 2 | 3 | 4 | 5–8 | 9–16 | Runs of ≤8 | Runs of ≤4 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 8.0 | 8.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 8.0 | 8.0 |
| 2 | 16 | 13.2 | 10.5 | 2.8 | 0.0 | 0.0 | 0.0 | 0.0 | 13.2 | 13.2 |
| 4 | 32 | 20.3 | 13.1 | 3.9 | 2.0 | 1.3 | 0.0 | 0.0 | 20.3 | 20.3 |
| 8 | 64 | 30.7 | 16.9 | 6.6 | 2.5 | 1.6 | 3.1 | 0.0 | 30.7 | 33.8 |
| 9 | 72 | 32.8 | 17.2 | 7.5 | 2.8 | 1.7 | 3.0 | 0.6 | 33.4 | 36.9 |
| 10 | 80 | 35.0 | 17.8 | 7.8 | 3.3 | 2.1 | 3.0 | 0.9 | 35.9 | 39.9 |
| 12 | 96 | 37.2 | 16.7 | 8.5 | 4.2 | 2.5 | 3.6 | 1.8 | 39.0 | 44.3 |
| 16 | 128 | 41.8 | 16.8 | 8.6 | 5.1 | 3.0 | 5.4 | 2.9 | 44.6 | 54.1 |

## The verify's attention

The grouped verify's split kernel ran in three orders (`run-split-*` runners, otherwise identical), two processes
each on code-1024 and code-4096 (`split-*-a`, `-b`), p50 ms of the grouped forward and picks:

| Context | Width | Main's order | Rows adjacent | Shared key staging |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 1 | 16.6 | 16.5 (0.99×) | 16.7 (1.01×) |
| 1,024 | 2 | 20.5 | 20.4 (1.00×) | 21.5 (1.05×) |
| 1,024 | 4 | 26.7 | 26.7 (1.00×) | 29.3 (1.10×) |
| 1,024 | 8 | 35.8 | 36.1 (1.01×) | 40.0 (1.12×) |
| 1,024 | 12 | 49.7 | 49.9 (1.00×) | 54.7 (1.10×) |
| 1,024 | 16 | 57.2 | 57.2 (1.00×) | 63.1 (1.10×) |
| 4,096 | 1 | 19.5 | 19.2 (0.98×) | 19.0 (0.97×) |
| 4,096 | 2 | 23.3 | 22.9 (0.98×) | 24.1 (1.04×) |
| 4,096 | 4 | 30.2 | 30.4 (1.00×) | 33.3 (1.10×) |
| 4,096 | 8 | 40.1 | 40.1 (1.00×) | 45.0 (1.12×) |
| 4,096 | 12 | 56.3 | 56.1 (1.00×) | 61.8 (1.10×) |
| 4,096 | 16 | 65.7 | 65.2 (0.99×) | 71.9 (1.10×) |

Rows of one stream next to each other in the grid change nothing. Staging a stream's cached keys once for up to four
rows (`i2v_split_runs`, byte-equal on 256 random chains and trees) is 10–12% slower from 4 rows on: its 24 KB of
threadgroup memory a group and its rows walked one after another cost more than the key reads it saves. The verify
keeps main's kernel.

## Controllers (reverted)

The grouped verify's three attention-input controllers first read each weight row once for eight rows
(`i2v_controls_down`, `i2v_controls_apply` in `group.scratch`, byte-equal at 1 to 16 rows). In that build's
timelines (`controllers-main-*`, `controllers-lane-*`, ms a round) they cost more at every width, from too few
threadgroups each walking its rows in turn:

| Build | batch | group-1 | group-2 | group-4 | group-8 | group-9 | group-12 | group-16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| main | 1.58 | 1.58 | 1.63 | 1.93 | 2.73 | 2.94 | 3.33 | 4.26 |
| multi-row controllers | 1.75 | 1.69 | 1.94 | 2.53 | 3.77 | 3.79 | 4.42 | 5.02 |

They were reverted before any widths or round run; the final build has main's controllers.

## Where the verify spends

Metal 4 timelines on code-1024 (`run.py widths --timeline 12`: 20 repetitions, 12 stamped after 8 warm), main → V3, ms
a round by category. Past eight rows `prefill.zig` names every grouped round `inkling2-group-8`, so 9, 12 and 16 rows
each ran in a process of their own. A stamp a dispatch adds about 5 µs: these spans attribute cost; they are not
timings.

| Category | batch | group-1 | group-2 | group-4 | group-8 | group-9 | group-12 | group-16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| routed experts, NVFP4 | 4.08 → 4.10 | 4.04 → 4.06 | 7.06 → 6.85 | 10.93 → 10.80 | 16.55 → 16.49 | 20.02 → 18.71 | 24.38 → 22.67 | 28.66 → 26.78 |
| routed experts, affine 4-bit | 0.35 → 0.20 | 0.30 → 0.20 | 0.33 → 0.32 | 0.89 → 0.92 | 1.64 → 1.61 | 1.83 → 1.81 | 2.39 → 2.38 | 3.15 → 3.18 |
| router, pick, combine | 2.22 → 2.26 | 2.15 → 2.18 | 2.06 → 2.27 | 2.21 → 2.22 | 2.32 → 2.35 | 2.36 → 2.49 | 2.53 → 2.51 | 2.67 → 2.59 |
| attention projections | 2.54 → 2.51 | 2.52 → 2.58 | 2.94 → 3.06 | 3.40 → 3.53 | 3.38 → 3.55 | 6.05 → 4.35 | 6.07 → 4.38 | 6.19 → 4.50 |
| o-projection and head | 1.92 → 1.90 | 1.91 → 1.88 | 2.20 → 2.12 | 2.27 → 2.28 | 2.34 → 2.35 | 4.42 → 3.33 | 4.48 → 3.71 | 4.68 → 4.13 |
| attention | 3.50 → 3.62 | 3.54 → 3.54 | 3.72 → 3.81 | 4.20 → 4.31 | 5.53 → 5.65 | 5.90 → 5.96 | 6.91 → 6.86 | 8.19 → 8.07 |
| convolutions | 2.22 → 2.33 | 2.29 → 2.30 | 2.15 → 2.29 | 2.22 → 2.33 | 2.32 → 2.26 | 2.26 → 2.33 | 2.27 → 2.21 | 2.16 → 2.11 |
| controllers | 1.62 → 1.67 | 1.63 → 1.61 | 1.68 → 1.73 | 1.95 → 2.04 | 2.75 → 2.81 | 2.94 → 2.90 | 3.52 → 3.50 | 4.27 → 4.18 |
| depth mixes and norms | 3.60 → 3.78 | 3.61 → 3.80 | 3.66 → 3.73 | 3.65 → 3.69 | 3.64 → 3.78 | 3.65 → 3.73 | 3.82 → 3.74 | 3.76 → 3.71 |
| other kernels and stamps | 0.18 → 0.16 | 0.15 → 0.15 | 0.36 → 0.32 | 0.18 → 0.18 | 0.25 → 0.18 | 0.18 → 0.19 | 0.20 → 0.18 | 0.20 → 0.19 |
| round span | 21.81 → 22.09 | 21.73 → 21.89 | 25.74 → 26.08 | 31.49 → 31.87 | 40.33 → 40.59 | 49.17 → 45.37 | 56.14 → 51.71 | 63.49 → 59.00 |

Past eight rows the verify is 4.0–4.5 ms cheaper a round: the fused attention projections 1.7 ms, the head and
o-projection 0.6–1.1 ms and the routed experts 1.3–1.9 ms. Up to eight rows the kernels are main's except down's two-
and three-member runs (−0.1 to −0.2 ms of experts at 2 to 4 rows); the other categories differ by up to 0.2 ms either
way in kernels that are main's, which is these single processes' spread.
## The plan's cost table

`verify.Plan` prices each tree's rows from a fixed table, so no time measured in a round changes a tree. DF2b's
table was rested and stepped 10.5 ms at nine rows, where its two-tile kernels began; the rounds measured its 16 rows
14% above it. V3's table comes from one provisional round process of this build with DF2b's table
(`provisional-a`, the six prompts, 512 tokens each), which verified the rows below:

| Rows | Rounds | Drafter ran | Verify ms mean | Verify p50 | Committed | Round ms | Committed a round-ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6 | 4 | 18.57 | 19.48 | 1.00 | 21.6 | 0.046 |
| 2 | 56 | 56 | 22.25 | 22.67 | 1.79 | 26.9 | 0.066 |
| 3 | 104 | 104 | 25.89 | 25.87 | 2.16 | 30.5 | 0.071 |
| 4 | 47 | 47 | 29.15 | 28.94 | 2.55 | 33.8 | 0.075 |
| 5 | 128 | 128 | 31.13 | 31.62 | 2.98 | 35.7 | 0.083 |
| 6 | 105 | 105 | 33.69 | 33.74 | 3.40 | 38.2 | 0.089 |
| 7 | 116 | 116 | 36.46 | 36.84 | 3.96 | 41.0 | 0.096 |
| 8 | 197 | 196 | 38.14 | 38.66 | 5.28 | 42.6 | 0.124 |
| 16 | 27 | 0 | 58.08 | 56.54 | 14.00 | 58.1 | 0.241 |

draft ms over the 756 rounds that ran the drafter: mean 4.54, p50 4.58

Widths that rounds verify 20 times or more take their in-round mean (2 to 8 and 16 rows). Widths 1, 9, 10 and 12 take
the sustained widths runs (`widths-lane-s`, `-s2`) scaled by the in-round to sustained ratio of their measured
neighbours, and 11 and 13 to 15 the line between, made non-decreasing (`run.py costtable`):

| Rows | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DF2b (rested) | 17.9 | 21.8 | 24.9 | 28.0 | 30.2 | 32.4 | 34.5 | 36.7 | 47.2 | 49.5 | 51.8 | 54.0 | 56.2 | 58.5 | 60.8 | 63.0 |
| V3 (in rounds) | 18.3 | 22.3 | 25.9 | 29.2 | 31.1 | 33.7 | 36.5 | 38.1 | 42.7 | 45.4 | 47.9 | 50.4 | 52.4 | 54.3 | 56.2 | 58.1 |

The step at nine rows falls from 10.5 to 4.6 ms. The draft stays priced at 4.5 ms (4.54 ms mean over the 756 rounds
that ran it). The round pairs below ran with this table.
## Decision 17 budgets

Decision 17 allows at most 20, 25, 30 and 40 ms a round at 4, 5, 6 and 8 mean committed tokens, 5 ms a committed
token (`run.py budgets`, V3's six processes):

| Prompt | Committed/round | Round ms | Budget at that commitment ms | ms a committed token | Meets |
| --- | ---: | ---: | ---: | ---: | --- |
| code-1024 | 7.63 | 40.7 | 38.1 | 5.34 | no |
| prose-1024 | 4.48 | 40.4 | 22.4 | 9.01 | no |
| structured-1024 | 4.29 | 40.5 | 21.5 | 9.44 | no |
| code-4096 | 2.94 | 38.0 | 14.7 | 12.94 | no |
| prose-4096 | 2.92 | 37.1 | 14.6 | 12.70 | no |
| structured-4096 | 3.62 | 39.5 | 18.1 | 10.90 | no |

No prompt meets its budget; code-1024 is the closest yet, 5.34 ms a committed token (DF2b: 6.63, main here: 6.08).

## What bounds verify next

- **Experts at four to eight live rows.** At eight rows the routed NVFP4 experts take 16.5 ms of the round
  (timeline) against 11.6 ms for the 30.7 distinct experts' bytes; alone, gate/up runs at 1.51× its bytes and down at
  1.54×. Every layout tried for experts of four to eight members (V2's tiles with other geometries, runs of four,
  one output column a simdgroup, two slots a threadgroup) measured at least V2's cost. tf's chains fix the arithmetic
  per weight, and an 8x8 tile computes eight rows' products whatever its live rows.
- **What every width pays.** At one row the round spends about 13 ms outside the experts and dense projections:
  depth mixes and norms 3.8 ms, attention 3.5, convolutions 2.3, router 2.3, controllers 1.6. That is why the one-row
  step stays at about twice its 7.65 ms floor (V1). Except attention, these are dispatched from `prefill.zig` through
  `kernels.zig`'s pipelines (the MLP controller, the router, the affine 4-bit experts in layers 20–23), outside this
  lane's files. Attention grows 2.1 ms from one row to eight; its row-adjacent order and shared key staging did not
  help.
- **Past eight rows** the head (2.95 ms alone at 16 rows, 3.1× its bytes) and the experts (gate/up 1.59×, down 1.79×
  at 16 rows) are compute-bound: 16 rows of products per weight.
- **Acceptance.** As DF2b found, drafter rounds of eight rows commit about five tokens in 45 ms (0.111 tokens a ms
  here); 200 tok/s needs 0.2. The 16-row lookup rounds commit 13.3 tokens in 61 ms (0.218).
## What changed

All in `zig/src/families/inkling2_loop/`: `kernels/experts.metal`, `kernels/rows.metal`, `rows.zig`, `verify.zig`,
`kbench.zig`, `dflash_run.zig` and `attention_test.zig`. No shared core, prefill, host, build or other family file.

- **Routed NVFP4 experts** (`kernels/experts.metal` behind `I2T_ROUTED`, compiled after `rows.metal` into `rows.zig`'s
  library; the model's own expert library does not define it). A threadgroup is one block of an expert's stored rows
  and a run of at most eight of its members in slot order, led by the run's first slot (`i2e_run`). An expert with
  nine to sixteen members takes two runs, so no run is padded past eight rows: main ran every expert of more than
  eight members as a 16-row tile pair. Runs of one to three members run scalar code, four to eight V2's 8x8 tiles.
  - Up to eight rows, gate/up keeps main's scalar code (V2's `i2t_columns`) and a one-row step's down keeps main's
    one-row path (`i2e_down1`). Down's runs of two and three members now decode each group once for all of them
    (`i2e_scalar`, four inputs a load); main padded them to 8-row tiles.
  - Past eight rows both projections take the new scalar path for runs of one to three.
- **Dense and projection tiles past eight rows** (`kernels/rows.metal`): one 8-row weight tile a simdgroup against
  both batch tiles (`i2t_dense16_8_1`, `i2t_projections16_8_1`) instead of two weight tiles (V2's `T = 2`). The head,
  o-projection and fused attention projections at 9 to 16 rows.
- **The plan's cost table** (`verify.zig`), refitted from this lane's in-round verify costs (below).
- **Measurement** (`kbench.zig`, `dflash_run.zig`, the receipt's `run.py`): `dflash-kernels` warms up, runs widths 1 to
  16 and checks each served output byte for byte; `dflash-widths` takes up to 16 widths. The sweeps ran on two
  development commits whose `dflash-kernels` also timed named candidates (`285624262`, `2ccad3140`).

Kept from main after measurement: the controllers, the verify's attention split and the one-row step's kernels (its
down runs main's one-row code, moved into `i2e_down1`). V2's expert tiles moved from `rows.metal` to `experts.metal`
unchanged; `group.zig` and `verify.metal` are main's.

## Correctness

The contract is V1's, V2's and DF2b's, unchanged (Decision 18): every row of a batch step or grouped verify has its
path's one-row steps' logits, picks, taps and committed caches bit for bit, so greedy speculation equals greedy
decoding. Every check is bit equality.

- **Experts** (`attention_test.zig`): the served dispatch against `i2p_gu` and `i2p_down` at the served shapes (k 2304
  and 1152; main's test used an intermediate width of 48, so each of down's lane chains held one group) at 1 to 16 rows, on
  random picks over 12 experts, on every row routed the same 8, and on layouts whose experts hold every member count
  from 1 to 16 at each width (the test asserts the counts it saw). Every candidate of the sweeps passed the same test
  (1,008 and 1,320 cases) before it was timed.
- **Dense and projection tiles** (`rows_test.zig`, unchanged): the served tiles against `i2a_dense` and
  `i2a_batch_projections` at 2 to 16 rows (120 and 28 cases); the sweeps' candidates passed it too (944 and 220).
- **Checkpoints**, unchanged tests over the new kernels: `batch_test.zig`, `group_test.zig` (grouped and tree verify
  rows equal to their paths' one-row steps, commits), `dflash_test.zig` (tree speculation equal to plain greedy) and
  the served host's test.
- **Every timed run** checked on the way: each kernel candidate equal to the one-row kernel on the checkpoint's
  weights and routes, every grouped width equal to its one-row steps in every widths run, every round reply equal
  to plain greedy.

On the final code (`e3b30971c`), `python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu` exited 0
(`.work/suite/20261124T160438`, 16:04–16:30): inkling-host 10/10 steps and 35/35 tests, inkling-gpu 17/17 steps and
179/179 tests with none skipped (`suite-summary.json`, `suite-inkling-host.txt`, `suite-inkling-gpu.txt`). Its tree
speculation test ran 262 rounds, 10 of them past eight rows, equal to plain greedy and its caches; heavy.py's charged
peak was 23.5 GB (footprint 21.6 GB). `zig build test-flint-draft-host` exited 0 (47/47 steps, 87/87 tests;
`flint-draft-host.txt`). `tools/zig/lean_new.py` against `53a3b8894` finds 0 new problems (256 on both), and `git diff
--check` is clean.

## Conditions

- **Machine.** Apple M5 Max (18 CPU cores, 40 GPU cores, 48 GB), macOS 27.2.
- **Lock and gates.** Every timed run, model load and Metal test went through `campaign/heavy.py` (the GPU lock and
  the 28 GB cap). Each timed `run.py` process passed `gpu-gate-ready -1` (60% limits) and `perf-gate-ready`
  (`PERF_GATE_MAX_BUSY_CORES=4`) before and after it; `widths-lane-b` failed its closing check once and was run
  again. Readings are in each `NAME.host.json`: other processes held 20–36% of GPU time around most runs
  (WindowServer and the terminal), 0.2–0.4% around the provisional rounds. `INKLING_LOAD_LIMIT_GB=26`; caches held at
  most 4,096 + 512 + 16 positions. The rested widths runs idled 60 s inside the lock before loading.
- **One timing block.** The coordinator granted the GPU to this lane from 14:21; no other lane timed or compiled on it
  meanwhile. Phase 1 (sweeps a and b, main's kernels, the split orders) ran 14:22–14:29, sweeps c and d 14:30–14:32,
  the final profiles and widths 14:46–15:13, the provisional rounds 15:13–15:15, the round pairs (two sessions back
  to back) from 15:18 and the kernels-alone comparison after them.
- **One compile inside a timed window.** An offline Metal syntax check (`xcrun metal -c`, about 2 s of one core) ran
  at 14:37:00 during the first phase-2 attempt's `profile-main-9`. That attempt was stopped for the controllers and
  its files kept as `controllers-*`; every profile in the tables above was taken again afterwards.
- **Thermal and memory.** `pmset -g therm` recorded no thermal or performance warning before or after any run;
  heavy.py saw memory pressure 1–2, no swap growth, and a charged peak of at most 20.6 GB (footprint 17.7 GB).
- **Builds.** Main: `git archive 53a3b8894` plus the widths-cell patch (`run-main`), and for the kernels alone also
  this lane's `kbench.zig` (`run-main-k`). V3: `285624262` for sweeps a and b, `2ccad3140` for c and d, `f26907c53`
  with `group.zig`'s split order set three ways for the split runs, `3b12cd35b` for the stopped controller attempt,
  `d16de328c` for the profiles, widths and provisional rounds, and `e3b30971c` (its plan table only differs) for the
  round pairs and the kernels alone.

## Reproduction

From the worktree. Runners (`--prefix` directories outside the tree):

```sh
git archive 53a3b8894 | tar -x -C MAIN_SRC
# MAIN_SRC/zig/src/families/inkling2_loop/dflash_run.zig: dflash-widths' chosen, samples, forward_only and gpu_only
# arrays from [8] to [16]
(cd MAIN_SRC && zig build tf-inkling2-loop-run -Dcpu=apple_m1 --prefix RUN_MAIN)
cp zig/src/families/inkling2_loop/kbench.zig MAIN_SRC/zig/src/families/inkling2_loop/kbench.zig
(cd MAIN_SRC && zig build tf-inkling2-loop-run -Dcpu=apple_m1 --prefix RUN_MAIN_K)
zig build tf-inkling2-loop-run -Dcpu=apple_m1 --prefix RUN_LANE
```

The sweeps' runners are the same command at `285624262` and `2ccad3140`; the split orders' at `f26907c53` with
`const split_kernel: Split = .splits_first;` (or `.rows_first`, `.runs`) in `group.zig`. Then, with `M`, `MK`, `L`
those runners' `bin/tf-inkling2-loop-run`, `S1`, `S2` the sweeps', `X_<order>` the split orders' and
`D=campaign/kernels/inkling-v3-experts-runs`:

```sh
P=$D/picks16-code-1024.json
python3 -I $D/run.py kernels $S1 $D/prompts.json kernels-sweep-a --picks $P --variants all
python3 -I $D/run.py kernels $M $D/prompts.json kernels-main-a --picks $P
python3 -I $D/run.py kernels $S1 $D/prompts.json kernels-sweep-b --picks $P --variants all
for run in "splits_first a" "rows_first a" "runs a" "runs b" "rows_first b" "splits_first b"; do
  set -- $run; python3 -I $D/run.py widths $X_$1 $D/code-1k4k.json split-$1-$2 --reps 15 --cells 1,2,4,8,12,16
done
V=i2e_gate_up_f3,i2e_gate_up_o3,i2e_gate_up16_f3,i2e_gate_up16_o3,i2e_down_f3,i2e_down_m,i2e_down_o1,i2e_down_o3
V=$V,i2e_down16_f3,i2e_down16_o1,i2e_down16_o3,i2t_dense_8_2,i2t_dense_8_1,i2t_dense_8_4,i2t_dense16_8_1
V=$V,i2t_dense16_h8_2,i2t_projections_8_2,i2t_projections_8_1,i2t_projections16_8_1,i2t_projections16_h8_2
for n in c d; do python3 -I $D/run.py kernels $S2 $D/prompts.json kernels-sweep-$n --picks $P --variants $V; done
for side in "$M main" "$L lane"; do set -- $side
  python3 -I $D/run.py widths $1 $D/code-1024.json profile-$2-8 --reps 20 --timeline 12 --cells 0,1,2,4,8
  for w in 9 12 16; do python3 -I $D/run.py widths $1 $D/code-1024.json profile-$2-$w --reps 20 --timeline 12 --cells $w; done
done
W=0,1,2,4,8,9,10,12,16
for run in "$M widths-main-a" "$L widths-lane-a" "$L widths-lane-b" "$M widths-main-b"; do
  set -- $run; python3 -I $D/run.py widths $1 $D/prompts.json $2 --reps 9 --rest 60 --cells $W; done
for run in "$M widths-main-s" "$L widths-lane-s" "$L widths-lane-s2" "$M widths-main-s2"; do
  set -- $run; python3 -I $D/run.py widths $1 $D/prompts.json $2 --reps 9 --load 20 --cells $W; done
python3 -I $D/run.py bench $L $D/prompts.json provisional-a --drafts tree --tokens 512
python3 -I $D/run.py costtable provisional-a --widths widths-lane-s,widths-lane-s2
# (verify.zig's table set from that output; rebuild L)
for run in "$M main-a" "$L lane-a" "$L lane-b" "$M main-b" "$M main-c" "$L lane-c" \
           "$L lane-d" "$M main-d" "$M main-e" "$L lane-e" "$L lane-f" "$M main-f"; do
  set -- $run; python3 -I $D/run.py bench $1 $D/prompts.json $2 --drafts tree --tokens 512; done
for run in "$MK kernels-cmp-main-a" "$L kernels-cmp-lane-a" "$L kernels-cmp-lane-b" "$MK kernels-cmp-main-b"; do
  set -- $run; python3 -I $D/run.py kernels $1 $D/prompts.json $2 --picks $P; done
python3 -I $D/run.py pairs main-a,main-b,main-c,main-d,main-e,main-f lane-a,lane-b,lane-c,lane-d,lane-e,lane-f
python3 -I $D/run.py roundtable lane-a,lane-b,lane-c,lane-d,lane-e,lane-f
python3 -I $D/run.py roundtable main-a,main-b,main-c,main-d,main-e,main-f
python3 -I $D/run.py costs lane-a,lane-b,lane-c,lane-d,lane-e,lane-f
python3 -I $D/run.py costs main-a,main-b,main-c,main-d,main-e,main-f
python3 -I $D/run.py widthcompare widths-main-a,widths-main-b widths-lane-a,widths-lane-b
python3 -I $D/run.py widthcompare widths-main-s,widths-main-s2 widths-lane-s,widths-lane-s2
python3 -I $D/run.py variants kernels-sweep-a,kernels-sweep-b
python3 -I $D/run.py variants kernels-sweep-c,kernels-sweep-d
python3 -I campaign/suite/suite.py run --names inkling-host,inkling-gpu
```

`attribution` reads the `.txt` copies of the timeline logs (`.log` files are not tracked), for example
`python3 -I $D/run.py attribution profile-lane-8.txt inkling2-batch=batch inkling2-group-8=group-8
profile-lane-16.txt:inkling2-group-8=group-16`.

## Open problems

- **Experts of four to eight live rows** still cost about 1.5× their bytes at eight rows (above); V2's named lever
  stays open under tf's arithmetic.
- **Gate/up at 4 and 6 rows** measured 4–6% slower than main's kernel alone (both V3 processes), though its code is
  main's scalar path and V2's tiles; not traced. The rounds' verify at those widths was not slower.
- **Inconclusive round gain.** The interval includes 1.0, and the second session ran beside an outside soak. A
  repeat on a quiet host would settle code-1024's 1.09 and whether the whole is above 3%.
- **The cost table** comes from one provisional process (with DF2b's table) plus sustained widths runs; widths 11, 13
  to 15 are interpolated and rounds now pick some of them.
- **Outside this lane's files:** the affine 4-bit experts (layers 20–23), the router and the MLP controller in the
  batch path use `kernels.zig`'s pipelines dispatched from `prefill.zig`, and the depth mixes' grid is fixed there.
- **Not in the kernel-contract runner.** As V2 and DF2b recorded, none of the family's kernels is registered there;
  the new instantiations are checked by the byte tests in `attention_test.zig` and `rows_test.zig` only.
- **Timeline names stop at eight rows** (`prefill.zig`), so each width past eight needs a process of its own.
