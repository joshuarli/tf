# Inkling 2 Mini depth snapshot normalization experiment

Decision: revert. The indexed, cached-normalization candidate preserved bytes and removed work, but its
83-token decode was slower in all five gated pairs. Median step wall time rose from 22.691 ms to 22.974 ms
(+1.25%); the baseline range was 22.545–22.898 ms and the candidate range was 22.887–23.104 ms. This is
one local experiment, not evidence of a general limit on depth caching. The candidate and its test are in
commits `ad3329cf2` and `55818c5dc`; the final tree restores the original decode.

The candidate normalized each block at publication, copied the normalized core bank before the pass rewind,
and normalized each pass output once. Raw partials stayed in `x` for the convolution paths. Each mix read
indexed snapshot rows and a separately normalized current row, in the original row order. The focused GPU
test compared the old copy-and-normalize kernels against this path bit for bit, including the prelude,
three passes, rewinds, and previous-pass messages. Full teacher-forced logits were byte-identical to main
for greeting (25 rows), sql (33), prose_review (35), and cells_code (1,178); reset reproduced 25 rows.

Apple M5 Max, 48 GiB, model `Inkling-2-Mini@ed40e7da625f`, main runner `318cdda7f`; the candidate's base
`dd3ae4a28` differs from main only in the campaign plan. The prompt was the first 400 bytes of
`campaign/prompts/dev/code-1024.txt`, tokenized to 83 IDs. Both runners used `bench 14 2`: 14 measured
steps after two warmups, one process at a time under the campaign's 28 GB heavy-command cap. Each pair
alternated execution order; every run had a quiet `gpu-gate-ready -1` before it and a quiet
`GPU_GATE_IGNORE_PIDS=<runner pid> gpu-gate-ready -1` afterwards. No run was discarded.

| Pair | Main median ms/step | Candidate median ms/step |
| --- | ---: | ---: |
| 1 | 22.691 | 22.968 |
| 2 | 22.626 | 22.887 |
| 3 | 22.545 | 23.005 |
| 4 | 22.898 | 22.974 |
| 5 | 22.818 | 23.104 |

Separate Metal 4 timelines, four sampled rounds after eight warm, show `i2_copy` 499 → 109 calls per
step and `i2_rms` 513 → 499 calls per step. The timeline reports dispatches, not grid dimensions. Counting
the depth rows in the executed schedule gives 332 ordinary mix rows plus 48 bank rows before, versus 96
current rows, eight published blocks and two pass outputs after: 380 → 106 rows. Total timeline dispatches
were 2,614 → 2,402; the baseline timeline omitted 192 `i2_tanh` stages present in the candidate timeline,
so that total is not an isolated count of the change. The candidate's RMS dispatches are smaller and more
frequent than the old batched depth normalizations; the indexed mix also costs more per call in this profile.
The profile is instrumented and does not establish which effect explains the end-to-end regression.

Reproduction (set `PATH=/opt/homebrew/bin:$PATH`, `TMPDIR=~/repos/tf/.work/tmp`,
and `ZIG_GLOBAL_CACHE_DIR=~/repos/tf/.work/zig-global-cache`):

```sh
H=~/repos/tf/campaign/heavy.py
M=/Users/josh/models/Inkling-2-Mini@ed40e7da625f
D=~/repos/tf/.work/tmp
B=~/repos/tf/zig-out/bin/tf-inkling2-loop-run
C=./zig-out/bin/tf-inkling2-loop-run
python3 "$H" -- zig build test-inkling2-loop -Dcpu=apple_m1
zig build tf-inkling2-loop-run -Dcpu=apple_m1
head -c 400 campaign/prompts/dev/code-1024.txt > "$D/d2-prompt.txt"
python3 "$H" -- "$C" "$M/tokenizer.json" "$D/d2-prompt.txt" tokenize "$D/d2-ids.json"
for n in greeting sql prose_review cells_code; do
  ids=~/repos/tf/.work/parity/inkling2-mini/metal/$n.tokens.json
  python3 "$H" -- "$B" "$M" "$ids" logits "$D/d2-$n-base.bin"
  python3 "$H" -- "$C" "$M" "$ids" logits "$D/d2-$n-cand.bin"
  cmp "$D/d2-$n-base.bin" "$D/d2-$n-cand.bin"
done
python3 "$H" -- "$C" "$M" ~/repos/tf/.work/parity/inkling2-mini/metal/greeting.tokens.json reset "$D/d2-reset.bin"
```

For each timeline and benchmark run, the heavy command launched a shell which ran the pre-gate, started
`$B` or `$C` with `"$M" "$D/d2-ids.json" bench 14 2` in the background, waited for it, and ran the
post-gate with that child's PID ignored. Timeline runs set `TF_METAL4_TIMELINE=4` and were reduced with
`python3 campaign/kernels/inkling_profile.py timeline --log <log>`. Timings exclude timeline instrumentation.

The measurement command was:

```sh
python3 "$H" -- bash "$D/d2-measure.sh"
```

The script ran the following sequence, retrying a failed gate after 20 seconds and stopping after one hour;
`timeline_base` and `timeline_cand` also set `TF_METAL4_TIMELINE=4`. Each log was written under `$D`.

```sh
for kind in timeline_base timeline_cand base cand cand base base cand cand base base cand; do
  bin=$B; [[ $kind == *cand* ]] && bin=$C
  gpu-gate-ready -1
  "$bin" "$M" "$D/d2-ids.json" bench 14 2 > "$D/d2-$kind.log" 2>&1 &
  runner_pid=$!
  wait "$runner_pid"
  GPU_GATE_IGNORE_PIDS=$runner_pid gpu-gate-ready -1
done
python3 campaign/kernels/inkling_profile.py timeline --log "$D/d2-timeline_base-kept.log"
python3 campaign/kernels/inkling_profile.py timeline --log "$D/d2-timeline_cand-kept.log"
```
