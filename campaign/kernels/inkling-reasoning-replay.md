# Inkling 2 Mini: replaying earlier reasoning in multi-turn chats

**Status: not run.** The harness is built and tested, but no model reply was collected.
Every family and effort level below is not run, and this receipt makes no claim about whether stripping earlier reasoning changes answer quality.

## The question

A post claimed that a harness that does not send Inkling's thinking back makes the model "completely useless", scoring "lower than a 9B model", and that the official card warns about it.
The campaign had already established four points before this lane started. The Hugging Face card for `thinkingmachines/Inkling-2-Mini` has no such warning; it mentions degraded performance in long multi-turn conversations, and that minimum reasoning effort does not fully disable reasoning.
The checkpoint template renders an assistant message's `reasoning_content` as a `<|content_thinking|>` block. The TensorFold server renders it, and the fx harness replays it.
The experiment measures what a client loses when it sends earlier turns back without that reasoning.

## Design

`campaign/quality/reasoning_replay.py` runs each conversation in two arms under greedy decoding:

- **replay** sends each earlier assistant turn back as `{"reasoning_content": ..., "content": ...}`, the message the fx workload clients build;
- **stripped** sends the same turns as `{"content": ...}` only.

The client memoises replies by request body. Both arms therefore share one turn-1 request, and each arm builds its later turns from its own replies.
Three families of 40 conversations have five turns each and answers that code checks:

- **chain** carries a running value through the fx workload's `transform_i` functions, one transform per turn. Each reply is `Value: N`, so stripping removes no needed information. Scoring separates absolute accuracy from step accuracy against the arm's own previous answer.
- **recall** asks for terse results of multi-step transforms, then for intermediate values that only the model's reasoning held.
- **code** has eight Python projects in five variants each. Each turn adds one requirement, and the turn's cumulative tests run in the macOS sandbox of `campaign/quality/sandbox.py`.

`score` reports per-turn accuracy for both arms and McNemar's exact test on each turn's discordant pairs.
It also reports a 10,000-draw conversation bootstrap of the turn 2 to 5 difference, with a project-cluster interval for the code family. It adds a sign test, reasoning and answer token counts, truncations, context refusals and paired failures.
As a check on the replay arm, it compares the turn-2 replay prompt with the stripped prompt and the turn-1 reasoning token count, to show that the server rendered the replayed reasoning.

## Results

| Family | `minimal` (effort 0.1) | `high` (effort 0.9) |
|---|---|---|
| chain | not run | not run |
| recall | not run | not run |
| code | not run | not run |

No accuracy, paired difference, token count, truncation count or determinism recheck exists.

## Why nothing ran

All model work runs under `campaign/heavy.py`, which serialises the GPU across lanes. About a dozen lanes were queued on the lock throughout.

1. A pilot queued at 13:54 was withdrawn before it acquired the lock and folded into a full minimal-effort run.
2. That run acquired the lock at 14:42. The server refused to start in about 20 s:
   `--parallel 4 --context 32768: Inkling needs 27413676032 bytes including caches and reserves under a 26000000000-byte load limit; each session fits at most 21982 tokens`.
   With four sessions under `INKLING_LOAD_LIMIT_GB=26`, the campaign's usual 32768-token context does not fit. The harness now defaults to `--context 21504`, and it records a prompt that this window refuses as a failed turn.
3. The run was re-queued, withdrawn at 15:01 while still waiting to add a per-effort deadline, then re-queued with a separate high-effort run.
4. Both jobs were cancelled at 15:09, still waiting for the lock, when the campaign wound down. Neither started a server.

Total server time was the one refused start.

## What was verified

- `zig build native -Dcpu=apple_m1 -j1` at `69651393c` exited 0.
- `python3 -I campaign/quality/test_reasoning_replay.py` passes 19 tests. They cover task generation, both arms' histories, the request memo, the deadline and family interleaving, context refusals and retries, value parsing, chain step scoring, McNemar and bootstrap values, and a scripted end-to-end score.
- In that test, every one of the 40 code conversations has turn tests that pass that turn's reference solution and reject the previous turn's, in process and in the sandbox for each project's first variant. Replies that exit early, loop, crash or omit code fail.
- `make -C campaign -f tests/tests.mk evidence-tests`, which now runs the new test, exits 0.
- From the source: the template quotes both `"minimal"` and `"high"` in its effort map, so `coerceEffort` in `zig/src/server/fields.zig` passes both names through unchanged. The template maps them to 0.1 and 0.9.

## Reproduction

From the worktree, with the checkpoint installed. Each `run` holds the GPU lock for its whole duration and resumes from `$OUT/responses.jsonl`.

```sh
export PATH=/opt/homebrew/bin:$PATH TMPDIR=$HOME/repos/tf/.work/tmp/q2/tmp ZIG_GLOBAL_CACHE_DIR=$HOME/repos/tf/.work/zig-global-cache
zig build native -Dcpu=apple_m1 -j1
OUT=$HOME/repos/tf/.work/tmp/q2/main
RUN="python3 -I campaign/quality/reasoning_replay.py run --out $OUT --serve $PWD/zig-out/native/bin/tensorfold-native --model-dir $HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd --context 21504 --max-tokens minimal=2048,high=4096 --conversations 40"
INKLING_LOAD_LIMIT_GB=26 python3 campaign/heavy.py -- $RUN --efforts minimal --deadline-minutes 40 --recheck 8
INKLING_LOAD_LIMIT_GB=26 python3 campaign/heavy.py -- $RUN --efforts high --deadline-minutes 65 --recheck 4
python3 -I campaign/quality/reasoning_replay.py score --out $OUT --max-tokens minimal=2048,high=4096 --partial
python3 -I campaign/quality/test_reasoning_replay.py
```

The reply lengths are unmeasured, so the two deadlines are estimates that keep the total near two hours of server time. A deadline stops new conversations, not the ones in flight.
Families alternate in the queue, so a cut leaves them at about the same count, and `score --partial` scores the conversations whose arms both completed.
