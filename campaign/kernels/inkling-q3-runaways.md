# Q3: Long-Reasoning Runaways Trace (`code-c02`, `code-c13`, `retrieval-t01`)

## Summary

Traced the full untruncated `reasoning_content` and `content` trajectories of `code-c02`, `code-c13`, and `retrieval-t01` under both default half prefill (`INKLING_PREFILL_HALF` unset) and exact prefill (`INKLING_PREFILL_HALF=off`) using `tensorfold-native` (`80ac21904`, `max_tokens=8192`, `temperature=0`, `seed=1234`, drafter loaded).

**Verdict:** No TensorFold numerical or implementation defect was found. `retrieval-t01` under half prefill is not a repetition loop at all (it is coherent, non-repeating exhaustive reasoning that runs out of the 8,192-token cap), while `code-c02` (`off`) and `code-c13` (both `half` and `off`) enter standard greedy (`temperature=0`) repetition attractors of the `oQ4e` checkpoint after matching cleanly for hundreds to thousands of characters.

## Full-Trajectory Results

| Case | Mode | Completion tokens | Finish | Reasoning chars | Content chars | Max 12-gram count in tail | Cause |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| `code-c02` | `half` (default) | 1,561 | `stop` | 4,112 | 863 | 1 | Passes cleanly (`11.6 s`) |
| `code-c02` | `off` (exact) | 8,192 | `length` | 17,015 | 0 | — (`'o'` char run) | Diverges at char 127; enumerates `3a4b...14l` then enters `'o'` repetition attractor |
| `code-c13` | `half` (default) | 8,192 | `length` | 27,365 | 0 | 44 | Agrees with `off` for 1,123 chars; repeats `'Should we allow "1/2" with internal newline? strip handles.'` |
| `code-c13` | `off` (exact) | 8,192 | `length` | 19,595 | 0 | 78 | Agrees with `half` for 1,123 chars; repeats `'What about "1/2/3"? error. Good.'` |
| `retrieval-t01` | `half` (default) | 8,192 | `length` | 26,125 | 0 | 1 | Finds needle `QX-7419` in first ~100 tokens; writes exhaustive non-repeating excerpt-by-excerpt technical review through Excerpt 5 until hitting 8,192 cap |
| `retrieval-t01` | `off` (exact) | 4,066 | `stop` | 12,116 | 2,683 | 1 | Finds needle `QX-7419`, chooses `"compact review"` at char 433, finishes cleanly |

## Detailed Findings

1. **`retrieval-t01` is greedy verbosity, not a loop or numeric bug:** Both `half` and `off` emit the exact same first 433 characters, including finding the needle (`"So the answer is QX-7419. Need to end with [[QX-7419]]."`). At char 433, `half` branches into `"But wait — the user also asked for a technical review of the excerpts."` and produces 26,125 characters of coherent, non-repeating Zig code review (max 12-gram count = 1 in the last 400 words) before hitting `max_tokens=8192`. `off` branches into `"But the larger prompt asks for a technical review of excerpts... I'll do a compact review"` and finishes at 4,066 tokens.
2. **`code-c13` is an intrinsic greedy attractor under both prefill modes:** `half` and `off` produce identical output for the first 1,123 characters before a 1-space formatting difference (`Fraction(1,2)` vs `Fraction(1, 2)`), after which both paths independently enter classic greedy repetition loops inside `<think>` (`44×` and `78×` exact 12-gram repeats).
3. **`code-c02` passes under the default (`half`) prefill:** Under `off`, it diverges at char 127 and falls into a single-character `'o'` repetition loop.
4. Combined with the passing `inkling-omlx-oracle` gate (`2,856/3,015 = 94.73%` top-1, mean KL `0.0199132`) and PH2/Q4 full-suite net runaways (`2` under `half` vs `3` under `off`), no target-engine fix is warranted.
