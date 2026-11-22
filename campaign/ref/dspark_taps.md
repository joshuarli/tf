# Inkling 2 Mini DSpark contract

The source references below are the MIT-licensed archives pinned in `campaign/sources.lock.json`: TokenSpeed
`tokenspeed@947b25bc92d7` and MLX-VLM `mlx-vlm@f1618d6d52be`. Line numbers refer to their extracted trees.

The target executes physical layers 0–5, layers 6–17 in three passes, then layers 18–23
(`mlx-vlm@f1618d6d52be/mlx_vlm/models/inkling2/config.py:74-83`). The checkpoint's four trained taps are
`[5, 23, 41, 48]`. Capture the raw partial entering each indexed visit after any loop rewind, not a completed
layer or a sum of the residual bank. Those intermediate taps enter layer 5, layer 11 in pass 1, and layer 17 in
pass 2; 48 is the final depth-mixed normalized output. Both implementations do this
(`tokenspeed@947b25bc92d7/python/tokenspeed/runtime/models/inkling2.py:640-671`,
`mlx-vlm@f1618d6d52be/mlx_vlm/models/inkling2/language.py:688-712`); TokenSpeed's test explicitly rejects adding
the prior embedding to the partial (`tokenspeed@947b25bc92d7/test/runtime/models/test_inkling2_capture.py:81-127`).

For each block, use the target's normalized token embedding at row 0 as the anchor and the learned mask embedding
for the remaining rows; the proposal rows are 1 through `block_size - 1`. The checkpoint's `block_size` is 8,
so one anchor precedes seven proposals (`tokenspeed@947b25bc92d7/python/tokenspeed/runtime/models/inkling2_mtp.py:680-690,715-733`).

The `rnn_hidden` Markov head has checkpoint `markov_rank` 256. It projects the previous token embedding, current
proposal hidden state, last accepted target features and recurrent state; gated recurrence produces a correction
for each sequentially sampled proposal (`tokenspeed@947b25bc92d7/python/tokenspeed/runtime/models/inkling2_mtp.py:303-318,338-352,437-495`).
Initialize the recurrent state to zero for every block and feed each sampled token into the next step; do not carry
that state across blocks (`tokenspeed@947b25bc92d7/python/tokenspeed/runtime/models/inkling2_mtp.py:449-467,491-495`).

After target verification, select the last accepted target row's captured features for the next round, mix the taps
for the recurrent head, and use the block anchor as its first previous token
(`tokenspeed@947b25bc92d7/python/tokenspeed/runtime/execution/drafter/inkling2.py:197-229,312-323`,
`tokenspeed@947b25bc92d7/python/tokenspeed/runtime/models/inkling2_mtp.py:618-629`). Verification, rollback and the
drafter itself remain future work; this document does not establish native numerical or GPU correctness.
