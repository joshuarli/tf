# Inkling 2 Mini oQ4e

Status: checkpoint loading and the family-local affine tests pass. Gated throughput and oMLX ID comparisons are pending.

The new affine kernels join the family-wide kernel-contract registration item required before promotion out of the
campaign. This lane uses family-local FP64 tests and parity under the campaign's explicit ruling.

The checkpoint's four `model-*.safetensors` headers contain 1,102 tensors in total. Layer 0's attention uses
`model.layers.0.attn.in_proj.weight` with shape `[8600, 576]` and dtype U32, plus BF16 scales and biases.
Its NVFP4 `w13.weight` has shape `[128, 2304, 288]` and dtype U32, with U8 scales and F32 global scales.
Layer 20's `w13.weight` has the same U32 shape but BF16 scales and biases for affine quantization.
Layer 5 uses separate `q_proj`, `k_proj`, and `v_proj` tensors. The current tensor map still expects
ModelOpt tensor names. Sliding `in_proj` rows are q, k, v, relative, gate, then sink, with widths
3072, 1024, 1024, 384, 3072, and 24. Global `q_proj` rows are q, relative, gate, then sink.
Global `k_proj` and `v_proj` each append 32 dynamic convolution rows to 1024 key or value rows.
Loop passes can supply distinct q, k, and v inputs, so those slices must be projected separately on those passes.

The 16 saved histories produced 3,015 rows with no non-finite logits. Overall top-1 agreement with the f32 oracle
was 2,763 of 3,015 positions (91.64%). At an f32 top-two margin of at least 0.1, agreement was 2,681 of 2,862
positions (93.68%). Mean KL, averaged across histories, was 0.03935111. Both long histories exceeded 1,024 tokens.
The standard f32-versus-f32 parity command exited 2 because the checkpoint weights differ. Its existing exception
labels do not establish the cause of differences between these checkpoints.

The oracle check verified 16 histories. The captures used `RUNNER_MODE=prefill`, `PREFILL_CHUNK=512`, f32
activations, the oQ4e checkpoint and the locally built runner. The initial comparator was stopped because it began
optional traces outside this lane. The rerun used `REF_VENV="''"` to disable those traces and reused the 16 captures.
The fidelity values above came from a separate comparison streamed eight rows at a time.

With the native checkpoint in `TF_INKLING2_LOOP_DIR`, `python3 campaign/heavy.py -- zig build test-inkling2-loop
-Dcpu=apple_m1 -j1` exited 0. With `TF_INKLING2_LOOP_OQ4E_DIR` set to the oQ4e checkpoint, the focused
`python3 campaign/heavy.py -- zig build test-inkling2-loop-affine -Dcpu=apple_m1 -j1` exited 0. The latter checks
8-bit dense and 4-bit expert projections against dequantized FP64 weights and checks finite oQ4e decode logits.
The aligned oQ4e one-token probe peaked at 20.6 GB charged under the 28 GB cap. This is not a peak for the
remaining long-context and batch measurements.
