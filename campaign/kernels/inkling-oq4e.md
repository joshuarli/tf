# Inkling 2 Mini oQ4e

Status: implementation in progress. No model measurements have been completed.

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

The native regression test, affine FP64 test, greedy generation, fidelity comparisons, throughput measurements,
batchbench, and peak charged memory are unverified. No GPU workload was started.
