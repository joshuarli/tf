# Inkling 2 Mini oQ4e

Status: blocked by the lane's writable scope. No implementation or model measurements were completed.

The requested affine dense and expert Metal operators need kernel contract registrations. The native contributor
instructions require registration in `zig/tests/contract/metal_main.zig` and launch records in
`zig/src/core/contract/registry.zig` (`docs/recipes/SKILL.md:80-88`). Both paths are outside the lane's writable
paths. `docs/kernel-contract.md:151-160` also requires a contract test type under `zig/tests/contract/`.
The required family-local FP64 test does not replace those registrations. The parent must coordinate those changes
or expand this lane's scope before the operators can satisfy the repository's correctness requirements.

The checkpoint's four `model-*.safetensors` headers contain 1,102 tensors in total. Layer 0's attention uses
`model.layers.0.attn.in_proj.weight` with shape `[8600, 576]` and dtype U32, plus BF16 scales and biases.
Its NVFP4 `w13.weight` has shape `[128, 2304, 288]` and dtype U32, with U8 scales and F32 global scales.
Layer 20's `w13.weight` has the same U32 shape but BF16 scales and biases for affine quantization.
Layer 5 uses separate `q_proj`, `k_proj`, and `v_proj` tensors. The existing loader admits one shard and only
ModelOpt quantization metadata (`zig/src/families/inkling2_loop/config.zig:119-138,276-290`). Its tensor map
expects the `model.llm` namespace and separate BF16 projections (`zig/src/families/inkling2_loop/model.zig:185-195,253-256,334-340`).

The native regression test, affine FP64 test, greedy generation, fidelity comparisons, throughput measurements,
batchbench, and peak charged memory are unverified. No GPU workload was started.
