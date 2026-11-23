# Colab experiments

This harness runs small CUDA and CPU experiments on Google Colab through the official
[`colab` CLI](https://github.com/googlecolab/google-colab-cli) (v0.8.0). Runs use the compute units of the
existing Colab subscription.

Use it for numerical and precision questions that do not depend on Metal. The M5 Max under Metal stays the only
performance platform for this campaign. CUDA timings are recorded as CUDA timings and are never Metal evidence.
TensorFold has no Inkling 2 Mini CUDA path.

## Verdict (2026-11-22): not needed for this campaign

The harness works. One T4 smoke used under 0.01 CU. The campaign does not use it, for four reasons:

- **The critical path is Metal-specific.** Staged bf16 expert weights for TensorOps (E4) and few-row verify
  overheads (V4) depend on Apple GPU behaviour, and CUDA timings cannot stand in for them.
- **Numerical studies are cheaper locally.** Operand-rounding error for fp16, bf16 and fp8 is a property of the
  formats, so the Mac's CPU computes it at no cost. F1's per-layer sensitivity needs activations captured by
  the Metal runner, and analysing those dumps is CPU work.
- **Full-model checks do not fit.** The f32 oracle needs about 104 GB. Parity, quality and Pi run against the
  native Metal engine. An L4 has 24 GB, and there is no Inkling CUDA path.
- **The limit is GPU timing, not machine count.** Only one model timing runs on the M5 at a time, and Colab
  cannot take that load.

Revisit Colab for any of these:
- large CUDA-portable sweeps, such as a tuned F2 fp8 activation-quantization study;
- work on TensorFold's CUDA engine, which the measured driver 580 and CUDA 13 satisfy;
- Linux x86_64 build checks with `env-probe`.

## Quickstart

```sh
uv tool install google-colab-cli                        # once, if `colab` is missing
gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
make -C campaign colab-probe                            # auth, balance, owned sessions; allocates nothing
make -C campaign colab-list                             # experiments and what each needs
make -C campaign colab-smoke GPU=T4 MAX_CU=0.25         # allocate, check, run, retrieve, release
make -C campaign colab-bench GPU=L4 NAMES=precision-error MAX_CU=0.5
make -C campaign colab-cleanup                          # stop any session this harness owns and left running
make -C campaign colab-test                             # the mocked-CLI tests, no network
```

`GPU` takes `CPU`, `T4` or `L4`. A100, H100 and G4 are refused because they need the owner's approval.
`MAX_CU` is required on every allocating target.

## What a run does

The harness calls the CLI with `--auth=adc`, and with `--config .work/colab/sessions.json` so that it sees only
the sessions it created. Each run follows these steps:

1. **Admit.** Read the balance with `colab usage`. Refuse before any allocation if any of these holds:
   - authentication is missing;
   - the accelerator is unknown or needs approval;
   - an experiment needs a newer SM than the accelerator has;
   - the selection is empty or unknown;
   - the estimate exceeds `MAX_CU` or the balance.

   The estimate is the ceiling rate multiplied by the deadline. The deadline is 180 s of setup plus each
   experiment's declared time. The CLI publishes no price list, so `colab.py` holds conservative ceilings: CPU
   0.2, T4 2.0 and L4 5.0 CU/h.
2. **Allocate one runtime.** It is named `tf-<gpu>-<random>` and recorded in `.work/colab/owned.json`. Then read
   the live rate. If the measured increase is above the ceiling, the runtime is stopped and nothing runs, so the
   budget argument holds.
3. **Run the named experiments in order in that one runtime**, sharing its installed packages and caches.
   - Each script is composed with `remote/common.py`, and the composed copy is kept beside the receipt.
   - Each runs under `colab exec` with the remaining deadline as its timeout.
   - An experiment marked as using source gets an upload of `build.zig`, `build.zig.zon` and `zig/` from the
     working tree, uncommitted changes included. The upload and its sha256 fingerprint cover exactly those
     files. The remote side extracts the tarball only when its recorded fingerprint differs.
   - A result whose fingerprint differs from the local one is rejected, so stale remote source is never
     benchmarked.
4. **Check each result.** Download each `result.json` before teardown. A result is an error if it is missing,
   if it reports a GPU other than the requested one, or if its exec failed.
5. **Release the runtime** with `colab stop` in a `finally`. This runs on success, error, deadline, Ctrl-C and
   SIGTERM. A failed stop keeps the session in `owned.json` and names `colab-cleanup` in the receipt.
6. **Record consumption** as the balance before minus the balance after. The balance has 0.01 CU resolution.

Nothing buys compute units, keeps a runtime alive between invocations, or sends keepalive traffic. Model
downloads are not part of any experiment.

Each run writes `.work/colab/<UTC stamp>-<session>/`. It holds `receipt.json`, which records:
- the session, accelerator, budget, estimate, live rate, balances, consumed compute units and wall time;
- the commit, the dirty flag and the source fingerprint;
- the result of each experiment, including the runtime's GPU, compute capability, VRAM, RAM, driver, nvcc,
  torch, shapes, precision, timing method and correctness against fp64;
- exclusions with their reasons.

The directory also holds a log and the composed script for each experiment. `.work/` is scratch: copy
conclusions into a tracked record.

## Experiments

| Name | Runs on | Uses source | What it answers |
| --- | --- | --- | --- |
| `smoke` | any runtime; CPU reports no GPU | no | The requested GPU is present, plus a 64×2304×2304 f32 GEMM (TF32 off) against fp64 with the `fp4_ref.zig` sumBound. |
| `env-probe` | any | yes | Driver against R580 and CUDA 13; installs Zig 0.17.1 Linux x86_64, checks its sha256 against `bootstrap.lock.json`, and runs `zig build -l` on the uploaded tree. |
| `precision-error` | sm_75+ | no | Operand-precision error at Inkling expert shapes. Weights are exact in every format. Arms: f32, fp16, bf16 (sm_80+), and fp8 E4M3 with per-tensor scale (sm_89+). Arms the device cannot run are recorded as exclusions. |

## Tested results

The mocked-CLI suite (`make -C campaign colab-test`, 19 tests) covers:
- refusals before allocation: approval-only GPUs, SM too low, unknown selections, budget and balance;
- missing authentication;
- a quota refusal at allocation;
- a live rate above the ceiling;
- a failed experiment, a missing artifact and the wrong GPU;
- interruption and deadline;
- stale source;
- owned-only cleanup and a failed stop;
- the dirty-tree fingerprint.

One real T4 smoke ran on 2026-11-22: `make -C campaign colab-smoke GPU=T4 MAX_CU=0.25`.

| | Result |
| --- | --- |
| Exit and wall time | 0, in 18.5 s |
| Estimate against measured | 0.167 CU estimated; the live rate measured 1.15 CU/h |
| Balance | 406.02 before and after, so under 0.01 CU at the balance's resolution |
| Runtime | Tesla T4, sm_75, 14.6 GB VRAM, 12.7 GB RAM, 2 CPUs |
| Software | driver 580.82.07, nvcc CUDA 13.0, torch 2.14.1+cu130 |
| GEMM | 0.327 ms median, worst error 3.1e-4 of the bound, none outside it |
| Release | released, with 0 active assignments afterward |

`env-probe` and `precision-error` have not run on Colab yet.

## Which experiments suit Colab

- **CPU-portable** (cheaper locally, and often fine on a Colab CPU runtime): the evidence fixtures, quality
  and prompt-corpus tests, and operand-rounding analysis. With exact weights, the operand-rounding error is a
  property of the formats, not the device.
- **CUDA-portable** (insight transfers, timings do not):
  - fp16, bf16 and fp8 operand error at expert shapes, which bears on the F1 and F2 fast-precision lanes;
  - E4M3 activation-quantization error with per-tensor, per-row and per-block scales;
  - accumulation-order sensitivity.
- **Metal-only**: every performance question (prefill, verify, DSpark timing, TensorOps), the native parity
  runs, served benches, quality and Pi. These stay on the M5 Max.
- **Full checkpoint**: the quality and Pi suites, and per-layer sensitivity on real activations. Running the
  20.2 GB checkpoint on CUDA would be a separate campaign. It needs a CUDA Inkling path or a torch reference
  sized to an L4's 24 GB, and an explicit download opt-in. It is not attempted here.

## T4 or L4

- **T4 by default.** It measured 1.15 CU/h. It covers the f32 and fp16 numerics and the harness itself.
- **L4 when the question needs bf16 or fp8.** Its sm_89 is the cheapest SM that runs all four
  `precision-error` arms. That answers F2's E4M3 question directly, and the T4 cannot. The L4's live rate has
  not been measured; the harness refuses it above 5.0 CU/h.
- **Both are irrelevant to TensorFold's CUDA build here.** The T4 is below its sm_80 floor. The L4 is only a
  source-build target, and no Inkling 2 Mini CUDA kernels exist for it.

## Blockers

- **No Inkling 2 Mini CUDA path.** Full-model CUDA inference is a separate campaign.
- **Consumption below 0.01 CU is not resolved.** Short runs show 0.00 consumed; the live rate is the better
  measure for them.
- **Default auth.** The CLI's default oauth2 flow prompts for a browser code, so this harness requires ADC.
