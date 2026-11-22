# Runs after common.py in one composed file; runtime() and write() come from it.
import time

import numpy as np

result = {"experiment": "smoke", "runtime": runtime(), "status": "error"}
try:
    import torch
    result["torch"] = torch.__version__
    result["cuda_available"] = torch.cuda.is_available()
    if not torch.cuda.is_available():
        raise RuntimeError("torch sees no CUDA device")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    props = torch.cuda.get_device_properties(0)
    result["device"] = {"name": props.name, "sm": f"{props.major}{props.minor}",
                        "vram_gb": round(props.total_memory / 2**30, 1)}
    k, rows, n = 2304, 64, 2304
    rng = np.random.default_rng(0)
    x = rng.standard_normal((rows, k)).astype(np.float32)
    w = rng.standard_normal((k, n)).astype(np.float32)
    reference = x.astype(np.float64) @ w.astype(np.float64)
    bound = 4 * (k + 64) * 2.0**-24 * (np.abs(x).astype(np.float64) @ np.abs(w).astype(np.float64)) \
        + 2 * 2.0**-24 * np.abs(reference)
    xd, wd = torch.from_numpy(x).cuda(), torch.from_numpy(w).cuda()
    for _ in range(3):
        y = xd @ wd
    torch.cuda.synchronize()
    times = []
    for _ in range(20):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        y = xd @ wd
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    error = np.abs(y.cpu().numpy().astype(np.float64) - reference)
    result.update(
        shape={"rows": rows, "k": k, "n": n}, precision="f32 operands, f32 accumulate, TF32 off",
        timing={"warmup": 3, "samples": 20, "median_ms": float(np.median(times)), "min_ms": float(min(times)),
                "max_ms": float(max(times)), "method": "CUDA events around one matmul; CUDA timing, not Metal evidence"},
        correctness={"reference": "fp64 numpy", "bound": "4*(K+64)*2^-24*sum|x*w| + 2*2^-24*|y| (fp4_ref.zig sumBound)",
                     "max_abs": float(error.max()), "worst_error_over_bound": float((error / bound).max()),
                     "outside_bound": int((error > bound).sum())})
    result["status"] = "ok" if result["correctness"]["outside_bound"] == 0 else "error"
except Exception as error:
    result["reason"] = f"{type(error).__name__}: {error}"
write(result)
