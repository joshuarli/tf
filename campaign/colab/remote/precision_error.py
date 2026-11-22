# Runs after common.py in one composed file; runtime() and write() come from it.
import numpy as np

result = {"experiment": "precision-error", "runtime": runtime(), "status": "error", "arms": [], "exclusions": []}
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)


def weights(rng, k, n):
    """NVFP4-shaped weights: E2M1 values with a power-of-two scale per 16-value block along K, so every weight is
    exact in fp16, bf16 and E4M3 and only the activation operand's precision differs between arms."""
    values = E2M1[rng.integers(0, 8, (k, n))] * rng.choice([-1, 1], (k, n))
    scales = 2.0 ** rng.integers(-8, -3, (k // 16, n)).repeat(16, axis=0)
    return (values * scales).astype(np.float32)


try:
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    props = torch.cuda.get_device_properties(0)
    sm = props.major * 10 + props.minor
    result["device"] = {"name": props.name, "sm": sm, "torch": torch.__version__}
    rng = np.random.default_rng(0)
    for k in (2304, 1152):
        w = weights(rng, k, 2304)
        for rows in (32, 64):
            x = (rng.standard_normal((rows, k)) * np.exp(rng.standard_normal((rows, 1)))).astype(np.float32)
            exact = x.astype(np.float64) @ w.astype(np.float64)
            bound = 4 * (k + 64) * 2.0**-24 * (np.abs(x).astype(np.float64) @ np.abs(w).astype(np.float64)) \
                + 2 * 2.0**-24 * np.abs(exact)
            xd, wd = torch.from_numpy(x).cuda(), torch.from_numpy(w).cuda()
            arms = {"f32": (0, lambda: xd @ wd), "fp16": (70, lambda: (xd.half() @ wd.half()).float()),
                    "bf16": (80, lambda: (xd.bfloat16() @ wd.bfloat16()).float())}
            if hasattr(torch, "_scaled_mm"):
                def fp8():
                    scale = xd.abs().amax().clamp(min=1e-30) / 448.0
                    xq = (xd / scale).to(torch.float8_e4m3fn)
                    wq = wd.to(torch.float8_e4m3fn).t().contiguous().t()
                    one = torch.ones((), device="cuda")
                    return torch._scaled_mm(xq, wq, scale_a=scale.reshape(()), scale_b=one, out_dtype=torch.float32)
                arms["fp8-e4m3-per-tensor"] = (89, fp8)
            for arm, (need, fn) in arms.items():
                cell = {"arm": arm, "k": k, "n": 2304, "rows": rows}
                if sm < need:
                    result["exclusions"].append({**cell, "reason": f"needs sm_{need}, device is sm_{sm}"})
                    continue
                for _ in range(3):
                    y = fn()
                torch.cuda.synchronize()
                times = []
                for _ in range(20):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    y = fn()
                    end.record()
                    torch.cuda.synchronize()
                    times.append(start.elapsed_time(end))
                error = np.abs(y.cpu().numpy().astype(np.float64) - exact)
                cell.update(max_abs=float(error.max()), max_rel=float((error / np.maximum(np.abs(exact), 1)).max()),
                            outside_f32_bound=int((error > bound).sum()), results=int(error.size),
                            median_ms=float(np.median(times)), min_ms=float(min(times)), max_ms=float(max(times)))
                result["arms"].append(cell)
    result["method"] = ("Weights exact in every operand format; error is against fp64 of the f32 activations, so it "
                        "isolates activation-operand rounding plus device accumulation. fp16 and bf16 arms produce "
                        "half-width outputs before the float cast. Timing: 3 warmups, 20 CUDA-event samples; CUDA "
                        "numbers, not Metal evidence.")
    result["status"] = "ok"
except Exception as error:
    result["reason"] = f"{type(error).__name__}: {error}"
write(result)
