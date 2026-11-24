"""buckets.py ROWS TABLE...: the split-timeline tables (inkling_profile.py timeline --split) in P3c's buckets, ms per 512 rows"""
import re, sys
BUCKETS = [
    ("Routed NVFP4 matrices", r"^(i2p_expert_units_(up|down)|i2pu_units_(up|down)|i2pu_wide_(up|down)|i2pk_units_(up|down)|i2p_expert_matrix_wide|i2p_expert_matrix)$"),
    ("Dense projections (staging, split, GEMM)", r"^(i2as_|i2ps_)"),
    ("f32 projections (router, controllers)", r"^(i2af_gemm|i2af_gate_ctl)$"),
    ("Grouped low-occupancy experts", r"^(i2p_gu_grouped|i2p_down_grouped|i2pg_gu|i2pg_down)$"),
    ("Attention", r"^(i2p_attn_units|i2pa_attn_units64|i2p_attn_half|i2p_attn_tiled|i2p_attn)$"),
    ("Affine-4 experts", r"^(i2p_affine_units_(up|down)|i2pu_affine_(up|down)|i2pu_wide_affine_(up|down)|i2pk_affine_(up|down)|i2a_half_group_(up|down))$"),
    ("Relative bias", r"^i2p_relbias"),
]
rows = float(sys.argv[1])
tables = [a for a in sys.argv[2:] if not a.startswith("--")]
cols = []
for path in tables:
    text = open(path).read()
    kernels = {}
    span = None
    for line in text.splitlines():
        m = re.match(r"^(\d+) segments, (\d+) dispatches, ([\d.]+) ms summed GPU span", line)
        if m:
            span = float(m.group(3)); dispatches = int(m.group(2))
        m = re.match(r"^(\S+)\s+([\d.]+)\s+(\d+)$", line)
        if m:
            kernels[m.group(1)] = float(m.group(2))
    sums = {name: 0.0 for name, _ in BUCKETS}
    sums["Everything else"] = 0.0
    for k, ms in kernels.items():
        for name, pat in BUCKETS:
            if re.search(pat, k):
                sums[name] += ms
                break
        else:
            sums["Everything else"] += ms
    sums["Summed span"] = span or sum(kernels.values())
    cols.append((path, sums, kernels))
scale = 512.0 / rows
names = [n for n, _ in BUCKETS] + ["Everything else", "Summed span"]
print("| Bucket | " + " | ".join(p.split("/")[-1].replace(".table", "") for p, _, _ in cols) + " |")
print("| --- |" + " ---: |" * len(cols))
for n in names:
    print(f"| {n} | " + " | ".join(f"{s[n] * scale:.1f}" for _, s, _ in cols) + " |")
if "--kernels" in sys.argv:
    every = {}
    for i, (_, _, kernels) in enumerate(cols):
        for k, ms in kernels.items():
            if not any(re.search(pat, k) for _, pat in BUCKETS):
                every.setdefault(k, [0.0] * len(cols))[i] = ms
    print()
    print("| Everything else, by kernel | " + " | ".join(p.split("/")[-1].replace(".table", "") for p, _, _ in cols) + " |")
    print("| --- |" + " ---: |" * len(cols))
    for k, v in sorted(every.items(), key=lambda kv: -max(kv[1]))[:24]:
        print(f"| `{k}` | " + " | ".join(f"{x * scale:.2f}" for x in v) + " |")
