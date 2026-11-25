"""buckets.py LOG...: one profiled prefill chunk's GPU time by kernel class, from split Metal 4 timeline logs.

Each LOG comes from `tf-inkling2-prefix-check profile` with TF_INKLING_PREFILL_PROFILE_BASE at the chunk's base: that
chunk's layers are rounds of their own (inkling2-slot-N, pass copies, head, vocabulary). Rounds named inkling2-prefill
are whole unprofiled chunks before the base (and the profiled chunk's embedding), so they are left out. Times are ms a
chunk: the timeline's ms a round, summed over the chunk's rounds.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling_profile import TIMELINE_HEADER, parse_timeline  # noqa: E402

CLASSES = [
    ("attention", r"^(i2p_attn|i2pa_attn|i2p_relbias|i2p_rel_rows)"),
    ("experts", r"^(i2p_expert|i2pu_|i2pk_|i2pg_|i2p_gu|i2p_down|i2p_affine_units|i2a_half_group|i2a_group)"),
    ("routing", r"^(i2p_route|i2p_router|i2p_pick|i2p_moe_|i2p_combine)"),
    ("dense", r"^(i2as_|i2ps_|i2af_|i2p_dense|i2p_gemm|i2p_cap|i2p_small|i2_head|i2p_head|q8_head|i2_unembed)"),
]
SPLIT = re.compile(r"(?=^(?:info: )?metal 4 timeline: \S+, \d+ rounds after)", re.MULTILINE)


def classify(name: str) -> str:
    for label, pattern in CLASSES:
        if re.search(pattern, name):
            return label
    return "glue"


def chunk(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    kernels: dict[str, float] = {}
    for block in SPLIT.split(path.read_text(errors="replace"))[1:]:
        header = TIMELINE_HEADER.search(block)
        if header is None or header.group(1) == "inkling2-prefill":
            continue
        for kernel in parse_timeline(block)["kernels"]:
            kernels[kernel["name"]] = kernels.get(kernel["name"], 0.0) + kernel["ms"]
    classes: dict[str, float] = {}
    for name, ms in kernels.items():
        classes[classify(name)] = classes.get(classify(name), 0.0) + ms
    return classes, kernels


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    columns = [chunk(p) for p in paths]
    labels = [label for label, _ in CLASSES] + ["glue"]
    print("| ms a 1,024-row chunk | " + " | ".join(p.stem for p in paths) + " |")
    print("| --- |" + " ---: |" * len(paths))
    for label in labels:
        print(f"| {label} | " + " | ".join(f"{c.get(label, 0.0):.1f}" for c, _ in columns) + " |")
    print("| total | " + " | ".join(f"{sum(c.values()):.1f}" for c, _ in columns) + " |")
    print()
    every: dict[str, list[float]] = {}
    for i, (_, kernels) in enumerate(columns):
        for name, ms in kernels.items():
            every.setdefault(name, [0.0] * len(columns))[i] = ms
    print("| kernel | class | " + " | ".join(p.stem for p in paths) + " |")
    print("| --- | --- |" + " ---: |" * len(paths))
    for name, values in sorted(every.items(), key=lambda item: -max(item[1])):
        print(f"| `{name}` | {classify(name)} | " + " | ".join(f"{v:.2f}" for v in values) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
