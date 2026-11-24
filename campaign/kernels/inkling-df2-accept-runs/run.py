"""DF2 acceptance runs: DFlash2 speculation on Inkling 2 Mini oQ4e, committed tokens a round and where drafts fail.

usage:
  run.py tails RUNNER OUT.json [--total 1500]
  run.py bench RUNNER PROMPTS.json NAME [--bits 4] [--tokens 512] [--drafts chain|lookup]
               [--context incremental|rebuilt|f32_taps|first128|first512|window8192]
  run.py accept NAME [NAME ...]
  run.py copies NAME PROMPTS.json [--gram 4]
  run.py rope

`tails` cuts each 4K dev prompt of DF1's `prompts.json` to its chat head, its instruction paragraph and its last
content tokens, `--total` tokens in all. `bench` runs `tf-inkling2-loop-run ... dflash` under campaign/heavy.py (the
GPU lock and the 28 GB cap) and writes NAME.jsonl here. Acceptance is deterministic; these runs take no timing gate,
so their round times are not measurements. `accept` prints committed tokens a round, per quarter of each reply and
per draft depth. `copies` sorts every draft the target judged by where its target token's `--gram`-gram (the
token and the ones before it) occurred earlier in the request: inside the drafter's window, only before it, or
nowhere, and prints the share each class accepted. `rope` compiles `rope.swift` and runs core/draft_ops.metal's
`tf_full_rope` on the GPU at positions up to 33,000, with Metal's fast and precise math, against float64 rotations,
and writes rope.txt.
"""

import argparse
import collections
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parents[1]
REPO = CAMPAIGN.parent
HEAVY = CAMPAIGN / "heavy.py"
DF1 = CAMPAIGN / "kernels/inkling-dflash2-runs"
TARGET = Path.home() / "models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd"
DRAFTER = Path.home() / "models/Inkling-2-Mini-DFlash2@4f9088ea66c2"
USER, TEXT = 200000, 200004


def tokenize(runner: Path, text: str) -> list[int]:
    scratch = REPO / ".work/tmp/df2-prompts"
    scratch.mkdir(parents=True, exist_ok=True)
    src, out = scratch / "text.txt", scratch / "ids.json"
    src.write_text(text)
    subprocess.run([str(runner), str(TARGET / "tokenizer.json"), str(src), "tokenize", str(out)], check=True,
                   capture_output=True)
    return json.loads(out.read_text())


def tails(args: argparse.Namespace) -> None:
    runner = Path(args.runner).resolve()
    manifest = []
    for p in json.loads((DF1 / "prompts.json").read_text()):
        if not p["name"].endswith("-4096"):
            continue
        ids = p["ids"]
        start = next(i for i in range(len(ids) - 1) if ids[i] == USER and ids[i + 1] == TEXT) + 2
        text = (CAMPAIGN / f"prompts/dev/{p['name']}.txt").read_text()
        instruction = tokenize(runner, text[:text.index("\n\n") + 2])
        if ids[start:start + len(instruction)] != instruction:
            raise SystemExit(f"{p['name']}: the instruction paragraph does not tokenize as the prompt's prefix")
        keep = args.total - start - len(instruction) - 1
        cut = ids[start:start + len(instruction)] + ids[-1 - keep:-1]
        manifest.append({"name": f"{p['name']}-tail{args.total}", "ids": ids[:start] + cut + [ids[-1]]})
    Path(args.out).write_text(json.dumps(manifest))
    print(f"{args.out}: " + ", ".join(f"{p['name']} {len(p['ids'])}" for p in manifest))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bench(args: argparse.Namespace) -> None:
    runner = Path(args.runner).resolve()
    out = HERE / f"{args.name}.jsonl"
    command = [str(runner), str(TARGET), str(Path(args.prompts).resolve()), "dflash", str(DRAFTER), str(args.tokens),
               str(out), str(args.bits), "entering", args.context, args.drafts]
    env = dict(os.environ, TMPDIR=str(REPO / ".work/tmp"), INKLING_LOAD_LIMIT_GB="26")
    (REPO / ".work/tmp").mkdir(parents=True, exist_ok=True)
    with (HERE / f"{args.name}.log").open("w") as log:
        code = subprocess.run(["python3", str(HEAVY), "--", *command], cwd=REPO, env=env, stdout=log,
                              stderr=subprocess.STDOUT).returncode
    if code != 0:
        raise SystemExit(f"{args.name}: exit {code}; see {args.name}.log")
    meta = {"kind": "binary", "runner": str(runner), "runner_sha256": sha256(runner), "target": str(TARGET),
            "drafter": str(DRAFTER), "prompts": str(Path(args.prompts).resolve().relative_to(REPO)),
            "prompts_sha256": sha256(Path(args.prompts)), "context": args.context, "drafts": args.drafts}
    out.write_text(json.dumps(meta) + "\n" + out.read_text())


def records(name: str) -> list[dict]:
    path = Path(name) if name.endswith(".jsonl") else HERE / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def accept(args: argparse.Namespace) -> None:
    print("| Run | Prompt | Equal | Rounds | Committed/round | Quarters | Accepted at depth 1..7 | Zero / full | "
          "Lookup rounds, accepted each | Drafter rounds' accepted each |")
    print("| --- | --- | --- | ---: | ---: | --- | --- | --- | --- | ---: |")
    for name in args.names:
        rows = records(name)
        rounds = collections.defaultdict(list)
        for r in rows:
            if r["kind"] == "round":
                rounds[r["prompt"]].append(r)
        for s in (r for r in rows if r["kind"] == "spec"):
            rs = rounds[s["prompt"]]
            emitted = sum(r["emitted"] for r in rs)
            quarters = [[0, 0] for _ in range(4)]
            done = 0
            for r in rs:
                q = quarters[min(3, done * 4 // emitted)]
                q[0] += r["emitted"]
                q[1] += 1
                done += r["emitted"]
            full = [r for r in rs if r["offered"] == 7]
            depth = []
            for d in range(1, 8):
                reach = sum(1 for r in full if r["accepted"] >= d - 1)
                depth.append(sum(1 for r in full if r["accepted"] >= d) / reach if reach else 0.0)
            zero = sum(1 for r in full if r["accepted"] == 0)
            whole = sum(1 for r in full if r["accepted"] == 7)
            looked = [r["accepted"] for r in rs if r.get("source") == "lookup"]
            drafted = [r["accepted"] for r in rs if r.get("source", "drafter") == "drafter"]
            lookups = f"{len(looked)}, {sum(looked) / len(looked):.2f}" if looked else "0"
            print(f"| {Path(name).stem} | {s['prompt']} | {'yes' if s['equal_plain'] else 'NO'} | {len(rs)} | "
                  f"{emitted / len(rs):.2f} | {' / '.join(f'{a / b:.2f}' for a, b in quarters)} | "
                  f"{' '.join(f'{v:.2f}' for v in depth)} | {zero} / {whole} | {lookups} | "
                  f"{sum(drafted) / len(drafted):.2f} |")


def copies(args: argparse.Namespace) -> None:
    window = 2047
    prompts = {p["name"]: p["ids"] for p in json.loads(Path(args.prompts).read_text())}
    rows = records(args.name)
    replies = {r["prompt"]: r["tokens"] for r in rows if r["kind"] == "plain"}
    print("| Prompt | In the window | Only before it | Nowhere |")
    print("| --- | ---: | ---: | ---: |")
    for s in (r for r in rows if r["kind"] == "spec"):
        seq = prompts[s["prompt"]] + replies[s["prompt"]]
        ends = collections.defaultdict(list)
        for t in range(args.gram - 1, len(seq)):
            ends[tuple(seq[t - args.gram + 1:t + 1])].append(t)
        counts = {c: [0, 0] for c in ("near", "far", "novel")}
        for r in rows:
            if r["kind"] != "round" or r["prompt"] != s["prompt"]:
                continue
            for k in range(1, min(r["accepted"] + 1, r["offered"]) + 1):
                t = r["position"] + k
                before = [j for j in ends[tuple(seq[t - args.gram + 1:t + 1])] if j < t]
                c = "novel" if not before else "near" if before[-1] >= t - window else "far"
                counts[c][0] += k <= r["accepted"]
                counts[c][1] += 1
        judged = sum(n for _, n in counts.values())
        print(f"| {s['prompt']} | " + " | ".join(
            f"{a}/{n} = {a / n:.2f} ({n / judged:.0%})" if n else "0/0" for a, n in counts.values()) + " |")


def rope(_: argparse.Namespace) -> None:
    binary = REPO / ".work/tmp/df2-rope"
    subprocess.run(["swiftc", "-O", "-o", str(binary), str(HERE / "rope.swift"), "-framework", "Metal"], check=True)
    kernels = REPO / "zig/kernels/metal/core/draft_ops.metal"
    with (HERE / "rope.txt").open("w") as out:
        for math in ("fast", "precise"):
            subprocess.run(["python3", str(HEAVY), "--", str(binary), str(kernels), math], cwd=REPO, check=True,
                           stdout=out, stderr=subprocess.STDOUT)
    print((HERE / "rope.txt").read_text(), end="")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("tails")
    a.add_argument("runner")
    a.add_argument("out")
    a.add_argument("--total", type=int, default=1500)
    a = sub.add_parser("bench")
    a.add_argument("runner")
    a.add_argument("prompts")
    a.add_argument("name")
    a.add_argument("--bits", type=int, default=4)
    a.add_argument("--tokens", type=int, default=512)
    a.add_argument("--context", default="incremental",
                   choices=["incremental", "rebuilt", "f32_taps", "first128", "first512", "window8192"])
    a.add_argument("--drafts", default="chain", choices=["chain", "lookup"])
    a = sub.add_parser("accept")
    a.add_argument("names", nargs="+")
    a = sub.add_parser("copies")
    a.add_argument("name")
    a.add_argument("prompts")
    a.add_argument("--gram", type=int, default=4)
    sub.add_parser("rope")
    args = p.parse_args()
    {"tails": tails, "bench": bench, "accept": accept, "copies": copies, "rope": rope}[args.action](args)


if __name__ == "__main__":
    sys.exit(main())
