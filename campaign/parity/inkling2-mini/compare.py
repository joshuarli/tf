"""Compare the Metal teacher-forced pass with the float32 reference on the parity histories.

Each history is judged by the declared numerical contract in
``campaign/ref/gate_policy.json`` through ``campaign/ref/logit_gate.py``: the
expected token/row counts from histories.json, the vocabulary shape,
finiteness, and the contract bounds. A history that fails any of those makes
the overall verdict fail. Failing histories fall into three explicit
categories: within contract; failing with a recorded, evidence-backed
explanation from gate_exceptions.json (which never turns a failure into a
pass); failing unexplained. The verdict is "qualified" only when every history
is within contract.

Candidate outputs are accepted only with a matching provenance receipt (see
provenance.py): the runner binary sha256, the model identity, the
token-history sha256 and the arithmetic policy must name the current inputs,
or the output is refused. Reference outputs are the saved oracle files,
verified by oracle.py; their observed digests are recorded in the report.

For each history whose error crosses ``--tensor-tol``, the first divergent
tensor at the onset position is found by tracing the onset position's prefix
on both sides: the reference's ``--trace-row`` writes one summary line per
tensor for that position's row alone in execution order, the runner's
``trace <out> <at>`` folds only that position, and the first overlapping name
(the slot-indexed mixin, normin, attn_out, ffn_inp and l_out tensors, in
execution order) whose summary differs by more than the tolerance is the
first tensor to diverge. The onset-1 prefix is traced the same way as the
clean control. Both traces run inside the campaign's memory ceiling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "ref"))

import provenance  # noqa: E402
from logit_gate import check as gate_check  # noqa: E402
from logit_gate import load_contract  # noqa: E402

DEFAULT_VOCAB = 200058
DEFAULT_CONTRACT = "f32-vs-f32"


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def percentiles(values: np.ndarray) -> str:
    if values.size == 0:
        return "n/a"
    q = np.percentile(values, [50, 90, 99, 100])
    return f"p50 {q[0]:.3e}, p90 {q[1]:.3e}, p99 {q[2]:.3e}, max {q[3]:.3e}"


OVERLAP_PREFIXES = ("mixin-", "normin-", "attn_out-", "ffn_inp-", "l_out-")


def read_trace(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 5:
            out[fields[0]] = [float(x) for x in fields[1:5]]
    return out


def fmt4(v: list[float]) -> str:
    return "(mean {:.6g}, std {:.6g}, min {:.6g}, max {:.6g})".format(*v)


def first_divergence(ref_path: Path, metal_path: Path, pos: int, tol: float) -> str:
    if not ref_path.exists() or not metal_path.exists():
        missing = ref_path if not ref_path.exists() else metal_path
        return f"no trace at {missing}"
    mine = read_trace(metal_path)
    suffix = f"-row{pos}"
    checked = 0
    for line in ref_path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 5 or not fields[0].endswith(suffix):
            continue
        base = fields[0][: -len(suffix)]
        if not base.startswith(OVERLAP_PREFIXES):
            continue
        want = [float(x) for x in fields[1:5]]
        checked += 1
        got = mine.get(base)
        if got is None:
            return f"{base}: absent from the Metal trace"
        delta = max(abs(a - b) for a, b in zip(got, want))
        if delta > tol:
            return (f"{base}: reference {fmt4(want)} against Metal {fmt4(got)}, "
                    f"largest difference {delta:.3e} over {checked} overlapping tensors")
    if checked == 0:
        return f"no position-{pos} rows for the overlapping tensors in {ref_path}"
    return f"every one of {checked} overlapping tensors agrees within {tol:g}"


def _trace_tag(runner_sha: str | None, tokens: list[int], ref_py_sha: str | None) -> str:
    tag = ""
    if runner_sha:
        tag += f".r{runner_sha[:12]}"
    if ref_py_sha:
        tag += f".f{ref_py_sha[:12]}"
    digest = hashlib.sha256(json.dumps(tokens).encode("utf-8")).hexdigest()[:12]
    return f"{tag}.t{digest}"


def trace_prefix(name: str, tokens: list[int], pos: int, args, runner_sha: str | None,
                 ref_py_sha: str | None) -> tuple[Path, Path]:
    """Write both sides' traces for the prefix ending at one position, each inside the memory ceiling."""
    metal_dir, ref_dir = Path(args.metal_traces), Path(args.ref_traces)
    metal_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)
    tag = _trace_tag(runner_sha, tokens[:pos + 1], ref_py_sha)
    ids = metal_dir / f"{name}.p{pos}{tag}.tokens.json"
    ids.write_text(json.dumps(tokens[:pos + 1]))
    metal = metal_dir / f"{name}.p{pos}{tag}.trace"
    if not metal.exists():
        done = subprocess.run(["python3", args.heavy, "--", args.runner, args.model, str(ids),
                               "trace", str(metal), str(pos)], capture_output=True, text=True)
        if done.returncode != 0:
            return Path(f"the runner's trace failed: {done.stderr.strip()[-200:]}"), metal
    ref = ref_dir / f"{name}.p{pos}{tag}.trace"
    if not ref.exists():
        venv = Path(args.ref_venv) / "bin" / "python"
        cmd = ["python3", args.heavy, "--", str(venv), args.reference, "--model", args.model,
               "--tokens", " ".join(str(t) for t in tokens[:pos + 1]),
               "--out", str(ref_dir / f"{name}.p{pos}{tag}.npy"),
               "--trace", str(ref), "--trace-row", str(pos),
               "--router-report", str(ref_dir / f"{name}.p{pos}{tag}.routers.json"),
               "--expert-cache-gib", "10"]
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode != 0:
            return metal, Path(f"the reference's trace failed: {done.stderr.strip()[-200:]}")
    return ref, metal


def tightest_routers(ref_dir: Path, name: str, pos: int, tag: str, count: int = 3) -> str:
    path = ref_dir / f"{name}.p{pos}{tag}.routers.json"
    if not path.exists():
        return f"no router report at {path}"
    rows = json.loads(path.read_text())
    ranked = sorted(((abs(v["gap_8_9"]), slot) for slot, v in rows.items()))
    bits = []
    for gap, slot in ranked[:count]:
        v = rows[slot]
        bits.append(f"slot {slot} (layer {v['layer']}): picks {v['picks']} against 9th {v['ninth']}, "
                    f"8th/9th selection gap {v['gap_8_9']:.3g} (prob gap {v['gap_8_9_prob']:.3g})")
    return "; ".join(bits)


def current_identities(args) -> tuple[str | None, list[str], str | None, str]:
    """Hash the current runner and model inputs; missing identities are reported, never invented."""
    problems: list[str] = []
    runner_sha: str | None = args.runner_sha256
    if runner_sha is None and args.runner and Path(args.runner).exists():
        runner_sha = provenance.sha256_file(Path(args.runner))
    if runner_sha is None:
        problems.append("no runner identity: pass --runner or --runner-sha256")
    model_digest: str | None = args.model_identity
    if model_digest is None and args.model and Path(args.model).exists():
        model_digest = provenance.model_identity(Path(args.model))
    if model_digest is None:
        problems.append("no model identity: pass --model or --model-identity")
    return runner_sha, problems, model_digest, args.arithmetic_policy


def load_exceptions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("exceptions", [])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--histories", required=True)
    parser.add_argument("--ref-dir", required=True)
    parser.add_argument("--metal-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--vocab", type=int, default=DEFAULT_VOCAB)
    parser.add_argument("--contract", default=DEFAULT_CONTRACT)
    parser.add_argument("--policy-path", default=str(HERE.parents[1] / "ref" / "gate_policy.json"))
    parser.add_argument("--exceptions", default=str(HERE / "gate_exceptions.json"))
    parser.add_argument("--arithmetic-policy", default=provenance.ARITHMETIC_POLICY)
    parser.add_argument("--tensor-tol", type=float, default=1e-3)
    parser.add_argument("--runner")
    parser.add_argument("--runner-sha256")
    parser.add_argument("--model")
    parser.add_argument("--model-identity")
    parser.add_argument("--heavy")
    parser.add_argument("--ref-venv")
    parser.add_argument("--reference")
    parser.add_argument("--metal-traces", default="")
    parser.add_argument("--ref-traces", default="")
    args = parser.parse_args()

    histories = json.loads(Path(args.histories).read_text())
    ref_dir, metal_dir = Path(args.ref_dir), Path(args.metal_dir)
    report = Path(args.report)
    args.metal_traces = args.metal_traces or str(metal_dir / "trace")
    args.ref_traces = args.ref_traces or str(ref_dir.parent / "trace-ref")
    can_trace = all([args.runner, args.model, args.heavy, args.ref_venv, args.reference,
                     Path(args.runner).exists() if args.runner else False,
                     Path(args.model).exists() if args.model else False,
                     Path(args.heavy).exists() if args.heavy else False])

    base_contract = load_contract(args.contract, Path(args.policy_path))
    contract = base_contract if base_contract.vocab == args.vocab else type(base_contract)(
        base_contract.name, args.vocab, base_contract.bounds)

    runner_sha, identity_problems, model_digest, policy = current_identities(args)
    args.model_identity_digest = model_digest or ""
    ref_py_sha: str | None = None
    if args.reference and Path(args.reference).exists():
        ref_py_sha = provenance.sha256_file(Path(args.reference))

    total = confident_total = agree_total = agree_confident = 0
    kls: list[float] = []
    abs_all: list[np.ndarray] = []
    rel_all: list[np.ndarray] = []
    rows: list[str] = []
    divergences: list[str] = []
    problems: list[str] = list(identity_problems)
    long_histories = 0
    within = explained = unexplained = 0
    per_history: list[dict] = []
    exceptions = load_exceptions(Path(args.exceptions))

    oracle_receipts_path = HERE.parents[1] / "ref" / "oracle_receipts.json"
    recorded_outputs: dict = {}
    if oracle_receipts_path.exists():
        recorded_outputs = json.loads(oracle_receipts_path.read_text()).get("outputs", {})

    for entry in histories:
        name, tokens = entry["name"], entry["token_ids"]
        expected = len(tokens)
        token_sha = provenance.token_history_sha(tokens)
        if len(tokens) > 1024:
            long_histories += 1
        ref_path, got_path = ref_dir / f"{name}.npy", metal_dir / f"{name}.bin"
        history_problems: list[str] = []
        ref_digest: str | None = None
        if not ref_path.exists() or not got_path.exists():
            history_problems.append(f"{name}: missing {'reference' if not ref_path.exists() else 'Metal'} logits")
        else:
            try:
                reference = np.load(ref_path)
            except (OSError, ValueError) as exc:
                history_problems.append(f"{name}: cannot load reference logits: {exc}")
                reference = None
            if history_problems:
                pass
            elif reference.shape[0] != expected:
                history_problems.append(f"{name}: reference holds {reference.shape[0]} rows, "
                                        f"histories.json names {expected} tokens")
            elif reference.shape[1] != args.vocab:
                history_problems.append(f"{name}: reference width {reference.shape[1]}, "
                                        f"expected vocabulary {args.vocab}")
            else:
                ref_digest = provenance.sha256_file(ref_path)
                mine_size = got_path.stat().st_size
                if mine_size % (4 * args.vocab) != 0:
                    history_problems.append(f"{name}: Metal logits are {mine_size} bytes, "
                                            f"not whole rows of {args.vocab}")
                    mine = None
                else:
                    mine = np.fromfile(got_path, dtype=np.float32).reshape(-1, args.vocab)
                    if mine.shape[0] != expected:
                        history_problems.append(f"{name}: Metal holds {mine.shape[0]} rows, "
                                                f"histories.json names {expected} tokens")
                        mine = None
                if not history_problems and runner_sha is not None and model_digest is not None:
                    receipt_problems = provenance.check_receipt(
                        got_path, runner_sha256=runner_sha, model_identity_digest=model_digest,
                        token_sha256=token_sha, policy=policy, vocab=args.vocab, rows=expected)
                    history_problems.extend(f"{name}: {p}" for p in receipt_problems)
                    if receipt_problems:
                        mine = None
                elif not history_problems:
                    history_problems.append(f"{name}: refusing to compare without runner and model identities")
                    mine = None
        if history_problems:
            problems.extend(history_problems)
            per_history.append({"name": name, "tokens": expected, "status": "unexplained",
                                "gate": None, "problems": history_problems,
                                "ref_digest": ref_digest, "token_sha256": token_sha,
                                "observed_onset": None})
            unexplained += 1
            rows.append(f"| {name} | {expected} | n/a | n/a | n/a | n/a | n/a | unexplained |")
            continue

        result = gate_check(reference, mine, expected, contract)

        n = result.positions or expected
        total += n
        confident_total += result.confident
        kls.append(float(result.mean_kl))
        err = np.abs(reference.astype(np.float64) - mine.astype(np.float64))
        scale = np.maximum(np.abs(reference).max(axis=1, keepdims=True), 1e-12)
        rel = err / scale
        gate_problems = list(result.problems)
        if gate_problems:
            history_problems.extend(f"{name}: {p}" for p in gate_problems)
            problems.extend(f"{name}: {p}" for p in gate_problems)
        abs_all.append(err.max(axis=1))
        rel_all.append(rel.max(axis=1))

        agree = mine.argmax(axis=1) == reference.argmax(axis=1)
        agree_total += int(agree.sum())
        ranked = np.sort(reference, axis=1)
        confident = (ranked[:, -1] - ranked[:, -2]) >= contract.bounds.confident_margin
        agree_confident += int((agree & confident).sum())
        conf_pct = agree[confident].mean() * 100 if confident.any() else float("nan")

        exception = next((e for e in exceptions if e.get("history") == name), None)
        numerical_only = history_problems and all(
            not p.startswith(f"{name}: receipt") and "refusing" not in p and "missing" not in p
            and "holds" not in p and "width" not in p and "load" not in p
            for p in history_problems)
        over_tol = np.nonzero(err.max(axis=1) > args.tensor_tol)[0]
        observed_onset: int | None = int(over_tol[0]) if over_tol.size else None
        if not history_problems:
            status = "within contract"
            within += 1
        elif exception is not None and numerical_only:
            status = "explained divergence"
            explained += 1
        else:
            status = "unexplained"
            unexplained += 1
        per_history.append({"name": name, "tokens": expected, "status": status,
                            "observed_onset": observed_onset,
                            "gate": {"positions": result.positions, "confident": result.confident,
                                     "top1": result.top1, "confident_top1": result.confident_top1,
                                     "max_abs": result.max_abs, "mean_abs": result.mean_abs,
                                     "max_position_kl": result.max_position_kl,
                                     "mean_kl": result.mean_kl,
                                     "worst_position": result.worst_position},
                            "problems": history_problems, "ref_digest": ref_digest,
                            "token_sha256": token_sha,
                            "exception": exception})
        rows.append(f"| {name} | {expected} | {result.max_abs:.3e} | {result.mean_kl:.3e} "
                    f"| {result.top1 * 100:.2f}% | {result.confident_top1 * 100:.2f}% "
                    f"| {result.confident} | {status} |")

        if history_problems:
            divergences.append(f"\n### {name} gate problems: " + "; ".join(history_problems))
        worst_order = np.argsort(err.max(axis=1))[::-1][:5]
        outlier_bits = ", ".join(f"position {int(i)} (max abs {err[i].max():.3e}, "
                                 f"KL {(np.exp(log_softmax(reference[i:i+1])) * (log_softmax(reference[i:i+1]) - log_softmax(mine[i:i+1]))).sum():.3e})"
                                 for i in worst_order)
        divergences.append(f"\n### {name} outliers: {outlier_bits}")
        disagreed = np.nonzero(~agree)[0]
        for i in disagreed[:8]:
            margin = float(ranked[i, -1] - ranked[i, -2])
            divergences.append(
                f"\n### {name} position {int(i)}: reference top-1 {int(reference[i].argmax())}, "
                f"Metal top-1 {int(mine[i].argmax())}\n"
                f"- reference top-2 margin {margin:.4g}, largest logit error {err[i].max():.4g}")
        if len(disagreed) > 8:
            divergences.append(f"\n### {name}: {len(disagreed) - 8} further top-1 disagreements")
        over = np.nonzero(err.max(axis=1) > args.tensor_tol)[0]
        if over.size:
            onset = int(over[0])
            tag = _trace_tag(runner_sha, tokens[:onset + 1], ref_py_sha)
            if not can_trace:
                divergences.append(
                    f"\n### {name} onset position {onset}: largest logit error {err[onset].max():.4g}\n"
                    "- first divergent tensor: not traced: no runner, model or memory ceiling was available")
            else:
                ref_tr, metal_tr = trace_prefix(name, tokens, onset, args, runner_sha, ref_py_sha)
                if ref_tr.exists() and metal_tr.exists():
                    where = first_divergence(ref_tr, metal_tr, onset, args.tensor_tol)
                    ctrl = "no earlier position"
                    if onset > 0:
                        ref_c, metal_c = trace_prefix(name, tokens, onset - 1, args, runner_sha, ref_py_sha)
                        if ref_c.exists() and metal_c.exists():
                            ctrl = first_divergence(ref_c, metal_c, onset - 1, args.tensor_tol)
                        else:
                            ctrl = str(ref_c) if not ref_c.exists() else str(metal_c)
                    routers = tightest_routers(Path(args.ref_traces), name, onset, tag)
                    divergences.append(
                        f"\n### {name} onset position {onset}: largest logit error {err[onset].max():.4g}, "
                        f"KL {(np.exp(log_softmax(reference[onset:onset+1])) * (log_softmax(reference[onset:onset+1]) - log_softmax(mine[onset:onset+1]))).sum():.4g}\n"
                        f"- first divergent tensor: {where}\n"
                        f"- clean control at {onset - 1}: {ctrl}\n"
                        f"- tightest reference router 8th/9th gaps at {onset}: {routers}")
                else:
                    bad = str(ref_tr) if not ref_tr.exists() else str(metal_tr)
                    divergences.append(
                        f"\n### {name} onset position {onset}: largest logit error {err[onset].max():.4g}\n"
                        f"- first divergent tensor: {bad}")

    abs_flat = np.concatenate(abs_all) if abs_all else np.array([])
    rel_flat = np.concatenate(rel_all) if rel_all else np.array([])
    overall = 100.0 * agree_total / total if total else 0.0
    confident_pct = 100.0 * agree_confident / confident_total if confident_total else float("nan")

    uncovered = [h["name"] for h in per_history
                 if not any(h["name"] in str(k) for k in recorded_outputs)]
    gap_note = ("Open provenance gap: campaign/ref/oracle_receipts.json records output digests only for "
                "the ref-check outputs, not for these parity histories' reference .npy files; the reference "
                "files are verified by oracle.py shape/finiteness/token-count checks, and their observed "
                "digests are recorded below. The current build identity is not attached to those older files.")

    if within == len(histories) and len(histories) > 0:
        verdict = "parity: qualified"
        failed = False
    else:
        verdict = (f"parity: not qualified: {within} within contract, "
                   f"{explained} explained divergences, {unexplained} unexplained")
        failed = True

    b = contract.bounds
    lines = [
        "# Inkling 2 Mini parity: Metal against the float32 reference",
        "",
        "Arithmetic: NVFP4 and MXFP8 weights decoded from the checkpoint's own bytes, float32 activations. This is",
        "not the official activation-quantized path and no equivalence with it is claimed.",
        "",
        f"Contract: {contract.name} over a {contract.vocab}-token vocabulary: per-logit max abs {b.max_abs:g}, "
        f"mean abs {b.mean_abs:g}, per-position max KL {b.max_position_kl:g}, mean KL {b.mean_kl:g}, "
        f"teacher top-2 margin {b.confident_margin:g} with confident top-1 {b.confident_top1:.0%}.",
        "",
        "## Summary",
        "",
        f"- positions compared: {total} over {len(histories)} histories "
        f"({long_histories} past the 1,024-token sliding window)",
        f"- mean KL: {float(np.mean(kls)):.6g}" if kls else "- mean KL: n/a",
        f"- top-1 agreement overall: {overall:.2f}%",
        f"- top-1 agreement on positions with reference top-2 margin >= {b.confident_margin:g}: "
        f"{confident_pct:.2f}% over {confident_total} positions",
        f"- absolute logit error: {percentiles(abs_flat)}",
        f"- relative logit error: {percentiles(rel_flat)}",
        f"- qualification: {within} within contract, {explained} explained divergences, "
        f"{unexplained} unexplained",
        "",
        "## Per history",
        "",
        "| history | tokens | max abs error | mean KL | top-1 | confident top-1 | confident | status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        *rows,
        "",
        "## Provenance",
        "",
        f"- runner sha256: {runner_sha}",
        f"- model identity: {model_digest}",
        f"- arithmetic policy: {policy}",
        f"- {gap_note if uncovered else 'campaign/ref/oracle_receipts.json records output digests covering these reference files.'}",
        "",
        "| history | token-history sha256 | reference sha256 | candidate receipt |",
        "| --- | --- | --- | --- |",
        *[f"| {h['name']} | {h['token_sha256']} | {h['ref_digest']} "
          f"| {'refused' if any('receipt' in p or 'refusing' in p for p in h['problems']) else 'matches'} |"
          for h in per_history],
        "",
        "## Divergences",
    ]
    lines += divergences if divergences else ["", "Every position agrees on top-1."]
    lines += ["",
              "## Verdict",
              "",
              verdict,
              ""]
    if failed:
        lines += ["Qualification is incomplete: every failing history above must return to within contract.",
                  ""]
    if exceptions:
        lines += ["Recorded explanations (evidence only; none of these turns a failure into a pass):"]
        for e in exceptions:
            hist = next((h for h in per_history if h["name"] == e.get("history")), None)
            observed = hist.get("observed_onset") if hist else None
            drift = ("; this run's observed onset differs from the recorded one"
                     if observed is not None and observed != e.get("onset_position") else "")
            lines.append(f"- {e.get('history')}: recorded onset {e.get('onset_position')}, "
                         f"observed onset in this run {observed}{drift}; {e.get('mechanism')} "
                         f"(evidence: {e.get('evidence')})")
        lines += [""]
    report.write_text("\n".join(lines) + "\n")

    console = verdict if not failed else (
        f"parity: FAIL: qualification incomplete: {within} within contract, "
        f"{explained} explained divergences, {unexplained} unexplained")
    for line in lines[:16]:
        print(line)
    print(f"report: {report}")
    for p in problems:
        print(f"problem: {p}", file=sys.stderr)
    print(console)
    if failed:
        print(console, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
