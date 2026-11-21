"""Numerical gate for teacher-forced logit comparisons under a declared arithmetic contract."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

POLICY = Path(__file__).with_name("gate_policy.json")


@dataclass(frozen=True)
class Bounds:
    max_abs: float
    mean_abs: float
    max_position_kl: float
    mean_kl: float
    confident_margin: float
    confident_top1: float


@dataclass(frozen=True)
class Contract:
    name: str
    vocab: int
    bounds: Bounds


@dataclass
class Result:
    positions: int = 0
    confident: int = 0
    top1: float = float("nan")
    confident_top1: float = float("nan")
    max_abs: float = float("nan")
    mean_abs: float = float("nan")
    max_position_kl: float = float("nan")
    mean_kl: float = float("nan")
    worst_position: int = -1
    problems: list[str] = field(default_factory=list)


def load_contract(name: str, policy: Path = POLICY) -> Contract:
    contracts = json.loads(policy.read_text())["contracts"]
    if name not in contracts:
        raise SystemExit(f"unknown arithmetic contract {name!r}; declared: {', '.join(sorted(contracts))}")
    entry = contracts[name]
    return Contract(name, int(entry["vocab"]), Bounds(**entry["bounds"]))


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def check(teacher: np.ndarray, candidate: np.ndarray, positions: int, contract: Contract) -> Result:
    """Compare a candidate's logits [positions, vocab] with the teacher's; every failed condition is a problem."""
    out = Result()
    expected = (positions, contract.vocab)
    for label, array in (("teacher", teacher), ("candidate", candidate)):
        if array.ndim != 2 or array.shape != expected:
            out.problems.append(f"{label} shape {array.shape}, expected {expected}")
    if out.problems:
        return out
    if positions == 0:
        out.problems.append("no positions to compare")
        return out
    for label, array in (("teacher", teacher), ("candidate", candidate)):
        bad = ~np.isfinite(array)
        if bad.any():
            rows = np.unique(np.nonzero(bad)[0])
            out.problems.append(f"{label} has {int(bad.sum())} non-finite logits in rows {rows[:8].tolist()}")
    if out.problems:
        return out

    b = contract.bounds
    out.positions = positions
    error = np.abs(teacher.astype(np.float64) - candidate.astype(np.float64))
    lt, lc = log_softmax(teacher), log_softmax(candidate)
    kl = (np.exp(lt) * (lt - lc)).sum(axis=1)
    top_t, top_c = teacher.argmax(axis=1), candidate.argmax(axis=1)
    ranked = np.sort(teacher, axis=1)
    confident = (ranked[:, -1] - ranked[:, -2]) >= b.confident_margin
    agree = top_t == top_c

    out.confident = int(confident.sum())
    out.top1 = float(agree.mean())
    out.confident_top1 = float(agree[confident].mean()) if out.confident else float("nan")
    out.max_abs = float(error.max())
    out.mean_abs = float(error.mean())
    out.max_position_kl = float(kl.max())
    out.mean_kl = float(kl.mean())
    out.worst_position = int(error.max(axis=1).argmax())

    if out.confident == 0:
        out.problems.append(f"no position has a teacher top-2 margin >= {b.confident_margin}")
    elif out.confident_top1 < b.confident_top1:
        wrong = np.nonzero(confident & ~agree)[0]
        out.problems.append(f"confident top-1 {out.confident_top1:.4%} < {b.confident_top1:.4%} at positions {wrong[:8].tolist()}")
    for metric, bound in (("max_abs", b.max_abs), ("mean_abs", b.mean_abs),
                          ("max_position_kl", b.max_position_kl), ("mean_kl", b.mean_kl)):
        value = getattr(out, metric)
        if value > bound:
            out.problems.append(f"{metric} {value:.4g} exceeds the {contract.name} bound {bound:.4g}")
    return out
