import sys

import numpy as np

from logit_gate import Bounds, Contract, check

VOCAB = 8
CONTRACT = Contract("test", VOCAB, Bounds(max_abs=1e-3, mean_abs=1e-4, max_position_kl=1e-6, mean_kl=1e-8,
                                          confident_margin=0.1, confident_top1=1.0))


def teacher(rows: int = 4) -> np.ndarray:
    rng = np.random.default_rng(7)
    base = rng.normal(0, 3, size=(rows, VOCAB)).astype(np.float32)
    base[np.arange(rows), np.arange(rows) % VOCAB] += 10
    return base


def fails(t: np.ndarray, c: np.ndarray, positions: int, needle: str) -> None:
    problems = check(t, c, positions, CONTRACT).problems
    assert any(needle in p for p in problems), f"expected a problem containing {needle!r}, got {problems}"


def test_close_logits_pass() -> None:
    t = teacher()
    c = t + np.float32(2e-5)
    assert check(t, c, 4, CONTRACT).problems == []


def test_same_argmax_distribution_corruption_fails() -> None:
    t = np.zeros((1, VOCAB), dtype=np.float32)
    c = np.zeros((1, VOCAB), dtype=np.float32)
    t[0, :3] = [2, 1, 0]
    c[0, :3] = [200, 100, 0]
    fails(t, c, 1, "max_abs")
    fails(t, c, 1, "kl")


def test_nan_row_fails() -> None:
    t = teacher()
    c = t.copy()
    t[2, 5] = np.nan
    fails(t, c, 4, "non-finite")


def test_empty_evidence_fails() -> None:
    empty = np.empty((0, VOCAB), dtype=np.float32)
    fails(empty, empty, 0, "no positions")


def test_confidently_wrong_token_fails() -> None:
    t = teacher()
    c = t.copy()
    c[1] = np.roll(c[1], 1)
    fails(t, c, 4, "confident top-1")


def test_shape_and_position_count_fail() -> None:
    t = teacher()
    fails(t, t[:3], 4, "shape")
    fails(t, t, 5, "shape")
    fails(t[:, :VOCAB - 1], t[:, :VOCAB - 1], 4, "shape")


def test_no_confident_position_fails() -> None:
    t = np.zeros((2, VOCAB), dtype=np.float32)
    fails(t, t.copy(), 2, "no position has a teacher top-2 margin")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} logit gate checks passed")
    sys.exit(0)
