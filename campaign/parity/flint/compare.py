"""Compare teacher-forced logits: full error distribution, per-history table and first divergent tensors.

Both engines write bf16 bits as [tokens, vocab]; each row r is the distribution of token r + 1 given tokens 0..r.
The report answers Decision 10's questions on this corpus: the mean KL to the oracle, top-1 agreement, and top-1
agreement on the rows whose top-2 margins are at least 0.1 on both sides.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

VOCAB = 150144
PERCENTILES = (0, 1, 5, 25, 50, 75, 95, 99, 99.9, 100)
MARGIN = 0.1


def bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def load(path: Path, tokens: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint16)
    if raw.size != tokens * VOCAB:
        raise SystemExit(f"{path}: {raw.size} words, expected {tokens} * {VOCAB}")
    return raw.reshape(tokens, VOCAB)


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    peak = x.max(-1, keepdims=True)
    return x - peak - np.log(np.exp(x - peak).sum(-1, keepdims=True))


def row_stats(tf: np.ndarray, ref: np.ndarray) -> dict:
    a, b = bits_to_f32(tf), bits_to_f32(ref)
    absolute = np.abs(a - b)
    scale = np.maximum(np.abs(a), 1e-3)
    relative = absolute / scale
    pa, pb = log_softmax(a), log_softmax(b)
    kl = (np.exp(pb) * (pb - pa)).sum(-1)
    ta, tb = a.argmax(-1), b.argmax(-1)
    sorted_a = np.sort(a, -1)
    sorted_b = np.sort(b, -1)
    margin_a = sorted_a[:, -1] - sorted_a[:, -2]
    margin_b = sorted_b[:, -1] - sorted_b[:, -2]
    agree = ta == tb
    decisive = (margin_a >= MARGIN) & (margin_b >= MARGIN)
    return {
        "absolute": absolute,
        "relative": relative,
        "kl": kl,
        "agree": agree,
        "margin_a": margin_a,
        "margin_b": margin_b,
        "decisive": decisive,
        "top1_a": ta,
        "top1_b": tb,
    }


def summarize(rows: dict) -> dict:
    absolute = rows["absolute"].reshape(-1)
    relative = rows["relative"].reshape(-1)
    kl = rows["kl"]
    return {
        "rows": int(kl.shape[0]),
        "positions": int(absolute.shape[0]),
        "absolute": {f"p{p:g}": float(np.percentile(absolute, p)) for p in PERCENTILES}
        | {"mean": float(absolute.mean())},
        "relative": {f"p{p:g}": float(np.percentile(relative, p)) for p in PERCENTILES}
        | {"mean": float(relative.mean())},
        "kl": {f"p{p:g}": float(np.percentile(kl, p)) for p in PERCENTILES}
        | {"mean": float(kl.mean())},
        "top1_agreement": float(rows["agree"].mean()),
        "top1_agree_rows": int(rows["agree"].sum()),
        "top2_margin": {
            "tf_p1": float(np.percentile(rows["margin_a"], 1)),
            "tf_p50": float(np.percentile(rows["margin_a"], 50)),
            "tf_p99": float(np.percentile(rows["margin_a"], 99)),
            "oracle_p1": float(np.percentile(rows["margin_b"], 1)),
            "oracle_p50": float(np.percentile(rows["margin_b"], 50)),
            "oracle_p99": float(np.percentile(rows["margin_b"], 99)),
            "tf_below_0.1": int((rows["margin_a"] < MARGIN).sum()),
            "oracle_below_0.1": int((rows["margin_b"] < MARGIN).sum()),
        },
        "decisive_rows": int(rows["decisive"].sum()),
        "decisive_agreement": float(rows["agree"][rows["decisive"]].mean()) if rows["decisive"].any() else None,
    }


def divergent_row(tf: np.ndarray, ref: np.ndarray, row: int, top: int = 5) -> dict:
    a, b = bits_to_f32(tf[row]), bits_to_f32(ref[row])
    difference = np.abs(a - b)
    order = np.argsort(-difference)
    impulse = np.abs(a - b)[:, None]  # per-vocabulary error, the first tensor a trace can reach
    return {
        "row": row,
        "max_absolute": float(difference.max()),
        "mean_absolute": float(difference.mean()),
        "argmax_absolute_vocab": int(order[0]),
        "tf_bit_exact_with_oracle": bool(np.array_equal(tf[row], ref[row])),
        "largest": [
            {
                "token": int(index),
                "tf": float(a[index]),
                "oracle": float(b[index]),
                "absolute": float(difference[index]),
                "tf_bits": int(tf[row, index]),
                "oracle_bits": int(ref[row, index]),
            }
            for index in order[:top]
        ],
        "vocabulary_error_quantiles": {
            f"p{p:g}": float(np.percentile(impulse, p)) for p in PERCENTILES
        },
    }


def self_reference(histories: Path, ref_dir: Path, other_dir: Path, label: str) -> dict:
    """The oracle against another of its own paths: the spread that belongs to accumulation order alone."""
    corpus = json.loads(histories.read_text())
    combined = {"absolute": [], "relative": [], "kl": [], "agree": [], "margin_a": [], "margin_b": [], "decisive": [], "top1_a": [], "top1_b": []}
    per_history = {}
    for entry in corpus["histories"]:
        name, tokens = entry["name"], entry["tokens"]
        rows = row_stats(load(other_dir / f"{name}.bin", tokens), load(ref_dir / f"{name}.bin", tokens))
        summary = summarize(rows)
        per_history[name] = {"kind": entry["kind"], "tokens": tokens, **summary}
        for key in combined:
            combined[key].append(rows[key])
    all_rows = {
        key: np.concatenate([value.reshape(-1) if value.ndim > 1 else value for value in combined[key]])
        for key in combined
    }
    return {"label": label, "overall": summarize(all_rows), "per_history": per_history}


def main(
    histories: Path,
    tf_dir: Path,
    ref_dir: Path,
    out: Path,
    logits_dir: Path,
    other_dir: Path | None = None,
    other_label: str = "",
    oracle_note: str = "",
    oracle_equivalence: str = "",
) -> int:
    corpus = json.loads(histories.read_text())
    per_history = {}
    combined = {"absolute": [], "relative": [], "kl": [], "agree": [], "margin_a": [], "margin_b": [], "decisive": [], "top1_a": [], "top1_b": []}
    for entry in corpus["histories"]:
        name = entry["name"]
        tokens = entry["tokens"]
        tf = load(tf_dir / f"{name}.bin", tokens)
        ref = load(ref_dir / f"{name}.bin", tokens)
        rows = row_stats(tf, ref)
        summary = summarize(rows)
        per_history[name] = {
            "kind": entry["kind"],
            "tokens": tokens,
            "ids_sha256": entry["ids_sha256"],
            "logits_sha256": {
                "tensorfold": hashlib.sha256((tf_dir / f"{name}.bin").read_bytes()).hexdigest(),
                "oracle": hashlib.sha256((ref_dir / f"{name}.bin").read_bytes()).hexdigest(),
            },
            **summary,
        }
        for key in combined:
            combined[key].append(rows[key])
    all_rows = {
        key: np.concatenate([value.reshape(-1) if value.ndim > 1 else value for value in combined[key]])
        for key in combined
    }
    overall = summarize(all_rows)
    mismatched = np.nonzero(~all_rows["agree"])[0]
    order = np.argsort(-all_rows["kl"])
    outliers = [
        {
            "global_position": int(index),
            "kl": float(all_rows["kl"][index]),
            "tf_top1": int(all_rows["top1_a"][index]),
            "oracle_top1": int(all_rows["top1_b"][index]),
            "tf_margin": float(all_rows["margin_a"][index]),
            "oracle_margin": float(all_rows["margin_b"][index]),
        }
        for index in order[:5]
    ]
    result = {
        "corpus_sha256": hashlib.sha256(histories.read_bytes()).hexdigest(),
        "margin_floor": MARGIN,
        "histories": len(per_history),
        "histories_over_window": int(sum(1 for entry in corpus["histories"] if entry["tokens"] > 513)),
        "tokens": int(all_rows["kl"].shape[0]),
        "positions": int(all_rows["absolute"].size),
        "oracle_note": oracle_note,
        "oracle_equivalence": oracle_equivalence,
        "overall": overall,
        "per_history": per_history,
        "disagreeing_rows": [
            {
                "tf_top1": int(all_rows["top1_a"][index]),
                "oracle_top1": int(all_rows["top1_b"][index]),
                "tf_margin": float(all_rows["margin_a"][index]),
                "oracle_margin": float(all_rows["margin_b"][index]),
            }
            for index in mismatched
        ],
        "outliers_by_kl": outliers,
        "divergent_rows": [],
        "oracle_self_reference": (
            self_reference(histories, ref_dir, other_dir, other_label) if other_dir is not None else None
        ),
    }
    logits = out.parent
    logits.mkdir(parents=True, exist_ok=True)
    positions = np.concatenate([np.arange(entry["tokens"]) for entry in corpus["histories"]])
    per_row_max = np.concatenate(
        [
            np.abs(
                bits_to_f32(load(tf_dir / f"{entry['name']}.bin", entry["tokens"]))
                - bits_to_f32(load(ref_dir / f"{entry['name']}.bin", entry["tokens"]))
            ).max(axis=1)
            for entry in corpus["histories"]
        ]
    )
    rows_at_zero = positions == 0
    result["localization"] = {
        "row_zero_mean_max_absolute": float(per_row_max[rows_at_zero].mean()),
        "row_zero_max_absolute": float(per_row_max[rows_at_zero].max()),
        "all_rows_mean_max_absolute": float(per_row_max.mean()),
        "first_row_of_every_history_is_the_worst_band": bool(rows_at_zero.mean() > 0 and per_row_max[rows_at_zero].mean() > per_row_max[~rows_at_zero].mean()),
        "position_bands": [
            {
                "from": int(lo),
                "to": int(hi),
                "rows": int(((positions >= lo) & (positions < hi)).sum()),
                "mean_max_absolute": float(per_row_max[(positions >= lo) & (positions < hi)].mean()),
            }
            for lo, hi in ((0, 1), (1, 4), (4, 16), (16, 64), (64, 128), (128, 256), (256, 512), (512, 513), (513, 10**9))
            if ((positions >= lo) & (positions < hi)).any()
        ],
    }
    if other_dir is not None:
        other_rows = np.concatenate(
            [
                np.abs(bits_to_f32(load(other_dir / f"{entry['name']}.bin", entry["tokens"])) - bits_to_f32(load(ref_dir / f"{entry['name']}.bin", entry["tokens"]))).max(axis=1)
                for entry in corpus["histories"]
            ]
        )
        reference_stable = other_rows <= 0.5
        result["localization"]["rows_stable_for_oracle_itself"] = int(reference_stable.sum())
        result["localization"]["rows_stable_for_oracle_itself_mean_tf_error"] = float(per_row_max[reference_stable].mean())
        result["localization"]["rows_unstable_for_oracle_itself_mean_tf_error"] = float(per_row_max[~reference_stable].mean())
        result["localization"]["classification_agreement"] = int(((per_row_max <= 0.5) == reference_stable).sum())
        decode_top1 = np.concatenate(
            [
                bits_to_f32(load(ref_dir / f"{entry['name']}.bin", entry["tokens"])).argmax(axis=1)
                for entry in corpus["histories"]
            ]
        )
        other_top1 = np.concatenate(
            [
                bits_to_f32(load(other_dir / f"{entry['name']}.bin", entry["tokens"])).argmax(axis=1)
                for entry in corpus["histories"]
            ]
        )
        tf_top1 = all_rows["top1_a"]
        disagreements = tf_top1 != decode_top1
        result["disagreement_breakdown"] = {
            "against_oracle_decode": int(disagreements.sum()),
            "resolved_by_oracle_other_path": int((disagreements & (tf_top1 == other_top1)).sum()),
            "where_oracle_also_differs_from_itself": int((disagreements & (other_top1 != decode_top1)).sum()),
            "top1_agreement_against_other_path": float((tf_top1 == other_top1).mean()),
            "oracle_paths_agree": float((other_top1 == decode_top1).mean()),
        }
        result["localization"]["per_row_error"] = {
            "tf": {f"p{p:g}": float(np.percentile(per_row_max, p)) for p in PERCENTILES} | {"mean": float(per_row_max.mean())},
            "oracle_self": {f"p{p:g}": float(np.percentile(other_rows, p)) for p in PERCENTILES} | {"mean": float(other_rows.mean())},
            "correlation": float(np.corrcoef(per_row_max, other_rows)[0, 1]),
            "rows_tf_at_most_oracle_self": int((per_row_max <= other_rows).sum()),
            "rows": int(per_row_max.size),
        }
    at = 0
    for entry in corpus["histories"]:
        name = entry["name"]
        tokens = entry["tokens"]
        tf = load(tf_dir / f"{name}.bin", tokens)
        ref = load(ref_dir / f"{name}.bin", tokens)
        rows = row_stats(tf, ref)
        bad = np.nonzero(~rows["agree"])[0]
        if bad.size == 0:
            at += tokens
            continue
        worst = bad[np.argmax(rows["kl"][bad])]
        (logits_dir).mkdir(parents=True, exist_ok=True)
        tf[worst].tofile(logits_dir / f"{name}.row{int(worst)}.tf.bin")
        ref[worst].tofile(logits_dir / f"{name}.row{int(worst)}.oracle.bin")
        entry_out = {
            "history": name,
            "kind": entry["kind"],
            "tokens": tokens,
            "global_position": int(at + worst),
            **divergent_row(tf, ref, int(worst)),
        }
        if other_dir is not None:
            other = load(other_dir / f"{name}.bin", tokens)
            same_row = np.abs(bits_to_f32(other[worst]) - bits_to_f32(ref[worst]))
            other_top1 = int(bits_to_f32(other[worst]).argmax())
            tf_top1 = int(bits_to_f32(tf[worst]).argmax())
            ref_top1 = int(bits_to_f32(ref[worst]).argmax())
            entry_out["oracle_self_reference"] = {
                "label": other_label,
                "max_absolute": float(same_row.max()),
                "mean_absolute": float(same_row.mean()),
                "top1": other_top1,
                "tf_top1": tf_top1,
                "oracle_top1": ref_top1,
                "self_reference_agrees_with_oracle": bool(other_top1 == ref_top1),
                "tf_agrees_with_self_reference": bool(tf_top1 == other_top1),
            }
            other[worst].tofile(logits_dir / f"{name}.row{int(worst)}.oracle-other.bin")
        result["divergent_rows"].append(entry_out)
        at += tokens
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(
        json.dumps(
            {
                "histories": result["histories"],
                "tokens": result["tokens"],
                "mean_kl": overall["kl"]["mean"],
                "top1": overall["top1_agreement"],
                "decisive_top1": overall["decisive_agreement"],
                "outliers": len(result["divergent_rows"]),
                "self_reference_kl": None
                if result["oracle_self_reference"] is None
                else result["oracle_self_reference"]["overall"]["kl"]["mean"],
                "self_reference_top1": None
                if result["oracle_self_reference"] is None
                else result["oracle_self_reference"]["overall"]["top1_agreement"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histories", type=Path, required=True)
    parser.add_argument("--tf", type=Path, required=True)
    parser.add_argument("--ref", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--logits-out", type=Path, required=True)
    parser.add_argument("--other", type=Path, default=None, help="another oracle path to compare the oracle against")
    parser.add_argument("--other-label", default="oracle chunked native prefill")
    parser.add_argument("--oracle-note", default="")
    parser.add_argument("--oracle-equivalence", default="")
    args = parser.parse_args()
    raise SystemExit(
        main(
            args.histories,
            args.tf,
            args.ref,
            args.out,
            args.logits_out,
            args.other,
            args.other_label,
            args.oracle_note,
            args.oracle_equivalence,
        )
    )
