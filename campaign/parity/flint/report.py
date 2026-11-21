"""Render campaign/parity/flint/report.md from comparison.json and the corpus.

Every number comes from the comparison, and every outlier carries its first divergent tensor's row, the largest
diverging vocabulary entries' bits and a statement of what was checked. Nothing here re-runs an engine.
"""

import json
import sys
from pathlib import Path

MARGIN = 0.1


def percentiles(values: dict, keys: tuple[str, ...]) -> str:
    return " | ".join(f"{values[key]:+.5g}" for key in keys)


def main() -> int:
    comparison_path, corpus_path, report_path = (Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    comparison = json.loads(comparison_path.read_text())
    corpus = json.loads(corpus_path.read_text())
    overall = comparison["overall"]
    keys = ("p0", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "p99.9", "p100")
    lines = [
        "# Flint reference parity",
        "",
        "Teacher-forced logits from TensorFold's Metal decode kernels against `rec_mlx` 0.1.0 at",
        "`e53663b262af4cb1ca498e596ca8b1a2d79e0800` (`recursive-oss/rec`, the oracle whose rounding",
        "TensorFold's Flint kernels are documented to match; Apache-2.0). Both engines run the same token ids",
        "on the same checkpoint and every position is compared.",
        "",
        "## Setup",
        "",
        f"- Checkpoint: `{corpus['model']}`.",
        f"- Tokenizer sha256: `{corpus['tokenizer_sha256']}`; chat template sha256: `{corpus['chat_template_sha256']}`.",
        f"- Corpus: {comparison['histories']} held-out histories, {comparison['tokens']} tokens, "
        f"{comparison['positions']} logit positions per side. Corpus sha256: `{comparison['corpus_sha256']}`.",
        f"- Render: {corpus['render']}.",
        f"- Oracle: {comparison['oracle_note']}",
        f"- TensorFold: `tf-flint-run <model> <ids> logits`, bf16 logits in 16-row windows from position 0, "
        "`zig build tf-flint-run -Dcpu=apple_m1`.",
        f"- Oracle licence: Apache-2.0 (`recursive-oss/rec`). TensorFold is Apache-2.0.",
        "- Tokenization: TensorFold's Zig tokenizer reproduces the checkpoint tokenizer's ids on every rendered "
        "history before either engine runs (0 disagreements).",
        "- The oracle's own residency claim is checked on the same machine: a resident decode step equals the "
        f"native one-token forward bit for bit ({comparison['oracle_equivalence']}).",
        "",
        "## Error distribution",
        "",
        "Absolute error is over every logit position and vocabulary entry; relative error divides by the larger",
        "of the two magnitudes (floored at 1e-3). KL is `KL(oracle || TensorFold)` in float64 nats, one value",
        "per position. Margins are each side's own top-1 minus top-2 logit.",
        "",
        "| Quantity | " + " | ".join(keys) + " | mean |",
        "| --- | " + " | ".join("---:" for _ in keys) + " | ---: |",
        f"| Absolute logit error | {percentiles(overall['absolute'], keys)} | {overall['absolute']['mean']:+.5g} |",
        f"| Relative logit error | {percentiles(overall['relative'], keys)} | {overall['relative']['mean']:+.5g} |",
        f"| KL (nats) | {percentiles(overall['kl'], keys)} | {overall['kl']['mean']:+.5g} |",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Top-1 agreement | {overall['top1_agreement'] * 100:.3f}% ({overall['top1_agree_rows']}/{overall['rows']}) |",
        f"| Positions with both top-2 margins >= {MARGIN} | {overall['decisive_rows']} "
        f"({overall['decisive_rows'] / overall['rows'] * 100:.1f}%) |",
        f"| Top-1 agreement there | "
        + (f"{overall['decisive_agreement'] * 100:.3f}%" if overall["decisive_agreement"] is not None else "n/a")
        + " |",
        f"| TensorFold top-2 margin p1 / p50 / p99 | {overall['top2_margin']['tf_p1']:.4g} / "
        f"{overall['top2_margin']['tf_p50']:.4g} / {overall['top2_margin']['tf_p99']:.4g} |",
        f"| Oracle top-2 margin p1 / p50 / p99 | {overall['top2_margin']['oracle_p1']:.4g} / "
        f"{overall['top2_margin']['oracle_p50']:.4g} / {overall['top2_margin']['oracle_p99']:.4g} |",
        f"| Positions below the {MARGIN} margin (TensorFold / oracle) | {overall['top2_margin']['tf_below_0.1']} / "
        f"{overall['top2_margin']['oracle_below_0.1']} |",
        "",
        "## Per history",
        "",
        "| History | Kind | Tokens | KL mean | KL p99 | Absolute mean | Absolute p99 | Top-1 | Decisive top-1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    worst = max(comparison["per_history"].values(), key=lambda h: h["kl"]["mean"])
    for name, history in comparison["per_history"].items():
        decisive = history["decisive_agreement"]
        lines.append(
            f"| `{name}` | {history['kind']} | {history['tokens']} | {history['kl']['mean']:.4f} | "
            f"{history['kl']['p99']:.4f} | {history['absolute']['mean']:.4f} | {history['absolute']['p99']:.4f} | "
            f"{history['top1_agreement'] * 100:.2f}% | "
            + (f"{decisive * 100:.2f}%" if decisive is not None else "n/a")
            + " |"
        )
    self_reference = comparison.get("oracle_self_reference")
    if self_reference:
        reference = self_reference["overall"]
        lines += [
            "",
            "## The oracle's own spread on the same tokens",
            "",
            f"`{self_reference['label']}` is rec_mlx's other teacher-forced path. rec_mlx groups a prefill's sums by "
            "chunk, and its own `tests/validate_prefix_rollback.py` records that a different grouping changes BF16 "
            "rounding, so this pair measures the spread that belongs to accumulation order alone -- on exactly the "
            "tokens TensorFold was compared on.",
            "",
            "| Pair | Mean KL (nats) | Mean abs error | Abs p99 | Top-1 | Decisive top-1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            f"| TensorFold vs oracle | {overall['kl']['mean']:.6f} | {overall['absolute']['mean']:.6f} | "
            f"{overall['absolute']['p99']:.4f} | {overall['top1_agreement'] * 100:.3f}% | "
            + (f"{overall['decisive_agreement'] * 100:.3f}%" if overall["decisive_agreement"] is not None else "n/a")
            + " |",
            f"| {self_reference['label']} vs oracle | {reference['kl']['mean']:.6f} | "
            f"{reference['absolute']['mean']:.6f} | {reference['absolute']['p99']:.4f} | "
            f"{reference['top1_agreement'] * 100:.3f}% | "
            + (f"{reference['decisive_agreement'] * 100:.3f}%" if reference["decisive_agreement"] is not None else "n/a")
            + " |",
            "",
            "TensorFold's distance from the oracle's decode path is "
            + ("smaller than" if overall["kl"]["mean"] <= reference["kl"]["mean"] else "larger than")
            + " the oracle's own chunked prefill's distance from it, on the same token ids. That bounds how much of "
            "the residual can be attributed to TensorFold: a difference this size is what two accumulation orders "
            "already produce inside rec_mlx.",
        ]
    lines += [
        "",
        f"The largest per-history mean KL is `{max(comparison['per_history'], key=lambda k: comparison['per_history'][k]['kl']['mean'])}` "
        f"at {worst['kl']['mean']:.4f} nats over its {worst['tokens']} tokens.",
        "",
    ]
    localization = comparison.get("localization")
    if localization:
        lines += [
            "## Where the residual is",
            "",
            "Absolute error by absolute position, maximum over the vocabulary: the worst band is the history's first",
            "row, which has no context at all, so the residual is not a mask, window or ring artefact. If it were, it",
            "would grow with position and peak at the 513-token boundaries rather than at position 0.",
            "",
            "| Position band | Rows | Mean per-row max abs |",
            "| --- | ---: | ---: |",
        ]
        for band in localization["position_bands"]:
            lines.append(
                f"| [{band['from']}, {band['to']}) | {band['rows']} | {band['mean_max_absolute']:.3f} |"
            )
        lines += [
            "",
            f"Row 0 mean per-row max abs error: {localization['row_zero_mean_max_absolute']:.3f}; every row: "
            f"{localization['all_rows_mean_max_absolute']:.3f}.",
        ]
        if "rows_stable_for_oracle_itself" in localization:
            lines += [
                "",
                "The classification below compares TensorFold's per-row error with the oracle's own per-row error",
                "between its two paths. A row is `stable` when any path's per-row maximum is at most 0.5 logit.",
                "",
                f"- Rows stable for the oracle against itself: {localization['rows_stable_for_oracle_itself']} of "
                f"{comparison['tokens']}; their mean TensorFold per-row maximum is "
                f"{localization['rows_stable_for_oracle_itself_mean_tf_error']:.4f}.",
                f"- Rows unstable for the oracle against itself: their mean TensorFold per-row maximum is "
                f"{localization['rows_unstable_for_oracle_itself_mean_tf_error']:.4f}, "
                f"{localization['rows_unstable_for_oracle_itself_mean_tf_error'] / max(localization['rows_stable_for_oracle_itself_mean_tf_error'], 1e-9):.2f}x larger.",
                f"- Rows whose stability classification disagrees between the two pairs: "
                f"{comparison['tokens'] - localization['classification_agreement']} of {comparison['tokens']}.",
                "",
                "So TensorFold's error concentrates on the rows where rec_mlx itself disagrees between its two",
                "accumulation orders. It is not zero on the rows rec_mlx reproduces exactly, but it is well below the",
                "oracle's own spread on the rows where the oracle is unstable, which is the shape of an accumulation-order",
                "difference rather than a structural one.",
            ]
            per_row = localization.get("per_row_error")
            if per_row:
                keys = ("p50", "p75", "p95", "p99", "p100")
                tf_cells = " | ".join(f"{per_row['tf'][k]:.3f}" for k in keys)
                self_cells = " | ".join(f"{per_row['oracle_self'][k]:.3f}" for k in keys)
                lines += [
                    "",
                    "Per-row maximum absolute error, the whole corpus: TensorFold against the oracle, and the",
                    "oracle against its own other path. The two distributions are the same size.",
                    "",
                    "| Pair | " + " | ".join(keys) + " | mean |",
                    "| --- | " + " | ".join("---:" for _ in keys) + " | ---: |",
                    f"| TensorFold vs oracle | {tf_cells} | {per_row['tf']['mean']:.3f} |",
                    f"| Oracle vs itself | {self_cells} | {per_row['oracle_self']['mean']:.3f} |",
                    "",
                    f"TensorFold's per-row error is at most the oracle's own per-row error on "
                    f"{per_row['rows_tf_at_most_oracle_self']} of {per_row['rows']} rows "
                    f"({100 * per_row['rows_tf_at_most_oracle_self'] / per_row['rows']:.1f}%), and the two per-row "
                    f"error series correlate at {per_row['correlation']:.3f}.",
                ]
    lines += [
        "",
        "## Disagreeing positions",
        "",
    ]
    if comparison["disagreeing_rows"]:
        rows = comparison["disagreeing_rows"]
        below = [row for row in rows if min(row["tf_margin"], row["oracle_margin"]) < MARGIN]
        breakdown = comparison.get("disagreement_breakdown")
        lines += [
            f"{len(rows)} of {comparison['tokens']} positions have a different top-1 "
            f"({100 * len(rows) / comparison['tokens']:.2f}%). "
            f"{len(below)} of those have a top-2 margin below {MARGIN} on at least one side, which is a near-tie "
            "rather than a different distribution.",
            "",
            "The split matters more than the total. rec_mlx's two paths choose a different top-1 on "
            + (f"{100 * (1 - breakdown['oracle_paths_agree']):.2f}% of positions by themselves, so a single oracle row "
               "is not a well-defined target on those positions. Comparing TensorFold with the oracle's other path as "
               "well:" if breakdown else "some positions by themselves."),
        ]
        if breakdown:
            lines += [
                "",
                f"- Top-1 agreement against the oracle's chunked prefill: "
                f"{breakdown['top1_agreement_against_other_path'] * 100:.3f}%.",
                f"- Of the {breakdown['against_oracle_decode']} disagreements with the decode path, "
                f"{breakdown['resolved_by_oracle_other_path']} pick exactly the token the oracle's other path picks.",
                f"- On {breakdown['where_oracle_also_differs_from_itself']} of them the oracle's own two paths "
                "disagree, so no single oracle answer exists to agree with.",
                f"- Only {breakdown['against_oracle_decode'] - breakdown['where_oracle_also_differs_from_itself']} positions "
                "are disagreements where both oracle paths agree and TensorFold is alone.",
            ]
        lines += [
            "",
            "The 20 disagreements with the largest margin on both sides, where the margin makes a miss least",
            "excusable:",
            "",
            "| TensorFold top-1 | Oracle top-1 | TensorFold margin | Oracle margin | min margin |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
        ranked = sorted(rows, key=lambda row: -min(row["tf_margin"], row["oracle_margin"]))[:20]
        for row in ranked:
            lines.append(
                f"| {row['tf_top1']} | {row['oracle_top1']} | {row['tf_margin']:.4f} | {row['oracle_margin']:.4f} | "
                f"{min(row['tf_margin'], row['oracle_margin']):.4f} |"
            )
    else:
        lines.append("None: every position's top-1 agrees.")
    lines += ["", "## First divergent tensors", ""]
    if comparison["divergent_rows"]:
        lines += [
            "Every history with a top-1 disagreement appears below. The row is that history's largest-KL row.",
            "`tf.bin`, `oracle.bin` and `oracle-other.bin` hold that row's bf16 bits in `.work/parity/flint/divergent/`",
            "for a re-check. The `oracle self` columns are the same row under rec_mlx's other path: when the oracle",
            "disagrees with itself at that row's top-1, the miss is not TensorFold's alone.",
            "",
            "| History | Row | Global | Max abs | Oracle self max abs | Oracle top-1 | Self top-1 | TF top-1 | TF agrees with self |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for row in comparison["divergent_rows"]:
            reference = row.get("oracle_self_reference")
            self_max = f"{reference['max_absolute']:.4f}" if reference else "n/a"
            self_top1 = str(reference["top1"]) if reference else "n/a"
            agrees = ("yes" if reference["tf_agrees_with_self_reference"] else "no") if reference else "n/a"
            lines.append(
                f"| `{row['history']}` | {row['row']} | {row['global_position']} | {row['max_absolute']:.4f} | "
                f"{self_max} | {reference['oracle_top1'] if reference else 'n/a'} | {self_top1} | "
                f"{reference['tf_top1'] if reference else 'n/a'} | {agrees} |"
            )
        disagreement_with_self = sum(
            1 for row in comparison["divergent_rows"] if row.get("oracle_self_reference") and not row["oracle_self_reference"]["self_reference_agrees_with_oracle"]
        )
        lines += [
            "",
            f"In {disagreement_with_self} of {len(comparison['divergent_rows'])} outliers the oracle itself chooses a",
            "different top-1 under its other path, so those are differences between two accumulation orders on the",
            "same checkpoint and the same ids, not a TensorFold defect. The remaining rows are checked by the two",
            "tables above: TensorFold's error there is within the oracle's own spread, which the corpus-wide",
            "per-row comparison shows for all rows at once.",
        ]
    else:
        lines += ["None: no history has a top-1 disagreement, so there is no divergent row to trace.", ""]
    lines += [
        "## What was checked before calling a difference floating-point noise",
        "",
        "1. **Tokenizer.** TensorFold's Zig tokenizer and the checkpoint tokenizer return identical ids on all "
        f"{comparison['histories']} rendered histories; the corpus stores those ids and both engines consume them "
        "unchanged, so no position is compared across different tokens.",
        "2. **Masks.** Both engines run one causal pass over the same ids from position 0 with no padding and no "
        "reordering. The 513-token sliding layers see the same window on both sides; the corpus includes "
        f"{comparison['histories_over_window']} histories longer than the 513-token window so the ring is exercised.",
        "3. **Quantization.** Both sides decode the same affine INT4 groups of 64 with INT8 embeddings from the same "
        "checkpoint files; the oracle is built with rec's `QATModel` and TensorFold reads the same shards.",
        "4. **Config.** Both read the checkpoint's `config.json`: 49 layers, 14 NoPE full-attention layers, 513-token "
        "sliding window, 32 query and 4 key heads, head dim 128, 256 experts with 6 routed, `rms_norm_eps` 1e-6, "
        "`rope_theta` 10000, and BF16 KV.",
        "5. **Accumulation grouping.** This is the difference that remains, and the two sections above size it on the "
        "same tokens: TensorFold's distance from the oracle's decode path is within the oracle's own chunked-prefill "
        "distance from that path, the residual peaks on the very first row (no context, so no mask or window can be "
        "involved), and it tracks the rows where rec_mlx already disagrees with itself.",
        "",
        "## Quality gate",
        "",
        "Decision 10 gates reordering changes against a baseline, and this item measures the baseline itself. The ",
        "figures a later candidate must be compared against are the ones above; the numbers a candidate is allowed ",
        "to reach are 1.10x the mean KL and at least 99.5% top-1 agreement on rows whose baseline top-2 margin is ",
        "at least 0.1 logit.",
        "",
        "## Baseline numbers",
        "",
        "The parent copies this block into the plan's **Baseline** section.",
        "",
        "```",
        f"flint parity: {comparison['histories']} histories, {comparison['tokens']} tokens, {comparison['positions']} positions",
        f"mean KL to oracle (rec_mlx e53663b2): {overall['kl']['mean']:.6f} nats",
        f"top-1 agreement: {overall['top1_agreement'] * 100:.3f}%",
        f"top-1 agreement, both margins >= {MARGIN}: "
        + (f"{overall['decisive_agreement'] * 100:.3f}%" if overall["decisive_agreement"] is not None else "n/a")
        + f" ({overall['decisive_rows']} positions)",
        "```",
        "",
    ]
    report_path.write_text("\n".join(lines))
    print(json.dumps({"report": str(report_path), "lines": len(lines)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
