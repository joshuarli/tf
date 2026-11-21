"""Regenerate campaign/scorecard.md from campaign/runs alone, deterministically."""

import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUTPUT = HERE / "scorecard.md"

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
        18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}

METRICS = ("decode_tps", "prompt_ttft_tps", "weighted_prefill_tps", "ttft_s", "total_s")


def gib(value: float | None) -> str:
    return "—" if value is None else f"{value / 2**30:.2f}"


def number(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def load() -> tuple[list[dict], list[dict], list[str], int]:
    trials, unavailable, files, malformed = [], [], [], 0
    if not RUNS.is_dir():
        return trials, unavailable, files, malformed
    for path in sorted(RUNS.glob("**/*.jsonl")):
        files.append(path.relative_to(HERE.parent).as_posix())
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                malformed += 1
                continue
            if record.get("kind") == "unavailable":
                unavailable.append(record)
            elif record.get("kind") == "trial":
                trials.append(record)
            else:
                malformed += 1
    return trials, unavailable, files, malformed


def stats(values: list[float]) -> dict:
    values = sorted(values)
    if not values:
        return {"n": 0, "median": None, "lo": None, "hi": None, "min": None, "max": None}
    median = statistics.median(values)
    if len(values) < 2:
        return {"n": 1, "median": median, "lo": None, "hi": None, "min": median, "max": median}
    half = T975.get(len(values) - 1, 1.96) * statistics.stdev(values) / math.sqrt(len(values))
    return {"n": len(values), "median": median, "lo": median - half, "hi": median + half,
            "min": values[0], "max": values[-1]}


def with_interval(summary: dict, digits: int = 1) -> str:
    if summary["n"] == 0:
        return "—"
    if summary["n"] == 1:
        return number(summary["median"], digits)
    return f"{number(summary['median'], digits)} [{number(summary['lo'], digits)}–{number(summary['hi'], digits)}]"


def with_range(summary: dict, digits: int = 2) -> str:
    if summary["n"] == 0:
        return "—"
    if summary["n"] == 1:
        return number(summary["median"], digits)
    return f"{number(summary['median'], digits)} [{number(summary['min'], digits)}–{number(summary['max'], digits)}]"


def gib_interval(summary: dict) -> str:
    if summary["n"] == 0:
        return "—"
    if summary["n"] == 1:
        return gib(summary["median"])
    return f"{gib(summary['median'])} [{gib(summary['lo'])}–{gib(summary['hi'])}]"


def text(value) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value) or "—"
    return "—" if value is None else str(value)


def group(trials: list[dict]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for record in trials:
        if not record.get("valid"):
            continue
        requests = record.get("requests") or {}
        cold = requests.get("cold") or {}
        if not cold.get("decode_tps"):
            continue
        key = (record.get("engine"), record.get("mode"), record.get("cell"), record.get("prompt_cache_gib"))
        groups.setdefault(key, []).append(record)
    return groups


def summarize(records: list[dict]) -> dict:
    cold_decode = [r["requests"]["cold"]["decode_tps"] for r in records if r["requests"].get("cold", {}).get("decode_tps")]
    warm_decode = [r["requests"]["warm"]["decode_tps"] for r in records if r["requests"].get("warm", {}).get("decode_tps")]
    prefill = []
    for r in records:
        cold = r["requests"].get("cold") or {}
        prompt, seconds = cold.get("prompt_tokens"), cold.get("engine_prefill_s")
        if isinstance(prompt, int) and isinstance(seconds, (int, float)) and seconds > 0:
            prefill.append(prompt / seconds)
    ttft = [r["requests"]["cold"]["ttft_s"] * 1000 for r in records if r["requests"].get("cold", {}).get("ttft_s")]
    totals = [r["requests"]["cold"]["total_s"] for r in records if r["requests"].get("cold", {}).get("total_s")]
    peak = [r["footprint"]["peak_bytes"] for r in records
            if (r.get("footprint") or {}).get("peak_bytes")]
    accept = [r["requests"]["cold"]["acceptance_rate"] for r in records
              if isinstance(r["requests"].get("cold", {}).get("acceptance_rate"), (int, float))
              and (r["requests"]["cold"].get("drafted_tokens") or 0) > 0]
    rounds = [r["requests"]["cold"]["committed_per_verification"] for r in records
              if isinstance(r["requests"].get("cold", {}).get("committed_per_verification"), (int, float))]
    speculative = [r for r in records if (r["requests"].get("cold") or {}).get("verification_rounds")]
    return {"pairs": len({r.get("pair") for r in records if r.get("pair") is not None}),
            "decode": stats(cold_decode), "decode_warm": stats(warm_decode),
            "prefill": stats(prefill), "ttft_ms": stats(ttft), "total_s": stats(totals),
            "peak": stats(peak), "acceptance": stats(accept), "committed": stats(rounds),
            "speculative_trials": len(speculative)}


def natural(records: list[dict]) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    for record in records:
        run = record.get("natural_termination") or {}
        if not run:
            continue
        key = (record.get("engine"), record.get("mode"), record.get("cell"), record.get("prompt_cache_gib"))
        entry = out.setdefault(key, {"classes": {}, "reasons": {}, "tokens": []})
        entry["classes"][run.get("completion")] = entry["classes"].get(run.get("completion"), 0) + 1
        for reason in run.get("finish_reasons") or []:
            entry["reasons"][reason] = entry["reasons"].get(reason, 0) + 1
        if isinstance(run.get("tokens"), int):
            entry["tokens"].append(run["tokens"])
    return out


def paired_comparisons(groups: dict[tuple, list[dict]]) -> list[dict]:
    """Per-pair baseline/candidate decode ratios for groups that share mode, cell and cache."""
    by_role: dict[tuple, dict[int, float]] = {}
    for (engine, mode, cell, cache), records in groups.items():
        for record in records:
            role = (record.get("binary") or {}).get("role")
            value = record["requests"]["cold"].get("decode_tps")
            if role and value:
                by_role.setdefault((mode, cell, cache), {}).setdefault(engine, {}).setdefault(
                    record.get("pair", 0), []).append(value)
    out = []
    for (mode, cell, cache), engines in sorted(by_role.items(), key=lambda item: (item[0][1], item[0][0])):
        names = sorted(engines)
        for index, baseline in enumerate(names):
            for candidate in names[index + 1:]:
                ratios = []
                for pair in sorted(set(engines[baseline]) & set(engines[candidate])):
                    b = statistics.median(engines[baseline][pair])
                    c = statistics.median(engines[candidate][pair])
                    if b and c:
                        ratios.append(b / c)
                if ratios:
                    out.append({"mode": mode, "cell": cell, "cache": cache, "baseline": baseline,
                                "candidate": candidate, "ratios": ratios,
                                "stats": stats([math.log(r) for r in ratios]),
                                "geomean": math.exp(statistics.fmean(math.log(r) for r in ratios))})
    return out


def render(trials: list[dict], unavailable: list[dict], files: list[str], malformed: int) -> str:
    groups = group(trials)
    valid = [r for r in trials if r.get("valid")]
    invalid = [r for r in trials if not r.get("valid")]
    summaries = {key: summarize(records) for key, records in groups.items()}
    lines = ["# Benchmark scorecard", "",
             "Generated from `campaign/runs/` by `make -C campaign scorecard`. Rates are medians over measured "
             "trials; brackets are a 95% confidence interval when more than one trial entered a cell.", "",
             f"- Run files: {len(files)}", f"- Trials recorded: {len(trials)} ({len(valid)} valid, {len(invalid)} invalid)",
             f"- Unavailable cell records: {len(unavailable)}", f"- Malformed lines skipped: {malformed}", ""]

    lines += ["## Matched", "",
              "Same engine, mode, cell and prompt-cache size, so every row compares like with like.", "",
              "| Engine | Mode | Cell | Cache GiB | Trials | Pairs | Cold decode tok/s | Warm decode tok/s | Prefill tok/s | TTFT ms | Total s | Peak GiB | Accept | Tokens/round |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key in sorted(groups, key=lambda k: (str(k[0]), str(k[1]), k[2] or 0, k[3] if k[3] is not None else -1)):
        engine, mode, cell, cache = key
        summary = summaries[key]
        lines.append(
            f"| {engine} | {mode} | {cell} | {cache} | {summary['decode']['n']} | {summary['pairs']} | "
            f"{with_interval(summary['decode'])} | "
            f"{with_interval(summary['decode_warm'])} | {with_interval(summary['prefill'])} | "
            f"{with_interval(summary['ttft_ms'])} | {with_interval(summary['total_s'], 2)} | "
            f"{gib_interval(summary['peak'])} | {with_range(summary['acceptance'], 3)} | "
            f"{with_range(summary['committed'], 2)} |")
    if not groups:
        lines.append("| — | — | — | — | 0 | — | — | — | — | — | — | — | — |")

    comparisons = paired_comparisons(groups)
    lines += ["", "### Paired baseline/candidate", "",
              "Per-pair ratios of baseline cold decode rate to candidate; the interval is on the log ratios.", ""]
    if comparisons:
        lines += ["| Mode | Cell | Cache GiB | Baseline | Candidate | Pairs | Baseline/candidate | 95% CI |",
                  "| --- | ---: | ---: | --- | --- | ---: | ---: | --- |"]
        for row in comparisons:
            interval = ""
            if row["stats"]["n"] > 1:
                low = math.exp(row["stats"]["lo"])
                high = math.exp(row["stats"]["hi"])
                interval = f"{low:.3f}–{high:.3f}"
            lines.append(f"| {row['mode']} | {row['cell']} | {row['cache']} | {row['baseline']} | {row['candidate']} | "
                         f"{row['stats']['n']} | {row['geomean']:.3f} | {interval or '—'} |")
    else:
        lines.append("No run pairs a baseline and a candidate process for the same mode and cell.")

    lines += ["", "## Best useful", "",
              "The fastest measured configuration per cell, among engines and modes that actually ran.", ""]
    best: dict[tuple, list[tuple[float, tuple]]] = {}
    for key, summary in summaries.items():
        if summary["decode"]["n"]:
            best.setdefault((key[2], key[3]), []).append((summary["decode"]["median"], key))
    lines += ["| Cell | Cache GiB | Best engine | Mode | Cold decode tok/s | Trials | Other measured |",
              "| ---: | ---: | --- | --- | ---: | ---: | --- |"]
    if best:
        for cell, cache in sorted(best, key=lambda item: (item[0] or 0, item[1] if item[1] is not None else -1)):
            rows = sorted(best[(cell, cache)], key=lambda item: (-item[0], str(item[1])))
            rate, key = rows[0]
            others = ", ".join(f"{k[0]} {k[1]} {number(summaries[k]['decode']['median'])}" for _, k in rows[1:]) or "—"
            lines.append(f"| {cell} | {cache} | {key[0]} | {key[1]} | {number(rate)} | "
                         f"{summaries[key]['decode']['n']} | {others} |")
    else:
        lines.append("| — | — | — | — | — | 0 | — |")

    lines += ["", "## Natural termination", "",
              "Requests sent without `ignore_eos` (they do not enter the fixed-length rates above).", ""]
    outcomes = natural(trials)
    if outcomes:
        lines += ["| Engine | Mode | Cell | Cache GiB | Completions | Finish reasons | Mean tokens |",
                  "| --- | --- | ---: | ---: | --- | --- | ---: |"]
        for key in sorted(outcomes, key=lambda k: (str(k[0]), str(k[1]), k[2] or 0, k[3] if k[3] is not None else -1)):
            entry = outcomes[key]
            completions = ", ".join(f"{name} {count}" for name, count in sorted(entry["classes"].items(), key=str)) or "—"
            reasons = ", ".join(f"{name} {count}" for name, count in sorted(entry["reasons"].items(), key=str)) or "—"
            mean = statistics.fmean(entry["tokens"]) if entry["tokens"] else None
            lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {completions} | {reasons} | {number(mean)} |")
    else:
        lines.append("No natural-termination requests were recorded.")

    lines += ["", "## Unavailable", ""]
    if unavailable:
        lines += ["| Engine | Mode | Cell | Reason |", "| --- | --- | ---: | --- |"]
        seen = set()
        for record in sorted(unavailable, key=lambda r: (str(r.get("engine")), str(r.get("mode")), r.get("cell") or 0,
                                                         str(r.get("reason")))):
            key = (record.get("engine"), record.get("mode"), record.get("cell"), record.get("reason"))
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"| {record.get('engine')} | {record.get('mode')} | {record.get('cell')} | "
                         f"{record.get('reason')} |")
    else:
        lines.append("Every recorded mode had a runnable configuration.")

    if invalid:
        lines += ["", "## Invalid trials", "",
                  "Host pressure, throttling or a short reply marked these out of the rates.", "",
                  "| Engine | Mode | Cell | Pair | Reasons |", "| --- | --- | ---: | ---: | --- |"]
        for record in sorted(invalid, key=lambda r: (str(r.get("engine")), str(r.get("mode")), r.get("cell") or 0,
                                                      r.get("pair") or 0, str(r.get("invalid_reasons")))):
            reasons = "; ".join(str(reason) for reason in record.get("invalid_reasons") or []) or text(record.get("error"))
            lines.append(f"| {record.get('engine')} | {record.get('mode')} | {record.get('cell')} | "
                         f"{record.get('pair')} | {reasons} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    trials, unavailable, files, malformed = load()
    document = render(trials, unavailable, files, malformed)
    OUTPUT.write_text(document, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(HERE.parent)}: {len(trials)} trials, {len(unavailable)} unavailable records",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
