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

HOST_FIELDS = ("product", "os", "memory_bytes", "recommended_max_working_set_bytes",
               "performance_cores", "efficiency_cores", "gpu_cores", "metal")
REQUIRED_CONTRACT = ("model_key", "model_revision", "model_aggregate_sha256", "quantization",
                     "host_signature", "context_policy", "output_policy", "cache_policy", "acceleration", "cell")


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


def describe(values) -> dict:
    values = sorted(values)
    if not values:
        return {"n": 0, "median": None, "min": None, "max": None}
    return {"n": len(values), "median": statistics.median(values), "min": values[0], "max": values[-1]}


def host_signature(host) -> dict | None:
    if not isinstance(host, dict):
        return None
    signature = {key: host.get(key) for key in HOST_FIELDS if key in host}
    return signature or None


def quant_label(key, hashes) -> str | None:
    if not key:
        return None
    for row in (hashes or {}).get("config") or []:
        if row.get("name", "").rsplit("/", 1)[-1] == "hf_quant_config.json":
            return f"{key}:{str(row.get('sha256', ''))[:12]}"
    return key.rsplit("-", 1)[-1] if "-" in key else key


def pairing_of(record: dict) -> dict:
    """The experiment/session/process-pair identity a record declares, or what a legacy record actually holds."""
    declared = record.get("pairing")
    binary = record.get("binary") or {}
    if isinstance(declared, dict):
        contract = declared.get("contract")
        return {"experiment": declared.get("experiment"), "session": declared.get("session"),
                "process_pair": declared.get("process_pair"),
                "role": declared.get("role") or binary.get("role"),
                "contract": dict(contract) if isinstance(contract, dict) else None}
    model = record.get("model") or {}
    hashes = model.get("hashes") or {}
    run_id, cell, mode, pair = record.get("run_id"), record.get("cell"), record.get("mode"), record.get("pair")
    process_pair = f"{run_id}:{cell}:{mode}:{pair}" if run_id is not None and pair is not None else None
    cache = record.get("prompt_cache_gib")
    contract = {
        "model_key": model.get("key"),
        "model_revision": hashes.get("revision"),
        "model_aggregate_sha256": hashes.get("aggregate_sha256"),
        "quantization": quant_label(model.get("key"), hashes),
        "host_signature": host_signature(record.get("host")),
        "output_policy": None,
        "cache_policy": {"prompt_cache_gib": cache} if cache is not None else None,
        "acceleration": {"mode": mode, "flags": list(record.get("mode_flags") or [])} if mode else None,
        "cell": cell,
    }
    return {"experiment": run_id, "session": run_id, "process_pair": process_pair,
            "role": binary.get("role"), "contract": contract}


def prompt_key(record: dict):
    prompt = record.get("prompt") or {}
    return (prompt.get("name"), prompt.get("sha256"))


def cold_rate(record: dict) -> float | None:
    value = ((record.get("requests") or {}).get("cold") or {}).get("decode_tps")
    return value if isinstance(value, (int, float)) and value > 0 else None


def contract_signature(contract) -> str:
    return json.dumps(contract, sort_keys=True, separators=(",", ":"))


def contract_problem(left, right) -> str | None:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return "contract not recorded"
    for field in REQUIRED_CONTRACT:
        if left.get(field) is None or right.get(field) is None:
            return f"contract field {field} not recorded"
    if contract_signature(left) != contract_signature(right):
        for field in REQUIRED_CONTRACT:
            if left.get(field) != right.get(field):
                return f"incompatible {field}: baseline {left.get(field)!r} vs candidate {right.get(field)!r}"
        return "incompatible contract"
    return None


def observed_problem(record: dict, contract: dict) -> str | None:
    acceleration = contract.get("acceleration") or {}
    declared = acceleration.get("mode")
    observed = record.get("acceleration_observed")
    if isinstance(observed, dict):
        if observed.get("declared") not in (None, declared):
            return f"acceleration observed as {observed.get('declared')!r}, declared {declared!r}"
        drafted = observed.get("drafted_tokens") or 0
        if declared == "ar" and drafted > 0:
            return f"ar mode drafted {drafted} tokens"
        if declared == "drafter" and drafted == 0:
            return "drafter mode drafted no tokens, so the declared acceleration did not run"
    cache = record.get("cache_observed")
    if isinstance(cache, dict):
        expected = (contract.get("cache_policy") or {}).get("prompt_cache_gib")
        if cache.get("prompt_cache_gib") != expected:
            return f"cache observed at {cache.get('prompt_cache_gib')} GiB, declared {expected}"
        if expected == 0 and (cache.get("cold_cached_tokens") or 0) > 0:
            return f"cold-prefill cell reported {cache['cold_cached_tokens']} cached tokens"
    return None


def unit_problem(unit: dict) -> str | None:
    baseline, candidate = unit["records"]["baseline"], unit["records"]["candidate"]
    if not baseline:
        return "missing baseline counterpart"
    if not candidate:
        return "missing candidate counterpart"
    for role in ("baseline", "candidate"):
        if unit["pairings"][role]["role"] != role:
            return f"{role} record declares role {unit['pairings'][role]['role']!r}"
    problem = contract_problem(unit["pairings"]["baseline"]["contract"],
                               unit["pairings"]["candidate"]["contract"])
    if problem:
        return problem
    for role, records in (("baseline", baseline), ("candidate", candidate)):
        contract = unit["pairings"][role]["contract"]
        for record in records:
            problem = observed_problem(record, contract)
            if problem:
                return f"{role} {problem}"
    for role, records in (("baseline", baseline), ("candidate", candidate)):
        keys = [prompt_key(record) for record in records]
        if any(sha is None for _, sha in keys):
            return f"{role} record has no prompt hash"
        if len(keys) != len(set(keys)):
            return f"duplicate {role} prompt counterpart"
    base_keys = {prompt_key(record) for record in baseline}
    cand_keys = {prompt_key(record) for record in candidate}
    if base_keys != cand_keys:
        missing = sorted(base_keys ^ cand_keys, key=str)
        return f"prompt counterparts differ: {missing}"
    for record in baseline:
        expected = (record.get("requests") or {}).get("cold") or {}
        value = expected.get("decode_tps")
        if not isinstance(value, (int, float)) or value <= 0:
            return "baseline record has no positive cold decode rate"
    for record in candidate:
        value = ((record.get("requests") or {}).get("cold") or {}).get("decode_tps")
        if not isinstance(value, (int, float)) or value <= 0:
            return "candidate record has no positive cold decode rate"
    return None


def build_unit(process_pair: str, unit: dict) -> dict:
    baseline = {prompt_key(record): record for record in unit["records"]["baseline"]}
    candidate = {prompt_key(record): record for record in unit["records"]["candidate"]}
    ratios = []
    for key in sorted(baseline, key=str):
        ratios.append(cold_rate(candidate[key]) / cold_rate(baseline[key]))
    contract = unit["pairings"]["baseline"]["contract"]
    ordered = sorted(baseline, key=str)
    return {"experiment": unit["pairings"]["baseline"]["experiment"],
            "session": unit["pairings"]["baseline"]["session"],
            "process_pair": process_pair, "contract": contract, "ratios": ratios,
            "pair_geomean": math.exp(statistics.fmean(math.log(ratio) for ratio in ratios)),
            "baseline_engine": baseline[ordered[0]].get("engine"),
            "candidate_engine": candidate[ordered[0]].get("engine"),
            "baseline_sha256": (baseline[ordered[0]].get("binary") or {}).get("sha256"),
            "candidate_sha256": (candidate[ordered[0]].get("binary") or {}).get("sha256"),
            "cell": contract.get("cell"), "mode": (contract.get("acceleration") or {}).get("mode"),
            "cache": (contract.get("cache_policy") or {}).get("prompt_cache_gib"),
            "prompts": len(ratios)}


def collect_pairs(trials: list[dict]) -> tuple[list[dict], list[dict]]:
    units: dict[str, dict] = {}
    for record in trials:
        if not record.get("valid") or cold_rate(record) is None:
            continue
        pairing = pairing_of(record)
        process_pair, role = pairing["process_pair"], pairing["role"]
        if not process_pair or role not in ("baseline", "candidate"):
            continue
        unit = units.setdefault(process_pair, {"pairings": {}, "records": {"baseline": [], "candidate": []}})
        unit["pairings"].setdefault(role, pairing)
        unit["records"][role].append(record)
    accepted, rejected = [], []
    for process_pair in sorted(units):
        unit = units[process_pair]
        for role in ("baseline", "candidate"):
            unit["pairings"].setdefault(role, {"role": None, "contract": None, "experiment": None,
                                               "session": None})
        problem = unit_problem(unit)
        if problem:
            rejected.append({"process_pair": process_pair, "reason": problem,
                             "experiment": unit["pairings"]["baseline"]["experiment"],
                             "session": unit["pairings"]["baseline"]["session"]})
        else:
            accepted.append(build_unit(process_pair, unit))
    return accepted, rejected


def paired_comparisons(units: list[dict]) -> list[dict]:
    """Combine accepted process pairs only inside one declared experiment with one matching contract."""
    groups: dict[tuple, list[dict]] = {}
    for unit in units:
        key = (unit["experiment"], contract_signature(unit["contract"]),
               unit["baseline_sha256"], unit["candidate_sha256"])
        groups.setdefault(key, []).append(unit)
    out = []
    for key in sorted(groups, key=lambda item: (str(item[0]), item[1], str(item[2]), str(item[3]))):
        members = groups[key]
        logs = [math.log(unit["pair_geomean"]) for unit in members]
        sessions = len({unit["session"] for unit in members})
        summary = {"n": len(logs), "median": statistics.fmean(logs), "min": min(logs), "max": max(logs)}
        low = high = None
        if len(logs) >= 2:
            half = T975.get(len(logs) - 1, 1.96) * statistics.stdev(logs) / math.sqrt(len(logs))
            low, high = summary["median"] - half, summary["median"] + half
        first = members[0]
        out.append({"experiment": first["experiment"], "mode": first["mode"], "cell": first["cell"],
                    "cache": first["cache"], "baseline": first["baseline_engine"],
                    "candidate": first["candidate_engine"], "pairs": len(logs), "sessions": sessions,
                    "speedup": math.exp(summary["median"]),
                    "low": math.exp(low) if low is not None else None,
                    "high": math.exp(high) if high is not None else None,
                    "min": math.exp(summary["min"]), "max": math.exp(summary["max"]),
                    "repeatability": len(logs) >= 6 and sessions >= 2})
    return out


def measure(records: list[dict], getter) -> dict:
    """Descriptive values per process pair; the pair, not a prompt inside it, is the repetition unit."""
    by_pair: dict[str, list[float]] = {}
    for record in records:
        value = getter(record)
        if isinstance(value, (int, float)):
            by_pair.setdefault(pairing_of(record)["process_pair"] or "unpaired", []).append(value)
    pair_medians = [statistics.median(values) for _, values in sorted(by_pair.items(), key=lambda item: str(item[0]))]
    sessions = {pairing_of(record)["session"] for record in records if pairing_of(record)["session"]}
    values = pair_medians if len(pair_medians) >= 2 else [value for values in by_pair.values() for value in values]
    summary = describe(values)
    summary["pairs"] = len(pair_medians)
    summary["sessions"] = len(sessions)
    summary["repeatability"] = len(pair_medians) >= 6 and len(sessions) >= 2
    return summary


def with_interval(summary: dict, digits: int = 1) -> str:
    if summary["n"] == 0:
        return "—"
    if summary["n"] == 1:
        return number(summary["median"], digits)
    return f"{number(summary['median'], digits)} [{number(summary['min'], digits)}–{number(summary['max'], digits)}]"


def with_range(summary: dict, digits: int = 2) -> str:
    return with_interval(summary, digits)


def gib_interval(summary: dict) -> str:
    if summary["n"] == 0:
        return "—"
    if summary["n"] == 1:
        return gib(summary["median"])
    return f"{gib(summary['median'])} [{gib(summary['min'])}–{gib(summary['max'])}]"


def text(value) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value) or "—"
    return "—" if value is None else str(value)


def group(trials: list[dict]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for record in trials:
        if not record.get("valid"):
            continue
        if cold_rate(record) is None:
            continue
        key = (record.get("engine"), record.get("mode"), record.get("cell"), record.get("prompt_cache_gib"))
        groups.setdefault(key, []).append(record)
    return groups


def summarize(records: list[dict]) -> dict:
    def get(path):
        def read(record):
            value = record
            for part in path:
                value = (value or {}).get(part) if isinstance(value, dict) else None
            return value if isinstance(value, (int, float)) else None
        return read

    def prefill(record):
        cold = (record.get("requests") or {}).get("cold") or {}
        prompt, seconds = cold.get("prompt_tokens"), cold.get("engine_prefill_s")
        return prompt / seconds if isinstance(prompt, int) and isinstance(seconds, (int, float)) and seconds > 0 else None

    def acceptance(record):
        cold = (record.get("requests") or {}).get("cold") or {}
        value = cold.get("acceptance_rate")
        return value if isinstance(value, (int, float)) and (cold.get("drafted_tokens") or 0) > 0 else None

    return {"decode": measure(records, lambda r: (r.get("requests") or {}).get("cold", {}).get("decode_tps")),
            "decode_warm": measure(records, lambda r: (r.get("requests") or {}).get("warm", {}).get("decode_tps")),
            "prefill": measure(records, prefill),
            "ttft_ms": measure(records, lambda r: ((r.get("requests") or {}).get("cold") or {}).get("ttft_s") * 1000),
            "total_s": measure(records, lambda r: ((r.get("requests") or {}).get("cold") or {}).get("total_s")),
            "sampled_peak": measure(records, lambda r: (r.get("footprint") or {}).get("peak_bytes")),
            "lifetime_peak": measure(records, lambda r: (r.get("footprint") or {}).get("process_lifetime_peak_bytes")),
            "device_peak": measure(records, lambda r: (r.get("footprint") or {}).get("device_peak_bytes")),
            "acceptance": measure(records, acceptance),
            "committed": measure(records, lambda r: (r.get("requests") or {}).get("cold", {}).get("committed_per_verification")),
            "speculative_trials": len([r for r in records if ((r.get("requests") or {}).get("cold") or {}).get("verification_rounds")]),
            "pairs": len({pairing_of(r)["process_pair"] for r in records if pairing_of(r)["process_pair"]}),
            "sessions": len({pairing_of(r)["session"] for r in records if pairing_of(r)["session"]})}


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


def repeatability_text(summary: dict) -> str:
    if summary["pairs"] >= 6 and summary["sessions"] >= 2:
        return "measured"
    return "unmeasured"


def render(trials: list[dict], unavailable: list[dict], files: list[str], malformed: int) -> str:
    groups = group(trials)
    valid = [r for r in trials if r.get("valid")]
    invalid = [r for r in trials if not r.get("valid")]
    summaries = {key: summarize(records) for key, records in groups.items()}
    lines = ["# Benchmark scorecard", "",
             "Generated from `campaign/runs/` by `make -C campaign scorecard`. Rates are medians; brackets are "
             "descriptive ranges across process-pair medians (or across prompts when only one pair ran). "
             "Repeatability is measured only by the six-pair, two-session protocol; otherwise it is marked "
             "\"unmeasured\" even when a range is shown.", "",
             f"- Run files: {len(files)}", f"- Trials recorded: {len(trials)} ({len(valid)} valid, {len(invalid)} invalid)",
             f"- Unavailable cell records: {len(unavailable)}", f"- Malformed lines skipped: {malformed}", ""]

    lines += ["## Matched", "",
              "Same engine, mode, cell and prompt-cache size. Sampled request peak is the `/memory` high-water mark "
              "after readiness; lifetime peak is the kernel's whole-process high-water mark and is not gated on the "
              "250 ms samples.", "",
              "| Engine | Mode | Cell | Cache GiB | Trials | Pairs | Cold decode tok/s | Warm decode tok/s | Prefill tok/s | TTFT ms | Total s | Sampled request peak GiB | Lifetime peak GiB | Accept | Tokens/round | Repeatability |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for key in sorted(groups, key=lambda k: (str(k[0]), str(k[1]), k[2] or 0, k[3] if k[3] is not None else -1)):
        engine, mode, cell, cache = key
        summary = summaries[key]
        lines.append(
            f"| {engine} | {mode} | {cell} | {cache} | {summary['decode']['n']} | {summary['pairs']} | "
            f"{with_interval(summary['decode'])} | "
            f"{with_interval(summary['decode_warm'])} | {with_interval(summary['prefill'])} | "
            f"{with_interval(summary['ttft_ms'])} | {with_interval(summary['total_s'], 2)} | "
            f"{gib_interval(summary['sampled_peak'])} | {gib_interval(summary['lifetime_peak'])} | "
            f"{with_range(summary['acceptance'], 3)} | {with_range(summary['committed'], 2)} | "
            f"{repeatability_text(summary['decode'])} |")
    if not groups:
        lines.append("| — | — | — | — | 0 | — | — | — | — | — | — | — | — | — | — | — |")

    units, rejected = collect_pairs(trials)
    comparisons = paired_comparisons(units)
    lines += ["", "### Paired baseline/candidate", "",
              "Speedup is candidate cold decode throughput over baseline; a value above 1.0 is faster. A unit is one "
              "process pair whose baseline and candidate records share prompt hashes, model and quantization, host, "
              "cell and output policy, prompt-cache policy and acceleration mode. Pairs combine only inside one "
              "declared experiment, and the interval is a t interval on per-pair log ratios; with fewer than six "
              "pairs across two sessions, it is descriptive and repeatability is unmeasured. Weights are predeclared: "
              "each prompt within a cell carries equal weight, and each independent process pair carries equal weight "
              "in the interval.", ""]
    if comparisons:
        lines += ["| Experiment | Mode | Cell | Cache GiB | Baseline | Candidate | Pairs | Sessions | Speedup (cand/base) | 95% CI | Repeatability |",
                  "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | --- | --- |"]
        for row in comparisons:
            interval = f"{row['low']:.3f}–{row['high']:.3f}" if row["low"] is not None else "—"
            lines.append(f"| {row['experiment']} | {row['mode']} | {row['cell']} | {row['cache']} | "
                         f"{row['baseline']} | {row['candidate']} | {row['pairs']} | {row['sessions']} | "
                         f"{row['speedup']:.3f} | {interval} | "
                         f"{'measured' if row['repeatability'] else 'unmeasured'} |")
    else:
        lines.append("No run pairs a baseline and a candidate process under one declared experiment with matching "
                     "prompt, model, host, policy and acceleration contracts.")

    if rejected:
        lines += ["", "#### Rejected pairs", "",
                  "These process pairs were not combined into a matched result.", "",
                  "| Experiment | Process pair | Reason |", "| --- | --- | --- |"]
        for row in rejected:
            lines.append(f"| {row['experiment']} | {row['process_pair']} | {row['reason']} |")

    lines += ["", "## Best useful", "",
              "The fastest measured configuration per cell, among engines, modes and configurations that actually "
              "ran. Heterogeneous configurations stay here rather than entering a matched result.", ""]
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
