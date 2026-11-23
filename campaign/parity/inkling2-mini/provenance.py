"""Receipts that key each candidate output to the exact inputs that produced it.

A receipt sits beside its output (``<name>.bin`` gets ``<name>.receipt.json``) and
records the runner binary sha256, the model identity digest (checkpoint
directory listing plus full hashes of its small config/quant descriptor files),
the token-history sha256, the arithmetic policy string, and the output's own
shape and digest. ``check`` fails when any input changed, so the caller
regenerates instead of reusing a stale file; ``omlx_gate.py`` refuses outputs
whose receipt is missing or names different inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ARITHMETIC_POLICY = "f32-activations"

SMALL_FILE_LIMIT = 64 * 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def token_history_sha(token_ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(token_ids).encode("utf-8")).hexdigest()


def model_identity(model_dir: Path) -> str:
    manifest: list[dict] = []
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        entry: dict = {
            "path": str(path.relative_to(model_dir)),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if stat.st_size <= SMALL_FILE_LIMIT:
            entry["sha256"] = sha256_file(path)
        manifest.append(entry)
    return hashlib.sha256(json.dumps(manifest).encode("utf-8")).hexdigest()


def capture_digest(output_sha256: dict[str, str]) -> str:
    """One digest naming a set of captures: the sha256 of their name-to-output-sha256 map, keys sorted."""
    return hashlib.sha256(json.dumps(output_sha256, sort_keys=True).encode("utf-8")).hexdigest()


def receipt_path_for(output: Path) -> Path:
    return output.with_name(output.name + ".receipt.json")


def write_receipt(output: Path, *, runner_sha256: str, model_identity_digest: str,
                  token_sha256: str, policy: str, vocab: int, rows: int) -> Path:
    receipt = {
        "schema": 1,
        "runner_sha256": runner_sha256,
        "model_identity": model_identity_digest,
        "token_sha256": token_sha256,
        "arithmetic_policy": policy,
        "vocab": vocab,
        "rows": rows,
        "output_sha256": sha256_file(output),
    }
    path = receipt_path_for(output)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return path


def check_receipt(output: Path, *, runner_sha256: str, model_identity_digest: str,
                  token_sha256: str, policy: str, vocab: int, rows: int) -> list[str]:
    if not output.exists():
        return [f"{output.name}: no output file"]
    path = receipt_path_for(output)
    if not path.exists():
        return [f"{output.name}: no receipt at {path.name}; refusing a file of unknown provenance"]
    try:
        receipt = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return [f"{output.name}: unreadable receipt: {exc}"]
    expected = {"runner_sha256": runner_sha256, "model_identity": model_identity_digest,
                "token_sha256": token_sha256, "arithmetic_policy": policy,
                "vocab": vocab, "rows": rows}
    problems = []
    for key, want in expected.items():
        if receipt.get(key) != want:
            problems.append(f"{output.name}: receipt names {key} {receipt.get(key)!r}, "
                            f"current inputs need {want!r}")
    return problems


def _current_inputs(args) -> tuple[str, str, str, int]:
    runner_sha256 = sha256_file(Path(args.runner))
    model_digest = model_identity(Path(args.model))
    tokens = json.loads(Path(args.tokens).read_text())
    rows = len(tokens)
    return runner_sha256, model_digest, token_history_sha(tokens), rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["write", "check"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--policy", default=ARITHMETIC_POLICY)
    parser.add_argument("--vocab", type=int, required=True)
    args = parser.parse_args()

    output = Path(args.output)
    runner_sha256, model_digest, token_sha, rows = _current_inputs(args)
    if args.action == "write":
        size_rows = output.stat().st_size // 4 // args.vocab
        if size_rows != rows:
            print(f"{output.name}: file holds {size_rows} rows, tokens name {rows}", file=sys.stderr)
            return 1
        write_receipt(output, runner_sha256=runner_sha256, model_identity_digest=model_digest,
                      token_sha256=token_sha, policy=args.policy, vocab=args.vocab, rows=rows)
        print(f"{output.name}: receipt written for {rows} rows")
        return 0
    problems = check_receipt(output, runner_sha256=runner_sha256,
                             model_identity_digest=model_digest, token_sha256=token_sha,
                             policy=args.policy, vocab=args.vocab, rows=rows)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        return 1
    print(f"{output.name}: receipt matches the current inputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
