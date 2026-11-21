"""Write or verify the suite manifest: every case's id, kind, source, licence and sha256."""

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CASES = HERE / "cases.json"
MANIFEST = HERE / "manifest.json"


def digest(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def build(document: dict) -> dict:
    entries = []
    for case in document["cases"]:
        entry = {"id": case["id"], "kind": case["kind"], "sets": case["sets"], "source": case["source"],
                 "licence": case["licence"], "sha256": digest(case)}
        if case["kind"] == "retrieval":
            path = REPO / case["prompt_file"]
            data = path.read_bytes()
            entry["files"] = [{"path": case["prompt_file"], "sha256": hashlib.sha256(data).hexdigest(),
                               "bytes": len(data), "cell": case["cell"]}]
        entries.append(entry)
    counts = {}
    for entry in entries:
        counts[entry["kind"]] = counts.get(entry["kind"], 0) + 1
    return {"schema": 1, "suite": {"cases": len(entries), "dev": sum(1 for c in entries if "dev" in c["sets"]),
                                   "kinds": counts,
                                   "prompt_tokens": {"4k": 4096, "32k": 32768}},
            "cases_sha256": hashlib.sha256(CASES.read_bytes()).hexdigest(), "cases": entries}


def main() -> int:
    document = json.loads(CASES.read_text(encoding="utf-8"))
    manifest = build(document)
    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if "--check" in sys.argv[1:]:
        if not MANIFEST.is_file() or MANIFEST.read_text(encoding="utf-8") != text:
            print(f"{MANIFEST} does not match cases.json and the retrieval prompts", file=sys.stderr)
            return 1
        print(f"{MANIFEST} matches {manifest['suite']['cases']} cases", file=sys.stderr)
        return 0
    MANIFEST.write_text(text, encoding="utf-8")
    print(f"wrote {MANIFEST} with {manifest['suite']['cases']} cases", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
