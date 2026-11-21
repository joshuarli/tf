"""Assert TensorFold's Zig tokenizer and the oracle's tokenizer agree on every rendered history, then stop.

The corpora's ids come from the checkpoint tokenizer the oracle loads. TensorFold's engine has no tokenizer, so its
own tokenizer (zig/src/core/tokenizer, driven by tokenizer-check) must reproduce those ids on the same text before
either engine's logits mean anything. Renders each history's text once and compares ids, not decoded text.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    corpus_path, checker, tokenizer_path = (Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    corpus = json.loads(corpus_path.read_text())
    cases = {"encode": [entry["text"].encode().hex() for entry in corpus["histories"]], "decode": []}
    with tempfile.TemporaryDirectory() as folder:
        cases_path, out_path = Path(folder) / "cases.json", Path(folder) / "out.json"
        cases_path.write_text(json.dumps(cases))
        subprocess.run([str(checker), str(tokenizer_path), str(cases_path), str(out_path)], check=True)
        got = json.loads(out_path.read_text())["encode"]
    failures = []
    for entry, ids in zip(corpus["histories"], got, strict=True):
        if ids != entry["ids"]:
            first = next((i for i, (a, b) in enumerate(zip(ids, entry["ids"])) if a != b), min(len(ids), len(entry["ids"])))
            failures.append({"history": entry["name"], "first_divergence": first, "count": len(entry["ids"]), "zig": len(ids)})
    if failures:
        print(json.dumps({"agreement": False, "failures": failures}, indent=1))
        return 1
    print(json.dumps({"agreement": True, "histories": len(got), "tokens": sum(len(ids) for ids in got)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
