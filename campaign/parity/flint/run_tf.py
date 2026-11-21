"""Write every history's ids for tf-flint-run and run it, one process per history, through campaign/heavy.py (GPU lock and memory cap).

tf-flint-run takes the ids file directly, so this only has to place them where it expects and refuse to start
if the corpus and the checkpoint disagree on shape.
"""

import json
import subprocess
import sys
from pathlib import Path

VOCAB = 150144


def main() -> int:
    corpus_path, binary, model_dir, out_dir, heavy = (Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]))
    corpus = json.loads(corpus_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    ids_dir = out_dir / "ids"
    ids_dir.mkdir(exist_ok=True)
    for entry in corpus["histories"]:
        ids_path = ids_dir / f"{entry['name']}.json"
        ids_path.write_text(json.dumps(entry["ids"]))
        logits_path = out_dir / f"{entry['name']}.bin"
        subprocess.run(["python3", str(heavy), "--", str(binary), str(model_dir), str(ids_path), "logits", str(logits_path)], check=True)
        if logits_path.stat().st_size != entry["tokens"] * VOCAB * 2:
            raise SystemExit(f"{entry['name']}: {logits_path.stat().st_size} bytes, expected {entry['tokens'] * VOCAB * 2}")
        print(json.dumps({"history": entry["name"], "tokens": entry["tokens"], "bytes": logits_path.stat().st_size}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
