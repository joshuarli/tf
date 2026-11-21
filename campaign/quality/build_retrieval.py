"""Build the eight context-retrieval prompts from the fixed campaign prompt corpus, at 4K and 32K prompt tokens."""

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "campaign" / "prompts"))

from flint_tokenizer import FlintTokenizer  # noqa: E402

CASES = HERE / "cases.json"
TOKENIZER = Path.home() / "models" / "flint-int4@3b93f9053b53" / "tokenizer.json"
CHAT_OVERHEAD = 2


def paragraph_cut(text: str, position: float) -> int:
    wanted = int(len(text) * position)
    found = text.find("\n\n", wanted)
    return found if found >= 0 else len(text)


def clean_prefix(tokenizer: FlintTokenizer, tokens: list[int], keep: int) -> str:
    while keep > 0:
        try:
            return tokenizer.decode(tokens[:keep])
        except UnicodeDecodeError:
            keep -= 1
    return ""


def compose(tokenizer: FlintTokenizer, source: str, needle: str, question: str, cell: int,
            position: float) -> tuple[str, int]:
    target = cell - CHAT_OVERHEAD
    block = "\n\n" + needle + "\n\n"
    question_block = "\n\n" + question
    tokens = tokenizer.encode(source)
    keep = target - len(tokenizer.encode(block)) - len(tokenizer.encode(question_block))
    best = None
    for _ in range(80):
        keep = max(1, min(len(tokens), keep))
        left = clean_prefix(tokenizer, tokens, keep)
        cut = paragraph_cut(left, position)
        content = left[:cut] + block + left[cut:] + question_block
        count = len(tokenizer.encode(content))
        if best is None or abs(count - target) < abs(best[1] - target) or \
                (abs(count - target) == abs(best[1] - target) and count > best[1]):
            best = (content, count)
        if count == target:
            break
        moved = max(1, min(len(tokens), keep + target - count))
        if moved == keep:
            break
        keep = moved
    if best is None or abs(best[1] - target) > 2:
        raise ValueError(f"could not reach {target} content tokens, best was {best[1] if best else None}")
    if best[0].count(needle) != 1:
        raise ValueError("the needle does not appear exactly once")
    return best


def build(document: dict) -> list[dict]:
    tokenizer = FlintTokenizer(TOKENIZER)
    written = []
    for entry in document["cases"]:
        if entry["kind"] != "retrieval":
            continue
        source = (REPO / entry["source"]).read_text(encoding="utf-8")
        for marker in (entry["needle"], entry["answer"]):
            if marker in source:
                raise ValueError(f"{entry['id']}: {marker!r} already occurs in {entry['source']}")
        text, tokens = compose(tokenizer, source, entry["needle"], entry["question"], entry["cell"],
                               entry["position"])
        path = REPO / entry["prompt_file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append({"id": entry["id"], "path": entry["prompt_file"], "tokens": tokens,
                        "sha256": hashlib.sha256(text.encode()).hexdigest()})
        print(f"{entry['id']}: {tokens} content tokens (+{CHAT_OVERHEAD} chat), "
              f"sha256 {written[-1]['sha256'][:12]}", file=sys.stderr)
    return written


def main() -> int:
    document = json.loads(CASES.read_text(encoding="utf-8"))
    if "--check" in sys.argv[1:]:
        tokenizer = FlintTokenizer(TOKENIZER)
        bad = []
        for entry in document["cases"]:
            if entry["kind"] != "retrieval":
                continue
            text = (REPO / entry["prompt_file"]).read_text(encoding="utf-8")
            if abs(len(tokenizer.encode(text)) - (entry["cell"] - CHAT_OVERHEAD)) > 2:
                bad.append(f"{entry['id']}: token count changed")
            elif text.count(entry["needle"]) != 1 or not text.rstrip().endswith(entry["question"]):
                bad.append(f"{entry['id']}: needle or question missing")
        print("\n".join(bad) if bad else "retrieval prompts verified", file=sys.stderr)
        return 1 if bad else 0
    build(document)
    return 0


if __name__ == "__main__":
    sys.exit(main())
