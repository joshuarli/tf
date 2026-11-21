"""Render the four teacher-forced probe histories and pin their token ids.

The rendering follows the pinned ``chat_template.jinja`` for user-only
histories: the template emits the default thinking-effort system message before
the first non-system message, then ``<|message_user|><|content_text|>...<|end_message|>``
for each user turn. Tokenisation goes through the pinned llama.cpp tokenizer
(``llama_logits --tokenize-text``), so the ids match the GGUF vocabulary.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HISTORIES = [
    ("greet", "Hi"),
    ("code", "Write a Python function that returns the sum of two integers."),
    ("prose", "The quick brown fox jumps over the lazy dog, and then it keeps running down the lane."),
    ("math", "What is 17 multiplied by 23? Reply with the number only."),
]


def render_user(content: str) -> str:
    effort = "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
    return effort + f"<|message_user|><|content_text|>{content}<|end_message|>"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-logits", required=True)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    entries = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, content in HISTORIES:
            text = render_user(content)
            text_path = Path(tmp) / f"{name}.txt"
            ids_path = Path(tmp) / f"{name}.json"
            text_path.write_text(text, encoding="utf-8")
            subprocess.run([args.llama_logits, "--model", args.gguf,
                            "--tokenize-text", str(text_path), "--out", str(ids_path)],
                           check=True, capture_output=True)
            ids = json.loads(ids_path.read_text())
            entries.append({"name": name, "text": content, "token_ids": ids})

    Path(args.out).write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    for entry in entries:
        print(f"{entry['name']}: {len(entry['token_ids'])} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
