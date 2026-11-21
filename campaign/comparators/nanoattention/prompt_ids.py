"""The checkpoint's chat-templated prompt ids, exactly as the served model counts them.

nanoattention takes token ids, not text. Rendering the checkpoint's chat_template.jinja
with the plain tokenizers library reproduces the 1024/4096/32768 prompt counts the
tensorfold server reports, so every engine receives the same ids.
"""

import json
import sys
from pathlib import Path

from jinja2 import Environment
from tokenizers import Tokenizer


def prompt_ids(model_dir: Path, text: str) -> list[int]:
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    template = Environment().from_string((model_dir / "chat_template.jinja").read_text())
    rendered = template.render(messages=[{"role": "user", "content": text}], add_generation_prompt=True)
    return tokenizer.encode(rendered).ids


if __name__ == "__main__":
    model_dir, path = Path(sys.argv[1]), Path(sys.argv[2])
    print(json.dumps(prompt_ids(model_dir, path.read_text(encoding="utf-8"))))
