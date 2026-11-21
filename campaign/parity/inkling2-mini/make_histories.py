"""Render and tokenise the 16 held-out teacher-forced histories for the Inkling 2 Mini parity check.

None of these texts is one of the four I0 used. Each is rendered the way a served request is: the pinned
``chat_template.jinja``'s default thinking-effort system turn, then one user turn,
``<|message_user|><|content_text|>...<|end_message|>``. Tokenisation goes through the pinned llama.cpp tokenizer
(``llama_logits --tokenize-json``), the vocabulary I0 checked against the checkpoint. ``--verify i0`` re-renders
I0's own four histories and requires their leading ids to match, which is the check that this rendering path is the
template's.

Run from the repository root after ``make -C campaign ref-gguf`` (the GGUF) and the ``llama_logits`` binary:

    python3 campaign/parity/inkling2-mini/make_histories.py \
        --llama-logits .work/build/llama_logits --gguf .work/build/inkling2-mini-f32.gguf --verify i0

It writes ``histories.json``: per history, its name, its text, its token ids and the sha256 of those ids' bytes.
``harbour`` and ``cells_code`` run past the 1,024-token sliding window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

EFFORT = "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"

LONG_PROSE = """The harbour at first light is a study in patience. Gulls wheel above the breakwater while the fishing boats nose
out of the channel one after another, their diesels settling into a slow uneven rhythm that carries across the water.
A man on the quay coils a rope he has coiled ten thousand times, and the motion has worn a smooth pale patch into the
tarred planks beneath his boots. Further along, the ice house breathes its cold breath into the warm air, and the smell
of salt and diesel and crushed shellfish settles into everything.

By seven the market is loud. The stalls arrive in a rough order that looks accidental and is not: the vegetable
growers take the north row because the sun reaches it first, the fishmongers take the shaded middle so their ice lasts
past noon, and the flower sellers hold the corner where the bus stops because a waiting crowd is a buying crowd. The
haggling is a formal thing, almost a dance. A price is named, refused, named again with a small theatrical injury, and
then accepted, and both parties look faintly disappointed and entirely satisfied.

The town itself climbs the hill in terraces. The oldest houses sit lowest, closest to the water, built when the only
transport was a boat and a strong back. Their walls are thick and their windows small and their doorways low enough to
make a tall visitor duck. Higher up the houses thin out into gardens, and the gardens thin out into gorse and rock, and
above that the hill simply continues, indifferent, until it becomes the moor.

There is a school halfway up with a bell that has been rung at the wrong time for a hundred and forty years. The story
goes that the bell was cast with a flaw, and that the flaw makes it sound a full second after it is struck, so that
generations of children have learned to flinch at a sound that has already happened. Nobody has ever replaced the bell.
It is considered part of the town, like the tide or the mayor.

In the afternoon the wind changes and the harbour turns choppy and the boats come back in early. The market folds
itself away with the same unhurried ritual as it arrived. The gulls follow the gutting tables and then the gutting
tables are hosed down and the gulls give up and go to sit on the roof of the ice house in a long grey line, all facing
the same way, like a committee.

Evening comes slowly in summer. The light lingers on the water and the hill stays warm long after the valley has gone
blue. People walk out along the breakwater for no reason at all, which is the best reason. Someone is always playing a
guitar badly near the lifeboat station. Someone is always mending a net that does not need mending, because mending a
net is a way of thinking.

The town has a newspaper that prints eight pages and a library that opens four days a week and a museum of the
lifeboat, which is small and free and staffed by a retired coxswain who will tell you, if you let him, about the night
in nineteen sixty-one when the wind reached force eleven and the crew went out anyway and came back. He tells it the
same way every time, in the same order, with the same pauses, and the pauses are the point.

Winter is different. The visitors go and the town contracts to its bones. The harbour empties except for two boats
that work through, and the market moves indoors to the hall with the leaking roof, and the wind comes off the moor
with nothing to stop it for thirty miles. But the bell still rings a second late, and the gulls still line up on the
ice house, and the coxswain still tells his story to whoever comes in out of the cold.

Nothing here is remarkable and that is exactly the point of it. A place that has decided what it is does not need to
keep deciding. The ferry runs twice a day whether or not anyone is waiting, the school teaches the same arithmetic it
taught to the parents of its pupils, and the ice house keeps its long slow argument with the summer. When the wind
finally drops in the evening the whole town seems to exhale at once, and the last light goes off the hill, and the
bell, a second late, says that the day is over.

The coxswain keeps a ledger of every launch in his own hand, going back to the year his father took over the station.
Most of the entries are ordinary: a yacht with a fouled propeller, a dinghy adrift, a dog that swam too far and had to
be lifted out of the water by the scruff of its neck and was not grateful about it at all. A few are not ordinary.
Those are written in the same steady hand as the others, which is the only remarkable thing about them, and when he
reads them aloud in the museum on a wet Tuesday afternoon the visitors go very quiet and the rain goes on against the
windows and nobody looks at the clock.

There is a phrase in the town for the hour before the light fails, when the work is done and the boats are tied and
the gulls have settled and the hill has begun to hold the last of the warmth. They call it the long minute. It is not
a minute and nobody can say exactly when it starts, and if you ask three people you will get three answers, but
everyone agrees when it has arrived, and everyone stops what they are doing for it, and then it is over and the lamps
come on along the quay and the evening begins properly.

That is the whole of the place, near enough. A harbour, a hill, a bell with a flaw in it, and a habit of stopping for
a minute that is not a minute. It will not make anyone rich and it does not want to. It only wants to keep going,
which it does, one tide at a time.
"""

LONG_CODE = """def solve_part_one(lines):
    total = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        left = int(parts[0])
        right = int(parts[1])
        if left < right:
            total += right - left
        else:
            total += left - right
    return total


def solve_part_two(lines):
    counts = {}
    for line in lines:
        for word in line.split():
            word = word.strip(".,;:!?").lower()
            if not word:
                continue
            counts[word] = counts.get(word, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ordered[:10]


class Grid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.cells = [[0 for _ in range(cols)] for _ in range(rows)]

    def get(self, r, c):
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            return 0
        return self.cells[r][c]

    def set(self, r, c, value):
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.cells[r][c] = value

    def neighbours(self, r, c):
        out = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                out.append(self.get(r + dr, c + dc))
        return out

    def step(self, rule):
        nxt = Grid(self.rows, self.cols)
        for r in range(self.rows):
            for c in range(self.cols):
                live = sum(1 for n in self.neighbours(r, c) if n)
                nxt.set(r, c, rule(self.get(r, c), live))
        return nxt

    def render(self, on="#", off="."):
        lines = []
        for row in self.cells:
            lines.append("".join(on if cell else off for cell in row))
        return "\\n".join(lines)


def parse_blocks(text):
    blocks = []
    current = []
    for line in text.splitlines():
        if line.strip() == "":
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line)
    if current:
        blocks.append(current)
    return blocks


def checksum(data, modulus=1000003):
    h = 0
    for byte in data:
        h = (h * 131 + byte) % modulus
    return h


def chunk(seq, size):
    for start in range(0, len(seq), size):
        yield seq[start:start + size]


def merge_sorted(a, b):
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def flatten(tree):
    if isinstance(tree, dict):
        out = []
        for key in sorted(tree):
            out.append(key)
            out.extend(flatten(tree[key]))
        return out
    if isinstance(tree, (list, tuple)):
        out = []
        for item in tree:
            out.extend(flatten(item))
        return out
    return [tree]


def sliding_windows(values, width):
    out = []
    for i in range(len(values)):
        lo = max(0, i + 1 - width)
        out.append(sum(values[lo:i + 1]))
    return out
"""

HISTORIES = [
    ("harbour", LONG_PROSE),
    ("cells_code", LONG_CODE),
    ("greeting", "Hello. What can you help me with today?"),
    ("explain_sliding", "Explain in two sentences why a sliding-window attention layer keeps less state than a full one."),
    ("unit_test", "Write a pytest function that checks a function named add returns the sum of its arguments."),
    ("refactor", "Rename the variable count to total in this snippet and say what else changes: count = 0."),
    ("math_mod", "What is 2 to the power of 20, and what is that number modulo 1000?"),
    ("math_frac", "A tank fills at 3 litres a minute and drains at 2.5. How full is it after 40 minutes?"),
    ("dialogue", "Passage: 'We should leave now,' said Mira. 'The tide turns at six.' Question: when does the tide turn?"),
    ("list_steps", "List four steps to reproduce a failing build, one per line, numbered."),
    ("structured", 'Return only JSON with keys "name" and "count" for a list of three apples.'),
    ("summary", "Summarise in one sentence: the city council approved the budget after a long debate about transit funding."),
    ("sql", "Write a SQL query counting orders per customer, largest first, for tables customers and orders."),
    ("prose_review", "Rewrite this sentence without the passive voice: The report was written by the intern in two days."),
    ("code_trace", "Trace this loop and give the final value: x = 1; for i in range(4): x = x * 2 + i"),
    ("token_budget", "Which is cheaper at 40 tokens a second: 90 seconds of work or 3,600 tokens?"),
]


def render(content: str) -> str:
    return EFFORT + f"<|message_user|><|content_text|>{content}<|end_message|>"


def tokenize(llama_logits: str, gguf: str, texts: list[tuple[str, str]]) -> dict[str, list[int]]:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "texts.json"
        src.write_text(json.dumps([{"name": n, "text": render(t)} for n, t in texts]), encoding="utf-8")
        dst = Path(tmp) / "ids.json"
        subprocess.run([llama_logits, "--model", gguf, "--tokenize-json", str(src), "--out", str(dst)],
                       check=True, stderr=subprocess.DEVNULL)
        out = json.loads(dst.read_text(encoding="utf-8"))
    return {entry["name"]: entry["token_ids"] for entry in out}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-logits", required=True)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--out", default=str(HERE / "histories.json"))
    parser.add_argument("--verify", default=None, help="a reference histories.json to check the rendering against")
    args = parser.parse_args(argv)

    ids = tokenize(args.llama_logits, args.gguf, HISTORIES)
    if args.verify:
        reference = json.loads(Path(args.verify).read_text(encoding="utf-8"))
        wanted = {entry["name"]: entry["token_ids"] for entry in reference}
        checked = [(entry["name"], entry["text"]) for entry in reference if "text" in entry][:4]
        got = tokenize(args.llama_logits, args.gguf, checked)
        for name, _ in checked:
            if name in wanted and got.get(name) != wanted[name]:
                print(f"render mismatch for {name}: {got.get(name)} != {wanted[name]}", file=sys.stderr)
                return 1
        print(f"render check: {len(checked)} histories tokenize to the reference's ids exactly")

    histories = []
    for name, text in HISTORIES:
        tokens = ids[name]
        digest = hashlib.sha256(json.dumps(tokens).encode("utf-8")).hexdigest()
        histories.append({"name": name, "text": text, "tokens": len(tokens), "sha256": digest, "token_ids": tokens})
    out = Path(args.out)
    out.write_text(json.dumps(histories, indent=2) + "\n", encoding="utf-8")
    for entry in histories:
        print(f"{entry['name']}: {entry['tokens']} tokens, sha256 {entry['sha256'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
