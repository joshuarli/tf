"""Build the fixed Flint prompt corpus and confirm each rendered chat prompt's length with the served tokenizer."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "campaign" / "bench"))
sys.path.insert(0, str(REPO / "tools"))

from flint_tokenizer import FlintTokenizer  # noqa: E402
import bench_openai  # noqa: E402
import harness  # noqa: E402

LENGTHS = (1024, 4096, 32768)
TOPICS = ("code", "prose", "structured")
SETS = ("dev", "holdout")
QUALIFY = "qualify"
QUALIFY_LENGTHS = (1024, 4096)

INSTRUCTIONS = {
    "code": (
        "You are reviewing a large native inference codebase. Below are consecutive excerpts from several source "
        "files, each introduced by a line of the form ===== path =====. Work through them and produce a technical "
        "review: for each excerpt name the function or type it defines, state one correctness or performance risk "
        "with the identifiers that carry it, and propose one typed refactor. Be specific and cite names from the text."
    ),
    "prose": (
        "The following are consecutive excerpts from engineering notes and design documents. Reason carefully about "
        "the argument they build. Restate the central claim in your own words, set out the evidence offered for it, "
        "identify the weakest inference and why it is weak, and say what measurement or observation would settle it."
    ),
    "structured": (
        "The following are consecutive excerpts from configuration and documentation files. Read them and return "
        "exactly one JSON object, with no prose around it, matching this schema: {\"summary\": string, "
        "\"components\": [{\"name\": string, \"kind\": string}], \"decisions\": [{\"id\": string, \"reason\": string}]}. "
        "Use names that appear in the excerpts and keep every string under forty words."
    ),
}

GLOBS = {
    "code": ["zig/src/**/*.zig", "zig/build/*.zig", "tools/bench_*.py", "build.zig"],
    "prose": ["docs/**/*.md", "README.md", "RUNBOOK.md", "CONTRIBUTING.md", "SECURITY.md", "campaign/*.md"],
    "structured": ["*.json", "campaign/*.json", "campaign/*.toml", "tools/*.json", "zig/**/*.json",
                   "docs/**/*.md", "*.yml", "*.yaml", "*.sb"],
}

SKIP_PARTS = ("node_modules", ".work", "zig-out", ".zig-cache", "fixtures/golden", "/golden/",
              "campaign/prompts", "campaign/runs")
MIN_BYTES, MAX_BYTES = 1024, 512 * 1024


_TRACKED: dict[tuple[str, ...], list[Path]] = {}
_POOLS: dict[tuple[str, tuple[str, ...]], list[Path]] = {}


def tracked(globs: list[str]) -> list[Path]:
    key = tuple(globs)
    if key not in _TRACKED:
        out = subprocess.run(["git", "ls-files", "--", *globs], cwd=REPO, capture_output=True, text=True,
                             check=True).stdout
        _TRACKED[key] = [REPO / line for line in out.splitlines() if line]
    return _TRACKED[key]


def usable(path: Path, forbidden: list[str]) -> bool:
    rel = path.relative_to(REPO).as_posix()
    if any(part in rel for part in SKIP_PARTS):
        return False
    try:
        if not MIN_BYTES <= path.stat().st_size <= MAX_BYTES:
            return False
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False
    return not any(marker in text for marker in forbidden)


def pool(topic: str, forbidden: list[str]) -> list[Path]:
    key = (topic, tuple(forbidden))
    if key not in _POOLS:
        seen, out = set(), []
        for path in tracked(GLOBS[topic]):
            if path.is_file() and path not in seen and usable(path, forbidden):
                seen.add(path)
                out.append(path)
        _POOLS[key] = out
    return _POOLS[key]


def partition(paths: list[Path]) -> dict[str, list[Path]]:
    """Disjoint source halves by a stable path hash, so no held-out prompt reuses a dev prompt's text."""
    halves: dict[str, list[Path]] = {"dev": [], "holdout": []}
    for path in paths:
        rel = path.relative_to(REPO).as_posix()
        halves["dev" if int(hashlib.sha256(rel.encode()).hexdigest(), 16) % 2 == 0 else "holdout"].append(path)
    return halves


def order(paths: list[Path], set_name: str, topic: str) -> list[Path]:
    def key(path: Path) -> tuple[str, str]:
        rel = path.relative_to(REPO).as_posix()
        return hashlib.sha256(f"{set_name}|{topic}|{rel}".encode()).hexdigest(), rel
    return sorted(paths, key=key)


def exact_prefix(tokenizer: FlintTokenizer, text: str, tokens: int) -> tuple[str, int]:
    target = tokens
    prefix = ""
    for _ in range(8):
        prefix, got = tokenizer.prefix_text(text, target)
        count = len(tokenizer.encode(prefix))
        if count == target:
            return prefix, count
        target += target - count
        if target <= 0:
            break
    raise ValueError(f"could not land exactly {tokens} content tokens (reached {count})")


def assemble(topic: str, length: int, set_name: str, tokenizer: FlintTokenizer,
             forbidden: list[str], cited: frozenset[str] = frozenset()) -> tuple[str, list[dict]]:
    if set_name == QUALIFY:
        sources = [p for p in pool(topic, forbidden) if p.relative_to(REPO).as_posix() not in cited]
    else:
        sources = partition(pool(topic, forbidden))[set_name]
    candidates = order(sources, set_name, topic)
    if not candidates:
        raise ValueError(f"no usable {set_name} sources for {topic}")
    instruction = INSTRUCTIONS[topic]
    pieces: list[tuple[Path, str]] = []
    assembled = instruction + "\n\n"
    for path in candidates:
        text = path.read_text(encoding="utf-8")
        piece = f"===== {path.relative_to(REPO).as_posix()} =====\n{text}\n\n"
        pieces.append((path, piece))
        assembled += piece
        if len(tokenizer.encode(assembled)) >= length:
            break
    content, count = exact_prefix(tokenizer, assembled, length - 2)
    sources, offset = [], 0
    for path, piece in pieces:
        start = assembled.find(piece, offset)
        if start < 0:
            continue
        offset = start + len(piece)
        end_in_prefix = start + len(piece) <= len(content)
        sources.append({"path": path.relative_to(REPO).as_posix(), "sha256": harness.sha256_file(path),
                        "bytes": path.stat().st_size, "complete": end_in_prefix})
        if not end_in_prefix:
            break
    return content, sources


def source_hash(entries: list[dict]) -> str:
    return hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_corpus(tokenizer: FlintTokenizer, lengths: list[int]) -> dict:
    forbidden = ["\x00", "<<<", ">>>", "<|", "|>", *tokenizer.added]
    manifest = {"schema": 1,
                "model": "flint-int4@3b93f9053b53",
                "tokenizer": "campaign/prompts/flint_tokenizer.py: byte-level BPE read from the checkpoint's tokenizer.json",
                "tokenizer_sha256": harness.sha256_file(Path.home() / "models" / "flint-int4@3b93f9053b53" / "tokenizer.json"),
                "rendered_chat_overhead_tokens": 2,
                "length_note": ("content is cut to length - 2 local tokens; the served chat template adds "
                                "<|user|> and <|assistant|>, so usage.prompt_tokens is the cell"),
                "prompts": []}
    for set_name in SETS:
        for topic in TOPICS:
            for length in lengths:
                name = f"{topic}-{length}" if set_name == "dev" else f"heldout-{topic}-{length}"
                content, sources = assemble(topic, length, set_name, tokenizer, forbidden)
                relative = f"campaign/prompts/{set_name}/{name}.txt"
                target = REPO / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                manifest["prompts"].append({
                    "name": name, "set": set_name, "topic": topic, "cell": length, "path": relative,
                    "sha256": hashlib.sha256(content.encode()).hexdigest(), "bytes": len(content.encode()),
                    "local_content_tokens": len(tokenizer.encode(content)),
                    "measured_prompt_tokens": None, "sources_sha256": source_hash(sources), "sources": sources,
                    "instruction": INSTRUCTIONS[topic],
                })
                print(f"built {relative} ({len(content.encode())} bytes, "
                      f"{len(tokenizer.encode(content))} local content tokens)", file=sys.stderr)
    return manifest


def append_qualify(tokenizer: FlintTokenizer, manifest: dict) -> None:
    """Append the qualify set: prompts whose sources no dev or holdout prompt cites, so no fitted prior has seen them."""
    if any(prompt["set"] == QUALIFY for prompt in manifest["prompts"]):
        raise SystemExit("the manifest already has the qualify set")
    forbidden = ["\x00", "<<<", ">>>", "<|", "|>", *tokenizer.added]
    cited = frozenset(source["path"] for prompt in manifest["prompts"] for source in prompt["sources"])
    for topic in TOPICS:
        for length in QUALIFY_LENGTHS:
            name = f"{QUALIFY}-{topic}-{length}"
            content, sources = assemble(topic, length, QUALIFY, tokenizer, forbidden, cited)
            cited |= {source["path"] for source in sources}
            relative = f"campaign/prompts/{QUALIFY}/{name}.txt"
            target = REPO / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            manifest["prompts"].append({
                "name": name, "set": QUALIFY, "topic": topic, "cell": length, "path": relative,
                "sha256": hashlib.sha256(content.encode()).hexdigest(), "bytes": len(content.encode()),
                "local_content_tokens": len(tokenizer.encode(content)),
                "measured_prompt_tokens": None, "sources_sha256": source_hash(sources), "sources": sources,
                "instruction": INSTRUCTIONS[topic],
            })
            print(f"built {relative} ({len(content.encode())} bytes)", file=sys.stderr)


def measure(manifest: dict, binary: Path, model: Path, log_path: Path, set_name: str | None = None) -> None:
    environment = harness.conditions()
    if environment["pressure_reasons"]:
        raise SystemExit(f"host is not in a measurable state: {environment['pressure_reasons']}")
    with harness.Server(binary, model, ["--no-drafts"], 0, log_path) as server:
        served = server.models()
        if not served:
            raise SystemExit("server listed no model id")
        model_id = served[0]
        print(f"serving {model_id}; startup {server.timing()}", file=sys.stderr)
        for prompt in manifest["prompts"]:
            if set_name is not None and prompt["set"] != set_name:
                continue
            text = (REPO / prompt["path"]).read_text(encoding="utf-8")
            item = {"name": prompt["name"], "kind": "chat", "prompt": text,
                    "thinking_switch": False, "thinking_applicable": None}
            run = bench_openai.stream(server.base, model_id, item, 1, 0.0, None)
            measured = run.get("prompt_tokens")
            prompt["measured_prompt_tokens"] = measured
            prompt["measured_cache_state"] = run.get("cache_state")
            if measured != prompt["cell"]:
                raise SystemExit(f"{prompt['name']}: served prompt_tokens {measured} != cell {prompt['cell']}")
            ttft = run.get("ttft_s")
            print(f"measured {prompt['name']}: {measured} prompt tokens "
                  f"(cold {'{:.2f}s'.format(ttft) if ttft is not None else 'unmeasured'})", file=sys.stderr)
        manifest["measured_by" if set_name is None else f"{set_name}_measured_by"] = {"binary_sha256": harness.sha256_file(binary), "host": harness.host_summary()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", default=",".join(str(n) for n in LENGTHS))
    parser.add_argument("--measure", action="store_true", help="confirm every prompt with the served tokenizer")
    parser.add_argument("--check", action="store_true", help="verify the committed manifest against local counts")
    parser.add_argument("--qualify", action="store_true",
                        help="append the qualify set to the committed manifest, leaving dev and holdout unchanged")
    parser.add_argument("--binary", type=Path, default=harness.BASELINE_BIN)
    parser.add_argument("--model-dir", type=Path, default=None)
    args = parser.parse_args()
    lengths = [int(value) for value in args.lengths.split(",")]
    tokenizer_path = Path.home() / "models" / "flint-int4@3b93f9053b53" / "tokenizer.json"
    tokenizer = FlintTokenizer(tokenizer_path)
    if args.check:
        manifest = json.loads(harness.PROMPT_MANIFEST.read_bytes())
        bad = []
        for prompt in manifest["prompts"]:
            text = (REPO / prompt["path"]).read_text(encoding="utf-8")
            if hashlib.sha256(text.encode()).hexdigest() != prompt["sha256"]:
                bad.append(f"{prompt['name']}: sha256 changed")
            elif len(tokenizer.encode(text)) != prompt["local_content_tokens"]:
                bad.append(f"{prompt['name']}: local token count changed")
            elif prompt.get("measured_prompt_tokens") != prompt["cell"]:
                bad.append(f"{prompt['name']}: no confirmed served length")
        print("\n".join(bad) if bad else f"{len(manifest['prompts'])} prompts verified", file=sys.stderr)
        return 1 if bad else 0
    if args.qualify:
        manifest = json.loads(harness.PROMPT_MANIFEST.read_bytes())
        append_qualify(tokenizer, manifest)
        if args.measure:
            harness.TMP.mkdir(parents=True, exist_ok=True)
            measure(manifest, args.binary, args.model_dir or harness.model_dir("flint-int4"),
                    harness.TMP / "prompts-measure-qualify.log", QUALIFY)
        harness.PROMPT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {harness.PROMPT_MANIFEST} with {len(manifest['prompts'])} prompts", file=sys.stderr)
        return 0
    manifest = write_corpus(tokenizer, lengths)
    if args.measure:
        model = args.model_dir or harness.model_dir("flint-int4")
        harness.TMP.mkdir(parents=True, exist_ok=True)
        measure(manifest, args.binary, model, harness.TMP / "prompts-measure.log")
    harness.PROMPT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {harness.PROMPT_MANIFEST} with {len(manifest['prompts'])} prompts", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
