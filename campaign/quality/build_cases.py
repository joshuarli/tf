"""The fixed 48-case quality suite: 16 code, 12 structured, 12 reasoning and 8 retrieval cases."""

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "cases.json"
QUALITY_JSON = REPO / "web" / "tensorfold_web" / "data" / "quality.json"

CODE_PROMPT = (
    "You are given one Python 3 programming task. Reply with exactly one fenced code block labelled python and "
    "nothing outside it. The grader imports your module and appends assertions; every required name must be defined "
    "at module level. Do not read stdin, print, or guard the task behind __main__.\n\n{spec}"
)

STRUCTURED_JSON_PROMPT = (
    "Respond with exactly one JSON value and no other text. It must match the JSON Schema given after the task.\n\n"
    "Task: {task}\n\nJSON Schema: {schema}"
)

TOOL_PROMPT = "Call the required tool. Task: {task}"

REASONING_SOURCE = "web/tensorfold_web/data/quality.json"
REASONING_LICENCE = {"gsm8k": "MIT", "mmlu": "MIT"}


def case(cid, kind, sets, source, licence, **extra):
    entry = {"id": cid, "kind": kind, "sets": sets, "source": source, "licence": licence, **extra}
    return entry


CODE_DEV = {"c01", "c02", "c03"}

CODE = [
    ("c01", "chunk_sums", "Write `chunk_sums(xs, size)`: split the list of integers `xs` into consecutive chunks of "
     "length `size` and return the list of each chunk's sum. The final chunk may be shorter. An empty list returns "
     "`[]`; a `size` below 1 raises `ValueError`.",
     "assert chunk_sums([], 3) == []\n"
     "assert chunk_sums([1, 2, 3, 4, 5], 2) == [3, 7, 5]\n"
     "assert chunk_sums([1], 1) == [1]\n"
     "assert chunk_sums([0, 0, 0], 5) == [0]\n"
     "assert chunk_sums([-1, 1, -1, 1], 3) == [-1, 1]\n"
     "try:\n"
     "    chunk_sums([1], 0)\n"
     "except ValueError:\n"
     "    pass\n"
     "else:\n"
     "    raise AssertionError('size 0 must raise ValueError')\n"),
    ("c02", "rle", "Write `rle_encode(s)` and `rle_decode(s)` for strings of ASCII letters. Encoding replaces each run "
     "with its decimal length followed by the character, so `aaabb` becomes `3a2b`; the empty string encodes to the "
     "empty string. Decoding reverses it and raises `ValueError` on a token with no positive run length, a dangling "
     "length, or any character that is not a letter.",
     "assert rle_encode('') == ''\n"
     "assert rle_encode('aaabb') == '3a2b'\n"
     "assert rle_encode('abc') == '1a1b1c'\n"
     "assert rle_decode('') == ''\n"
     "assert rle_decode('3a2b') == 'aaabb'\n"
     "assert rle_decode(rle_encode('mississippi')) == 'mississippi'\n"
     "for bad in ('3', 'a2', '2ab', '0a'):\n"
     "    try:\n"
     "        rle_decode(bad)\n"
     "    except ValueError:\n"
     "        pass\n"
     "    else:\n"
     "        raise AssertionError(bad + ' must raise ValueError')\n"),
    ("c03", "base36", "Write `base36(n)`: render the non-negative integer `n` in lowercase base 36, using digits "
     "0-9 then letters a-z. Zero is `\"0\"`. A negative `n` raises `ValueError`.",
     "assert base36(0) == '0'\n"
     "assert base36(35) == 'z'\n"
     "assert base36(36) == '10'\n"
     "assert base36(71) == '1z'\n"
     "assert base36(1295) == 'zz'\n"
     "assert base36(1296) == '100'\n"
     "try:\n"
     "    base36(-1)\n"
     "except ValueError:\n"
     "    pass\n"
     "else:\n"
     "    raise AssertionError('negative must raise ValueError')\n"),
    ("c04", "merge_intervals", "Write `merge_intervals(intervals)`: given inclusive `[start, end]` integer intervals, "
     "return a new sorted list in which overlapping or touching intervals are merged. The input must not be mutated.",
     "src = [[1, 3], [2, 6], [8, 10], [15, 18]]\n"
     "assert merge_intervals(src) == [[1, 6], [8, 10], [15, 18]]\n"
     "assert src == [[1, 3], [2, 6], [8, 10], [15, 18]]\n"
     "assert merge_intervals([]) == []\n"
     "assert merge_intervals([[5, 6], [1, 2]]) == [[1, 2], [5, 6]]\n"
     "assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]\n"
     "assert merge_intervals([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]\n"),
    ("c05", "LRU", "Write a class `LRU(capacity)` with `get(key)` returning the stored value or -1, and `put(key, "
     "value)`. It holds at most `capacity` entries (capacity at least 1) and evicts the least recently used entry; "
     "both `get` and `put` count as use.",
     "c = LRU(2)\n"
     "c.put(1, 1)\n"
     "c.put(2, 2)\n"
     "assert c.get(1) == 1\n"
     "c.put(3, 3)\n"
     "assert c.get(2) == -1\n"
     "assert c.get(3) == 3\n"
     "c.put(4, 4)\n"
     "assert c.get(1) == -1\n"
     "assert c.get(3) == 3\n"
     "assert c.get(4) == 4\n"
     "one = LRU(1)\n"
     "one.put(5, 50)\n"
     "assert one.get(5) == 50\n"
     "one.put(6, 60)\n"
     "assert one.get(5) == -1\n"),
    ("c06", "is_valid_parens", "Write `is_valid_parens(s)`: `s` contains only the six bracket characters "
     "`()[]{}`; return True when they are correctly nested and balanced, and False otherwise. The empty string is "
     "valid.",
     "assert is_valid_parens('') is True\n"
     "assert is_valid_parens('()[]{}') is True\n"
     "assert is_valid_parens('([{}])') is True\n"
     "assert is_valid_parens('([)]') is False\n"
     "assert is_valid_parens('(') is False\n"
     "assert is_valid_parens(')(') is False\n"
     "assert is_valid_parens('{[()()]}') is True\n"),
    ("c07", "top_k_words", "Write `top_k_words(text, k)`: lowercase the text, take words of ASCII letters and "
     "digits plus apostrophes, count them, and return the `k` most frequent as `(word, count)` tuples ordered by "
     "count descending then word ascending. Fewer than `k` distinct words returns all of them; `k` below 1 returns "
     "`[]`.",
     "text = 'The cat and the dog and the bird. A CAT, a dog!'\n"
     "assert top_k_words(text, 2) == [('the', 3), ('a', 2)]\n"
     "assert top_k_words(text, 10) == [('the', 3), ('a', 2), ('and', 2), ('cat', 2), ('dog', 2), ('bird', 1)]\n"
     "assert top_k_words('', 3) == []\n"
     "assert top_k_words('x y', 0) == []\n"),
    ("c08", "rotate90", "Write `rotate90(matrix)`: given a square list-of-lists, return a new matrix rotated 90 "
     "degrees clockwise. The input must not be mutated.",
     "m = [[1, 2], [3, 4]]\n"
     "assert rotate90(m) == [[3, 1], [4, 2]]\n"
     "assert m == [[1, 2], [3, 4]]\n"
     "assert rotate90([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [[7, 4, 1], [8, 5, 2], [9, 6, 3]]\n"
     "assert rotate90([]) == []\n"),
    ("c09", "isbn13_valid", "Write `isbn13_valid(s)`: strip spaces and hyphens, require exactly 13 ASCII digits, and "
     "return True when the alternating 1,3 weighted sum of the digits is a multiple of 10. Anything else returns "
     "False.",
     "assert isbn13_valid('9780306406157') is True\n"
     "assert isbn13_valid('978-0-306-40615-7') is True\n"
     "assert isbn13_valid(' 9780306406157 ') is True\n"
     "assert isbn13_valid('9780306406158') is False\n"
     "assert isbn13_valid('978030640615') is False\n"
     "assert isbn13_valid('978030640615X') is False\n"),
    ("c10", "parse_ini", "Write `parse_ini(text)`: parse INI text into a dict mapping section name to a dict of "
     "string keys and values. Lines starting with `;` or `#` are comments, blank lines are ignored, a line "
     "`[name]` opens a section, and `key = value` sets a stripped key and value with the last duplicate winning. "
     "Keys before any section belong to the section named by the empty string. Any other non-blank line raises "
     "`ValueError`.",
     "text = '\\n'.join([\n"
     "    '; comment',\n"
     "    'name = root',\n"
     "    '[db]',\n"
     "    'host=localhost',\n"
     "    'port = 5432',\n"
     "    '# note',\n"
     "    '[app]',\n"
     "    'debug = false',\n"
     "    'name = app',\n"
     "])\n"
     "got = parse_ini(text)\n"
     "assert got[''] == {'name': 'root'}\n"
     "assert got['db'] == {'host': 'localhost', 'port': '5432'}\n"
     "assert got['app'] == {'debug': 'false', 'name': 'app'}\n"
     "try:\n"
     "    parse_ini('[x]\\nbroken line\\n')\n"
     "except ValueError:\n"
     "    pass\n"
     "else:\n"
     "    raise AssertionError('malformed line must raise ValueError')\n"),
    ("c11", "first_at_least", "Write `first_at_least(xs, target)`: `xs` is sorted ascending; return the index of "
     "the first element greater than or equal to `target`, or `len(xs)` when there is none. Use binary search.",
     "assert first_at_least([], 5) == 0\n"
     "assert first_at_least([1, 2, 2, 3, 5], 2) == 1\n"
     "assert first_at_least([1, 2, 2, 3, 5], 4) == 4\n"
     "assert first_at_least([1, 2, 2, 3, 5], 9) == 5\n"
     "assert first_at_least([-3, -1, 0], -2) == 1\n"),
    ("c12", "sliding_max", "Write `sliding_max(xs, k)`: return the maximum of every window of `k` consecutive "
     "elements. `k` at least 1; a list shorter than `k` returns `[]`.",
     "assert sliding_max([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]\n"
     "assert sliding_max([1], 1) == [1]\n"
     "assert sliding_max([1, 2], 3) == []\n"
     "assert sliding_max([9, 8, 7, 6], 2) == [9, 8, 7]\n"),
    ("c13", "to_fraction", "Write `to_fraction(s)`: parse an integer, a decimal, or a `p/q` fraction into a reduced "
     "`(numerator, denominator)` tuple with a positive denominator. Raise `ValueError` for anything unparsable and "
     "for a zero denominator.",
     "assert to_fraction('1/2') == (1, 2)\n"
     "assert to_fraction('6/4') == (3, 2)\n"
     "assert to_fraction('0.75') == (3, 4)\n"
     "assert to_fraction('-2.5') == (-5, 2)\n"
     "assert to_fraction('4') == (4, 1)\n"
     "assert to_fraction('-3/9') == (-1, 3)\n"
     "try:\n"
     "    to_fraction('1/0')\n"
     "except ValueError:\n"
     "    pass\n"
     "else:\n"
     "    raise AssertionError('zero denominator must raise ValueError')\n"),
    ("c14", "flatten", "Write `flatten(xs)`: return a new list holding the non-list, non-tuple values of the nested "
     "structure `xs` in order. Lists and tuples are traversed at any depth; strings and other values are leaves.",
     "assert flatten([]) == []\n"
     "assert flatten([1, [2, [3, [4]]], 5]) == [1, 2, 3, 4, 5]\n"
     "assert flatten([[], [[], [1]], ()]) == [1]\n"
     "assert flatten(((1, (2,)), 3)) == [1, 2, 3]\n"
     "assert flatten(['ab', ['cd']]) == ['ab', 'cd']\n"),
    ("c15", "split_csv_line", "Write `split_csv_line(line)`: split one CSV record on commas that are outside "
     "double-quoted fields. Quotes are removed, `\"\"` inside a quoted field is one quote, and an unbalanced quote "
     "raises `ValueError`. The empty line yields `[\"\"]`.",
     "assert split_csv_line('a,b,c') == ['a', 'b', 'c']\n"
     "assert split_csv_line('a,\"b,c\",d') == ['a', 'b,c', 'd']\n"
     "assert split_csv_line('\"he said \"\"hi\"\"\",x') == ['he said \"hi\"', 'x']\n"
     "assert split_csv_line('') == ['']\n"
     "try:\n"
     "    split_csv_line('\"unterminated,x')\n"
     "except ValueError:\n"
     "    pass\n"
     "else:\n"
     "    raise AssertionError('unbalanced quote must raise ValueError')\n"),
    ("c16", "canonical_json", "Write `canonical_json(obj)`: return compact JSON text with object keys sorted and "
     "non-ASCII characters kept as UTF-8, using the separators `,` and `:` and no spaces.",
     "assert canonical_json({'b': 1, 'a': [True, None]}) == '{\"a\":[true,null],\"b\":1}'\n"
     "assert canonical_json({'x': 'h\\u00e9llo'}) == '{\"x\":\"h\\u00e9llo\"}'\n"
     "assert canonical_json([]) == '[]'\n"
     "assert canonical_json({'k': {'z': 2, 'y': 1}}) == '{\"k\":{\"y\":1,\"z\":2}}'\n"),
]


def code_cases():
    out = []
    for cid, name, spec, tests in CODE:
        sets = ["dev", "full"] if cid in CODE_DEV else ["full"]
        out.append(case(f"code-{cid}", "code", sets, "campaign/quality/build_cases.py#CODE", "Apache-2.0",
                        name=name, prompt=CODE_PROMPT.format(spec=spec), tests=tests,
                        prompt_sha256=hashlib.sha256(CODE_PROMPT.format(spec=spec).encode()).hexdigest()))
    return out


JSON_DEV = {"s01", "s02"}

JSON_CASES = [
    ("s01", "weather", "Report tomorrow's weather for Reykjavik.",
     {"type": "object", "properties": {"city": {"type": "string"}, "temp_c": {"type": "number"},
                                       "conditions": {"type": "string", "enum": ["clear", "cloudy", "rain", "snow"]}},
      "required": ["city", "temp_c", "conditions"], "additionalProperties": False}, None),
    ("s02", "inventory", "Return exactly three inventory records, each with an integer id and a tag from alpha, beta, gamma.",
     {"type": "object", "properties": {"items": {"type": "array", "minItems": 3, "maxItems": 3, "items": {
         "type": "object", "properties": {"id": {"type": "integer"},
                                          "tag": {"type": "string", "enum": ["alpha", "beta", "gamma"]}},
         "required": ["id", "tag"], "additionalProperties": False}}}, "required": ["items"],
      "additionalProperties": False}, None),
    ("s03", "service", "Describe one service with its name and integer replica count between 1 and 10, plus a list of "
     "enabled features from cache, metrics and tracing.",
     {"type": "object", "properties": {"service": {"type": "object", "properties": {
         "name": {"type": "string"}, "replicas": {"type": "integer", "minimum": 1, "maximum": 10}},
         "required": ["name", "replicas"], "additionalProperties": False},
         "features": {"type": "array", "items": {"type": "string", "enum": ["cache", "metrics", "tracing"]}}},
      "required": ["service", "features"], "additionalProperties": False}, None),
    ("s04", "contact", "Return one contact with a name of 1 to 40 characters, an email, and a country from IE, PT or JP.",
     {"type": "object", "properties": {"name": {"type": "string", "minLength": 1, "maxLength": 40},
                                       "email": {"type": "string"},
                                       "country": {"type": "string", "enum": ["IE", "PT", "JP"]}},
      "required": ["name", "email", "country"], "additionalProperties": False}, None),
    ("s05", "stats", "Report a result with a boolean ok flag, a score between 0 and 1, and a non-negative integer count.",
     {"type": "object", "properties": {"ok": {"type": "boolean"}, "score": {"type": "number", "minimum": 0,
                                                                            "maximum": 1},
                                       "count": {"type": "integer", "minimum": 0}},
      "required": ["ok", "score", "count"], "additionalProperties": False}, None),
    ("s06", "severity", "Classify an incident with a severity and a scope.",
     {"type": "object", "properties": {"severity": {"type": "string", "enum": ["low", "medium", "high"]},
                                       "scope": {"type": "string", "enum": ["local", "regional", "global"]}},
      "required": ["severity", "scope"], "additionalProperties": False}, None),
    ("s07", "directions", "Return the four cardinal directions north, south, east and west, each exactly once, as a "
     "JSON array of strings.",
     {"type": "array", "minItems": 4, "maxItems": 4,
      "items": {"type": "string", "enum": ["north", "south", "east", "west"]}},
     ["north", "south", "east", "west"]),
    ("s08", "groups", "Return one or two groups, each with a name and exactly two member names.",
     {"type": "object", "properties": {"groups": {"type": "array", "minItems": 1, "maxItems": 2, "items": {
         "type": "object", "properties": {"name": {"type": "string"},
                                          "members": {"type": "array", "minItems": 2, "maxItems": 2,
                                                      "items": {"type": "string"}}},
         "required": ["name", "members"], "additionalProperties": False}}},
      "required": ["groups"], "additionalProperties": False}, None),
]


def json_cases():
    out = []
    for cid, name, task, schema, exact_set in JSON_CASES:
        schema_text = json.dumps(schema, separators=(",", ":"))
        sets = ["dev", "full"] if cid in JSON_DEV else ["full"]
        out.append(case(f"structured-{cid}", "structured", sets,
                        "campaign/quality/build_cases.py#JSON_CASES", "Apache-2.0", name=name,
                        mode="json_schema", task=task, schema=schema_text, exact_set=exact_set,
                        prompt=STRUCTURED_JSON_PROMPT.format(task=task, schema=schema_text)))
    return out


TOOL_DEV = {"s09"}

TOOL_CASES = [
    ("s09", "weather", "Give the weather in Lisbon in Celsius.", "required", None,
     [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
                                        "parameters": {"type": "object", "properties": {
                                            "city": {"type": "string"},
                                            "unit": {"type": "string", "enum": ["c", "f"]}},
                                            "required": ["city", "unit"], "additionalProperties": False}}}],
     "get_weather"),
    ("s10", "convert", "Convert 12.5 kilometres to miles.", "named", "convert",
     [{"type": "function", "function": {"name": "convert", "description": "Convert a measurement between units.",
                                        "parameters": {"type": "object", "properties": {
                                            "amount": {"type": "number"},
                                            "from": {"type": "string", "enum": ["km", "mi"]},
                                            "to": {"type": "string", "enum": ["km", "mi"]}},
                                            "required": ["amount", "from", "to"], "additionalProperties": False}}},
      {"type": "function", "function": {"name": "ping", "description": "Check availability.",
                                        "parameters": {"type": "object", "properties": {"host": {"type": "string"}},
                                                       "required": ["host"], "additionalProperties": False}}}],
     "convert"),
    ("s11", "search", "Search for notes about tide tables and lunar phases, at most 3 queries and at most 20 results.",
     "required", None,
     [{"type": "function", "function": {"name": "search", "description": "Search the note archive.",
                                        "parameters": {"type": "object", "properties": {
                                            "query": {"type": "array", "minItems": 1, "maxItems": 3,
                                                      "items": {"type": "string"}},
                                            "limit": {"type": "integer", "minimum": 1, "maximum": 20}},
                                            "required": ["query", "limit"], "additionalProperties": False}}}],
     "search"),
    ("s12", "population", "Look up the population of Lisbon.", "required", None,
     [{"type": "function", "function": {"name": "population_lookup", "description": "Population of a city.",
                                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                       "required": ["city"], "additionalProperties": False}}},
      {"type": "function", "function": {"name": "translate", "description": "Translate text.",
                                        "parameters": {"type": "object", "properties": {
                                            "text": {"type": "string"},
                                            "target": {"type": "string", "enum": ["en", "pt", "ja"]}},
                                            "required": ["text", "target"], "additionalProperties": False}}}],
     "population_lookup"),
]


def tool_cases():
    out = []
    for cid, name, task, choice_kind, choice_name, tools, expected in TOOL_CASES:
        choice = "required" if choice_kind == "required" else {"type": "function", "function": {"name": choice_name}}
        sets = ["dev", "full"] if cid in TOOL_DEV else ["full"]
        out.append(case(f"structured-{cid}", "structured", sets,
                        "campaign/quality/build_cases.py#TOOL_CASES", "Apache-2.0", name=name,
                        mode="tool_call", task=task, tools=tools, tool_choice=choice, expected_name=expected,
                        prompt=TOOL_PROMPT.format(task=task)))
    return out


REASONING_IDS = [("gsm8k", "gsm8k-167"), ("gsm8k", "gsm8k-83"), ("mmlu", "mmlu-163"),
                 ("gsm8k", "gsm8k-3"), ("gsm8k", "gsm8k-23"), ("gsm8k", "gsm8k-86"),
                 ("gsm8k", "gsm8k-120"), ("gsm8k", "gsm8k-221"), ("gsm8k", "gsm8k-278"),
                 ("mmlu", "mmlu-593"), ("mmlu", "mmlu-600"), ("mmlu", "mmlu-1768")]

REASONING_DEV = {"gsm8k-167", "gsm8k-83", "mmlu-163"}


def reasoning_cases():
    data = json.loads(QUALITY_JSON.read_text(encoding="utf-8"))
    index = {}
    for category, block in data["categories"].items():
        for item in block["items"]:
            index[(category, item["id"])] = item
    out = []
    for number, (category, item_id) in enumerate(REASONING_IDS, 1):
        item = index[(category, item_id)]
        sets = ["dev", "full"] if item_id in REASONING_DEV else ["full"]
        out.append(case(f"reasoning-{item_id}", "reasoning", sets,
                        f"{REASONING_SOURCE}:{category}:{item_id}", REASONING_LICENCE[category],
                        category=category, item=item))
    return out


RETRIEVAL_SPECS = [
    ("t01", "campaign/prompts/dev/code-4096.txt", 4096, 0.15, "ZX-51", "Zephyr Point relay station", "QX-7419",
     ["dev", "full"]),
    ("t02", "campaign/prompts/dev/prose-4096.txt", 4096, 0.5, "ZX-52", "Ferngate water tower", "VN-6403",
     ["dev", "full"]),
    ("t03", "campaign/prompts/dev/structured-4096.txt", 4096, 0.85, "ZX-53", "Marlowe depot", "KT-3187",
     ["full"]),
    ("t04", "campaign/prompts/holdout/heldout-code-4096.txt", 4096, 0.35, "ZX-54", "Ashby cable hut", "TQ-9052",
     ["full"]),
    ("t05", "campaign/prompts/dev/code-32768.txt", 32768, 0.1, "ZX-55", "Norwood pump station", "PL-5317",
     ["dev", "full"]),
    ("t06", "campaign/prompts/dev/prose-32768.txt", 32768, 0.5, "ZX-56", "Cairn Hill observatory", "RM-2277",
     ["full"]),
    ("t07", "campaign/prompts/dev/structured-32768.txt", 32768, 0.9, "ZX-57", "Belmont substation", "HD-4182",
     ["full"]),
    ("t08", "campaign/prompts/holdout/heldout-code-32768.txt", 32768, 0.65, "ZX-58", "Oakhurst relay", "BG-8642",
     ["full"]),
]


def retrieval_cases():
    out = []
    for cid, source, cell, position, record, entity, answer, sets in RETRIEVAL_SPECS:
        needle = (f"Facility record {record}: the access code for the {entity} is {answer}. "
                  f"The {entity} was inspected on 14 March.")
        question = (f"According to the facility records in the text, what is the access code for the {entity}? "
                    f"End your reply with the code in double square brackets, like [[AB-0000]].")
        out.append(case(f"retrieval-{cid}", "retrieval", sets, source, "Apache-2.0", cell=cell,
                        position=position, needle=needle, answer=answer, question=question,
                        prompt_file=f"campaign/quality/cases/retrieval/{cid}.txt",
                        source_sha256=hashlib.sha256((REPO / source).read_bytes()).hexdigest()))
    return out


def build():
    cases = code_cases() + json_cases() + tool_cases() + reasoning_cases() + retrieval_cases()
    ids = [c["id"] for c in cases]
    if len(cases) != 48 or len(set(ids)) != 48:
        raise SystemExit(f"the suite must hold 48 distinct cases, got {len(cases)} with {len(set(ids))} ids")
    for kind, want in (("code", 16), ("structured", 12), ("reasoning", 12), ("retrieval", 8)):
        got = sum(1 for c in cases if c["kind"] == kind)
        if got != want:
            raise SystemExit(f"kind {kind} must hold {want} cases, got {got}")
    dev = [c for c in cases if "dev" in c["sets"]]
    if len(dev) != 12 or len({c["kind"] for c in dev}) != 4:
        raise SystemExit(f"the dev subset must hold 12 cases across four kinds, got {len(dev)}")
    return {"schema": 1, "cases": cases}


def main() -> int:
    document = build()
    text = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if "--check" in sys.argv[1:]:
        if not OUT.is_file() or OUT.read_text(encoding="utf-8") != text:
            print(f"{OUT} does not match build_cases.py", file=sys.stderr)
            return 1
        print(f"{OUT} matches build_cases.py ({len(document['cases'])} cases)", file=sys.stderr)
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT} with {len(document['cases'])} cases", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
