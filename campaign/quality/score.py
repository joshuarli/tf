"""Score one quality case's reply: tests for code, schema for structured, exact match for reasoning and retrieval."""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "web"))
sys.path.insert(0, str(HERE))

from tensorfold_web import quality  # noqa: E402

import sandbox  # noqa: E402

FENCE = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.DOTALL)
BRACKET = re.compile(r"\[\[([^\[\]]{1,64})\]\]")


def errors(instance, schema, path="$"):
    out = []
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(instance, dict):
            return [f"{path}: expected object"]
        for key in schema.get("required", []):
            if key not in instance:
                out.append(f"{path}: missing {key}")
        properties = schema.get("properties", {})
        for key, value in instance.items():
            if key in properties:
                out.extend(errors(value, properties[key], f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                out.append(f"{path}: unexpected {key}")
    elif kind == "array":
        if not isinstance(instance, list):
            return [f"{path}: expected array"]
        if len(instance) < schema.get("minItems", 0):
            out.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            out.append(f"{path}: more than {schema['maxItems']} items")
        if "items" in schema:
            for index, value in enumerate(instance):
                out.extend(errors(value, schema["items"], f"{path}[{index}]"))
    elif kind == "string":
        if not isinstance(instance, str):
            return [f"{path}: expected string"]
        if "minLength" in schema and len(instance) < schema["minLength"]:
            out.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            out.append(f"{path}: longer than {schema['maxLength']}")
    elif kind == "integer":
        if isinstance(instance, bool) or not isinstance(instance, int):
            return [f"{path}: expected integer"]
    elif kind == "number":
        if isinstance(instance, bool) or not isinstance(instance, (int, float)):
            return [f"{path}: expected number"]
    elif kind == "boolean":
        if not isinstance(instance, bool):
            return [f"{path}: expected boolean"]
    elif kind == "null":
        if instance is not None:
            return [f"{path}: expected null"]
    if "enum" in schema and instance not in schema["enum"]:
        out.append(f"{path}: {instance!r} is not one of {schema['enum']}")
    if "minimum" in schema and isinstance(instance, (int, float)) and instance < schema["minimum"]:
        out.append(f"{path}: below {schema['minimum']}")
    if "maximum" in schema and isinstance(instance, (int, float)) and instance > schema["maximum"]:
        out.append(f"{path}: above {schema['maximum']}")
    return out


def code_block(text: str) -> str | None:
    blocks = FENCE.findall(text)
    if blocks:
        return blocks[-1].strip() + "\n"
    return None


def score_code(case: dict, content: str, work: Path) -> tuple[bool, str]:
    solution = code_block(content)
    if solution is None:
        return False, "no fenced python code block"
    result = sandbox.run(solution + "\n" + case["tests"], work)
    if result["passed"]:
        return True, "tests passed"
    if result["ended"] == "timeout":
        return False, f"stopped at the {sandbox.WALL_S} s wall limit"
    detail = (result["stderr"] or result["stdout"]).strip().splitlines()
    tail = detail[-1] if detail else f"exit {result['exit']}"
    return False, f"tests failed: {tail}"[:300]


def score_structured(case: dict, message: dict) -> tuple[bool, str]:
    content = message.get("content") or ""
    if case["mode"] == "json_schema":
        text = content.strip()
        if not text:
            return False, "empty response"
        try:
            value = json.loads(text)
        except ValueError:
            return False, "response is not one JSON value"
        problems = errors(value, json.loads(case["schema"]))
        if problems:
            return False, "; ".join(problems)[:300]
        if case.get("exact_set") is not None and sorted(value) != sorted(case["exact_set"]):
            return False, f"values {value!r} are not {case['exact_set']!r}"
        return True, "schema and values match"
    calls = message.get("tool_calls") or []
    if not calls:
        return False, "no tool call"
    if len(calls) != 1:
        return False, f"{len(calls)} tool calls, expected one"
    call = calls[0]
    function = call.get("function") or {}
    name = function.get("name") or ""
    if name != case["expected_name"]:
        return False, f"called {name!r}, expected {case['expected_name']!r}"
    offered = next((tool["function"] for tool in case["tools"] if tool["function"]["name"] == name), None)
    if offered is None:
        return False, f"unknown tool {name!r}"
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return False, "arguments are not JSON"
    problems = errors(arguments, offered["parameters"])
    if problems:
        return False, "; ".join(problems)[:300]
    return True, f"{name} arguments match"


def score_reasoning(case: dict, content: str) -> tuple[bool, str]:
    passed, read = quality.score(case["category"], case["item"], content)
    return passed, read


def score_retrieval(case: dict, content: str) -> tuple[bool, str]:
    text = quality.answer_text(content)
    found = BRACKET.findall(text)
    if not found:
        return False, "no [[...]] answer"
    got = found[-1].strip()
    return got == case["answer"], got


def score(case: dict, message: dict, work: Path) -> tuple[bool, str]:
    kind = case["kind"]
    if kind == "code":
        return score_code(case, message.get("content") or "", work)
    if kind == "structured":
        return score_structured(case, message)
    if kind == "reasoning":
        return score_reasoning(case, message.get("content") or "")
    if kind == "retrieval":
        return score_retrieval(case, message.get("content") or "")
    raise ValueError(f"unknown case kind {kind!r}")
