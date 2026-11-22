"""The campaign's total context budget, including the rendered prompt and reserved reply tokens."""

import json
import urllib.request

CONTEXT_TOKENS = 32768


def context_problem(prompt_tokens: int, max_tokens: int) -> str | None:
    if type(prompt_tokens) is not int or type(max_tokens) is not int or min(prompt_tokens, max_tokens) < 0:
        raise ValueError("context counts must be nonnegative integers")
    if prompt_tokens + max_tokens <= CONTEXT_TOKENS:
        return None
    return (f"campaign context budget exceeded: {prompt_tokens} rendered prompt tokens + {max_tokens} reply tokens "
            f"> {CONTEXT_TOKENS}; prompt must fit {max(0, CONTEXT_TOKENS - max_tokens)} tokens; "
            "the fixture was not truncated")


def rendered_prompt_tokens(base: str, body: dict) -> int:
    request = urllib.request.Request(f"{base}/v1/tokenize", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.loads(response.read())
    count = result.get("count")
    if type(count) is not int or count < 0:
        raise ValueError("tokenizer returned no valid rendered prompt count")
    return count
