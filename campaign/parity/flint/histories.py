"""Write the held-out parity corpus and assert both engines tokenize it identically.

The texts are authored below (no repeats, no generated filler), rendered with the checkpoint's own chat template
including assistant reasoning, then tokenized once. The stored ids come from the checkpoint tokenizer; the parity
target asserts TensorFold's Zig tokenizer returns the same ids on the same rendered text before either engine runs.
"""

import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer
from rec_mlx.chat import load_tokenizer, normalize_messages

MODEL = Path.home() / "models" / "flint-int4@3b93f9053b53"
OUT = Path(__file__).with_name("histories.json")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run a project's test command and return its tail",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "select": {"type": ["string", "null"]},
                    "timeout_s": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    }
]

HISTORIES = [
    {
        "name": "linked-list-reversal",
        "kind": "code",
        "messages": [
            {"role": "system", "content": "You are a precise coding assistant. Prefer small, tested functions."},
            {"role": "user", "content": "Reverse a singly linked list in Python, iteratively, and explain the pointer updates."},
            {
                "role": "assistant",
                "reasoning_content": "Three pointers suffice. Save the successor before relinking, then advance. The head becomes the old tail.",
                "content": (
                    "```python\n"
                    "class Node:\n"
                    "    def __init__(self, value, next=None):\n"
                    "        self.value = value\n"
                    "        self.next = next\n\n"
                    "def reverse(head: Node | None) -> Node | None:\n"
                    "    prev = None\n"
                    "    while head is not None:\n"
                    "        head.next, prev, head = prev, head, head.next\n"
                    "    return prev\n"
                    "```\n\n"
                    "Each iteration flips exactly one link before moving on, so the list reverses in place with O(1) extra space."
                ),
            },
            {"role": "user", "content": "Now the recursive form, and say what the stack costs on a 100k-node list."},
        ],
    },
    {
        "name": "json-invoice-records",
        "kind": "structured",
        "messages": [
            {"role": "system", "content": "Answer with strict JSON. No prose outside the JSON object."},
            {"role": "user", "content": "Produce two invoice records for Nordic clients in October 2026, with a nested line list and integer cents."},
            {
                "role": "assistant",
                "reasoning_content": "Two records, distinct ids and currencies. Amounts in integer cents, ISO dates, and a totals object.",
                "content": (
                    "{\n"
                    '  "invoices": [\n'
                    "    {\n"
                    '      "id": "INV-2026-1041",\n'
                    '      "customer": {"name": "Fjordlys AS", "country": "NO", "vat_id": "NO991234567MVA"},\n'
                    '      "issued": "2026-10-03",\n'
                    '      "currency": "NOK",\n'
                    '      "lines": [\n'
                    '        {"sku": "TRAIN-REMOTE", "qty": 4, "unit_cents": 125000},\n'
                    '        {"sku": "SEAT-EXTRA", "qty": 1, "unit_cents": 34900}\n'
                    "      ],\n"
                    '      "totals": {"net_cents": 534900, "vat_cents": 133725, "gross_cents": 668625},\n'
                    '      "terms_days": 30\n'
                    "    },\n"
                    "    {\n"
                    '      "id": "INV-2026-1042",\n'
                    '      "customer": {"name": "Sjöberg & Söner AB", "country": "SE", "vat_id": "SE556789012301"},\n'
                    '      "issued": "2026-10-19",\n'
                    '      "currency": "SEK",\n'
                    '      "lines": [\n'
                    '        {"sku": "LIC-ANNUAL", "qty": 1, "unit_cents": 899000},\n'
                    '        {"sku": "SUPPORT-HR", "qty": 12, "unit_cents": 15900}\n'
                    "      ],\n"
                    '      "totals": {"net_cents": 1089800, "vat_cents": 272450, "gross_cents": 1362250},\n'
                    '      "terms_days": 14\n'
                    "    }\n"
                    "  ]\n"
                    "}"
                ),
            },
            {"role": "user", "content": "Recalculate the second record's totals with 25% VAT and show only the changed numbers."},
        ],
    },
    {
        "name": "tool-call-test-selection",
        "kind": "structured",
        "messages": [
            {"role": "system", "content": "You can call run_tests. Prefer the narrowest selection that still covers the change."},
            {"role": "user", "content": "I changed the tokenizer's handling of combining marks. Run the relevant tests with a 900 second budget."},
            {
                "role": "assistant",
                "reasoning_content": "The change is in the tokenizer contract, so select the tokenizer tests and not the whole suite.",
                "content": "Running the tokenizer tests only.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_tests",
                            "arguments": {"path": "zig/src/core/tokenizer", "select": "normalizer|combining", "timeout_s": 900},
                        }
                    }
                ],
            },
            {"role": "tool", "content": "9 passed, 1 skipped in 42.8 s"},
            {"role": "user", "content": "The skip is new. What are the usual reasons a tokenizer mark test would skip?"},
        ],
        "tools": TOOLS,
    },
    {
        "name": "zig-slice-bounds",
        "kind": "code",
        "messages": [
            {"role": "user", "content": "In Zig 0.17, why does `list.items[start..end]` panic at runtime rather than type-error, and how do I make it loud at compile time?"},
            {
                "role": "assistant",
                "reasoning_content": "Runtime slice bounds are checked in safety builds. Compile-time knowledge needs comptime-known lengths or a check with std.debug.assert.",
                "content": (
                    "A slice's bounds are values, so they are checked at run time in safety builds and are unchecked in "
                    "release-fast. To make the check statically visible, keep the bound as a `comptime` value and slice "
                    "with a fixed-length array type:\n\n"
                    "```zig\n"
                    "fn first4(items: []const u8) !*const [4]u8 {\n"
                    "    if (items.len < 4) return error.TooShort;\n"
                    "    return items[0..4];\n"
                    "}\n"
                    "```\n\n"
                    "The explicit length test states the precondition once; the returned pointer-to-array then carries the "
                    "length in its type, so callers cannot widen it by accident."
                ),
            },
            {"role": "user", "content": "How does that interact with `std.mem.alignForward` when the alignment is a runtime value?"},
        ],
    },
    {
        "name": "rust-borrow-tangle",
        "kind": "prose",
        "messages": [
            {"role": "user", "content": "This Rust snippet fails to compile. Explain the borrow conflict precisely and fix it without clones."},
            {
                "role": "assistant",
                "reasoning_content": "The loop borrows the map immutably while the body wants to insert. Split collect and mutate phases.",
                "content": (
                    "The failure is an immutable borrow of `entries` held across a mutable one: the `for` loop's iterator "
                    "keeps `&entries` alive for the whole body, so `entries.insert` inside it is rejected. The fix is to "
                    "collect the work first, end the borrow, and then mutate:\n\n"
                    "```rust\n"
                    "let pending: Vec<(String, u32)> = entries\n"
                    "    .iter()\n"
                    "    .filter(|(_, count)| **count == 0)\n"
                    "    .map(|(key, _)| (key.clone(), 1))\n"
                    "    .collect();\n\n"
                    "for (key, value) in pending {\n"
                    "    entries.insert(key, value);\n"
                    "}\n"
                    "```\n\n"
                    "The clone is of a key already owned by the map, so no bulk copy of the map happens; the alternative is "
                    "`retain` plus a second pass, which reads worse for this shape."
                ),
            },
            {"role": "user", "content": "Could `entry` plus a manual index loop avoid the temporary vector, and what does that cost in code?"},
        ],
    },
    {
        "name": "retry-backoff-analysis",
        "kind": "prose",
        "messages": [
            {"role": "system", "content": "Be concrete about failure modes and arithmetic."},
            {"role": "user", "content": "We retry a flaky upload with exponential backoff, 5 attempts, base 200 ms, full jitter off. Estimate the worst-case added latency and when it is the wrong policy."},
            {
                "role": "assistant",
                "reasoning_content": "Sum 200, 400, 800, 1600 ms between five attempts is 3.0 s of waiting plus the timeouts. Full jitter would spread contention.",
                "content": (
                    "Four gaps of 200, 400, 800 and 1600 ms sum to 3.0 s, so the worst case adds three seconds on top of "
                    "however long each attempt spends before failing.\n\n"
                    "That policy is wrong when many clients retry in lockstep: without jitter they line up again on the "
                    "server's recovery edge, so the first retry storm is merely delayed. It is also wrong for interactive "
                    "requests with a user-visible deadline under about four seconds, where a single retry plus an honest "
                    "error beats four silent waits. Read-mostly idempotent uploads tolerate it; writes that may already "
                    "have committed do not, because a duplicate upload is a correctness problem rather than a delay."
                ),
            },
            {"role": "user", "content": "Now add decorrelated jitter and say what it changes about the tail."},
        ],
    },
    {
        "name": "observability-plan",
        "kind": "prose",
        "messages": [
            {"role": "system", "content": "You are an SRE reviewing a plan. Name the failure it does not catch."},
            {"role": "user", "content": "Review this plan: add p99 latency and error-rate alerts per endpoint, sample 1% of traces, and page on three consecutive bad minutes."},
            {
                "role": "assistant",
                "reasoning_content": "It catches burns but not slow saturation, and sampling 1% can hide rare fatal errors. Three minutes is long for a payment path.",
                "content": (
                    "It catches sustained burns and misses three things. First, saturation: a queue that drains slower than "
                    "it fills raises no error rate and may keep p99 under target until it collapses. Second, rare fatal "
                    "paths: at 1% sampling an error that hits one request in ten thousand may not appear in a trace for "
                    "hours, so the alert fires with no usable example. Third, scope: three consecutive bad minutes on a "
                    "payment path means minutes of failed charges, so that budget belongs to a batch job, not checkout.\n\n"
                    "Add headroom and queue-depth alerts, keep 100% of error traces with a cheap tail-based rule, and give "
                    "the payment path a one-minute page with a lower absolute threshold."
                ),
            },
            {"role": "user", "content": "Which of those three additions would you ship first if only one quarter of the work fits?"},
        ],
    },
    {
        "name": "token-bucket-ts",
        "kind": "code",
        "messages": [
            {"role": "user", "content": "Implement a token bucket in TypeScript with monotonic time, and make refill exact for fractional rates."},
            {
                "role": "assistant",
                "reasoning_content": "Store tokens as a float and the last refill instant. Compute elapsed from performance.now in ms, add rate times seconds, clamp at capacity.",
                "content": (
                    "```ts\n"
                    "export class TokenBucket {\n"
                    "  private tokens: number;\n"
                    "  private updatedMs: number;\n"
                    "  constructor(private readonly capacity: number, private readonly perSecond: number, nowMs = performance.now()) {\n"
                    "    this.tokens = capacity;\n"
                    "    this.updatedMs = nowMs;\n"
                    "  }\n\n"
                    "  take(count = 1, nowMs = performance.now()): boolean {\n"
                    "    const elapsedS = Math.max(0, nowMs - this.updatedMs) / 1000;\n"
                    "    this.tokens = Math.min(this.capacity, this.tokens + elapsedS * this.perSecond);\n"
                    "    this.updatedMs = nowMs;\n"
                    "    if (this.tokens < count) return false;\n"
                    "    this.tokens -= count;\n"
                    "    return true;\n"
                    "  }\n"
                    "}\n"
                    "```\n\n"
                    "Refill is computed lazily on each call, so a fractional rate stays exact instead of truncating per tick."
                ),
            },
            {"role": "user", "content": "Make it safe under concurrent callers without a lock, and say what invariant that relies on."},
        ],
    },
    {
        "name": "binary-search-edges",
        "kind": "code",
        "messages": [
            {"role": "user", "content": "Show a binary search over a sorted array of duplicate keys that finds the first index >= target, and prove the loop terminates."},
            {
                "role": "assistant",
                "reasoning_content": "Lower bound with half-open interval. mid = lo + (lo+hi)>>1 keeps mid < hi so the interval shrinks.",
                "content": (
                    "```python\n"
                    "def lower_bound(values: list[int], target: int) -> int:\n"
                    "    lo, hi = 0, len(values)\n"
                    "    while lo < hi:\n"
                    "        mid = lo + (hi - lo) // 2\n"
                    "        if values[mid] < target:\n"
                    "            lo = mid + 1\n"
                    "        else:\n"
                    "            hi = mid\n"
                    "    return lo\n"
                    "```\n\n"
                    "Termination: `mid` satisfies lo <= mid < hi, so either `lo` rises past `mid` or `hi` falls to `mid`; "
                    "each iteration shortens the half-open interval [lo, hi) by at least one element. The invariant is that "
                    "every index before `lo` is < target and every index from `hi` on is >= target, which also gives the "
                    "answer directly: `lo` is the first index >= target, and len(values) when none exists."
                ),
            },
            {"role": "user", "content": "Extend it to return both bounds in one pass and note the guard for an empty input."},
        ],
    },
    {
        "name": "long-module-scheduler",
        "kind": "code",
        "messages": [
            {"role": "user", "content": "Review this module: it schedules rate-limited tasks across a small worker pool. Point out the concurrency bug and the missing cancellation path."},
            {
                "role": "assistant",
                "reasoning_content": "The queue is guarded, but the in-flight set is appended outside the lock, so the drain loop can miss a task. Cancellation is absent.",
                "content": (
                    "```python\n"
                    "import asyncio, time\n"
                    "from dataclasses import dataclass, field\n\n"
                    "@dataclass\n"
                    "class Scheduler:\n"
                    "    workers: int = 4\n"
                    "    budget_per_second: int = 200\n"
                    "    _queue: asyncio.Queue = field(default_factory=asyncio.Queue)\n"
                    "    _inflight: set = field(default_factory=set)\n"
                    "    _issued: int = 0\n\n"
                    "    async def submit(self, task) -> None:\n"
                    "        await self._queue.put(task)\n\n"
                    "    async def _worker(self):\n"
                    "        while True:\n"
                    "            task = await self._queue.get()\n"
                    "            self._inflight.add(task)\n"
                    "            try:\n"
                    "                await task.run()\n"
                    "            finally:\n"
                    "                self._inflight.discard(task)\n"
                    "                self._queue.task_done()\n\n"
                    "    async def run(self, until):\n"
                    "        while time.monotonic() < until:\n"
                    "            if self._issued >= self.budget_per_second:\n"
                    "                await asyncio.sleep(0.05)\n"
                    "                self._issued = 0\n"
                    "            workers = [asyncio.create_task(self._worker()) for _ in range(self.workers)]\n"
                    "            self._issued += len(workers)\n"
                    "            await asyncio.gather(*workers)\n"
                    "```\n\n"
                    "Two problems. `run` creates a fresh worker set each loop and waits for all of them, so `_worker`'s "
                    "unbounded loop keeps `gather` from returning: the budget check never runs again. And nothing cancels "
                    "in-flight work, so shutdown waits on tasks that may never finish.\n\n"
                    "Start the pool once, let the rate limiter live in the worker's loop, and add a cancellation event the "
                    "worker checks between awaits while `run` cancels and awaits the pool."
                ),
            },
            {"role": "user", "content": "Rewrite only the parts you flagged, keeping the public methods stable."},
        ],
    },
    {
        "name": "transformer-attention-math",
        "kind": "prose",
        "messages": [
            {"role": "user", "content": "Derive why scaled dot-product attention divides by sqrt(d_k), and how a learned per-head sink changes the softmax denominator."},
            {
                "role": "assistant",
                "reasoning_content": "The dot product of independent unit-variance vectors has variance d_k, so scaling restores unit variance before softmax. A sink adds one more term to the denominator, so probabilities sum below one.",
                "content": (
                    "If query and key entries are independent with unit variance, their dot product over d_k terms has "
                    "variance d_k. Without scaling, logits grow with dimension, softmax saturates, and gradients through it "
                    "vanish, so dividing by the standard deviation sqrt(d_k) keeps the logits near unit scale regardless of "
                    "head width.\n\n"
                    "A learned sink is an extra logit appended per head before the softmax. Because softmax normalizes over "
                    "all terms, the attention weights on real keys then sum to 1 - exp(sink) / sum, which is strictly below "
                    "one. The remainder is probability mass the head can spend on attending to nothing, and its learned "
                    "value is per head, so one head can hold a mostly-flat distribution while another stays sharp."
                ),
            },
            {"role": "user", "content": "With 32 heads and a sink of 2.0, what fraction of mass does the sink take when the real logits are all zero?"},
        ],
    },
    {
        "name": "dijkstra-walkthrough",
        "kind": "code",
        "messages": [
            {"role": "system", "content": "You teach algorithms with runnable code and a worked trace."},
            {"role": "user", "content": "Implement Dijkstra with a binary heap and show a trace on a small weighted graph. I care about the lazy-deletion detail."},
            {
                "role": "assistant",
                "reasoning_content": "Use heapq with (distance, node) pairs and skip a popped entry whose distance is stale. The stale entries are cheap because each relaxation pushes at most one.",
                "content": (
                    "```python\n"
                    "import heapq\n"
                    "from dataclasses import dataclass\n\n"
                    "@dataclass(frozen=True)\n"
                    "class Edge:\n"
                    "    to: str\n"
                    "    weight: int\n\n"
                    "def shortest_paths(graph: dict[str, list[Edge]], source: str) -> dict[str, int]:\n"
                    "    best: dict[str, int] = {source: 0}\n"
                    "    heap: list[tuple[int, str]] = [(0, source)]\n"
                    "    while heap:\n"
                    "        distance, node = heapq.heappop(heap)\n"
                    "        if distance > best.get(node, float('inf')):\n"
                    "            continue\n"
                    "        for edge in graph.get(node, ()):\n"
                    "            candidate = distance + edge.weight\n"
                    "            if candidate < best.get(edge.to, float('inf')):\n"
                    "                best[edge.to] = candidate\n"
                    "                heapq.heappush(heap, (candidate, edge.to))\n"
                    "    return best\n"
                    "```\n\n"
                    "The `continue` is the lazy deletion. Python's heap has no decrease-key, so a shorter path to a node "
                    "pushes a second entry and the older, longer one stays in the heap. When it eventually surfaces its "
                    "distance no longer matches `best`, so it is discarded. Each edge relaxation pushes at most once, so "
                    "the heap holds at most |E| entries and the loop stays O((V + E) log E).\n\n"
                    "Trace on A -> B (4), A -> C (1), C -> B (2), B -> D (5), C -> D (8). Start with (0, A). Popping A "
                    "relaxes B to 4 and C to 1, so the heap is [(1, C), (4, B)]. C is popped at 1 and relaxes B to 3 and D "
                    "to 9, giving [(3, B), (4, B), (9, D)]. B at 3 relaxes D to 8, so the stale (4, B) is skipped when it "
                    "surfaces, and the final distances are A 0, C 1, B 3, D 8."
                ),
            },
            {"role": "user", "content": "Now handle a negative edge: show exactly where the greedy loop breaks and what a Bellman-Ford fallback would detect."},
            {
                "role": "assistant",
                "reasoning_content": "A negative edge means a settled node can later improve, which Dijkstra never revisits. Bellman-Ford relaxes every edge V-1 times and then reports an improvement as a negative cycle.",
                "content": (
                    "Add B -> C with weight -3. Dijkstra pops B at 3 and would relax C to 0, but C was already settled at "
                    "1 and the algorithm never revisits it, so the reported distance 1 is wrong. The greedy invariant is "
                    "that a popped node's distance is final, and a negative edge breaks exactly that: a node can still be "
                    "improved after it is popped.\n\n"
                    "Bellman-Ford keeps the same `best` map but relaxes all edges V-1 times, so improvements propagate "
                    "through the whole graph. A final pass that still improves an entry proves a negative cycle reachable "
                    "from the source, which is the check worth keeping even if the fast path is Dijkstra."
                ),
            },
            {"role": "user", "content": "Sketch A* on top of this code and say what an inadmissible heuristic does to the result."},
        ],
    },
    {
        "name": "rollout-postmortem",
        "kind": "prose",
        "messages": [
            {"role": "system", "content": "You write terse, factual incident reviews. Attribute causes, do not assign blame."},
            {"role": "user", "content": "Write the postmortem for a canary deploy that took 8% of traffic and then rolled back after 11 minutes. The metric that fired was a 4x rise in 5xx on checkout. Include the timeline, the two plausible causes and the action items."},
            {
                "role": "assistant",
                "reasoning_content": "Structure: impact, timeline, what we know, what we do not, actions. The two candidates are a schema change in the payment serializer and a retry storm from the new timeout.",
                "content": (
                    "**Impact.** From 14:02 to 14:13 UTC, 8% of checkout traffic hit the canary. 5xx rose from 0.3% to "
                    "1.2% on the canary slice and stayed flat on the control slice, so roughly 1,400 requests failed. No "
                    "charges were double-captured; the failures were all before capture.\n\n"
                    "**Timeline.** 13:58 the canary service accepted 8% of traffic. 14:02 the 5xx rate on the slice "
                    "crossed the page threshold and the on-call acknowledged. 14:05 the payment serializer's error logs "
                    "showed `KeyError: 'amount_minor'` on about one request in two hundred. 14:09 the deploy was paused "
                    "at 8%. 14:13 rollback finished and the error rate returned to baseline within two minutes.\n\n"
                    "**Two plausible causes.** First, the canary build expects `amount_minor` while the older clients and "
                    "the queue's dead-letter replay still send `amount`, so the serializer's new strict field name fails "
                    "on mixed traffic; that matches the intermittent rate and the exact exception. Second, the new 2 s "
                    "payment timeout is shorter than the 2.5 s p99 of the capture call, so the client retries, and each "
                    "retry multiplies load during the slow window. The second would raise latency before errors, which we "
                    "did not see, so the field rename is the leading cause and the retry behaviour is a contributing "
                    "amplifier rather than the trigger.\n\n"
                    "**What we do not know.** Whether the canary's request mix matched production is unverified; the "
                    "slice is chosen by hash of the account id, not by endpoint, so it could over-represent replay "
                    "traffic. We also have not reproduced the failure against a captured payload.\n\n"
                    "**Actions.** Accept both field names for one release with a deprecation log, and add a contract "
                    "test that replays a recorded dead-letter payload. Raise the payment timeout to 4 s and cap retries "
                    "at one with full jitter. Make the canary select by endpoint and account, and gate checkout on a "
                    "captured-traffic check rather than hash alone. Add an alert on serializer field errors, which "
                    "would have paged two minutes before the rate threshold."
                ),
            },
            {"role": "user", "content": "Two of those actions are cheap and two are structural. Rank them by the risk they retire per unit of effort, and name the one that would have prevented this incident outright."},
        ],
    },
    {
        "name": "arrow-etl-pipeline",
        "kind": "code",
        "messages": [
            {"role": "system", "content": "Prefer typed, streaming code with explicit schemas."},
            {"role": "user", "content": "Write a streaming CSV-to-Arrow ETL in Python that validates types, partitions by day, and never loads a whole file. Use the standard library plus pyarrow."},
            {
                "role": "assistant",
                "reasoning_content": "Read line by line, batch into record batches of a fixed size, write partitioned datasets through a DatasetWriter per day. Validate with a schema before appending so one bad row fails loudly.",
                "content": (
                    "```python\n"
                    "import csv, datetime as dt, pathlib\n"
                    "import pyarrow as pa, pyarrow.dataset as ds\n\n"
                    "SCHEMA = pa.schema([\n"
                    "    ('trip_id', pa.string()),\n"
                    "    ('depot', pa.string()),\n"
                    "    ('departed', pa.timestamp('s')),\n"
                    "    ('dwell_s', pa.int32()),\n"
                    "    ('fare_cents', pa.int64()),\n"
                    "])\n\n"
                    "def rows(path: pathlib.Path, batch: int = 2048):\n"
                    "    with path.open(newline='', encoding='utf-8') as handle:\n"
                    "        reader = csv.DictReader(handle)\n"
                    "        chunk = []\n"
                    "        for raw in reader:\n"
                    "            chunk.append({\n"
                    "                'trip_id': raw['trip_id'],\n"
                    "                'depot': raw['depot'],\n"
                    "                'departed': dt.datetime.fromisoformat(raw['departed']),\n"
                    "                'dwell_s': int(raw['dwell_s']),\n"
                    "                'fare_cents': int(raw['fare_cents']),\n"
                    "            })\n"
                    "            if len(chunk) == batch:\n"
                    "                yield chunk\n"
                    "                chunk = []\n"
                    "        if chunk:\n"
                    "            yield chunk\n\n"
                    "def write(source: pathlib.Path, out: pathlib.Path) -> None:\n"
                    "    writers = {}\n"
                    "    for chunk in rows(source):\n"
                    "        table = pa.Table.from_pylist(chunk, schema=SCHEMA)\n"
                    "        day = table.column('departed')[0].as_py().date() if table.num_rows else None\n"
                    "        if day is None:\n"
                    "            continue\n"
                    "        if day not in writers:\n"
                    "            writers[day] = ds.dataset(out, schema=SCHEMA, partition_schema=pa.schema([('day', pa.date32())])).get_writer()\n"
                    "        writers[day].write_table(table)\n"
                    "    for writer in writers.values():\n"
                    "        writer.close()\n"
                    "```\n\n"
                    "Every batch is schema-checked by `from_pylist`, so a bad integer fails at the offending row instead "
                    "of silently producing nulls. Batches are written as they arrive, so peak memory is one batch."
                ),
            },
            {"role": "user", "content": "That partition loop has a leftover stub and groups by a timestamp column, which is wrong. Fix both, and make a row with an unparseable date fail with its line number."},
        ],
    },
    {
        "name": "debug-session-nan-loss",
        "kind": "code",
        "messages": [
            {"role": "system", "content": "You are pair debugging with an engineer. Ask for the one datum you need, state hypotheses explicitly, and never claim a fix without a check."},
            {"role": "user", "content": "My tiny transformer's loss goes NaN at step 340, always the same step. AdamW, lr 3e-4, cosine schedule, bf16 autocast, batch 32, seq 512. Grad clipping at 1.0 is on. Where do I look first?"},
            {
                "role": "assistant",
                "reasoning_content": "Deterministic step number points at data, not randomness. Ask whether step 340 is a new shard boundary and whether the clip runs before the optimizer step or before the unscale.",
                "content": (
                    "The fact that it is always step 340 is the important clue: a random instability would move. Two "
                    "candidates fit a fixed step.\n\n"
                    "First, step 340 may be the first batch of a shard whose examples differ, for instance one with a "
                    "zero-length target or an all-masked label row. Second, the clip may run before the loss is unscaled, "
                    "so the effective clip is 1/65536 and gradients survive at full magnitude.\n\n"
                    "Give me one datum: the loss value and gradient norm at steps 338, 339 and 340. If the norm jumps from "
                    "a normal 0.8 to thousands in one step, it is data. If the norm is clipped and flat but the loss still "
                    "diverges, it is the optimizer path."
                ),
            },
            {"role": "user", "content": "338: loss 2.14, norm 0.91. 339: loss 2.13, norm 0.93. 340: loss inf, norm 1840. Then everything is NaN."},
            {
                "role": "assistant",
                "reasoning_content": "A norm jump of 2000x in one step with a flat loss is data or numerics, not the schedule. Ask for the batch's max token id and whether an attention mask is all ones for a padded row.",
                "content": (
                    "That is data or a numerics edge, not the schedule: the loss is flat, then the norm explodes 2000x "
                    "before any schedule change.\n\n"
                    "Check the offending batch for (a) a token id at the vocabulary boundary that indexes past the "
                    "embedding, and (b) a fully padded row whose attention mask lets every key be masked, so softmax "
                    "divides by zero and the backward pass turns that NaN into an inf gradient. Both are common and both "
                    "reproduce at a fixed step when the sampler is a fixed permutation.\n\n"
                    "Log `input_ids.max()`, `attention_mask.sum(-1).min()`, and the loss before and after the unscale at "
                    "step 340, and I will tell you which one it is from the three numbers."
                ),
            },
            {"role": "user", "content": "max id 50256 which equals vocab_size - 1, so that is fine. min mask sum is 0, so one row is entirely padding. Loss before unscale is 2.11, after unscale 2.11 too."},
            {
                "role": "assistant",
                "reasoning_content": "A fully masked row makes softmax over an all -inf row, which is NaN. The fix is to skip such rows in the loss and to make the mask helper never produce an all-false row, e.g. by leaving the first position visible.",
                "content": (
                    "Found it. A row with mask sum 0 gives every key `-inf`, so softmax over that row is 0/0 and the "
                    "gradient is NaN regardless of the clip. The loss stays finite for the batch because you divide by "
                    "the number of tokens, but the NaN gradient poisons Adam's moment estimates on that step and every "
                    "later step inherits it.\n\n"
                    "Two changes, both small. In the loss, compute per-token cross entropy and use a denominator that "
                    "counts only unmasked tokens, so a fully masked row contributes nothing. In the mask builder, refuse "
                    "to produce an all-false row for a padded example: either drop the example or set one position "
                    "visible, which matches what the tokenizer does for an empty completion.\n\n"
                    "Then add a check that runs one forward and backward on a synthetic all-padding batch and asserts "
                    "every gradient is finite. That check fails today and passes after the change, so it is the proof."
                ),
            },
            {"role": "user", "content": "Is there any chance the cosine schedule causes this instead, given it is near the warmup end? And would bf16 vs fp32 change the diagnosis?"},
            {
                "role": "assistant",
                "reasoning_content": "The schedule changes lr smoothly and cannot jump the norm 2000x; bf16 would show as noisy loss earlier, not a single deterministic explosion with an all-masked row present.",
                "content": (
                    "No: cosine moves the learning rate by a fraction of a percent per step near the end of warmup, so "
                    "it cannot multiply a gradient norm by two thousand. It also cannot be bf16-specific in the way you "
                    "describe, because bf16 instability shows as gradually noisier loss with occasional spikes across "
                    "many steps, whereas a single deterministic step with an all-masked row present is structural.\n\n"
                    "One caveat worth keeping: with bf16 autocast the exponent range is the same as fp32 but the mantissa "
                    "is shorter, so a loss that reaches a large but finite value can round to inf earlier. That would "
                    "change where you see the NaN, not whether the all-masked row poisons the gradient. Fix the row, keep "
                    "the finite-gradient check, and re-run; if the loss is still noisy afterwards, then look at the "
                    "schedule and the clip ordering."
                ),
            },
        ],
    },
    {
        "name": "changelog-structured",
        "kind": "structured",
        "messages": [
            {"role": "system", "content": "Return a single JSON changelog object with conventional commits grouped by type."},
            {"role": "user", "content": "Turn these raw entries into a release note: fixed a bf16 rounding bug in the norm kernel; added KV ring reuse; removed the dead draft path; made the CLI refuse unknown flags."},
            {
                "role": "assistant",
                "reasoning_content": "Group by conventional type, keep each subject imperative and short, and include the compatibility note for the removed path.",
                "content": (
                    "{\n"
                    '  "version": "1.4.0",\n'
                    '  "date": "2026-11-21",\n'
                    '  "groups": {\n'
                    '    "fix": [\n'
                    '      {"scope": "metal", "subject": "round the row norm once at the bf16 point"}\n'
                    "    ],\n"
                    '    "feat": [\n'
                    '      {"scope": "flint", "subject": "reuse sliding KV rings across prompt chunks"}\n'
                    "    ],\n"
                    '    "refactor": [\n'
                    '      {"scope": "flint", "subject": "delete the dead draft path"}\n'
                    "    ],\n"
                    '    "cli": [\n'
                    '      {"scope": "server", "subject": "refuse flags the family does not implement"}\n'
                    "    ]\n"
                    "  },\n"
                    '  "breaking": [\n'
                    '    {"change": "the draft flag no longer falls back silently", "migration": "pass --no-drafts explicitly"}\n'
                    "  ]\n"
                    "}"
                ),
            },
            {"role": "user", "content": "Reorder the groups so breaking changes come first and drop any empty group."},
        ],
    },
]


def rendered(model_dir: Path):
    checkpoint = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    from transformers import PreTrainedTokenizerFast

    checkpoint_hf = PreTrainedTokenizerFast.from_pretrained(model_dir, local_files_only=True, fix_mistral_regex=False)
    checkpoint_hf.chat_template = (model_dir / "chat_template.jinja").read_text()
    oracle = load_tokenizer(model_dir)
    return checkpoint, checkpoint_hf, oracle


def main() -> int:
    checkpoint, checkpoint_hf, oracle = rendered(MODEL)
    out = []
    for entry in HISTORIES:
        messages = entry["messages"]
        tools = entry.get("tools")
        text = checkpoint_hf.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True
        )
        ids = checkpoint_hf.apply_chat_template(
            messages, tools=tools, tokenize=True, add_generation_prompt=True, return_dict=False
        )
        oracle_text = oracle.apply_chat_template(
            normalize_messages(messages), tools=tools, tokenize=False, add_generation_prompt=True
        )
        oracle_ids = oracle.apply_chat_template(
            normalize_messages(messages), tools=tools, tokenize=True, add_generation_prompt=True, return_dict=False
        )
        if text != oracle_text or list(ids) != list(oracle_ids):
            raise SystemExit(f"{entry['name']}: the checkpoint template and rec_mlx's bundled template disagree")
        again = checkpoint.encode(text, add_special_tokens=False).ids
        if again != list(ids):
            raise SystemExit(f"{entry['name']}: tokenizer round trip disagrees with apply_chat_template")
        if not 0 < len(ids) <= 4096:
            raise SystemExit(f"{entry['name']}: {len(ids)} tokens is outside 1..4096")
        out.append(
            {
                "name": entry["name"],
                "kind": entry["kind"],
                "role_sequence": [message["role"] for message in messages],
                "reasoning_messages": sum(1 for message in messages if message.get("reasoning_content")),
                "tools": bool(tools),
                "tokens": len(ids),
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "ids_sha256": hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest(),
                "text": text,
                "ids": list(ids),
            }
        )
    payload = {
        "model": "RecursiveSI/flint-int4@3b93f9053b530488405ae9fca51d4900d6ce21f0",
        "tokenizer_sha256": hashlib.sha256((MODEL / "tokenizer.json").read_bytes()).hexdigest(),
        "chat_template_sha256": hashlib.sha256((MODEL / "chat_template.jinja").read_bytes()).hexdigest(),
        "render": "checkpoint chat_template.jinja, add_generation_prompt=True, assistant reasoning_content rendered as <think>",
        "provenance": "authored for this campaign; no model output and no repeated filler",
        "total_tokens": sum(entry["tokens"] for entry in out),
        "histories": out,
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "histories": len(out),
                "total_tokens": payload["total_tokens"],
                "min": min(entry["tokens"] for entry in out),
                "max": max(entry["tokens"] for entry in out),
                "kinds": sorted({entry["kind"] for entry in out}),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
