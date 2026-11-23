#!/usr/bin/env python3
"""Paired test of whether sending earlier reasoning back changes Inkling's later answers.

Each conversation runs in two arms under greedy decoding. `replay` sends every earlier assistant turn
back with its reasoning_content and its content, as the fx agent harness does; `stripped` sends the
content only. Both arms send the same first request, so they share turn 1, and each arm builds its
later history from its own replies.

Three task families have answers that code checks:
  chain   a running value through one transform per turn; the visible answer carries the whole state
  recall  terse answers to multi-step transforms, then questions about values that were never shown
  code    a Python function or class that gains one requirement per turn, tested in a sandbox

  reasoning_replay.py run --out DIR --serve BINARY --model-dir CHECKPOINT [--efforts minimal,high]
  reasoning_replay.py run --out DIR --base http://127.0.0.1:PORT/v1
  reasoning_replay.py score --out DIR [--efforts minimal,high]

`run` memoises every reply in DIR/responses.jsonl, keyed by the request body, so a rerun resumes where
the last one stopped. `score` rebuilds both arms' conversations from that file without a server, runs
the code family's tests in the macOS sandbox, and writes DIR/summary.json and DIR/report.md.
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import re
import signal
import socket
import string
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
for entry in (HERE, HERE.parent / "fx"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from bench_subagents import CONTEXT  # noqa: E402

FAMILIES = ("chain", "recall", "code")
ARMS = ("replay", "stripped")
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
TRANSFORMS = 115
MODULUS = 997
SYSTEM = "Review this code context:\n" + CONTEXT
VALUE_LINE = "Reply with only the line 'Value: <number>'."
SEED = "inkling-reasoning-replay"
CONVERSATIONS = 40
TURNS = 5
REQUEST_TIMEOUT = 1800
BOOTSTRAP_DRAWS = 10000


def transform(index: int, value: int) -> int:
    """transform_<index> from the fx workload's code context."""
    return (value * (index + 1) + index) % MODULUS


# Task families. Every turn holds the user text, the expected answer and what scoring needs.

def chain_conversation(index: int, turns: int = TURNS) -> dict:
    rng = random.Random(f"{SEED}-chain-{index}")
    start = rng.randrange(1, MODULUS)
    value, steps = start, []
    for number in range(turns):
        op = rng.randrange(TRANSFORMS)
        value = transform(op, value)
        user = (f"Start with the value {start}. Apply transform_{op} to it. {VALUE_LINE}" if number == 0
                else f"Apply transform_{op} to the current value. {VALUE_LINE}")
        steps.append({"user": user, "expect": value, "ops": [op], "recall": False})
    return {"id": f"chain-{index:02d}", "family": "chain", "system": SYSTEM, "start": start, "turns": steps}


def recall_conversation(index: int) -> dict:
    rng = random.Random(f"{SEED}-recall-{index}")
    start = rng.randrange(1, MODULUS)
    a, b, c, d, e = rng.sample(range(TRANSFORMS), 5)
    s1 = transform(a, start)
    s2 = transform(b, s1)
    s3 = transform(c, s2)
    s4 = transform(d, s3)
    s5 = transform(e, s4)
    final = "Reply with only the line 'Value: <number>' for the final result."
    steps = [
        {"user": f"Start with the value {start}. Apply transform_{a}, then transform_{b}, then transform_{c}, "
                 f"in that order. {final}", "expect": s3, "ops": [a, b, c], "recall": False},
        {"user": f"Now apply transform_{d} and then transform_{e} to that result. {final}", "expect": s5,
         "ops": [d, e], "recall": False},
        {"user": f"In my first message, what was the value right after applying transform_{a}? {VALUE_LINE}",
         "expect": s1, "ops": [], "recall": True},
        {"user": f"In my first message, what was the value right after applying transform_{b}? {VALUE_LINE}",
         "expect": s2, "ops": [], "recall": True},
        {"user": f"In my second message, what was the value right after applying transform_{d}? {VALUE_LINE}",
         "expect": s4, "ops": [], "recall": True},
    ]
    return {"id": f"recall-{index:02d}", "family": "recall", "system": SYSTEM, "start": start, "turns": steps}


WORDS = ["alpha", "beta", "gamma", "red", "blue", "green", "cat", "dog", "the", "and", "of", "data", "model", "test"]
ODD_WORDS = ["2024", "v2", "C++", "it's", "ÉTÉ", "x_y", "a.b", "--", "!", "(draft)"]
FEW_WORDS = ["red", "blue", "green", "cat", "dog", "sun"]
FRUIT = ["apple", "pear", "fig", "kiwi", "plum"]


def _title(rng: random.Random) -> str:
    parts = []
    for _ in range(rng.randint(1, 6)):
        word = rng.choice(WORDS + ODD_WORDS)
        roll = rng.random()
        parts.append(word.upper() if roll < 0.2 else word.capitalize() if roll < 0.5 else word)
    text = parts[0]
    for part in parts[1:]:
        text += rng.choice([" ", "  ", "\t", " - ", ", ", "   "]) + part
    return rng.choice(["", " ", "  "]) + text + rng.choice(["", " ", "\n"])


def slug_cases(p: dict, turn: int, rng: random.Random) -> list:
    texts = ["Hello World", "  Leading and trailing  ", "Tabs\tand\nnewlines   here", "MiXeD CaSe 42",
             "C'est la vie!", "Ünïcödé text", "a - b _ c . d", "already-slugged_text.here", "   ", "",
             "Many     spaces     between     words", "Punctuation: commas, periods. And (brackets)!",
             "__init__ method", "--leading and trailing--", "2024 Annual Report: Q3 Results", "!!!",
             "e.g. 3.14 or 2,718", "a" * (p["n"] - 1) + " tail", "word " * 8]
    texts += [_title(rng) for _ in range(8)]
    return [{"fn": p["fn"], "args": [text]} for text in texts]


def evens_cases(p: dict, turn: int, rng: random.Random) -> list:
    cap = p["cap"]
    lists = [[], [1, 2, 3, 4], [-2, -4, 6], [0, 0, 1], [cap - 1, cap, cap + 1, cap + 2, 2 * cap + 3], [7],
             [-1, -3, -5], [2, 2, 2, 2, 2], [cap * 3, -cap * 3, 1, 8]]
    lists += [[rng.randint(-2 * cap, 2 * cap) for _ in range(rng.randint(3, 12))] for _ in range(8)]
    if turn >= 5:
        lists += [None, [None, 2, None, 4], [None], [cap + 2, None, -2, 6]]
    return [{"fn": p["fn"], "args": [nums]} for nums in lists]


def top_cases(p: dict, turn: int, rng: random.Random) -> list:
    pairs = [("the cat and the hat", 2), ("a b c a b a", 2), ("one two three four", 3), ("zeta alpha zeta alpha beta", 2),
             ("x", 3), ("", 2), ("red red blue blue green", 1), ("red red blue blue green", 10)]
    pairs += [(" ".join(rng.choice(FEW_WORDS) for _ in range(rng.randint(3, 15))), rng.randint(1, 4)) for _ in range(6)]
    if turn >= 2:
        pairs += [("The the THE cat Cat", 2), ("Apple apple banana Banana BANANA", 2), ("Dog cat DOG Cat dog", 3)]
    if turn >= 3:
        pairs += [("Hello, world! Hello? world... yes; yes: yes", 3), ("?!; , . word, word.", 2),
                  ("Stop. stop! STOP? go", 2)]
    cases = [{"fn": p["fn"], "args": [text, k]} for text, k in pairs]
    if turn >= 4:
        cases += [{"fn": p["fn"], "args": ["the cat and the hat and the mat", 3], "kwargs": {"stop": ["the", "and"]}},
                  {"fn": p["fn"], "args": ["The cat, THE dog. the end!", 2], "kwargs": {"stop": ["THE"]}},
                  {"fn": p["fn"], "args": ["red blue red", 2], "kwargs": {"stop": []}},
                  {"fn": p["fn"], "args": ["Alpha beta alpha Beta gamma", 5], "kwargs": {"stop": ["beta"]}}]
    return cases


def _duration(rng: random.Random, units: str) -> str:
    chosen = [unit for unit in units if rng.random() < 0.6] or [rng.choice(units)]
    return "".join(f"{rng.randint(0, 99)}{unit}" for unit in chosen)


def duration_cases(p: dict, turn: int, rng: random.Random) -> list:
    texts = ["2h15m", "45s", "1h30m10s", "3h", "90m", "0s", "1h1s", "12m30s", "100h", "7h7m7s"]
    texts += [_duration(rng, "hms") for _ in range(5)]
    if turn >= 2:
        texts += ["1d", "2d3h", "1d30m", "3d4h5m6s", "0d1s"] + [_duration(rng, "dhms") for _ in range(3)]
    if turn >= 3:
        texts += ["1h 30m", "  45s ", "2D 3H", "1 h 30 m", "10M", "1H2M3S"]
    if turn >= 4:
        texts += ["", "   ", "15", "h", "10x", "5w", "1m1h", "1h1h", "2s3m", "abc", "1h30", "1hh", "1h-30m"]
    if turn >= 5:
        texts += ["-1h30m", "-45s", "- 2d", "-1D1S", "-", "--1h"]
    return [{"fn": p["fn"], "args": [text]} for text in texts]


def rle_cases(p: dict, turn: int, rng: random.Random) -> list:
    m = p["m"]
    texts = ["aaabcc", "abc", "", "a", "z" * 25, "aabbaa", "  spaces  ", "!!??..", "mississippi", "x" * (2 * m) + "yz",
             "q" * (m + 1)]
    texts += ["".join(rng.choice("ab ") * rng.randint(1, 3 * m) for _ in range(rng.randint(1, 5))) for _ in range(6)]
    if turn >= 4:
        texts += ["aAab", "BBbbCc", "Hello", "Aa" * (m + 1)]
    if turn >= 5:
        texts += ["a1", "2024", "ab3c", "x9x"]
    return [{"fn": p["fn"], "args": [text]} for text in texts]


def inventory_cases(p: dict, turn: int, rng: random.Random) -> list:
    lists = [[("apple", 3), ("pear", 2), ("apple", 4)], [], [("fig", 1)], [("x", 5), ("y", -2), ("x", -5)],
             [("b", 2), ("a", 2), ("c", 2)], [("kiwi", 0)]]
    lists += [[(rng.choice(FRUIT), rng.randint(-3, 9)) for _ in range(rng.randint(1, 8))] for _ in range(6)]
    if turn >= 3:
        lists += [[("Apple", 2), ("apple", 3), ("APPLE", -1), ("Pear", 1)], [("Fig", 4), ("fig", -4), ("Kiwi", 2)]]
    if turn >= 4:
        lists += [[("apple", "3"), ("pear", 2.5), ("apple", 2)], [("plum", 1.0), ("plum", 1), ("Fig", "x")]]
    return [{"fn": p["fn"], "args": [entries]} for entries in lists]


def _bank_ops(rng: random.Random) -> list:
    ops, balance = [], 0
    for _ in range(rng.randint(2, 8)):
        if balance and rng.random() < 0.45:
            amount = rng.randint(1, balance)
            ops.append(("withdraw", amount))
            balance -= amount
        else:
            amount = rng.randint(1, 60)
            ops.append(("deposit", amount))
            balance += amount
    return ops


def bank_cases(p: dict, turn: int, rng: random.Random) -> list:
    sequences = [[("deposit", 100), ("withdraw", 30), ("deposit", 5)], [("deposit", 10), ("withdraw", 10)], []]
    sequences += [_bank_ops(rng) for _ in range(5)]
    if turn >= 2:
        sequences += [[("deposit", 20), ("withdraw", 50), ("withdraw", 20)], [("withdraw", 1)],
                      [("deposit", 5), ("withdraw", 6), ("deposit", 1), ("withdraw", 6)]]
    if turn >= 3:
        sequences += [[("deposit", 0), ("deposit", -5), ("deposit", 7), ("withdraw", 0), ("withdraw", -3)]]
    if turn >= 4:
        sequences += [[("history",), ("deposit", 40), ("withdraw", 15), ("withdraw", 99), ("deposit", -1), ("history",)],
                      [("deposit", 3), ("history",), ("deposit", 4), ("history",)]]
    if turn >= 5:
        sequences += [[("deposit", 10), ("withdraw", 4), ("undo",), ("history",), ("undo",), ("undo",), ("history",)],
                      [("undo",)],
                      [("deposit", 8), ("withdraw", 20), ("deposit", 2), ("undo",), ("withdraw", 8), ("undo",), ("history",)]]
    return [{"cls": p["cls"], "ops": ops} for ops in sequences]


ROMAN_PAIRS = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
               (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]


def _roman(n: int) -> str:
    out = []
    for value, letters in ROMAN_PAIRS:
        while n >= value:
            out.append(letters)
            n -= value
    return "".join(out)


def roman_cases(p: dict, turn: int, rng: random.Random) -> list:
    numbers = [1, 3, 4, 9, 14, 40, 44, 90, 99, 400, 444, 900, 1994, 2024, 3888, 3999]
    numbers += [rng.randint(1, 3999) for _ in range(6)]
    cases = [{"fn": p["fn"], "args": [n]} for n in numbers]
    if turn >= 2:
        cases += [{"fn": p["fn"], "args": [n]} for n in (0, -1, 4000, 10000)]
    if turn >= 3:
        cases += [{"fn": p["inv"], "args": [_roman(n)]} for n in [1, 4, 9, 14, 40, 90, 400, 1994, 2024, 3999]
                  + [rng.randint(1, 3999) for _ in range(5)]]
    if turn >= 4:
        cases += [{"fn": p["inv"], "args": [text]} for text in ("xiv", "  MCMXCIV ", "mmxxiv\n", " ix")]
    if turn >= 5:
        cases += [{"fn": p["inv"], "args": [text]}
                  for text in ("IIII", "IC", "", "VV", "IM", "MMMM", "ABC", "XM", "IIX", "VX", "  ")]
    return cases


PROJECTS = [
    {"key": "slug", "cases": slug_cases,
     "params": [{"fn": "slugify", "sep": "-", "n": 12, "default": "untitled"},
                {"fn": "make_slug", "sep": "_", "n": 16, "default": "item"},
                {"fn": "url_key", "sep": "-", "n": 20, "default": "page"},
                {"fn": "to_handle", "sep": ".", "n": 10, "default": "user"},
                {"fn": "clean_title", "sep": "_", "n": 14, "default": "entry"}],
     "asks": ["Write a Python function `$fn(title)` that returns `title` lowercased, with leading and trailing "
              "whitespace removed.",
              "Change `$fn` so that, after that, every run of whitespace inside the text is replaced by a single `$sep`.",
              "Next, after that replacement, delete every character that is not a lowercase ASCII letter, a digit "
              "or `$sep`.",
              "Next, collapse every run of repeated `$sep` characters into one, and remove `$sep` from both ends of "
              "the result.",
              "Finally, cut the result to at most $n characters, removing a `$sep` that the cut leaves at the end. "
              "If the final result is empty, return $default_r instead."],
     "refs": [r'''def $fn(title):
    return title.strip().lower()
''', r'''def $fn(title):
    return $sep_r.join(title.strip().lower().split())
''', r'''def $fn(title):
    text = $sep_r.join(title.strip().lower().split())
    return "".join(ch for ch in text if ch in "abcdefghijklmnopqrstuvwxyz0123456789" + $sep_r)
''', r'''def $fn(title):
    text = $sep_r.join(title.strip().lower().split())
    text = "".join(ch for ch in text if ch in "abcdefghijklmnopqrstuvwxyz0123456789" + $sep_r)
    while $sep_r * 2 in text:
        text = text.replace($sep_r * 2, $sep_r)
    return text.strip($sep_r)
''', r'''def $fn(title):
    text = $sep_r.join(title.strip().lower().split())
    text = "".join(ch for ch in text if ch in "abcdefghijklmnopqrstuvwxyz0123456789" + $sep_r)
    while $sep_r * 2 in text:
        text = text.replace($sep_r * 2, $sep_r)
    text = text.strip($sep_r)[:$n].rstrip($sep_r)
    return text or $default_r
''']},
    {"key": "evens", "cases": evens_cases,
     "params": [{"fn": "sum_evens", "cap": 100}, {"fn": "even_total", "cap": 50}, {"fn": "add_evens", "cap": 75},
                {"fn": "evens_sum", "cap": 250}, {"fn": "total_even", "cap": 1000}],
     "asks": ["Write a Python function `$fn(nums)` that returns the sum of the even integers in the list `nums`.",
              "Change it so that negative numbers are ignored entirely.",
              "Before the even check, replace any number greater than $cap with $cap.",
              "Return a tuple `(total, count)` instead, where `count` is how many numbers were added into the total.",
              "Skip any `None` items in `nums`, and return `(0, 0)` if `nums` itself is `None`."],
     "refs": [r'''def $fn(nums):
    return sum(x for x in nums if x % 2 == 0)
''', r'''def $fn(nums):
    return sum(x for x in nums if x >= 0 and x % 2 == 0)
''', r'''def $fn(nums):
    total = 0
    for x in nums:
        if x < 0:
            continue
        x = min(x, $cap)
        if x % 2 == 0:
            total += x
    return total
''', r'''def $fn(nums):
    total = count = 0
    for x in nums:
        if x < 0:
            continue
        x = min(x, $cap)
        if x % 2 == 0:
            total += x
            count += 1
    return (total, count)
''', r'''def $fn(nums):
    if nums is None:
        return (0, 0)
    total = count = 0
    for x in nums:
        if x is None or x < 0:
            continue
        x = min(x, $cap)
        if x % 2 == 0:
            total += x
            count += 1
    return (total, count)
''']},
    {"key": "topwords", "cases": top_cases,
     "params": [{"fn": "top_words"}, {"fn": "most_common_words"}, {"fn": "frequent_words"}, {"fn": "word_ranking"},
                {"fn": "leading_words"}],
     "asks": ["Write a Python function `$fn(text, k)` that splits `text` on whitespace and returns a list of the `k` "
              "most frequent words, most frequent first, with ties in alphabetical order. If there are fewer than "
              "`k` distinct words, return them all.",
              "Make the counting case-insensitive, and return the words in lowercase.",
              "Before counting, strip any of the characters `.,!?;:` from both ends of each word, and skip words "
              "that become empty.",
              "Add an optional parameter `stop`, empty by default, holding words to leave out of the counts; match "
              "them case-insensitively.",
              "Return a list of `(word, count)` tuples instead of bare words."],
     "refs": [r'''def $fn(text, k):
    counts = {}
    for word in text.split():
        counts[word] = counts.get(word, 0) + 1
    return sorted(counts, key=lambda w: (-counts[w], w))[:k]
''', r'''def $fn(text, k):
    counts = {}
    for word in text.split():
        word = word.lower()
        counts[word] = counts.get(word, 0) + 1
    return sorted(counts, key=lambda w: (-counts[w], w))[:k]
''', r'''def $fn(text, k):
    counts = {}
    for word in text.split():
        word = word.lower().strip(".,!?;:")
        if not word:
            continue
        counts[word] = counts.get(word, 0) + 1
    return sorted(counts, key=lambda w: (-counts[w], w))[:k]
''', r'''def $fn(text, k, stop=()):
    skip = {s.lower() for s in stop}
    counts = {}
    for word in text.split():
        word = word.lower().strip(".,!?;:")
        if not word or word in skip:
            continue
        counts[word] = counts.get(word, 0) + 1
    return sorted(counts, key=lambda w: (-counts[w], w))[:k]
''', r'''def $fn(text, k, stop=()):
    skip = {s.lower() for s in stop}
    counts = {}
    for word in text.split():
        word = word.lower().strip(".,!?;:")
        if not word or word in skip:
            continue
        counts[word] = counts.get(word, 0) + 1
    return [(w, counts[w]) for w in sorted(counts, key=lambda w: (-counts[w], w))[:k]]
''']},
    {"key": "duration", "cases": duration_cases,
     "params": [{"fn": "parse_duration"}, {"fn": "to_seconds"}, {"fn": "duration_seconds"}, {"fn": "seconds_in"},
                {"fn": "read_duration"}],
     "asks": ["Write a Python function `$fn(text)` that converts a duration such as `2h15m`, `45s` or `1h30m10s` "
              "into a total number of seconds, as an int. The units `h`, `m` and `s` each appear at most once, in "
              "that order, and any of them may be left out.",
              "Also accept a `d` unit for days, which comes before the hours.",
              "Ignore any spaces in the text, and accept the units in uppercase too.",
              "Raise `ValueError` if the text is not a valid duration, for example an empty string, a number "
              "without a unit, an unknown unit, or units that are repeated or out of order.",
              "Allow an optional leading `-` that makes the duration negative, so `-1h30m` is -5400."],
     "refs": [r'''import re


def $fn(text):
    match = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", text)
    hours, minutes, seconds = (int(part) if part else 0 for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds
''', r'''import re


def $fn(text):
    match = re.fullmatch(r"(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", text)
    days, hours, minutes, seconds = (int(part) if part else 0 for part in match.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds
''', r'''import re


def $fn(text):
    text = text.replace(" ", "")
    match = re.fullmatch(r"(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", text, re.IGNORECASE)
    days, hours, minutes, seconds = (int(part) if part else 0 for part in match.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds
''', r'''import re


def $fn(text):
    text = text.replace(" ", "")
    match = re.fullmatch(r"(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", text, re.IGNORECASE)
    if match is None or all(part is None for part in match.groups()):
        raise ValueError(f"not a duration: {text!r}")
    days, hours, minutes, seconds = (int(part) if part else 0 for part in match.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds
''', r'''import re


def $fn(text):
    text = text.replace(" ", "")
    sign = 1
    if text.startswith("-"):
        sign, text = -1, text[1:]
    match = re.fullmatch(r"(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", text, re.IGNORECASE)
    if match is None or all(part is None for part in match.groups()):
        raise ValueError(f"not a duration: {text!r}")
    days, hours, minutes, seconds = (int(part) if part else 0 for part in match.groups())
    return sign * (days * 86400 + hours * 3600 + minutes * 60 + seconds)
''']},
    {"key": "rle", "cases": rle_cases,
     "params": [{"fn": "rle", "m": 9, "m2": 11}, {"fn": "run_length", "m": 5, "m2": 7},
                {"fn": "compress_runs", "m": 7, "m2": 9}, {"fn": "encode_runs", "m": 4, "m2": 6},
                {"fn": "squeeze", "m": 6, "m2": 8}],
     "asks": ["Write a Python function `$fn(s)` that run-length encodes a string: each run of a repeated character "
              "becomes the character followed by the run length, so `aaabcc` becomes `a3b1c2`.",
              "Leave out the count when a run has length 1, so `aaabcc` becomes `a3bc2`.",
              "Split runs longer than $m into pieces of $m followed by the rest, so a run of $m2 `x` characters "
              "becomes `x${m}x2`.",
              "Treat uppercase and lowercase letters as the same character, and write the lowercase letter: `aAab` "
              "becomes `a3b`.",
              "Raise `ValueError` if the string contains a digit, because the output would be ambiguous."],
     "refs": [r'''def $fn(s):
    out = []
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(s[i] + str(j - i))
        i = j
    return "".join(out)
''', r'''def $fn(s):
    out = []
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(s[i] + (str(j - i) if j - i > 1 else ""))
        i = j
    return "".join(out)
''', r'''def $fn(s):
    out = []
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        run = j - i
        while run > 0:
            piece = min(run, $m)
            out.append(s[i] + (str(piece) if piece > 1 else ""))
            run -= piece
        i = j
    return "".join(out)
''', r'''def $fn(s):
    s = s.lower()
    out = []
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        run = j - i
        while run > 0:
            piece = min(run, $m)
            out.append(s[i] + (str(piece) if piece > 1 else ""))
            run -= piece
        i = j
    return "".join(out)
''', r'''def $fn(s):
    if any(ch in "0123456789" for ch in s):
        raise ValueError("digits cannot be encoded")
    s = s.lower()
    out = []
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        run = j - i
        while run > 0:
            piece = min(run, $m)
            out.append(s[i] + (str(piece) if piece > 1 else ""))
            run -= piece
        i = j
    return "".join(out)
''']},
    {"key": "inventory", "cases": inventory_cases,
     "params": [{"fn": "tally"}, {"fn": "stock_levels"}, {"fn": "inventory_totals"}, {"fn": "count_items"},
                {"fn": "merge_counts"}],
     "asks": ["Write a Python function `$fn(entries)` where `entries` is a list of `(item, qty)` pairs; return a "
              "dict that maps each item to the sum of its quantities.",
              "Leave out any item whose total is zero or negative.",
              "Treat item names case-insensitively, using the lowercase names as the keys.",
              "Skip any entry whose `qty` is not an `int`, such as a string or a float.",
              "Return a list of `(item, total)` tuples sorted by total, largest first, with ties in alphabetical "
              "order, instead of a dict."],
     "refs": [r'''def $fn(entries):
    totals = {}
    for item, qty in entries:
        totals[item] = totals.get(item, 0) + qty
    return totals
''', r'''def $fn(entries):
    totals = {}
    for item, qty in entries:
        totals[item] = totals.get(item, 0) + qty
    return {item: total for item, total in totals.items() if total > 0}
''', r'''def $fn(entries):
    totals = {}
    for item, qty in entries:
        item = item.lower()
        totals[item] = totals.get(item, 0) + qty
    return {item: total for item, total in totals.items() if total > 0}
''', r'''def $fn(entries):
    totals = {}
    for item, qty in entries:
        if type(qty) is not int:
            continue
        item = item.lower()
        totals[item] = totals.get(item, 0) + qty
    return {item: total for item, total in totals.items() if total > 0}
''', r'''def $fn(entries):
    totals = {}
    for item, qty in entries:
        if type(qty) is not int:
            continue
        item = item.lower()
        totals[item] = totals.get(item, 0) + qty
    kept = [(item, total) for item, total in totals.items() if total > 0]
    return sorted(kept, key=lambda pair: (-pair[1], pair[0]))
''']},
    {"key": "account", "cases": bank_cases,
     "params": [{"cls": "Account"}, {"cls": "Wallet"}, {"cls": "Ledger"}, {"cls": "Purse"}, {"cls": "CashBox"}],
     "asks": ["Write a Python class `$cls` with a `balance` attribute that starts at 0, a `deposit(amount)` method "
              "that adds the amount to it, and a `withdraw(amount)` method that subtracts the amount from it.",
              "`withdraw` must raise `ValueError` and leave the balance unchanged if the amount is larger than the "
              "balance.",
              "Both methods must raise `ValueError`, leaving the balance unchanged, if the amount is zero or negative.",
              "Add a `history()` method that returns a list of `('deposit', amount)` and `('withdraw', amount)` "
              "tuples for the successful operations, oldest first.",
              "Add an `undo()` method that reverses the most recent successful operation that has not been undone "
              "yet and removes it from the history. It raises `IndexError` when there is nothing left to undo."],
     "refs": [r'''class $cls:
    def __init__(self):
        self.balance = 0

    def deposit(self, amount):
        self.balance += amount

    def withdraw(self, amount):
        self.balance -= amount
''', r'''class $cls:
    def __init__(self):
        self.balance = 0

    def deposit(self, amount):
        self.balance += amount

    def withdraw(self, amount):
        if amount > self.balance:
            raise ValueError("insufficient funds")
        self.balance -= amount
''', r'''class $cls:
    def __init__(self):
        self.balance = 0

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("amount must be positive")
        self.balance += amount

    def withdraw(self, amount):
        if amount <= 0:
            raise ValueError("amount must be positive")
        if amount > self.balance:
            raise ValueError("insufficient funds")
        self.balance -= amount
''', r'''class $cls:
    def __init__(self):
        self.balance = 0
        self._history = []

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("amount must be positive")
        self.balance += amount
        self._history.append(("deposit", amount))

    def withdraw(self, amount):
        if amount <= 0:
            raise ValueError("amount must be positive")
        if amount > self.balance:
            raise ValueError("insufficient funds")
        self.balance -= amount
        self._history.append(("withdraw", amount))

    def history(self):
        return list(self._history)
''', r'''class $cls:
    def __init__(self):
        self.balance = 0
        self._history = []

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("amount must be positive")
        self.balance += amount
        self._history.append(("deposit", amount))

    def withdraw(self, amount):
        if amount <= 0:
            raise ValueError("amount must be positive")
        if amount > self.balance:
            raise ValueError("insufficient funds")
        self.balance -= amount
        self._history.append(("withdraw", amount))

    def history(self):
        return list(self._history)

    def undo(self):
        if not self._history:
            raise IndexError("nothing to undo")
        kind, amount = self._history.pop()
        self.balance += -amount if kind == "deposit" else amount
''']},
    {"key": "roman", "cases": roman_cases,
     "params": [{"fn": "to_roman", "inv": "from_roman"}, {"fn": "int_to_roman", "inv": "roman_to_int"},
                {"fn": "roman", "inv": "parse_roman"}, {"fn": "as_roman", "inv": "read_roman"},
                {"fn": "encode_roman", "inv": "decode_roman"}],
     "asks": ["Write a Python function `$fn(n)` that converts an integer from 1 to 3999 into a Roman numeral string, "
              "using the subtractive forms such as `IV` and `CM`.",
              "Make `$fn` raise `ValueError` for any integer outside 1 to 3999.",
              "Add a function `$inv(s)` that converts a Roman numeral string back into an int, and keep `$fn` in "
              "the same code.",
              "Make `$inv` accept lowercase numerals and ignore leading and trailing whitespace.",
              "Make `$inv` raise `ValueError` for any text that is not a numeral in the standard form `$fn` "
              "produces, such as `IIII`, `IC` or an empty string."],
     "refs": [r'''PAIRS = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"),
         (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]


def $fn(n):
    out = []
    for value, letters in PAIRS:
        while n >= value:
            out.append(letters)
            n -= value
    return "".join(out)
''', r'''PAIRS = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"),
         (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]


def $fn(n):
    if not 1 <= n <= 3999:
        raise ValueError("out of range")
    out = []
    for value, letters in PAIRS:
        while n >= value:
            out.append(letters)
            n -= value
    return "".join(out)
''', r'''PAIRS = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"),
         (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def $fn(n):
    if not 1 <= n <= 3999:
        raise ValueError("out of range")
    out = []
    for value, letters in PAIRS:
        while n >= value:
            out.append(letters)
            n -= value
    return "".join(out)


def $inv(s):
    total = 0
    for i, ch in enumerate(s):
        value = VALUES[ch]
        if i + 1 < len(s) and VALUES[s[i + 1]] > value:
            total -= value
        else:
            total += value
    return total
''', r'''PAIRS = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"),
         (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def $fn(n):
    if not 1 <= n <= 3999:
        raise ValueError("out of range")
    out = []
    for value, letters in PAIRS:
        while n >= value:
            out.append(letters)
            n -= value
    return "".join(out)


def $inv(s):
    s = s.strip().upper()
    total = 0
    for i, ch in enumerate(s):
        value = VALUES[ch]
        if i + 1 < len(s) and VALUES[s[i + 1]] > value:
            total -= value
        else:
            total += value
    return total
''', r'''PAIRS = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"),
         (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def $fn(n):
    if not 1 <= n <= 3999:
        raise ValueError("out of range")
    out = []
    for value, letters in PAIRS:
        while n >= value:
            out.append(letters)
            n -= value
    return "".join(out)


def $inv(s):
    s = s.strip().upper()
    if not s or any(ch not in VALUES for ch in s):
        raise ValueError(f"not a numeral: {s!r}")
    total = 0
    for i, ch in enumerate(s):
        value = VALUES[ch]
        if i + 1 < len(s) and VALUES[s[i + 1]] > value:
            total -= value
        else:
            total += value
    if not 1 <= total <= 3999 or $fn(total) != s:
        raise ValueError(f"not a standard numeral: {s!r}")
    return total
''']},
]
PROJECT = {project["key"]: project for project in PROJECTS}

DRIVER = r'''
def _kind(exc):
    for known in (ValueError, IndexError, TypeError, KeyError, AttributeError):
        if isinstance(exc, known):
            return known.__name__
    return type(exc).__name__


def observe(ns, case):
    if "cls" in case:
        account = ns[case["cls"]]()
        trace = []
        for op in case["ops"]:
            try:
                if op[0] == "history":
                    trace.append(("ok", [tuple(entry) for entry in account.history()]))
                    continue
                getattr(account, op[0])(*op[1:])
                trace.append(("ok", account.balance))
            except Exception as exc:
                trace.append(("raises", _kind(exc), getattr(account, "balance", None)))
        return trace
    function = ns[case["fn"]]
    try:
        return ("ok", function(*case["args"], **case.get("kwargs", {})))
    except Exception as exc:
        return ("raises", _kind(exc))
'''

HARNESS = r'''
def main():
    namespace = {"__name__": "solution"}
    try:
        exec(compile(SOURCE, "solution.py", "exec"), namespace)
    except BaseException as exc:
        return {"passed": 0, "total": len(CASES), "load_error": f"{type(exc).__name__}: {exc}"[:300]}
    passed, failures = 0, []
    for case, expected in CASES:
        try:
            got = observe(namespace, case)
        except BaseException as exc:
            got = ("crash", type(exc).__name__, str(exc)[:100])
        if got == expected:
            passed += 1
        elif len(failures) < 3:
            failures.append({"case": repr(case)[:300], "expected": repr(expected)[:300], "got": repr(got)[:300]})
    return {"passed": passed, "total": len(CASES), "failures": failures}


result = main()
with open("result.json", "w", encoding="utf-8") as handle:
    json.dump(result, handle)
sys.exit(0 if result["passed"] == result["total"] else 1)
'''


def fill(template: str, params: dict) -> str:
    values = {}
    for key, value in params.items():
        values[key] = str(value)
        values[key + "_r"] = repr(value)
    return string.Template(template).substitute(values)


def code_conversation(project: dict, variant: int) -> dict:
    params = project["params"][variant]
    steps = []
    for number, ask in enumerate(project["asks"]):
        tail = ("Reply with the code in a single ```python block." if number == 0
                else "Reply with the complete updated code in a single ```python block.")
        steps.append({"user": f"{fill(ask, params)} {tail}", "recall": False})
    return {"id": f"code-{project['key']}-{variant}", "family": "code", "project": project["key"], "variant": variant,
            "turns": steps}


def conversations(family: str, count: int = CONVERSATIONS, turns: int = TURNS) -> list:
    if family == "chain":
        return [chain_conversation(index, turns) for index in range(count)]
    if family == "recall":
        return [recall_conversation(index) for index in range(count)]
    if family == "code":
        ordered = [code_conversation(project, variant) for variant in range(5) for project in PROJECTS]
        if count > len(ordered):
            raise ValueError(f"the code family has {len(ordered)} conversations, not {count}")
        return ordered[:count]
    raise ValueError(f"unknown family {family!r}")


def reference(conversation: dict, turn: int) -> str:
    """The reference solution after `turn` requirements (1-based)."""
    project = PROJECT[conversation["project"]]
    return fill(project["refs"][turn - 1], project["params"][conversation["variant"]])


def observer():
    namespace = {}
    exec(compile(DRIVER, "driver.py", "exec"), namespace)
    return namespace["observe"]


def code_tests(conversation: dict, turn: int) -> list:
    """(case, expected) pairs for the requirements up to `turn`, the expectations from the reference."""
    project = PROJECT[conversation["project"]]
    params = project["params"][conversation["variant"]]
    rng = random.Random(f"{SEED}-code-{conversation['project']}-{conversation['variant']}")
    cases = project["cases"](params, turn, rng)
    namespace = {"__name__": "reference"}
    exec(compile(reference(conversation, turn), "reference.py", "exec"), namespace)
    observe = observer()
    return [(case, observe(namespace, case)) for case in cases]


def test_program(source: str, tests: list) -> str:
    return ("import json\nimport sys\n\nSOURCE = " + repr(source) + "\nCASES = " + repr(tests) + "\n" + DRIVER
            + HARNESS)


FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+.-]*)[^\n]*\n(.*?)```", re.S)


def extract_code(content: str) -> str | None:
    """Every fenced Python block in the reply, in order; None when there is none."""
    blocks = [body for language, body in FENCE.findall(content) if language.lower() in ("python", "py", "python3", "")]
    return "\n\n".join(blocks) if blocks else None


VALUE = re.compile(r"value\s*\**\s*[:=]\s*\**\s*(-?\d+)", re.I)
INTEGER = re.compile(r"-?\d+")


def parse_value(content: str) -> tuple[int | None, bool]:
    """(the answer, whether it came from a 'Value: N' line) from a reply's visible content."""
    found = VALUE.findall(content)
    if found:
        return int(found[-1]), True
    numbers = INTEGER.findall(content)
    return (int(numbers[-1]), False) if numbers else (None, False)


# Requests and histories.

def request_body(model: str, messages: list, effort: str, max_tokens: int) -> dict:
    return {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": False,
            "reasoning_effort": effort}


def assistant_message(arm: str, reply: dict) -> dict:
    """How an arm sends an earlier reply back; replay matches the fx workload's clients."""
    if arm == "replay" and reply["reasoning"]:
        return {"role": "assistant", "reasoning_content": reply["reasoning"], "content": reply["content"]}
    return {"role": "assistant", "content": reply["content"]}


def converse(client, model: str, conversation: dict, effort: str, arm: str, max_tokens: int, upto: int | None = None) -> list:
    """Run one arm of a conversation turn by turn from its own replies and return them."""
    messages = [{"role": "system", "content": conversation["system"]}] if conversation.get("system") else []
    replies = []
    for turn in conversation["turns"][:upto]:
        messages.append({"role": "user", "content": turn["user"]})
        reply = client.complete(request_body(model, list(messages), effort, max_tokens))
        replies.append(reply)
        messages.append(assistant_message(arm, reply))
    return replies


def body_key(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def parse_reply(document: dict) -> dict:
    choice = (document.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = document.get("usage") or {}
    runtime = document.get("tensorfold") or {}
    return {"content": message.get("content") or "", "reasoning": message.get("reasoning_content") or "",
            "finish_reason": choice.get("finish_reason"), "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "token_sha256": runtime.get("token_sha256"), "drafts": runtime.get("drafts"),
            "reasoning_effort": runtime.get("reasoning_effort"), "decode_seconds": runtime.get("seconds"),
            "prefill_seconds": runtime.get("prefill_seconds")}


class Missing(LookupError):
    """An offline replay met a request that no run has answered."""


class Client:
    """Chat completions memoised by request body; identical concurrent bodies are sent once."""

    def __init__(self, path: Path, base: str | None = None, timeout: float = REQUEST_TIMEOUT, log=sys.stderr):
        self.path, self.base, self.timeout, self.log = path, base, timeout, log
        self.memo, self.inflight = {}, {}
        self.lock = threading.Lock()
        self.sent = self.tokens = 0
        self.started = time.monotonic()
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        record = json.loads(line)
                        self.memo[record["key"]] = record["reply"]

    def complete(self, body: dict) -> dict:
        key = body_key(body)
        with self.lock:
            if key in self.memo:
                return self.memo[key]
            if self.base is None:
                raise Missing(f"no recorded reply for a {body['reasoning_effort']} request of "
                              f"{len(body['messages'])} messages")
            waiting = self.inflight.get(key)
            if waiting is None:
                waiting = self.inflight[key] = concurrent.futures.Future()
                owner = True
            else:
                owner = False
        if not owner:
            return waiting.result()
        try:
            started = time.monotonic()
            reply = self.send(body)
            reply["seconds"] = round(time.monotonic() - started, 3)
            with self.lock:
                self.memo[key] = reply
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"key": key, "body": body, "reply": reply}, ensure_ascii=False) + "\n")
                self.sent += 1
                self.tokens += reply["completion_tokens"] or 0
                rate = self.tokens / max(1e-9, time.monotonic() - self.started)
                print(f"reply {self.sent}: {body['reasoning_effort']} {len(body['messages'])} messages "
                      f"{reply['completion_tokens']} tokens ({reply['reasoning_tokens']} reasoning) "
                      f"{reply['finish_reason']} {reply['seconds']}s; {self.tokens} tokens at {rate:.1f}/s",
                      file=self.log, flush=True)
            waiting.set_result(reply)
            return reply
        except BaseException as exc:
            waiting.set_exception(exc)
            raise
        finally:
            with self.lock:
                self.inflight.pop(key, None)

    def send(self, body: dict, attempts: int = 3) -> dict:
        """One reply; a prompt the context window refuses is a failed turn, not a failed run."""
        request = urllib.request.Request(self.base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return parse_reply(json.loads(response.read()))
            except urllib.error.HTTPError as exc:
                with exc:
                    text = exc.read().decode("utf-8", "replace")
                if exc.code == 400 and "context_length_exceeded" in text:
                    return refused(text)
                if exc.code < 500 or attempt + 1 == attempts:
                    raise RuntimeError(f"HTTP {exc.code}: {text[:500]}") from exc
            except (urllib.error.URLError, ConnectionError) as exc:
                if attempt + 1 == attempts:
                    raise
            time.sleep(5)
        raise AssertionError("unreachable")


def refused(text: str) -> dict:
    return {"content": "", "reasoning": "", "finish_reason": "context_length", "prompt_tokens": None,
            "completion_tokens": 0, "reasoning_tokens": 0, "cached_tokens": None, "token_sha256": None,
            "drafts": None, "reasoning_effort": None, "decode_seconds": None, "prefill_seconds": None,
            "error": text[:500]}


def run_effort(client: Client, model: str, selected: list, effort: str, max_tokens: int, workers: int) -> None:
    """Every shared first turn, then both arms of every conversation."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        firsts = [pool.submit(converse, client, model, conversation, effort, ARMS[0], max_tokens, 1)
                  for conversation in selected]
        for future in firsts:
            future.result()
        arms = [pool.submit(converse, client, model, conversation, effort, arm, max_tokens)
                for conversation in selected for arm in ARMS]
        for future in concurrent.futures.as_completed(arms):
            future.result()


def recheck(client: Client, sample: int, seed: str = SEED) -> dict:
    """Resend a sample of answered bodies past the memo and count byte-identical replies."""
    with client.path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    chosen = random.Random(seed).sample(records, min(sample, len(records)))
    fields = ("content", "reasoning", "finish_reason", "completion_tokens")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        fresh = list(pool.map(lambda record: client.send(record["body"]), chosen))
    same = [all(record["reply"][field] == reply[field] for field in fields) for record, reply in zip(chosen, fresh)]
    rows = [{"key": record["key"], "effort": record["body"]["reasoning_effort"], "same": flag,
             "tokens": [record["reply"]["completion_tokens"], reply["completion_tokens"]]}
            for record, flag, reply in zip(chosen, same, fresh)]
    return {"sent": len(chosen), "identical": sum(same), "rows": rows}


# Serving.

class Server:
    """A tensorfold-native server this process starts, waits for and stops."""

    def __init__(self, binary: str, model_dir: str, out: Path, parallel: int, context: int, cache_gib: float):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        self.base = f"http://127.0.0.1:{port}/v1"
        self.log_path = out / f"server-{time.strftime('%Y%m%dT%H%M%S')}.log"
        env = dict(os.environ)
        env.setdefault("INKLING_LOAD_LIMIT_GB", "26")
        env.setdefault("INKLING_COPY_DRAFTS", "1")
        env["TENSORFOLD_NO_UPDATE_CHECK"] = "1"
        self.argv = [binary, "serve", model_dir, "--host", "127.0.0.1", "--port", str(port), "--parallel", str(parallel),
                     "--prompt-cache-gib", str(cache_gib), "--no-update-check", "--context", str(context),
                     "--temperature", "0"]
        self.log = self.log_path.open("wb")
        self.process = subprocess.Popen(self.argv, stdin=subprocess.DEVNULL, stdout=self.log, stderr=subprocess.STDOUT,
                                        env=env)

    def wait(self, timeout: float = 900) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.log.flush()
                tail = self.log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:]
                raise RuntimeError(f"the server exited with {self.process.returncode}: {' / '.join(tail)}")
            try:
                with urllib.request.urlopen(self.base + "/models", timeout=10) as response:
                    return json.loads(response.read())["data"][0]["id"]
            except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError):
                time.sleep(2)
        raise TimeoutError(f"the server was not ready after {timeout} s; see {self.log_path}")

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(60)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()


def budgets(spec: str, efforts: list) -> dict:
    """max_tokens per effort from `N` or `effort=N,...`."""
    if "=" not in spec:
        return {effort: int(spec) for effort in efforts}
    table = {}
    for part in spec.split(","):
        effort, _, value = part.partition("=")
        table[effort.strip()] = int(value)
    missing = [effort for effort in efforts if effort not in table]
    if missing:
        raise SystemExit(f"--max-tokens names no budget for {', '.join(missing)}")
    return table


def selection(args) -> tuple[list, list, dict]:
    families = [name.strip() for name in args.families.split(",") if name.strip()]
    efforts = [name.strip() for name in args.efforts.split(",") if name.strip()]
    for name in families:
        if name not in FAMILIES:
            raise SystemExit(f"unknown family {name!r}; choose from {', '.join(FAMILIES)}")
    for name in efforts:
        if name not in EFFORTS:
            raise SystemExit(f"unknown effort {name!r}; choose from {', '.join(EFFORTS)}")
    selected = [conversation for family in families for conversation in conversations(family, args.conversations, args.turns)]
    return selected, efforts, budgets(args.max_tokens, efforts)


def run(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    selected, efforts, limits = selection(args)
    server = None

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        if args.serve:
            server = Server(args.serve, args.model_dir, out, args.parallel, args.context, args.prompt_cache_gib)
            print(f"server: {' '.join(server.argv)}", file=sys.stderr, flush=True)
            served = server.wait()
            base = server.base
        else:
            base = args.base
            with urllib.request.urlopen(base.rstrip("/") + "/models", timeout=30) as response:
                served = json.loads(response.read())["data"][0]["id"]
        model = args.model or served
        client = Client(out / "responses.jsonl", base)
        with (out / "runs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": "start", "model": model, "served": served, "families": args.families,
                                     "efforts": efforts, "max_tokens": limits, "conversations": args.conversations,
                                     "turns": args.turns, "argv": server.argv if server else None,
                                     "time": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
        for effort in efforts:
            started = time.monotonic()
            run_effort(client, model, selected, effort, limits[effort], args.parallel)
            print(f"effort {effort}: done in {time.monotonic() - started:.0f} s", file=sys.stderr, flush=True)
        if args.recheck:
            result = recheck(client, args.recheck)
            with (out / "recheck.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result) + "\n")
            print(f"recheck: {result['identical']} of {result['sent']} replies identical", file=sys.stderr, flush=True)
        with (out / "runs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": "finish", "model": model, "sent": client.sent,
                                     "completion_tokens": client.tokens,
                                     "seconds": round(time.monotonic() - client.started, 1),
                                     "time": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
    finally:
        if server:
            server.stop()
    return 0


# Scoring.

class CodeChecker:
    """Runs each distinct reply program once in the sandbox and remembers the verdict."""

    def __init__(self, out: Path, wall_s: float = 20, cpu_s: int = 10):
        import sandbox
        self.sandbox, self.wall_s, self.cpu_s = sandbox, wall_s, cpu_s
        self.root = out / "sandbox"
        self.path = out / "code-results.jsonl"
        self.memo, self.lock = {}, threading.Lock()
        if self.path.exists():
            with self.path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        record = json.loads(line)
                        self.memo[record["key"]] = record["result"]
        self.tests = {}

    def suite(self, conversation: dict, turn: int) -> list:
        key = (conversation["id"], turn)
        if key not in self.tests:
            self.tests[key] = code_tests(conversation, turn)
        return self.tests[key]

    def check(self, conversation: dict, turn: int, content: str) -> dict:
        source = extract_code(content)
        if source is None:
            return {"passed": False, "detail": "no python block"}
        program = test_program(source, self.suite(conversation, turn))
        key = hashlib.sha256(program.encode()).hexdigest()
        with self.lock:
            if key in self.memo:
                return self.memo[key]
        work = self.root / key[:20]
        work.mkdir(parents=True, exist_ok=True)
        (work / "result.json").unlink(missing_ok=True)
        ran = self.sandbox.run(program, work, wall_s=self.wall_s, cpu_s=self.cpu_s)
        try:
            detail = json.loads((work / "result.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            detail = {}
        counted = (isinstance(detail, dict) and isinstance(detail.get("total"), int)
                   and isinstance(detail.get("passed"), int))
        if not counted:
            detail = {"ended": ran["ended"], "exit": ran["exit"], "stderr": ran["stderr"][-300:]}
        result = {"passed": bool(ran["passed"] and counted and detail["passed"] == detail["total"]), "detail": detail}
        with self.lock:
            self.memo[key] = result
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": key, "result": result}) + "\n")
        return result


def score_turns(conversation: dict, replies: list, checker: CodeChecker | None = None) -> list:
    """One row per turn: correctness, the parsed answer and the reply's token counts."""
    rows, previous = [], conversation.get("start")
    for number, (turn, reply) in enumerate(zip(conversation["turns"], replies), start=1):
        row = {"turn": number, "recall": turn["recall"], "finish_reason": reply["finish_reason"],
               "prompt_tokens": reply["prompt_tokens"], "completion_tokens": reply["completion_tokens"],
               "reasoning_tokens": reply["reasoning_tokens"],
               "answer_tokens": (reply["completion_tokens"] or 0) - (reply["reasoning_tokens"] or 0),
               "cached_tokens": reply["cached_tokens"], "reasoning_chars": len(reply["reasoning"])}
        if conversation["family"] == "code":
            result = checker.check(conversation, number, reply["content"])
            row.update(correct=result["passed"], detail=result["detail"])
        else:
            value, formatted = parse_value(reply["content"])
            row.update(answer=value, formatted=formatted, expect=turn["expect"], correct=value == turn["expect"])
            if turn["ops"]:
                if previous is None:
                    row["step"] = None
                else:
                    stepped = previous
                    for op in turn["ops"]:
                        stepped = transform(op, stepped)
                    row["step"] = value == stepped
                previous = value
        rows.append(row)
    return rows


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def bootstrap(differences: list, draws: int = BOOTSTRAP_DRAWS, seed: str = SEED) -> tuple[float, float, float]:
    """Mean of per-conversation differences and its 95% percentile interval."""
    n = len(differences)
    if n == 0:
        return math.nan, math.nan, math.nan
    rng = random.Random(seed)
    means = sorted(sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    return sum(differences) / n, means[int(0.025 * draws)], means[int(0.975 * draws) - 1]


def _mean(values: list) -> float | None:
    values = [value for value in values if value is not None]
    return round(sum(values) / len(values), 3) if values else None


def summarize(rows: dict, conversations_by_id: dict) -> dict:
    """Per family and effort: per-turn accuracy, paired tests and token counts. rows[(id, effort, arm)] -> turns."""
    summary = {}
    keys = sorted({(conversations_by_id[cid]["family"], effort) for cid, effort, _ in rows},
                  key=lambda key: (FAMILIES.index(key[0]), EFFORTS.index(key[1])))
    for family, effort in keys:
        ids = sorted(cid for cid, eff, arm in rows if eff == effort and arm == ARMS[0]
                     and conversations_by_id[cid]["family"] == family and (cid, eff, ARMS[1]) in rows)
        turns = max(len(rows[(cid, effort, ARMS[0])]) for cid in ids)
        per_turn = []
        for number in range(1, turns + 1):
            pairs = [(rows[(cid, effort, "replay")][number - 1], rows[(cid, effort, "stripped")][number - 1])
                     for cid in ids if len(rows[(cid, effort, "replay")]) >= number]
            b = sum(1 for a, s in pairs if a["correct"] and not s["correct"])
            c = sum(1 for a, s in pairs if s["correct"] and not a["correct"])
            entry = {"turn": number, "n": len(pairs), "recall": pairs[0][0]["recall"],
                     "replay": round(sum(a["correct"] for a, _ in pairs) / len(pairs), 3),
                     "stripped": round(sum(s["correct"] for _, s in pairs) / len(pairs), 3),
                     "replay_only": b, "stripped_only": c, "mcnemar_p": round(mcnemar(b, c), 4)}
            for arm, index in (("replay", 0), ("stripped", 1)):
                chosen = [pair[index] for pair in pairs]
                entry[f"{arm}_reasoning_tokens"] = _mean([row["reasoning_tokens"] for row in chosen])
                entry[f"{arm}_answer_tokens"] = _mean([row["answer_tokens"] for row in chosen])
                entry[f"{arm}_prompt_tokens"] = _mean([row["prompt_tokens"] for row in chosen])
                entry[f"{arm}_no_reasoning"] = sum(1 for row in chosen if not row["reasoning_tokens"])
                entry[f"{arm}_truncated"] = sum(1 for row in chosen if row["finish_reason"] == "length")
                entry[f"{arm}_context_refused"] = sum(1 for row in chosen if row["finish_reason"] == "context_length")
                if any("step" in row for row in chosen):
                    judged = [row["step"] for row in chosen if row.get("step") is not None]
                    entry[f"{arm}_step"] = round(sum(judged) / len(judged), 3) if judged else None
                    entry[f"{arm}_step_n"] = len(judged)
            per_turn.append(entry)

        def contrast(select, field="correct") -> dict:
            differences, better, worse = [], 0, 0
            for cid in ids:
                a = [row.get(field) for row in rows[(cid, effort, "replay")] if select(row)]
                s = [row.get(field) for row in rows[(cid, effort, "stripped")] if select(row)]
                kept = [(x, y) for x, y in zip(a, s) if x is not None and y is not None]
                if not kept:
                    continue
                difference = sum(x - y for x, y in kept) / len(kept)
                differences.append(difference)
                better += difference > 0
                worse += difference < 0
            mean, low, high = bootstrap(differences)
            return {"conversations": len(differences), "diff": round(mean, 3), "ci95": [round(low, 3), round(high, 3)],
                    "replay_better": better, "stripped_better": worse, "sign_p": round(mcnemar(better, worse), 4)}

        later = lambda row: row["turn"] >= 2  # noqa: E731
        block = {"conversations": len(ids), "turns": turns, "per_turn": per_turn,
                 "later_turns": {"replay": _mean([row["correct"] for cid in ids for row in rows[(cid, effort, "replay")] if later(row)]),
                                 "stripped": _mean([row["correct"] for cid in ids for row in rows[(cid, effort, "stripped")] if later(row)]),
                                 **contrast(later)}}
        if any("step" in row for cid in ids for row in rows[(cid, effort, "replay")] if later(row)):
            stepped = lambda row: later(row) and "step" in row  # noqa: E731
            block["later_steps"] = {
                "replay": _mean([row["step"] for cid in ids for row in rows[(cid, effort, "replay")] if stepped(row)]),
                "stripped": _mean([row["step"] for cid in ids for row in rows[(cid, effort, "stripped")] if stepped(row)]),
                **contrast(stepped, "step")}
        rendered = []
        for cid in ids:
            first, replay, stripped = (rows[(cid, effort, "replay")][0], rows[(cid, effort, "replay")][1:2],
                                       rows[(cid, effort, "stripped")][1:2])
            if replay and stripped and first["reasoning_tokens"] and replay[0]["prompt_tokens"] and stripped[0]["prompt_tokens"]:
                rendered.append(replay[0]["prompt_tokens"] - stripped[0]["prompt_tokens"] - first["reasoning_tokens"])
        block["replayed_reasoning_render"] = {
            "conversations": len(rendered), "turn1_without_reasoning": len(ids) - len(rendered),
            "extra_prompt_tokens_minus_reasoning_tokens": [min(rendered), max(rendered)] if rendered else None}
        if family == "recall":
            block["recall_turns"] = {
                "replay": _mean([row["correct"] for cid in ids for row in rows[(cid, effort, "replay")] if row["recall"]]),
                "stripped": _mean([row["correct"] for cid in ids for row in rows[(cid, effort, "stripped")] if row["recall"]]),
                **contrast(lambda row: row["recall"])}
        if family == "code":
            projects = sorted({conversations_by_id[cid]["project"] for cid in ids})
            by_project = {project: [cid for cid in ids if conversations_by_id[cid]["project"] == project]
                          for project in projects}
            cluster = []
            for project in projects:
                members = by_project[project]
                a = [row["correct"] for cid in members for row in rows[(cid, effort, "replay")] if later(row)]
                s = [row["correct"] for cid in members for row in rows[(cid, effort, "stripped")] if later(row)]
                cluster.append((sum(a) - sum(s)) / len(a))
            mean, low, high = bootstrap(cluster)
            block["later_turns"]["project_cluster_ci95"] = [round(low, 3), round(high, 3)]
        summary[f"{family}/{effort}"] = block
    return summary


def examples(rows: dict, replies: dict, conversations_by_id: dict, family: str, effort: str, limit: int = 3) -> list:
    """Paired turns where exactly one arm is right, both directions, earliest first."""
    found = []
    for (cid, eff, arm), turns in sorted(rows.items()):
        if arm != "replay" or eff != effort or conversations_by_id[cid]["family"] != family:
            continue
        other = rows[(cid, eff, "stripped")]
        for a, s in zip(turns, other):
            if a["correct"] != s["correct"]:
                number = a["turn"]
                ra, rs = replies[(cid, eff, "replay")][number - 1], replies[(cid, eff, "stripped")][number - 1]
                found.append({"id": cid, "turn": number, "winner": "replay" if a["correct"] else "stripped",
                              "user": conversations_by_id[cid]["turns"][number - 1]["user"][:240],
                              "expect": a.get("expect"),
                              "replay": {"answer": a.get("answer"), "content": ra["content"][-300:],
                                         "reasoning": ra["reasoning"][:400], "reasoning_tokens": ra["reasoning_tokens"],
                                         "detail": a.get("detail")},
                              "stripped": {"answer": s.get("answer"), "content": rs["content"][-300:],
                                           "reasoning": rs["reasoning"][:400], "reasoning_tokens": rs["reasoning_tokens"],
                                           "detail": s.get("detail")}})
                break
    chosen = [item for item in found if item["winner"] == "replay"][:limit]
    chosen += [item for item in found if item["winner"] == "stripped"][:limit]
    return chosen


def report(summary: dict) -> str:
    lines = []
    for name, block in summary.items():
        lines += [f"### {name} ({block['conversations']} conversations)", "",
                  "| Turn | Replay | Stripped | Replay only | Stripped only | McNemar p | Reasoning tok R / S "
                  "| Answer tok R / S | Prompt tok R / S | Truncated R / S |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for entry in block["per_turn"]:
            step = ""
            if "replay_step" in entry:
                step = f" (step {entry['replay_step']} / {entry['stripped_step']})"
            label = f"{entry['turn']}{' recall' if entry['recall'] else ''}"
            lines.append(f"| {label} | {entry['replay']} | {entry['stripped']}{step} | {entry['replay_only']} "
                         f"| {entry['stripped_only']} | {entry['mcnemar_p']} "
                         f"| {entry['replay_reasoning_tokens']} / {entry['stripped_reasoning_tokens']} "
                         f"| {entry['replay_answer_tokens']} / {entry['stripped_answer_tokens']} "
                         f"| {entry['replay_prompt_tokens']} / {entry['stripped_prompt_tokens']} "
                         f"| {entry['replay_truncated']} / {entry['stripped_truncated']} |")
        for label in ("later_turns", "later_steps", "recall_turns"):
            if label in block:
                part = block[label]
                extra = f", project-cluster CI {part['project_cluster_ci95']}" if "project_cluster_ci95" in part else ""
                lines.append(f"\n{label.replace('_', ' ')}: replay {part['replay']}, stripped {part['stripped']}, "
                             f"diff {part['diff']} (95% CI {part['ci95']}{extra}); conversations replay better "
                             f"{part['replay_better']}, stripped better {part['stripped_better']}, sign p {part['sign_p']}")
        refused_turns = [(entry["turn"], entry["replay_context_refused"], entry["stripped_context_refused"])
                         for entry in block["per_turn"] if entry["replay_context_refused"] or entry["stripped_context_refused"]]
        if refused_turns:
            lines.append(f"\ncontext-window refusals (turn, replay, stripped): {refused_turns}")
        render = block["replayed_reasoning_render"]
        lines.append(f"\nturn-2 replay prompt minus stripped prompt minus turn-1 reasoning tokens: "
                     f"{render['extra_prompt_tokens_minus_reasoning_tokens']} over {render['conversations']} "
                     f"conversations; turn 1 had no reasoning in {render['turn1_without_reasoning']}")
        lines.append("")
    return "\n".join(lines)


def score(args) -> int:
    out = Path(args.out)
    selected, efforts, limits = selection(args)
    runs = [json.loads(line) for line in (out / "runs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    model = args.model or [run["model"] for run in runs if run.get("event") == "start"][-1]
    client = Client(out / "responses.jsonl")
    checker = CodeChecker(out) if any(c["family"] == "code" for c in selected) else None
    by_id = {conversation["id"]: conversation for conversation in selected}
    replies, missing = {}, []
    for effort in efforts:
        for conversation in selected:
            for arm in ARMS:
                try:
                    replies[(conversation["id"], effort, arm)] = converse(client, model, conversation, effort, arm,
                                                                          limits[effort])
                except Missing:
                    missing.append((conversation["id"], effort, arm))
    if missing:
        print(f"score: {len(missing)} conversation arms have unanswered turns, first {missing[0]}", file=sys.stderr)
        if not args.partial:
            return 1
    complete = {key for key in replies if all((key[0], key[1], arm) in replies for arm in ARMS)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.sandbox_workers) as pool:
        futures = {key: pool.submit(score_turns, by_id[key[0]], replies[key], checker) for key in sorted(complete)}
        rows = {key: future.result() for key, future in futures.items()}
    summary = summarize(rows, by_id)
    samples = {name: examples(rows, replies, by_id, *name.split("/")) for name in summary}
    document = {"model": model, "max_tokens": limits, "summary": summary, "examples": samples}
    (out / "summary.json").write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    with (out / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for (cid, effort, arm), turns in sorted(rows.items()):
            for row in turns:
                handle.write(json.dumps({"id": cid, "effort": effort, "arm": arm, **row}, ensure_ascii=False) + "\n")
    text = report(summary)
    (out / "report.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "score"):
        sub = commands.add_parser(name)
        sub.add_argument("--out", required=True, help="directory for replies, logs and results")
        sub.add_argument("--families", default=",".join(FAMILIES))
        sub.add_argument("--efforts", default="minimal,high")
        sub.add_argument("--conversations", type=int, default=CONVERSATIONS, help="per family")
        sub.add_argument("--turns", type=int, default=TURNS, help="chain turns; recall and code have five")
        sub.add_argument("--max-tokens", default="2048", help="N, or effort=N pairs such as minimal=2048,high=4096")
        sub.add_argument("--model", help="model id for the requests; default: the served id")
    run_parser = commands.choices["run"]
    target = run_parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--base", help="an OpenAI-compatible base URL such as http://127.0.0.1:8080/v1")
    target.add_argument("--serve", help="tensorfold-native binary to start for this run")
    run_parser.add_argument("--model-dir", help="checkpoint for --serve")
    run_parser.add_argument("--parallel", type=int, default=4, help="server sessions and client workers")
    run_parser.add_argument("--context", type=int, default=21504,
                            help="tokens per session; four sessions fit at most 21982 under INKLING_LOAD_LIMIT_GB=26")
    run_parser.add_argument("--prompt-cache-gib", type=float, default=2)
    run_parser.add_argument("--recheck", type=int, default=0, help="resend this many answered bodies at the end")
    score_parser = commands.choices["score"]
    score_parser.add_argument("--partial", action="store_true", help="score the complete pairs when some are missing")
    score_parser.add_argument("--sandbox-workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.serve and not args.model_dir:
            parser.error("--serve needs --model-dir")
        return run(args)
    return score(args)


if __name__ == "__main__":
    sys.exit(main())
