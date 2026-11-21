"""Rates helpers for a tiny text classifier."""

import re

WORD = re.compile(r"[a-z']+")


def normalize(text):
    return " ".join(WORD.findall(text.lower()))


def word_count(text):
    return len(normalize(text).split())


def term_frequency(text):
    counts = {}
    for word in normalize(text).split():
        counts[word] = counts.get(word, 0) + 1
    return counts


def top_terms(text, k):
    freq = term_frequency(text)
    ranked = sorted(freq.items(), key=lambda pair: (-pair[1], pair[0]))
    return [word for word, _ in ranked[:k]]
