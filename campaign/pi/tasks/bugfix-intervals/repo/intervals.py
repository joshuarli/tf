"""Interval merging for booking calendars."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    start: int
    end: int


def merge(intervals):
    ordered = sorted(intervals, key=lambda span: span.start)
    merged = []
    for span in ordered:
        if merged and span.start <= merged[-1].end:
            last = merged[-1]
            merged[-1] = Interval(last.start, max(last.end, span.end))
        else:
            merged.append(span)
    return merged


def total_covered(intervals):
    return sum(span.end - span.start for span in sorted(intervals, key=lambda span: span.start))
