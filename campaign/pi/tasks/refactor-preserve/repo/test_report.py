import report

cases = [
    ([], {"count": 0, "total": 0, "mean": 0}),
    ([50], {"count": 1, "total": 50, "mean": 50}),
    ([None, -5, 150, 60], {"count": 4, "total": 160, "mean": 40}),
]
for scores, expected in cases:
    assert report.summarize(scores) == expected, (scores, report.summarize(scores))

assert report.grade_bucket(95) == "A"
assert report.grade_bucket(85) == "B"
assert report.grade_bucket(70) == "C"
assert report.grade_bucket(69.9) == "F"
print("ok")
