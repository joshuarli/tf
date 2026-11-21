from intervals import Interval, merge, total_covered

assert merge([Interval(1, 3), Interval(2, 6)]) == [Interval(1, 6)]
assert merge([Interval(1, 2), Interval(2, 3)]) == [Interval(1, 3)]
assert total_covered([Interval(0, 4), Interval(2, 6), Interval(10, 12)]) == 8
print("ok")
