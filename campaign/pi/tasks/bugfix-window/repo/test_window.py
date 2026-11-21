import window

assert window.rolling_max([1, 3, 2], 1) == [1, 3, 2]
assert window.rolling_max([1, 3, 2], 2) == [3, 3]
assert window.rolling_mean([2, 4], 2) == [3.0]
print("ok")
