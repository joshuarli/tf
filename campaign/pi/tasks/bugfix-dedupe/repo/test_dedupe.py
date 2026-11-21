import dedupe

assert dedupe.dedupe([1, 2, 1, 3, 2, 1]) == [1, 2, 3]
assert dedupe.dedupe([]) == []
assert dedupe.dedupe(["a", "b", "a"]) == ["a", "b"]
print("ok")
