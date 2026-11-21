import rates

assert rates.normalize("Hello,   World!") == "hello world"
assert rates.word_count("one two  three") == 3
assert rates.term_frequency("a b a")["a"] == 2
assert rates.top_terms("b b a c c c", 2) == ["c", "b"]
print("ok")
