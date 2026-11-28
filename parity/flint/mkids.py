import json, sys
from tokenizers import Tokenizer
t = Tokenizer.from_file("/Users/josh/models/flint-int4@3b93f9053b53/tokenizer.json")
base = "Explain why a hash table's load factor matters, then show a two-line Python example with a dict and a set. " * 40
ids = t.encode(base, add_special_tokens=False).ids
for n in (64, 256):
    json.dump(ids[:n], open(f"probe_{n}.json","w"))
    print(n, len(ids[:n]))
