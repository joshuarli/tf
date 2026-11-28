import json, sys
from tokenizers import Tokenizer
t = Tokenizer.from_file("/Users/josh/models/flint-int4@3b93f9053b53/tokenizer.json")
ids = t.encode("Explain why a hash table's load factor matters, then show a two-line Python example.", add_special_tokens=False).ids
print(len(ids))
json.dump(ids, open("probe_ids.json","w"))
