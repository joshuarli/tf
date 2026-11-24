"""qualcmp.py BASE.json CAND.json: per-case pass and byte equality of content, reasoning and tool calls between two quality runs."""
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
ca = {c["id"]: c for c in a["cases"]}
cb = {c["id"]: c for c in b["cases"]}
same = 0
print("| case | base | candidate | reply bytes |")
print("| --- | --- | --- | --- |")
for cid in ca:
    x, y = ca[cid], cb.get(cid)
    if y is None:
        print(f"| {cid} | {x['passed']} | missing | - |"); continue
    eq = all(x.get(k) == y.get(k) for k in ("content", "reasoning", "tool_calls", "finish_reason"))
    same += eq
    print(f"| {cid} | {'pass' if x['passed'] else 'fail'} | {'pass' if y['passed'] else 'fail'} | {'equal' if eq else 'differ'} |")
print(f"\n{same}/{len(ca)} cases byte-equal; base {a['totals']}, candidate {b['totals']}")
