"""chunks.py LENGTH...: the served host's prefill chunks for a fresh prompt (prefix reuse on, fork window 256, interval 128, chunk 1024)"""
import sys
def chunks(n, window=256, interval=128, chunk=1024, reuse=True):
    at, out = 0, []
    first = ((max(n - window, 0)) + interval - 1) // interval * interval
    nxt = max(interval, max(first, ((at + interval - 1) // interval) * interval))
    while at < n:
        if reuse and nxt == at and nxt < n and nxt > 0:
            nxt += interval
        remaining = n - 1 - at if reuse and at < n - 1 else n - at
        take = min(chunk, min(remaining, nxt - at if reuse and nxt > at else remaining))
        out.append(take)
        at += take
    return out
for a in sys.argv[1:]:
    n = int(a)
    print(n, chunks(n))
