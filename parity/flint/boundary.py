"""Rule out mask/ring off-by-one: error by position, and at the 513-token ring and 16-row window boundaries."""
import json, sys
from pathlib import Path
import numpy as np
V=150144
def f32(b): return (b.astype(np.uint32)<<16).view(np.float32)
def load(p,n): return np.fromfile(p,dtype=np.uint16).reshape(n,V)
corpus=json.load(open(sys.argv[1])); tfd=Path(sys.argv[2]); rd=Path(sys.argv[3])
per=[]; poss=[]; names=[]
for h in corpus["histories"]:
    n=h["tokens"]
    a=f32(load(tfd/f"{h['name']}.bin",n)); b=f32(load(rd/f"{h['name']}.bin",n))
    per.append(np.abs(a-b).max(1)); poss.append(np.arange(n)); names.append(h["name"])
per=np.concatenate(per); poss=np.concatenate(poss)
bins=[(0,1),(1,4),(4,16),(16,64),(64,128),(128,256),(256,512),(512,513),(513,600),(600,1200)]
print(f"{'position band':16s} {'rows':>5} {'mean maxabs':>12} {'p90':>8}")
for lo,hi in bins:
    m=(poss>=lo)&(poss<hi)
    if m.sum(): print(f"[{lo},{hi})".ljust(16), f"{m.sum():5d}", f"{per[m].mean():12.3f}", f"{np.percentile(per[m],90):8.3f}")
print()
for k in (512,513,514,16,32,256):
    at=np.nonzero(poss==k)[0]; nb=np.nonzero(poss==k-1)[0]
    if at.size and nb.size:
        print(f"position {k:4d}: mean {per[at].mean():.3f}   position {k-1:4d}: mean {per[nb].mean():.3f}")
print()
# first row of each history (no context) vs the mean
firsts=np.nonzero(poss==0)[0]
print("row 0 (no context): mean", round(float(per[firsts].mean()),3), "all rows mean", round(float(per.mean()),3))
