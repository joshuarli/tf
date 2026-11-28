import json, numpy as np
from pathlib import Path
V=150144
def f32(b): return (b.astype(np.uint32)<<16).view(np.float32)
def load(p,n): return np.fromfile(p,dtype=np.uint16).reshape(n,V)
def lsm(x):
    x=x.astype(np.float64); m=x.max(-1,keepdims=True)
    return x-m-np.log(np.exp(x-m).sum(-1,keepdims=True))
def kl(a,b):
    pa,pb=lsm(a),lsm(b); return float((np.exp(pa)*(pa-pb)).sum(-1).mean())
d=json.load(open("/Users/josh/repos/tensorfold-m5/.work/wt/f2-parity/campaign/parity/flint/histories.json"))
tf=np.concatenate([f32(load(Path("tf")/f"{e['name']}.bin",e["tokens"])) for e in d["histories"]])
rec=np.concatenate([f32(load(Path("oracle")/f"{e['name']}.bin",e["tokens"])) for e in d["histories"]])
ch=np.concatenate([f32(load(Path("oracle-chunked")/f"{e['name']}.bin",e["tokens"])) for e in d["histories"]])
at,ar,ac=tf.argmax(1),rec.argmax(1),ch.argmax(1); N=len(tf)
print(f"top1: TF/decode {int((at==ar).sum())}/{N}={100*(at==ar).mean():.3f}%  TF/chunked {int((at==ac).sum())}/{N}={100*(at==ac).mean():.3f}%  chunked/decode {int((ac==ar).sum())}/{N}={100*(ac==ar).mean():.3f}%")
dis=np.nonzero(at!=ar)[0]
print(f"of {len(dis)} TF-vs-decode disagreements: TF matches chunked {int((at[dis]==ac[dis]).sum())}; chunked also differs from decode {int((ac[dis]!=ar[dis]).sum())}")
print(f"KL(rec||tf) {kl(rec,tf):.6f}  KL(rec||ch) {kl(rec,ch):.6f}  KL(tf||rec) {kl(tf,rec):.6f}  KL(ch||tf) {kl(ch,tf):.6f}")
