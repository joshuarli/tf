"""Where does the residual live: per-row error shape, bias, and whether the same rows are unstable for rec itself."""
import json, sys
from pathlib import Path
import numpy as np
V=150144
def f32(b): return (b.astype(np.uint32)<<16).view(np.float32)
def load(p,n): return np.fromfile(p,dtype=np.uint16).reshape(n,V)
corpus=json.load(open(sys.argv[1])); tfd=Path(sys.argv[2]); rd=Path(sys.argv[3]); cd=Path(sys.argv[4])
rows_tf=[]; rows_rec=[]; rows_ch=[]; owner=[]
for h in corpus["histories"]:
    n=h["tokens"]; a=load(tfd/f"{h['name']}.bin",n); b=load(rd/f"{h['name']}.bin",n); c=load(cd/f"{h['name']}.bin",n)
    rows_tf.append(a); rows_rec.append(b); rows_ch.append(c); owner += [h["name"]]*n
tf=np.concatenate(rows_tf); rec=np.concatenate(rows_rec); ch=np.concatenate(rows_ch)
A,B,C=f32(tf),f32(rec),f32(ch)
d_tf=np.abs(A-B); d_rr=np.abs(C-B)
print("rows",len(A))
mx_tf=d_tf.max(1); mx_rr=d_rr.max(1)
print("per-row max abs: TF-vs-rec percentiles", {p:round(float(np.percentile(mx_tf,p)),3) for p in (50,75,90,95,99,100)})
print("per-row max abs: rec-chunk-vs-rec percentiles", {p:round(float(np.percentile(mx_rr,p)),3) for p in (50,75,90,95,99,100)})
near_tf = mx_tf<=0.5; near_rr = mx_rr<=0.5
print(f"rows with per-row max<=0.5: TF {near_tf.sum()}/{len(A)} ({near_tf.mean()*100:.1f}%), rec-self {near_rr.sum()}/{len(A)} ({near_rr.mean()*100:.1f}%)")
both = near_tf & near_rr
print("agreement of 'unstable' classification: both near", both.sum(), "tf-only near", (near_tf&~near_rr).sum(), "rec-only near", (near_rr&~near_tf).sum())
if (~near_rr).any() and near_rr.any():
    print("rows unstable for rec: their TF max abs mean", round(float(mx_tf[~near_rr].mean()),3), "vs stable", round(float(mx_tf[near_rr].mean()),3))
signed=(A-B)
print("signed mean over all entries:", float(signed.mean()), "abs mean", float(d_tf.mean()))
print("TF top1 idx == rec top1 of rec-chunk top1? recompute")
tfa,tfa2=A.argmax(1),B.argmax(1); cha=C.argmax(1)
print("top1 tf/rec", int((tfa==tfa2).sum()), "rec/recchunk", int((cha==tfa2).sum()))
print("rows where rec and recchunk disagree:", int((cha!=tfa2).sum()), "; of those TF sides with rec:", int(((cha!=tfa2)&(tfa==tfa2)).sum()), "with chunk:", int(((cha!=tfa2)&(tfa==cha)).sum()))
