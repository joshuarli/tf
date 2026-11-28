import json, subprocess
from pathlib import Path
import numpy as np
import mlx.core as mx
from rec_mlx.kernels import build_decoder
from rec_mlx.quantized import QATModel
ROOT=Path("/Users/josh/repos/tensorfold-m5/.work/wt/f2-parity"); W=Path(".")
M=Path.home()/"models"/"flint-int4@3b93f9053b53"; LOCK="/Users/josh/repos/tensorfold-m5/.work/gpu.lock"; V=150144
def f32(a): return (a.astype(np.uint32)<<16).view(np.float32)
mx.set_cache_limit(2*1024**3)
model=QATModel(M); engine,_=build_decoder(model, optimized=True); mx.eval(model.native.parameters())
for n in (1,2,4,16):
    toks=json.load(open(f"scale_{n}.json"))
    subprocess.run(["lockf",LOCK,str(ROOT/"zig-out/bin/tf-flint-run"),str(M),f"scale_{n}.json","logits",f"scale_{n}_tf.bin"],check=True)
    tf=np.fromfile(f"scale_{n}_tf.bin",dtype=np.uint16).reshape(n,V)
    st=engine.prefill_state(n,limit=n+1); st["initial"]=0; st["kv"].grow(n); st["capacity"]=st["kv"].capacity
    dec=np.stack([np.array(engine.forward(st,t)[0].astype(mx.bfloat16).view(mx.uint16)).reshape(V) for t in toks])
    a,b=f32(tf),f32(dec)
    d=np.abs(a-b)
    print(f"n={n:3d} bits_equal={np.array_equal(tf,dec)} max={d.max():.4f} mean={d.mean():.5f} p99={np.percentile(d,99):.4f} top1={int((a.argmax(1)==b.argmax(1)).sum())}/{n} scale={np.abs(a).max():.1f}")
    if n==1:
        row=d.reshape(-1); order=np.argsort(-row)[:5]
        print("   top-5 divergent vocab:", [(int(i), float(a[0,i]), float(b[0,i])) for i in order])
