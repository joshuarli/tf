import sys
import numpy as np
V = 200058
a = np.fromfile(sys.argv[1], dtype=np.float32).reshape(-1, V)
b = np.fromfile(sys.argv[2], dtype=np.float32).reshape(-1, V)
split = int(sys.argv[3])
ai = a.view(np.uint32); bi = b.view(np.uint32)
eq = (ai == bi).all(axis=1)
print("rows", a.shape[0], "bit-equal rows before split:", int(eq[:split].sum()), "of", split, "; after:", int(eq[split:].sum()), "of", a.shape[0]-split)
print("first differing row:", int(np.argmin(eq)) if not eq.all() else None)
d = np.abs(a.astype(np.float64) - b.astype(np.float64)).max(axis=1)
print("max abs diff before split: %.6g after: %.6g" % (d[:split].max(), d[split:].max()))
print("top-1 agreement: %.4f" % (a.argmax(axis=1) == b.argmax(axis=1)).mean())
def lsm(x):
    x = x.astype(np.float64); m = x.max(axis=1, keepdims=True); return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))
la, lb = lsm(a), lsm(b)
kl = (np.exp(la) * (la - lb)).sum(axis=1)
print("mean KL(a||b) before split: %.3g after: %.3g" % (kl[:split].mean(), kl[split:].mean()))
