"""spread.py REF.bin CAND.bin: teacher-forced agreement of two all-row logit dumps of one prompt."""
import sys
import numpy as np

V = 200058
ref = np.fromfile(sys.argv[1], dtype=np.float32).reshape(-1, V)
cand = np.fromfile(sys.argv[2], dtype=np.float32).reshape(-1, V)
equal = (ref.view(np.uint32) == cand.view(np.uint32)).all(axis=1)
top2 = np.sort(ref, axis=1)[:, -2:]
margin = top2[:, 1] - top2[:, 0]
agree = ref.argmax(axis=1) == cand.argmax(axis=1)
wide = margin >= 0.1
def lsm(x):
    x = x.astype(np.float64)
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))
lr, lc = lsm(ref), lsm(cand)
kl = (np.exp(lr) * (lr - lc)).sum(axis=1)
delta = np.abs(ref.astype(np.float64) - cand.astype(np.float64)).max(axis=1)
row = int(delta.argmax())
print(f"rows {ref.shape[0]}; bit-equal rows {int(equal.sum())}; top-1 {agree.mean() * 100:.2f}%; "
      f"top-1 at margin >= 0.1 {agree[wide].sum()}/{wide.sum()} ({agree[wide].mean() * 100:.2f}%); "
      f"mean KL {kl.mean():.5f}; max KL {kl.max():.3f}; max |dlogit| {delta.max():.3f} at row {row}")
