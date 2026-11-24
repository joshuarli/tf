"""Find the row with the largest logit change between two chunk schedules and the routing flip behind it."""
import sys
import numpy as np

V, E, K, SLOTS = 200058, 128, 8, 48
ORDER = list(range(0, 18)) + list(range(24, 48)) + list(range(18, 24))


def layer(slot):
    return slot if slot < 24 else 6 + (slot - 24) % 12


def routes(path, n):
    raw = np.fromfile(path, dtype=np.float32)
    per = n * (K + E) + E
    out = []
    for s in range(SLOTS):
        block = raw[s * per:(s + 1) * per]
        picks = block[:n * K].view(np.int32).reshape(n, K)
        router = block[n * K:n * (K + E)].reshape(n, E)
        bias = block[n * (K + E):]
        out.append((picks, router, bias))
    return out


def sel(router, bias):
    return 1.0 / (1.0 + np.exp(-router.astype(np.float64))) + bias.astype(np.float64)


a = np.fromfile(sys.argv[1], dtype=np.float32).reshape(-1, V)
b = np.fromfile(sys.argv[2], dtype=np.float32).reshape(-1, V)
split = int(sys.argv[5])
d = np.abs(a.astype(np.float64) - b.astype(np.float64))
row = int(d.max(axis=1).argmax())
tok = int(d[row].argmax())
print(f"max |dlogit| {d[row, tok]:.4f} at row {row} token {tok}: {a[row, tok]:.4f} vs {b[row, tok]:.4f}")
print(f"row {row} top-1 {int(a[row].argmax())} vs {int(b[row].argmax())}; mean |dlogit| over the row {d[row].mean():.4f}")
ra = routes(sys.argv[3] + ".0", a.shape[0])
rb = routes(sys.argv[4] + ".0", split) if row < split else routes(sys.argv[4] + f".{split}", a.shape[0] - split)
rrow = row if row < split else row - split
flips_total = 0
for s in ORDER:
    pa, la, bias = ra[s]
    pb, lb, _ = rb[s]
    sa, sb = set(pa[row].tolist()), set(pb[rrow].tolist())
    if sa != sb:
        flips_total += 1
first = None
for s in ORDER:
    pa, la, bias = ra[s]
    pb, lb, _ = rb[s]
    sa, sb = set(pa[row].tolist()), set(pb[rrow].tolist())
    drift = np.abs(la[row].astype(np.float64) - lb[rrow].astype(np.float64)).max()
    if sa != sb and first is None:
        first = s
        xa, xb = sel(la[row], bias), sel(lb[rrow], bias)
        ordera, orderb = np.sort(xa)[::-1], np.sort(xb)[::-1]
        print(f"first top-8 change at slot {s} (layer {layer(s)}), router logit drift there {drift:.3g}")
        print(f"  only in A: {sorted(sa - sb)} only in B: {sorted(sb - sa)}")
        for e in sorted(sa ^ sb):
            print(f"  expert {e}: score A {xa[e]:.6f} B {xb[e]:.6f} (router {la[row, e]:.5f} vs {lb[rrow, e]:.5f})")
        print(f"  8th/9th score margin A {ordera[7] - ordera[8]:.6f} B {orderb[7] - orderb[8]:.6f}")
    elif first is None:
        print(f"slot {s:2d} (layer {layer(s):2d}): same top-8, router logit drift {drift:.3g}")
print(f"row {row}: top-8 sets differ at {flips_total} of {SLOTS} slots")
rows = min(a.shape[0], split)
changed = 0
margins = []
for s in ORDER:
    pa, la, bias = ra[s]
    pb, lb, _ = rb[s]
    for r in range(rows):
        if set(pa[r].tolist()) != set(pb[r].tolist()):
            changed += 1
            x = np.sort(sel(la[r], bias))[::-1]
            margins.append(x[7] - x[8])
print(f"rows 0..{rows - 1}: {changed} of {rows * SLOTS} row-slots change their top-8; median 8th/9th margin there "
      f"{np.median(margins):.3g}, 90th percentile {np.percentile(margins, 90):.3g}")
allm = []
pa, la, bias = ra[ORDER[0]]
for s in ORDER:
    pa, la, bias = ra[s]
    x = np.sort(sel(la[:rows], bias), axis=1)[:, ::-1]
    allm.append(x[:, 7] - x[:, 8])
allm = np.concatenate(allm)
print(f"all row-slots: median 8th/9th margin {np.median(allm):.3g}")
