"""Localize an Inkling 2 Mini logit divergence between TensorFold and oMLX, layer visit by layer visit.

    capture  oMLX's model on the oQ4e checkpoint, teacher-forced over one history. Optionally in f32 (every bf16 or
             f16 parameter widened, f32 embeddings fed in), without the loop controllers' bf16 rounding, with keys
             and values rounded to f16 as TensorFold stores them, one token a forward and eagerly (so that every
             matmul is a single-row product), and with TensorFold's expert choices imposed on one row. Writes the
             logits and, for the chosen rows, every visit's attention sub-layer output, FFN input, layer output and
             router selection scores.
    visits   Two sides' per-visit agreement at chosen positions: relative L2 differences and expert sets. A side is
             `tf:DIR` (the runner's `tracerow` files `DIR/tracerow-<pos>.bin` and their `.router` sidecars) or an
             npz from `capture`.
    kl       KL(reference || candidate) per position between two captures of one history, with window means,
             medians and the positions above a threshold.

Run `capture` under `campaign/heavy.py` in the pinned oMLX environment (`.work/venv-omlx/bin/python -I -B`), with
HF_HUB_OFFLINE=1; `visits` and `kl` need only numpy. The visit order is oMLX's schedule: the prelude (layers 0 to 5),
three passes over the core (layers 6 to 17), then the tail (18 to 23). TensorFold numbers the same 48 visits as
slots 0 to 17, 24 to 35, 36 to 47 and 18 to 23.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PARITY = HERE.parent / "parity" / "inkling2-mini"
VOCAB = 200058
VISITS = 48
EXPERTS = 128
TOPK = 8
HIDDEN = 2304
GLOBAL_LAYERS = {5, 11, 17, 23}
TRACE_ROWS = 9


def tf_slot(visit: int) -> int:
    if visit < 18:
        return visit
    if visit < 30:
        return 24 + visit - 18
    if visit < 42:
        return 36 + visit - 30
    return 18 + visit - 42


def layer_of(visit: int) -> int:
    slot = tf_slot(visit)
    return slot if slot < 24 else 6 + (slot - 24) % 12


def parse_rows(text: str) -> list[int]:
    return sorted({r for part in text.split(",") if part for r in range(*(int(x) for x in part.split(":")))})


def capture(args) -> int:
    import time

    import mlx.core as mx
    import mlx.nn as nn
    from omlx.patches.inkling2 import kernels as K
    from omlx.patches.inkling2 import model as M
    from omlx.patches.inkling2 import vlm

    ids = {h["name"]: h for h in json.loads(Path(args.histories).read_text())}[args.name]["token_ids"]
    rows = parse_rows(args.rows)
    began = time.time()
    model, _ = vlm.load(args.model)
    print(f"loaded in {time.time() - began:.1f}s", flush=True)
    f32 = args.dtype == "f32"
    if f32:
        model.apply(lambda x: x.astype(mx.float32) if x.dtype in (mx.bfloat16, mx.float16) else x)
        mx.eval(model.parameters())
        left = [k for k, v in nn.utils.tree_flatten(model.parameters()) if v.dtype in (mx.bfloat16, mx.float16)]
        if left:
            raise SystemExit(f"parameters still not f32: {left[:5]}")
    if args.exact_controller:
        def mix_messages(x, message, down, gain_forget):
            n, b, length, hidden = x.shape
            z = mx.stack([M._unit_rms(x), M._unit_rms(message)], axis=1).reshape(2 * n, b * length, hidden)
            low = (z @ down.reshape(2 * n, hidden, -1)).reshape(n, 2, b * length, -1).sum(axis=1)
            gates = (low @ gain_forget).reshape(n, b, length, 2 * hidden)
            return (1.0 - mx.tanh(gates[..., hidden:])) * x + mx.tanh(gates[..., :hidden]) * message
        M._mix_messages = mix_messages
    if args.kv_f16:
        attend = M.Inkling2Attention.attend

        def attend_f16(self, q, k, v, rel, sink, kv, state=None):
            k = k.astype(mx.float16).astype(q.dtype)
            v = v.astype(mx.float16).astype(q.dtype)
            return attend(self, q, k, v, rel, sink, kv, state)
        M.Inkling2Attention.attend = attend_f16
    if args.eager:
        M._COMPILED_MAX_TOKENS = 0

    state = {"start": 0, "visit": 0, "moe": 0}
    imposed = []
    for spec in args.transplant:
        path, row = spec.rsplit(",", 1)
        imposed.append((int(row), np.fromfile(path, dtype=np.int32).reshape(VISITS, TOPK)))
    route = M.Inkling2MoE.route

    def route_imposed(self, x):
        inds, w = route(self, x)
        visit = state["moe"] % VISITS
        state["moe"] += 1
        for row, picks in imposed:
            r = row - state["start"]
            if 0 <= r < x.shape[0]:
                chosen = mx.array(picks[tf_slot(visit)].astype(np.uint32))
                scores = mx.sigmoid(M._row_matmul(x.astype(mx.float32), self.router_weight.T))[r][chosen]
                inds[r] = chosen.astype(inds.dtype)
                w[r] = scores * (self.route_scale / (scores.sum() + 1e-20))
        return inds, w
    M.Inkling2MoE.route = route_imposed

    caps: dict[str, dict] = {k: {} for k in ("attn_out", "ffn_inp", "l_out", "sel")}

    def post(self, out, gate, h, mlp_blocks, s2, s3, messages=None):
        # Inkling2Layer._post, recording the chosen rows.
        a = self.attn.output(out, gate)
        residual, s2 = K.short_conv(a, self.attn_conv, s2, self._conv_scale,
                                    residual=None if self.is_boundary else h, out_dtype=mx.float32)
        y = self.mlp_input(mlp_blocks, M._unit_rms(residual))[:, :, 0]
        if messages is not None:
            y = M._mix_messages(y[None], messages[None, :, :, 3], self.loop_down[3:],
                                self.loop_gain_forget[3:]).squeeze(0)
        xin = M._norm(self.mlp_norm, y).astype(h.dtype)
        hout, s3 = K.short_conv(self.mlp(xin), self.mlp_conv, s3, self._conv_scale, residual=residual,
                                dtype=h.dtype)
        visit = state["visit"]
        state["visit"] += 1
        want = [r for r in rows if state["start"] <= r < state["start"] + h.shape[1]]
        if want:
            at = mx.array([r - state["start"] for r in want])
            sel = mx.sigmoid(M._row_matmul(xin.reshape(-1, xin.shape[-1]).astype(mx.float32),
                                           self.mlp.router_weight.T)) + self.mlp.router_bias
            attn = residual if self.is_boundary else residual - h.astype(mx.float32)
            for key, val in (("attn_out", attn[0]), ("ffn_inp", residual[0]), ("l_out", hout[0]), ("sel", sel)):
                got = np.array(val[at].astype(mx.float32))
                for i, r in enumerate(want):
                    caps[key][(r, visit)] = got[i]
        return hout, s2, s3
    M.Inkling2Layer._post = post

    cache = model.make_cache()
    logits = np.lib.format.open_memmap(args.logits, mode="w+", dtype=np.float32, shape=(len(ids), VOCAB)) \
        if args.logits else None
    began = time.time()
    for start in range(0, len(ids), args.chunk):
        piece = mx.array([ids[start:start + args.chunk]])
        state["start"], state["visit"] = start, 0
        emb = model.model.embed_norm(model.model.embed(piece)) if f32 else None
        out = model.forward(piece, cache, emb)[0, :, :VOCAB]
        mx.eval(out, [c.state for c in cache])
        if logits is not None:
            logits[start:start + piece.shape[1]] = np.array(out)
    print(f"forward {len(ids)} tokens in {time.time() - began:.1f}s, MLX peak {mx.get_peak_memory() / 1e9:.1f} GB",
          flush=True)
    if logits is not None:
        logits.flush()
    if args.out and rows:
        arrays = {key: np.stack([np.stack([got[(r, v)] for r in rows]) for v in range(VISITS)])
                  for key, got in caps.items()}
        np.savez(args.out, rows=np.array(rows), **arrays)
        print("wrote", args.out, {k: v.shape for k, v in arrays.items()})
    return 0


def load_side(spec: str, pos: int) -> dict:
    """attn_out, ffn_inp and l_out [visits, hidden] and the router selection scores [visits, experts] at pos."""
    if spec.startswith("tf:"):
        path = Path(spec[3:]) / f"tracerow-{pos}.bin"
        x = np.fromfile(path, dtype=np.float32).reshape(VISITS, TRACE_ROWS, HIDDEN)
        router = np.fromfile(f"{path}.router", dtype=np.float32).reshape(VISITS, 2, EXPERTS).astype(np.float64)
        order = [tf_slot(v) for v in range(VISITS)]
        sel = 1 / (1 + np.exp(-router[order, 0])) + router[order, 1]
        return {"attn_out": x[order, 6], "ffn_inp": x[order, 7], "l_out": x[order, 8], "sel": sel}
    z = np.load(spec)
    r = list(z["rows"]).index(pos) if "rows" in z else pos - int(z["lo"])
    return {k: z[k][:, r].astype(np.float64 if k == "sel" else np.float32)
            for k in ("attn_out", "ffn_inp", "l_out", "sel")}


def top_set(scores: np.ndarray) -> tuple[set, float]:
    order = np.argsort(-scores, kind="stable")
    return set(order[:TOPK].tolist()), float(scores[order[TOPK - 1]] - scores[order[TOPK]])


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-30))


def visits(args) -> int:
    for pos in parse_rows(args.positions) if ":" in args.positions else [int(p) for p in args.positions.split(",")]:
        a, b = load_side(args.a, pos), load_side(args.b, pos)
        print(f"== position {pos}: {args.a} against {args.b}, relative L2 differences")
        print(" visit slot layer  attn_out   ffn_inp   moe_out     l_out  experts (a only / b only)  8th-9th gap a, b")
        for v in range(VISITS):
            sa, ga = top_set(a["sel"][v])
            sb, gb = top_set(b["sel"][v])
            moe = rel(a["l_out"][v] - a["ffn_inp"][v], b["l_out"][v] - b["ffn_inp"][v])
            sets = "same" if sa == sb else f"{sorted(sa - sb)} / {sorted(sb - sa)}"
            kind = "G" if layer_of(v) in GLOBAL_LAYERS else " "
            print(f" {v:5d} {tf_slot(v):4d} {layer_of(v):4d}{kind} {rel(a['attn_out'][v], b['attn_out'][v]):9.2e}"
                  f" {rel(a['ffn_inp'][v], b['ffn_inp'][v]):9.2e} {moe:9.2e} {rel(a['l_out'][v], b['l_out'][v]):9.2e}"
                  f"  {sets:>24s}  {ga:.2g}, {gb:.2g}")
    return 0


def open_rows(spec: str, name: str):
    sys.path.insert(0, str(PARITY))
    from omlx_gate import open_capture
    path = Path(spec)
    if path.is_file():
        if path.suffix == ".npy":
            return np.load(path, mmap_mode="r")
        return np.memmap(path, dtype=np.float32, mode="r").reshape(-1, VOCAB)
    data, why = open_capture(path, name, VOCAB)
    if data is None:
        raise SystemExit(why)
    return data


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def kl(args) -> int:
    ref, cand = open_rows(args.ref, args.name), open_rows(args.cand, args.name)
    n = min(ref.shape[0], cand.shape[0])
    out = np.zeros(n)
    for s in range(0, n, 64):
        a = log_softmax(np.asarray(ref[s:s + 64], np.float64))
        b = log_softmax(np.asarray(cand[s:s + 64], np.float64))
        out[s:s + len(a)] = (np.exp(a) * (a - b)).sum(axis=1)
    for part in args.windows.split(","):
        lo, hi = (int(x) for x in part.split(":"))
        x = out[lo:min(hi, n)]
        print(f"{lo}-{min(hi, n) - 1}: mean {x.mean():.4f} median {np.median(x):.4f}")
    spikes = np.nonzero(out > args.above)[0]
    print(f"above {args.above}:", ", ".join(f"{int(p)} ({out[p]:.2f})" for p in spikes))
    if args.at:
        print(", ".join(f"{p}: {out[int(p)]:.4f}" for p in args.at.split(",")))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--model", default=str(Path.home() / "models" / "Inkling-2-Mini-oQ4e-mtp@3eea042751cd"))
    c.add_argument("--histories", default=str(PARITY / "histories.json"))
    c.add_argument("--name", default="harbour")
    c.add_argument("--dtype", choices=("bf16", "f32"), default="bf16")
    c.add_argument("--exact-controller", action="store_true", help="no bf16 rounding in the loop controllers")
    c.add_argument("--kv-f16", action="store_true", help="round keys and values to f16 before attention")
    c.add_argument("--eager", action="store_true", help="never compile short forwards (needed for --rows)")
    c.add_argument("--chunk", type=int, default=2048, help="tokens a forward")
    c.add_argument("--rows", default="", help="lo:hi[,lo:hi...] positions whose visits to record")
    c.add_argument("--transplant", action="append", default=[],
                   help="PICKS,ROW: route that row's visits to the experts of a runner .picks sidecar")
    c.add_argument("--logits", default="", help="float32 [tokens, vocab] .npy")
    c.add_argument("--out", default="", help="npz of the recorded visits")
    v = sub.add_parser("visits")
    v.add_argument("a")
    v.add_argument("b")
    v.add_argument("positions", help="comma-separated positions")
    k = sub.add_parser("kl")
    k.add_argument("ref", help="a capture directory, or one .npy or raw .bin file")
    k.add_argument("cand")
    k.add_argument("--name", default="harbour")
    k.add_argument("--windows", default="0:1024,1024:1336,1280:1336")
    k.add_argument("--above", type=float, default=1.0)
    k.add_argument("--at", default="")
    args = ap.parse_args()
    return {"capture": capture, "visits": visits, "kl": kl}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
