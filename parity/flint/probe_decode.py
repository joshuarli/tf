import json, sys, time
from pathlib import Path
import numpy as np
import mlx.core as mx
from rec_mlx.kernels import build_decoder
from rec_mlx.quantized import QATModel

VOCAB = 150144
corpus = json.loads(Path('/Users/josh/repos/tensorfold-m5/.work/wt/f2-parity/campaign/parity/flint/histories.json').read_text())
name = sys.argv[1] if len(sys.argv) > 1 else 'debug-session-nan-loss'
entry = [h for h in corpus['histories'] if h['name'] == name][0]
tokens = entry['ids']
T = len(tokens)
model = QATModel(Path.home() / 'models' / 'flint-int4@3b93f9053b53')
engine, _ = build_decoder(model, optimized=True)
mx.eval(model.native.parameters())
native = model.native
ref = np.fromfile(f'/Users/josh/repos/tensorfold-m5/.work/parity/flint/oracle/{name}.bin', dtype=np.uint16).reshape(-1, VOCAB)

state = engine.prefill_state(T + 1, limit=T + 1)
state['initial'] = 0
state['kv'].grow(T + 1)
state['capacity'] = state['kv'].capacity
cache = model.make_cache(T + 1, state['kv'])
rows = [0, 10, 100, 500, 512, 600, T - 1]
got = {}
t0 = time.perf_counter()
for t, tok in enumerate(tokens):
    h = native.model.embed_tokens(mx.array([[tok]], mx.int32))
    for layer, c in zip(native.layers, cache):
        h = layer(h, None, c)
    if t in rows:
        logits = native.lm_head(native.model.norm(h)).reshape(-1).astype(mx.bfloat16)
        mx.eval(logits)
        got[t] = np.array(logits.view(mx.uint16))
print(json.dumps({'seconds': time.perf_counter() - t0, 'tokens': T}))
for t in rows:
    same = bool(np.array_equal(got[t], ref[t]))
    d = np.abs((got[t].astype(np.uint32) << 16).view(np.float32) - (ref[t].astype(np.uint32) << 16).view(np.float32)).max()
    print(json.dumps({'row': t, 'bits_equal_bin': same, 'max_abs': float(d), 'top1': int(got[t].argmax()), 'bin_top1': int(ref[t].argmax())}))
