"""CPU float32 teacher-forced reference for the Inkling 2 Mini ``inkling2_loop`` graph.

The graph is transcribed from the pinned llama.cpp commit ``40b7a546``:
``src/models/inkling2.cpp`` (layer, relative attention, MoE, output) and
``src/models/inkling2-loop.cpp`` (core loop, controllers, per-slot state).

Quantized expert weights are decoded straight from the checkpoint's
NVFP4/MXFP8 bytes, one expert at a time, so the model is never materialized in
float32: attention, norms and the embeddings are read lazily from the
safetensors mmap, and the expert cache is bounded.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)


def _e4m3_table() -> np.ndarray:
    codes = np.arange(256, dtype=np.uint8)
    exponent = ((codes & 0x7F) >> 3).astype(np.int32)
    mantissa = (codes & 7).astype(np.float32)
    values = np.where(exponent == 0, np.ldexp(mantissa, -9), np.ldexp(1 + mantissa / 8, exponent - 7))
    values = np.where(codes & 0x80, -values, values)
    values[(codes & 0x7F) == 0x7F] = np.nan
    return values.astype(np.float32)


E4M3 = _e4m3_table()
DTYPE_SIZES = {"BF16": 2, "F16": 2, "F32": 4, "U8": 1, "F8_E4M3": 1, "I32": 4}


class Safetensors:
    """Read-only safetensors reader backed by an mmap."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        with open(self.path, "rb") as handle:
            header_len = int.from_bytes(handle.read(8), "little")
            self.header = json.loads(handle.read(header_len))
        self.data_start = 8 + header_len
        self.names = {name for name in self.header if name != "__metadata__"}

    def info(self, name: str) -> dict:
        if name not in self.header:
            raise KeyError(f"{self.path.name}: no tensor {name!r}")
        return self.header[name]

    def array(self, name: str, shape=None, dtype=np.uint8) -> np.ndarray:
        info = self.info(name)
        begin = self.data_start + info["data_offsets"][0]
        target = tuple(info["shape"]) if shape is None else shape
        return np.ndarray(target, dtype=dtype, buffer=self.mm, offset=begin)

    def f32(self, name: str) -> np.ndarray:
        dtype = self.info(name)["dtype"]
        if dtype == "BF16":
            words = self.array(name, dtype=np.uint16)
            return (words.astype(np.uint32) << 16).view(np.float32)
        if dtype == "F32":
            return self.array(name, dtype=np.float32)
        raise ValueError(f"{name}: unsupported float dtype {dtype}")

    def bf16_rows(self, name: str, rows: np.ndarray) -> np.ndarray:
        info = self.info(name)
        width = int(np.prod(info["shape"][1:])) * 2
        begin = self.data_start + info["data_offsets"][0]
        out = np.empty((len(rows), width // 2), dtype=np.float32)
        for i, row in enumerate(rows):
            words = np.ndarray((width // 2,), dtype=np.uint16, buffer=self.mm,
                               offset=begin + int(row) * width)
            out[i] = (words.astype(np.uint32) << 16).view(np.float32)
        return out

    def rows(self, name: str, row_start: int, row_stop: int, row_elems: int) -> np.ndarray:
        """Bytes of rows [row_start, row_stop) of a row-major tensor."""
        info = self.info(name)
        row = row_elems * DTYPE_SIZES[info["dtype"]]
        begin = self.data_start + info["data_offsets"][0] + row_start * row
        return np.asarray(self.mm[begin:begin + (row_stop - row_start) * row])


def rms_norm(x: np.ndarray, weight: np.ndarray | None, eps: float) -> np.ndarray:
    inv = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(eps))
    out = x * inv
    return out if weight is None else out * weight


def l2_clip(x: np.ndarray, threshold: float, scale: np.ndarray | None = None) -> np.ndarray:
    # llama.cpp clip(): sqrt(sum_rows(x^2)) + 1e-6, clamp(threshold / norm, 0, 1)
    if scale is not None:
        x = x * scale
    norm = np.sqrt(np.sum(x * x, axis=-1, keepdims=True)) + np.float32(1e-6)
    return x * np.clip(np.float32(threshold) / norm, 0.0, 1.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class Inkling2LoopReference:
    def __init__(self, model_dir: Path, expert_cache_gib: float = 10.0):
        self.dir = Path(model_dir)
        self.config = json.loads((self.dir / "config.json").read_text())
        hp = self.config["text_config"]
        self.hp = hp
        index_path = self.dir / "model.safetensors.index.json"
        if index_path.is_file():
            files = sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
        else:
            files = sorted(p.name for p in self.dir.glob("*.safetensors")
                           if ".index." not in p.name and not p.name.startswith("mtp."))
        if len(files) != 1:
            raise ValueError(f"{self.dir}: expected one text safetensors file, found {files}")
        self.st = Safetensors(self.dir / files[0])

        self.n_embd = hp["hidden_size"]
        self.n_layer = hp["num_hidden_layers"]
        self.n_head = hp["num_attention_heads"]
        self.n_head_kv = hp["num_key_value_heads"]
        self.head_dim = hp["head_dim"]
        self.n_ff = hp["moe_intermediate_size"] if "moe_intermediate_size" in hp else hp["intermediate_size"]
        self.n_expert = hp["n_routed_experts"]
        self.n_expert_used = hp["num_experts_per_tok"]
        self.vocab = hp["vocab_size"]
        self.unpadded_vocab = hp["unpadded_vocab_size"]
        self.n_rel = hp["d_rel"]
        self.rel_extent = hp["rel_extent"]
        self.swa = hp["sliding_window_size"]
        self.block = hp["n_layers_per_attn_res_block"]
        self.eps = hp["rms_norm_eps"]
        self.qk_clip = hp["qk_l2_clip_threshold"]
        self.rel_clip = hp["rel_l2_clip_threshold"]
        self.gate_softcap = hp["mlp_gate_softcap"]
        self.up_softcap = hp["mlp_up_softcap"]
        self.route_scale = hp["route_scale"]
        self.logit_softcap = hp["final_logit_softcapping"]
        self.log_floor = hp["log_scaling_n_floor"]
        self.log_alpha = hp["log_scaling_alpha"]
        self.n_conv = hp.get("conv_kernel_size") or hp.get("sconv_kernel_size")
        loop = hp["loop_config"]
        self.core_start = loop["core_start"]
        self.core_end = loop["core_end"]
        self.n_passes = loop["num_passes"]
        self.gate_rank = loop["gate_rank"]
        self.n_slot = self.n_layer + (self.n_passes - 1) * (self.core_end - self.core_start)

        layer_types = hp.get("layer_types")
        if layer_types is None:
            local = set(hp["local_layer_ids"])
            layer_types = ["hybrid_sliding" if i in local else "hybrid" for i in range(self.n_layer)]
        self.is_swa = [t == "hybrid_sliding" for t in layer_types]

        self.trace = None
        self.trace_row = None
        self.router_path = None
        self.router_rows = {}
        self.force_picks = []
        self.row_dump = None
        self.row_stash = {}
        self._weights: dict[str, np.ndarray] = {}
        self._experts: dict[tuple[int, int], object] = {}
        self._expert_bytes = 0
        self._expert_limit = int(expert_cache_gib * 1024**3)

    # ---------------------------------------------------------------- weights

    def w(self, name: str) -> np.ndarray:
        out = self._weights.get(name)
        if out is None:
            out = np.ascontiguousarray(self.st.f32(name))
            self._weights[name] = out
        return out

    def layer_weight(self, layer: int, suffix: str) -> str:
        return f"model.llm.layers.{layer}.{suffix}"

    def query(self, name: str) -> np.ndarray:
        arr = self.w(name)
        return arr.reshape(1, -1) if arr.ndim == 1 else arr

    def conv_weight(self, layer: int, which: str, swa: bool) -> np.ndarray:
        prefix = self.layer_weight(layer, f"attn.{which}_sconv")
        if swa:
            return np.ascontiguousarray(self.st.f32(prefix + ".weight").squeeze(1).T)
        return np.ascontiguousarray(self.st.f32(prefix + ".static_w")[::-1])

    def post_conv_weight(self, layer: int, which: str) -> np.ndarray:
        name = self.layer_weight(layer, "attn_sconv.weight" if which == "attn" else "mlp_sconv.weight")
        return np.ascontiguousarray(self.st.f32(name).squeeze(1).T)

    def physical_layer(self, slot: int) -> int:
        if slot < self.n_layer:
            return slot
        return self.core_start + (slot - self.n_layer) % (self.core_end - self.core_start)

    # ---------------------------------------------------------------- experts

    def _decode_expert(self, layer: int, kind: str, expert: int) -> np.ndarray | None:
        name = self.layer_weight(layer, f"mlp.experts.{kind}_weight")
        if name not in self.st.names:
            return None
        info = self.st.info(name)
        rows = info["shape"][1]
        packed_rows, packed_cols = info["shape"][1], info["shape"][2]
        nvfp4 = info["dtype"] == "U8"

        chunk = 256
        parts = []
        for start in range(0, rows, chunk):
            stop = min(start + chunk, rows)
            block = self.st.rows(name, expert * packed_rows + start, expert * packed_rows + stop, packed_cols)
            if nvfp4:
                block = block.reshape(stop - start, packed_cols)
                values = np.empty((stop - start, packed_cols * 2), dtype=np.float32)
                values[:, 0::2] = E2M1[block & 15]
                values[:, 1::2] = E2M1[block >> 4]
            else:
                values = E4M3[block.reshape(stop - start, packed_cols)].astype(np.float32, copy=True)
            parts.append(values)
        values = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)

        scale_name = self.layer_weight(layer, f"mlp.experts.{kind}_weight_scale")
        scale_shape = tuple(self.st.info(scale_name)["shape"])
        scale_rows = scale_shape[1]
        scales = []
        for start in range(0, scale_rows, chunk):
            stop = min(start + chunk, scale_rows)
            raw = self.st.rows(scale_name, expert * scale_rows + start, expert * scale_rows + stop,
                               scale_shape[2])
            scales.append(raw.reshape(stop - start, scale_shape[2]))
        scales = scales[0] if len(scales) == 1 else np.concatenate(scales, axis=0)
        if nvfp4:
            block_scale = E4M3[scales]
            values = values.reshape(values.shape[0], -1, 16)
            values = (values * block_scale[:, :, None]).reshape(values.shape[0], -1)
            amax = np.float32(self.st.f32(self.layer_weight(layer, f"mlp.experts.{kind}_weight_amax"))[expert])
            denom = np.float32(self.st.f32(
                self.layer_weight(layer, f"mlp.experts.{kind}_weight_outer_scale_denominator"))[0])
            values *= np.float32(amax / denom)
        else:
            i32 = scales.astype(np.int32) - 127
            factors = np.ldexp(np.ones(i32.shape, dtype=np.float32), i32)
            values = values.reshape(values.shape[0], -1, 32)
            values = (values * factors[:, :, None]).reshape(values.shape[0], -1)
        if not np.isfinite(values).all():
            raise ValueError(f"{name}: expert {expert}: non-finite decoded weight")
        return np.ascontiguousarray(values)

    @staticmethod
    def _nbytes(array: np.ndarray) -> int:
        return array.nbytes

    def expert_pair(self, layer: int, expert: int) -> tuple[np.ndarray, np.ndarray]:
        key = (layer, expert)
        found = self._experts.get(key)
        if found is not None:
            return found
        w13 = self._decode_expert(layer, "w13", expert)
        w2 = self._decode_expert(layer, "w2", expert)
        if w13 is None or w2 is None:
            raise ValueError(f"layer {layer}: expert {expert} has no packed weights")
        # the checkpoint interleaves the gate and up rows: even rows are gate
        fused = np.ascontiguousarray(np.concatenate([w13[0::2], w13[1::2]], axis=0))
        del w13
        pair = (fused, w2)
        if self._nbytes(fused) + self._nbytes(w2) + self._expert_bytes <= self._expert_limit:
            self._experts[key] = pair
            self._expert_bytes += self._nbytes(fused) + self._nbytes(w2)
        return pair

    def _record(self, name: str, arr: np.ndarray) -> None:
        if self.trace is None:
            return
        flat = np.asarray(arr, dtype=np.float32).ravel()
        self.trace.write(f"{name} {flat.mean():.9g} {flat.std():.9g} {flat.min():.9g} {flat.max():.9g}"
                         f" {flat[0]:.9g} {flat[1]:.9g} {flat[2]:.9g}\n")
        if self.trace_row is not None and arr.ndim == 2 and arr.shape[0] > self.trace_row:
            row = np.asarray(arr[self.trace_row], dtype=np.float32)
            self.trace.write(f"{name}-row{self.trace_row} {row.mean():.9g} {row.std():.9g}"
                             f" {row.min():.9g} {row.max():.9g}\n")
        if (self.row_dump is not None and self.trace_row is not None and arr.ndim == 2
                and arr.shape[0] > self.trace_row and "-row" not in name
                and name.startswith(("mixin-", "normin-", "attn_out-", "ffn_inp-", "l_out-"))):
            self.row_stash[name] = np.ascontiguousarray(arr[self.trace_row], dtype=np.float32)

    # ---------------------------------------------------------------- graph

    def _mix(self, blocks: list[np.ndarray], cur: np.ndarray, query: np.ndarray) -> np.ndarray:
        states = [rms_norm(block, None, 1e-6) for block in reversed(blocks)]
        states.append(rms_norm(cur, None, 1e-6))
        stacked = np.stack(states, axis=0)
        scores = np.einsum("dtn,kn->kdt", stacked, query, optimize=True)
        scores *= np.float32(1.0 / math.sqrt(self.n_embd))
        scores -= scores.max(axis=1, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(axis=1, keepdims=True)
        return np.einsum("kdt,dtn->ktn", weights, stacked, optimize=True)

    def _control(self, cur: np.ndarray, message: np.ndarray, layer: int, ctrl: str) -> np.ndarray:
        gate = np.concatenate([rms_norm(cur, None, 1e-6), rms_norm(message, None, 1e-6)], axis=-1)
        g = gate @ self.w(self.layer_weight(layer, f"controllers.{ctrl}.down_weight")).T
        gain = np.tanh(g @ self.w(self.layer_weight(layer, f"controllers.{ctrl}.gain_weight")).T)
        forget = np.tanh(g @ self.w(self.layer_weight(layer, f"controllers.{ctrl}.forget_weight")).T)
        return cur - forget * cur + gain * message

    def _convolve(self, x: np.ndarray, weight: np.ndarray, dynamic: np.ndarray | None) -> np.ndarray:
        n_conv = self.n_conv
        tokens = x.shape[0]
        padded = np.concatenate([np.zeros((n_conv - 1, x.shape[1]), dtype=np.float32), x], axis=0)
        out = np.array(x, dtype=np.float32, copy=True)
        for k in range(n_conv):
            out += padded[k:k + tokens] * weight[k]
        if dynamic is not None:
            heads = dynamic.shape[1]
            width = x.shape[1] // heads
            for lag in range(n_conv):
                window = padded[n_conv - 1 - lag:n_conv - 1 - lag + tokens]
                out += (window.reshape(tokens, heads, width) * dynamic[:, :, lag][:, :, None]).reshape(tokens, -1)
        return out * np.float32(1.0 / math.sqrt((2 if dynamic is not None else 1) * n_conv + 1))

    def _tau(self, tokens: int) -> np.ndarray:
        pos = np.arange(tokens, dtype=np.float32)
        scaled = np.maximum(np.float32(1.0), (pos + np.float32(1.0)) / np.float32(self.log_floor))
        return np.float32(1.0) + np.float32(self.log_alpha) * np.log(scaled)

    def _attention(self, layer: int, swa: bool, q_in: np.ndarray, k_in: np.ndarray, v_in: np.ndarray,
                   tau: np.ndarray) -> np.ndarray:
        tokens = q_in.shape[0]
        D, H, KV = self.head_dim, self.n_head, self.n_head_kv
        q = (q_in @ self.w(self.layer_weight(layer, "attn.wq_du.weight")).T).reshape(tokens, H, D)
        k = (k_in @ self.w(self.layer_weight(layer, "attn.wk_dv.weight")).T).reshape(tokens, KV, D)
        v = (v_in @ self.w(self.layer_weight(layer, "attn.wv_dv.weight")).T).reshape(tokens, KV, D)
        r = (q_in @ self.w(self.layer_weight(layer, "attn.wr_du.weight")).T).reshape(tokens, H, self.n_rel)

        q_scale = self.w(self.layer_weight(layer, "attn.q_l2_clip.scale"))
        r_scale = self.w(self.layer_weight(layer, "attn.rel_l2_clip.scale"))
        q = l2_clip(q, self.qk_clip, q_scale[None])
        r = l2_clip(r, self.rel_clip, r_scale[None])

        k_dyn = v_dyn = None
        if not swa:
            k_dyn = (k_in @ self.w(self.layer_weight(layer, "attn.k_sconv.weight_proj")).T).reshape(
                tokens, KV, self.n_conv)
            v_dyn = (v_in @ self.w(self.layer_weight(layer, "attn.v_sconv.weight_proj")).T).reshape(
                tokens, KV, self.n_conv)
        k = self._convolve(k.reshape(tokens, KV * D), self.conv_weight(layer, "k", swa), k_dyn).reshape(tokens, KV, D)
        v = self._convolve(v.reshape(tokens, KV * D), self.conv_weight(layer, "v", swa), v_dyn).reshape(tokens, KV, D)
        k = l2_clip(k, self.qk_clip)
        v = rms_norm(v, None, self.eps)

        if not swa:
            q = q * tau[:, None, None]
            r = r * tau[:, None, None]
        slot = self._slot
        self._record(f"Qcur-{slot}", q)
        self._record(f"Kcur-{slot}", k)
        self._record(f"Vcur-{slot}", v)
        self._record(f"Rcur-{slot}", r)

        sinks = q_in @ self.w(self.layer_weight(layer, "attn.attn_sink_proj.weight")).T
        bias = self.w(self.layer_weight(layer, "attn.rel_logits_proj.proj"))
        expected = self.swa if swa else self.rel_extent
        if bias.shape[1] != expected:
            raise ValueError(f"layer {layer}: relative bias extent {bias.shape[1]} != {expected}")
        extent = bias.shape[1]
        bias = np.concatenate([bias, np.zeros((bias.shape[0], 1), dtype=np.float32)], axis=1)
        table = np.einsum("rq,thr->thq", bias, r, optimize=True) - sinks[:, :, None]

        raw = np.arange(tokens)[:, None] - np.arange(tokens)[None, :]
        index = np.where((raw >= 0) & (raw < extent), raw, extent)
        kq_b = np.take_along_axis(table, index[:, None, :].repeat(H, axis=1), axis=2)
        self._record(f"kq_b-{slot}", kq_b)

        k_exp = np.repeat(k, H // KV, axis=1)
        v_exp = np.repeat(v, H // KV, axis=1)
        qk = np.einsum("thd,jhd->thj", q, k_exp, optimize=True)
        self._record(f"kq-{slot}", qk)
        scores = qk * np.float32(1.0 / math.sqrt(D)) + kq_b
        self._record(f"kq_plus_kq_b-{slot}", scores)

        allowed = raw >= 0
        if swa:
            allowed &= raw < self.swa
        scores = np.where(allowed[:, None, :], scores, np.float32(-np.inf))
        peak = np.maximum(scores.max(axis=-1, keepdims=True), np.float32(0.0))
        weights = np.exp(scores - peak)
        weights = weights / (np.exp(-peak) + weights.sum(axis=-1, keepdims=True))
        self._record(f"kq_soft_max-{slot}", weights)
        out = np.einsum("thj,jhd->thd", weights, v_exp, optimize=True).reshape(tokens, H * D)
        self._record(f"kqv_out-{slot}", out)
        out = out * sigmoid(q_in @ self.w(self.layer_weight(layer, "attn.gate_attn.weight")).T)
        return self._convolve(out @ self.w(self.layer_weight(layer, "attn.wo_ud.weight")).T,
                              self.post_conv_weight(layer, "attn"), None)

    def _moe(self, layer: int, x: np.ndarray) -> np.ndarray:
        logits = x @ self.w(self.layer_weight(layer, "mlp.gate.weight")).T
        probs = sigmoid(logits)
        selection = probs + self.w(self.layer_weight(layer, "mlp.gate.bias"))
        order = np.argsort(-selection, axis=1, kind="stable")[:, :self.n_expert_used]
        if self.force_picks:
            order = order.copy()
            for text in self.force_picks:
                fields = [int(v) for v in text.split(",")]
                wanted_slot, wanted_expert = fields[0], fields[1]
                wanted_row = fields[2] if len(fields) > 2 else order.shape[0] - 1
                if wanted_slot == self._slot:
                    if wanted_expert in order[wanted_row]:
                        raise ValueError(f"slot {wanted_slot}: forced expert"
                                        f" {wanted_expert} already picked")
                    order[wanted_row, -1] = wanted_expert
        if self.router_path is not None:
            positions = np.arange(selection.shape[0])
            ninth_all = np.argsort(-selection, axis=1, kind="stable")[:, self.n_expert_used]
            gap_all = selection[positions, order[:, -1]] - selection[positions, ninth_all]
            last = selection.shape[0] - 1
            self.router_rows[str(self._slot)] = {
                "layer": layer,
                "picks": [int(v) for v in order[last]],
                "ninth": int(ninth_all[last]),
                "gap_8_9": float(gap_all[last]),
                "gap_8_9_prob": float(probs[last, order[last, -1]] - probs[last, ninth_all[last]]),
                "forced": [text for text in self.force_picks
                             if int(text.split(",")[0]) == self._slot],
                "picks_all": [[int(v) for v in row] for row in order],
                "ninth_all": [int(v) for v in ninth_all],
                "gaps_all": [float(v) for v in gap_all],
            }
        denom = np.take_along_axis(probs, order, axis=1).sum(axis=1) + np.float32(1e-20)
        slot = self._slot
        self._record(f"ffn_moe_in-{slot}", x)
        self._record(f"ffn_moe_logits-{slot}", logits)
        self._record(f"ffn_moe_probs-{slot}", probs)
        self._record(f"ffn_moe_weights-{slot}", np.take_along_axis(probs, order, axis=1))
        self._record(f"ffn_moe_weights_norm-{slot}", np.take_along_axis(probs, order, axis=1) / denom[:, None])
        self._record(f"ffn_moe_weights_scaled-{slot}", np.take_along_axis(probs, order, axis=1) / denom[:, None]
                     * np.float32(self.route_scale))

        out = np.zeros_like(x)
        for expert in np.unique(order):
            rows = np.nonzero((order == expert).any(axis=1))[0]
            fused, w2 = self.expert_pair(layer, int(expert))
            gate_up = x[rows] @ fused.T
            gate = gate_up[:, :self.n_ff]
            up = gate_up[:, self.n_ff:]
            self._record(f"ffn_moe_gate_up-{slot}", gate_up)
            self._record(f"ffn_moe_gate-{slot}", gate)
            self._record(f"ffn_moe_up-{slot}", up)
            act = np.float32(self.gate_softcap) * np.tanh(gate / np.float32(self.gate_softcap)) * sigmoid(gate)
            if self.up_softcap > 0:
                up = np.float32(self.up_softcap) * np.tanh(up / np.float32(self.up_softcap))
            act = act * up
            self._record(f"ffn_moe_situ-{slot}", act)
            down = act @ w2.T
            self._record(f"ffn_moe_down-{slot}", down)
            scale = probs[rows, expert] / denom[rows] * np.float32(self.route_scale)
            out[rows] += scale[:, None] * down
        return out

    def _layer(self, slot: int, ip: int, cur: np.ndarray, blocks: list[np.ndarray],
               messages: np.ndarray | None) -> np.ndarray:
        layer = self.physical_layer(slot)
        swa = self.is_swa[layer]
        inp_sa = cur
        inputs = [cur, cur, cur]
        if layer > 0:
            mixed = self._mix(blocks, cur, self.query(f"model.llm.attn_res.attn_queries.{layer - 1}"))
            inputs = [mixed[min(j, mixed.shape[0] - 1)] for j in range(3)]
        if messages is not None:
            for j, ctrl in enumerate(("attn", "k", "v")):
                inputs[j] = self._control(inputs[j], messages[j], layer, ctrl)

        for j, value in enumerate(inputs):
            self._record(f"mixin-{slot}-{j}", value)
        if swa:
            norm = self.w(self.layer_weight(layer, "attn_norm.weight"))
            if messages is not None:
                inputs = [rms_norm(x, norm, self.eps) for x in inputs]
            else:
                shared = rms_norm(inputs[0], norm, self.eps)
                inputs = [shared, shared, shared]
        else:
            inputs = [
                rms_norm(inputs[0], self.w(self.layer_weight(layer, "q_input_norm.weight")), self.eps),
                rms_norm(inputs[1], self.w(self.layer_weight(layer, "k_input_norm.weight")), self.eps),
                rms_norm(inputs[2], self.w(self.layer_weight(layer, "v_input_norm.weight")), self.eps),
            ]

        for j, value in enumerate(inputs):
            self._record(f"normin-{slot}-{j}", value)
        self._slot = slot
        cur = self._attention(layer, swa, inputs[0], inputs[1], inputs[2], self._tau(inp_sa.shape[0]))
        self._record(f"attn_out-{slot}", cur)
        if ip % self.block == 0:
            blocks.append(inp_sa)
        else:
            cur = inp_sa + cur

        ffn_inp = cur
        self._record(f"ffn_inp-{slot}", cur)
        cur = self._mix(blocks, cur, self.query(f"model.llm.attn_res.mlp_queries.{layer}"))[0]
        if messages is not None:
            cur = self._control(cur, messages[3], layer, "mlp")
        cur = rms_norm(cur, self.w(self.layer_weight(layer, "mlp_norm.weight")), self.eps)
        cur = self._moe(layer, cur)
        self._record(f"ffn_out-{slot}", cur)
        cur = self._convolve(cur, self.post_conv_weight(layer, "ffn"), None)
        out = ffn_inp + cur
        self._record(f"l_out-{slot}", out)
        return out

    def _loop_queries(self) -> list[np.ndarray]:
        q = self.st.f32("model.llm.middle_cycle_cross_pass_queries")
        kv = self.st.f32("model.llm.middle_cycle_cross_pass_kv_queries")
        return [np.stack([q[2 * i], kv[2 * i], kv[2 * i + 1], q[2 * i + 1]])
                for i in range(self.core_end - self.core_start)]

    def _lm_head(self, cur: np.ndarray) -> np.ndarray:
        logits = np.empty((cur.shape[0], self.vocab), dtype=np.float32)
        step = 16384
        for start in range(0, self.vocab, step):
            stop = min(start + step, self.vocab)
            words = self.st.array("model.llm.unembed.weight", shape=(self.vocab, self.n_embd), dtype=np.uint16)
            rows = (np.asarray(words[start:stop]).astype(np.uint32) << 16).view(np.float32)
            logits[:, start:stop] = cur @ rows.T
        cap = np.float32(self.logit_softcap)
        return (cap * np.tanh(logits / cap))[:, :self.unpadded_vocab]

    def logits(self, token_ids: list[int]) -> np.ndarray:
        ids = np.asarray(token_ids, dtype=np.int64)
        cur = rms_norm(self.st.bf16_rows("model.llm.embed.weight", ids),
                       self.w("model.llm.embed_norm.weight"), self.eps)

        blocks: list[np.ndarray] = []
        for il in range(self.core_start):
            cur = self._layer(il, il, cur, blocks, None)
        prelude = cur
        prelude_blocks = list(blocks)

        queries = self._loop_queries()
        for pass_index in range(self.n_passes):
            bank: list[np.ndarray] = []
            bank_out = None
            if pass_index > 0:
                bank = list(blocks[len(prelude_blocks) + 1:])
                bank_out = cur
                cur = prelude
                blocks = list(prelude_blocks)
            for ip in range(self.core_start, self.core_end):
                slot = ip if pass_index == 0 else self.n_layer + (pass_index - 1) * (
                    self.core_end - self.core_start) + (ip - self.core_start)
                messages = None if pass_index == 0 else self._mix(bank, bank_out, queries[ip - self.core_start])
                cur = self._layer(slot, ip, cur, blocks, messages)

        for il in range(self.core_end, self.n_layer):
            cur = self._layer(il, il, cur, blocks, None)

        cur = self._mix(blocks, cur, self.query("model.llm.attn_res.output_query"))[0]
        cur = rms_norm(cur, self.w("model.llm.norm.weight"), self.eps)
        self._record("result_norm", cur)
        out = self._lm_head(cur)
        self._record("result_output", out)
        if self.router_path is not None:
            Path(self.router_path).write_text(json.dumps(self.router_rows, indent=1) + "\n")
            self.router_rows = {}
        if self.row_dump is not None:
            np.savez(self.row_dump, **{k: v for k, v in self.row_stash.items()})
            self.row_stash = {}
        return out


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU float32 teacher-forced Inkling 2 Mini reference")
    parser.add_argument("--model", required=True)
    parser.add_argument("--histories", help="JSON list of {name, token_ids}")
    parser.add_argument("--out-dir", help="write <name>.npy logits for every history")
    parser.add_argument("--tokens", help="comma- or space-separated token ids")
    parser.add_argument("--out", help="write one .npy logits file")
    parser.add_argument("--expert-cache-gib", type=float, default=10.0)
    parser.add_argument("--trace", help="write per-tensor mean/std/min/max, like llama_logits --dump")
    parser.add_argument("--trace-row", type=int, help="also write stats for one token row")
    parser.add_argument("--router-report", help="write per-slot router top-8 picks and 8th/9th gaps for the last row")
    parser.add_argument("--force-pick", action="append", default=[],
                        help="SLOT,EXPERT[,ROW]: rerun with one row's 8th pick replaced (repeatable)")
    parser.add_argument("--row-dump", help="write an .npz of full last-row vectors for the traced tensors")
    args = parser.parse_args()

    model = Inkling2LoopReference(Path(args.model), args.expert_cache_gib)
    trace = open(args.trace, "w", encoding="utf-8") if args.trace else None
    model.trace = trace
    model.trace_row = args.trace_row
    model.router_path = args.router_report
    model.force_picks = args.force_pick
    model.row_dump = args.row_dump
    if args.histories:
        histories = json.loads(Path(args.histories).read_text())
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for entry in histories:
            logits = model.logits(entry["token_ids"])
            np.save(out_dir / f"{entry['name']}.npy", logits)
            top = np.argmax(logits, axis=1)
            print(f"{entry['name']}: {len(entry['token_ids'])} tokens, "
                  f"top-1 {' '.join(map(str, top.tolist()))}")
        if trace:
            trace.close()
        return 0
    if args.tokens:
        ids = [int(x) for x in args.tokens.replace(",", " ").split()]
        logits = model.logits(ids)
        np.save(args.out, logits)
        print(f"wrote {logits.shape} logits to {args.out}")
        if trace:
            trace.close()
        return 0
    parser.error("give --histories/--out-dir or --tokens/--out")
    return 2


if __name__ == "__main__":
    sys.exit(main())
