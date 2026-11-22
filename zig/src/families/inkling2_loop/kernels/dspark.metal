#include <metal_stdlib>
using namespace metal;

struct Matrix { uint rows, inner, count; };
struct Norm { uint width, rows; float epsilon; };
struct Mix { uint rows, width; };
struct Rope { uint rows, heads, dim, position; float theta; };
struct Attention { uint context, heads, kv_heads, dim, window; };

kernel void ds_matrix(const device float* x [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                      device float* y [[buffer(2)]], constant Matrix& a [[buffer(3)]],
                      uint2 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    uint row = group.x * 8 + sg;
    if (row >= a.rows || group.y >= a.count) return;
    float sum = 0;
    for (uint i = lane; i < a.inner; i += 32)
        sum = fma(x[group.y * a.inner + i], float(w[(size_t)row * a.inner + i]), sum);
    sum = simd_sum(sum);
    if (lane == 0) y[group.y * a.rows + row] = sum;
}

kernel void ds_norm(const device float* x [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                    device float* y [[buffer(2)]], constant Norm& a [[buffer(3)]],
                    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sums[8];
    float sum = 0;
    for (uint i = tid; i < a.width; i += 256) {
        float v = x[row * a.width + i];
        sum = fma(v, v, sum);
    }
    sum = simd_sum(sum);
    if (lane == 0) sums[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = lane < 8 ? sums[lane] : 0;
        v = simd_sum(v);
        if (lane == 0) sums[0] = rsqrt(v / float(a.width) + a.epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < a.width; i += 256)
        y[row * a.width + i] = x[row * a.width + i] * sums[0] * float(w[i]);
}

kernel void ds_mix(const device float* x [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                   device float* y [[buffer(2)]], constant Mix& a [[buffer(3)]],
                   uint i [[thread_position_in_grid]]) {
    if (i >= a.rows * a.width) return;
    float v0 = float(w[0]), v1 = float(w[1]), v2 = float(w[2]), v3 = float(w[3]);
    float mx = max(max(v0, v1), max(v2, v3));
    float4 e = exp(float4(v0, v1, v2, v3) - mx);
    e /= e.x + e.y + e.z + e.w;
    uint row = i / a.width, col = i % a.width;
    y[i] = x[(row * 4 + 0) * a.width + col] * e.x + x[(row * 4 + 1) * a.width + col] * e.y
         + x[(row * 4 + 2) * a.width + col] * e.z + x[(row * 4 + 3) * a.width + col] * e.w;
}

kernel void ds_add(const device float* x [[buffer(0)]], const device float* y [[buffer(1)]],
                   device float* z [[buffer(2)]], constant uint& count [[buffer(3)]],
                   uint i [[thread_position_in_grid]]) {
    if (i < count) z[i] = x[i] + y[i];
}

kernel void ds_copy(const device float* x [[buffer(0)]], device float* y [[buffer(1)]],
                    constant uint& count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < count) y[i] = x[i];
}

kernel void ds_silu(const device float* x [[buffer(0)]], const device float* y [[buffer(1)]],
                    device float* z [[buffer(2)]], constant uint& count [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
    if (i < count) z[i] = (x[i] / (1.0f + exp(-x[i]))) * y[i];
}

kernel void ds_rope(device float* x [[buffer(0)]], constant Rope& a [[buffer(1)]],
                    uint i [[thread_position_in_grid]]) {
    uint half_width = a.dim / 2;
    if (i >= a.rows * a.heads * half_width) return;
    uint row = i / (a.heads * half_width), head = (i / half_width) % a.heads, col = i % half_width;
    uint off = (row * a.heads + head) * a.dim + col;
    float angle = float(a.position + row) * pow(a.theta, -2.0f * float(col) / float(a.dim));
    float s = sin(angle), c = cos(angle), v0 = x[off], v1 = x[off + half_width];
    x[off] = v0 * c - v1 * s;
    x[off + half_width] = v0 * s + v1 * c;
}

kernel void ds_scores(const device float* q [[buffer(0)]], const device float* ck [[buffer(1)]],
                      const device float* bk [[buffer(2)]], device float* scores [[buffer(3)]],
                      constant Attention& a [[buffer(4)]], uint3 g [[thread_position_in_grid]]) {
    uint row = g.x, head = g.y, pos = g.z;
    if (row >= 8 || head >= a.heads || pos >= a.context + 8) return;
    uint kh = head / (a.heads / a.kv_heads);
    const device float* key = pos < a.context ? ck + (pos * a.kv_heads + kh) * a.dim
        : bk + ((pos - a.context) * a.kv_heads + kh) * a.dim;
    float sum = 0;
    for (uint i = 0; i < a.dim; ++i) sum = fma(q[(row * a.heads + head) * a.dim + i], key[i], sum);
    scores[(row * a.heads + head) * (a.context + 8) + pos] = sum * rsqrt(float(a.dim));
}

kernel void ds_attention(const device float* scores [[buffer(0)]], const device float* cv [[buffer(1)]],
                         const device float* bv [[buffer(2)]], device float* out [[buffer(3)]],
                         constant Attention& a [[buffer(4)]], uint3 g [[thread_position_in_grid]]) {
    uint row = g.x, head = g.y, col = g.z;
    if (row >= 8 || head >= a.heads || col >= a.dim) return;
    uint start = a.window > 0 && a.context > a.window ? a.context - a.window : 0;
    const device float* s = scores + (row * a.heads + head) * (a.context + 8);
    float maximum = -INFINITY;
    for (uint i = start; i < a.context + 8; ++i) maximum = max(maximum, s[i]);
    float sum = 0, value = 0;
    uint kh = head / (a.heads / a.kv_heads);
    for (uint i = start; i < a.context + 8; ++i) {
        float p = exp(s[i] - maximum);
        float v = i < a.context ? cv[(i * a.kv_heads + kh) * a.dim + col]
                                 : bv[((i - a.context) * a.kv_heads + kh) * a.dim + col];
        sum += p;
        value = fma(p, v, value);
    }
    out[(row * a.heads + head) * a.dim + col] = value / sum;
}

kernel void ds_anchor(const device bfloat* embed [[buffer(0)]], const device bfloat* mask [[buffer(1)]],
                      const device uint* token [[buffer(2)]], device float* x [[buffer(3)]],
                      constant uint& width [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= 8 * width) return;
    x[i] = i < width ? float(embed[(size_t)token[0] * width + i]) : float(mask[i % width]);
}

kernel void ds_rnn_input(const device float* state [[buffer(0)]], const device uint* token [[buffer(1)]],
                         const device bfloat* lookup [[buffer(2)]], const device float* hidden [[buffer(3)]],
                         const device float* features [[buffer(4)]], device float* out [[buffer(5)]],
                         constant uint2& a [[buffer(6)]], uint i [[thread_position_in_grid]]) {
    if (i >= 5120) return;
    if (i < 256) out[i] = state[i];
    else if (i < 512) out[i] = float(lookup[(size_t)token[0] * 256 + i - 256]);
    else if (i < 2816) out[i] = hidden[a.x * 2304 + i - 512];
    else out[i] = features[i - 2816];
}

kernel void ds_recur(const device float* projection [[buffer(0)]], const device bfloat* bias [[buffer(1)]],
                     device float* state [[buffer(2)]], device float* latent [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= 256) return;
    float gate = 1.0f / (1.0f + exp(-(projection[i] + float(bias[i]))));
    float candidate = tanh(projection[256 + i] + float(bias[256 + i]));
    state[i] = gate * state[i] + (1.0f - gate) * candidate;
    latent[i] = tanh(projection[512 + i] + float(bias[512 + i]));
}

kernel void ds_bias(device float* x [[buffer(0)]], const device bfloat* bias [[buffer(1)]],
                    constant uint& count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < count) x[i] += float(bias[i]);
}

kernel void ds_logits(const device float* base [[buffer(0)]], const device float* correction [[buffer(1)]],
                      device float* logits [[buffer(2)]], constant uint& count [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i < count) logits[i] = 30.0f * tanh((base[i] + correction[i]) / 30.0f);
}

kernel void ds_argmax(const device float* logits [[buffer(0)]], device uint* token [[buffer(1)]],
                      constant uint& count [[buffer(2)]], uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float best_values[8];
    threadgroup uint best_ids[8];
    float value = -INFINITY;
    uint id = 0;
    for (uint i = tid; i < count; i += 256) {
        float v = logits[i];
        if (v > value || (v == value && i < id)) { value = v; id = i; }
    }
    float v = simd_max(value);
    uint t = simd_min(value == v ? id : UINT_MAX);
    if (lane == 0) { best_values[sg] = v; best_ids[sg] = t; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float x = lane < 8 ? best_values[lane] : -INFINITY;
        uint y = lane < 8 ? best_ids[lane] : UINT_MAX;
        float m = simd_max(x);
        uint j = simd_min(x == m ? y : UINT_MAX);
        if (lane == 0) token[0] = j;
    }
}
