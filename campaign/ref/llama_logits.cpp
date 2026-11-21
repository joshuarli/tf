// Teacher-forced logits from a llama.cpp GGUF checkpoint.
//
// Modes:
//   --ids "1 2 3" --out logits.bin          one history from token ids
//   --ids-file ids.json --out logits.bin    ids from a JSON array or {"token_ids": [...]}
//   --tokenize-text text.txt --out ids.json tokens with parse_special=true, add_special=false
//
// Logits are written as int32 n_tokens, int32 n_vocab, then n_tokens*n_vocab float32.

#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::string read_file(const char * path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fprintf(stderr, "cannot read %s\n", path);
        exit(2);
    }
    std::stringstream buf;
    buf << in.rdbuf();
    return buf.str();
}

static std::vector<int32_t> parse_ids(const std::string & text) {
    std::string cleaned;
    bool after_key = text.find("token_ids") == std::string::npos;
    for (char c : text) {
        if (c == '[' || c == ']' || c == ',' || c == '{' || c == '}' || c == '"' || c == ':') {
            cleaned.push_back(' ');
        } else {
            cleaned.push_back(c);
        }
    }
    std::vector<int32_t> ids;
    std::stringstream ss(cleaned);
    std::string word;
    while (ss >> word) {
        if (!after_key) {
            if (word == "token_ids") {
                after_key = true;
            }
            continue;
        }
        char * end = nullptr;
        long value = strtol(word.c_str(), &end, 10);
        if (end && *end == '\0' && !word.empty() && (isdigit((unsigned char) word[0]) || word[0] == '-')) {
            ids.push_back(static_cast<int32_t>(value));
        }
    }
    return ids;
}

static void write_json_ids(const char * path, const std::vector<int32_t> & ids) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", path);
        exit(2);
    }
    fputc('[', f);
    for (size_t i = 0; i < ids.size(); ++i) {
        fprintf(f, i ? ",%d" : "%d", ids[i]);
    }
    fputs("]\n", f);
    fclose(f);
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * out_path = nullptr;
    const char * ids_arg = nullptr;
    const char * ids_file = nullptr;
    const char * tokenize_text = nullptr;
    const char * tokenize_json = nullptr;
    const char * dump_path = nullptr;
    int dump_row = -1;
    int ngl = 0;
    int threads = 6;

    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s needs a value\n", name);
                exit(2);
            }
            return argv[++i];
        };
        if (!strcmp(argv[i], "--model")) model_path = need("--model");
        else if (!strcmp(argv[i], "--out")) out_path = need("--out");
        else if (!strcmp(argv[i], "--ids")) ids_arg = need("--ids");
        else if (!strcmp(argv[i], "--ids-file")) ids_file = need("--ids-file");
        else if (!strcmp(argv[i], "--tokenize-text")) tokenize_text = need("--tokenize-text");
        else if (!strcmp(argv[i], "--tokenize-json")) tokenize_json = need("--tokenize-json");
        else if (!strcmp(argv[i], "--ngl")) ngl = atoi(need("--ngl"));
        else if (!strcmp(argv[i], "--threads")) threads = atoi(need("--threads"));
        else if (!strcmp(argv[i], "--dump")) dump_path = need("--dump");
        else if (!strcmp(argv[i], "--dump-row")) dump_row = atoi(need("--dump-row"));
        else {
            fprintf(stderr, "unknown argument %s\n", argv[i]);
            return 2;
        }
    }
    if (!model_path || !out_path) {
        fprintf(stderr, "usage: llama_logits --model m.gguf --out out.bin (--ids ... | --ids-file ... | --tokenize-text ...)\n");
        return 2;
    }

    llama_backend_init();
    // the oracle runs on the CPU: a 104 GB f32 GGUF cannot be resident on the GPU
    static ggml_backend_dev_t cpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    static ggml_backend_dev_t devices[] = { cpu_device, nullptr };
    llama_model_params mparams = llama_model_default_params();
    mparams.devices = devices;
    mparams.n_gpu_layers = ngl;
    mparams.load_mode = LLAMA_LOAD_MODE_MMAP;
    mparams.lazy_mode = LLAMA_LAZY_MODE_ON;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", model_path);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (tokenize_json) {
        // minimal JSON reader for [{"name": "...", "text": "..."}]
        std::string all = read_file(tokenize_json);
        FILE * out = fopen(out_path, "wb");
        if (!out) { fprintf(stderr, "cannot write %s\n", out_path); return 2; }
        fputs("[\n", out);
        size_t pos = 0;
        bool first = true;
        while ((pos = all.find("\"name\"", pos)) != std::string::npos) {
            auto read_string = [&](const char * key, size_t from) -> std::pair<std::string, size_t> {
                size_t k = all.find(key, from);
                k = all.find(':', k);
                size_t q0 = all.find('"', k);
                size_t q1 = q0 + 1;
                std::string value;
                while (q1 < all.size() && all[q1] != '"') {
                    if (all[q1] == '\\') { value.push_back(all[q1 + 1]); q1 += 2; }
                    else { value.push_back(all[q1]); ++q1; }
                }
                return {value, q1 + 1};
            };
            auto [name, after_name] = read_string("\"name\"", pos);
            auto [text, after_text] = read_string("\"text\"", after_name);
            pos = after_text;
            std::vector<llama_token> tokens(text.size() + 8);
            int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tokens.data(),
                                       (int32_t) tokens.size(), false, true);
            if (n < 0) { fprintf(stderr, "tokenize failed for %s\n", name.c_str()); return 1; }
            tokens.resize(n);
            fprintf(out, "%s{\"name\":\"%s\",\"token_ids\":[", first ? "" : ",\n", name.c_str());
            for (int32_t i = 0; i < n; ++i) fprintf(out, i ? ",%d" : "%d", tokens[i]);
            fputs("]}", out);
            first = false;
            fprintf(stderr, "%s: %d tokens\n", name.c_str(), n);
        }
        fputs("\n]\n", out);
        fclose(out);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }

    if (tokenize_text) {
        std::string text = read_file(tokenize_text);
        std::vector<llama_token> tokens(text.size() + 8);
        int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tokens.data(),
                                   (int32_t) tokens.size(), false, true);
        if (n < 0) {
            fprintf(stderr, "tokenize failed: %d\n", n);
            return 1;
        }
        tokens.resize(n);
        write_json_ids(out_path, tokens);
        fprintf(stderr, "tokenized %d tokens\n", n);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }

    std::vector<int32_t> ids;
    if (ids_arg) ids = parse_ids(ids_arg);
    else if (ids_file) ids = parse_ids(read_file(ids_file));
    if (ids.empty()) {
        fprintf(stderr, "no token ids\n");
        return 2;
    }
    if ((int) ids.size() > 4096) {
        fprintf(stderr, "too many tokens\n");
        return 2;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t) (ids.size() + 8);
    cparams.n_batch = (uint32_t) ids.size();
    cparams.n_ubatch = (uint32_t) ids.size();
    cparams.n_threads = threads;
    cparams.n_threads_batch = threads;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.offload_kqv = false;
    cparams.op_offload = false;
    // the default KV cache is F16; the oracle keeps K/V in F32
    cparams.type_k = GGML_TYPE_F32;
    cparams.type_v = GGML_TYPE_F32;
    static int s_dump_row = dump_row;
    if (dump_path) {
        cparams.cb_eval = [](ggml_tensor * t, bool ask, void * user_data) -> bool {
            if (ask) {
                return t->name[0] != '\0';
            }
            const char * path = static_cast<const char *>(user_data);
            if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) {
                return true;
            }
            const int64_t n = ggml_nelements(t);
            if (n <= 0 || n > 4 * 1024 * 1024) {
                return true;
            }
            std::vector<uint8_t> raw(ggml_nbytes(t));
            ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
            const float * f32 = reinterpret_cast<const float *>(raw.data());
            double sum = 0.0, sq = 0.0;
            float lo = INFINITY, hi = -INFINITY;
            for (int64_t i = 0; i < n; ++i) {
                float v;
                if (t->type == GGML_TYPE_F32) v = f32[i];
                else if (t->type == GGML_TYPE_F16) v = ggml_fp16_to_fp32(reinterpret_cast<const ggml_fp16_t *>(raw.data())[i]);
                else {
                    uint16_t b = reinterpret_cast<const uint16_t *>(raw.data())[i];
                    uint32_t w = uint32_t(b) << 16;
                    v = *reinterpret_cast<float *>(&w);
                }
                if (!std::isfinite(v)) v = 0.0f;
                sum += v; sq += double(v) * v;
                lo = std::min(lo, v); hi = std::max(hi, v);
            }
            double mean = sum / n;
            double var = sq / n - mean * mean;
            FILE * out = fopen(path, "ab");
            if (out) {
                fprintf(out, "%s %.9g %.9g %.9g %.9g %lld %lld %lld %lld", t->name, mean, std::sqrt(var < 0 ? 0 : var), lo, hi, (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3]);
                for (int64_t i = 0; i < n && i < 3; ++i) {
                    float v = t->type == GGML_TYPE_F32 ? f32[i]
                            : t->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(reinterpret_cast<const ggml_fp16_t *>(raw.data())[i])
                            : 0.0f;
                    fprintf(out, " %.9g", v);
                }
                if (s_dump_row >= 0 && t->ne[1] > s_dump_row && t->nb[0] == ggml_type_size(t->type)) {
                    const char * rowp = reinterpret_cast<const char *>(raw.data()) + s_dump_row * t->nb[1];
                    double rsum = 0.0, rsq = 0.0;
                    float rlo = INFINITY, rhi = -INFINITY;
                    for (int64_t i = 0; i < t->ne[0]; ++i) {
                        float v;
                        if (t->type == GGML_TYPE_F32) v = reinterpret_cast<const float *>(rowp)[i];
                        else if (t->type == GGML_TYPE_F16) v = ggml_fp16_to_fp32(reinterpret_cast<const ggml_fp16_t *>(rowp)[i]);
                        else {
                            uint16_t b = reinterpret_cast<const uint16_t *>(rowp)[i];
                            uint32_t w = uint32_t(b) << 16;
                            v = *reinterpret_cast<float *>(&w);
                        }
                        if (!std::isfinite(v)) v = 0.0f;
                        rsum += v; rsq += double(v) * v;
                        rlo = std::min(rlo, v); rhi = std::max(rhi, v);
                    }
                    double rmean = rsum / t->ne[0];
                    fprintf(out, "\n%s-row%d %.9g %.9g %.9g %.9g", t->name, s_dump_row, rmean,
                            std::sqrt(std::max(0.0, rsq / t->ne[0] - rmean * rmean)), rlo, rhi);
                }
                fputc('\n', out);
                fclose(out);
            }
            return true;
        };
        cparams.cb_eval_user_data = const_cast<char *>(dump_path);
    }
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    llama_batch batch = llama_batch_init((int32_t) ids.size(), 0, 1);
    batch.n_tokens = (int32_t) ids.size();
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = ids[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    FILE * f = fopen(out_path, "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", out_path);
        return 2;
    }
    const int32_t n_tokens = batch.n_tokens;
    fwrite(&n_tokens, sizeof(n_tokens), 1, f);
    fwrite(&n_vocab, sizeof(n_vocab), 1, f);
    for (int32_t i = 0; i < n_tokens; ++i) {
        const float * logits = llama_get_logits_ith(ctx, i);
        if (!logits) {
            fprintf(stderr, "missing logits for token %d\n", i);
            return 1;
        }
        fwrite(logits, sizeof(float), n_vocab, f);
    }
    fclose(f);
    fprintf(stderr, "wrote %d x %d logits to %s\n", n_tokens, n_vocab, out_path);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
