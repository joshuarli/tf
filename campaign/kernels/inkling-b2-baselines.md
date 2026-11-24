# Inkling 2 Mini corrected agent baselines

Item B2, 2026-11-23, M5 Max 48 GiB, on the corrected reasoning-replay workload from `69651393c`. TensorFold leads oMLX at one and two agents and trails it at four, and it matches oMLX at eight. Current llama.cpp, on a different 4-bit file, trails TensorFold at one and two agents, matches it on the warm four-agent repetition, and cannot run eight agents inside the 28 GB budget. The T3 target of 1.10× the strongest engine in every cell is not met: TensorFold fails it at four and eight agents.

TensorFold's aggregate rate does not rise past two agents (76.6, 72.0 and 76.6 tok/s warm at two, four and eight), while oMLX and llama.cpp both gain from two to four. Its four-agent mean time to first token is 998 ms warm, against 623 ms for oMLX and 501 ms for llama.cpp. Multi-agent scheduling, not kernel speed, is the next bottleneck.

## Engines

| Engine | Revision | Weights | Settings |
| --- | --- | --- | --- |
| TensorFold | main `038f24cce` (P3c and D2) | oQ4e `3eea042751cd` | `--parallel N`, `--prompt-cache-gib 2`, `--context 4096`, copy drafts on, `INKLING_LOAD_LIMIT_GB=26` |
| oMLX 0.8.3 | `e72868dac12d`, MLX 0.32.4 | oQ4e `3eea042751cd` | `--max-concurrent-requests 4`, `--memory-guard-gb 26`, MTP off or on |
| llama.cpp | upstream `af5c58347fdb` (2026-11-24 01:12 UTC) | `Xylolabs/Inkling-2-Mini-GGUF` NVFP4-XL `109af3770438`, 17.6 GB | `-ngl 999 --load-mode none -np N -c 4096·N --jinja --temp 0`, `--cache-ram 2048` MiB unless noted |

TensorFold and oMLX serve the same weights, so their rows form the controlled comparison. llama.cpp serves other weights and belongs in the best-practical table. Its first turn renders to the same 2,572 prompt tokens as TensorFold's and produced the same 94-token reply.

## Results

Aggregate output tok/s over the four-turn workload. Repetition 1 is the first pass, repetition 2 repeats it with the server warm.

| Agents | TensorFold | oMLX, MTP off | oMLX, MTP on | llama.cpp | TensorFold over oMLX, MTP off |
| ---: | --- | --- | --- | --- | --- |
| 1 | 37.69, 57.24 | 27.33, 52.89 | 29.50, 67.17 | 23.23, 34.15 | 1.38, 1.08 |
| 2 | 55.17, 76.56 | 52.16, 59.83 | 52.10, 56.98 | 30.73, 52.71 | 1.06, 1.28 |
| 4 | 64.49, 72.00 | 77.18, 80.43 | 77.47, 82.75 | 36.72, 72.78 (prompt cache off) | 0.84, 0.90 |
| 8 | 69.05, 76.56 | 75.28, 76.51 | 74.72, 75.29 | over budget | 0.92, 1.00 |

Peak charged memory, in GB:

| Agents | TensorFold | oMLX, MTP off | oMLX, MTP on | llama.cpp |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 19.33 | 20.50 | 22.60 | 20.44 |
| 2 | 20.61 | 20.10 | 20.58 | 23.83 |
| 4 | 21.61 | 24.40 | 25.69 | 27.80 |
| 8 | 23.60 | 24.68 | 25.04 | over 28 |

Mean time to first token at four agents, repetitions 1 and 2: TensorFold 1,438 and 998 ms, oMLX with MTP off 870 and 623 ms, and llama.cpp 4,231 and 501 ms.

## What differs between the runs

- **Server lifetime.** TensorFold and llama.cpp start one server per agent count. oMLX serves all four counts from one server, in the order 1, 2, 4, 8, so its first repetition at two agents and above runs on a warm process. Repetition 2 is the fairer comparison for those cells.
- **Slots.** oMLX's eight-agent cell is eight clients on four slots. Its earlier receipt found that eight slots exceed the 28 GB cap.
- **Gates.** The TensorFold and llama.cpp sweeps check the CPU and GPU gates before and after each agent count. The oMLX script checks them once at the start and once at the end.
- **Tokens.** TensorFold generated 268, 540, 1,087 and 2,172 tokens at one, two, four and eight agents. oMLX's counts differ by up to 2%, and llama.cpp's by up to 6%, because their arithmetic differs.

## llama.cpp inside the budget

Each llama.cpp slot holds about 2.2 GB of state at 4,096 tokens (its log reports a 2,218 MiB prompt state), against 307 MB for a TensorFold session. With `--cache-ram 2048` the four-agent server exceeded the 28 GB cap at 28.2 GB. With the RAM prompt cache off it fit at 27.8 GB. Eight agents exceeded the cap at 28.3 GB even with 8-bit keys and values (`-fa on -ctk q8_0 -ctv q8_0`), a lossy setting that ran 35.51 and 68.65 tok/s at four agents.

The concurrent-versus-solo equality check could not run on llama.cpp: every solo request from `tools/bench_concurrent.py` failed, so no equality claim is made for it.

At kernel level, `llama-bench` on the same file and flags reports 899 tok/s prefill at 1,024 tokens, 716 tok/s at 4,096, and 37.0 tok/s generation at 64 tokens.

`campaign/heavy.py` charges memory-mapped weights twice when Metal wires them: once as resident pages and again as wired growth beyond the footprint. A memory-mapped llama.cpp server was stopped at 35.8 GB charged with a 0.6 GB footprint. Loading with `--load-mode none` puts the weights in the footprint, so they are charged once.

## Not yet measured

The best llama.cpp Inkling stack (#32808 indexed bias, #32752 and #32883 chunking, #32855 depth-mix order, all open), mlx-serve and MTPLX are not yet measured. oMLX has not been run on the one-server-per-count protocol.

## Reproduction

Raw records are in `campaign/kernels/inkling-b2-runs/`.

```sh
make -C campaign fetch NAMES="llama.cpp-upstream inkling-2-mini-nvfp4-xl-gguf"
mkdir -p .work/src/llama.cpp-upstream@af5c58347fdb/tree
tar -xzf .work/src/llama.cpp-upstream@af5c58347fdb/llama.cpp-af5c58347fdbc2cd19e65b278e81c8f528f8a175.tar.gz \
  -C .work/src/llama.cpp-upstream@af5c58347fdb/tree --strip-components 1
(cd .work/src/llama.cpp-upstream@af5c58347fdb/tree && cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF && cmake --build build --target llama-server llama-bench)
INKLING_MODEL="$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" sh campaign/fx/sweep-subagents.sh N 4096
sh campaign/comparators/llama-cpp/sweep-subagents.sh N 4096
LLAMA_TAG=f16-nocache LLAMA_CACHE_MIB=0 sh campaign/comparators/llama-cpp/sweep-subagents.sh 4 4096
OMLX_MTP=off sh campaign/comparators/omlx/sweep-concurrency.sh
OMLX_MTP=on sh campaign/comparators/omlx/sweep-concurrency.sh
```

Every sweep ran with `.work/tmp/gates` (from [the f16 KV procedure](inkling-f16-kv.md#served-agent-sweeps)) first on `PATH` and `PERF_GATE_MAX_BUSY_CORES=4`. Each agent count ran once.
