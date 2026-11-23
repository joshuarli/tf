# oMLX concurrency: stopped at the memory cap

oMLX 0.8.3 loaded `Inkling-2-Mini-oQ4e-mtp@3eea042751cd` offline. The required MTP-off sweep stopped during its first eight-agent cell when `heavy.py` charged 29.1 GB against the 28 GB cap. Metal then reported insufficient memory, and the client exited with an incomplete streamed response. The sweep exited 1. No post-run timing gate ran, so the partial timings are not qualified benchmark results.

An initial run used a fresh oMLX settings directory. oMLX enabled Lightning MTP automatically for the checkpoint. That run also stopped at eight agents, at 28.9 GB charged. The second run explicitly set `mtp_enabled` to false in the isolated server settings. `omlx serve --help` has no MTP flag. oMLX supports the setting per model.

The MTP-off run completed one repetition each at one, two, and four agents before the cap breach. The client observed cached prompt tokens, including 2,571, 2,623, and 2,672 on successive turns of the first agent. The first turn had zero cached tokens. The eight-agent cell, second repetitions, standard concurrent benchmark, output equality comparisons, and post-run timing gate remain unverified. These observations do not establish a throughput comparison with TensorFold.

At the MTP-off run's start, `memory_pressure` reported 83% system-wide memory free, and swap usage was 8,634.50 MB. After the failed run, memory free was 82% and swap usage was 8,618.50 MB. These host-wide values are snapshots, not process memory.

From the repository root, with the pinned model and oMLX environment already installed, the command below reproduces the MTP-off attempt. It uses `heavy.py` with its 28 GB default cap and creates isolated server settings under `.work/tmp/omlx-concurrency-off`. The script may stop at the same cap.

```sh
sh campaign/comparators/omlx/sweep-concurrency.sh
```

The comparator copies the multi-turn client because the original requires TensorFold-specific server log records. It retains the prompts, concurrency, token limit, and two repetitions, and reads cached-token counts from the streaming usage response. `make -C campaign omlx-tests` exited 0 with six tests. Make also printed two Python syntax errors while evaluating unrelated shell expressions; they did not change its exit status.
