# Pi harness (H1)

Coding-agent serving check: drive `pi` 1.5.0 against the local TensorFold server and score eight
small fixed tasks.

```bash
make -C campaign pi-harness MODEL=flint            # 8 tasks, writes campaign/pi/results/<date>-flint.json
make -C campaign pi-harness MODEL=flint BIN=/path/to/tensorfold-native
make -C campaign pi-harness MODEL=flint PI_ONLY=bugfix-dedupe
```

Each task under `tasks/<name>/` holds `repo/` (the fixture the agent edits), `prompt.txt` and
`check.py`, a hidden acceptance program run on the host after the agent stops. The runner copies
`repo/` to `.work/pi/<name>/repo`, gives `pi` that directory under `sandbox-exec -f campaign/pi/sandbox.sb`
(writes only in `.work/pi/<name>` and `.work/tmp`; network only to localhost; `~/.pi` denied), and
reads the JSON event stream pi writes for turns and per-turn latency. Tool-call parse errors come from
the server's `kept_tool_calls_{unoffered,malformed}_total` counters and the server log.

The agent directory is copied into `.work/pi/<name>/agent` because pi creates lock and credential
files beside `models.json`; `campaign/pi/agent/models.json` is the committed source and defines the
`tensorfold` provider at `http://127.0.0.1:8888/v1`. `pi` is launched as

```
pi --no-extensions --no-mcp --no-skills -nc --no-session --mode json \
   --provider tensorfold --model flint --tools read,bash,edit,write -p '<prompt>'
```

with `PI_CODING_AGENT_DIR`, `PI_OFFLINE=1`, `PI_TELEMETRY=0` and `TMPDIR` inside the task's scratch.
If nothing answers on port 8888 the runner starts the baseline binary on it under the GPU lock and
stops only that process; an already-serving server is reused and left running.
