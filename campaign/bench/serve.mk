BENCH_PY := PYTHONPATH=$(CURDIR)/bench python3 -c
MODEL_DIR_OF = $(shell $(BENCH_PY) 'import harness; print(harness.model_dir(harness.model_key("$(1)")))')
MODEL_LIMIT_OF = $(shell $(BENCH_PY) 'import harness; print(harness.load_limit_var(harness.model_key("$(1)")) + "=" + str(harness.load_limit_gb(harness.model_key("$(1)"))))')
MODEL_FLAGS_OF = $(shell $(BENCH_PY) 'import harness; key = harness.model_key("$(1)"); print(" ".join(harness.default_serve_args(key) + harness.model_serve_args(key)))')
serve_preview = serve $(call MODEL_DIR_OF,$(1)) under $(call MODEL_LIMIT_OF,$(1)) through $(CURDIR)/heavy.py: tensorfold-native serve $(call MODEL_DIR_OF,$(1)) --host 127.0.0.1 --port <free> --parallel 1 --context 32768 --prompt-cache-gib $(2) --no-update-check $(call MODEL_FLAGS_OF,$(1))
