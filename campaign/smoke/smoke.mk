SMOKE_BIN ?= $(BASELINE_WT)/zig-out/native/bin/tensorfold-native
MODEL_LIMIT_VAR := $(shell $(BENCH_PY) 'import harness; print(harness.load_limit_var(harness.model_key("$(MODEL)")))')
MODEL_LIMIT_GB := $(shell $(BENCH_PY) 'import harness; print(harness.load_limit_gb(harness.model_key("$(MODEL)")))')

.PHONY: smoke

smoke:
	mkdir -p $(WORK)/logs/smoke
	$(if $(filter flint,$(MODEL)),,@echo "$(call serve_preview,$(MODEL),0)")
	$(HEAVY) python3 smoke/smoke.py --bin $(SMOKE_BIN) --model $(MODEL) --mode $(MODE) --load-limit-var $(MODEL_LIMIT_VAR) --load-limit-gb $(MODEL_LIMIT_GB) --log $(WORK)/logs/smoke/$(MODEL)-$(MODE).log
