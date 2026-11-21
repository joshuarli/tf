SMOKE_BIN ?= $(BASELINE_WT)/zig-out/native/bin/tensorfold-native
LOAD_LIMIT_GB := $(shell python3 -c 'import heavy; print(heavy.FLINT_LOAD_LIMIT_GB)')

.PHONY: smoke

smoke:
	mkdir -p $(WORK)/logs/smoke
	$(HEAVY) python3 smoke/smoke.py --bin $(SMOKE_BIN) --model $(MODEL) --mode $(MODE) --load-limit-gb $(LOAD_LIMIT_GB) --log $(WORK)/logs/smoke/$(MODEL)-$(MODE).log
