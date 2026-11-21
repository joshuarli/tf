SMOKE_BIN ?= $(BASELINE_WT)/zig-out/native/bin/tensorfold-native
LOAD_LIMIT_GB := $(shell jaq -r '.metal.recommendedMaxWorkingSetSize | tonumber * 0.9 / 1e9 | floor' host.json)

.PHONY: smoke

smoke:
	mkdir -p $(WORK)/logs/smoke
	lockf $(GPU_LOCK) python3 smoke/smoke.py --bin $(SMOKE_BIN) --model $(MODEL) --mode $(MODE) --load-limit-gb $(LOAD_LIMIT_GB) --log $(WORK)/logs/smoke/$(MODEL)-$(MODE).log
