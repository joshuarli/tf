NANO_PROMPTS ?= code-512,code-1024
NANO_STEPS ?= 512
NANO_PAIRS ?= 3
NANO_VARIANT ?= optimized
NANO_NATIVE := $(WORK)/build/nanoattention-native/bin/native-compare

.PHONY: nanoattention-prepare nanoattention-setup nanoattention-smoke nanoattention-bench nanoattention-plan nanoattention-tests

nanoattention-tests:
	$(UV) run --frozen --project . python -B -m unittest discover -s comparators/nanoattention -p 'test_*.py'

nanoattention-prepare:
	$(UV) run --frozen --project . python fetch.py --only nanoattention
	$(UV) run --frozen --project . python comparators/nanoattention/setup.py --prepare-only

nanoattention-setup: check-zig
	$(UV) run --frozen --project . python fetch.py --only nanoattention inkling-2-mini
	$(UV) run --frozen --project . python comparators/nanoattention/setup.py
	$(UV) run --frozen --project . python comparators/mlx-vlm/build_native.py --output $(NANO_NATIVE)

nanoattention-smoke:
	$(UV) run --frozen --project . python comparators/nanoattention/inkling.py \
		--prompts code-512 --steps 16 --pairs 1 --variant $(NANO_VARIANT)

nanoattention-bench:
	$(UV) run --frozen --project . python comparators/nanoattention/inkling.py \
		--prompts $(NANO_PROMPTS) --steps $(NANO_STEPS) --pairs $(NANO_PAIRS) --variant $(NANO_VARIANT)

nanoattention-plan:
	$(UV) run --frozen --project . python comparators/nanoattention/inkling.py --plan \
		--prompts $(NANO_PROMPTS) --steps $(NANO_STEPS) --pairs $(NANO_PAIRS) --variant $(NANO_VARIANT)
