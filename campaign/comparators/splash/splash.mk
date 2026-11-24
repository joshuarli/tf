SPLASH_ROOT := $(WORK)/src/splash@2afc7337aacc
SPLASH_MODEL ?= $(HOME)/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd
SPLASH_CELLS ?= ~1024,~4096
SPLASH_PAIRS ?= 1
SPLASH_PROFILE ?= controlled
SPLASH_VERIFY_ROWS ?= auto

.PHONY: splash-setup splash-bench

splash-setup:
	$(MAKE) fetch NAMES="splash inkling-2-mini-dflash2"
	mkdir -p $(SPLASH_ROOT)/x $(WORK)/splash-home
	tar -xzf $(SPLASH_ROOT)/splash-1.7.2-arm64-macos26.tar.gz -C $(SPLASH_ROOT)/x

splash-bench:
	cd $(CAMPAIGN_DIR) && env SPLASH_PROFILE=$(SPLASH_PROFILE) SPLASH_VERIFY_ROWS=$(SPLASH_VERIFY_ROWS) \
		$(UV) run --project $(CAMPAIGN_DIR) python comparators/run.py splash --model-dir $(SPLASH_MODEL) \
		--cells '$(SPLASH_CELLS)' --pairs $(SPLASH_PAIRS) --mode spec-$(SPLASH_PROFILE)-$(SPLASH_VERIFY_ROWS) \
		$(if $(PROMPTS),--prompts '$(PROMPTS)',)

SPLASH_CURVE ?= 1024,4096,8192,16384,31744

.PHONY: splash-curve
splash-curve:
	cd $(CAMPAIGN_DIR) && env SPLASH_PROFILE=$(SPLASH_PROFILE) SPLASH_VERIFY_ROWS=$(SPLASH_VERIFY_ROWS) \
		$(UV) run --project $(CAMPAIGN_DIR) python comparators/serve_curve.py splash --model-dir $(SPLASH_MODEL) \
		--lengths $(SPLASH_CURVE) --out runs/$(shell date -u +%Y-%m-%d)/splash-prefill-$(SPLASH_PROFILE)-$(notdir $(SPLASH_MODEL)).jsonl
