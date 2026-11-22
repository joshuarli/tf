UV ?= ~/.local/bin/uv
CMP_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
CAMPAIGN_DIR := $(abspath $(CMP_DIR)/..)
FLINT_DIR ?= $(HOME)/models/flint-int4@3b93f9053b53
CMP_CELLS ?= ~1024,~4096
CMP_PAIRS ?= 3
CMP_ENV := TMPDIR=$(TMPDIR) HF_HOME=$(HF_HOME) UV_CACHE_DIR=$(UV_CACHE_DIR)

.PHONY: comparators comparator-rec comparator-exo-main comparator-exo-2533 comparator-mlx-lm comparator-nanoattention session-b-ar session-b-copy

comparators: comparator-rec comparator-mlx-lm

define comparator-run
	cd $(CAMPAIGN_DIR) && env $(CMP_ENV) UV_PROJECT_ENVIRONMENT=$(WORK)/venv-$(1) \
		$(UV) run --frozen --project comparators/$(1) python comparators/run.py $(1) \
		--model-dir $(FLINT_DIR) --cells '$(CMP_CELLS)' --pairs $(CMP_PAIRS) $(if $(PROMPTS),--prompts '$(PROMPTS)',)
endef

comparator-rec:
	@test -f $(FLINT_DIR)/flint-int4.json || { echo "missing $(FLINT_DIR); run make -C campaign fetch NAMES=flint-int4" >&2; exit 2; }
	$(call comparator-run,rec)

comparator-mlx-lm:
	@test -f $(FLINT_DIR)/config.json || { echo "missing $(FLINT_DIR); run make -C campaign fetch NAMES=flint-int4" >&2; exit 2; }
	$(call comparator-run,mlx-lm)

session-b-ar:
	cd $(CAMPAIGN_DIR) && $(UV) run --project $(CAMPAIGN_DIR) python comparators/session.py \
		--model flint --mode ar --cells '$(CMP_CELLS)' --session B --pairs-start 4 --pairs-stop 6

session-b-copy:
	cd $(CAMPAIGN_DIR) && $(UV) run --project $(CAMPAIGN_DIR) python comparators/session.py \
		--model flint --mode copy --cells '$(CMP_CELLS)' --session B --pairs-start 4 --pairs-stop 6
