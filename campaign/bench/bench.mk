UV ?= ~/.local/bin/uv
MODEL ?= flint
MODE ?= ar
CELLS ?= 1024,4096,32768
PROMPTS ?= all
PAIRS ?= 1
BIN ?= $(WORK)/wt/baseline/zig-out/native/bin/tensorfold-native
CANDIDATE_BIN ?=

.PHONY: bench scorecard prompts prompts-check

bench:
	$(if $(filter flint,$(MODEL)),,@echo "$(call serve_preview,$(MODEL),2)")
	$(UV) run --project $(CURDIR) python $(CURDIR)/bench/bench.py \
		--model $(MODEL) --mode $(MODE) --cells $(CELLS) --prompts '$(PROMPTS)' --pairs $(PAIRS) \
		--bin '$(BIN)' $(if $(CANDIDATE_BIN),--candidate-bin '$(CANDIDATE_BIN)',)

scorecard:
	$(UV) run --project $(CURDIR) python $(CURDIR)/scorecard.py

prompts:
	$(UV) run --project $(CURDIR) python $(CURDIR)/prompts/build.py --measure

prompts-check:
	$(UV) run --project $(CURDIR) python $(CURDIR)/prompts/build.py --check
