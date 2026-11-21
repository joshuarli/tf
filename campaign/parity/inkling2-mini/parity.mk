MODEL ?= inkling2-mini
# The fragment's own directory, independent of how many fragments the campaign Makefile included before it.
# `$(lastword $(MAKEFILE_LIST))` is this fragment while it is being read, so this is its own path.
CAMPAIGN_DIR ?= $(abspath $(dir $(lastword $(MAKEFILE_LIST)))../..)
SELF ?= $(CAMPAIGN_DIR)/parity/inkling2-mini/parity.mk
PARITY_DIR ?= $(CAMPAIGN_DIR)/parity/inkling2-mini/
TREE ?= $(abspath $(CAMPAIGN_DIR)/..)
MAIN_ROOT ?= $(abspath $(shell git rev-parse --path-format=absolute --git-common-dir)/..)
WORK ?= $(MAIN_ROOT)/.work
HEAVY_PY ?= $(firstword $(wildcard $(CAMPAIGN_DIR)/heavy.py $(MAIN_ROOT)/campaign/heavy.py))
REF_VENV ?= $(WORK)/venv-ref
REF_SCRIPT ?= $(CAMPAIGN_DIR)/ref/inkling2_loop_ref.py
UV ?= ~/.local/bin/uv
REF_UV ?= $(UV) run --frozen --project $(CAMPAIGN_DIR)/ref
COMPARE ?= $(PARITY_DIR)compare.py
INKLING_ORACLE ?= $(PARITY_DIR)oracle.py
INKLING_PARITY_HISTORIES ?= $(PARITY_DIR)histories.json
WEIGHTS_ROOT ?= $(HOME)/models
INKLING_DIR ?= $(WEIGHTS_ROOT)/Inkling-2-Mini@ed40e7da625f
REPORT ?= $(PARITY_DIR)report.md
REF_LOGITS ?= $(WORK)/parity/inkling2-mini/ref
METAL_LOGITS ?= $(WORK)/parity/inkling2-mini/metal
RUNNER ?= $(TREE)/zig-out/bin/tf-inkling2-loop-run
MIN_MARGIN ?= 0.1
AGREEMENT ?= 99.5
VOCAB ?= 200058

.PHONY: parity-inkling2-mini verify-oracle metal-logits clean-parity

parity-inkling2-mini: $(REPORT)

# The oracle's logits are the saved, receipt-checked float32 outputs: this item compares against them and does not
# recompute them, because a corpus rerun of the reference is the campaign's expensive step, not this item's.
verify-oracle: $(INKLING_PARITY_HISTORIES) $(INKLING_ORACLE)
	$(REF_UV) python $(INKLING_ORACLE) --histories $(INKLING_PARITY_HISTORIES) --dir $(REF_LOGITS)

# The same histories through the Metal runner, one model load a time inside the campaign's memory ceiling.
metal-logits: $(INKLING_PARITY_HISTORIES)
	@test -x $(RUNNER) || \
		{ echo "missing $(RUNNER); run zig build tf-inkling2-loop-run -Dcpu=apple_m1" >&2; exit 2; }
	@test -n "$(HEAVY_PY)" || { echo "missing campaign/heavy.py: every model load runs inside its ceiling" >&2; exit 2; }
	@mkdir -p $(METAL_LOGITS)
	@set -e; for name in $$(jaq -r '.[].name' $(INKLING_PARITY_HISTORIES)); do \
		jaq --arg n "$$name" '.[] | select(.name == $$n) | .token_ids' $(INKLING_PARITY_HISTORIES) > $(METAL_LOGITS)/$$name.tokens.json; \
		tokens=$$(jaq 'length' $(METAL_LOGITS)/$$name.tokens.json); \
		if [ -f $(METAL_LOGITS)/$$name.bin ] && \
		   [ "$$(( $$(stat -f%z $(METAL_LOGITS)/$$name.bin) / 4 / $(VOCAB) ))" = "$$tokens" ]; then \
			echo "$$name: $${tokens} tokens, already written"; continue; \
		fi; \
		python3 $(HEAVY_PY) -- $(RUNNER) $(INKLING_DIR) $(METAL_LOGITS)/$$name.tokens.json logits $(METAL_LOGITS)/$$name.bin; \
	done

$(REPORT): $(INKLING_PARITY_HISTORIES) $(COMPARE) $(INKLING_ORACLE) | verify-oracle metal-logits
	@test -n "$(HEAVY_PY)" || { echo "missing campaign/heavy.py: every model load runs inside its ceiling" >&2; exit 2; }
	$(REF_UV) python $(COMPARE) --histories $(INKLING_PARITY_HISTORIES) --ref-dir $(REF_LOGITS) --metal-dir $(METAL_LOGITS) \
		--report $(REPORT) --min-margin $(MIN_MARGIN) --agreement $(AGREEMENT) \
		--runner $(RUNNER) --model $(INKLING_DIR) --heavy $(HEAVY_PY) --ref-venv $(REF_VENV) --reference $(REF_SCRIPT)

clean-parity:
	rm -rf $(WORK)/parity/inkling2-mini $(REPORT)
