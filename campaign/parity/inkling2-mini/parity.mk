MODEL ?= inkling2-mini
SELF := $(abspath $(lastword $(MAKEFILE_LIST)))
PARITY_DIR := $(dir $(SELF))
TREE := $(abspath $(PARITY_DIR)../../..)
CAMPAIGN_DIR := $(TREE)/campaign
MAIN_ROOT := $(abspath $(shell git rev-parse --path-format=absolute --git-common-dir)/..)
WORK := $(MAIN_ROOT)/.work
HEAVY := $(firstword $(wildcard $(CAMPAIGN_DIR)/heavy.py $(MAIN_ROOT)/campaign/heavy.py))
REF_VENV := $(WORK)/venv-ref
REF_SCRIPT := $(CAMPAIGN_DIR)/ref/inkling2_loop_ref.py
UV ?= ~/.local/bin/uv
REF_UV := $(UV) run --frozen --project $(CAMPAIGN_DIR)/ref
COMPARE := $(PARITY_DIR)compare.py
ORACLE := $(PARITY_DIR)oracle.py
HISTORIES := $(PARITY_DIR)histories.json
WEIGHTS_ROOT ?= $(HOME)/models
INKLING_DIR := $(WEIGHTS_ROOT)/Inkling-2-Mini@ed40e7da625f
REPORT := $(PARITY_DIR)report.md
REF_LOGITS := $(WORK)/parity/inkling2-mini/ref
METAL_LOGITS := $(WORK)/parity/inkling2-mini/metal
RUNNER := $(TREE)/zig-out/bin/tf-inkling2-loop-run
MIN_MARGIN ?= 0.1
AGREEMENT ?= 99.5
VOCAB := 200058

.PHONY: parity-inkling2-mini verify-oracle metal-logits clean-parity

parity-inkling2-mini: $(REPORT)

# The oracle's logits are the saved, receipt-checked float32 outputs: this item compares against them and does not
# recompute them, because a corpus rerun of the reference is the campaign's expensive step, not this item's.
verify-oracle: $(HISTORIES)
	$(REF_UV) python $(ORACLE) --histories $(HISTORIES) --dir $(REF_LOGITS)

# The same histories through the Metal runner, one model load a time inside the campaign's memory ceiling.
metal-logits: $(HISTORIES)
	@test -x $(RUNNER) || \
		{ echo "missing $(RUNNER); run zig build tf-inkling2-loop-run -Dcpu=apple_m1" >&2; exit 2; }
	@test -n "$(HEAVY)" || { echo "missing campaign/heavy.py: every model load runs inside its ceiling" >&2; exit 2; }
	@mkdir -p $(METAL_LOGITS)
	@set -e; for name in $$(jaq -r '.[].name' $(HISTORIES)); do \
		jaq --arg n "$$name" '.[] | select(.name == $$n) | .token_ids' $(HISTORIES) > $(METAL_LOGITS)/$$name.tokens.json; \
		tokens=$$(jaq 'length' $(METAL_LOGITS)/$$name.tokens.json); \
		if [ -f $(METAL_LOGITS)/$$name.bin ] && \
		   [ "$$(( $$(stat -f%z $(METAL_LOGITS)/$$name.bin) / 4 / $(VOCAB) ))" = "$$tokens" ]; then \
			echo "$$name: $${tokens} tokens, already written"; continue; \
		fi; \
		python3 $(HEAVY) -- $(RUNNER) $(INKLING_DIR) $(METAL_LOGITS)/$$name.tokens.json logits $(METAL_LOGITS)/$$name.bin; \
	done

$(REPORT): $(HISTORIES) $(COMPARE) $(ORACLE)
	@$(MAKE) --no-print-directory -f $(SELF) verify-oracle metal-logits
	@test -n "$(HEAVY)" || { echo "missing campaign/heavy.py: every model load runs inside its ceiling" >&2; exit 2; }
	$(REF_UV) python $(COMPARE) --histories $(HISTORIES) --ref-dir $(REF_LOGITS) --metal-dir $(METAL_LOGITS) \
		--report $(REPORT) --min-margin $(MIN_MARGIN) --agreement $(AGREEMENT) \
		--runner $(RUNNER) --model $(INKLING_DIR) --heavy $(HEAVY)

clean-parity:
	rm -rf $(WORK)/parity/inkling2-mini $(REPORT)
