MODEL ?= inkling2-mini
PARITY_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
CAMPAIGN_DIR := $(abspath $(PARITY_DIR)../..)
MAIN_ROOT := $(abspath $(shell git rev-parse --path-format=absolute --git-common-dir)/..)
WORK := $(MAIN_ROOT)/.work
GPU_LOCK := $(WORK)/gpu.lock
REF_VENV := $(WORK)/venv-ref
REF_SCRIPT := $(CAMPAIGN_DIR)/ref/inkling2_loop_ref.py
UV ?= ~/.local/bin/uv
REF_UV := $(UV) run --frozen --project $(CAMPAIGN_DIR)/ref
COMPARE := $(PARITY_DIR)compare.py
CACHE := $(PARITY_DIR)ref_cache.py
TRACE := $(PARITY_DIR)trace_cache.py
HISTORIES := $(PARITY_DIR)histories.json
WEIGHTS_ROOT ?= $(HOME)/models
INKLING_DIR := $(WEIGHTS_ROOT)/Inkling-2-Mini@ed40e7da625f
REPORT := $(PARITY_DIR)report.md
REF_LOGITS := $(WORK)/parity/inkling2-mini/ref
METAL_LOGITS := $(WORK)/parity/inkling2-mini/metal
REF_TRACES := $(WORK)/parity/inkling2-mini/trace
RUNNER := $(MAIN_ROOT)/zig-out/bin/tf-inkling2-loop-run
MIN_MARGIN ?= 0.1
AGREEMENT ?= 99.5

.PHONY: parity-inkling2-mini ref-logits metal-logits ref-traces clean-parity

parity-inkling2-mini: $(REPORT)

# Every history's logits from the float32 reference, one history at a time under the GPU lock and cached by the
# history's sha256, so a rerun with unchanged histories recomputes nothing.
ref-logits: $(HISTORIES)
	@test -f $(INKLING_DIR)/model-00001-of-00001.safetensors || \
		{ echo "missing $(INKLING_DIR); run make -C $(CAMPAIGN_DIR) fetch NAMES=inkling-2-mini" >&2; exit 2; }
	$(REF_UV) python $(CACHE) --model $(INKLING_DIR) --histories $(HISTORIES) --out $(REF_LOGITS) \
		--reference $(REF_SCRIPT) --venv $(REF_VENV) --lock $(GPU_LOCK) --expert-cache-gib 10

# The same histories through the Metal runner, one process under the GPU lock at a time.
metal-logits: $(HISTORIES)
	@test -x $(RUNNER) || \
		{ echo "missing $(RUNNER); run zig build tf-inkling2-loop-run -Dcpu=apple_m1" >&2; exit 2; }
	@mkdir -p $(METAL_LOGITS)
	@set -e; for name in $$(jaq -r '.[].name' $(HISTORIES)); do \
		jaq --arg n "$$name" '.[] | select(.name == $$n) | .token_ids' $(HISTORIES) > $(METAL_LOGITS)/$$name.tokens.json; \
		lockf $(GPU_LOCK) $(RUNNER) $(INKLING_DIR) $(METAL_LOGITS)/$$name.tokens.json logits $(METAL_LOGITS)/$$name.bin; \
	done

$(REPORT): $(HISTORIES) $(COMPARE) $(CACHE)
	@$(MAKE) --no-print-directory -f $(lastword $(MAKEFILE_LIST)) ref-logits metal-logits
	$(REF_UV) python $(COMPARE) --histories $(HISTORIES) --ref-dir $(REF_LOGITS) --metal-dir $(METAL_LOGITS) \
		--report $(REPORT) --min-margin $(MIN_MARGIN) --agreement $(AGREEMENT)

clean-parity:
	rm -rf $(WORK)/parity/inkling2-mini $(REPORT)
