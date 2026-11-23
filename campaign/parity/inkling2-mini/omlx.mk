# The oMLX oracle: teacher-forced oMLX logits on the oQ4e checkpoint, and the gate that holds TensorFold to them.
# Every fragment shares the campaign Makefile's namespace, so these variables carry an OMLX_ORACLE_ prefix.
OMLX_ORACLE_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
OMLX_ORACLE_CAMPAIGN := $(abspath $(OMLX_ORACLE_DIR)/../..)
OMLX_ORACLE_TREE := $(abspath $(OMLX_ORACLE_CAMPAIGN)/..)
OMLX_ORACLE_WORK ?= $(abspath $(shell git -C $(OMLX_ORACLE_DIR) rev-parse --path-format=absolute --git-common-dir)/../.work)
OMLX_ORACLE_HEAVY ?= python3 $(OMLX_ORACLE_CAMPAIGN)/heavy.py --
OMLX_ORACLE_MODEL ?= $(HOME)/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd
OMLX_ORACLE_REF ?= $(OMLX_ORACLE_WORK)/parity/inkling2-mini/omlx-ref
OMLX_ORACLE_VENV ?= $(OMLX_ORACLE_WORK)/venv-omlx
OMLX_ORACLE_ENV ?= env 'PATH=$(OMLX_ORACLE_VENV)/bin:$(PATH)' XDG_CACHE_HOME=$(OMLX_ORACLE_WORK)/xdg-cache \
	HF_HOME=$(OMLX_ORACLE_WORK)/hf-home TMPDIR=$(OMLX_ORACLE_WORK)/tmp PYTHONDONTWRITEBYTECODE=1 \
	OMLX_WITH_CUSTOM_KERNEL=0 HF_HUB_OFFLINE=1 INKLING_LOAD_LIMIT_GB=26
OMLX_ORACLE_CAPTURE_ARGS ?=
# The gate's runner mode and chunk come from the gate file, so they always match the recorded baseline.
OMLX_ORACLE_GATE ?= $(OMLX_ORACLE_DIR)/omlx_gate.json
OMLX_ORACLE_MODE = $(shell jaq -r .candidate.runner_mode $(OMLX_ORACLE_GATE))
OMLX_ORACLE_CHUNK = $(shell jaq -r .candidate.prefill_chunk $(OMLX_ORACLE_GATE))
OMLX_ORACLE_RUNNER ?= $(OMLX_ORACLE_TREE)/zig-out/bin/tf-inkling2-loop-run
# Lane-scoped so concurrent lanes never share candidate captures; receipts decide whether a capture is reused.
OMLX_ORACLE_LANE ?= $(subst /,_,$(shell git -C $(OMLX_ORACLE_DIR) rev-parse --abbrev-ref HEAD 2>/dev/null || echo local))
OMLX_ORACLE_BINS ?= $(OMLX_ORACLE_WORK)/parity/inkling2-mini/omlx-gate/$(OMLX_ORACLE_LANE)-$(OMLX_ORACLE_MODE)-$(OMLX_ORACLE_CHUNK)
OMLX_ORACLE_PYTHON ?= $(UV) run --frozen --project $(OMLX_ORACLE_CAMPAIGN)/ref python -I -B

.PHONY: omlx-oracle-capture omlx-oracle-runner omlx-oracle-gate

# Capture, or resume, the oracle in one oMLX load inside the memory ceiling. Needs `make -C campaign omlx-setup`.
omlx-oracle-capture:
	$(OMLX_ORACLE_ENV) $(OMLX_ORACLE_HEAVY) $(OMLX_ORACLE_VENV)/bin/python -I -B $(OMLX_ORACLE_DIR)/omlx_capture.py \
		--out $(OMLX_ORACLE_REF) $(OMLX_ORACLE_CAPTURE_ARGS)

omlx-oracle-runner:
	cd $(OMLX_ORACLE_TREE) && zig build tf-inkling2-loop-run -Dcpu=apple_m1 -j1

# The runner's captures come from parity.mk's metal-logits, one model load a history under heavy.py, so they carry
# the same receipts as every parity capture. The comparison then streams them against the oracle.
omlx-oracle-gate: omlx-oracle-runner
	INKLING_LOAD_LIMIT_GB=26 $(MAKE) -C $(OMLX_ORACLE_CAMPAIGN) -f parity/inkling2-mini/parity.mk metal-logits \
		RUNNER_MODE=$(OMLX_ORACLE_MODE) PREFILL_CHUNK=$(OMLX_ORACLE_CHUNK) RUNNER=$(OMLX_ORACLE_RUNNER) \
		INKLING_DIR=$(OMLX_ORACLE_MODEL) METAL_LOGITS=$(OMLX_ORACLE_BINS)
	$(OMLX_ORACLE_PYTHON) $(OMLX_ORACLE_DIR)/omlx_gate.py gate --gate $(OMLX_ORACLE_GATE) --oracle $(OMLX_ORACLE_REF) \
		--cand $(OMLX_ORACLE_BINS) --runner $(OMLX_ORACLE_RUNNER) --model $(OMLX_ORACLE_MODEL) \
		--json $(OMLX_ORACLE_BINS).json --report $(OMLX_ORACLE_BINS).md
