UV ?= ~/.local/bin/uv
FLINT_REPO := $(abspath $(CURDIR)/..)
FLINT_DIR := parity/flint
FLINT_MODEL_DIR := $(HOME)/models/flint-int4@3b93f9053b53
FLINT_ORACLE_ENV := $(WORK)/venv-rec
FLINT_LOGITS := $(WORK)/parity/flint
FLINT_CORPUS := $(FLINT_DIR)/histories.json
FLINT_REPORT := $(FLINT_DIR)/report.md
FLINT_ORACLE := rec_mlx 0.1.0 at e53663b262af4cb1ca498e596ca8b1a2d79e0800 (recursive-oss/rec, Apache-2.0), resident decode one token at a time
FLINT_EQUIV := row 10 of linked-list-reversal, the equivalence asserted in tests/validate_resident.py, bit-identical
FLINT_ENV := TMPDIR=$(TMPDIR) HF_HOME=$(HF_HOME) UV_CACHE_DIR=$(UV_CACHE_DIR) UV_PROJECT_ENVIRONMENT=$(FLINT_ORACLE_ENV)
FLINT_PY := $(FLINT_ORACLE_ENV)/bin/python

.PHONY: parity-flint

parity-flint: check-zig
	mkdir -p $(TMPDIR) $(FLINT_LOGITS)
	cd $(FLINT_REPO) && $(FLINT_ENV) $(UV) sync --frozen --project campaign/$(FLINT_DIR)
	cd $(FLINT_REPO) && $(FLINT_ENV) $(UV) run --frozen --project campaign/$(FLINT_DIR) \
		python campaign/$(FLINT_DIR)/histories.py
	cd $(FLINT_REPO) && TMPDIR=$(TMPDIR) ZIG_GLOBAL_CACHE_DIR=$(ZIG_GLOBAL_CACHE_DIR) \
		zig build tf-flint-run -Dcpu=apple_m1
	cd $(FLINT_REPO) && TMPDIR=$(TMPDIR) ZIG_GLOBAL_CACHE_DIR=$(ZIG_GLOBAL_CACHE_DIR) \
		zig build-exe zig/src/core/tokenizer/check.zig -O ReleaseSafe -femit-bin=$(FLINT_LOGITS)/tokenizer-check
	cd $(FLINT_REPO) && $(FLINT_PY) campaign/$(FLINT_DIR)/tokenizer_identity.py \
		campaign/$(FLINT_CORPUS) $(FLINT_LOGITS)/tokenizer-check $(FLINT_MODEL_DIR)/tokenizer.json
	cd $(FLINT_REPO) && $(FLINT_PY) campaign/$(FLINT_DIR)/run_tf.py \
		campaign/$(FLINT_CORPUS) $(FLINT_REPO)/zig-out/bin/tf-flint-run $(FLINT_MODEL_DIR) $(FLINT_LOGITS)/tf $(HEAVY_PY)
	cd $(FLINT_REPO) && $(HEAVY) $(FLINT_PY) campaign/$(FLINT_DIR)/oracle.py \
		campaign/$(FLINT_CORPUS) $(FLINT_MODEL_DIR) $(FLINT_LOGITS)/oracle 10
	cd $(FLINT_REPO) && $(HEAVY) $(FLINT_PY) campaign/$(FLINT_DIR)/oracle_chunked.py \
		campaign/$(FLINT_CORPUS) $(FLINT_MODEL_DIR) $(FLINT_LOGITS)/oracle-chunked 4096
	cd $(FLINT_REPO) && $(FLINT_PY) campaign/$(FLINT_DIR)/tokenizer_identity.py \
		campaign/$(FLINT_CORPUS) $(FLINT_LOGITS)/tokenizer-check $(FLINT_MODEL_DIR)/tokenizer.json
	cd $(FLINT_REPO) && $(HEAVY) $(FLINT_PY) campaign/$(FLINT_DIR)/compare.py \
		--histories campaign/$(FLINT_CORPUS) --tf $(FLINT_LOGITS)/tf --ref $(FLINT_LOGITS)/oracle \
		--out $(FLINT_LOGITS)/comparison.json --logits-out $(FLINT_LOGITS)/divergent \
		--other $(FLINT_LOGITS)/oracle-chunked \
		--oracle-note '$(FLINT_ORACLE)' --oracle-equivalence '$(FLINT_EQUIV)'
	cd $(FLINT_REPO) && $(FLINT_PY) campaign/$(FLINT_DIR)/trace.py \
		--corpus campaign/$(FLINT_CORPUS) --model-dir $(FLINT_MODEL_DIR) --logits $(FLINT_LOGITS) \
		--binary $(FLINT_REPO)/zig-out/bin/tf-flint-run --python $(FLINT_PY) \
		--oracle-script campaign/$(FLINT_DIR)/oracle_trace.py --heavy $(HEAVY_PY) \
		--out $(FLINT_LOGITS) --exact-histories debug-session-nan-loss,json-invoice-records
	cd $(FLINT_REPO) && $(FLINT_PY) campaign/$(FLINT_DIR)/report.py \
		$(FLINT_LOGITS)/comparison.json campaign/$(FLINT_CORPUS) campaign/$(FLINT_REPORT) \
		$(FLINT_LOGITS)/trace.json
