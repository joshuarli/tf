XGRAMMAR_REPO := https://github.com/mlc-ai/xgrammar
XGRAMMAR_TAG := v0.2.8
XGRAMMAR_DIR ?= $(WORK)/src/xgrammar-v0.2.8
BASELINE_WT ?= $(WORK)/wt/baseline
GRAMMAR_LIB := $(BASELINE_WT)/zig-out/native/lib/libtfgrammar.dylib

.PHONY: quality-xgrammar quality-grammar

$(XGRAMMAR_DIR):
	mkdir -p $(dir $(XGRAMMAR_DIR))
	git clone --branch $(XGRAMMAR_TAG) --recurse-submodules $(XGRAMMAR_REPO) $(XGRAMMAR_DIR)

quality-xgrammar: | $(XGRAMMAR_DIR)
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --project $(CURDIR) python $(CURDIR)/quality/verify_pins.py --dir $(XGRAMMAR_DIR)

quality-grammar: quality-xgrammar
	cd $(BASELINE_WT) && zig build grammar -Dxgrammar=$(XGRAMMAR_DIR)
	test -f $(GRAMMAR_LIB)
	@echo "built $(GRAMMAR_LIB)"
