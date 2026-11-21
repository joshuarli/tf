MODEL ?= flint
SET ?= dev
BIN ?= $(WORK)/wt/baseline/zig-out/native/bin/tensorfold-native
QUALIFY_RESULT ?=
QUALIFY_BASELINE ?=

.PHONY: quality-flint quality-inkling2-mini quality-cases quality-cases-check quality-qualify

quality-flint: quality-grammar
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --project $(CURDIR) python $(CURDIR)/quality/run.py \
		--model flint --set $(SET) --bin '$(BIN)'

quality-inkling2-mini: quality-grammar
	@echo "$(call serve_preview,inkling2-mini,2)"
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --project $(CURDIR) python $(CURDIR)/quality/run.py \
		--model inkling2-mini --set $(SET) --bin '$(BIN)'

quality-qualify:
	@test -n '$(QUALIFY_RESULT)' -a -n '$(QUALIFY_BASELINE)' || { \
		echo "quality-qualify needs QUALIFY_RESULT=<run result.json> QUALIFY_BASELINE=<frozen baseline result.json>" >&2; \
		exit 2; }
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --project $(CURDIR) python $(CURDIR)/quality/qualify.py \
		--result '$(QUALIFY_RESULT)' --baseline '$(QUALIFY_BASELINE)'

quality-cases:
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --project $(CURDIR) python $(CURDIR)/quality/build_cases.py
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --project $(CURDIR) python $(CURDIR)/quality/build_retrieval.py
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --project $(CURDIR) python $(CURDIR)/quality/manifest.py

quality-cases-check:
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --project $(CURDIR) python $(CURDIR)/quality/build_cases.py --check
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --project $(CURDIR) python $(CURDIR)/quality/build_retrieval.py --check
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --project $(CURDIR) python $(CURDIR)/quality/manifest.py --check
