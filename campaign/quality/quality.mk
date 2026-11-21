MODEL ?= flint
SET ?= dev
BIN ?= $(WORK)/wt/baseline/zig-out/native/bin/tensorfold-native

.PHONY: quality-flint quality-cases quality-cases-check

quality-flint: quality-grammar
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --project $(CURDIR) python $(CURDIR)/quality/run.py \
		--model $(MODEL) --set $(SET) --bin '$(BIN)'

quality-cases:
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --project $(CURDIR) python $(CURDIR)/quality/build_cases.py
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --project $(CURDIR) python $(CURDIR)/quality/build_retrieval.py
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --project $(CURDIR) python $(CURDIR)/quality/manifest.py

quality-cases-check:
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --project $(CURDIR) python $(CURDIR)/quality/build_cases.py --check
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --project $(CURDIR) python $(CURDIR)/quality/build_retrieval.py --check
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --project $(CURDIR) python $(CURDIR)/quality/manifest.py --check
