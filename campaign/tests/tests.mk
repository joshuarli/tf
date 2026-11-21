UV ?= ~/.local/bin/uv

.PHONY: evidence-tests

evidence-tests:
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --project $(CURDIR) python $(CURDIR)/tests/test_evidence.py
