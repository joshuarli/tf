UV ?= ~/.local/bin/uv

.PHONY: evidence-tests

evidence-tests:
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --project $(CURDIR) python $(CURDIR)/tests/test_evidence.py
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --project $(CURDIR) python $(CURDIR)/tests/test_inkling_selection.py
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --project $(CURDIR) python $(CURDIR)/tests/test_context_cap.py
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --project $(CURDIR) python $(CURDIR)/tests/test_suite.py
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --project $(CURDIR) python $(CURDIR)/fx/test_subagents.py
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --project $(CURDIR) python $(CURDIR)/quality/test_reasoning_replay.py
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --project $(CURDIR) python $(CURDIR)/comparators/nanoattention/test_inkling.py
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --project $(CURDIR) python $(CURDIR)/comparators/nanoattention/test_setup.py
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --project $(CURDIR)/ref python $(CURDIR)/parity/inkling2-mini/test_parity_gate.py
