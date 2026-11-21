PYTHONDONTWRITEBYTECODE=1

.PHONY: evidence-tests

evidence-tests:
	PYTHONDONTWRITEBYTECODE=1 python3 $(CURDIR)/tests/test_evidence.py
