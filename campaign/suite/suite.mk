TIER ?= edit
NAMES ?=
AREAS ?=
SKIP ?=
SUITE := python3 -I suite/suite.py
SUITE_SEL = --tier $(TIER) --names '$(NAMES)' --areas '$(AREAS)' --skip '$(SKIP)'

.PHONY: bench-list bench-check bench-suite bench-dry bench-full

bench-list:
	@$(SUITE) list

bench-check:
	@$(SUITE) check $(SUITE_SEL)

bench-dry:
	@$(SUITE) run --dry-run $(SUITE_SEL)

bench-suite:
	@$(SUITE) run $(SUITE_SEL)

bench-full:
	@test "$(FULL)" = requested || { echo "bench-full needs FULL=requested: it runs every tier, including the 48-case quality set" >&2; exit 2; }
	@$(SUITE) run --tier full --full-requested --names '$(NAMES)' --areas '$(AREAS)' --skip '$(SKIP)'
