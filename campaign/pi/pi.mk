UV ?= ~/.local/bin/uv
PI_MODEL ?= flint
PI_ONLY ?=
PI_BIN ?=

.PHONY: pi-harness-flint

pi-harness-flint:
	PI_BIN='$(PI_BIN)' $(UV) run --project $(CURDIR) python $(CURDIR)/pi/run.py --model flint \
		--bin '$(BIN)' --pi-model $(PI_MODEL) $(if $(PI_ONLY),--only '$(PI_ONLY)',)
