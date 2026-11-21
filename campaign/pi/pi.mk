UV ?= ~/.local/bin/uv
PI_MODEL ?= flint
PI_ONLY ?=
PI_BIN ?=
QUALIFY_RESULT ?=
QUALIFY_BASELINE ?=

.PHONY: pi-harness-flint pi-qualify

pi-harness-flint:
	PI_BIN='$(PI_BIN)' $(UV) run --project $(CURDIR) python $(CURDIR)/pi/run.py --model flint \
		--bin '$(BIN)' --pi-model $(PI_MODEL) $(if $(PI_ONLY),--only '$(PI_ONLY)',)

pi-qualify:
	@test -n '$(QUALIFY_RESULT)' || { \
		echo "pi-qualify needs QUALIFY_RESULT=<pi result.json> (QUALIFY_BASELINE=<frozen result.json> for loss)" >&2; \
		exit 2; }
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --project $(CURDIR) python $(CURDIR)/pi/qualify.py \
		--result '$(QUALIFY_RESULT)' $(if $(QUALIFY_BASELINE),--baseline '$(QUALIFY_BASELINE)',)
