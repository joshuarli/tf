OMLX_WORK := $(WORK)
OMLX_ENV := env 'PATH=$(WORK)/venv-omlx/bin:$(PATH)' UV_CACHE_DIR=$(WORK)/uv-cache UV_PYTHON_INSTALL_DIR=$(WORK)/uv-python XDG_CACHE_HOME=$(WORK)/xdg-cache HF_HOME=$(WORK)/hf-home TMPDIR=$(WORK)/tmp PYTHONDONTWRITEBYTECODE=1 OMLX_WITH_CUSTOM_KERNEL=0
OMLX_PYTHON := $(WORK)/venv-omlx/bin/python
OMLX_INTERPRETER := $(HOME)/.local/share/uv/python/cpython-3.13-macos-aarch64-none/bin/python3.13
OMLX_PROMPT ?= code-1024
OMLX_MODE ?= both
OMLX_MODEL ?= $(HOME)/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd

.PHONY: omlx-setup omlx-tests omlx-smoke omlx-cell

omlx-setup:
	mkdir -p $(WORK)/tmp $(WORK)/xdg-cache $(WORK)/hf-home $(WORK)/uv-cache
	$(OMLX_ENV) $(OMLX_INTERPRETER) comparators/omlx/setup.py --validate-lock
	$(OMLX_ENV) $(MAKE) fetch NAMES=omlx
	$(OMLX_ENV) $(MAKE) fetch NAMES=inkling-2-mini-oq4e-mtp
	$(OMLX_ENV) $(OMLX_INTERPRETER) comparators/omlx/setup.py
	$(OMLX_ENV) UV_PROJECT_ENVIRONMENT=$(WORK)/venv-omlx UV_PYTHON_DOWNLOADS=never $(UV) sync --locked --require-build-hashes --project comparators/omlx --python $(OMLX_INTERPRETER)
	$(OMLX_ENV) $(OMLX_PYTHON) comparators/omlx/setup.py --verify-install

omlx-tests:
	$(OMLX_ENV) python3 -B -m unittest discover -s comparators/omlx -p 'test_*.py'

omlx-smoke:
	$(OMLX_ENV) HF_HUB_OFFLINE=1 INKLING_LOAD_LIMIT_GB=26 GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60 $(OMLX_PYTHON) comparators/omlx/run.py --smoke

omlx-cell:
	$(OMLX_ENV) HF_HUB_OFFLINE=1 INKLING_LOAD_LIMIT_GB=26 GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60 $(OMLX_PYTHON) comparators/omlx/run.py --prompt '$(OMLX_PROMPT)' --mode '$(OMLX_MODE)' --model '$(OMLX_MODEL)'
