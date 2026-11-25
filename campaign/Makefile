UV ?= ~/.local/bin/uv
NAMES ?=
BASELINE_REV := fa13df5dd1fdd23ada775510a464c9aa03a39e09
MAIN_ROOT := $(abspath $(shell git rev-parse --path-format=absolute --git-common-dir)/..)
WORK := $(MAIN_ROOT)/.work
BASELINE_WT := $(WORK)/wt/baseline
GPU_LOCK := $(WORK)/gpu.lock
HEAVY_PY := $(CURDIR)/heavy.py
HEAVY := python3 $(HEAVY_PY) --
LOGS := $(WORK)/logs/baseline
export TMPDIR := $(WORK)/tmp
export ZIG_GLOBAL_CACHE_DIR := $(WORK)/zig-global-cache
export UV_CACHE_DIR := $(WORK)/uv-cache
export HF_HOME := $(WORK)/hf-home
ZIG_PINNED := $(shell jaq -r '.inputs[] | select(.role == "toolchain") | .version' bootstrap.lock.json)

.PHONY: fetch host check-zig baseline-toolchain build-baseline baseline-timings

fetch:
	$(UV) run --project . python fetch.py $(if $(NAMES),--only $(NAMES),)

host:
	$(UV) run --project . python host.py

check-zig:
	@test "$$(zig version)" = "$(ZIG_PINNED)" || { echo "zig $$(zig version) does not match pinned $(ZIG_PINNED)" >&2; exit 1; }

$(BASELINE_WT):
	mkdir -p $(TMPDIR)
	git worktree add --detach $(BASELINE_WT) $(BASELINE_REV)

baseline-toolchain: | $(BASELINE_WT)
	printf '%s\n' "$(ZIG_PINNED)" > $(BASELINE_WT)/.zig-version
	sed 's/minimum_zig_version = "[^"]*"/minimum_zig_version = "$(ZIG_PINNED)"/' $(BASELINE_WT)/build.zig.zon > $(BASELINE_WT)/build.zig.zon.tmp
	mv $(BASELINE_WT)/build.zig.zon.tmp $(BASELINE_WT)/build.zig.zon
	test "$$(git -C $(BASELINE_WT) rev-parse HEAD)" = "$(BASELINE_REV)"
	test "$$(git -C $(BASELINE_WT) diff --name-only | tr '\n' ' ')" = ".zig-version build.zig.zon "

build-baseline: check-zig baseline-toolchain
	cd $(BASELINE_WT) && zig build native -Dcpu=apple_m1 -Dversion=1.0.0
	sha256sum $(BASELINE_WT)/zig-out/native/bin/tensorfold-native | tee $(WORK)/baseline.sha256

baseline-timings: check-zig baseline-toolchain
	mkdir -p $(LOGS)
	rm -rf $(BASELINE_WT)/.zig-cache $(BASELINE_WT)/zig-out $(ZIG_GLOBAL_CACHE_DIR)/o
	cd $(BASELINE_WT) && /usr/bin/time -p zig build native -Dcpu=apple_m1 -Dversion=1.0.0 2> $(LOGS)/native-cold.log
	cd $(BASELINE_WT) && /usr/bin/time -p zig build native -Dcpu=apple_m1 -Dversion=1.0.0 2> $(LOGS)/native-warm.log
	cd $(BASELINE_WT) && /usr/bin/time -p zig build test-flint-draft-host > $(LOGS)/flint-draft-host.log 2>&1
	cd $(BASELINE_WT) && $(HEAVY) /usr/bin/time -p zig build test test-golden -Dcpu=apple_m1 --summary all > $(LOGS)/test-golden.log 2>&1 || echo "test test-golden exited $$?" >> $(LOGS)/test-golden.log

.PHONY: parity quality pi-harness

parity: parity-$(MODEL)

quality: quality-$(MODEL)

pi-harness: pi-harness-$(MODEL)

include $(sort $(wildcard */*.mk */*/*.mk))
