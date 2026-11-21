# Teacher-forced reference check for Inkling 2 Mini (campaign item I0).
#
# ref-check runs the CPU float32 reference (campaign/ref/inkling2_loop_ref.py)
# and llama.cpp at the pinned commit over the four short histories in
# campaign/ref/histories.json, then compares their logits. Every process that
# loads a full model runs under the campaign GPU lock, one at a time.

REF_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
CAMPAIGN_DIR := $(abspath $(REF_DIR)..)
MODEL ?= inkling2-mini

LLAMA_COMMIT := 40b7a546defe1f60edd1338c719552ffc58c89a0
LLAMA_TGZ := $(WORK)/src/llama.cpp@40b7a546defe/llama.cpp-$(LLAMA_COMMIT).tar.gz
LLAMA_SRC := $(WORK)/build/llama.cpp-src
LLAMA_BUILD := $(WORK)/build/llama.cpp
LLAMA_BIN := $(LLAMA_BUILD)/bin
LLAMA_LOGITS := $(WORK)/build/llama_logits
LLAMA_CMAKE := -DCMAKE_BUILD_TYPE=Release -DLLAMA_METAL=ON -DLLAMA_CURL=OFF \
	-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=ON

WEIGHTS_ROOT ?= $(HOME)/models
INKLING_DIR := $(WEIGHTS_ROOT)/Inkling-2-Mini@ed40e7da625f
GGUF := $(WORK)/build/inkling2-mini-f32.gguf
CONVERT_VENV := $(WORK)/venv-convert
REF_VENV := $(WORK)/venv-ref
REF_OUT := $(WORK)/ref/inkling2-mini
REF_UV := $(UV) run --frozen --project $(REF_DIR)
HISTORIES := $(REF_DIR)histories.json

.PHONY: ref-check ref-gguf ref-llama-build

ref-check: $(LLAMA_LOGITS) $(GGUF)
	@test "$(MODEL)" = "inkling2-mini" || { echo "ref-check supports MODEL=inkling2-mini only" >&2; exit 2; }
	@test -f $(INKLING_DIR)/model-00001-of-00001.safetensors || \
		{ echo "missing $(INKLING_DIR); run make -C campaign fetch NAMES=inkling-2-mini" >&2; exit 2; }
	@rm -rf $(REF_OUT) && mkdir -p $(REF_OUT)/ref $(REF_OUT)/llama
	lockf $(GPU_LOCK) env UV_PROJECT_ENVIRONMENT=$(REF_VENV) $(REF_UV) python $(REF_DIR)inkling2_loop_ref.py \
		--model $(INKLING_DIR) --histories $(HISTORIES) --out-dir $(REF_OUT)/ref \
		--expert-cache-gib 10
	@set -e; for name in $(shell jaq -r '.[].name' $(HISTORIES)); do \
		ids=$$(jaq -r --arg n "$$name" '.[] | select(.name == $$n) | .token_ids | @csv' $(HISTORIES) | tr -d '"'); \
		lockf $(GPU_LOCK) $(LLAMA_LOGITS) --model $(GGUF) --ids "$$ids" \
			--out $(REF_OUT)/llama/$$name.bin --threads 6 >> $(REF_OUT)/llama.log 2>&1; \
	done
	env UV_PROJECT_ENVIRONMENT=$(REF_VENV) $(REF_UV) python $(REF_DIR)compare_logits.py --histories $(HISTORIES) \
		--ref-dir $(REF_OUT)/ref --llama-dir $(REF_OUT)/llama --min-margin 0.1

# the pinned source tarball, through the campaign lock file
$(LLAMA_TGZ):
	$(MAKE) -C $(CAMPAIGN_DIR) fetch NAMES=llama.cpp

$(LLAMA_SRC)/.pinned: $(LLAMA_TGZ)
	rm -rf $(LLAMA_SRC)
	mkdir -p $(LLAMA_SRC)
	tar xzf $(LLAMA_TGZ) -C $(LLAMA_SRC) --strip-components=1
	touch $@

ref-llama-build: | $(LLAMA_SRC)/.pinned
	cmake -S $(LLAMA_SRC) -B $(LLAMA_BUILD) $(LLAMA_CMAKE)
	cmake --build $(LLAMA_BUILD) -j 10 --target llama-cli

$(LLAMA_BIN)/libllama.dylib: | $(LLAMA_SRC)/.pinned
	cmake -S $(LLAMA_SRC) -B $(LLAMA_BUILD) $(LLAMA_CMAKE)
	cmake --build $(LLAMA_BUILD) -j 10 --target llama-cli

$(LLAMA_LOGITS): $(REF_DIR)llama_logits.cpp $(LLAMA_BIN)/libllama.dylib | $(LLAMA_SRC)/.pinned
	c++ -std=c++17 -O2 -Wno-deprecated-declarations \
		-I$(LLAMA_SRC)/include -I$(LLAMA_SRC)/ggml/include $< -o $@ \
		-L$(LLAMA_BIN) -lllama -lggml -lggml-base -lggml-cpu -Wl,-rpath,$(LLAMA_BIN)

# The converter dequantizes NVFP4/MXFP8 to float32 without requantizing, so the
# GGUF keeps the checkpoint's exact decoded values. It needs its own torch venv.
$(CONVERT_VENV)/bin/python:
	$(UV) venv --python 3.12 $(CONVERT_VENV)
	$(UV) pip install --python $(CONVERT_VENV)/bin/python torch numpy pyyaml sentencepiece tqdm transformers

ref-gguf: $(GGUF)
$(GGUF): $(CONVERT_VENV)/bin/python | $(LLAMA_SRC)/.pinned
	cd $(LLAMA_SRC) && $(CONVERT_VENV)/bin/python convert_hf_to_gguf.py $(INKLING_DIR) \
		--outfile $(GGUF) --outtype f32
