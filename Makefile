SHELL := /bin/bash

# Auto-detect platform and architecture
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
IS_WINDOWS := $(filter MINGW% MSYS% CYGWIN%,$(UNAME_S))

# Cross-platform defaults; Windows block below overrides these.
LINK := ln -sf
EXE :=
PICKBRAIN_DOWNLOAD := download

# Determine features and flags based on platform
ifeq ($(UNAME_S),Darwin)
  ifeq ($(UNAME_M),arm64)
    # Apple Silicon: Metal GPU + Accelerate BLAS
    CLI_FEATURES := t5-quantized,metal,progress
    NAPI_FEATURES := t5-quantized,metal,napi
    RUSTFLAGS_EXTRA :=
    TARGET := aarch64-apple-darwin
  else
    # Intel Mac: CPU-only with FBGEMM + hybrid-dequant
    CLI_FEATURES := t5-quantized,fbgemm,hybrid-dequant,progress
    NAPI_FEATURES := t5-quantized,fbgemm,hybrid-dequant,napi
    RUSTFLAGS_EXTRA := -C target-feature=+avx2,+fma
    TARGET := x86_64-apple-darwin
  endif
  PICKBRAIN_FEATURES := $(CLI_FEATURES),embed-assets
else ifeq ($(UNAME_S),Linux)
  CLI_FEATURES := t5-quantized,fbgemm,hybrid-dequant,progress
  NAPI_FEATURES := t5-quantized,fbgemm,napi
  RUSTFLAGS_EXTRA :=
  TARGET :=
else ifneq (,$(IS_WINDOWS))
  # Native Windows x86_64 via Git Bash / MSYS2 / Cygwin. Uses OpenVINO backend
  # (README §Feature flags). Uses host Rust target (x86_64-pc-windows-msvc).
  CLI_FEATURES := t5-openvino,fbgemm,progress
  NAPI_FEATURES := t5-openvino,fbgemm,napi
  RUSTFLAGS_EXTRA := -C target-feature=+avx2
  TARGET :=
  PICKBRAIN_FEATURES := $(CLI_FEATURES),embed-assets
  PICKBRAIN_DOWNLOAD := ovdownload
  LINK := cp -f
  EXE := .exe
endif

# Binary paths
ifdef TARGET
  CLI_BIN := target/$(TARGET)/release/warp-cli$(EXE)
  PICKBRAIN_BIN := target/$(TARGET)/release/examples/pickbrain$(EXE)
  BUILD_TARGET := --target $(TARGET)
else
  CLI_BIN := target/release/warp-cli$(EXE)
  PICKBRAIN_BIN := target/release/examples/pickbrain$(EXE)
  BUILD_TARGET :=
endif

export RUSTFLAGS += $(RUSTFLAGS_EXTRA)

# === Python environment ===

assets:
	mkdir -p assets

ifneq (,$(IS_WINDOWS))
# Windows (OpenVINO path): quantize-openvino.py imports downloadweights, whose
# module-level code downloads the HF snapshot and writes assets/config.json,
# assets/tokenizer.json. So one script produces all four ovdownload assets and
# we skip the uv venv entirely. Requires system Python with openvino, nncf,
# torch, transformers, huggingface_hub, safetensors installed.
assets/config.json assets/tokenizer.json assets/xtr-ov-int4.bin assets/xtr-ov-int4.xml: | assets
	python quantize-openvino.py
else
env/pyvenv.cfg:
	uv venv env

env/bin/transformers: env/pyvenv.cfg
	(source env/*/activate && uv pip install -r requirements.txt)

assets/config.json assets/tokenizer.json xtr.safetensors: env/bin/transformers | assets
	(source env/*/activate && python downloadweights.py)

assets/xtr.gguf: xtr.safetensors | assets
	cargo run -p quantize-tool xtr.safetensors assets/xtr.gguf

assets/xtr-ov-int4.bin assets/xtr-ov-int4.xml:
	python quantize-openvino.py
endif

download: assets assets/config.json assets/tokenizer.json assets/xtr.gguf

ovdownload: assets/config.json assets/tokenizer.json assets/xtr-ov-int4.bin assets/xtr-ov-int4.xml

# === Build targets ===

warp-cli: download
	cargo build --release $(BUILD_TARGET) --features $(CLI_FEATURES) --bin warp-cli
	$(LINK) $(CLI_BIN) ./warp-cli$(EXE)

pickbrain: $(PICKBRAIN_DOWNLOAD)
	cargo build --release $(BUILD_TARGET) --features $(PICKBRAIN_FEATURES) --example pickbrain
	$(LINK) $(PICKBRAIN_BIN) ./pickbrain$(EXE)

pickbrain-install: pickbrain
	mkdir -p $(HOME)/bin $(HOME)/.claude/skills/pickbrain $(HOME)/.codex/skills/pickbrain $(HOME)/.pickbrain/assets
	ln -f $(realpath pickbrain$(EXE)) $(HOME)/bin/pickbrain$(EXE)
	rm -f $(HOME)/.claude/skills/pickbrain/skill.md $(HOME)/.codex/skills/pickbrain/skill.md
	cp skills/pickbrain/SKILL.md $(HOME)/.claude/skills/pickbrain/SKILL.md
	cp skills/pickbrain-codex/SKILL.md $(HOME)/.codex/skills/pickbrain/SKILL.md
ifneq (,$(IS_WINDOWS))
	cp assets/xtr-ov-int4.xml assets/xtr-ov-int4.bin assets/config.json assets/tokenizer.json $(HOME)/.pickbrain/assets/
endif

macintel:
	RUSTFLAGS='-C target-cpu=haswell' cargo build --release --target x86_64-apple-darwin --features t5-quantized,fbgemm,hybrid-dequant,progress

winintel: ovdownload
	RUSTFLAGS='-C target-feature=+avx2' cargo xwin build --release --target x86_64-pc-windows-msvc --features t5-openvino,fbgemm,progress

ifdef TARGET
  LIB_BIN := target/$(TARGET)/release/libwitchcraft.dylib
else
  LIB_BIN := target/release/libwitchcraft.dylib
endif

module:
	cargo build --release --target aarch64-apple-darwin --features t5-quantized,metal,napi
	cargo build --release --target x86_64-apple-darwin --features t5-quantized,fbgemm,hybrid-dequant,napi
	lipo -create target/aarch64-apple-darwin/release/libwitchcraft.dylib target/x86_64-apple-darwin/release/libwitchcraft.dylib -output target/release/warp-macos-universal.node
	ln -sf target/release/warp-macos-universal.node warp.node

test: download
	RUST_LOG=debug cargo llvm-cov nextest --release --features napi,$(CLI_FEATURES) --lcov --output-path lcov.info
	genhtml lcov.info

bench:
	cargo run -p t5-bench --release --features hybrid-dequant,ov,fbgemm

# === Dataset targets ===

nfcorpus: warp-cli
	rm -rf mydb.sqlite*
	$(CLI_BIN) readcsv datasets/nfcorpus.tsv
	$(CLI_BIN) embed
	$(CLI_BIN) index

nfcorpus-score: warp-cli
	$(CLI_BIN) hybridcsv $(HOME)/src/xtr-warp/beir/nfcorpus/questions.test.tsv warp-results.txt

run: module
	ln -sf target/release/warp-macos-universal.node warp.node
	node index.cjs

.PHONY: download build warp-cli pickbrain pickbrain-install module win test bench nfcorpus nfcorpus-score run
