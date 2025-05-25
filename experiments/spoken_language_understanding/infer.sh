#!/usr/bin/env bash

# Copyright 2025 Keqi Deng (University of Cambridge)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# ======= Default config =======
LS_ROOT=""
SAVE_DIR=""
CHECKPOINT_FILENAME=avg_10_checkpoint.pt

# ======= Option parsing =======
. ./../tools/utils/parse_options.sh || exit 1;

# ======= Run inference =======
fairseq-generate ${LS_ROOT} \
  --config-yaml config.yaml --gen-subset test --task speech_to_text \
  --path ${SAVE_DIR}/${CHECKPOINT_FILENAME} \
  --max-tokens 130000 --beam 10 --scoring ic_accuracy
