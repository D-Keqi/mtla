#!/usr/bin/env bash

# Copyright 2025 Keqi Deng (University of Cambridge)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# ======= Default config =======
MUSTC_ROOT=""
ST_SAVE_DIR=""
CHECKPOINT_FILENAME=avg_10_checkpoint.pt

# ======= Option parsing =======
. ./../tools/parse_options.sh || exit 1;

# ======= Run inference =======
fairseq-generate ${MUSTC_ROOT}/en-de \
  --config-yaml config_st.yaml --gen-subset tst-COMMON_st --task speech_to_text \
  --path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
  --max-tokens 50000 --beam 50 --scoring sacrebleu
