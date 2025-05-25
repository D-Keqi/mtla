#!/usr/bin/env bash

# Copyright 2025 Keqi Deng (University of Cambridge)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# ======= Default config =======
SUM_SAVE_DIR=""
CHECKPOINT_FILENAME=avg_10_checkpoint.pt

# ======= Option parsing =======
. ./../tools/utils/parse_options.sh || exit 1;

# ======= Run inference =======
fairseq-generate xsum_data/bin \
  --path ${SUM_SAVE_DIR}/${CHECKPOINT_FILENAME} \
  --max-tokens 70000 --beam 10 --scoring rouge_l --skip-invalid-size-inputs-valid-test --remove-bpe --temperature 0.9
