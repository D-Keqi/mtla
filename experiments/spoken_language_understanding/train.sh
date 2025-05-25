#!/usr/bin/env bash

# Copyright 2025 Keqi Deng (University of Cambridge)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# ======= Default config =======
DATA_ROOT=""
SAVE_DIR=""

# ======= Option parsing =======
. ./../tools/utils/parse_options.sh || exit 1;

# ======= Check required =======
if [[ -z "$DATA_ROOT" || -z "$SAVE_DIR" ]]; then
  echo "Please set DATA_ROOT and SAVE_DIR."
  exit 1
fi

# ======= Run training =======
fairseq-train ${DATA_ROOT} --save-dir ${SAVE_DIR} \
  --distributed-world-size 1 --config-yaml config.yaml --train-subset train --valid-subset devel \
  --num-workers 4 --max-tokens 18000 --max-update 300000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy_with_ctc_ic --ctc-weight 1.0 --label-smoothing 0.1 --report-accuracy \
  --arch conformer_decoder_only_MTLA_cross_xm --share-decoder-input-output-embed --dropout 0.3 --conv-kernel-sizes "5" \
  --optimizer adam --lr 2e-4 --lr-scheduler inverse_sqrt --warmup-updates 50000 \
  --clip-norm 10.0 --seed 1 --update-freq 1 --keep-last-epochs 1 --keep-best-checkpoints 10 --maximize-best-checkpoint-metric --best-checkpoint-metric accuracy \
  2>&1 | tee -a ${SAVE_DIR}/train.log
