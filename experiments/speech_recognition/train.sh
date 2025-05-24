#!/usr/bin/env bash

# Copyright 2025 Keqi Deng (University of Cambridge)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# ======= Default config =======
DATA_ROOT=""
SAVE_DIR=""

# ======= Option parsing =======
. ./../tools/parse_options.sh || exit 1;

# ======= Check required =======
if [[ -z "$DATA_ROOT" || -z "$SAVE_DIR" ]]; then
  echo "Please set DATA_ROOT and SAVE_DIR."
  exit 1
fi

# ======= Run training =======
fairseq-train ${DATA_ROOT} --save-dir ${SAVE_DIR} \
  --distributed-world-size 1 --config-yaml config.yaml --train-subset train_sp --valid-subset dev \
  --num-workers 4 --max-tokens 7500000 --max-update 100000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy_with_ctc --label-smoothing 0.1 --report-accuracy \
  --arch s2t_decoder_only_MTLA_cross_xm --conv-kernel-sizes "5" --share-decoder-input-output-embed --use-audio-input \
  --optimizer adam --lr 2e-4 --lr-scheduler inverse_sqrt --warmup-updates 15000 --max-source-positions 999999 \
  --clip-norm 1.0 --seed 1 --update-freq 4 --keep-last-epochs 1 --keep-best-checkpoints 10 --maximize-best-checkpoint-metric --best-checkpoint-metric accuracy \
  2>&1 | tee -a ${SAVE_DIR}/train.log
