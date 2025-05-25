#!/usr/bin/env bash

# Copyright 2025 Keqi Deng (University of Cambridge)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# ======= Default config =======
SUM_SAVE_DIR=""

# ======= Option parsing =======
. ./../tools/utils/parse_options.sh || exit 1;

# ======= Check required =======
if [[ -z "$SUM_SAVE_DIR" ]]; then
  echo "Please set SUM_SAVE_DIR."
  exit 1
fi

# ======= Run training =======
fairseq-train \
    xsum_data/bin \
    --arch transformer_decoder_only_MTLA_cross_xm --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 2e-4 --lr-scheduler inverse_sqrt --warmup-updates 15000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 40000 --update-freq 2 --max-update 60000 \
    --save-dir ${SUM_SAVE_DIR} \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --skip-invalid-size-inputs-valid-test \
    --keep-last-epochs 1 --keep-best-checkpoints 10 --maximize-best-checkpoint-metric --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    2>&1 | tee -a ${SUM_SAVE_DIR}/train.log
