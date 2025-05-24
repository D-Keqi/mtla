#!/usr/bin/env bash

# Copyright 2025 Keqi Deng (University of Cambridge)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# ======= Default config =======
MUSTC_ROOT=""
ST_SAVE_DIR=""
ASR_SAVE_DIR=""

# ======= Option parsing =======
. ./../tools/parse_options.sh || exit 1;

# ======= Check required =======
if [[ -z "$MUSTC_ROOT" || -z "$ST_SAVE_DIR" ]]; then
  echo "Please set MUSTC_ROOT and ST_SAVE_DIR."
  exit 1
fi

# ======= Run training =======
fairseq-train ${MUSTC_ROOT}/en-de \
  --distributed-world-size 1 --config-yaml config_st.yaml --train-subset train_st --valid-subset dev_st \
  --save-dir ${ST_SAVE_DIR} --num-workers 4 --max-tokens 40000 --max-update 100000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
  --arch s2t_decoder_only_MTLA_cross_xm --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8 --keep-last-epochs 1 --keep-best-checkpoints 10 --maximize-best-checkpoint-metric --best-checkpoint-metric accuracy\
  --load-pretrained-encoder-from ${ASR_SAVE_DIR}\
  2>&1 | tee -a ${ST_SAVE_DIR}/train.log
