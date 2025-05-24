#!/usr/bin/env bash

# Copyright 2025 Keqi Deng (University of Cambridge)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# ======= Default config =======
MUSTC_ROOT=""
ASR_SAVE_DIR=""
ST_SAVE_DIR=""
skip_asr=true
vocab_type=unigram
asr_vocab_size=5000
st_vocab_size=8000
langpair=en-de

# ======= Parse command-line options =======
. ./../tools/parse_options.sh || exit 1;

# ======= Validate inputs =======
if [[ -z "$MUSTC_ROOT" ]]; then
  echo "Error: MUSTC_ROOT must be set. Use --MUSTC_ROOT /path/to/mustc"
  exit 1
fi

# ======= Prepare ASR and ST data =======
if [ "$skip_asr" != "true" ]; then
  echo "Stage 1: Preparing ASR data..."
  python examples/speech_to_text/prep_mustc_data.py \
    --data-root ${MUSTC_ROOT} --task asr \
    --vocab-type ${vocab_type} --vocab-size ${asr_vocab_size}
else
  echo "Stage 1: Skipped ASR data preparation."
fi

echo "Stage 2: Preparing ST data..."
python examples/speech_to_text/prep_mustc_data.py \
  --data-root ${MUSTC_ROOT} --task st \
  --vocab-type ${vocab_type} --vocab-size ${st_vocab_size}

# ======= Train ASR model =======
if [ "$skip_asr" != "true" ]; then
  echo "Stage 3: Training ASR model..."
  fairseq-train ${MUSTC_ROOT}/${langpair} \
    --config-yaml config_asr.yaml --train-subset train_asr --valid-subset dev_asr \
    --save-dir ${ASR_SAVE_DIR} --num-workers 4 --max-tokens 40000 --max-update 100000 \
    --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
    --arch s2t_transformer_m --optimizer adam --lr 1e-3 --lr-scheduler inverse_sqrt \
    --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8
else
  echo "Stage 3: Skipped ASR model training."
fi

echo "Done."

