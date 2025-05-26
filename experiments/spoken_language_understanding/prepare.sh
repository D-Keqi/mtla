#!/usr/bin/env bash

# Copyright 2025 Keqi Deng (University of Cambridge)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

set -e
set -u
set -o pipefail

train_set="train"
valid_set="devel"
test_sets="test devel"

# ======= Preprocess data following ESPnet =======
./espnet.sh \
    --lang en --stage 0 --stop-stage 5 \
    --ngpu 1 \
    --use_lm false \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --nbpe 5000 \
    --token_type word\
    --feats_type raw\
    --audio_format "flac.ark" \
    --max_wav_duration 30 \
    --feats_normalize utterance_mvn\
    --inference_nj 8 \
    --inference_asr_model valid.acc.ave_10best.pth\
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"

data_dir=slurp_data
if [ ! -d ${data_dir} ]; then
  mkdir -p ${data_dir}
fi

# ======= Convert Kaldi-style data form into TSV manifest =======
python ./local/kaldi_to_tsv.py \
  --base_dir ./dump/raw \
  --output_dir ${data_dir} \
  --subsets test devel train_sp

# ======= Generate vocab file =======
tail -n +3 ./data/en_token_list/word/tokens.txt | head -n -1 | awk '{print $0, 1}' > ${data_dir}/vocab.txt

# ======= Create a symbolic link to config.yaml =======
ln -s local/config.yaml ${data_dir}/.

