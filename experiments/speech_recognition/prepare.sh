#!/usr/bin/env bash

# Copyright 2025 Keqi Deng (University of Cambridge)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
set -e
set -u
set -o pipefail

# You may set 'mic' to:
#  ihm [individual headset mic- the default which gives best results]
#  sdm1 [single distant microphone- the current script allows you only to select
#        the 1st of 8 microphones]
#  mdm8 [multiple distant microphones-- currently we only support averaging over
#       the 8 source microphones].
# ... by calling this script as, for example,
# ./run.sh --mic sdm1
# ./run.sh --mic mdm8
mic=ihm

train_set=${mic}_train
valid_set=${mic}_dev
test_sets="${mic}_eval ${mic}_dev"

speed_perturb_factors="0.9 1.0 1.1"

# ======= Preprocess data following ESPnet =======
./espnet.sh \
    --lang en --stage 0 --stop-stage 4 \
    --local_data_opts "--mic ${mic}" \
    --nbpe 100 \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"

data_dir=ami_data
if [ ! -d ${data_dir} ]; then
  mkdir -p ${data_dir}
fi

# ======= Convert Kaldi-style data form into TSV manifest =======
python ./local/kaldi_to_tsv.py \
  --base_dir ./dump/raw \
  --output_dir ${data_dir} \
  --subsets ihm_eval ihm_train_sp ihm_dev

# ======= Generate BPE vocab using Fairseq toolkit =======
python ./local/generate_vocab.py \
  --text-file ./dump/raw/${train_set}_sp/text \
  --output-prefix ${data_dir}/spm_unigram100 \
  --vocab-type bpe \
  --vocab-size 100
