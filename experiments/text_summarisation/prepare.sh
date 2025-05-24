#!/usr/bin/env bash

# Copyright 2025 Keqi Deng (University of Cambridge)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

set -e

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=30000

# Initialise directory
OUTDIR=xsum_data
prep=$OUTDIR
tmp=$prep/tmp
mkdir -p $tmp $prep

####################################
### Step 1: Load the data via HuggingFace
####################################
echo "Loading XSum data..."
python -c "
from datasets import load_dataset
import json

dataset = load_dataset('xsum')
splits = ['train', 'validation', 'test']

for split in splits:
    with open(f'$tmp/{split}.raw.source', 'w') as src_f, \\
         open(f'$tmp/{split}.raw.target', 'w') as tgt_f:
        for item in dataset[split]:
            article = ' '.join(item['document'].split())
            summary = ' '.join(item['summary'].split())
            src_f.write(article + '\n')
            tgt_f.write(summary + '\n')
"

####################################
### Step 2: Tokenisation
####################################
echo "Tokenising..."
for split in train validation test; do
    for l in source target; do
        perl $NORM_PUNC en < $tmp/$split.raw.$l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 8 -a -l en > $tmp/$split.tok.$l
    done
done

####################################
### Step 3: BPE Training and Apply
####################################
echo "Training joint BPE..."
cat $tmp/train.tok.source $tmp/train.tok.target > $tmp/train.combined
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $tmp/train.combined > $prep/code

echo "Applying BPE..."
for split in train validation test; do
    for l in source target; do
        python $BPEROOT/apply_bpe.py -c $prep/code < $tmp/$split.tok.$l > $tmp/$split.bpe.$l
    done
done

for split in train validation test; do
  ln -sf $(realpath $tmp/${split}.bpe.source) $prep/${split}.source
  ln -sf $(realpath $tmp/${split}.bpe.target) $prep/${split}.target
done

####################################
### Step 4: Fairseq binarization
####################################
echo "Binarizing..."
fairseq-preprocess \
    --source-lang source \
    --target-lang target \
    --trainpref $prep/train \
    --validpref $prep/validation \
    --testpref $prep/test \
    --destdir $prep/bin \
    --workers 60 \
    --joined-dictionary

echo "Done! Final data directory:"
tree $prep/bin

