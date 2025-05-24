# Text Summarisation Example on XSum

## Data Preparation
Download the data and BPE training:
```bash
# Download XSum data, Tokenisation, BPE Training and Apply, and Fairseq binarization
bash prepare.sh
```
The processed data is saved in the `xsum_data` folder.

## Model Training
Second, train the model on the processed XSum data. By default, MTLA is used as the decoder-only self-attention structure, i.e. `--arch transformer_decoder_only_MTLA_cross_xm`.
To train other models:
- MHA: `--arch transformer_decoder_only_roformer_cross_xm`
- MLA: `--arch transformer_decoder_only_MLA_cross_xm`
```bash
bash train.sh --SUM_SAVE_DIR /path/to/save_text_sum_model
```

Third, after training is completed, apply the model averaging technique.
```bash
CHECKPOINT_FILENAME=avg_10_checkpoint.pt
python ./../tools/fairseq/scripts/average_checkpoints.py \
  --inputs ${SUM_SAVE_DIR} --num-best-checkpoints 10 \
  --output "${SUM_SAVE_DIR}/${CHECKPOINT_FILENAME}"
```

## Model Inference
Finally, run inference. The summarisation quality (ROUGE scores: ROUGE-1, ROUGE-2, and ROUGE-L), inference time, and GPU memory usage will all be reported.
```bash
bash infer.sh --SUM_SAVE_DIR /path/to/save_text_sum_model
```
