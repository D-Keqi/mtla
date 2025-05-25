# Speech Recognition Example on AMI

## Data Preparation
Download the data, process it following [ESPnet](https://github.com/espnet/espnet), convert it into TSV manifests, and then train a BPE tokenizer following [Fairseq](https://github.com/facebookresearch/fairseq) style.
```bash
bash prepare.sh
```
The processed data is saved in the `ami_data` folder.

## Model Training
Second, train the model on the processed AMI data. By default, MTLA is used as the decoder-only self-attention structure, i.e. `--arch s2t_decoder_only_MTLA_cross_xm`.
To train other models:
- MHA: `--arch s2t_decoder_only_roformer_cross_xm`
- MLA: `--arch s2t_decoder_only_MLA_cross_xm`
```bash
bash train.sh --DATA_ROOT ami_data --SAVE_DIR /path/to/save_asr_model
```

Third, after training is completed, apply the model averaging technique.
```bash
CHECKPOINT_FILENAME=avg_10_checkpoint.pt
python ./../tools/fairseq/scripts/average_checkpoints.py \
  --inputs ${SAVE_DIR} --num-best-checkpoints 10 \
  --output "${SAVE_DIR}/${CHECKPOINT_FILENAME}"
```

## Model Inference
Fourth, pre-extract and save the SSL features before inference:
``` bash
python local/wav2ssl.py \
  --checkpoint /path/to/save_asr_model/avg_10_checkpoint.pt \
  --data-dir ami_data \
  --input-tsv ami_data/ihm_eval.tsv \
  --output-tsv ami_data/ihm_eval_infer.tsv \
  --output-dir /path/to/feature_save_dir
```

Finally, run inference. The ASR quality (WER), inference time, and GPU memory usage will all be reported.
```bash
bash infer.sh --DATA_ROOT ami_data --SAVE_DIR /path/to/save_asr_model
```
