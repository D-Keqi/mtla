# Spoken Language Understanding Example on SLURP

## Data Preparation
Download the [SLURP] (https://github.com/pswietojanski/slurp) data and set its path to the value of `SLURP` in `local.data.sh`. 
Follow the processing steps from [ESPnet](https://github.com/espnet/espnet), then convert the data into TSV manifests.
```bash
bash prepare.sh
```
The processed data is saved in the `slurp_data` folder.

## Model Training
Second, train the model on the processed SLURP data. By default, MTLA is used as the decoder-only self-attention structure, i.e. `--arch s2t_decoder_only_MTLA_cross_xm`.
To train other models:
- MHA: `--arch s2t_decoder_only_roformer_cross_xm`
- MLA: `--arch s2t_decoder_only_MLA_cross_xm`
```bash
bash train.sh --DATA_ROOT slurp_data --SAVE_DIR /path/to/save_slu_model
```

Third, after training is completed, apply the model averaging technique.
```bash
CHECKPOINT_FILENAME=avg_10_checkpoint.pt
python ./../tools/fairseq/scripts/average_checkpoints.py \
  --inputs ${SAVE_DIR} --num-best-checkpoints 10 \
  --output "${SAVE_DIR}/${CHECKPOINT_FILENAME}"
```

## Model Inference
Finally, run inference. The intent identifiaction quality (accuracy), inference time, and GPU memory usage will all be reported.
```bash
bash infer.sh --DATA_ROOT ami_data --SAVE_DIR /path/to/save_slu_model
```
