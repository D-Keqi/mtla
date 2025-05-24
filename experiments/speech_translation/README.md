# Speech Translation (ST) Example on MuST-C

This example generally follows the Fairseq MuST-C [example](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_text/docs/mustc_example.md).

## Data Preparation
[Download](https://ict.fbk.eu/must-c) and unpack MuST-C data to a path `${MUSTC_ROOT}/en-de`. Note that English-German (En-De) is used as the default here, modify it 
accordingly to conduct experiments with other language pairs.

First, some additional Python packages are required.
```bash
# Additional Python packages for ST data processing/model training
pip install pandas torchaudio soundfile sentencepiece
```
Second, preprocess the data, including Fbank feature extraction, TSV manifest generation, and training an ASR model to initialise the encoder of the ST model. Note 
that training the ASR model is optional and is skipped by default if a pre-trained ASR model is already available, such as those provided 
by [Fairseq](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_text/docs/mustc_example.md).
```bash
# Generate TSV manifests, features, vocabulary,
# and configuration for each language, while skipping ASR training
bash prepare.sh --MUSTC_ROOT ${MUSTC_ROOT}
```
To train a new ASR model from scratch, run:
```bash
# Generate TSV manifests, features, vocabulary,
# and configuration for each language, while also training ASR
bash prepare.sh --skip_asr false --MUSTC_ROOT ${MUSTC_ROOT} --ASR_SAVE_DIR /path/to/save_ASR_model
```
Third, train the ST model. By default, MTLA is used as the decoder-only self-attention structure, i.e. `--arch s2t_decoder_only_MTLA_cross_xm`.
To train other models:
- MHA: `--arch s2t_decoder_only_roformer_cross_xm`
- MLA: `--arch s2t_decoder_only_MLA_cross_xm`
- MQA: `--arch s2t_decoder_only_MQAroformer_cross_xm`
- GQA: `--arch s2t_decoder_only_GQAroformer_cross_xm`
```bash
bash train.sh --MUSTC_ROOT ${MUSTC_ROOT} --ASR_SAVE_DIR /path/to/save_ASR_model/***.pt --ST_SAVE_DIR /path/to/save_ST_model
```
Fourth, after training is completed, apply the model averaging technique.
```bash
CHECKPOINT_FILENAME=avg_10_checkpoint.pt
python ./../tools/fairseq/scripts/average_checkpoints.py \
  --inputs ${ST_SAVE_DIR} --num-best-checkpoints 10 \
  --output "${ST_SAVE_DIR}/${CHECKPOINT_FILENAME}"
```
Finally, run inference. The translation quality (BLEU score), inference time, and GPU memory usage will all be reported.
```bash
bash infer.sh --MUSTC_ROOT ${MUSTC_ROOT} --ST_SAVE_DIR /path/to/save_ST_model
```
