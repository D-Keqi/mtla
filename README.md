# MTLA: Multi-head Temporal Latent Attention

![MTLA](assets/mtla.png "Multi-head Temporal Latent Attention")
> **Multi-head Temporal Latent Attention**\
> Keqi Deng, Philip C. Woodland\
> Paper: https://arxiv.org/abs/2505.13544

## About

**MTLA** is a novel attention mechanism building on [DeepSeek](https://github.com/deepseek-ai/DeepSeek-V3) MLA, with a key innovation: **temporal compression of the key-value cache**. This enables more efficient self-attention and significantly reduces memory footprint during inference, making it particularly valuable for decoder-only architectures such as LLMs. Built on [PyTorch](http://pytorch.org/), this project also serves as an open-source, decoder-only toolkit for end-to-end speech and language processing, covering tasks such as text summarisation, speech translation, speech recognition, spoken language understanding, and so on, with fully featured setup recipes.

## Key Features

### Supported Attention Mechanisms
- **Attention**: Multi-head Attention ([MHA](https://arxiv.org/pdf/1706.03762)), Multi-Query Attention ([MQA](https://arxiv.org/pdf/1911.02150)), Grouped-Query Attention ([GQA](https://arxiv.org/pdf/2305.13245)), Multi-head Latent Attention ([MLA](https://arxiv.org/pdf/2405.04434)), and Multi-head Temporal Latent Attention ([MTLA](https://arxiv.org/pdf/2505.13544))
- **Positional Encoding**: Rotary Position Embedding ([RoPE](https://arxiv.org/pdf/2104.09864)), and [Decoupled Rotary Position Embedding](https://arxiv.org/pdf/2405.04434)

### Complete Setup Recipes
- Tasks: speech translation (MuST-C), speech recognition (AMI), spoken language understanding (SLURP), and text summarisation (XSum)
- Data Processing: [Fairseq](https://github.com/facebookresearch/fairseq)-style Fbank feature extraction and compression into `zip` file, and [ESPnet2](https://github.com/espnet/espnet)-style speech data processing with raw audio saved in `flac` or `ark` format
- Feature Extraction: Fbank online/offline extraction, and self-supervised learning representations as features, using upstream models in [S3PRL](https://github.com/s3prl/s3prl)

### Evaluation
- **Parallel Inference**: Fairseq-style parallel beam search over batches containing multiple data samples
- **Quality Evaluation**: BLEU, WER, classification accuracy, and ROUGE (ROUGE-1, ROUGE-2, and ROUGE-L)
- **Efficiency Evaluation**: inference time spent, and GPU memory (including activation memory and the storage of key-value cache) consumed on inference

## Installation and Usage
- If you only need the Python MTLA module, simply clone this repository and refer to the following example:
  ``` python
  import torch
  from MTLA import MultiheadTemporalLatentAttention
  
  batch, length, dim = 2, 64, 512
  x = torch.randn(batch, length, dim)
  pos = torch.arange(0, length).float().view(1, -1) # Position information
  model = MultiheadTemporalLatentAttention(
      embed_dim=dim, # Model dimension
      num_heads=8,  # Attention heads of queries
  )
  y = model(query=x, key=x, value=x, position=pos)
  assert y.shape == x.shape
  ```
- If you intend to run the full experiments, please install the project as described below before proceeding to the examples in the `experiments` directory.
  * [PyTorch](http://pytorch.org/) version >= 1.10.0
  * Python version >= 3.8
  ``` bash
  cd experiments/tools/fairseq
  pip install --editable ./
  ```

## Citation

If you use this codebase, or otherwise find our work valuable, please cite MTLA:
```
@article{mtla,
  title={Multi-head Temporal Latent Attention},
  author={Deng, Keqi and Woodland, Philip C},
  journal={arXiv preprint arXiv:2505.13544},
  year={2025}
}
```
