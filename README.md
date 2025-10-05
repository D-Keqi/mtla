# MTLA: Multi-head Temporal Latent Attention

![MTLA](assets/mtla.png "Multi-head Temporal Latent Attention")
> **Multi-head Temporal Latent Attention**\
> *Keqi Deng, Philip C. Woodland*  
> 📄 [Paper on arXiv](https://arxiv.org/abs/2505.13544)  
> 🎉 **Accepted at NeurIPS 2025!**  
## About

**MTLA** is a novel attention mechanism building on [DeepSeek](https://github.com/deepseek-ai/DeepSeek-V3) MLA, with a key innovation: **temporal compression of the key-value cache**. This enables more efficient self-attention and significantly reduces memory footprint during inference, making it particularly valuable for decoder-only architectures such as LLMs. Built on [PyTorch](http://pytorch.org/), this project also serves as an open-source, decoder-only toolkit for end-to-end speech and language processing, covering tasks such as text summarisation, speech translation, speech recognition, spoken language understanding, and so on, with fully featured setup recipes.

## Key Features

### Supported Attention Mechanisms
- **Attention**: Multi-head Attention ([MHA](https://arxiv.org/pdf/1706.03762)), Multi-Query Attention ([MQA](https://arxiv.org/pdf/1911.02150)), Grouped-Query Attention ([GQA](https://arxiv.org/pdf/2305.13245)), Multi-head Latent Attention ([MLA](https://arxiv.org/pdf/2405.04434)), and Multi-head Temporal Latent Attention ([MTLA](https://arxiv.org/pdf/2505.13544))
- **Positional Encoding**: Rotary Position Embedding ([RoPE](https://arxiv.org/pdf/2104.09864)), and [Decoupled Rotary Position Embedding](https://arxiv.org/pdf/2405.04434)
- **FlashAttention**: [Extended FlashAttention-2](https://github.com/D-Keqi/flash-attention) for MTLA inference
- **HuggingFace Transformers**: Support [HuggingFace Transformers](https://github.com/huggingface/transformers) toolkit usage to train LLMs based on MTLA

### Complete Setup Recipes
- **Tasks**: speech translation (MuST-C), speech recognition (AMI), spoken language understanding (SLURP), and text summarisation (XSum)
- **Data Processing**: [Fairseq](https://github.com/facebookresearch/fairseq)-style Fbank feature extraction and compression into `zip` file, and [ESPnet2](https://github.com/espnet/espnet)-style speech data processing with raw audio saved in `flac` or `ark` format
- **Feature Extraction**: Fbank online/offline extraction, and self-supervised learning representations as features, using upstream models in [S3PRL](https://github.com/s3prl/s3prl)
- **Notebook Demo**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/D-Keqi/mtla/blob/main/assets/MTLA.ipynb)

### Evaluation
- **Parallel Inference**: Fairseq-style parallel beam search over batches containing multiple data samples
- **Quality Evaluation**: BLEU, WER, classification accuracy, and ROUGE (ROUGE-1, ROUGE-2, and ROUGE-L)
- **Efficiency Evaluation**: inference time spent, and GPU memory (including activation memory and the storage of key-value cache) consumed on inference

## Installation and Usage
- If you only need the Python MTLA module, simply clone this repository or pip install:
  ``` bash
  pip install mtla
  ```
  Then refer to the following example:
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
  **A notebook demo of training with MTLA and performing beam search inference refers to**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/D-Keqi/mtla/blob/main/assets/MTLA.ipynb)
- Optional: **FlashAttention backend** for **MTLA inference**. We provide an optional FlashAttention backend to accelerate MTLA inference. This feature is disabled by default. To enable it, please install our customised FlashAttention fork:
  ``` bash
  git clone https://github.com/D-Keqi/flash-attention.git
  cd flash-attention
  python setup.py install
  ```
  - FlashAttention requires a CUDA-capable GPU with **PyTorch 2.7.0** and **CUDA 12.6** (tested working versions).
  - Only **fp16 (`torch.float16`)** or **bf16 (`torch.bfloat16`)** dtypes are supported.  
  - If FlashAttention is not installed, MTLA will automatically fall back to the standard PyTorch implementation.
  
  Refer to the example below to use our extended FlashAttention for MTLA inference：
  ```python
  import torch
  from MTLA import MultiheadTemporalLatentAttention
  
  batch, length, dim = 2, 16, 512
  dtype = torch.float16  # or torch.bfloat16
  device = "cuda"
  
  x = torch.randn(batch, length, dim, device=device, dtype=dtype)
  pos = torch.arange(0, length, device=device, dtype=torch.float32).view(1, -1)
  
  model = MultiheadTemporalLatentAttention(
      embed_dim=dim,
      num_heads=8,
  ).to(device, dtype=dtype)
  model.eval()
  
  # Incremental inference with FlashAttention-based MTLA
  incremental_state = {}
  outputs = []
  for t in range(length):
      out = model(
          query=x[:, t:t+1],
          key=x[:, t:t+1],
          value=x[:, t:t+1],
          position=pos[:, t:t+1],
          incremental_state=incremental_state,
          use_flashattn_infer=True,  # Enable FlashAttention
      )
      outputs.append(out)
  
  y = torch.cat(outputs, dim=1)
  print("Output shape:", y.shape)  # should be [batch, length, dim]
  ```
- If you want to use **MTLA** through **HuggingFace Transformers** or train an **LLM** based on MTLA, you just need to `import mtla`, then you can load MTLA-based models as easily as you would load any other model in Transformers. See the example below for reference:
  ``` python
  # If you want to build a MTLA-based LLM from scratch
  from mtla import LlamaMTLAConfig, LlamaMTLAForCausalLM
  from transformers import AutoModelForCausalLM, AutoTokenizer
  base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B") # Just an example
  base_config = base_model.config
  config = LlamaMTLAConfig(**vars(base_config))
  config.down_rate = 2 # You can play this and other MTLA-specific parameters
  model = LlamaMTLAForCausalLM(config)

  # If you want to load a MTLA-based pre-trained LLM
  import mtla
  from transformers import AutoModelForCausalLM, AutoTokenizer
  model = AutoModelForCausalLM.from_pretrained("mtla/model/path")
  tokenizer = AutoTokenizer.from_pretrained("mtla/model/path")
  # Then you can use e.g. model.generate() function just like other LLMs
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
@inproceedings{deng2025mtla,
  title={Multi-head Temporal Latent Attention},
  author={Deng, Keqi and Woodland, Philip C},
  booktitle={Proc. NeurIPS},
  address={San Diego, USA},
  year={2025}
}
```
