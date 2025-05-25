# -*- coding: utf-8 -*-
"""
Created on Mon May 19 00:50:20 2025

@author: Keqi Deng (University of Cambridge)
"""

import argparse
from examples.speech_to_text.data_utils import gen_config_yaml, gen_vocab
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def generate_vocab(text_file, output_prefix, vocab_type="bpe", vocab_size=100):
    """
    Generate a vocabulary file (e.g., BPE or unigram) from the input text.
    Assumes the text file has one utterance per line, optionally prefixed by an ID.
    """
    with NamedTemporaryFile(mode="w") as f:
        # Clean and write only the text part (skip the ID if present)
        with open(text_file) as src:
            for line in src:
                parts = line.strip().split()
                if len(parts) > 1:
                    content = " ".join(
                        parts[1:]
                    )  # Skip the first token (usually an ID)
                else:
                    content = parts[0]
                f.write(content + "\n")

        # Call Fairseq's built-in function to generate the vocabulary
        gen_vocab(
            Path(f.name),
            Path(output_prefix),
            vocab_type,
            vocab_size,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text-file",
        required=True,
        help="Input text file (one sentence per line, with optional ID prefix)",
    )
    parser.add_argument(
        "--output-prefix", required=True, help="Output prefix (e.g., 'bpe_model')"
    )
    parser.add_argument(
        "--vocab-type",
        default="bpe",
        choices=["bpe", "unigram", "char"],
        help="Type of vocabulary model",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=100, help="Size of the vocabulary"
    )
    args = parser.parse_args()

    generate_vocab(args.text_file, args.output_prefix, args.vocab_type, args.vocab_size)

    gen_config_yaml(
        Path(os.path.dirname(args.output_prefix)),
        spm_filename=os.path.basename(args.output_prefix) + ".model",
        yaml_filename=f"config.yaml",
        specaugment_policy=None,
        cmvn_type=None,
        extra={"use_audio_input": True},
    )

    gen_config_yaml(
        Path(os.path.dirname(args.output_prefix)),
        spm_filename=os.path.basename(args.output_prefix) + ".model",
        yaml_filename=f"config_infer.yaml",
        specaugment_policy=None,
        cmvn_type=None,
    )
