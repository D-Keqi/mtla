# -*- coding: utf-8 -*-
"""
Created on Mon May 19 00:50:20 2025

@author: Keqi Deng (University of Cambridge)
"""

import os
import argparse
import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm
from fairseq import checkpoint_utils, tasks, options


def load_fairseq_model(checkpoint_path, data_dir):
    """Load a fairseq speech-to-text model from checkpoint and data directory."""
    parser = options.get_generation_parser()
    input_args = ["--path", checkpoint_path, data_dir]
    args = options.parse_args_and_arch(parser, input_args)
    args.task = "speech_to_text"
    args.config_yaml = "config.yaml"

    task = tasks.setup_task(args)

    models, _model_args = checkpoint_utils.load_model_ensemble(
        [checkpoint_path], arg_overrides={"data": data_dir}, task=task
    )

    model = models[0]
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    return model, task


def forward_encoder_torchscript(model, src_tokens, src_lengths):
    """TorchScript-compatible encoder forward pass for speech input."""
    if model.frontend is not None:
        with torch.cuda.amp.autocast(enabled=False):
            src_tokens, src_lengths = model.frontend(src_tokens, src_lengths)

            if model.normalize is not None:
                src_tokens, src_lengths = model.normalize(src_tokens, src_lengths)

        src_tokens, src_lengths = model.preencoder(src_tokens, src_lengths)
        src_tokens = model.new_norm(src_tokens)

    return src_tokens, src_lengths


def process_eval_tsv(tsv_path, output_tsv, output_dir, model):
    """Process input eval.tsv and extract audio features, saving them as .npy."""
    os.makedirs(output_dir, exist_ok=True)

    with open(tsv_path, "r") as fin, open(output_tsv, "w") as fout:
        header = fin.readline()
        fout.write(header)

        for line_num, line in enumerate(tqdm(fin, desc="Processing"), start=2):
            parts = line.strip().split("\t")
            if len(parts) < 5:
                print(f"Skipping invalid line {line_num}: {line.strip()}")
                continue

            utt_id, audio_path, old_length, text, speaker = parts

            try:
                audio, sample_rate = sf.read(audio_path)
            except Exception as e:
                print(f"Error reading {audio_path}: {e}")
                continue

            audio_input = torch.from_numpy(audio).float().unsqueeze(0).cuda()
            src_length = torch.tensor(int(old_length)).long().unsqueeze(0).cuda()

            with torch.no_grad():
                src_tokens, src_lengths = forward_encoder_torchscript(
                    model, audio_input, src_length
                )

            saved_src = src_tokens.squeeze(0).cpu().numpy()
            new_length = src_lengths[0].item()

            output_path = os.path.join(output_dir, f"{utt_id}.npy")
            np.save(output_path, saved_src)

            new_line = "\t".join([utt_id, output_path, str(new_length), text, speaker])
            fout.write(new_line + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract features using Fairseq S2T model"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to Fairseq data directory"
    )
    parser.add_argument(
        "--input-tsv", type=str, required=True, help="Path to input eval.tsv file"
    )
    parser.add_argument(
        "--output-tsv",
        type=str,
        required=True,
        help="Path to output .tsv file with feature paths",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save extracted features",
    )
    args = parser.parse_args()

    model, _ = load_fairseq_model(args.checkpoint, args.data_dir)

    process_eval_tsv(
        tsv_path=args.input_tsv,
        output_tsv=args.output_tsv,
        output_dir=args.output_dir,
        model=model,
    )


if __name__ == "__main__":
    main()
