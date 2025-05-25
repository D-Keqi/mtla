# -*- coding: utf-8 -*-
"""
Created on Mon May 19 00:50:20 2025

@author: Keqi Deng (University of Cambridge)
"""

import os
import argparse
from pathlib import Path


def kaldi_to_fairseq_tsv(
    text_file,
    wav_scp_file,
    utt2num_frames_file,
    utt2spk_file,
    output_tsv,
    sample_rate=16000,
    frame_shift_ms=10,
):
    """
    Convert Kaldi-format data to Fairseq TSV format.
    """
    # Read input files
    with open(text_file) as f:
        text = {line.split()[0]: " ".join(line.split()[1:]) for line in f}
    with open(wav_scp_file) as f:
        wav = {line.split()[0]: line.split()[1] for line in f}
    with open(utt2num_frames_file) as f:
        n_samples = {line.split()[0]: int(line.split()[1]) for line in f}
    with open(utt2spk_file) as f:
        speaker = {line.split()[0]: line.split()[1] for line in f}

    # Frame shift in samples
    frame_shift_samples = int(frame_shift_ms * sample_rate / 1000)
    n_frames = (
        n_samples  # Modify if necessary: e.g., for raw sample counts -> frame count
    )

    # Write TSV file
    with open(output_tsv, "w") as f:
        f.write("id\taudio\tn_frames\ttgt_text\tspeaker\n")
        for utt_id in text.keys():
            f.write(
                f"{utt_id}\t{wav[utt_id]}\t{n_frames[utt_id]}\t{text[utt_id]}\t{speaker[utt_id]}\n"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Convert Kaldi-style data folders to Fairseq TSV format."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory containing Kaldi-format subdirectories.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save TSV files."
    )
    parser.add_argument(
        "--subsets",
        type=str,
        nargs="+",
        default=["ihm_train_sp", "ihm_dev", "ihm_eval"],
        help="List of Kaldi subsets to process.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for subset in args.subsets:
        subset_dir = os.path.join(args.base_dir, subset)
        print(f"Processing subset: {subset}")
        kaldi_to_fairseq_tsv(
            text_file=os.path.join(subset_dir, "text"),
            wav_scp_file=os.path.join(subset_dir, "wav.scp"),
            utt2num_frames_file=os.path.join(subset_dir, "utt2num_samples"),
            utt2spk_file=os.path.join(subset_dir, "utt2spk"),
            output_tsv=os.path.join(args.output_dir, f"{subset}.tsv"),
        )


if __name__ == "__main__":
    main()
