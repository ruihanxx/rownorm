#!/usr/bin/env python
"""
Utility script to download and preprocess OpenWebText for the GPT-2 example.

After running:

    cd examples/gpt2
    python prepare_openwebtext.py

you should get:

    examples/gpt2/data/openwebtext-bin/train.bin
    examples/gpt2/data/openwebtext-bin/val.bin

which are exactly the files expected by `train.py` (see the `--dataset_dir`
argument, whose default is `data/openwebtext-bin`).
"""

import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm


def encode_dataset(text_iter, tokenizer: GPT2TokenizerFast, desc: str):
    """
    Encode an iterable of texts into a single 1D numpy array of uint16 token ids.

    We use GPT-2's tokenizer and append an EOS token after each document to keep
    documents separated while still forming one continuous token stream.
    """
    all_ids = []
    eos_id = tokenizer.eos_token_id

    for text in tqdm(text_iter, desc=desc):
        # Basic safety: make sure text is a string
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if eos_id is not None:
            ids.append(eos_id)
        all_ids.extend(ids)

    # GPT-2 vocab size is 50257, so uint16 is safe (max id 50256)
    return np.array(all_ids, dtype=np.uint16)


def main():
    script_dir = Path(__file__).resolve().parent

    # Where we will write the bin files
    out_dir = script_dir / "data" / "openwebtext-bin"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_bin_path = out_dir / "train.bin"
    val_bin_path = out_dir / "val.bin"

    print(f"Output directory: {out_dir}")

    # -------------------------------------------------------------------------
    # 1) Load OpenWebText via Hugging Face datasets
    # -------------------------------------------------------------------------
    #
    # This uses the community OpenWebText dataset. You can change the name here
    # if you want to use a different variant or a local copy.
    #
    print("Loading OpenWebText dataset from Hugging Face (split='train')...")
    dataset = load_dataset("openwebtext", split="train")
    print(f"Loaded {len(dataset):,} documents.")

    # Simple 90/10 train/val split at the document level
    print("Splitting into train/val (90/10)...")
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_texts = split["train"]["text"]
    val_texts = split["test"]["text"]

    # -------------------------------------------------------------------------
    # 2) Load GPT-2 tokenizer
    # -------------------------------------------------------------------------
    print("Loading GPT-2 tokenizer (GPT2TokenizerFast from transformers)...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Ensure EOS token exists (it does for 'gpt2', but we check for clarity)
    if tokenizer.eos_token_id is None:
        raise RuntimeError("GPT-2 tokenizer has no EOS token defined.")

    # -------------------------------------------------------------------------
    # 3) Encode and save
    # -------------------------------------------------------------------------
    print("Encoding train split...")
    train_ids = encode_dataset(train_texts, tokenizer, desc="Encoding train")
    print(f"Train tokens: {train_ids.size:,}")
    print(f"Writing to {train_bin_path} ...")
    train_ids.tofile(train_bin_path)

    print("Encoding val split...")
    val_ids = encode_dataset(val_texts, tokenizer, desc="Encoding val")
    print(f"Val tokens: {val_ids.size:,}")
    print(f"Writing to {val_bin_path} ...")
    val_ids.tofile(val_bin_path)

    print("\n✅ Done.")
    print(f"Train bin: {train_bin_path}")
    print(f"Val bin:   {val_bin_path}")
    print("\nNow you can run GPT-2 training with the default:")
    print("  --dataset_dir data/openwebtext-bin")


if __name__ == "__main__":
    main()

