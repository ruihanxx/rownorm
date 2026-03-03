# MOGA

Official PyTorch implementation of the paper:  
**On the Width Scaling of Neural Optimizers Under Matrix Operator Norms I: Row/Column Normalization and Hyperparameter Transfer.**

This repository provides **two fully reproducible training examples**:

- `examples/gpt2/` — GPT-2 pre-training on OpenWebText  
  - includes learning-rate transfer experiments  
  - includes large/standard token budget experiments
- `examples/llama/` — LLaMA pre-training on C4  
  - includes large/standard token budget experiments

The sections below list the **exact steps** to reproduce each example.

---

## 0. Prerequisites

- The provided scripts are configured for **4× H100 GPUs**.
- If your hardware differs (fewer GPUs / different GPU type), you should adjust training parameters (e.g., batch size, gradient accumulation, sequence length, etc.) accordingly.

---

## 1. GPT-2 (OpenWebText)

### Step 1 — Install dependencies

```bash
cd examples/gpt2
conda create -n gpt python=3.12
conda activate gpt
pip install -r requirements.txt
```

### Step 2 — Prepare the OpenWebText dataset

Run the preprocessing script to download/build the OpenWebText training data used by the GPT-2 example:

```bash
python prepare_openwebtext.py
```

### Step 3 — Run the training script

Use the provided launcher script to start the large-token experiment with MOGA (configured for 4 GPUs):

```bash
# run the 130M GPT-2 large-token experiment with MOGA (uses 4 GPUs)
bash scripts/large_token/moga_small.sh
```

After training starts, you can monitor results via the log directory:

#### check training loss log at folder './logs_llm' ####

---

## 2. LLaMA (C4)

### Step 1 — Install dependencies

```bash
cd examples/llama
conda create -n llama python=3.10
conda activate llama
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Step 2 — Prepare the C4 dataset

Download/prepare the C4 dataset using the provided script:

```bash
bash download_c4.sh 
```

### Step 3 — Run the training script

Launch the large-token experiment (configured for 4 GPUs):

```bash
# run the 130M LLaMA large-token experiment with MOGA (uses 4 GPUs)
bash scripts/large_token/llama_130m_moga.sh
```

Training logs are written here:

#### check training loss log at folder './logs_llm' ####

---

## Acknowledgement

The experimental settings for the LLaMA part of this work are based on [Conda](https://github.com/jie040109/Conda). We thank the authors for granting permission to use their implementation.
