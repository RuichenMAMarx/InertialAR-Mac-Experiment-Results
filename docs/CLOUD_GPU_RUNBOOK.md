# Cloud GPU runbook

## Recommended machine

- Linux
- NVIDIA A100 40 GB, L40S 48 GB, or a 24 GB RTX 4090 for bounded inference
- At least 32 GB host RAM
- At least 30 GB free disk
- Python 3.11
- PyTorch 2.3.0 with CUDA 12.x
- FlashAttention 2.6.2

The released QM9 checkpoint is 7,172,585,482 bytes. Start with one generated
molecule per batch and monitor GPU memory before increasing batch size.

## Installation

```bash
git clone https://github.com/HaoruiLi46/InertialAR.git
cd InertialAR

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r env.txt
python -m pip install -e .
python -m pip install huggingface_hub
```

Install the FlashAttention wheel that matches Python, CUDA, and PyTorch, using
the exact release URL documented by the upstream repository.

## Download only QM9

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Haoruili46/InertialAR",
    repo_type="model",
    allow_patterns=["data/QM9/**", "ckpt/QM9/**"],
    local_dir=".",
)
PY
```

## Verify the environment

```bash
python - <<'PY'
import torch
import flash_attn
import rdkit

print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
print("cuda_available", torch.cuda.is_available())
PY
```

## Bounded official-checkpoint generation

```bash
NUM_GENERATE=10 \
BATCH_SIZE=1 \
DEVICE=cuda \
DATA_ROOT=./data \
CKPT_PATH=./ckpt/QM9 \
AUTO_EVAL=1 \
bash scripts/generation/generate_qm9.sh
```

Record the command, commit hash, GPU model, package versions, wall time, peak
GPU memory, generated NPZ, and evaluation JSON. Increase sample count only
after this run completes without memory or dtype errors.

## Representation and steering study

The Mac reproduction adds `return_representations=True` to the model. Port the
same non-default return path to the cloud checkout, load the official
checkpoint strictly, and first cache representations without intervention.

For each selected label:

1. Fit probes on fixed data splits.
2. Estimate a direction from training data only.
3. Normalize the direction using the training representation scale.
4. Intervene separately at `r_type_context` and `r_coord_condition`.
5. Generate with identical random seeds for every alpha.
6. Report target hit rate and molecular quality metrics together.

