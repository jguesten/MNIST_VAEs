#!/usr/bin/env bash
# F4: class-conditional samples of the conditional VAE, next to the unconditional VAE.
set -euo pipefail
cd "$(dirname "$0")/.."

python -u run.py --config configs/VAE.yml --set model.beta=1 name=VAE_b1
python -u run.py --config configs/VAE.yml --set model.dim_latent=2 name=VAE_b1_d2


python -u run.py --config configs/cVAE.yml
python -u run.py --config configs/cVAE.yml --set model.dim_latent=2 name=cVAE_d2


jupyter nbconvert --to notebook --execute --inplace notebooks/d5_cvae.ipynb
echo "F4 written to figures/"
