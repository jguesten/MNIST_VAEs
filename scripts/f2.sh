#!/usr/bin/env bash
# F2: latent scatter (d = 2) and latent traversals (d = 10, beta = 1 and 4).
set -euo pipefail
cd "$(dirname "$0")/.."

python -u run.py --config configs/VAE.yml --set model.dim_latent=2 name=VAE_d2
for beta in 1 4; do
    python -u run.py --config configs/VAE.yml --set model.beta=$beta name=VAE_b${beta}
done

jupyter nbconvert --to notebook --execute --inplace notebooks/d2_vae.ipynb
echo "F2 written to figures/"
