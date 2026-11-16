#!/usr/bin/env bash
# F1: rate–distortion plot, beta sweep at d = 10, 3 seeds each.
set -euo pipefail
cd "$(dirname "$0")/.."

for beta in 0.5 1 2 4; do
    python -u run.py --config configs/VAE.yml --set model.beta=$beta name=VAE_b${beta}
done
python -u evaluate_test.py outputs/VAE_b0.5 outputs/VAE_b1 outputs/VAE_b2 outputs/VAE_b4

jupyter nbconvert --to notebook --execute --inplace notebooks/d3_beta_sweep.ipynb
echo "F1 written to figures/"
