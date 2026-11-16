#!/usr/bin/env bash
# F3: label-efficiency curve, linear probes on frozen encoders (beta = 1, 4) vs. CNN from scratch.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f splits/subsets.json ] || python -u make_subsets.py

for beta in 0.5 1 4; do
    python -u run.py --config configs/VAE.yml --set model.beta=$beta name=VAE_b${beta}
done
python -u run_probes.py --config configs/probes.yml

for n in 100 300 1000 3000 10000 50000; do
    for s in 0 1 2; do
        python -u run.py --config configs/CNN_subset.yml \
            --set data.train_subset.n=$n data.train_subset.seed=$s name=CNN_n${n}_s${s}
    done
done
python -u evaluate_test.py outputs/CNN_n*_s*

jupyter nbconvert --to notebook --execute --inplace notebooks/d4_probes.ipynb
echo "F3 written to figures/"
