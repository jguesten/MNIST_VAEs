# Variational autoencoders on MNIST

Representation-learning project: VAEs with Bernoulli decoders on MNIST, a β sweep,
linear probes for label efficiency, conditional VAE with learned class means, and VAE adaptations: convolutional layers and an added inverse-autoregressive-flow. 

Every figure in the report can be reproduced from a fresh clone with one command:

| Figure | Content | Command |
|---|---|---|
| F1 | Rate–distortion plot, β ∈ {0.5, 1, 2, 4}, d = 10 | `bash scripts/f1.sh` |
| F2 | 2D latent scatter (d = 2) and latent traversals (d = 10) | `bash scripts/f2.sh` |
| F3 | Label-efficiency curve: VAE probes (β = 1, 4) vs. CNN | `bash scripts/f3.sh` |
| F4 | Conditional-VAE class-conditional samples | `bash scripts/f4.sh` |

Each script trains whatever runs the figure needs, evaluates them on the test set,
and then executes the corresponding notebook, which writes the figure to `figures/`.
Runs that already exist in `outputs/` are skipped, so scripts can be rerun safely and
share runs with each other (e.g. F2 and F3 reuse the β = 1 and β = 4 models from F1).

## Setup

Tested with Python 3.11 and the package versions in `requirements.txt`
(PyTorch 2.2.2, the last release available for Intel Macs).

```bash
git clone <repository-url>
cd vae_mnist
conda create -n vae python=3.11
conda activate vae
conda env config vars set PYTHONNOUSERSITE=1    # ignore packages in ~/.local
conda deactivate && conda activate vae
python -m pip install -r requirements.txt
```

MNIST is downloaded automatically to `data/` on the first run.
All commands are run from the repository root.

## Data and evaluation protocol

- **Split:** the official 60,000 training images are split into 50,000 for training and
  10,000 for validation with a fixed seed (`split_seed: 0` in every config).
  The 10,000 official test images are used only for final numbers.
- **Validation** is used during development: training curves, checking settings.
- **Test** evaluation happens only in `evaluate_test.py` (models) and in the final probe
  run (`run_probes.py`), after all settings were fixed.
- **Seeds:** each VAE configuration is trained with 3 seeds (0, 1, 2); figures report the
  mean ± 1 SD over seeds. The label-efficiency experiment uses 3 class-balanced
  subsets per n (`splits/subsets.json`, created by `make_subsets.py`); its error
  bands are ± 1 SD over subsets.

## Repository layout

```
configs/          experiment settings (YAML)
src/              models, data loading, training loop, probes, plotting helpers
notebooks/        one notebook per figure; load saved results and draw the figure
scripts/          f1.sh ... f4.sh: one command per figure
run.py            train one experiment (all seeds in its config)
evaluate_test.py  test-set metrics for trained runs
make_subsets.py   class-balanced labelled subsets for the probes and CNN baseline
run_probes.py     linear probes on frozen encoders
outputs/          trained runs (created; not in git)
figures/          generated figures (created; not in git)
splits/           labelled subsets (committed, so everyone uses the same images)
```

## Running individual experiments

```bash
python -u run.py --config configs/vae.yml                                  # one experiment
python -u run.py --config configs/vae.yml --set model.beta=4 name=VAE_b4_d10  # with overrides
python -u evaluate_test.py outputs/VAE_b4_d10                              # test metrics
```

`--set` overrides any config value (`section.key=value`); every run saves the config it
actually used to `outputs/<name>/seed_<n>/config.yaml`, next to `model.pt`,
`history.json` and, after evaluation, `test_metrics.json`.

## Approximate runtimes (CPU)

Training times depend strongly on the machine; on a laptop CPU, one VAE run of 30 epochs
takes several minutes. F1 trains 12 runs, F3 trains 18 small CNN runs plus the probes,
so expect F1 and F3 to take the longest.

## Use of coding assistants
Crucial code such as the model classes, the ELBO, reparametrisation trick, KL terms, probe, as well as VAE extensions were written by me. Boilerplate code such as data loaders, plotting, logging, scripts and this README were written by a coding assistant.