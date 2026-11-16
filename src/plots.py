# src/plots.py
"""Shared helpers for the analysis notebooks: loading saved runs and making figures.

Experiments are passed as a dict {label: experiment folder}, where each folder
(outputs/<name>/) contains one seed_<n>/ subfolder per run with config.yaml,
model.pt, history.json and, after evaluate_test.py, test_metrics.json.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.ticker import MaxNLocator

from src.models import build_model
# ----------------------------------------------------------------------------
# Loading saved runs
# ----------------------------------------------------------------------------

def seed_dirs(exp_dir):
    """All seed_<n> folders of one experiment, sorted."""
    dirs = sorted(Path(exp_dir).glob("seed_*"))
    if not dirs:
        raise FileNotFoundError(f"no seed_* folders in {Path(exp_dir).resolve()}")
    return dirs


def load_histories(exp_dir):
    """One training history (dict of per-epoch lists) per seed."""
    return [json.load(open(d / "history.json")) for d in seed_dirs(exp_dir)]


def load_test_metrics(exp_dir):
    """One test-metric dict per seed, as written by evaluate_test.py."""
    return [json.load(open(d / "test_metrics.json")) for d in seed_dirs(exp_dir)]


def load_model(run_dir):
    """Rebuild a model from its saved config and load its weights (eval mode, CPU)."""
    run_dir = Path(run_dir)
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    model = build_model(cfg["model"])
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    return model.eval(), cfg


def mean_sd(histories, key, sign=1.0):
    """Mean and sample SD over seeds of one history entry, per epoch."""
    values = sign * np.array([h[key] for h in histories])          # (n_seeds, n_epochs)
    sd = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros(values.shape[1:])
    return values.mean(axis=0), sd


def summarize(values):
    """(mean, sample SD, n) of a list of per-seed numbers."""
    values = np.asarray(values, dtype=float)
    sd = values.std(ddof=1) if len(values) > 1 else 0.0
    return values.mean(), sd, len(values)


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------

@torch.no_grad()
def encode(model, x):
    """Deterministic latent code: model.encode if defined, else the encoder mean."""
    if hasattr(model, "encode"):
        return model.encode(x)
    return model.encoder(x)[:, 0]


def to_image(t):
    """(n, 1, 28, 28) tensor -> (n, 28, 28) numpy array for imshow."""
    return t.squeeze(1).detach().cpu().numpy()


def clean_axes(ax):
    """Remove ticks and frame, but keep the axis so a ylabel can still be shown."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def integer_epochs(ax):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def image_grid(images, n_rows, n_cols):
    """(n_rows*n_cols, 28, 28) array -> one (n_rows*28, n_cols*28) canvas."""
    h, w = images.shape[1:]
    return images.reshape(n_rows, n_cols, h, w).transpose(0, 2, 1, 3).reshape(n_rows * h, n_cols * w)


# ----------------------------------------------------------------------------
# Training curves and test results (any model)
# ----------------------------------------------------------------------------

def plot_curves(ax, experiments, curves, ylabel, title=None, band=True):
    """Mean ± SD over seeds of history entries, one colour per experiment.

    curves: list of (key, name, linestyle, sign), e.g.
        [("train_loss", "train", "-", 1), ("val_loss", "val", "--", 1)]
    sign=-1 flips a curve, e.g. to show -train_loss as an ELBO.
    """
    for label, exp_dir in experiments.items():
        hs = load_histories(exp_dir)
        epochs = np.arange(1, len(hs[0][curves[0][0]]) + 1)
        color = None
        for key, name, style, sign in curves:
            m, s = mean_sd(hs, key, sign)
            suffix = f" {name}" if len(curves) > 1 else ""
            line, = ax.plot(epochs, m, style, color=color, label=f"{label}{suffix}")
            color = line.get_color()                      # same colour for this experiment
            if band:
                ax.fill_between(epochs, m - s, m + s, color=color, alpha=0.2)
    ax.set(xlabel="Epoch", ylabel=ylabel)
    if title:
        ax.set_title(title)
    integer_epochs(ax)
    ax.legend()


def plot_test_bars(ax, experiments, key="val_acc", percent=True, title=None):
    """Bar chart of a test metric, mean ± SD over seeds. Returns {label: (mean, sd, n)}."""
    results = {label: summarize([m[key] for m in load_test_metrics(exp_dir)])
               for label, exp_dir in experiments.items()}
    labels = list(results)
    means = [results[l][0] for l in labels]
    sds = [results[l][1] for l in labels]

    bars = ax.bar(labels, means, yerr=sds, capsize=6)
    fmt = (lambda v: f"{v:.2%}") if percent else (lambda v: f"{v:.2f}")
    ax.bar_label(bars, labels=[fmt(m) for m in means], padding=8)
    if percent:                                   # zoom in: accuracies are all close to 1
        ax.set_ylim(min(m - s for m, s in zip(means, sds)) - 0.02, 1.0)
    if title:
        ax.set_title(title)
    return results


# ----------------------------------------------------------------------------
# VAE figures
# ----------------------------------------------------------------------------

@torch.no_grad()
def plot_reconstructions(experiments, images, n=10):
    """Originals in the top row, one row of reconstructions per experiment (first seed)."""
    x = images[:n]
    rows = 1 + len(experiments)
    fig, axes = plt.subplots(rows, n, figsize=(n * 1.1, rows * 1.25), squeeze=False)

    originals = to_image(x)
    for i in range(n):
        axes[0, i].imshow(originals[i], cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_ylabel("original", rotation=0, ha="right", va="center", labelpad=12)

    for row, (label, exp_dir) in enumerate(experiments.items(), start=1):
        model, _ = load_model(seed_dirs(exp_dir)[0])
        recon = to_image(model.decoder(encode(model, x)))
        for i in range(n):
            axes[row, i].imshow(recon[i], cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(label, rotation=0, ha="right", va="center", labelpad=12)

    for ax in axes.flat:
        clean_axes(ax)
    fig.suptitle("Images and reconstructions")
    fig.tight_layout()
    return fig


@torch.no_grad()
def latent_codes(model, loader):
    """Deterministic latent codes and labels for every image in a loader."""
    zs, ys = [], []
    for images, labels in loader:
        zs.append(encode(model, images))
        ys.append(labels)
    return torch.cat(zs).numpy(), torch.cat(ys).numpy()


def plot_latent_scatter(model, loader, title="Latent codes (d = 2)"):
    """Scatter of 2D latent codes coloured by digit, with the prior's 2-SD circle."""
    if model.dim_latent != 2:
        raise ValueError(f"needs a model with dim_latent = 2, got {model.dim_latent}")
    z, y = latent_codes(model, loader)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sc = ax.scatter(z[:, 0], z[:, 1], c=y, cmap="tab10", vmin=-0.5, vmax=9.5, s=2, alpha=0.6)
    fig.colorbar(sc, ax=ax, ticks=range(10), label="digit")
    ax.add_patch(plt.Circle((0, 0), 2, fill=False, linestyle="--", color="gray"))
    ax.set_aspect("equal")
    ax.set(title=title, xlabel="$z_1$", ylabel="$z_2$")
    fig.tight_layout()
    return fig


@torch.no_grad()
def plot_prior_samples(experiments, n_rows=8, n_cols=8, seed=0):
    """Decode z ~ N(0, I) for each experiment (first seed); same z for equal dim_latent."""
    fig, axes = plt.subplots(1, len(experiments), figsize=(5 * len(experiments), 5.4), squeeze=False)
    for ax, (label, exp_dir) in zip(axes[0], experiments.items()):
        model, _ = load_model(seed_dirs(exp_dir)[0])
        gen = torch.Generator().manual_seed(seed)
        z = torch.randn(n_rows * n_cols, model.dim_latent, generator=gen)
        samples = to_image(model.decoder(z))                # prior samples skip any encoder flow
        ax.imshow(image_grid(samples, n_rows, n_cols), cmap="gray", vmin=0, vmax=1)
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle("Samples from the prior")
    fig.tight_layout()
    return fig


@torch.no_grad()
def plot_traversal(model, image, values=np.linspace(-3, 3, 9)):
    """One row per latent dimension: vary that dimension, keep the others at the image's code."""
    z = encode(model, image[None])                          # (1, d)
    d = model.dim_latent
    fig, axes = plt.subplots(d, len(values), figsize=(len(values) * 0.9, d * 0.9), squeeze=False)
    for j in range(d):
        zs = z.repeat(len(values), 1)
        zs[:, j] = torch.as_tensor(values, dtype=zs.dtype)
        imgs = to_image(model.decoder(zs))
        for k in range(len(values)):
            axes[j, k].imshow(imgs[k], cmap="gray", vmin=0, vmax=1)
            clean_axes(axes[j, k])
        axes[j, 0].set_ylabel(f"$z_{{{j + 1}}}$", rotation=0, ha="right", va="center", labelpad=8)
    for k, v in enumerate(values):
        axes[-1, k].set_xlabel(f"{v:.1f}")
    fig.suptitle("Latent traversals")
    fig.tight_layout()
    return fig


@torch.no_grad()
def active_units(model, loader, threshold=0.01):
    """Variance over images of each latent code dimension, and the number above threshold."""
    z, _ = latent_codes(model, loader)
    var = z.var(axis=0, ddof=1)
    return var, int((var > threshold).sum())


# add to src/plots.py
from collections import defaultdict


def probe_curve(results, encoder, key="test_acc"):
    """{n: [accuracy per subset]} for one encoder from run_probes.py's results."""
    curve = defaultdict(list)
    for r in results:
        if r["encoder"] == encoder:
            curve[r["n"]].append(r[key])
    return dict(curve)


def cnn_curve(outputs_dir, ns, subset_seeds, key="val_acc", name="CNN_n{n}_s{s}"):
    """{n: [accuracy per subset]} from the CNN runs' test_metrics.json files."""
    curve = {}
    for n in ns:
        curve[n] = []
        for s in subset_seeds:
            run = Path(outputs_dir) / name.format(n=n, s=s) / "seed_0"
            curve[n].append(json.load(open(run / "test_metrics.json"))[key])
    return curve

def plot_label_efficiency(ax, curves, title="Label efficiency"):
    """Accuracy vs number of labels, log x-axis, mean ± SD over subsets per curve.

    curves: {label: {n: [accuracies]}}
    """
    for label, curve in curves.items():
        ns = sorted(curve)
        stats = [summarize(curve[n]) for n in ns]
        means = np.array([m for m, _, _ in stats])
        sds = np.array([s for _, s, _ in stats])
        line, = ax.plot(ns, means, "o-", label=label)
        ax.fill_between(ns, means - sds, means + sds, color=line.get_color(), alpha=0.2)

    all_ns = sorted({n for curve in curves.values() for n in curve})
    ax.set_xscale("log")
    ax.set_xticks(all_ns, [f"{n // 1000}k" if n >= 1000 else str(n) for n in all_ns])
    ax.minorticks_off()
    ax.set(xlabel="Number of labelled training images n", ylabel="Test accuracy", title=title)
    ax.grid(alpha=0.3)
    ax.legend()
