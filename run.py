# run.py
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data import get_loaders
from src.models import build_model
from src.train import fit


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def apply_overrides(cfg, overrides):
    """Apply KEY=VALUE overrides, e.g. model.beta=2, to the nested config dict."""
    for item in overrides:
        key, value = item.split("=", 1)
        *parents, last = key.split(".")
        section = cfg
        for p in parents:
            section = section[p]
        section[last] = yaml.safe_load(value)      # "2" -> 2, "0.5" -> 0.5, "abc" -> "abc"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. model.beta=2 name=VAE_b2")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    apply_overrides(cfg, args.set)

    for seed in cfg["seeds"]:
        out_dir = Path("outputs") / cfg["name"] / f"seed_{seed}"
        if (out_dir / "history.json").exists():
            print(f"[{cfg['name']}] seed {seed}: already done, skipping")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f)                  # the settings actually used

        print(f"[{cfg['name']}] seed {seed}: training")
        set_seed(seed)
        train_loader, val_loader, _ = get_loaders(**cfg["data"])
        model = build_model(cfg["model"])
        history = fit(model, train_loader, val_loader, **cfg["training"])

        torch.save(model.state_dict(), out_dir / "model.pt")
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f)
        print(f"[{cfg['name']}] seed {seed}: saved to {out_dir}")


if __name__ == "__main__":
    main()