# evaluate_test.py
"""Evaluate trained models on the test set and save test_metrics.json per run.

Usage: python -u evaluate_test.py outputs/MLP outputs/CNN_n*_s*
"""
import argparse
import json
from pathlib import Path

import torch
import yaml

from src.data import get_loaders
from src.models import build_model
from src.train import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dirs", nargs="+", help="experiment folders in outputs/")
    args = parser.parse_args()

    test_loader = None
    for exp_dir in args.exp_dirs:
        for run_dir in sorted(Path(exp_dir).glob("seed_*")):
            cfg = yaml.safe_load(open(run_dir / "config.yaml"))
            if test_loader is None:
                data_cfg = {k: v for k, v in cfg["data"].items() if k != "train_subset"}
                _, _, test_loader = get_loaders(**data_cfg)
            model = build_model(cfg["model"])
            model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
            metrics = evaluate(model, test_loader)
            with open(run_dir / "test_metrics.json", "w") as f:
                json.dump(metrics, f)
            print(f"{run_dir}: {metrics}")


if __name__ == "__main__":
    main()