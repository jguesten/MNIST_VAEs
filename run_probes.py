# run_probes.py
import argparse, json
from pathlib import Path

import torch
import yaml

from src.data import get_loaders, load_subset
from src.probes import extract_features, fit_probe, probe_accuracy
from src.plots import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
cfg = yaml.safe_load(open(parser.parse_args().config))

train_loader, val_loader, test_loader = get_loaders(**cfg["data"])
train_eval = torch.utils.data.DataLoader(train_loader.dataset, batch_size=256, shuffle=False)

results = []
for label, run_dir in cfg["encoders"].items():
    model, _ = load_model(run_dir)
    x_train, y_train = extract_features(model, train_eval)
    x_val, y_val = extract_features(model, val_loader)
    x_test, y_test = extract_features(model, test_loader)
    for n in cfg["ns"]:
        for seed in cfg["subset_seeds"]:

            idx = torch.tensor(load_subset(n, seed, cfg["subsets"]))
            probe, mean, std = fit_probe(x_train[idx], y_train[idx], **cfg["probe"])
            acc = probe_accuracy(probe, mean, std, x_val, y_val)
            test_acc = probe_accuracy(probe, mean, std, x_test, y_test)
            results.append({"encoder": label, "n": n, "subset_seed": seed, "val_acc": acc, "test_acc": test_acc})
            print(f"{label} n={n} seed={seed}: acc:{acc:.4f}")

out = Path("outputs") / cfg["name"]
out.mkdir(parents=True, exist_ok=True)
json.dump(results, open(out / "results.json", "w"), indent=1)