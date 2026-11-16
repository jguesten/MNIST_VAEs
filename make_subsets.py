# make_subsets.py
import json
from src.data import get_loaders, balanced_subset
from pathlib import Path

out = Path("splits") / "subsets.json"
out.parent.mkdir(parents=True, exist_ok=True)

train_split = get_loaders(batch_size=256, val_size=10000, split_seed=0)[0].dataset
labels = train_split.dataset.targets[train_split.indices]

subsets = {n: {s: balanced_subset(labels, n, s) for s in [0, 1, 2]}
           for n in [100, 300, 1000, 3000, 10000]}
subsets[50000] = {s: list(range(len(labels))) for s in [0, 1, 2]}

with open(out, "w") as f:
    json.dump(subsets, f)