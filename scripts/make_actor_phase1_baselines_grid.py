import csv
import os
from itertools import product

dataset = "Actor"
models = ["GCN", "GAT", "GraphSAGE"]
num_layers_list = [2, 4]
hidden_channels_list = [64, 128]
split_indices = list(range(10))
seed = 1

rows = []
exp_id = 0

for model, num_layers, hidden_channels, split_idx in product(
    models, num_layers_list, hidden_channels_list, split_indices
):
    rows.append({
        "experiment_id": exp_id,
        "dataset": dataset,
        "model": model,
        "num_layers": num_layers,
        "hidden_channels": hidden_channels,
        "seed": seed,
        "split_idx": split_idx,
    })
    exp_id += 1

out_path = "configs/actor_phase1_baselines.csv"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "experiment_id",
            "dataset",
            "model",
            "num_layers",
            "hidden_channels",
            "seed",
            "split_idx",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} experiments to {os.path.abspath(out_path)}")
