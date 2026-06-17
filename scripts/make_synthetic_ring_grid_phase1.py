import csv
import os
from itertools import product

models = ["GCN", "GAT", "GraphSAGE"]
distances = [2, 4, 8, 16]
hidden_channels_list = [32, 64, 128]
seeds = [1, 2, 3]

rows = []
exp_id = 0

for model, distance, hidden_channels, seed in product(
    models, distances, hidden_channels_list, seeds
):
    rows.append({
        "experiment_id": exp_id,
        "model": model,
        "distance": distance,
        "num_layers": distance,
        "hidden_channels": hidden_channels,
        "seed": seed,
    })
    exp_id += 1

out_path = os.path.expanduser("~/gnn_thesis_cluster/configs/synthetic_ring_grid_phase1.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["experiment_id", "model", "distance", "num_layers", "hidden_channels", "seed"],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} experiments to {out_path}")
