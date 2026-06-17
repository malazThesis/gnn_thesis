import csv
import os
from itertools import product

datasets = ["Cora", "CiteSeer", "PubMed"]
models = ["GCN", "GAT", "GraphSAGE", "GCNPairNorm", "GATPairNorm", "GraphSAGEPairNorm"]
num_layers_list = [8, 16, 32]
hidden_channels_list = [64, 128]
seeds = [1, 2, 3]

rows = []
exp_id = 0
for dataset, model, num_layers, hidden_channels, seed in product(
    datasets, models, num_layers_list, hidden_channels_list, seeds
):
    rows.append({
        "experiment_id": exp_id,
        "dataset": dataset,
        "model": model,
        "num_layers": num_layers,
        "hidden_channels": hidden_channels,
        "seed": seed,
    })
    exp_id += 1

out_path = os.path.expanduser("~/gnn_thesis_cluster/configs/realworld_grid_phase2.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["experiment_id", "dataset", "model", "num_layers", "hidden_channels", "seed"],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} experiments to {out_path}")
