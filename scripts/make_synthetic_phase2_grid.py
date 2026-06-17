import csv
import os
from itertools import product

models = ["GCN", "GAT", "GraphSAGE"]
seeds = [1, 2, 3]
hidden_channels = 64

rows = []
exp_id = 0

# Ring
for model, distance, seed in product(models, [4, 8, 16], seeds):
    for num_layers in [2, distance, 2 * distance]:
        rows.append({
            "experiment_id": exp_id,
            "topology": "Ring",
            "model": model,
            "distance": distance,
            "target_node": distance,
            "num_layers": num_layers,
            "hidden_channels": hidden_channels,
            "seed": seed,
        })
        exp_id += 1

# CrossedRing
# use representative target nodes discovered earlier for the chosen distances
crossed_targets = {2: 2, 4: 4, 8: 8}
for model, distance, seed in product(models, [2, 4, 8], seeds):
    for num_layers in [2, distance, 2 * distance]:
        rows.append({
            "experiment_id": exp_id,
            "topology": "CrossedRing",
            "model": model,
            "distance": distance,
            "target_node": crossed_targets[distance],
            "num_layers": max(2, num_layers),
            "hidden_channels": hidden_channels,
            "seed": seed,
        })
        exp_id += 1

# CliquePath
# representative targets from the current CliquePath construction
clique_targets = {2: 4, 3: 8, 4: 12, 5: 16}
for model, distance, seed in product(models, [2, 3, 4, 5], seeds):
    for num_layers in [2, distance, 2 * distance]:
        rows.append({
            "experiment_id": exp_id,
            "topology": "CliquePath",
            "model": model,
            "distance": distance,
            "target_node": clique_targets[distance],
            "num_layers": max(2, num_layers),
            "hidden_channels": hidden_channels,
            "seed": seed,
        })
        exp_id += 1

out_path = os.path.expanduser("~/gnn_thesis_cluster/configs/synthetic_phase2_depth_grid.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "experiment_id",
            "topology",
            "model",
            "distance",
            "target_node",
            "num_layers",
            "hidden_channels",
            "seed",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} experiments to {out_path}")
