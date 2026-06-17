import csv
import os
from itertools import product
from collections import deque

import torch


def build_cliquepath_graph(num_cliques: int, clique_size: int):
    edges = []

    for c in range(num_cliques):
        start = c * clique_size
        nodes = list(range(start, start + clique_size))
        for i in nodes:
            for j in nodes:
                if i != j:
                    edges.append([i, j])

    for c in range(num_cliques - 1):
        a = c * clique_size
        b = (c + 1) * clique_size
        edges.append([a, b])
        edges.append([b, a])

    num_nodes = num_cliques * clique_size
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index, num_nodes


def shortest_path_distances(num_nodes: int, edge_index, source: int):
    adj = [[] for _ in range(num_nodes)]
    for u, v in edge_index.t().tolist():
        adj[u].append(v)

    dist = [-1] * num_nodes
    dist[source] = 0
    q = deque([source])

    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)

    return dist


def representative_targets_by_distance(num_nodes: int, edge_index, source: int = 0):
    distances = shortest_path_distances(num_nodes, edge_index, source)
    mapping = {}
    for node, d in enumerate(distances):
        if node == source or d < 0:
            continue
        if d not in mapping:
            mapping[d] = node
    return mapping


models = ["GCN", "GAT", "GraphSAGE"]
hidden_channels_list = [32, 64, 128]
seeds = [1, 2, 3]

edge_index, num_nodes = build_cliquepath_graph(num_cliques=5, clique_size=4)
distance_to_target = representative_targets_by_distance(num_nodes, edge_index, source=0)
distances = sorted(distance_to_target.keys())

rows = []
exp_id = 0

for model, distance, hidden_channels, seed in product(
    models, distances, hidden_channels_list, seeds
):
    rows.append({
        "experiment_id": exp_id,
        "model": model,
        "distance": distance,
        "target_node": distance_to_target[distance],
        "num_layers": max(2, distance),
        "hidden_channels": hidden_channels,
        "seed": seed,
    })
    exp_id += 1

out_path = os.path.expanduser("~/gnn_thesis_cluster/configs/synthetic_cliquepath_grid_phase1.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["experiment_id", "model", "distance", "target_node", "num_layers", "hidden_channels", "seed"],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Available CliquePath distances: {distances}")
print(f"Wrote {len(rows)} experiments to {out_path}")
