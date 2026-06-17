import argparse
import csv
import json
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_crossed_ring_graph(num_nodes: int):
    edges = []

    # normal ring edges
    for i in range(num_nodes):
        j = (i + 1) % num_nodes
        edges.append([i, j])
        edges.append([j, i])

    # chord / crossed connections
    half = num_nodes // 2
    for i in range(half):
        j = (i + half) % num_nodes
        edges.append([i, j])
        edges.append([j, i])

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


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


def make_crossed_ring_transfer_sample(
    num_nodes: int,
    feature_dim: int,
    source: int,
    target: int,
    target_value: int,
):
    edge_index = build_crossed_ring_graph(num_nodes)
    distances = shortest_path_distances(num_nodes, edge_index, source)
    distance = distances[target]
    assert distance >= 0

    x = torch.zeros((num_nodes, feature_dim), dtype=torch.float)
    x[target, target_value % feature_dim] = 1.0

    y = torch.tensor(target_value % feature_dim, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    data.source_index = torch.tensor([source], dtype=torch.long)
    data.target_index = torch.tensor([target], dtype=torch.long)
    data.distance = torch.tensor([distance], dtype=torch.long)
    data.label = y
    return data


def build_crossed_ring_dataset(target_node, num_samples, num_nodes=64, feature_dim=8):
    source = 0
    dataset = []
    for i in range(num_samples):
        target_value = i % feature_dim  # balanced labels
        data = make_crossed_ring_transfer_sample(
            num_nodes=num_nodes,
            feature_dim=feature_dim,
            source=source,
            target=target_node,
            target_value=target_value,
        )
        dataset.append(data)
    return dataset


class NodeTransferGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.0):
        super().__init__()
        assert num_layers >= 2
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        source_repr = x[data.source_index]
        return self.classifier(source_repr)


class NodeTransferGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.0):
        super().__init__()
        assert num_layers >= 2
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        source_repr = x[data.source_index]
        return self.classifier(source_repr)


class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, heads=4, dropout=0.0):
        super().__init__()
        assert num_layers >= 2
        self.dropout = dropout
        hidden_per_head = max(8, hidden_channels // heads)

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_per_head, heads=heads, dropout=dropout))

        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_per_head * heads, hidden_per_head, heads=heads, dropout=dropout)
            )

        self.convs.append(
            GATConv(hidden_per_head * heads, hidden_channels, heads=1, concat=False, dropout=dropout)
        )

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class NodeTransferGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.0):
        super().__init__()
        self.encoder = GATEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            heads=4,
            dropout=dropout,
        )
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x = self.encoder(data.x, data.edge_index)
        source_repr = x[data.source_index]
        return self.classifier(source_repr)


def make_model(model_name, feature_dim, hidden_channels, num_layers):
    if model_name == "GCN":
        return NodeTransferGCN(feature_dim, hidden_channels, feature_dim, num_layers=num_layers, dropout=0.0)
    if model_name == "GraphSAGE":
        return NodeTransferGraphSAGE(feature_dim, hidden_channels, feature_dim, num_layers=num_layers, dropout=0.0)
    if model_name == "GAT":
        return NodeTransferGAT(feature_dim, hidden_channels, feature_dim, num_layers=num_layers, dropout=0.0)
    raise ValueError(f"Unknown model: {model_name}")


def train_epoch(model, dataset, optimizer, device):
    model.train()
    total_loss = 0.0
    for data in dataset:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.label.unsqueeze(0))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataset)


@torch.no_grad()
def evaluate(model, dataset, device):
    model.eval()
    correct = 0
    for data in dataset:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1).item()
        if pred == int(data.label.item()):
            correct += 1
    return correct / len(dataset)


def load_grid_row(grid_path: str, task_id: int):
    with open(grid_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[task_id]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=str, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--out-dir", type=str, default="runs/synthetic_crossed_ring_phase1")
    args = parser.parse_args()

    cfg = load_grid_row(args.grid, args.task_id)

    model_name = cfg["model"]
    distance = int(cfg["distance"])
    target_node = int(cfg["target_node"])
    num_layers = int(cfg["num_layers"])
    hidden_channels = int(cfg["hidden_channels"])
    seed = int(cfg["seed"])

    set_seed(seed)

    feature_dim = 8
    train_dataset = build_crossed_ring_dataset(
        target_node=target_node, num_samples=160, num_nodes=64, feature_dim=feature_dim
    )
    test_dataset = build_crossed_ring_dataset(
        target_node=target_node, num_samples=80, num_nodes=64, feature_dim=feature_dim
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(model_name, feature_dim, hidden_channels, num_layers).to(device)

    learning_rate = 0.01 if model_name != "GAT" else 0.005
    weight_decay = 5e-4 if model_name != "GAT" else 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history = []
    best_test_acc = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_dataset, optimizer, device)
        train_acc = evaluate(model, train_dataset, device)
        test_acc = evaluate(model, test_dataset, device)

        history.append({
            "epoch": epoch,
            "loss": train_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
        })

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch

    final_train_acc = evaluate(model, train_dataset, device)
    final_test_acc = evaluate(model, test_dataset, device)

    result = {
        "topology": "CrossedRing",
        "model": model_name,
        "seed": seed,
        "distance": distance,
        "target_node": target_node,
        "num_layers": num_layers,
        "hidden_channels": hidden_channels,
        "epochs": args.epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "final_train_acc": final_train_acc,
        "final_test_acc": final_test_acc,
        "best_epoch": best_epoch,
        "best_test_acc": best_test_acc,
        "chance_level": 1.0 / feature_dim,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    stem = f"CrossedRing_{model_name}_D{distance}_L{num_layers}_H{hidden_channels}_seed{seed}"

    with open(os.path.join(args.out_dir, f"{stem}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    with open(os.path.join(args.out_dir, f"{stem}_history.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "train_acc", "test_acc"])
        writer.writeheader()
        writer.writerows(history)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
