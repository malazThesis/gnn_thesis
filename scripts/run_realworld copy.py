
# OVERSMOOTHING_TRACKING_PATCH_V1
import atexit
from pathlib import Path as MetricsPath

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis_metrics.oversmoothing import (
    LayerEmbeddingRecorder,
    append_metrics_csv,
    collect_layer_metrics,
    default_metric_epochs,
    model_gradient_norm,
)

import argparse
import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from dataset_loader_realworld_extra import load_extra_dataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset_any(name: str, root: str, split_idx: int = 0):
    if name in ["Cora", "CiteSeer", "PubMed"]:
        dataset = Planetoid(root=os.path.join(root, "Planetoid", name), name=name)
        data = dataset[0]
        return dataset, data

    if name in ["Roman-empire", "Amazon-Photo", "Amazon-Computers", "Coauthor-CS", "Coauthor-Physics", "Actor", "Chameleon", "Squirrel", "Cornell", "Texas", "Wisconsin", "Amazon-Ratings"]:
        return load_extra_dataset(name=name, root=root, split_idx=split_idx)

    raise ValueError(f"Unknown dataset: {name}")


class PairNorm(nn.Module):
    def __init__(self, scale: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.scale = scale
        self.eps = eps

    def forward(self, x):
        col_mean = x.mean(dim=0, keepdim=True)
        x = x - col_mean
        row_norm = torch.sqrt((x ** 2).sum(dim=1, keepdim=True) + self.eps)
        mean_norm = row_norm.mean()
        return self.scale * x / (mean_norm + self.eps)


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        assert num_layers >= 2
        self.convs = nn.ModuleList()
        self.dropout = dropout

        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GCNPairNorm(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        assert num_layers >= 2
        self.convs = nn.ModuleList()
        self.pns = nn.ModuleList()
        self.dropout = dropout

        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.pns.append(PairNorm())

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.pns.append(PairNorm())

        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for conv, pn in zip(self.convs[:-1], self.pns):
            x = conv(x, edge_index)
            x = pn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        assert num_layers >= 2
        self.convs = nn.ModuleList()
        self.dropout = dropout

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GraphSAGEPairNorm(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        assert num_layers >= 2
        self.convs = nn.ModuleList()
        self.pns = nn.ModuleList()
        self.dropout = dropout

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.pns.append(PairNorm())

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.pns.append(PairNorm())

        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for conv, pn in zip(self.convs[:-1], self.pns):
            x = conv(x, edge_index)
            x = pn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, heads=4, dropout=0.6):
        super().__init__()
        assert num_layers >= 2
        self.dropout = dropout
        self.convs = nn.ModuleList()

        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
            )
        self.convs.append(
            GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        )

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GATPairNorm(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, heads=4, dropout=0.6):
        super().__init__()
        assert num_layers >= 2
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.pns = nn.ModuleList()

        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.pns.append(PairNorm())

        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
            )
            self.pns.append(PairNorm())

        self.convs.append(
            GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        )

    def forward(self, x, edge_index):
        for conv, pn in zip(self.convs[:-1], self.pns):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = pn(x)
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


def build_model(model_name, dataset, hidden_channels, num_layers):
    if model_name == "GCN":
        return GCN(dataset.num_features, hidden_channels, dataset.num_classes, num_layers=num_layers, dropout=0.5)
    if model_name == "GCNPairNorm":
        return GCNPairNorm(dataset.num_features, hidden_channels, dataset.num_classes, num_layers=num_layers, dropout=0.5)
    if model_name == "GraphSAGE":
        return GraphSAGE(dataset.num_features, hidden_channels, dataset.num_classes, num_layers=num_layers, dropout=0.5)
    if model_name == "GraphSAGEPairNorm":
        return GraphSAGEPairNorm(dataset.num_features, hidden_channels, dataset.num_classes, num_layers=num_layers, dropout=0.5)
    if model_name == "GAT":
        gat_hidden = max(8, hidden_channels // 8)
        return GAT(dataset.num_features, gat_hidden, dataset.num_classes, num_layers=num_layers, heads=4, dropout=0.6)
    if model_name == "GATPairNorm":
        gat_hidden = max(8, hidden_channels // 8)
        return GATPairNorm(dataset.num_features, gat_hidden, dataset.num_classes, num_layers=num_layers, heads=4, dropout=0.6)
    raise ValueError(f"Unknown model: {model_name}")


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    result = {}
    for split_name, mask in {
        "train": data.train_mask,
        "val": data.val_mask,
        "test": data.test_mask,
    }.items():
        correct = (pred[mask] == data.y[mask]).sum().item()
        total = int(mask.sum())
        result[split_name] = correct / total
    return result


def load_grid_row(grid_path: str, task_id: int):
    with open(grid_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[task_id]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=str, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="runs/realworld")
    parser.add_argument(
        "--track-oversmoothing",
        action="store_true",
        help="Record layerwise oversmoothing metrics.",
    )
    parser.add_argument(
        "--oversmoothing-epochs",
        type=str,
        default="",
        help="Comma-separated epochs to record.",
    )
    parser.add_argument(
        "--oversmoothing-max-nodes",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--oversmoothing-max-edges",
        type=int,
        default=200000,
    )

    args = parser.parse_args()

    _oversmoothing_epochs = (
        {
            int(value.strip())
            for value in args.oversmoothing_epochs.split(',')
            if value.strip()
        }
        if args.oversmoothing_epochs.strip()
        else default_metric_epochs(
            int(getattr(args, "epochs", 200))
        )
    )


    cfg = load_grid_row(args.grid, args.task_id)

    dataset_name = cfg["dataset"]
    model_name = cfg["model"]
    num_layers = int(cfg["num_layers"])
    hidden_channels = int(cfg["hidden_channels"])

    seed = int(cfg.get("seed", 1))
    split_idx = int(cfg.get("split_idx", 0))

    set_seed(seed)

    dataset, data = load_dataset_any(dataset_name, root=args.data_root, split_idx=split_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    model = build_model(model_name, dataset, hidden_channels, num_layers).to(device)

    default_learning_rate = (
        0.005
        if "GAT" in model_name
        else 0.01
    )

    raw_learning_rate = cfg.get(
        "lr",
        "",
    )

    learning_rate = (
        float(raw_learning_rate)
        if str(raw_learning_rate).strip()
        else default_learning_rate
    )
    weight_decay = 5e-4 if "GAT" not in model_name else 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    _oversmoothing_recorder = (
        LayerEmbeddingRecorder(model)
        if args.track_oversmoothing
        else None
    )
    if _oversmoothing_recorder is not None:
        atexit.register(
            _oversmoothing_recorder.close
        )

    _metric_scope = dict(locals())

    def _metric_lookup(
        names,
        default=None,
    ):
        for name in names:
            value = _metric_scope.get(name)

            if (
                value is not None
                and not isinstance(
                    value,
                    torch.nn.Module,
                )
                and isinstance(
                    value,
                    (
                        str,
                        int,
                        float,
                        bool,
                    ),
                )
            ):
                return value

        for container_name in (
            "config",
            "cfg",
            "row",
            "params",
            "job",
            "task",
            "run_config",
        ):
            source = _metric_scope.get(
                container_name
            )

            if not hasattr(source, "get"):
                continue

            for name in names:
                try:
                    value = source.get(name)
                except Exception:
                    continue

                if value is not None:
                    return value

        for name in names:
            value = getattr(
                args,
                name,
                None,
            )

            if value is not None:
                return value

        return default

    _metric_dataset = str(
        _metric_lookup(
            ("dataset_name", "dataset"),
            "dataset",
        )
    )
    _metric_model = str(
        _metric_lookup(
            (
                "model_name",
                "model_type",
                "model",
            ),
            "model",
        )
    )
    _metric_layers = _metric_lookup(
        ("num_layers", "layers"),
        "x",
    )
    _metric_hidden = _metric_lookup(
        (
            "hidden_channels",
            "hidden_dim",
            "hidden",
        ),
        "x",
    )
    _metric_seed = _metric_lookup(
        ("seed",),
        0,
    )
    _metric_split_idx = _metric_lookup(
        ("split_idx", "split"),
        None,
    )

    _metric_out_dir = MetricsPath(
        getattr(
            args,
            "out_dir",
            "runs",
        )
    )

    _metric_existing_stem = None

    for _metric_candidate in (
        _metric_scope.values()
    ):
        if not isinstance(
            _metric_candidate,
            (str, MetricsPath),
        ):
            continue

        _metric_candidate_name = (
            MetricsPath(
                _metric_candidate
            ).name
        )

        for _metric_suffix in (
            "_history.csv",
            "_summary.json",
        ):
            if _metric_candidate_name.endswith(
                _metric_suffix
            ):
                _metric_existing_stem = (
                    _metric_candidate_name[
                        :-len(_metric_suffix)
                    ]
                )
                break

        if _metric_existing_stem is not None:
            break

    if _metric_existing_stem is not None:
        _metric_stem = _metric_existing_stem
    else:
        _metric_stem = (
            _metric_dataset
            + "_"
            + _metric_model
            + "_L"
            + str(_metric_layers)
            + "_H"
            + str(_metric_hidden)
            + "_seed"
            + str(_metric_seed)
        )

        if _metric_split_idx is not None:
            _metric_stem += (
                "_split"
                + str(_metric_split_idx)
            )

    _oversmoothing_csv = (
        _metric_out_dir
        / (
            _metric_stem
            + "_oversmoothing.csv"
        )
    )


    history = []
    best_val_acc = -1.0
    best_test_acc = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        loss = train(model, data, optimizer)

        _metric_epoch = int(epoch)
        if (
            _oversmoothing_recorder is not None
            and _metric_epoch in _oversmoothing_epochs
        ):
            model.eval()
            _oversmoothing_recorder.start()
            with torch.no_grad():
                _ = model(data.x, data.edge_index)
            _oversmoothing_recorder.stop()

            _metric_rows = collect_layer_metrics(
                embeddings=_oversmoothing_recorder.embeddings,
                edge_index=data.edge_index,
                metadata={
                    "dataset": _metric_dataset,
                    "model": _metric_model,
                    "seed": _metric_seed,
                    "split_idx": _metric_split_idx,
                    "num_layers": _metric_layers,
                    "hidden_channels": _metric_hidden,
                },
                epoch=_metric_epoch,
                gradient_norm=model_gradient_norm(model),
                max_pairwise_nodes=args.oversmoothing_max_nodes,
                max_rank_nodes=args.oversmoothing_max_nodes,
                max_edges=args.oversmoothing_max_edges,
            )
            append_metrics_csv(
                _oversmoothing_csv,
                _metric_rows,
            )

        accs = evaluate(model, data)

        history.append({
            "epoch": epoch,
            "loss": loss,
            "train_acc": accs["train"],
            "val_acc": accs["val"],
            "test_acc": accs["test"],
        })

        if accs["val"] > best_val_acc:
            best_val_acc = accs["val"]
            best_test_acc = accs["test"]
            best_epoch = epoch

    final_accs = evaluate(model, data)

    result = {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "split_idx": split_idx,
        "num_layers": num_layers,
        "hidden_channels": hidden_channels,
        "epochs": args.epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "final_train_acc": final_accs["train"],
        "final_val_acc": final_accs["val"],
        "final_test_acc": final_accs["test"],
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_test_acc_at_best_val": best_test_acc,
    }

    os.makedirs(args.out_dir, exist_ok=True)

    stem = f"{dataset_name}_{model_name}_L{num_layers}_H{hidden_channels}_seed{seed}_split{split_idx}"

    with open(os.path.join(args.out_dir, f"{stem}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    with open(os.path.join(args.out_dir, f"{stem}_history.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "train_acc", "val_acc", "test_acc"])
        writer.writeheader()
        writer.writerows(history)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
