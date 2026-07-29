from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.utils import (
    coalesce,
    homophily,
    remove_self_loops,
    stochastic_blockmodel_graph,
)


def parse_float_list(value: str) -> list[float]:
    return [
        float(item.strip())
        for item in value.split(",")
        if item.strip()
    ]


def parse_int_list(value: str) -> list[int]:
    return [
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    ]


def make_stratified_masks(
    y: torch.Tensor,
    *,
    num_splits: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_nodes = y.numel()

    train_mask = torch.zeros(
        (num_nodes, num_splits),
        dtype=torch.bool,
    )
    val_mask = torch.zeros_like(train_mask)
    test_mask = torch.zeros_like(train_mask)

    classes = torch.unique(y).tolist()

    for split_idx in range(num_splits):
        generator = torch.Generator()
        generator.manual_seed(
            seed * 10_000 + split_idx
        )

        for class_id in classes:
            indices = torch.where(
                y == class_id
            )[0]

            permutation = indices[
                torch.randperm(
                    indices.numel(),
                    generator=generator,
                )
            ]

            num_train = max(
                1,
                int(train_ratio * len(permutation)),
            )
            num_val = max(
                1,
                int(val_ratio * len(permutation)),
            )

            train_indices = permutation[:num_train]
            val_indices = permutation[
                num_train:num_train + num_val
            ]
            test_indices = permutation[
                num_train + num_val:
            ]

            train_mask[
                train_indices,
                split_idx,
            ] = True
            val_mask[
                val_indices,
                split_idx,
            ] = True
            test_mask[
                test_indices,
                split_idx,
            ] = True

    return train_mask, val_mask, test_mask


def probabilities(
    *,
    num_nodes: int,
    target_homophily: float,
    average_degree: float,
) -> tuple[float, float]:
    """
    For two approximately equal blocks:

        p_in  = h * scale
        p_out = (1-h) * scale

    The scale is selected to approximately preserve the
    requested expected degree.
    """
    block_size = num_nodes // 2

    denominator = (
        (block_size - 1) * target_homophily
        + block_size * (1.0 - target_homophily)
    )

    scale = average_degree / denominator

    p_in = target_homophily * scale
    p_out = (1.0 - target_homophily) * scale

    if max(p_in, p_out) > 1.0:
        raise ValueError(
            "Requested degree is too large for this "
            "number of nodes and homophily."
        )

    return p_in, p_out


def generate_graph(
    *,
    num_nodes: int,
    feature_dim: int,
    feature_signal: float,
    average_degree: float,
    target_homophily: float,
    graph_seed: int,
    num_splits: int,
    train_ratio: float,
    val_ratio: float,
) -> Data:
    torch.manual_seed(graph_seed)

    first_block = num_nodes // 2
    second_block = num_nodes - first_block

    y = torch.cat(
        [
            torch.zeros(
                first_block,
                dtype=torch.long,
            ),
            torch.ones(
                second_block,
                dtype=torch.long,
            ),
        ]
    )

    p_in, p_out = probabilities(
        num_nodes=num_nodes,
        target_homophily=target_homophily,
        average_degree=average_degree,
    )

    edge_probs = [
        [p_in, p_out],
        [p_out, p_in],
    ]

    edge_index = stochastic_blockmodel_graph(
        [first_block, second_block],
        edge_probs,
        directed=False,
    )

    edge_index, _ = remove_self_loops(edge_index)
    edge_index = coalesce(
        edge_index,
        num_nodes=num_nodes,
    )

    feature_generator = torch.Generator()
    feature_generator.manual_seed(
        graph_seed * 1000 + 17
    )

    direction = torch.randn(
        feature_dim,
        generator=feature_generator,
    )
    direction = direction / direction.norm().clamp_min(
        1e-12
    )

    class_sign = (
        y.float() * 2.0 - 1.0
    ).unsqueeze(1)

    noise = torch.randn(
        (num_nodes, feature_dim),
        generator=feature_generator,
    )

    x = (
        noise
        + class_sign
        * (feature_signal / 2.0)
        * direction.unsqueeze(0)
    )

    train_mask, val_mask, test_mask = (
        make_stratified_masks(
            y,
            num_splits=num_splits,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=graph_seed,
        )
    )

    realized_homophily = float(
        homophily(
            edge_index,
            y,
            method="edge",
        )
    )

    realized_average_degree = (
        edge_index.size(1) / num_nodes
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    data.dataset_name = (
        f"CSBM-H{target_homophily:.1f}"
    )
    data.target_homophily = target_homophily
    data.realized_homophily = realized_homophily
    data.target_average_degree = average_degree
    data.realized_average_degree = (
        realized_average_degree
    )
    data.feature_signal = feature_signal
    data.train_ratio = train_ratio
    data.val_ratio = val_ratio
    data.graph_seed = graph_seed
    data.num_classes = 2

    return data


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--homophily-values",
        type=str,
        default="0.1,0.5,0.9",
    )
    parser.add_argument(
        "--graph-seeds",
        type=str,
        default="1",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--feature-signal",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--average-degree",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--num-splits",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
    )

    args = parser.parse_args()

    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("Invalid train ratio.")

    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("Invalid validation ratio.")

    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError(
            "Train and validation ratios must sum "
            "to less than one."
        )

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    homophily_values = parse_float_list(
        args.homophily_values
    )
    graph_seeds = parse_int_list(
        args.graph_seeds
    )

    manifest_rows = []

    for target_homophily in homophily_values:
        if not 0.0 <= target_homophily <= 1.0:
            raise ValueError(target_homophily)

        for graph_seed in graph_seeds:
            data = generate_graph(
                num_nodes=args.num_nodes,
                feature_dim=args.feature_dim,
                feature_signal=args.feature_signal,
                average_degree=args.average_degree,
                target_homophily=target_homophily,
                graph_seed=graph_seed,
                num_splits=args.num_splits,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
            )

            filename = (
                f"csbm_h{int(target_homophily * 10):02d}"
                f"_seed{graph_seed}.pt"
            )
            path = args.out_dir / filename

            torch.save(data, path)

            manifest_rows.append(
                {
                    "path": str(path),
                    "dataset": data.dataset_name,
                    "graph_seed": graph_seed,
                    "num_nodes": data.num_nodes,
                    "num_edges": data.num_edges,
                    "feature_dim": data.num_features,
                    "target_homophily":
                        target_homophily,
                    "realized_homophily":
                        data.realized_homophily,
                    "target_average_degree":
                        args.average_degree,
                    "realized_average_degree":
                        data.realized_average_degree,
                    "feature_signal":
                        args.feature_signal,
                    "train_ratio":
                        args.train_ratio,
                    "val_ratio":
                        args.val_ratio,
                    "num_splits":
                        args.num_splits,
                }
            )

            print(
                filename,
                f"nodes={data.num_nodes}",
                f"edges={data.num_edges}",
                (
                    "target_h="
                    f"{target_homophily:.3f}"
                ),
                (
                    "realized_h="
                    f"{data.realized_homophily:.3f}"
                ),
                (
                    "degree="
                    f"{data.realized_average_degree:.2f}"
                ),
            )

    manifest_path = (
        args.out_dir / "manifest.csv"
    )

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=manifest_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("Saved:", manifest_path)


if __name__ == "__main__":
    main()
