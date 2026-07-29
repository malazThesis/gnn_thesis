from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh


CONFIG_PATH = Path(
    "configs/csbm_classmetrics_temporal_fs050.csv"
)

OUTCOME_PATH = Path(
    "runs/csbm_classmetrics_temporal_fs050/"
    "csbm_l8_baseline_recovery_by_seed.csv"
)

OUTPUT_PATH = Path(
    "runs/csbm_classmetrics_temporal_fs050/"
    "csbm_graph_structure_recovery.csv"
)


def load_graph(path: Path):
    try:
        loaded = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        loaded = torch.load(
            path,
            map_location="cpu",
        )

    if (
        isinstance(loaded, dict)
        and "data" in loaded
    ):
        return loaded["data"]

    return loaded


def as_mask(
    value: torch.Tensor,
    split_idx: int = 0,
) -> np.ndarray:
    mask = value.detach().cpu()

    if mask.ndim == 2:
        mask = mask[:, split_idx]

    return mask.numpy().astype(bool)


def propagate(
    matrix: sparse.spmatrix,
    vector: np.ndarray,
    steps: int,
) -> np.ndarray:
    result = vector.astype(
        np.float64,
        copy=True,
    )

    for _ in range(steps):
        result = matrix @ result

    return np.asarray(result).reshape(-1)


def cosine_abs(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    denominator = (
        np.linalg.norm(left)
        * np.linalg.norm(right)
    )

    if denominator <= 0.0:
        return math.nan

    return float(
        abs(
            np.dot(left, right)
            / denominator
        )
    )


def orientation_free_accuracy(
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    prediction = scores >= 0
    target = labels > 0

    accuracy = float(
        np.mean(prediction == target)
    )

    return max(
        accuracy,
        1.0 - accuracy,
    )


def main() -> None:
    config = pd.read_csv(CONFIG_PATH)

    # Only one row per underlying graph is needed.
    graphs = (
        config[
            (config["model"] == "GraphSAGE")
            & (config["num_layers"] == 8)
        ]
        .drop_duplicates("data_path")
        .copy()
    )

    outcomes = pd.read_csv(OUTCOME_PATH)

    rows = []

    for record in graphs.itertuples(
        index=False
    ):
        graph_seed = int(record.seed)
        path = Path(record.data_path)

        data = load_graph(path)

        labels = (
            data.y.detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )

        num_nodes = int(
            data.num_nodes
        )

        edge_index = (
            data.edge_index.detach()
            .cpu()
            .numpy()
        )

        source = edge_index[0]
        target = edge_index[1]

        # Unique undirected edges without self-loops.
        lower = np.minimum(
            source,
            target,
        )
        upper = np.maximum(
            source,
            target,
        )

        valid = lower != upper
        lower = lower[valid]
        upper = upper[valid]

        keys = (
            lower.astype(np.int64)
            * num_nodes
            + upper.astype(np.int64)
        )

        _, unique_positions = np.unique(
            keys,
            return_index=True,
        )

        lower = lower[unique_positions]
        upper = upper[unique_positions]

        number_edges = len(lower)

        adjacency = sparse.coo_matrix(
            (
                np.ones(
                    2 * number_edges,
                    dtype=np.float64,
                ),
                (
                    np.concatenate(
                        [lower, upper]
                    ),
                    np.concatenate(
                        [upper, lower]
                    ),
                ),
            ),
            shape=(num_nodes, num_nodes),
        ).tocsr()

        degrees = np.asarray(
            adjacency.sum(axis=1)
        ).reshape(-1)

        realized_homophily = float(
            np.mean(
                labels[lower]
                == labels[upper]
            )
        )

        number_components, component_labels = (
            csgraph.connected_components(
                adjacency,
                directed=False,
            )
        )

        component_sizes = np.bincount(
            component_labels
        )

        giant_component_fraction = float(
            component_sizes.max()
            / num_nodes
        )

        adjacency_loop = (
            adjacency
            + sparse.identity(
                num_nodes,
                format="csr",
            )
        )

        loop_degree = np.asarray(
            adjacency_loop.sum(axis=1)
        ).reshape(-1)

        inverse_sqrt = np.power(
            loop_degree,
            -0.5,
            where=loop_degree > 0,
        )

        normalization = sparse.diags(
            inverse_sqrt
        )

        propagation = (
            normalization
            @ adjacency_loop
            @ normalization
        ).tocsr()

        classes = np.unique(labels)

        if len(classes) != 2:
            raise RuntimeError(
                "Expected binary labels."
            )

        class_signal = np.where(
            labels == classes[0],
            -1.0,
            1.0,
        )

        class_signal -= (
            class_signal.mean()
        )

        propagated_once = (
            propagation @ class_signal
        )

        class_rayleigh = float(
            np.dot(
                class_signal,
                propagated_once,
            )
            / np.dot(
                class_signal,
                class_signal,
            )
        )

        diagnostics = {}

        for steps in [1, 6, 7]:
            propagated = propagate(
                propagation,
                class_signal,
                steps,
            )

            diagnostics[
                f"class_signal_retention_{steps}"
            ] = float(
                np.linalg.norm(propagated)
                / np.linalg.norm(
                    class_signal
                )
            )

            diagnostics[
                f"class_alignment_abs_{steps}"
            ] = cosine_abs(
                class_signal,
                propagated,
            )

        train_mask = as_mask(
            data.train_mask,
            split_idx=0,
        )

        test_mask = as_mask(
            data.test_mask,
            split_idx=0,
        )

        train_neighbor_counts = (
            adjacency
            @ train_mask.astype(
                np.float64
            )
        )

        train_signal = np.zeros(
            num_nodes,
            dtype=np.float64,
        )

        train_signal[train_mask] = (
            class_signal[train_mask]
        )

        label_propagation = {}

        for steps in [6, 7]:
            scores = propagate(
                propagation,
                train_signal,
                steps,
            )

            label_propagation[
                f"train_label_prop_accuracy_{steps}"
            ] = orientation_free_accuracy(
                scores[test_mask],
                class_signal[test_mask],
            )

        try:
            eigenvalues = eigsh(
                propagation,
                k=6,
                which="LM",
                return_eigenvectors=False,
            )

            eigenvalues = np.asarray(
                eigenvalues
            )

            eigenvalues = eigenvalues[
                np.argsort(
                    np.abs(eigenvalues)
                )[::-1]
            ]

            spectral_radius = float(
                abs(eigenvalues[0])
            )

            second_eigenvalue_abs = float(
                abs(eigenvalues[1])
            )
        except Exception:
            spectral_radius = math.nan
            second_eigenvalue_abs = math.nan

        rows.append(
            {
                "graph_seed": graph_seed,
                "data_path": str(path),
                "num_nodes": num_nodes,
                "num_undirected_edges":
                    number_edges,
                "realized_homophily":
                    realized_homophily,
                "mean_degree":
                    float(degrees.mean()),
                "degree_std":
                    float(degrees.std()),
                "min_degree":
                    float(degrees.min()),
                "max_degree":
                    float(degrees.max()),
                "isolated_nodes":
                    int(
                        np.sum(degrees == 0)
                    ),
                "num_components":
                    int(number_components),
                "giant_component_fraction":
                    giant_component_fraction,
                "spectral_radius":
                    spectral_radius,
                "second_eigenvalue_abs":
                    second_eigenvalue_abs,
                "class_rayleigh":
                    class_rayleigh,
                "test_nodes_with_train_neighbor":
                    float(
                        np.mean(
                            train_neighbor_counts[
                                test_mask
                            ] > 0
                        )
                    ),
                "mean_train_neighbors_test":
                    float(
                        train_neighbor_counts[
                            test_mask
                        ].mean()
                    ),
                **diagnostics,
                **label_propagation,
            }
        )

    result = pd.DataFrame(rows)

    result = result.merge(
        outcomes[
            [
                "graph_seed",
                "outcome",
                "final_test_acc",
            ]
        ],
        on="graph_seed",
        validate="one_to_one",
    )

    result = result.sort_values(
        "graph_seed"
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        result.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.10f}",
        )
    )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
