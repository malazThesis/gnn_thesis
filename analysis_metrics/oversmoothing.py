from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch_geometric.nn import MessagePassing


METRIC_COLUMNS = [
    "dataset",
    "model",
    "seed",
    "split_idx",
    "num_layers",
    "hidden_channels",
    "epoch",
    "layer_index",
    "layer_name",
    "num_nodes_used",
    "num_labeled_nodes_used",
    "num_classes_used",
    "class_metric_scope",
    "within_class_cosine_distance",
    "between_class_cosine_distance",
    "class_separation_margin",
    "class_centroid_cosine_distance",
    "fisher_discriminant_ratio",
    "mean_pairwise_cosine_distance",
    "mean_edge_cosine_distance",
    "dirichlet_energy",
    "normalized_dirichlet_energy",
    "embedding_dim",
    "effective_rank",
    "effective_rank_ratio",
    "mean_embedding_norm",
    "mean_feature_variance",
    "gradient_norm",
]


def _first_tensor(value: Any) -> Tensor | None:
    """Extract the first tensor from common module-output structures."""
    if isinstance(value, Tensor):
        return value

    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor

    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor

    return None


def _sample_rows(
    x: Tensor,
    max_nodes: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    if x.size(0) <= max_nodes:
        return x

    indices = torch.randperm(
        x.size(0),
        device=x.device,
        generator=generator,
    )[:max_nodes]

    return x[indices]


def mean_pairwise_cosine_distance(
    x: Tensor,
    max_nodes: int = 2048,
    eps: float = 1e-12,
) -> float:
    """
    Mean cosine distance across sampled node pairs.

    Smaller values indicate increasingly similar node representations.
    """
    if x.ndim != 2 or x.size(0) < 2:
        return math.nan

    sampled = _sample_rows(x, max_nodes=max_nodes)
    sampled = F.normalize(sampled.float(), p=2, dim=-1, eps=eps)

    similarity = sampled @ sampled.T
    n = similarity.size(0)

    off_diagonal_sum = similarity.sum() - similarity.diagonal().sum()
    mean_similarity = off_diagonal_sum / (n * (n - 1))

    return float((1.0 - mean_similarity).item())


def _sample_labeled_rows(
    x: Tensor,
    labels: Tensor,
    label_mask: Tensor | None,
    max_nodes: int,
) -> tuple[Tensor, Tensor]:
    """
    Stratified sampling of labeled node representations.

    Sampling is approximately balanced over the available classes,
    which prevents large classes from completely dominating the
    pairwise diagnostic metrics.
    """
    if x.ndim != 2:
        return (
            x.new_empty((0, 0)),
            torch.empty(
                0,
                dtype=torch.long,
                device=x.device,
            ),
        )

    labels = labels.reshape(-1).to(
        device=x.device,
        dtype=torch.long,
    )

    if labels.numel() != x.size(0):
        raise ValueError(
            "Labels and embeddings have different "
            "numbers of nodes."
        )

    if label_mask is None:
        mask = torch.ones(
            x.size(0),
            dtype=torch.bool,
            device=x.device,
        )
    else:
        mask = label_mask.reshape(-1).to(
            device=x.device,
            dtype=torch.bool,
        )

        if mask.numel() != x.size(0):
            raise ValueError(
                "Label mask and embeddings have different "
                "numbers of nodes."
            )

    # Negative labels are conventionally treated as unlabeled.
    valid_indices = torch.nonzero(
        mask & (labels >= 0),
        as_tuple=False,
    ).flatten()

    if valid_indices.numel() == 0:
        return (
            x.new_empty((0, x.size(1))),
            labels.new_empty((0,)),
        )

    valid_labels = labels[valid_indices]
    classes = torch.unique(
        valid_labels,
        sorted=True,
    )

    if valid_indices.numel() <= max_nodes:
        return (
            x[valid_indices],
            valid_labels,
        )

    number_classes = int(classes.numel())
    base_quota = max_nodes // max(
        1,
        number_classes,
    )
    remainder = max_nodes % max(
        1,
        number_classes,
    )

    selected_chunks = []
    leftover_chunks = []

    for class_position, class_value in enumerate(
        classes
    ):
        class_indices = valid_indices[
            valid_labels == class_value
        ]

        permutation = torch.randperm(
            class_indices.numel(),
            device=x.device,
        )

        class_indices = class_indices[
            permutation
        ]

        quota = base_quota + (
            1
            if class_position < remainder
            else 0
        )

        take = min(
            quota,
            int(class_indices.numel()),
        )

        if take > 0:
            selected_chunks.append(
                class_indices[:take]
            )

        if take < class_indices.numel():
            leftover_chunks.append(
                class_indices[take:]
            )

    selected = (
        torch.cat(selected_chunks)
        if selected_chunks
        else valid_indices.new_empty((0,))
    )

    remaining = max_nodes - int(
        selected.numel()
    )

    if remaining > 0 and leftover_chunks:
        leftovers = torch.cat(
            leftover_chunks
        )

        permutation = torch.randperm(
            leftovers.numel(),
            device=x.device,
        )

        selected = torch.cat(
            [
                selected,
                leftovers[
                    permutation[:remaining]
                ],
            ]
        )

    return (
        x[selected],
        labels[selected],
    )


@torch.no_grad()
def class_separation_metrics(
    x: Tensor,
    labels: Tensor | None,
    label_mask: Tensor | None = None,
    max_nodes: int = 2048,
    eps: float = 1e-12,
) -> dict[str, float | int]:
    """
    Label-aware representation diagnostics.

    The cosine metrics are scale invariant. The Fisher ratio is the
    ratio of between-class to within-class scatter and is therefore
    also invariant to uniform global rescaling.

    Positive class_separation_margin means that representations from
    different classes are farther apart than representations from the
    same class.
    """
    empty_result: dict[str, float | int] = {
        "num_labeled_nodes_used": 0,
        "num_classes_used": 0,
        "within_class_cosine_distance":
            math.nan,
        "between_class_cosine_distance":
            math.nan,
        "class_separation_margin":
            math.nan,
        "class_centroid_cosine_distance":
            math.nan,
        "fisher_discriminant_ratio":
            math.nan,
    }

    if (
        labels is None
        or x.ndim != 2
        or x.size(0) < 2
    ):
        return empty_result

    sampled_x, sampled_labels = (
        _sample_labeled_rows(
            x,
            labels=labels,
            label_mask=label_mask,
            max_nodes=max_nodes,
        )
    )

    number_nodes = int(
        sampled_x.size(0)
    )

    if number_nodes < 2:
        return {
            **empty_result,
            "num_labeled_nodes_used":
                number_nodes,
        }

    classes = torch.unique(
        sampled_labels,
        sorted=True,
    )

    number_classes = int(
        classes.numel()
    )

    if number_classes < 2:
        return {
            **empty_result,
            "num_labeled_nodes_used":
                number_nodes,
            "num_classes_used":
                number_classes,
        }

    normalized = F.normalize(
        sampled_x.float(),
        p=2,
        dim=-1,
        eps=eps,
    )

    similarity = normalized @ normalized.T
    distance = 1.0 - similarity

    same_class = (
        sampled_labels[:, None]
        == sampled_labels[None, :]
    )

    diagonal = torch.eye(
        number_nodes,
        dtype=torch.bool,
        device=x.device,
    )

    within_mask = (
        same_class
        & ~diagonal
    )
    between_mask = ~same_class

    if within_mask.any():
        within_distance = float(
            distance[
                within_mask
            ].mean().item()
        )
    else:
        within_distance = math.nan

    if between_mask.any():
        between_distance = float(
            distance[
                between_mask
            ].mean().item()
        )
    else:
        between_distance = math.nan

    if (
        not math.isnan(within_distance)
        and not math.isnan(
            between_distance
        )
    ):
        separation_margin = (
            between_distance
            - within_distance
        )
    else:
        separation_margin = math.nan

    centroids = []

    for class_value in classes:
        class_rows = normalized[
            sampled_labels == class_value
        ]

        centroid = class_rows.mean(
            dim=0
        )

        centroids.append(centroid)

    centroid_matrix = torch.stack(
        centroids,
        dim=0,
    )

    centroid_matrix = F.normalize(
        centroid_matrix,
        p=2,
        dim=-1,
        eps=eps,
    )

    centroid_similarity = (
        centroid_matrix
        @ centroid_matrix.T
    )

    centroid_count = int(
        centroid_matrix.size(0)
    )

    centroid_off_diagonal = (
        centroid_similarity.sum()
        - centroid_similarity.diagonal().sum()
    )

    centroid_mean_similarity = (
        centroid_off_diagonal
        / (
            centroid_count
            * (centroid_count - 1)
        )
    )

    centroid_distance = float(
        (
            1.0
            - centroid_mean_similarity
        ).item()
    )

    float_x = sampled_x.float()
    grand_mean = float_x.mean(
        dim=0,
        keepdim=True,
    )

    within_scatter = torch.zeros(
        (),
        dtype=torch.float32,
        device=x.device,
    )
    between_scatter = torch.zeros(
        (),
        dtype=torch.float32,
        device=x.device,
    )

    for class_value in classes:
        class_rows = float_x[
            sampled_labels == class_value
        ]

        class_mean = class_rows.mean(
            dim=0,
            keepdim=True,
        )

        within_scatter = (
            within_scatter
            + (
                class_rows
                - class_mean
            ).pow(2).sum()
        )

        between_scatter = (
            between_scatter
            + class_rows.size(0)
            * (
                class_mean
                - grand_mean
            ).pow(2).sum()
        )

    fisher_ratio = float(
        (
            between_scatter
            / within_scatter.clamp_min(eps)
        ).item()
    )

    return {
        "num_labeled_nodes_used":
            number_nodes,
        "num_classes_used":
            number_classes,
        "within_class_cosine_distance":
            within_distance,
        "between_class_cosine_distance":
            between_distance,
        "class_separation_margin":
            separation_margin,
        "class_centroid_cosine_distance":
            centroid_distance,
        "fisher_discriminant_ratio":
            fisher_ratio,
    }


def mean_edge_cosine_distance(
    x: Tensor,
    edge_index: Tensor,
    max_edges: int = 200_000,
    eps: float = 1e-12,
) -> float:
    """
    Mean cosine distance between connected nodes.

    This is an edge-local MAD-style measure.
    """
    if x.ndim != 2 or edge_index.numel() == 0:
        return math.nan

    source = edge_index[0]
    target = edge_index[1]

    num_edges = source.numel()

    if num_edges > max_edges:
        indices = torch.randperm(
            num_edges,
            device=edge_index.device,
        )[:max_edges]

        source = source[indices]
        target = target[indices]

    source_x = F.normalize(
        x[source].float(),
        p=2,
        dim=-1,
        eps=eps,
    )
    target_x = F.normalize(
        x[target].float(),
        p=2,
        dim=-1,
        eps=eps,
    )

    cosine_similarity = (source_x * target_x).sum(dim=-1)
    cosine_distance = 1.0 - cosine_similarity

    return float(cosine_distance.mean().item())


def dirichlet_energy(
    x: Tensor,
    edge_index: Tensor,
    max_edges: int = 200_000,
) -> float:
    """
    Mean squared embedding difference across graph edges.

    Smaller values indicate smoother representations.
    """
    if x.ndim != 2 or edge_index.numel() == 0:
        return math.nan

    source = edge_index[0]
    target = edge_index[1]

    num_edges = source.numel()

    if num_edges > max_edges:
        indices = torch.randperm(
            num_edges,
            device=edge_index.device,
        )[:max_edges]

        source = source[indices]
        target = target[indices]

    differences = x[source].float() - x[target].float()
    squared_distance = differences.pow(2).sum(dim=-1)

    return float(squared_distance.mean().item())



def normalized_dirichlet_energy(
    x: Tensor,
    edge_index: Tensor,
    max_edges: int = 200_000,
    eps: float = 1e-12,
) -> float:
    """
    Dirichlet energy divided by mean squared embedding norm.

    Unlike raw Dirichlet energy, this value is less sensitive
    to a global change in embedding magnitude.
    """
    raw_energy = dirichlet_energy(
        x,
        edge_index,
        max_edges=max_edges,
    )

    if math.isnan(raw_energy):
        return math.nan

    mean_squared_norm = (
        x.float()
        .pow(2)
        .sum(dim=-1)
        .mean()
        .clamp_min(eps)
    )

    return float(
        raw_energy
        / float(mean_squared_norm.item())
    )


def effective_rank(
    x: Tensor,
    max_nodes: int = 2048,
    eps: float = 1e-12,
) -> float:
    """
    Entropy-based effective rank of the centered embedding matrix.

    A decreasing effective rank indicates dimensional collapse.
    """
    if x.ndim != 2 or min(x.shape) < 2:
        return math.nan

    sampled = _sample_rows(x, max_nodes=max_nodes)
    sampled = sampled.float()

    # The SVD is deliberately computed on CPU to avoid retaining
    # additional GPU memory during training.
    sampled = sampled.detach().cpu()
    sampled = sampled - sampled.mean(dim=0, keepdim=True)

    singular_values = torch.linalg.svdvals(sampled)
    singular_values = singular_values[
        singular_values > eps
    ]

    if singular_values.numel() == 0:
        return 0.0

    probabilities = singular_values / singular_values.sum()
    entropy = -(
        probabilities
        * torch.log(probabilities.clamp_min(eps))
    ).sum()

    return float(torch.exp(entropy).item())


def mean_embedding_norm(x: Tensor) -> float:
    if x.ndim != 2 or x.numel() == 0:
        return math.nan

    return float(
        x.float().norm(p=2, dim=-1).mean().item()
    )


def mean_feature_variance(x: Tensor) -> float:
    if x.ndim != 2 or x.size(0) < 2:
        return math.nan

    return float(
        x.float().var(dim=0, unbiased=False).mean().item()
    )


def model_gradient_norm(model: Module) -> float:
    squared_norm = 0.0
    found_gradient = False

    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        found_gradient = True
        gradient = parameter.grad.detach().float()
        squared_norm += float(
            gradient.pow(2).sum().item()
        )

    if not found_gradient:
        return math.nan

    return math.sqrt(squared_norm)


class LayerEmbeddingRecorder:
    """
    Records outputs of all torch_geometric MessagePassing modules.

    Usage:
        recorder = LayerEmbeddingRecorder(model)
        recorder.clear()
        logits = model(...)
        embeddings = recorder.embeddings
        recorder.close()
    """

    def __init__(self, model: Module):
        self.embeddings: dict[str, Tensor] = {}
        self._call_counts: dict[str, int] = {}
        self._handles = []
        self.enabled = False
        self.enabled = False

        for name, module in model.named_modules():
            is_message_passing = isinstance(
                module,
                MessagePassing,
            )
            is_pairnorm = (
                bool(name)
                and module.__class__.__name__ in {
                    "PairNorm",
                    "CenterNorm",
                    "ScaleNorm",
                }
            )

            if not (
                is_message_passing
                or is_pairnorm
            ):
                continue

            handle = module.register_forward_hook(
                self._make_hook(name)
            )
            self._handles.append(handle)

        if not self._handles:
            raise RuntimeError(
                "No torch_geometric MessagePassing layers "
                "were found in the model."
            )

    def _make_hook(self, name: str):
        def hook(
            module: Module,
            inputs: tuple[Any, ...],
            output: Any,
        ) -> None:
            if not self.enabled:
                return

            tensor = _first_tensor(output)

            if tensor is not None and tensor.ndim == 2:
                call_number = (
                    self._call_counts.get(name, 0)
                    + 1
                )
                self._call_counts[name] = call_number

                key = (
                    name
                    if call_number == 1
                    else f"{name}#{call_number}"
                )

                self.embeddings[key] = tensor.detach()

        return hook

    def clear(self) -> None:
        self.embeddings.clear()
        self._call_counts.clear()

    def start(self) -> None:
        self.clear()
        self.enabled = True

    def stop(self) -> None:
        self.enabled = False

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()

        self._handles.clear()
        self.embeddings.clear()
        self._call_counts.clear()
        self.enabled = False
        self.enabled = False


@torch.no_grad()
def collect_layer_metrics(
    *,
    embeddings: dict[str, Tensor],
    edge_index: Tensor,
    metadata: dict[str, Any],
    epoch: int,
    labels: Tensor | None = None,
    label_mask: Tensor | None = None,
    gradient_norm: float = math.nan,
    max_pairwise_nodes: int = 2048,
    max_rank_nodes: int = 2048,
    max_edges: int = 200_000,
) -> list[dict[str, Any]]:
    rows = []

    for layer_index, (layer_name, embedding) in enumerate(
        embeddings.items(),
        start=1,
    ):
        x = embedding.detach()

        rank = effective_rank(
            x,
            max_nodes=max_rank_nodes,
        )

        rank_denominator = max(
            1,
            min(
                int(x.size(0)),
                int(x.size(1)),
            ),
        )

        raw_energy = dirichlet_energy(
            x,
            edge_index,
            max_edges=max_edges,
        )

        normalized_energy = (
            normalized_dirichlet_energy(
                x,
                edge_index,
                max_edges=max_edges,
            )
        )

        class_metrics = (
            class_separation_metrics(
                x,
                labels=labels,
                label_mask=label_mask,
                max_nodes=max_pairwise_nodes,
            )
        )

        row = {
            "dataset": metadata.get("dataset"),
            "model": metadata.get("model"),
            "seed": metadata.get("seed"),
            "split_idx": metadata.get("split_idx"),
            "num_layers": metadata.get("num_layers"),
            "hidden_channels": metadata.get(
                "hidden_channels"
            ),
            "epoch": epoch,
            "layer_index": layer_index,
            "layer_name": layer_name,
            "num_nodes_used": min(
                int(x.size(0)),
                max_pairwise_nodes,
            ),
            **class_metrics,
            "class_metric_scope":
                metadata.get(
                    "class_metric_scope"
                ),
            "mean_pairwise_cosine_distance":
                mean_pairwise_cosine_distance(
                    x,
                    max_nodes=max_pairwise_nodes,
                ),
            "mean_edge_cosine_distance":
                mean_edge_cosine_distance(
                    x,
                    edge_index,
                    max_edges=max_edges,
                ),
            "dirichlet_energy": raw_energy,
            "normalized_dirichlet_energy":
                normalized_energy,
            "embedding_dim": int(x.size(1)),
            "effective_rank": rank,
            "effective_rank_ratio": (
                rank / rank_denominator
                if not math.isnan(rank)
                else math.nan
            ),
            "mean_embedding_norm":
                mean_embedding_norm(x),
            "mean_feature_variance":
                mean_feature_variance(x),
            "gradient_norm": gradient_norm,
        }

        rows.append(row)

    return rows


def append_metrics_csv(
    path: str | Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return

    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = output_path.exists()

    with output_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=METRIC_COLUMNS,
            extrasaction="ignore",
        )

        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)


def default_metric_epochs(total_epochs: int) -> set[int]:
    """
    Log densely at the start and sparsely later.
    """
    candidates = {
        1,
        2,
        4,
        8,
        16,
        32,
        50,
        75,
        100,
        125,
        150,
        175,
        total_epochs,
    }

    return {
        epoch
        for epoch in candidates
        if 1 <= epoch <= total_epochs
    }


def seed_metric_sampling(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
