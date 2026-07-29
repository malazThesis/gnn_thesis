from __future__ import annotations

import re
import shutil
from pathlib import Path


TARGET = Path("analysis_metrics/oversmoothing.py")
BACKUP = Path(
    "analysis_metrics/oversmoothing.py.before_metric_upgrade"
)
MARKER = "def normalized_dirichlet_energy("


def main():
    if not TARGET.exists():
        raise FileNotFoundError(TARGET)

    text = TARGET.read_text(encoding="utf-8")

    if MARKER in text:
        print("Metric upgrade already installed.")
        return

    if not BACKUP.exists():
        shutil.copy2(TARGET, BACKUP)
        print("Backup:", BACKUP)

    old_columns = '''    "dirichlet_energy",
    "effective_rank",
    "mean_embedding_norm",
'''

    new_columns = '''    "dirichlet_energy",
    "normalized_dirichlet_energy",
    "embedding_dim",
    "effective_rank",
    "effective_rank_ratio",
    "mean_embedding_norm",
'''

    if old_columns not in text:
        raise RuntimeError(
            "Metric column insertion point not found."
        )

    text = text.replace(
        old_columns,
        new_columns,
        1,
    )

    effective_rank_marker = "\ndef effective_rank(\n"

    normalized_function = '''
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


'''

    if effective_rank_marker not in text:
        raise RuntimeError(
            "effective_rank insertion point not found."
        )

    text = text.replace(
        effective_rank_marker,
        "\n" + normalized_function + "def effective_rank(\n",
        1,
    )

    old_init = '''        self.embeddings: dict[str, Tensor] = {}
        self._handles = []
        self.enabled = False
'''

    new_init = '''        self.embeddings: dict[str, Tensor] = {}
        self._call_counts: dict[str, int] = {}
        self._handles = []
        self.enabled = False
'''

    if old_init not in text:
        raise RuntimeError(
            "Recorder initialization block not found."
        )

    text = text.replace(
        old_init,
        new_init,
        1,
    )

    old_hook = '''            if tensor is not None and tensor.ndim == 2:
                self.embeddings[name] = tensor.detach()
'''

    new_hook = '''            if tensor is not None and tensor.ndim == 2:
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
'''

    if old_hook not in text:
        raise RuntimeError(
            "Recorder hook block not found."
        )

    text = text.replace(
        old_hook,
        new_hook,
        1,
    )

    old_clear = '''    def clear(self) -> None:
        self.embeddings.clear()
'''

    new_clear = '''    def clear(self) -> None:
        self.embeddings.clear()
        self._call_counts.clear()
'''

    if old_clear not in text:
        raise RuntimeError(
            "Recorder clear block not found."
        )

    text = text.replace(
        old_clear,
        new_clear,
        1,
    )

    old_close = '''        self._handles.clear()
        self.embeddings.clear()
        self.enabled = False
'''

    new_close = '''        self._handles.clear()
        self.embeddings.clear()
        self._call_counts.clear()
        self.enabled = False
'''

    if old_close in text:
        text = text.replace(
            old_close,
            new_close,
            1,
        )

    replacement_function = '''@torch.no_grad()
def collect_layer_metrics(
    *,
    embeddings: dict[str, Tensor],
    edge_index: Tensor,
    metadata: dict[str, Any],
    epoch: int,
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
'''

    pattern = re.compile(
        r"@torch\.no_grad\(\)\n"
        r"def collect_layer_metrics\(.*?"
        r"\n    return rows\n",
        flags=re.DOTALL,
    )

    text, count = pattern.subn(
        replacement_function,
        text,
        count=1,
    )

    if count != 1:
        raise RuntimeError(
            "collect_layer_metrics replacement failed."
        )

    compile(
        text,
        str(TARGET),
        "exec",
    )

    TARGET.write_text(
        text,
        encoding="utf-8",
    )

    print("Updated:", TARGET)


if __name__ == "__main__":
    main()
