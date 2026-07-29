import tempfile
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch_geometric.nn import GCNConv

from analysis_metrics.oversmoothing import (
    LayerEmbeddingRecorder,
    append_metrics_csv,
    collect_layer_metrics,
    model_gradient_norm,
)


class SmallGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(16, 32)
        self.conv2 = GCNConv(32, 3)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


def main():
    torch.manual_seed(1)

    num_nodes = 100

    x = torch.randn(num_nodes, 16)

    source = torch.arange(num_nodes)
    target = torch.roll(source, shifts=-1)

    edge_index = torch.stack(
        [
            torch.cat([source, target]),
            torch.cat([target, source]),
        ],
        dim=0,
    )

    y = torch.randint(
        low=0,
        high=3,
        size=(num_nodes,),
    )

    model = SmallGCN()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )

    recorder = LayerEmbeddingRecorder(model)

    recorder.start()
    logits = model(x, edge_index)
    recorder.stop()

    loss = nn.functional.cross_entropy(
        logits,
        y,
    )

    optimizer.zero_grad()
    loss.backward()

    gradient_norm = model_gradient_norm(model)

    rows = collect_layer_metrics(
        embeddings=recorder.embeddings,
        edge_index=edge_index,
        epoch=1,
        gradient_norm=gradient_norm,
        metadata={
            "dataset": "SmokeGraph",
            "model": "SmallGCN",
            "seed": 1,
            "split_idx": 0,
            "num_layers": 2,
            "hidden_channels": 32,
        },
    )

    output_path = (
        Path(tempfile.gettempdir())
        / "oversmoothing_metrics_smoke.csv"
    )

    if output_path.exists():
        output_path.unlink()

    append_metrics_csv(
        output_path,
        rows,
    )

    recorder.close()

    print(f"Loss: {loss.item():.6f}")
    print(f"Gradient norm: {gradient_norm:.6f}")
    print(f"Recorded layers: {len(rows)}")
    print(f"Saved: {output_path}")

    for row in rows:
        print(
            row["layer_name"],
            "pairwise_distance=",
            f"{row['mean_pairwise_cosine_distance']:.6f}",
            "edge_distance=",
            f"{row['mean_edge_cosine_distance']:.6f}",
            "effective_rank=",
            f"{row['effective_rank']:.6f}",
        )


if __name__ == "__main__":
    main()
