from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


RUN_DIR = Path(
    "runs/csbm_lowlabel_h01_parity"
)

METRIC_OUTPUT = (
    RUN_DIR / "csbm_h01_metric_timing.csv"
)
HISTORY_OUTPUT = (
    RUN_DIR / "csbm_h01_history_timing.csv"
)

SELECTED_DEPTHS = [7, 8, 9, 10]
SELECTED_EPOCHS = [1, 4, 8, 32, 200]


def module_number(name: object) -> int:
    match = re.search(
        r"\.(\d+)(?:#\d+)?$",
        str(name),
    )

    return int(match.group(1)) if match else -1


def select_hidden(
    frame: pd.DataFrame,
    *,
    model: str,
    hidden_channels: int,
) -> pd.Series:
    frame = frame[
        frame["layer_name"].notna()
    ].copy()

    if model.endswith("PairNorm"):
        candidates = frame[
            frame["layer_name"]
            .astype(str)
            .str.startswith("pns.")
        ].copy()
    else:
        candidates = frame[
            (
                frame["layer_name"]
                .astype(str)
                .str.startswith("convs.")
            )
            & (
                frame["embedding_dim"]
                == hidden_channels
            )
        ].copy()

    if candidates.empty:
        raise RuntimeError(
            f"No hidden layer found for {model}"
        )

    candidates["module_number"] = (
        candidates["layer_name"]
        .map(module_number)
    )

    return candidates.sort_values(
        ["module_number", "layer_index"]
    ).iloc[-1]


def main() -> None:
    metric_rows = []
    history_frames = []

    summary_files = sorted(
        RUN_DIR.glob("*_summary.json")
    )

    if len(summary_files) != 120:
        raise RuntimeError(
            f"Expected 120 summaries, "
            f"found {len(summary_files)}"
        )

    for summary_path in summary_files:
        with summary_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            summary = json.load(handle)

        metadata = {
            "feature_signal": float(
                summary["feature_signal"]
            ),
            "model": str(summary["model"]),
            "num_layers": int(
                summary["num_layers"]
            ),
            "graph_seed": int(
                summary["graph_seed"]
            ),
            "best_epoch": int(
                summary["best_epoch"]
            ),
            "best_test_acc": float(
                summary[
                    "best_test_acc_at_best_val"
                ]
            ),
        }

        metric_path = summary_path.with_name(
            summary_path.name.replace(
                "_summary.json",
                "_oversmoothing.csv",
            )
        )

        history_path = summary_path.with_name(
            summary_path.name.replace(
                "_summary.json",
                "_history.csv",
            )
        )

        metrics = pd.read_csv(metric_path)

        for epoch, epoch_frame in metrics.groupby(
            "epoch"
        ):
            hidden = select_hidden(
                epoch_frame,
                model=metadata["model"],
                hidden_channels=int(
                    summary["hidden_channels"]
                ),
            )

            metric_rows.append(
                {
                    **metadata,
                    "epoch": int(epoch),
                    "pairwise_distance": float(
                        hidden[
                            "mean_pairwise_cosine_distance"
                        ]
                    ),
                    "edge_distance": float(
                        hidden[
                            "mean_edge_cosine_distance"
                        ]
                    ),
                    "rank_ratio": float(
                        hidden[
                            "effective_rank_ratio"
                        ]
                    ),
                    "normalized_dirichlet": float(
                        hidden[
                            "normalized_dirichlet_energy"
                        ]
                    ),
                }
            )

        history = pd.read_csv(history_path)

        for column, value in metadata.items():
            history[column] = value

        history_frames.append(history)

    metric_runs = pd.DataFrame(metric_rows)

    history_runs = pd.concat(
        history_frames,
        ignore_index=True,
    )

    metric_summary = (
        metric_runs
        .groupby(
            [
                "feature_signal",
                "model",
                "num_layers",
                "epoch",
            ],
            as_index=False,
        )
        .agg(
            pairwise_mean=(
                "pairwise_distance",
                "mean",
            ),
            pairwise_std=(
                "pairwise_distance",
                "std",
            ),
            edge_mean=(
                "edge_distance",
                "mean",
            ),
            rank_ratio_mean=(
                "rank_ratio",
                "mean",
            ),
            normalized_dirichlet_mean=(
                "normalized_dirichlet",
                "mean",
            ),
        )
    )

    history_summary = (
        history_runs
        .groupby(
            [
                "feature_signal",
                "model",
                "num_layers",
                "epoch",
            ],
            as_index=False,
        )
        .agg(
            loss_mean=("loss", "mean"),
            train_acc_mean=(
                "train_acc",
                "mean",
            ),
            val_acc_mean=(
                "val_acc",
                "mean",
            ),
            test_acc_mean=(
                "test_acc",
                "mean",
            ),
            gradient_norm_mean=(
                "gradient_norm",
                "mean",
            ),
        )
    )

    metric_summary.to_csv(
        METRIC_OUTPUT,
        index=False,
    )

    history_summary.to_csv(
        HISTORY_OUTPUT,
        index=False,
    )

    metric_display = metric_summary[
        (
            metric_summary["num_layers"]
            .isin(SELECTED_DEPTHS)
        )
        & (
            metric_summary["epoch"]
            .isin(SELECTED_EPOCHS)
        )
    ]

    history_display = history_summary[
        (
            history_summary["num_layers"]
            .isin(SELECTED_DEPTHS)
        )
        & (
            history_summary["epoch"]
            .isin(SELECTED_EPOCHS)
        )
    ]

    print(
        "\n=== Representation timing ==="
    )
    print(
        metric_display[
            [
                "feature_signal",
                "model",
                "num_layers",
                "epoch",
                "pairwise_mean",
                "edge_mean",
                "rank_ratio_mean",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print(
        "\n=== Training timing ==="
    )
    print(
        history_display[
            [
                "feature_signal",
                "model",
                "num_layers",
                "epoch",
                "loss_mean",
                "train_acc_mean",
                "val_acc_mean",
                "gradient_norm_mean",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print("\nSaved:", METRIC_OUTPUT)
    print("Saved:", HISTORY_OUTPUT)


if __name__ == "__main__":
    main()
