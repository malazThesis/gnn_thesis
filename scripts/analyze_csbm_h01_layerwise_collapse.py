from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


RUN_DIR = Path(
    "runs/csbm_lowlabel_h01_parity"
)
PLOT_DIR = Path(
    "plots/csbm_lowlabel_h01_layerwise"
)
OUTPUT_CSV = (
    RUN_DIR
    / "csbm_h01_layerwise_collapse.csv"
)

DEPTHS = [7, 8]
EPOCHS = [1, 8, 32]


def module_number(name: object) -> int:
    match = re.search(
        r"\.(\d+)(?:#\d+)?$",
        str(name),
    )

    return int(match.group(1)) if match else -1


def load_layer_rows() -> pd.DataFrame:
    rows = []

    for summary_path in sorted(
        RUN_DIR.glob("*_summary.json")
    ):
        with summary_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            summary = json.load(handle)

        depth = int(summary["num_layers"])

        if depth not in DEPTHS:
            continue

        model = str(summary["model"])
        hidden_channels = int(
            summary["hidden_channels"]
        )

        metric_path = summary_path.with_name(
            summary_path.name.replace(
                "_summary.json",
                "_oversmoothing.csv",
            )
        )

        metrics = pd.read_csv(metric_path)

        metrics = metrics[
            metrics["epoch"].isin(EPOCHS)
            & metrics["layer_name"].notna()
        ].copy()

        if model.endswith("PairNorm"):
            metrics = metrics[
                metrics["layer_name"]
                .astype(str)
                .str.startswith("pns.")
            ].copy()
        else:
            metrics = metrics[
                (
                    metrics["layer_name"]
                    .astype(str)
                    .str.startswith("convs.")
                )
                & (
                    metrics["embedding_dim"]
                    == hidden_channels
                )
            ].copy()

        metrics["message_passing_stage"] = (
            metrics["layer_name"]
            .map(module_number)
            + 1
        )

        for metric in metrics.itertuples(
            index=False
        ):
            rows.append(
                {
                    "feature_signal": float(
                        summary["feature_signal"]
                    ),
                    "model": model,
                    "num_layers": depth,
                    "graph_seed": int(
                        summary["graph_seed"]
                    ),
                    "epoch": int(metric.epoch),
                    "message_passing_stage": int(
                        metric.message_passing_stage
                    ),
                    "pairwise_distance": float(
                        metric.mean_pairwise_cosine_distance
                    ),
                    "edge_distance": float(
                        metric.mean_edge_cosine_distance
                    ),
                    "rank_ratio": float(
                        metric.effective_rank_ratio
                    ),
                }
            )

    return pd.DataFrame(rows)


def aggregate(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    groups = [
        "feature_signal",
        "model",
        "num_layers",
        "epoch",
        "message_passing_stage",
    ]

    result = (
        frame.groupby(
            groups,
            as_index=False,
        )
        .agg(
            n=(
                "pairwise_distance",
                "count",
            ),
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
        )
    )

    result["pairwise_ci95"] = (
        result["pairwise_std"]
        / result["n"].pow(0.5)
        * result["n"].map(
            lambda n: float(
                stats.t.ppf(
                    0.975,
                    df=n - 1,
                )
            )
            if n > 1
            else 0.0
        )
    )

    return result


def plot_profiles(
    summary: pd.DataFrame,
) -> None:
    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for keys, frame in summary.groupby(
        ["feature_signal", "epoch"]
    ):
        feature_signal, epoch = keys

        plt.figure(figsize=(10, 6))

        for model_keys, model_frame in (
            frame.groupby(
                ["model", "num_layers"]
            )
        ):
            model, depth = model_keys

            model_frame = model_frame.sort_values(
                "message_passing_stage"
            )

            plt.errorbar(
                model_frame[
                    "message_passing_stage"
                ],
                model_frame["pairwise_mean"],
                yerr=model_frame[
                    "pairwise_ci95"
                ],
                marker="o",
                capsize=3,
                label=f"{model} L{depth}",
            )

        plt.xlabel("Message-passing stage")
        plt.ylabel(
            "Mean pairwise cosine distance"
        )
        plt.title(
            "Layerwise representation collapse "
            f"(FS={feature_signal:.2f}, "
            f"epoch={epoch})"
        )
        plt.legend()
        plt.tight_layout()

        path = (
            PLOT_DIR
            / (
                "layerwise_pairwise_"
                f"fs{int(feature_signal * 100):03d}"
                f"_epoch{epoch}.png"
            )
        )

        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print("Saved:", path)


def main() -> None:
    layer_rows = load_layer_rows()
    summary = aggregate(layer_rows)

    summary.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    plot_profiles(summary)

    print(
        "\n=== Layerwise profiles at epoch 1 ==="
    )

    print(
        summary[
            summary["epoch"] == 1
        ][
            [
                "feature_signal",
                "model",
                "num_layers",
                "message_passing_stage",
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

    print("\nSaved:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
