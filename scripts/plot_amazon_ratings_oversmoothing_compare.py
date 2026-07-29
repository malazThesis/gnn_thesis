from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RUN_DIR = Path(
    "runs/amazon_ratings_oversmoothing_compare"
)

PLOT_DIR = Path(
    "plots/amazon_ratings_oversmoothing_compare"
)

SUMMARY_CSV = (
    RUN_DIR
    / "amazon_ratings_oversmoothing_final_comparison.csv"
)


def load_metrics() -> pd.DataFrame:
    files = sorted(
        RUN_DIR.glob("*_oversmoothing.csv")
    )

    if len(files) != 2:
        raise RuntimeError(
            f"Expected two metrics files, found {len(files)}"
        )

    frames = []

    for path in files:
        frame = pd.read_csv(path)

        frame = frame[
            frame["layer_name"].notna()
        ].copy()

        frame["source_file"] = path.name
        frames.append(frame)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def select_stage(
    frame: pd.DataFrame,
    stage: str,
) -> pd.DataFrame:
    if stage == "hidden":
        mask = (
            (
                (frame["model"] == "GCN")
                & (frame["layer_name"] == "convs.0")
            )
            |
            (
                (frame["model"] == "GCNPairNorm")
                & (frame["layer_name"] == "pns.0")
            )
        )
    elif stage == "output":
        mask = (
            frame["layer_name"] == "convs.1"
        )
    else:
        raise ValueError(stage)

    selected = frame[mask].copy()

    expected_models = {
        "GCN",
        "GCNPairNorm",
    }

    found_models = set(
        selected["model"].unique()
    )

    if found_models != expected_models:
        raise RuntimeError(
            f"{stage}: expected {expected_models}, "
            f"found {found_models}"
        )

    return selected


def plot_metric(
    frame: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    plt.figure(figsize=(9, 6))

    for model, model_frame in frame.groupby(
        "model"
    ):
        model_frame = model_frame.sort_values(
            "epoch"
        )

        plt.plot(
            model_frame["epoch"],
            model_frame[metric],
            marker="o",
            label=model,
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    output = PLOT_DIR / filename

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("Saved:", output)


def load_accuracy() -> pd.DataFrame:
    rows = []

    for path in sorted(
        RUN_DIR.glob("*_summary.json")
    ):
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            summary = json.load(handle)

        rows.append(
            {
                "model": summary["model"],
                "best_test_acc":
                    summary[
                        "best_test_acc_at_best_val"
                    ],
                "best_val_acc":
                    summary["best_val_acc"],
                "best_epoch":
                    summary["best_epoch"],
            }
        )

    return pd.DataFrame(rows)


def main():
    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = load_metrics()

    hidden = select_stage(
        metrics,
        "hidden",
    )
    output = select_stage(
        metrics,
        "output",
    )

    plot_metric(
        hidden,
        "mean_pairwise_cosine_distance",
        "Mean pairwise cosine distance",
        "Hidden-space node separation",
        "hidden_pairwise_distance.png",
    )

    plot_metric(
        hidden,
        "mean_edge_cosine_distance",
        "Mean edge cosine distance",
        "Hidden-space neighbor separation",
        "hidden_edge_distance.png",
    )

    plot_metric(
        hidden,
        "effective_rank_ratio",
        "Effective-rank ratio",
        "Hidden-space dimensional diversity",
        "hidden_effective_rank_ratio.png",
    )

    plot_metric(
        hidden,
        "normalized_dirichlet_energy",
        "Normalized Dirichlet energy",
        "Hidden-space normalized Dirichlet energy",
        "hidden_normalized_dirichlet_energy.png",
    )

    plot_metric(
        output,
        "mean_pairwise_cosine_distance",
        "Mean pairwise cosine distance",
        "Output-space node separation",
        "output_pairwise_distance.png",
    )

    plot_metric(
        output,
        "effective_rank_ratio",
        "Effective-rank ratio",
        "Output-space dimensional diversity",
        "output_effective_rank_ratio.png",
    )

    final_epoch = metrics["epoch"].max()

    final_hidden = hidden[
        hidden["epoch"] == final_epoch
    ][
        [
            "model",
            "mean_pairwise_cosine_distance",
            "mean_edge_cosine_distance",
            "normalized_dirichlet_energy",
            "effective_rank",
            "effective_rank_ratio",
            "mean_embedding_norm",
        ]
    ].copy()

    final_output = output[
        output["epoch"] == final_epoch
    ][
        [
            "model",
            "mean_pairwise_cosine_distance",
            "mean_edge_cosine_distance",
            "normalized_dirichlet_energy",
            "effective_rank",
            "effective_rank_ratio",
            "mean_embedding_norm",
        ]
    ].copy()

    final_hidden = final_hidden.add_prefix(
        "hidden_"
    ).rename(
        columns={
            "hidden_model": "model",
        }
    )

    final_output = final_output.add_prefix(
        "output_"
    ).rename(
        columns={
            "output_model": "model",
        }
    )

    accuracy = load_accuracy()

    comparison = (
        accuracy
        .merge(
            final_hidden,
            on="model",
            how="left",
        )
        .merge(
            final_output,
            on="model",
            how="left",
        )
        .sort_values("model")
    )

    comparison.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    print("\nFinal comparison:")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nSaved:", SUMMARY_CSV)


if __name__ == "__main__":
    main()
