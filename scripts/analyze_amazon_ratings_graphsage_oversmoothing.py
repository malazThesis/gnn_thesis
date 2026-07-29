from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EXPERIMENTS = {
    "Shallow L4": Path(
        "runs/amazon_ratings_graphsage_os_phase1"
    ),
    "Deep L8": Path(
        "runs/amazon_ratings_graphsage_os_phase2"
    ),
}

OUTPUT_DIR = Path(
    "plots/amazon_ratings_graphsage_oversmoothing"
)

OUTPUT_CSV = Path(
    "runs/amazon_ratings_graphsage_oversmoothing_summary.csv"
)


def module_number(name: str) -> int:
    match = re.search(r"\.(\d+)(?:#\d+)?$", str(name))

    if match is None:
        return -1

    return int(match.group(1))


def load_experiment(
    label: str,
    directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_files = sorted(
        directory.glob("*_oversmoothing.csv")
    )
    summary_files = sorted(
        directory.glob("*_summary.json")
    )

    if len(metric_files) != 2:
        raise RuntimeError(
            f"{label}: expected two metric files, "
            f"found {len(metric_files)}"
        )

    if len(summary_files) != 2:
        raise RuntimeError(
            f"{label}: expected two summaries, "
            f"found {len(summary_files)}"
        )

    metric_frames = []

    for path in metric_files:
        frame = pd.read_csv(path)
        frame = frame[
            frame["layer_name"].notna()
        ].copy()
        frame["experiment"] = label
        metric_frames.append(frame)

    summary_rows = []

    for path in summary_files:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            row = json.load(handle)

        summary_rows.append(
            {
                "experiment": label,
                "model": row["model"],
                "num_layers": row["num_layers"],
                "hidden_channels": row[
                    "hidden_channels"
                ],
                "best_test_acc": row[
                    "best_test_acc_at_best_val"
                ],
                "best_val_acc": row["best_val_acc"],
                "best_epoch": row["best_epoch"],
            }
        )

    return (
        pd.concat(
            metric_frames,
            ignore_index=True,
        ),
        pd.DataFrame(summary_rows),
    )


def select_hidden_stage(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    selected_rows = []

    for (
        experiment,
        model,
        epoch,
    ), group in frame.groupby(
        ["experiment", "model", "epoch"]
    ):
        if model.endswith("PairNorm"):
            candidates = group[
                group["layer_name"]
                .astype(str)
                .str.startswith("pns.")
            ].copy()
        else:
            candidates = group[
                (
                    group["layer_name"]
                    .astype(str)
                    .str.startswith("convs.")
                )
                & (
                    group["embedding_dim"]
                    == group["hidden_channels"]
                )
            ].copy()

        if candidates.empty:
            raise RuntimeError(
                f"No hidden-stage row for "
                f"{experiment}, {model}, epoch {epoch}"
            )

        candidates["module_number"] = (
            candidates["layer_name"]
            .map(module_number)
        )

        selected_rows.append(
            candidates.sort_values(
                [
                    "module_number",
                    "layer_index",
                ]
            ).iloc[-1]
        )

    return pd.DataFrame(selected_rows)


def select_output_stage(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    selected_rows = []

    for (
        experiment,
        model,
        epoch,
    ), group in frame.groupby(
        ["experiment", "model", "epoch"]
    ):
        candidates = group[
            group["layer_name"]
            .astype(str)
            .str.startswith("convs.")
        ].copy()

        candidates["module_number"] = (
            candidates["layer_name"]
            .map(module_number)
        )

        selected_rows.append(
            candidates.sort_values(
                [
                    "module_number",
                    "layer_index",
                ]
            ).iloc[-1]
        )

    return pd.DataFrame(selected_rows)


def plot_metric(
    frame: pd.DataFrame,
    metric: str,
    ylabel: str,
    filename: str,
):
    for experiment, experiment_frame in frame.groupby(
        "experiment"
    ):
        plt.figure(figsize=(9, 6))

        for model, model_frame in experiment_frame.groupby(
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
        plt.title(f"{experiment}: {ylabel}")
        plt.legend()
        plt.tight_layout()

        safe_experiment = (
            experiment.lower()
            .replace(" ", "_")
        )

        output_path = (
            OUTPUT_DIR
            / f"{safe_experiment}_{filename}"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print("Saved:", output_path)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metric_frames = []
    summary_frames = []

    for label, directory in EXPERIMENTS.items():
        metrics, summaries = load_experiment(
            label,
            directory,
        )
        metric_frames.append(metrics)
        summary_frames.append(summaries)

    metrics = pd.concat(
        metric_frames,
        ignore_index=True,
    )

    summaries = pd.concat(
        summary_frames,
        ignore_index=True,
    )

    hidden = select_hidden_stage(metrics)
    output = select_output_stage(metrics)

    plot_metric(
        hidden,
        "mean_pairwise_cosine_distance",
        "Hidden pairwise cosine distance",
        "hidden_pairwise_distance.png",
    )

    plot_metric(
        hidden,
        "mean_edge_cosine_distance",
        "Hidden edge cosine distance",
        "hidden_edge_distance.png",
    )

    plot_metric(
        hidden,
        "effective_rank_ratio",
        "Hidden effective-rank ratio",
        "hidden_effective_rank_ratio.png",
    )

    plot_metric(
        hidden,
        "normalized_dirichlet_energy",
        "Hidden normalized Dirichlet energy",
        "hidden_normalized_dirichlet_energy.png",
    )

    plot_metric(
        output,
        "mean_pairwise_cosine_distance",
        "Output pairwise cosine distance",
        "output_pairwise_distance.png",
    )

    final_epoch = metrics["epoch"].max()

    final_hidden = hidden[
        hidden["epoch"] == final_epoch
    ][
        [
            "experiment",
            "model",
            "mean_pairwise_cosine_distance",
            "mean_edge_cosine_distance",
            "normalized_dirichlet_energy",
            "effective_rank",
            "effective_rank_ratio",
            "mean_embedding_norm",
        ]
    ].copy()

    final_hidden = final_hidden.rename(
        columns={
            column: f"hidden_{column}"
            for column in final_hidden.columns
            if column not in {
                "experiment",
                "model",
            }
        }
    )

    final_output = output[
        output["epoch"] == final_epoch
    ][
        [
            "experiment",
            "model",
            "mean_pairwise_cosine_distance",
            "mean_edge_cosine_distance",
            "normalized_dirichlet_energy",
            "effective_rank",
            "effective_rank_ratio",
            "mean_embedding_norm",
        ]
    ].copy()

    final_output = final_output.rename(
        columns={
            column: f"output_{column}"
            for column in final_output.columns
            if column not in {
                "experiment",
                "model",
            }
        }
    )

    result = (
        summaries
        .merge(
            final_hidden,
            on=["experiment", "model"],
            how="left",
        )
        .merge(
            final_output,
            on=["experiment", "model"],
            how="left",
        )
        .sort_values(
            ["experiment", "model"]
        )
    )

    result.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print("\nFinal comparison:")
    print(
        result.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nSaved:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
