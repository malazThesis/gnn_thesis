from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


EXPERIMENTS = {
    "Shallow L4": Path(
        "runs/amazon_ratings_graphsage_os_phase1"
    ),
    "Deep L8": Path(
        "runs/amazon_ratings_graphsage_os_phase2"
    ),
}

OUTPUT = Path(
    "runs/amazon_ratings_graphsage_best_epoch_metrics.csv"
)


def module_number(name: str) -> int:
    match = re.search(
        r"\.(\d+)(?:#\d+)?$",
        str(name),
    )
    return int(match.group(1)) if match else -1


def select_hidden(
    frame: pd.DataFrame,
    model: str,
) -> pd.Series:
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
                == frame["hidden_channels"]
            )
        ].copy()

    if candidates.empty:
        raise RuntimeError(
            f"No hidden representation found for {model}"
        )

    candidates["module_number"] = (
        candidates["layer_name"].map(module_number)
    )

    return candidates.sort_values(
        ["module_number", "layer_index"]
    ).iloc[-1]


def select_output(
    frame: pd.DataFrame,
) -> pd.Series:
    candidates = frame[
        frame["layer_name"]
        .astype(str)
        .str.startswith("convs.")
    ].copy()

    candidates["module_number"] = (
        candidates["layer_name"].map(module_number)
    )

    return candidates.sort_values(
        ["module_number", "layer_index"]
    ).iloc[-1]


def extract_metrics(
    *,
    experiment: str,
    summary_path: Path,
    metric_path: Path,
) -> list[dict]:
    with summary_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        summary = json.load(handle)

    metrics = pd.read_csv(metric_path)
    metrics = metrics[
        metrics["layer_name"].notna()
    ].copy()

    recorded_epochs = sorted(
        metrics["epoch"].unique()
    )

    best_epoch = int(summary["best_epoch"])

    nearest_best_epoch = min(
        recorded_epochs,
        key=lambda value: abs(value - best_epoch),
    )

    checkpoints = {
        "nearest_best_epoch": nearest_best_epoch,
        "final_epoch": max(recorded_epochs),
    }

    rows = []

    for checkpoint, recorded_epoch in checkpoints.items():
        epoch_frame = metrics[
            metrics["epoch"] == recorded_epoch
        ].copy()

        hidden = select_hidden(
            epoch_frame,
            summary["model"],
        )
        output = select_output(epoch_frame)

        rows.append(
            {
                "experiment": experiment,
                "model": summary["model"],
                "num_layers": summary["num_layers"],
                "hidden_channels":
                    summary["hidden_channels"],
                "checkpoint": checkpoint,
                "target_best_epoch": best_epoch,
                "recorded_epoch": int(recorded_epoch),
                "best_test_acc":
                    summary[
                        "best_test_acc_at_best_val"
                    ],
                "best_val_acc":
                    summary["best_val_acc"],
                "hidden_pairwise_distance":
                    hidden[
                        "mean_pairwise_cosine_distance"
                    ],
                "hidden_edge_distance":
                    hidden[
                        "mean_edge_cosine_distance"
                    ],
                "hidden_normalized_dirichlet":
                    hidden[
                        "normalized_dirichlet_energy"
                    ],
                "hidden_effective_rank_ratio":
                    hidden[
                        "effective_rank_ratio"
                    ],
                "output_pairwise_distance":
                    output[
                        "mean_pairwise_cosine_distance"
                    ],
                "output_edge_distance":
                    output[
                        "mean_edge_cosine_distance"
                    ],
                "output_effective_rank_ratio":
                    output[
                        "effective_rank_ratio"
                    ],
            }
        )

    return rows


def main():
    rows = []

    for experiment, directory in EXPERIMENTS.items():
        summaries = sorted(
            directory.glob("*_summary.json")
        )

        if len(summaries) != 2:
            raise RuntimeError(
                f"{experiment}: expected two summaries, "
                f"found {len(summaries)}"
            )

        for summary_path in summaries:
            metric_path = summary_path.with_name(
                summary_path.name.replace(
                    "_summary.json",
                    "_oversmoothing.csv",
                )
            )

            if not metric_path.exists():
                raise FileNotFoundError(metric_path)

            rows.extend(
                extract_metrics(
                    experiment=experiment,
                    summary_path=summary_path,
                    metric_path=metric_path,
                )
            )

    result = pd.DataFrame(rows).sort_values(
        ["experiment", "checkpoint", "model"]
    )

    result.to_csv(OUTPUT, index=False)

    print(
        result.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
