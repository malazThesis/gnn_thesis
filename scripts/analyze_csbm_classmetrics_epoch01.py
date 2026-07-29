from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd
from scipy import stats


RUN_DIR = Path(
    "runs/csbm_classmetrics_epoch01"
)

RUN_OUTPUT = (
    RUN_DIR
    / "csbm_classmetrics_epoch01_run_level.csv"
)

SUMMARY_OUTPUT = (
    RUN_DIR
    / "csbm_classmetrics_epoch01_summary.csv"
)


METRICS = [
    "mean_pairwise_cosine_distance",
    "within_class_cosine_distance",
    "between_class_cosine_distance",
    "class_separation_margin",
    "class_centroid_cosine_distance",
    "fisher_discriminant_ratio",
    "effective_rank_ratio",
]


def module_number(name: object) -> int:
    match = re.search(
        r"\.(\d+)(?:#\d+)?$",
        str(name),
    )

    return int(match.group(1)) if match else -1


def mean_ci95(
    values: pd.Series,
) -> tuple[int, float, float, float]:
    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    n = len(values)

    if n == 0:
        return (
            0,
            math.nan,
            math.nan,
            math.nan,
        )

    mean = float(values.mean())

    if n == 1:
        return n, mean, 0.0, 0.0

    std = float(values.std(ddof=1))

    critical = float(
        stats.t.ppf(
            0.975,
            df=n - 1,
        )
    )

    ci95 = (
        critical
        * std
        / math.sqrt(n)
    )

    return n, mean, std, ci95


def select_representations(
    frame: pd.DataFrame,
    *,
    model: str,
    hidden_channels: int,
) -> dict[str, pd.Series]:
    frame = frame[
        frame["layer_name"].notna()
    ].copy()

    frame["module_number"] = (
        frame["layer_name"].map(
            module_number
        )
    )

    convolution_rows = frame[
        frame["layer_name"]
        .astype(str)
        .str.startswith("convs.")
    ].copy()

    if convolution_rows.empty:
        raise RuntimeError(
            f"No convolution rows found for {model}"
        )

    logits = convolution_rows.sort_values(
        [
            "module_number",
            "layer_index",
        ]
    ).iloc[-1]

    if model.endswith("PairNorm"):
        hidden_rows = frame[
            frame["layer_name"]
            .astype(str)
            .str.startswith("pns.")
        ].copy()
    else:
        hidden_rows = convolution_rows[
            convolution_rows["embedding_dim"]
            == hidden_channels
        ].copy()

    if hidden_rows.empty:
        raise RuntimeError(
            f"No final hidden row found for {model}"
        )

    final_hidden = hidden_rows.sort_values(
        [
            "module_number",
            "layer_index",
        ]
    ).iloc[-1]

    return {
        "final_hidden": final_hidden,
        "logits": logits,
    }


def load_run_level() -> pd.DataFrame:
    summary_files = sorted(
        RUN_DIR.glob("*_summary.json")
    )

    if len(summary_files) != 40:
        raise RuntimeError(
            f"Expected 40 summaries, "
            f"found {len(summary_files)}"
        )

    rows = []

    for summary_path in summary_files:
        with summary_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            summary = json.load(handle)

        metric_path = summary_path.with_name(
            summary_path.name.replace(
                "_summary.json",
                "_oversmoothing.csv",
            )
        )

        metrics = pd.read_csv(metric_path)

        required_columns = set(
            METRICS
            + [
                "epoch",
                "layer_name",
                "layer_index",
                "embedding_dim",
                "num_labeled_nodes_used",
                "num_classes_used",
                "class_metric_scope",
            ]
        )

        missing = sorted(
            required_columns
            - set(metrics.columns)
        )

        if missing:
            raise RuntimeError(
                f"{metric_path.name}: "
                f"missing columns {missing}"
            )

        epochs = sorted(
            int(value)
            for value in metrics[
                "epoch"
            ].unique()
        )

        if epochs != [0, 1]:
            raise RuntimeError(
                f"{metric_path.name}: "
                f"expected epochs [0, 1], "
                f"found {epochs}"
            )

        model = str(summary["model"])
        hidden_channels = int(
            summary["hidden_channels"]
        )

        for epoch in (0, 1):
            epoch_frame = metrics[
                metrics["epoch"] == epoch
            ].copy()

            selected = select_representations(
                epoch_frame,
                model=model,
                hidden_channels=hidden_channels,
            )

            for representation, row in (
                selected.items()
            ):
                result = {
                    "feature_signal": float(
                        summary["feature_signal"]
                    ),
                    "model": model,
                    "uses_pairnorm":
                        model.endswith("PairNorm"),
                    "num_layers": int(
                        summary["num_layers"]
                    ),
                    "graph_seed": int(
                        summary["graph_seed"]
                    ),
                    "split_idx": int(
                        summary["split_idx"]
                    ),
                    "epoch": epoch,
                    "representation":
                        representation,
                    "layer_name":
                        row["layer_name"],
                    "embedding_dim": int(
                        row["embedding_dim"]
                    ),
                    "num_labeled_nodes_used":
                        int(
                            row[
                                "num_labeled_nodes_used"
                            ]
                        ),
                    "num_classes_used":
                        int(
                            row[
                                "num_classes_used"
                            ]
                        ),
                    "class_metric_scope":
                        row["class_metric_scope"],
                }

                for metric in METRICS:
                    result[metric] = float(
                        row[metric]
                    )

                within = result[
                    "within_class_cosine_distance"
                ]
                between = result[
                    "between_class_cosine_distance"
                ]

                result[
                    "class_distance_ratio"
                ] = (
                    between / within
                    if within > 0.0
                    else math.nan
                )

                denominator = (
                    between
                    + within
                )

                result[
                    "normalized_class_margin"
                ] = (
                    (
                        between
                        - within
                    )
                    / denominator
                    if denominator > 0.0
                    else math.nan
                )

                global_distance = result[
                    "mean_pairwise_cosine_distance"
                ]

                result[
                    "margin_to_global_distance"
                ] = (
                    result[
                        "class_separation_margin"
                    ]
                    / global_distance
                    if global_distance > 0.0
                    else math.nan
                )

                rows.append(result)

    result = pd.DataFrame(rows)

    expected_rows = 40 * 2 * 2

    if len(result) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows, "
            f"found {len(result)}"
        )

    return result.sort_values(
        [
            "feature_signal",
            "model",
            "num_layers",
            "graph_seed",
            "epoch",
            "representation",
        ]
    ).reset_index(drop=True)


def build_summary(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    metrics = METRICS + [
        "class_distance_ratio",
        "normalized_class_margin",
        "margin_to_global_distance",
    ]

    rows = []

    group_columns = [
        "feature_signal",
        "model",
        "num_layers",
        "epoch",
        "representation",
    ]

    for keys, group in runs.groupby(
        group_columns
    ):
        row = {
            "feature_signal": keys[0],
            "model": keys[1],
            "num_layers": keys[2],
            "epoch": keys[3],
            "representation": keys[4],
            "n": len(group),
        }

        for metric in metrics:
            n, mean, std, ci95 = mean_ci95(
                group[metric]
            )

            row[f"{metric}_n"] = n
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = ci95

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        group_columns
    ).reset_index(drop=True)


def main() -> None:
    runs = load_run_level()
    summary = build_summary(runs)

    runs.to_csv(
        RUN_OUTPUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    display_columns = [
        "feature_signal",
        "model",
        "num_layers",
        "epoch",
        "representation",
        "mean_pairwise_cosine_distance_mean",
        "within_class_cosine_distance_mean",
        "between_class_cosine_distance_mean",
        "class_separation_margin_mean",
        "normalized_class_margin_mean",
        "fisher_discriminant_ratio_mean",
        "margin_to_global_distance_mean",
    ]

    print(
        summary[display_columns].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.8f}",
        )
    )

    print("\nSaved:", RUN_OUTPUT)
    print("Saved:", SUMMARY_OUTPUT)


if __name__ == "__main__":
    main()
