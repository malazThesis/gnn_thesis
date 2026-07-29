from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


RUN_DIR = Path(
    "runs/csbm_classmetrics_temporal_fs050"
)

PLOT_DIR = Path(
    "plots/csbm_l8_recovery"
)

OUTPUT_RUNS = (
    RUN_DIR
    / "csbm_l8_baseline_recovery_by_seed.csv"
)

OUTPUT_EARLY = (
    RUN_DIR
    / "csbm_l8_baseline_early_predictors.csv"
)

OUTPUT_GROUPS = (
    RUN_DIR
    / "csbm_l8_baseline_recovery_groups.csv"
)


def first_epoch_at_least(
    frame: pd.DataFrame,
    column: str,
    threshold: float,
) -> float:
    selected = frame[
        frame[column] >= threshold
    ]

    if selected.empty:
        return math.nan

    return float(
        selected["epoch"].min()
    )


def classify_outcome(
    test_accuracy: float,
) -> str:
    if test_accuracy >= 0.90:
        return "recovered"

    if test_accuracy <= 0.52:
        return "collapsed"

    return "partial"


def main() -> None:
    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metric_path = (
        RUN_DIR
        / "csbm_temporal_metric_run_level.csv"
    )

    history_path = (
        RUN_DIR
        / "csbm_temporal_history_run_level.csv"
    )

    metrics = pd.read_csv(metric_path)
    history = pd.read_csv(history_path)

    metrics = metrics[
        (metrics["model"] == "GraphSAGE")
        & (metrics["num_layers"] == 8)
    ].copy()

    history = history[
        (history["model"] == "GraphSAGE")
        & (history["num_layers"] == 8)
    ].copy()

    if history["graph_seed"].nunique() != 5:
        raise RuntimeError(
            "Expected five GraphSAGE L8 seeds."
        )

    run_rows = []

    for seed, frame in history.groupby(
        "graph_seed"
    ):
        frame = frame.sort_values("epoch")
        final = frame.iloc[-1]

        final_test = float(
            final["test_acc"]
        )

        run_rows.append(
            {
                "graph_seed": int(seed),
                "final_loss": float(
                    final["loss"]
                ),
                "final_train_acc": float(
                    final["train_acc"]
                ),
                "final_val_acc": float(
                    final["val_acc"]
                ),
                "final_test_acc": final_test,
                "outcome": classify_outcome(
                    final_test
                ),
                "first_train_60": (
                    first_epoch_at_least(
                        frame,
                        "train_acc",
                        0.60,
                    )
                ),
                "first_train_90": (
                    first_epoch_at_least(
                        frame,
                        "train_acc",
                        0.90,
                    )
                ),
                "first_val_90": (
                    first_epoch_at_least(
                        frame,
                        "val_acc",
                        0.90,
                    )
                ),
                "first_test_90": (
                    first_epoch_at_least(
                        frame,
                        "test_acc",
                        0.90,
                    )
                ),
            }
        )

    runs = pd.DataFrame(run_rows)

    final_hidden = metrics[
        metrics["representation"]
        == "final_hidden"
    ].copy()

    logits = metrics[
        metrics["representation"]
        == "logits"
    ].copy()

    early_epochs = [0, 1, 4, 8, 32, 200]

    metric_columns = [
        "mean_pairwise_cosine_distance",
        "class_separation_margin",
        "fisher_discriminant_ratio",
        "effective_rank_ratio",
    ]

    wide_parts = []

    for representation, frame in [
        ("hidden", final_hidden),
        ("logits", logits),
    ]:
        selected = frame[
            frame["epoch"].isin(
                early_epochs
            )
        ][
            [
                "graph_seed",
                "epoch",
                *metric_columns,
            ]
        ].copy()

        wide = selected.pivot(
            index="graph_seed",
            columns="epoch",
            values=metric_columns,
        )

        wide.columns = [
            (
                f"{representation}_"
                f"{metric}_epoch{epoch}"
            )
            for metric, epoch in wide.columns
        ]

        wide_parts.append(
            wide.reset_index()
        )

    early = runs.copy()

    for part in wide_parts:
        early = early.merge(
            part,
            on="graph_seed",
            validate="one_to_one",
        )

    early.to_csv(
        OUTPUT_EARLY,
        index=False,
    )

    runs.to_csv(
        OUTPUT_RUNS,
        index=False,
    )

    group_rows = []

    merged_metrics = final_hidden.merge(
        runs[
            [
                "graph_seed",
                "outcome",
                "final_test_acc",
            ]
        ],
        on="graph_seed",
        validate="many_to_one",
    )

    for keys, group in merged_metrics.groupby(
        [
            "outcome",
            "epoch",
        ]
    ):
        group_rows.append(
            {
                "outcome": keys[0],
                "epoch": keys[1],
                "n": group[
                    "graph_seed"
                ].nunique(),
                "pairwise_mean": group[
                    "mean_pairwise_cosine_distance"
                ].mean(),
                "class_margin_mean": group[
                    "class_separation_margin"
                ].mean(),
                "fisher_mean": group[
                    "fisher_discriminant_ratio"
                ].mean(),
                "rank_ratio_mean": group[
                    "effective_rank_ratio"
                ].mean(),
            }
        )

    groups = pd.DataFrame(
        group_rows
    ).sort_values(
        [
            "outcome",
            "epoch",
        ]
    )

    groups.to_csv(
        OUTPUT_GROUPS,
        index=False,
    )

    print(
        "\n=== L8 baseline outcome by seed ==="
    )

    print(
        runs.sort_values(
            "graph_seed"
        ).to_string(
            index=False,
            float_format=lambda value:
                f"{value:.10f}",
        )
    )

    display_columns = [
        "graph_seed",
        "outcome",
        "final_test_acc",
    ]

    for epoch in [0, 1, 4, 8, 32]:
        display_columns.extend(
            [
                (
                    "hidden_"
                    "mean_pairwise_cosine_distance_"
                    f"epoch{epoch}"
                ),
                (
                    "hidden_"
                    "class_separation_margin_"
                    f"epoch{epoch}"
                ),
            ]
        )

    print(
        "\n=== Early representation state ==="
    )

    print(
        early[
            display_columns
        ].sort_values(
            "graph_seed"
        ).to_string(
            index=False,
            float_format=lambda value:
                f"{value:.10f}",
        )
    )

    print(
        "\n=== Recovered versus collapsed ==="
    )

    print(
        groups.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.10f}",
        )
    )

    correlations = []

    candidate_columns = [
        column
        for column in early.columns
        if (
            column.startswith(
                "hidden_"
            )
            or column.startswith(
                "logits_"
            )
        )
        and not column.endswith(
            "epoch200"
        )
    ]

    for column in candidate_columns:
        valid = early[
            [
                column,
                "final_test_acc",
            ]
        ].dropna()

        if (
            len(valid) >= 3
            and valid[column].nunique() > 1
        ):
            result = stats.spearmanr(
                valid[column],
                valid["final_test_acc"],
            )

            correlations.append(
                {
                    "metric": column,
                    "spearman_rho": float(
                        result.statistic
                    ),
                    "pvalue": float(
                        result.pvalue
                    ),
                }
            )

    correlation_frame = pd.DataFrame(
        correlations
    ).sort_values(
        "spearman_rho",
        ascending=False,
    )

    print(
        "\n=== Descriptive early correlations ==="
    )

    print(
        correlation_frame.head(20).to_string(
            index=False,
            float_format=lambda value:
                f"{value:.10f}",
        )
    )

    for column, ylabel, filename, log_scale in [
        (
            "test_acc",
            "Test accuracy",
            "l8_test_accuracy_by_seed.png",
            False,
        ),
        (
            "mean_pairwise_cosine_distance",
            "Final-hidden pairwise distance",
            "l8_pairwise_by_seed.png",
            True,
        ),
        (
            "class_separation_margin",
            "Final-hidden class margin",
            "l8_class_margin_by_seed.png",
            True,
        ),
    ]:
        plt.figure(figsize=(10, 6))

        if column == "test_acc":
            source = history
        else:
            source = final_hidden

        for seed, frame in source.groupby(
            "graph_seed"
        ):
            frame = frame.sort_values("epoch")

            plt.plot(
                frame["epoch"],
                frame[column],
                marker=(
                    "o"
                    if column != "test_acc"
                    else None
                ),
                label=f"Seed {seed}",
            )

        if log_scale:
            plt.yscale(
                "symlog",
                linthresh=1e-9,
            )

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(
            f"GraphSAGE L8: {ylabel} by seed"
        )
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

    print("\nSaved:", OUTPUT_RUNS)
    print("Saved:", OUTPUT_EARLY)
    print("Saved:", OUTPUT_GROUPS)


if __name__ == "__main__":
    main()
