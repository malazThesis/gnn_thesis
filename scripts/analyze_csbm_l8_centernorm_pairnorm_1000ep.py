from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


RUN_DIR = Path(
    "runs/csbm_l8_centernorm_pairnorm_1000ep_fs050"
)

PLOT_DIR = Path(
    "plots/csbm_l8_centernorm_pairnorm_1000ep_fs050"
)

RUN_OUTPUT = (
    RUN_DIR
    / "csbm_l8_1000ep_run_level.csv"
)

HISTORY_OUTPUT = (
    RUN_DIR
    / "csbm_l8_1000ep_history_summary.csv"
)

METRIC_RUN_OUTPUT = (
    RUN_DIR
    / "csbm_l8_1000ep_metric_run_level.csv"
)

METRIC_SUMMARY_OUTPUT = (
    RUN_DIR
    / "csbm_l8_1000ep_metric_summary.csv"
)

SUMMARY_OUTPUT = (
    RUN_DIR
    / "csbm_l8_1000ep_model_summary.csv"
)

PAIRED_OUTPUT = (
    RUN_DIR
    / "csbm_l8_1000ep_paired_comparison.csv"
)

MODELS = [
    "GraphSAGECenterNorm",
    "GraphSAGEPairNorm",
]

SELECTED_EPOCHS = [
    1,
    4,
    8,
    32,
    100,
    200,
    400,
    600,
    800,
    1000,
]

METRIC_COLUMNS = [
    "mean_pairwise_cosine_distance",
    "mean_edge_cosine_distance",
    "normalized_dirichlet_energy",
    "effective_rank_ratio",
    "mean_embedding_norm",
    "class_separation_margin",
    "class_centroid_cosine_distance",
    "fisher_discriminant_ratio",
]


def graph_seed_from_summary(
    summary: dict,
    filename: str,
) -> int:
    dataset = str(
        summary.get("dataset", "")
    )

    match = re.search(
        r"-G(\d+)(?:-|$)",
        dataset,
    )

    if match:
        return int(match.group(1))

    data_path = str(
        summary.get(
            "data_path",
            summary.get("dataset_path", ""),
        )
    )

    match = re.search(
        r"seed(\d+)",
        data_path,
    )

    if match:
        return int(match.group(1))

    match = re.search(
        r"-G(\d+)_",
        filename,
    )

    if match:
        return int(match.group(1))

    raise RuntimeError(
        f"Cannot resolve graph seed: {filename}"
    )


def module_number(
    layer_name: object,
) -> int:
    match = re.search(
        r"\.(\d+)(?:#\d+)?$",
        str(layer_name),
    )

    return int(match.group(1)) if match else -1


def value_at_epoch(
    frame: pd.DataFrame,
    epoch: int,
    column: str,
) -> float:
    selected = frame[
        frame["epoch"] == epoch
    ]

    if selected.empty:
        return math.nan

    return float(
        selected.iloc[-1][column]
    )


def first_epoch_at_least(
    frame: pd.DataFrame,
    column: str,
    threshold: float,
) -> float:
    selected = frame[
        np.isfinite(frame[column])
        & (frame[column] >= threshold)
    ]

    if selected.empty:
        return math.nan

    return float(
        selected["epoch"].min()
    )


def first_post200_failure(
    frame: pd.DataFrame,
) -> float:
    selected = frame[
        (frame["epoch"] >= 200)
        & np.isfinite(frame["test_acc"])
        & (frame["test_acc"] < 0.90)
    ]

    if selected.empty:
        return math.nan

    return float(
        selected["epoch"].min()
    )


def finite_max(
    values: pd.Series,
) -> float:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).to_numpy(dtype=float)

    numeric = numeric[
        np.isfinite(numeric)
    ]

    if numeric.size == 0:
        return math.nan

    return float(numeric.max())


def finite_min(
    values: pd.Series,
) -> float:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).to_numpy(dtype=float)

    numeric = numeric[
        np.isfinite(numeric)
    ]

    if numeric.size == 0:
        return math.nan

    return float(numeric.min())


def final_hidden_row(
    metric_frame: pd.DataFrame,
    epoch: int,
) -> pd.Series:
    selected = metric_frame[
        metric_frame["epoch"] == epoch
    ].copy()

    selected = selected[
        selected["layer_name"]
        .astype(str)
        .str.startswith("pns.")
    ].copy()

    if selected.empty:
        raise RuntimeError(
            f"No normalization output at epoch {epoch}"
        )

    selected["module_number"] = (
        selected["layer_name"].map(
            module_number
        )
    )

    return selected.sort_values(
        [
            "module_number",
            "layer_index",
        ]
    ).iloc[-1]


def load_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    summary_files = sorted(
        RUN_DIR.glob("*_summary.json")
    )

    if len(summary_files) != 50:
        raise RuntimeError(
            f"Expected 50 summary files, "
            f"found {len(summary_files)}"
        )

    run_rows = []
    history_rows = []
    metric_rows = []

    for summary_path in summary_files:
        with summary_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            summary = json.load(handle)

        model = str(summary["model"])

        if model not in MODELS:
            raise RuntimeError(
                f"Unexpected model: {model}"
            )

        graph_seed = graph_seed_from_summary(
            summary,
            summary_path.name,
        )

        initialization_seed = int(
            summary["seed"]
        )

        history_path = summary_path.with_name(
            summary_path.name.replace(
                "_summary.json",
                "_history.csv",
            )
        )

        metric_path = summary_path.with_name(
            summary_path.name.replace(
                "_summary.json",
                "_oversmoothing.csv",
            )
        )

        history = pd.read_csv(
            history_path
        ).sort_values("epoch")

        metrics = pd.read_csv(
            metric_path
        )

        if len(history) != 1000:
            raise RuntimeError(
                f"{history_path}: expected 1000 "
                f"epochs, found {len(history)}"
            )

        history_numeric = history[
            [
                "loss",
                "train_acc",
                "val_acc",
                "test_acc",
                "gradient_norm",
            ]
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        history_failure = bool(
            not np.isfinite(
                history_numeric.to_numpy(
                    dtype=float
                )
            ).all()
        )

        final = history.iloc[-1]

        valid_validation = history[
            np.isfinite(history["val_acc"])
            & np.isfinite(history["test_acc"])
        ]

        best = valid_validation.loc[
            valid_validation[
                "val_acc"
            ].idxmax()
        ]

        post200 = history[
            history["epoch"] >= 200
        ]

        test_200 = value_at_epoch(
            history,
            200,
            "test_acc",
        )

        test_1000 = value_at_epoch(
            history,
            1000,
            "test_acc",
        )

        run_row = {
            "model":
                model,
            "graph_seed":
                graph_seed,
            "initialization_seed":
                initialization_seed,
            "learning_rate":
                0.5,
            "first_test_90":
                first_epoch_at_least(
                    history,
                    "test_acc",
                    0.90,
                ),
            "test_acc_epoch200":
                test_200,
            "test_acc_epoch400":
                value_at_epoch(
                    history,
                    400,
                    "test_acc",
                ),
            "test_acc_epoch600":
                value_at_epoch(
                    history,
                    600,
                    "test_acc",
                ),
            "test_acc_epoch800":
                value_at_epoch(
                    history,
                    800,
                    "test_acc",
                ),
            "test_acc_epoch1000":
                test_1000,
            "accuracy_change_1000_minus_200":
                test_1000 - test_200,
            "minimum_test_acc_after_200":
                finite_min(
                    post200["test_acc"]
                ),
            "first_epoch_below_90_after_200":
                first_post200_failure(
                    history
                ),
            "late_failure":
                bool(
                    np.isfinite(
                        first_post200_failure(
                            history
                        )
                    )
                ),
            "final_recovered":
                bool(
                    np.isfinite(test_1000)
                    and test_1000 >= 0.90
                ),
            "test_acc_at_best_val":
                float(best["test_acc"]),
            "best_val_epoch":
                int(best["epoch"]),
            "maximum_loss":
                finite_max(
                    history["loss"]
                ),
            "maximum_gradient_norm":
                finite_max(
                    history["gradient_norm"]
                ),
            "final_gradient_norm":
                float(
                    final["gradient_norm"]
                ),
            "history_numerical_failure":
                history_failure,
        }

        for record in history.to_dict(
            orient="records"
        ):
            history_rows.append(
                {
                    "model":
                        model,
                    "graph_seed":
                        graph_seed,
                    "initialization_seed":
                        initialization_seed,
                    **record,
                }
            )

        metric_failure = False
        metric_norm_values = []

        for epoch in SELECTED_EPOCHS:
            hidden = final_hidden_row(
                metrics,
                epoch,
            )

            metric_row = {
                "model":
                    model,
                "graph_seed":
                    graph_seed,
                "initialization_seed":
                    initialization_seed,
                "epoch":
                    epoch,
                "layer_name":
                    hidden["layer_name"],
            }

            for column in METRIC_COLUMNS:
                value = float(
                    hidden[column]
                )

                metric_row[column] = value

                if not np.isfinite(value):
                    metric_failure = True

            metric_norm_values.append(
                metric_row[
                    "mean_embedding_norm"
                ]
            )

            metric_rows.append(
                metric_row
            )

        norm_200 = next(
            row["mean_embedding_norm"]
            for row in metric_rows[::-1]
            if (
                row["model"] == model
                and row["graph_seed"]
                == graph_seed
                and row[
                    "initialization_seed"
                ]
                == initialization_seed
                and row["epoch"] == 200
            )
        )

        norm_1000 = next(
            row["mean_embedding_norm"]
            for row in metric_rows[::-1]
            if (
                row["model"] == model
                and row["graph_seed"]
                == graph_seed
                and row[
                    "initialization_seed"
                ]
                == initialization_seed
                and row["epoch"] == 1000
            )
        )

        run_row[
            "embedding_norm_epoch200"
        ] = norm_200

        run_row[
            "embedding_norm_epoch1000"
        ] = norm_1000

        run_row[
            "maximum_recorded_embedding_norm"
        ] = finite_max(
            pd.Series(metric_norm_values)
        )

        run_row[
            "log10_embedding_norm_epoch1000"
        ] = (
            math.log10(norm_1000)
            if (
                np.isfinite(norm_1000)
                and norm_1000 > 0
            )
            else math.nan
        )

        run_row[
            "embedding_norm_growth_orders"
        ] = (
            math.log10(norm_1000)
            - math.log10(norm_200)
            if (
                np.isfinite(norm_1000)
                and np.isfinite(norm_200)
                and norm_1000 > 0
                and norm_200 > 0
            )
            else math.nan
        )

        run_row[
            "metric_numerical_failure"
        ] = metric_failure

        run_rows.append(run_row)

    runs = pd.DataFrame(run_rows)
    histories = pd.DataFrame(history_rows)
    metric_runs = pd.DataFrame(metric_rows)

    keys = [
        "model",
        "graph_seed",
        "initialization_seed",
    ]

    if runs.duplicated(keys).any():
        raise RuntimeError(
            "Duplicate run-level rows."
        )

    counts = (
        runs.groupby("model")
        .size()
        .to_dict()
    )

    for model in MODELS:
        if counts.get(model, 0) != 25:
            raise RuntimeError(
                f"{model}: expected 25 runs, "
                f"found {counts.get(model, 0)}"
            )

    return runs, histories, metric_runs


def summarize_histories(
    histories: pd.DataFrame,
) -> pd.DataFrame:
    selected = histories[
        histories["epoch"].isin(
            SELECTED_EPOCHS
        )
    ].copy()

    return (
        selected.groupby(
            [
                "model",
                "epoch",
            ],
            as_index=False,
        )
        .agg(
            test_acc_mean=(
                "test_acc",
                "mean",
            ),
            test_acc_min=(
                "test_acc",
                "min",
            ),
            recovered_rate=(
                "test_acc",
                lambda values:
                    float(
                        np.mean(
                            np.asarray(values)
                            >= 0.90
                        )
                    ),
            ),
            loss_mean=(
                "loss",
                "mean",
            ),
            loss_max=(
                "loss",
                "max",
            ),
            gradient_norm_median=(
                "gradient_norm",
                "median",
            ),
            gradient_norm_max=(
                "gradient_norm",
                "max",
            ),
        )
    )


def summarize_metrics(
    metric_runs: pd.DataFrame,
) -> pd.DataFrame:
    return (
        metric_runs.groupby(
            [
                "model",
                "epoch",
            ],
            as_index=False,
        )
        .agg(
            pairwise_mean=(
                "mean_pairwise_cosine_distance",
                "mean",
            ),
            pairwise_min=(
                "mean_pairwise_cosine_distance",
                "min",
            ),
            class_margin_mean=(
                "class_separation_margin",
                "mean",
            ),
            class_margin_min=(
                "class_separation_margin",
                "min",
            ),
            rank_ratio_mean=(
                "effective_rank_ratio",
                "mean",
            ),
            embedding_norm_median=(
                "mean_embedding_norm",
                "median",
            ),
            embedding_norm_min=(
                "mean_embedding_norm",
                "min",
            ),
            embedding_norm_max=(
                "mean_embedding_norm",
                "max",
            ),
        )
    )


def summarize_models(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for model, group in runs.groupby(
        "model"
    ):
        rows.append(
            {
                "model":
                    model,
                "n":
                    len(group),
                "final_recovered_runs":
                    int(
                        group[
                            "final_recovered"
                        ].sum()
                    ),
                "late_failure_runs":
                    int(
                        group[
                            "late_failure"
                        ].sum()
                    ),
                "history_numerical_failure_runs":
                    int(
                        group[
                            "history_numerical_failure"
                        ].sum()
                    ),
                "metric_numerical_failure_runs":
                    int(
                        group[
                            "metric_numerical_failure"
                        ].sum()
                    ),
                "first_test_90_mean":
                    float(
                        group[
                            "first_test_90"
                        ].mean()
                    ),
                "test_acc_epoch200_mean":
                    float(
                        group[
                            "test_acc_epoch200"
                        ].mean()
                    ),
                "test_acc_epoch1000_mean":
                    float(
                        group[
                            "test_acc_epoch1000"
                        ].mean()
                    ),
                "test_acc_epoch1000_min":
                    float(
                        group[
                            "test_acc_epoch1000"
                        ].min()
                    ),
                "minimum_test_after_200_mean":
                    float(
                        group[
                            "minimum_test_acc_after_200"
                        ].mean()
                    ),
                "minimum_test_after_200_min":
                    float(
                        group[
                            "minimum_test_acc_after_200"
                        ].min()
                    ),
                "accuracy_change_1000_minus_200_mean":
                    float(
                        group[
                            "accuracy_change_1000_minus_200"
                        ].mean()
                    ),
                "maximum_gradient_norm_median":
                    float(
                        group[
                            "maximum_gradient_norm"
                        ].median()
                    ),
                "maximum_gradient_norm_max":
                    float(
                        group[
                            "maximum_gradient_norm"
                        ].max()
                    ),
                "embedding_norm_epoch200_median":
                    float(
                        group[
                            "embedding_norm_epoch200"
                        ].median()
                    ),
                "embedding_norm_epoch1000_median":
                    float(
                        group[
                            "embedding_norm_epoch1000"
                        ].median()
                    ),
                "embedding_norm_epoch1000_max":
                    float(
                        group[
                            "embedding_norm_epoch1000"
                        ].max()
                    ),
                "embedding_norm_growth_orders_median":
                    float(
                        group[
                            "embedding_norm_growth_orders"
                        ].median()
                    ),
            }
        )

    return pd.DataFrame(rows)


def paired_comparison(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "graph_seed",
        "initialization_seed",
    ]

    center = runs[
        runs["model"]
        == "GraphSAGECenterNorm"
    ].copy()

    pairnorm = runs[
        runs["model"]
        == "GraphSAGEPairNorm"
    ].copy()

    paired = center.merge(
        pairnorm,
        on=keys,
        suffixes=(
            "_center",
            "_pairnorm",
        ),
        validate="one_to_one",
    )

    comparison_columns = [
        "test_acc_epoch1000",
        "minimum_test_acc_after_200",
        "accuracy_change_1000_minus_200",
        "first_test_90",
        "maximum_gradient_norm",
        "log10_embedding_norm_epoch1000",
    ]

    output = {
        "n_pairs": len(paired),
        "pairnorm_only_final_recovered":
            int(
                (
                    paired[
                        "final_recovered_pairnorm"
                    ]
                    & ~paired[
                        "final_recovered_center"
                    ]
                ).sum()
            ),
        "centernorm_only_final_recovered":
            int(
                (
                    paired[
                        "final_recovered_center"
                    ]
                    & ~paired[
                        "final_recovered_pairnorm"
                    ]
                ).sum()
            ),
        "pairnorm_only_late_stable":
            int(
                (
                    ~paired[
                        "late_failure_pairnorm"
                    ]
                    & paired[
                        "late_failure_center"
                    ]
                ).sum()
            ),
        "centernorm_only_late_stable":
            int(
                (
                    ~paired[
                        "late_failure_center"
                    ]
                    & paired[
                        "late_failure_pairnorm"
                    ]
                ).sum()
            ),
    }

    for column in comparison_columns:
        difference = (
            paired[
                f"{column}_pairnorm"
            ]
            - paired[
                f"{column}_center"
            ]
        )

        finite = difference[
            np.isfinite(difference)
        ]

        output[
            f"{column}_mean_difference_"
            "pairnorm_minus_center"
        ] = (
            float(finite.mean())
            if len(finite)
            else math.nan
        )

        try:
            pvalue = float(
                stats.wilcoxon(
                    finite,
                    alternative="two-sided",
                ).pvalue
            )
        except ValueError:
            pvalue = math.nan

        output[
            f"{column}_wilcoxon_pvalue"
        ] = pvalue

    discordant = (
        output[
            "pairnorm_only_final_recovered"
        ]
        + output[
            "centernorm_only_final_recovered"
        ]
    )

    output[
        "final_recovery_mcnemar_pvalue"
    ] = (
        float(
            stats.binomtest(
                output[
                    "pairnorm_only_final_recovered"
                ],
                n=discordant,
                p=0.5,
            ).pvalue
        )
        if discordant
        else 1.0
    )

    return pd.DataFrame([output])


def make_plots(
    histories: pd.DataFrame,
    metric_summary: pd.DataFrame,
) -> None:
    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_summary = (
        histories.groupby(
            [
                "model",
                "epoch",
            ],
            as_index=False,
        )
        .agg(
            test_acc_mean=(
                "test_acc",
                "mean",
            ),
            test_acc_std=(
                "test_acc",
                "std",
            ),
        )
    )

    plt.figure(figsize=(9, 6))

    for model in MODELS:
        selected = history_summary[
            history_summary["model"] == model
        ]

        plt.plot(
            selected["epoch"],
            selected["test_acc_mean"],
            label=model,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Mean test accuracy")
    plt.ylim(0.85, 1.01)
    plt.title(
        "Long-term accuracy at learning rate 0.5"
    )
    plt.legend()
    plt.tight_layout()

    path = (
        PLOT_DIR
        / "test_accuracy_1000_epochs.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()
    print("Saved:", path)

    plt.figure(figsize=(9, 6))

    for model in MODELS:
        selected = metric_summary[
            metric_summary["model"] == model
        ]

        plt.plot(
            selected["epoch"],
            selected[
                "embedding_norm_median"
            ],
            marker="o",
            label=model,
        )

    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel(
        "Median final-hidden embedding norm"
    )
    plt.title(
        "Representation-scale stability"
    )
    plt.legend()
    plt.tight_layout()

    path = (
        PLOT_DIR
        / "embedding_norm_1000_epochs.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()
    print("Saved:", path)

    plt.figure(figsize=(9, 6))

    for model in MODELS:
        selected = metric_summary[
            metric_summary["model"] == model
        ]

        plt.plot(
            selected["epoch"],
            selected["class_margin_mean"],
            marker="o",
            label=model,
        )

    plt.xlabel("Epoch")
    plt.ylabel(
        "Mean class-separation margin"
    )
    plt.title(
        "Long-term class separation"
    )
    plt.legend()
    plt.tight_layout()

    path = (
        PLOT_DIR
        / "class_margin_1000_epochs.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()
    print("Saved:", path)


def main() -> None:
    runs, histories, metric_runs = load_data()

    history_summary = summarize_histories(
        histories
    )

    metric_summary = summarize_metrics(
        metric_runs
    )

    model_summary = summarize_models(
        runs
    )

    paired = paired_comparison(
        runs
    )

    runs.to_csv(
        RUN_OUTPUT,
        index=False,
    )

    history_summary.to_csv(
        HISTORY_OUTPUT,
        index=False,
    )

    metric_runs.to_csv(
        METRIC_RUN_OUTPUT,
        index=False,
    )

    metric_summary.to_csv(
        METRIC_SUMMARY_OUTPUT,
        index=False,
    )

    model_summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    paired.to_csv(
        PAIRED_OUTPUT,
        index=False,
    )

    make_plots(
        histories,
        metric_summary,
    )

    print("\n=== LANGZEITSTABILITÄT ===")

    print(
        model_summary.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.10f}",
        )
    )

    print(
        "\n=== ACCURACY UND GRADIENTEN "
        "ÜBER DIE ZEIT ==="
    )

    print(
        history_summary.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.10f}",
        )
    )

    print(
        "\n=== REPRÄSENTATIONSMETRIKEN ==="
    )

    print(
        metric_summary.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.10f}",
        )
    )

    print(
        "\n=== GEPAARTER VERGLEICH ==="
    )

    print(
        paired.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.10f}",
        )
    )

    print("\nSaved:", RUN_OUTPUT)
    print("Saved:", HISTORY_OUTPUT)
    print("Saved:", METRIC_RUN_OUTPUT)
    print("Saved:", METRIC_SUMMARY_OUTPUT)
    print("Saved:", SUMMARY_OUTPUT)
    print("Saved:", PAIRED_OUTPUT)


if __name__ == "__main__":
    main()
