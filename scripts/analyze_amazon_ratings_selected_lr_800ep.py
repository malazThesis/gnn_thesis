from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


RUN_DIR = Path(
    "runs/amazon_ratings_selected_lr_800ep"
)

PLOT_DIR = Path(
    "plots/amazon_ratings_selected_lr_800ep"
)

RUN_OUTPUT = (
    RUN_DIR
    / "amazon_ratings_800ep_run_level.csv"
)

MODEL_SUMMARY_OUTPUT = (
    RUN_DIR
    / "amazon_ratings_800ep_model_summary.csv"
)

EXTENSION_OUTPUT = (
    RUN_DIR
    / "amazon_ratings_400_to_800_extension.csv"
)

PAIRED_OUTPUT = (
    RUN_DIR
    / "amazon_ratings_800ep_pairnorm_effects.csv"
)

TRAJECTORY_OUTPUT = (
    RUN_DIR
    / "amazon_ratings_800ep_trajectory_summary.csv"
)

MODELS = [
    "GraphSAGE",
    "GraphSAGEPairNorm",
]

DEPTHS = [
    4,
    8,
]


def resolve_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    raise RuntimeError(
        f"Missing columns {candidates}; "
        f"available={frame.columns.tolist()}"
    )


def best_checkpoint(
    history: pd.DataFrame,
    *,
    maximum_epoch: int,
    epoch_column: str,
    val_column: str,
    test_column: str,
    train_column: str,
    loss_column: str,
) -> dict[str, float | int]:
    selected = history[
        history[epoch_column]
        <= maximum_epoch
    ].copy()

    selected = selected[
        np.isfinite(selected[val_column])
        & np.isfinite(selected[test_column])
    ]

    if selected.empty:
        raise RuntimeError(
            f"No valid checkpoint through "
            f"epoch {maximum_epoch}"
        )

    # First occurrence resolves exact validation ties.
    best = selected.loc[
        selected[val_column].idxmax()
    ]

    return {
        "epoch":
            int(best[epoch_column]),
        "val_acc":
            float(best[val_column]),
        "test_acc":
            float(best[test_column]),
        "train_acc":
            float(best[train_column]),
        "loss":
            float(best[loss_column]),
    }


def exact_sign_flip_pvalue(
    differences: np.ndarray,
) -> float:
    values = np.asarray(
        differences,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return math.nan

    observed = abs(
        float(values.mean())
    )

    extreme = 0
    total = 0

    for signs in itertools.product(
        [-1.0, 1.0],
        repeat=len(values),
    ):
        statistic = abs(
            float(
                np.mean(
                    values
                    * np.asarray(signs)
                )
            )
        )

        if statistic >= observed - 1e-15:
            extreme += 1

        total += 1

    return extreme / total


def paired_statistics(
    differences: pd.Series,
) -> dict[str, float | int]:
    values = pd.to_numeric(
        differences,
        errors="coerce",
    ).to_numpy(dtype=float)

    values = values[
        np.isfinite(values)
    ]

    n = len(values)

    if n < 2:
        raise RuntimeError(
            "Insufficient paired observations."
        )

    mean = float(
        values.mean()
    )

    standard_deviation = float(
        values.std(ddof=1)
    )

    standard_error = (
        standard_deviation
        / math.sqrt(n)
    )

    critical_value = float(
        stats.t.ppf(
            0.975,
            df=n - 1,
        )
    )

    try:
        wilcoxon_pvalue = float(
            stats.wilcoxon(
                values,
                alternative="two-sided",
            ).pvalue
        )
    except ValueError:
        wilcoxon_pvalue = math.nan

    return {
        "n_pairs":
            n,
        "mean_difference":
            mean,
        "mean_difference_percentage_points":
            100.0 * mean,
        "std_difference":
            standard_deviation,
        "median_difference":
            float(
                np.median(values)
            ),
        "minimum_difference":
            float(
                values.min()
            ),
        "maximum_difference":
            float(
                values.max()
            ),
        "positive_pairs":
            int(
                np.sum(values > 0)
            ),
        "negative_pairs":
            int(
                np.sum(values < 0)
            ),
        "zero_pairs":
            int(
                np.sum(values == 0)
            ),
        "ci95_low":
            mean
            - critical_value
            * standard_error,
        "ci95_high":
            mean
            + critical_value
            * standard_error,
        "paired_t_pvalue":
            float(
                stats.ttest_1samp(
                    values,
                    popmean=0.0,
                ).pvalue
            ),
        "wilcoxon_pvalue":
            wilcoxon_pvalue,
        "exact_sign_flip_pvalue":
            exact_sign_flip_pvalue(
                values
            ),
        "cohen_dz":
            (
                mean
                / standard_deviation
                if standard_deviation > 0
                else math.nan
            ),
    }


def holm_adjust(
    pvalues: list[float],
) -> list[float]:
    values = np.asarray(
        pvalues,
        dtype=float,
    )

    adjusted = np.full(
        len(values),
        np.nan,
    )

    valid = np.where(
        np.isfinite(values)
    )[0]

    order = valid[
        np.argsort(values[valid])
    ]

    running = 0.0
    total = len(order)

    for position, index in enumerate(order):
        candidate = (
            total - position
        ) * values[index]

        running = max(
            running,
            candidate,
        )

        adjusted[index] = min(
            running,
            1.0,
        )

    return adjusted.tolist()


def load_runs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    summary_files = sorted(
        RUN_DIR.glob("*_summary.json")
    )

    if len(summary_files) != 40:
        raise RuntimeError(
            f"Expected 40 summaries, "
            f"found {len(summary_files)}"
        )

    run_rows = []
    trajectory_rows = []

    for summary_path in summary_files:
        with summary_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            summary = json.load(handle)

        history_path = summary_path.with_name(
            summary_path.name.replace(
                "_summary.json",
                "_history.csv",
            )
        )

        history = pd.read_csv(
            history_path
        )

        epoch_column = resolve_column(
            history,
            ["epoch"],
        )

        loss_column = resolve_column(
            history,
            ["loss"],
        )

        train_column = resolve_column(
            history,
            [
                "train_acc",
                "train_accuracy",
                "train",
            ],
        )

        val_column = resolve_column(
            history,
            [
                "val_acc",
                "validation_acc",
                "val_accuracy",
                "val",
            ],
        )

        test_column = resolve_column(
            history,
            [
                "test_acc",
                "test_accuracy",
                "test",
            ],
        )

        for column in [
            epoch_column,
            loss_column,
            train_column,
            val_column,
            test_column,
        ]:
            history[column] = pd.to_numeric(
                history[column],
                errors="coerce",
            )

        if int(history[epoch_column].max()) != 800:
            raise RuntimeError(
                f"{history_path}: "
                f"maximum epoch is not 800"
            )

        best_400 = best_checkpoint(
            history,
            maximum_epoch=400,
            epoch_column=epoch_column,
            val_column=val_column,
            test_column=test_column,
            train_column=train_column,
            loss_column=loss_column,
        )

        best_800 = best_checkpoint(
            history,
            maximum_epoch=800,
            epoch_column=epoch_column,
            val_column=val_column,
            test_column=test_column,
            train_column=train_column,
            loss_column=loss_column,
        )

        final = history.sort_values(
            epoch_column
        ).iloc[-1]

        model = str(
            summary["model"]
        )

        depth = int(
            summary["num_layers"]
        )

        learning_rate = float(
            summary["learning_rate"]
        )

        seed = int(
            summary.get("seed", 1)
        )

        split_idx = int(
            summary.get("split_idx", 0)
        )

        run_rows.append(
            {
                "dataset":
                    str(
                        summary.get(
                            "dataset",
                            "Amazon-Ratings",
                        )
                    ),
                "model":
                    model,
                "num_layers":
                    depth,
                "hidden_channels":
                    int(
                        summary[
                            "hidden_channels"
                        ]
                    ),
                "learning_rate":
                    learning_rate,
                "seed":
                    seed,
                "split_idx":
                    split_idx,
                "best_epoch_400":
                    best_400["epoch"],
                "best_val_acc_400":
                    best_400["val_acc"],
                "test_acc_at_best_val_400":
                    best_400["test_acc"],
                "best_epoch_800":
                    best_800["epoch"],
                "best_val_acc_800":
                    best_800["val_acc"],
                "test_acc_at_best_val_800":
                    best_800["test_acc"],
                "validation_gain_800_minus_400":
                    (
                        best_800["val_acc"]
                        - best_400["val_acc"]
                    ),
                "test_gain_800_minus_400":
                    (
                        best_800["test_acc"]
                        - best_400["test_acc"]
                    ),
                "best_epoch_shift":
                    (
                        best_800["epoch"]
                        - best_400["epoch"]
                    ),
                "optimum_after_400":
                    bool(
                        best_800["epoch"] > 400
                    ),
                "optimum_at_800_cap":
                    bool(
                        best_800["epoch"] == 800
                    ),
                "optimum_last_10_percent":
                    bool(
                        best_800["epoch"] >= 720
                    ),
                "final_val_acc":
                    float(
                        final[val_column]
                    ),
                "final_test_acc":
                    float(
                        final[test_column]
                    ),
                "final_loss":
                    float(
                        final[loss_column]
                    ),
                "summary_path":
                    str(summary_path),
                "history_path":
                    str(history_path),
            }
        )

        for record in history.to_dict(
            orient="records"
        ):
            trajectory_rows.append(
                {
                    "model":
                        model,
                    "num_layers":
                        depth,
                    "learning_rate":
                        learning_rate,
                    "seed":
                        seed,
                    "split_idx":
                        split_idx,
                    "epoch":
                        int(
                            record[
                                epoch_column
                            ]
                        ),
                    "loss":
                        float(
                            record[
                                loss_column
                            ]
                        ),
                    "train_acc":
                        float(
                            record[
                                train_column
                            ]
                        ),
                    "val_acc":
                        float(
                            record[
                                val_column
                            ]
                        ),
                    "test_acc":
                        float(
                            record[
                                test_column
                            ]
                        ),
                }
            )

    runs = pd.DataFrame(run_rows)
    trajectories = pd.DataFrame(
        trajectory_rows
    )

    keys = [
        "model",
        "num_layers",
        "seed",
        "split_idx",
    ]

    if runs.duplicated(keys).any():
        raise RuntimeError(
            "Duplicate run-level rows."
        )

    counts = (
        runs.groupby(
            [
                "model",
                "num_layers",
            ]
        )
        .size()
    )

    if not (counts == 10).all():
        raise RuntimeError(
            "Expected ten runs per "
            "model/depth:\n"
            + counts.to_string()
        )

    return runs.sort_values(keys), trajectories


def summarize_models(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    return (
        runs.groupby(
            [
                "model",
                "num_layers",
                "learning_rate",
            ],
            as_index=False,
        )
        .agg(
            n=(
                "split_idx",
                "size",
            ),
            mean_best_val_acc_400=(
                "best_val_acc_400",
                "mean",
            ),
            mean_test_at_best_val_400=(
                "test_acc_at_best_val_400",
                "mean",
            ),
            mean_best_epoch_400=(
                "best_epoch_400",
                "mean",
            ),
            mean_best_val_acc_800=(
                "best_val_acc_800",
                "mean",
            ),
            std_best_val_acc_800=(
                "best_val_acc_800",
                "std",
            ),
            mean_test_at_best_val_800=(
                "test_acc_at_best_val_800",
                "mean",
            ),
            std_test_at_best_val_800=(
                "test_acc_at_best_val_800",
                "std",
            ),
            mean_best_epoch_800=(
                "best_epoch_800",
                "mean",
            ),
            optimum_after_400_count=(
                "optimum_after_400",
                "sum",
            ),
            optimum_at_800_cap_count=(
                "optimum_at_800_cap",
                "sum",
            ),
            optimum_last_10_percent_count=(
                "optimum_last_10_percent",
                "sum",
            ),
            mean_validation_gain_800_minus_400=(
                "validation_gain_800_minus_400",
                "mean",
            ),
            mean_test_gain_800_minus_400=(
                "test_gain_800_minus_400",
                "mean",
            ),
            mean_final_test_acc=(
                "final_test_acc",
                "mean",
            ),
        )
        .sort_values(
            [
                "num_layers",
                "model",
            ]
        )
    )


def extension_tests(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        model,
        depth,
    ), group in runs.groupby(
        [
            "model",
            "num_layers",
        ]
    ):
        statistics = paired_statistics(
            group[
                "test_gain_800_minus_400"
            ]
        )

        rows.append(
            {
                "model":
                    model,
                "num_layers":
                    depth,
                "learning_rate":
                    float(
                        group[
                            "learning_rate"
                        ].iloc[0]
                    ),
                "mean_test_400":
                    float(
                        group[
                            "test_acc_at_best_val_400"
                        ].mean()
                    ),
                "mean_test_800":
                    float(
                        group[
                            "test_acc_at_best_val_800"
                        ].mean()
                    ),
                "mean_val_400":
                    float(
                        group[
                            "best_val_acc_400"
                        ].mean()
                    ),
                "mean_val_800":
                    float(
                        group[
                            "best_val_acc_800"
                        ].mean()
                    ),
                "optimum_after_400_count":
                    int(
                        group[
                            "optimum_after_400"
                        ].sum()
                    ),
                "optimum_at_800_cap_count":
                    int(
                        group[
                            "optimum_at_800_cap"
                        ].sum()
                    ),
                **statistics,
            }
        )

    result = pd.DataFrame(rows)

    for column in [
        "paired_t_pvalue",
        "wilcoxon_pvalue",
        "exact_sign_flip_pvalue",
    ]:
        result[
            column.replace(
                "_pvalue",
                "_holm_pvalue",
            )
        ] = holm_adjust(
            result[column].tolist()
        )

    return result


def pairnorm_effects(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for depth in DEPTHS:
        baseline = runs[
            (
                runs["model"]
                == "GraphSAGE"
            )
            & (
                runs["num_layers"]
                == depth
            )
        ][
            [
                "seed",
                "split_idx",
                "learning_rate",
                "best_val_acc_800",
                "test_acc_at_best_val_800",
                "best_epoch_800",
            ]
        ].rename(
            columns={
                "learning_rate":
                    "baseline_learning_rate",
                "best_val_acc_800":
                    "baseline_val",
                "test_acc_at_best_val_800":
                    "baseline_test",
                "best_epoch_800":
                    "baseline_best_epoch",
            }
        )

        pairnorm = runs[
            (
                runs["model"]
                == "GraphSAGEPairNorm"
            )
            & (
                runs["num_layers"]
                == depth
            )
        ][
            [
                "seed",
                "split_idx",
                "learning_rate",
                "best_val_acc_800",
                "test_acc_at_best_val_800",
                "best_epoch_800",
            ]
        ].rename(
            columns={
                "learning_rate":
                    "pairnorm_learning_rate",
                "best_val_acc_800":
                    "pairnorm_val",
                "test_acc_at_best_val_800":
                    "pairnorm_test",
                "best_epoch_800":
                    "pairnorm_best_epoch",
            }
        )

        paired = baseline.merge(
            pairnorm,
            on=[
                "seed",
                "split_idx",
            ],
            validate="one_to_one",
        )

        paired["gain"] = (
            paired["pairnorm_test"]
            - paired["baseline_test"]
        )

        rows.append(
            {
                "num_layers":
                    depth,
                "baseline_learning_rate":
                    float(
                        paired[
                            "baseline_learning_rate"
                        ].iloc[0]
                    ),
                "pairnorm_learning_rate":
                    float(
                        paired[
                            "pairnorm_learning_rate"
                        ].iloc[0]
                    ),
                "baseline_mean_val":
                    float(
                        paired[
                            "baseline_val"
                        ].mean()
                    ),
                "pairnorm_mean_val":
                    float(
                        paired[
                            "pairnorm_val"
                        ].mean()
                    ),
                "baseline_mean_test":
                    float(
                        paired[
                            "baseline_test"
                        ].mean()
                    ),
                "pairnorm_mean_test":
                    float(
                        paired[
                            "pairnorm_test"
                        ].mean()
                    ),
                "baseline_mean_best_epoch":
                    float(
                        paired[
                            "baseline_best_epoch"
                        ].mean()
                    ),
                "pairnorm_mean_best_epoch":
                    float(
                        paired[
                            "pairnorm_best_epoch"
                        ].mean()
                    ),
                **paired_statistics(
                    paired["gain"]
                ),
            }
        )

    result = pd.DataFrame(rows)

    for column in [
        "paired_t_pvalue",
        "wilcoxon_pvalue",
        "exact_sign_flip_pvalue",
    ]:
        result[
            column.replace(
                "_pvalue",
                "_holm_pvalue",
            )
        ] = holm_adjust(
            result[column].tolist()
        )

    return result


def summarize_trajectories(
    trajectories: pd.DataFrame,
) -> pd.DataFrame:
    selected_epochs = (
        set(range(1, 801, 25))
        | {
            1,
            400,
            800,
        }
    )

    selected = trajectories[
        trajectories["epoch"].isin(
            selected_epochs
        )
    ]

    return (
        selected.groupby(
            [
                "model",
                "num_layers",
                "learning_rate",
                "epoch",
            ],
            as_index=False,
        )
        .agg(
            mean_train_acc=(
                "train_acc",
                "mean",
            ),
            mean_val_acc=(
                "val_acc",
                "mean",
            ),
            mean_test_acc=(
                "test_acc",
                "mean",
            ),
            mean_loss=(
                "loss",
                "mean",
            ),
        )
    )


def make_plots(
    trajectory_summary: pd.DataFrame,
) -> None:
    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for depth in DEPTHS:
        selected = trajectory_summary[
            trajectory_summary[
                "num_layers"
            ]
            == depth
        ]

        plt.figure(figsize=(9, 6))

        for model in MODELS:
            model_rows = selected[
                selected["model"] == model
            ].sort_values("epoch")

            plt.plot(
                model_rows["epoch"],
                model_rows["mean_val_acc"],
                label=model,
            )

        plt.axvline(
            400,
            linestyle="--",
        )

        plt.xlabel("Epoch")
        plt.ylabel(
            "Mean validation accuracy"
        )
        plt.title(
            f"Amazon-Ratings selected learning rates, L{depth}"
        )
        plt.legend()
        plt.tight_layout()

        path = (
            PLOT_DIR
            / f"validation_trajectory_L{depth}.png"
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
            model_rows = selected[
                selected["model"] == model
            ].sort_values("epoch")

            plt.plot(
                model_rows["epoch"],
                model_rows["mean_test_acc"],
                label=model,
            )

        plt.axvline(
            400,
            linestyle="--",
        )

        plt.xlabel("Epoch")
        plt.ylabel(
            "Mean test accuracy"
        )
        plt.title(
            f"Amazon-Ratings test trajectory, L{depth}"
        )
        plt.legend()
        plt.tight_layout()

        path = (
            PLOT_DIR
            / f"test_trajectory_L{depth}.png"
        )

        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()
        print("Saved:", path)


def main() -> None:
    runs, trajectories = load_runs()

    model_summary = summarize_models(
        runs
    )

    extension = extension_tests(
        runs
    )

    paired = pairnorm_effects(
        runs
    )

    trajectory_summary = (
        summarize_trajectories(
            trajectories
        )
    )

    runs.to_csv(
        RUN_OUTPUT,
        index=False,
    )

    model_summary.to_csv(
        MODEL_SUMMARY_OUTPUT,
        index=False,
    )

    extension.to_csv(
        EXTENSION_OUTPUT,
        index=False,
    )

    paired.to_csv(
        PAIRED_OUTPUT,
        index=False,
    )

    trajectory_summary.to_csv(
        TRAJECTORY_OUTPUT,
        index=False,
    )

    make_plots(
        trajectory_summary
    )

    print("\n=== MODELLZUSAMMENFASSUNG ===")

    print(
        model_summary.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.10f}",
        )
    )

    print(
        "\n=== EFFEKT DER VERLÄNGERUNG "
        "400 → 800 ==="
    )

    print(
        extension.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.10f}",
        )
    )

    print(
        "\n=== PAIRNORM-EFFEKT NACH "
        "800 EPOCHEN ==="
    )

    print(
        paired.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.10f}",
        )
    )

    print("\nSaved:", RUN_OUTPUT)
    print("Saved:", MODEL_SUMMARY_OUTPUT)
    print("Saved:", EXTENSION_OUTPUT)
    print("Saved:", PAIRED_OUTPUT)
    print("Saved:", TRAJECTORY_OUTPUT)


if __name__ == "__main__":
    main()
