from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


RUN_ROOT = Path(
    "runs/amazon_ratings_fair_lr"
)

PLOT_DIR = Path(
    "plots/amazon_ratings_fair_lr"
)

RUN_OUTPUT = (
    RUN_ROOT
    / "amazon_ratings_fair_lr_run_level.csv"
)

LR_SUMMARY_OUTPUT = (
    RUN_ROOT
    / "amazon_ratings_fair_lr_summary.csv"
)

SELECTED_OUTPUT = (
    RUN_ROOT
    / "amazon_ratings_fair_lr_selected_configs.csv"
)

PAIRED_OUTPUT = (
    RUN_ROOT
    / "amazon_ratings_fair_lr_selected_paired_tests.csv"
)

MATCHED_LR_OUTPUT = (
    RUN_ROOT
    / "amazon_ratings_fair_lr_matched_lr_effects.csv"
)

MODEL_ORDER = [
    "GraphSAGE",
    "GraphSAGEPairNorm",
]

DEPTH_ORDER = [
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
        f"None of these columns found: {candidates}. "
        f"Available: {frame.columns.tolist()}"
    )


def load_run_level() -> pd.DataFrame:
    rows = []

    summary_files = sorted(
        RUN_ROOT.glob(
            "lr_*/*_summary.json"
        )
    )

    if len(summary_files) != 240:
        raise RuntimeError(
            f"Expected 240 summaries, "
            f"found {len(summary_files)}"
        )

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

        if not history_path.exists():
            raise FileNotFoundError(
                history_path
            )

        history = pd.read_csv(
            history_path
        )

        epoch_column = resolve_column(
            history,
            ["epoch"],
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

        loss_column = resolve_column(
            history,
            ["loss"],
        )

        for column in [
            epoch_column,
            train_column,
            val_column,
            test_column,
            loss_column,
        ]:
            history[column] = pd.to_numeric(
                history[column],
                errors="coerce",
            )

        valid = history[
            np.isfinite(
                history[val_column]
            )
            & np.isfinite(
                history[test_column]
            )
        ]

        if valid.empty:
            raise RuntimeError(
                f"No finite validation rows: "
                f"{history_path}"
            )

        best_index = valid[
            val_column
        ].idxmax()

        best = valid.loc[best_index]
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

        split_idx = int(
            summary.get(
                "split_idx",
                0,
            )
        )

        seed = int(
            summary.get(
                "seed",
                1,
            )
        )

        maximum_epoch = int(
            history[epoch_column].max()
        )

        best_epoch = int(
            best[epoch_column]
        )

        rows.append(
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
                "best_epoch":
                    best_epoch,
                "best_val_acc":
                    float(
                        best[val_column]
                    ),
                "test_acc_at_best_val":
                    float(
                        best[test_column]
                    ),
                "train_acc_at_best_val":
                    float(
                        best[train_column]
                    ),
                "loss_at_best_val":
                    float(
                        best[loss_column]
                    ),
                "final_test_acc":
                    float(
                        final[test_column]
                    ),
                "final_val_acc":
                    float(
                        final[val_column]
                    ),
                "final_loss":
                    float(
                        final[loss_column]
                    ),
                "maximum_epoch":
                    maximum_epoch,
                "best_epoch_at_cap":
                    bool(
                        best_epoch
                        == maximum_epoch
                    ),
                "best_epoch_last_10_percent":
                    bool(
                        best_epoch
                        >= 0.9 * maximum_epoch
                    ),
                "history_path":
                    str(history_path),
                "summary_path":
                    str(summary_path),
            }
        )

    runs = pd.DataFrame(rows)

    keys = [
        "model",
        "num_layers",
        "learning_rate",
        "seed",
        "split_idx",
    ]

    if runs.duplicated(keys).any():
        duplicates = runs[
            runs.duplicated(
                keys,
                keep=False,
            )
        ]

        raise RuntimeError(
            "Duplicate runs:\n"
            + duplicates.to_string(
                index=False
            )
        )

    expected = (
        runs.groupby(
            [
                "model",
                "num_layers",
                "learning_rate",
            ]
        )
        .size()
    )

    if not (expected == 10).all():
        raise RuntimeError(
            "Expected ten splits per "
            "model/depth/lr:\n"
            + expected.to_string()
        )

    return runs.sort_values(keys)


def summarize_learning_rates(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        runs.groupby(
            [
                "model",
                "num_layers",
                "hidden_channels",
                "learning_rate",
            ],
            as_index=False,
        )
        .agg(
            n=(
                "split_idx",
                "size",
            ),
            mean_best_val_acc=(
                "best_val_acc",
                "mean",
            ),
            std_best_val_acc=(
                "best_val_acc",
                "std",
            ),
            mean_test_at_best_val=(
                "test_acc_at_best_val",
                "mean",
            ),
            std_test_at_best_val=(
                "test_acc_at_best_val",
                "std",
            ),
            mean_final_test_acc=(
                "final_test_acc",
                "mean",
            ),
            std_final_test_acc=(
                "final_test_acc",
                "std",
            ),
            mean_best_epoch=(
                "best_epoch",
                "mean",
            ),
            std_best_epoch=(
                "best_epoch",
                "std",
            ),
            best_epoch_at_cap_count=(
                "best_epoch_at_cap",
                "sum",
            ),
            best_epoch_last_10_percent_count=(
                "best_epoch_last_10_percent",
                "sum",
            ),
        )
    )

    return summary.sort_values(
        [
            "num_layers",
            "model",
            "learning_rate",
        ]
    )


def select_by_validation(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        model,
        depth,
    ), group in summary.groupby(
        [
            "model",
            "num_layers",
        ]
    ):
        # Primär: höchste mittlere Validation-Accuracy.
        # Exakter Gleichstand: kleinere Lernrate.
        selected = group.sort_values(
            [
                "mean_best_val_acc",
                "learning_rate",
            ],
            ascending=[
                False,
                True,
            ],
        ).iloc[0]

        row = selected.to_dict()

        row[
            "selection_rule"
        ] = (
            "maximum mean validation accuracy; "
            "smaller learning rate breaks exact ties"
        )

        rows.append(row)

    selected = pd.DataFrame(rows)

    selected["model_order"] = (
        selected["model"].map(
            {
                model: index
                for index, model
                in enumerate(MODEL_ORDER)
            }
        )
    )

    return (
        selected.sort_values(
            [
                "num_layers",
                "model_order",
            ]
        )
        .drop(
            columns="model_order"
        )
    )


def exact_sign_flip_pvalue(
    differences: np.ndarray,
) -> float:
    differences = np.asarray(
        differences,
        dtype=float,
    )

    differences = differences[
        np.isfinite(differences)
    ]

    if len(differences) == 0:
        return math.nan

    observed = abs(
        differences.mean()
    )

    extreme = 0
    total = 0

    for signs in itertools.product(
        [-1.0, 1.0],
        repeat=len(differences),
    ):
        statistic = abs(
            np.mean(
                differences
                * np.asarray(signs)
            )
        )

        if statistic >= observed - 1e-15:
            extreme += 1

        total += 1

    return extreme / total


def paired_statistics(
    differences: pd.Series,
) -> dict[str, float]:
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
            "Not enough paired observations."
        )

    mean = float(
        values.mean()
    )

    std = float(
        values.std(ddof=1)
    )

    standard_error = (
        std / math.sqrt(n)
    )

    critical = float(
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
        "mean_gain":
            mean,
        "mean_gain_percentage_points":
            100.0 * mean,
        "std_gain":
            std,
        "median_gain":
            float(
                np.median(values)
            ),
        "minimum_gain":
            float(
                values.min()
            ),
        "maximum_gain":
            float(
                values.max()
            ),
        "positive_splits":
            int(
                np.sum(values > 0)
            ),
        "negative_splits":
            int(
                np.sum(values < 0)
            ),
        "zero_splits":
            int(
                np.sum(values == 0)
            ),
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
        "mean_gain_ci95_low":
            mean
            - critical * standard_error,
        "mean_gain_ci95_high":
            mean
            + critical * standard_error,
        "cohen_dz":
            (
                mean / std
                if std > 0
                else math.nan
            ),
    }


def holm_adjust(
    values: list[float],
) -> list[float]:
    pvalues = np.asarray(
        values,
        dtype=float,
    )

    adjusted = np.full(
        len(pvalues),
        np.nan,
    )

    valid_indices = np.where(
        np.isfinite(pvalues)
    )[0]

    ordered = valid_indices[
        np.argsort(
            pvalues[valid_indices]
        )
    ]

    previous = 0.0
    total = len(ordered)

    for position, index in enumerate(
        ordered
    ):
        candidate = (
            total - position
        ) * pvalues[index]

        previous = max(
            previous,
            candidate,
        )

        adjusted[index] = min(
            previous,
            1.0,
        )

    return adjusted.tolist()


def selected_paired_tests(
    runs: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for depth in DEPTH_ORDER:
        baseline_lr = float(
            selected[
                (
                    selected["model"]
                    == "GraphSAGE"
                )
                & (
                    selected["num_layers"]
                    == depth
                )
            ][
                "learning_rate"
            ].iloc[0]
        )

        pairnorm_lr = float(
            selected[
                (
                    selected["model"]
                    == "GraphSAGEPairNorm"
                )
                & (
                    selected["num_layers"]
                    == depth
                )
            ][
                "learning_rate"
            ].iloc[0]
        )

        baseline = runs[
            (
                runs["model"]
                == "GraphSAGE"
            )
            & (
                runs["num_layers"]
                == depth
            )
            & np.isclose(
                runs["learning_rate"],
                baseline_lr,
            )
        ][
            [
                "seed",
                "split_idx",
                "test_acc_at_best_val",
                "best_val_acc",
                "best_epoch",
            ]
        ].rename(
            columns={
                "test_acc_at_best_val":
                    "baseline_test",
                "best_val_acc":
                    "baseline_val",
                "best_epoch":
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
            & np.isclose(
                runs["learning_rate"],
                pairnorm_lr,
            )
        ][
            [
                "seed",
                "split_idx",
                "test_acc_at_best_val",
                "best_val_acc",
                "best_epoch",
            ]
        ].rename(
            columns={
                "test_acc_at_best_val":
                    "pairnorm_test",
                "best_val_acc":
                    "pairnorm_val",
                "best_epoch":
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

        result = {
            "num_layers":
                depth,
            "baseline_selected_lr":
                baseline_lr,
            "pairnorm_selected_lr":
                pairnorm_lr,
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

        rows.append(result)

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


def matched_learning_rate_effects(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        depth,
        learning_rate,
    ), group in runs.groupby(
        [
            "num_layers",
            "learning_rate",
        ]
    ):
        baseline = group[
            group["model"]
            == "GraphSAGE"
        ][
            [
                "seed",
                "split_idx",
                "test_acc_at_best_val",
            ]
        ].rename(
            columns={
                "test_acc_at_best_val":
                    "baseline_test",
            }
        )

        pairnorm = group[
            group["model"]
            == "GraphSAGEPairNorm"
        ][
            [
                "seed",
                "split_idx",
                "test_acc_at_best_val",
            ]
        ].rename(
            columns={
                "test_acc_at_best_val":
                    "pairnorm_test",
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
                "learning_rate":
                    learning_rate,
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
                **paired_statistics(
                    paired["gain"]
                ),
            }
        )

    result = pd.DataFrame(rows)

    result[
        "exact_sign_flip_holm_pvalue"
    ] = holm_adjust(
        result[
            "exact_sign_flip_pvalue"
        ].tolist()
    )

    return result.sort_values(
        [
            "num_layers",
            "learning_rate",
        ]
    )


def make_plots(
    summary: pd.DataFrame,
) -> None:
    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for depth in DEPTH_ORDER:
        selected = summary[
            summary["num_layers"]
            == depth
        ]

        plt.figure(
            figsize=(9, 6)
        )

        for model in MODEL_ORDER:
            model_rows = selected[
                selected["model"]
                == model
            ].sort_values(
                "learning_rate"
            )

            plt.plot(
                model_rows[
                    "learning_rate"
                ],
                model_rows[
                    "mean_best_val_acc"
                ],
                marker="o",
                label=model,
            )

        plt.xscale("log")
        plt.xlabel("Learning rate")
        plt.ylabel(
            "Mean best validation accuracy"
        )
        plt.title(
            f"Amazon-Ratings validation tuning, L{depth}"
        )
        plt.legend()
        plt.tight_layout()

        path = (
            PLOT_DIR
            / f"validation_lr_curve_L{depth}.png"
        )

        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()
        print("Saved:", path)

        plt.figure(
            figsize=(9, 6)
        )

        for model in MODEL_ORDER:
            model_rows = selected[
                selected["model"]
                == model
            ].sort_values(
                "learning_rate"
            )

            plt.plot(
                model_rows[
                    "learning_rate"
                ],
                model_rows[
                    "mean_test_at_best_val"
                ],
                marker="o",
                label=model,
            )

        plt.xscale("log")
        plt.xlabel("Learning rate")
        plt.ylabel(
            "Mean test accuracy at best validation epoch"
        )
        plt.title(
            f"Amazon-Ratings test performance, L{depth}"
        )
        plt.legend()
        plt.tight_layout()

        path = (
            PLOT_DIR
            / f"test_lr_curve_L{depth}.png"
        )

        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()
        print("Saved:", path)


def main() -> None:
    runs = load_run_level()

    summary = summarize_learning_rates(
        runs
    )

    selected = select_by_validation(
        summary
    )

    paired = selected_paired_tests(
        runs,
        selected,
    )

    matched = matched_learning_rate_effects(
        runs
    )

    runs.to_csv(
        RUN_OUTPUT,
        index=False,
    )

    summary.to_csv(
        LR_SUMMARY_OUTPUT,
        index=False,
    )

    selected.to_csv(
        SELECTED_OUTPUT,
        index=False,
    )

    paired.to_csv(
        PAIRED_OUTPUT,
        index=False,
    )

    matched.to_csv(
        MATCHED_LR_OUTPUT,
        index=False,
    )

    make_plots(summary)

    print("\n=== ALLE LERNRATEN ===")

    print(
        summary.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.10f}",
        )
    )

    print(
        "\n=== NACH VALIDATION AUSGEWÄHLT ==="
    )

    print(
        selected[
            [
                "model",
                "num_layers",
                "learning_rate",
                "mean_best_val_acc",
                "mean_test_at_best_val",
                "std_test_at_best_val",
                "mean_best_epoch",
                "best_epoch_at_cap_count",
                "best_epoch_last_10_percent_count",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.10f}",
        )
    )

    print(
        "\n=== GEPAARTE TESTVERGLEICHE ==="
    )

    print(
        paired.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.10f}",
        )
    )

    print(
        "\n=== EFFEKT BEI GLEICHER LERNRATE ==="
    )

    print(
        matched[
            [
                "num_layers",
                "learning_rate",
                "baseline_mean_test",
                "pairnorm_mean_test",
                "mean_gain_percentage_points",
                "positive_splits",
                "exact_sign_flip_pvalue",
                "exact_sign_flip_holm_pvalue",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.10f}",
        )
    )

    print("\nSaved:", RUN_OUTPUT)
    print("Saved:", LR_SUMMARY_OUTPUT)
    print("Saved:", SELECTED_OUTPUT)
    print("Saved:", PAIRED_OUTPUT)
    print("Saved:", MATCHED_LR_OUTPUT)


if __name__ == "__main__":
    main()
