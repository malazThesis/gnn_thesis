from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


BASE_DIR = Path(
    "runs/csbm_l8_graph_init_factorial_fs050"
)

LR_DIR = Path(
    "runs/csbm_l8_lr_sensitivity_fs050"
)

OUTPUT_DIR = LR_DIR

RUN_OUTPUT = (
    OUTPUT_DIR
    / "csbm_l8_lr_run_level.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "csbm_l8_lr_summary.csv"
)

PAIRWISE_OUTPUT = (
    OUTPUT_DIR
    / "csbm_l8_lr_pairwise_tests.csv"
)

METRIC_RUN_OUTPUT = (
    OUTPUT_DIR
    / "csbm_l8_lr_metric_run_level.csv"
)

METRIC_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "csbm_l8_lr_metric_summary.csv"
)

GLOBAL_TEST_OUTPUT = (
    OUTPUT_DIR
    / "csbm_l8_lr_global_tests.csv"
)

LEARNING_RATES = [
    0.001,
    0.003,
    0.01,
    0.03,
]

METRIC_EPOCHS = [
    0,
    1,
    4,
    8,
    32,
    200,
]


def first_epoch_at_least(
    history: pd.DataFrame,
    column: str,
    threshold: float,
) -> float:
    selected = history[
        history[column] >= threshold
    ]

    if selected.empty:
        return math.nan

    return float(selected["epoch"].min())


def classify_outcome(
    test_accuracy: float,
) -> str:
    if test_accuracy >= 0.90:
        return "recovered"

    if test_accuracy <= 0.52:
        return "collapsed"

    return "partial"


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan

    proportion = successes / total
    denominator = 1.0 + z**2 / total

    center = (
        proportion
        + z**2 / (2.0 * total)
    ) / denominator

    half_width = (
        z
        * math.sqrt(
            proportion
            * (1.0 - proportion)
            / total
            + z**2
            / (4.0 * total**2)
        )
        / denominator
    )

    return (
        max(0.0, center - half_width),
        min(1.0, center + half_width),
    )


def holm_adjust(
    pvalues: list[float],
) -> list[float]:
    values = np.asarray(
        pvalues,
        dtype=float,
    )

    order = np.argsort(values)
    adjusted = np.empty_like(values)

    running = 0.0
    number_tests = len(values)

    for position, index in enumerate(order):
        value = values[index]

        if math.isnan(value):
            adjusted[index] = math.nan
            continue

        candidate = (
            number_tests - position
        ) * value

        running = max(
            running,
            candidate,
        )

        adjusted[index] = min(
            running,
            1.0,
        )

    return adjusted.tolist()


def resolve_graph_seed(
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

    match = re.search(
        r"-G(\d+)_",
        filename,
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

    raise RuntimeError(
        f"Cannot resolve graph seed for {filename}"
    )


def resolve_learning_rate(
    summary: dict,
    source_directory: Path,
) -> float:
    for key in [
        "lr",
        "learning_rate",
    ]:
        if summary.get(key) is not None:
            return float(summary[key])

    dataset = str(
        summary.get("dataset", "")
    )

    match = re.search(
        r"-LR(\d+p\d+)",
        dataset,
    )

    if match:
        return float(
            match.group(1).replace(
                "p",
                ".",
            )
        )

    if source_directory == BASE_DIR:
        return 0.01

    raise RuntimeError(
        "Cannot determine learning rate."
    )


def module_number(name: object) -> int:
    match = re.search(
        r"\.(\d+)(?:#\d+)?$",
        str(name),
    )

    return int(match.group(1)) if match else -1


def select_final_hidden(
    frame: pd.DataFrame,
    hidden_channels: int,
) -> pd.Series:
    frame = frame[
        frame["layer_name"]
        .astype(str)
        .str.startswith("convs.")
        & (
            frame["embedding_dim"]
            == hidden_channels
        )
    ].copy()

    if frame.empty:
        raise RuntimeError(
            "No final hidden representation found."
        )

    frame["module_number"] = (
        frame["layer_name"].map(
            module_number
        )
    )

    return frame.sort_values(
        [
            "module_number",
            "layer_index",
        ]
    ).iloc[-1]


def load_runs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    run_rows = []
    metric_rows = []

    directory_specs = [
        (BASE_DIR, 25),
        (LR_DIR, 75),
    ]

    for directory, expected_count in directory_specs:
        summary_files = sorted(
            directory.glob("*_summary.json")
        )

        selected_files = []

        for summary_path in summary_files:
            with summary_path.open(
                "r",
                encoding="utf-8",
            ) as handle:
                summary = json.load(handle)

            if (
                summary["model"] != "GraphSAGE"
                or int(summary["num_layers"]) != 8
            ):
                continue

            selected_files.append(
                (
                    summary_path,
                    summary,
                )
            )

        if len(selected_files) != expected_count:
            raise RuntimeError(
                f"{directory}: expected "
                f"{expected_count} baseline runs, "
                f"found {len(selected_files)}"
            )

        for summary_path, summary in selected_files:
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

            graph_seed = resolve_graph_seed(
                summary,
                summary_path.name,
            )

            initialization_seed = int(
                summary["seed"]
            )

            learning_rate = (
                resolve_learning_rate(
                    summary,
                    directory,
                )
            )

            final = history.iloc[-1]
            best = history.loc[
                history["val_acc"].idxmax()
            ]

            final_test_accuracy = float(
                final["test_acc"]
            )

            run_rows.append(
                {
                    "graph_seed":
                        graph_seed,
                    "initialization_seed":
                        initialization_seed,
                    "learning_rate":
                        learning_rate,
                    "final_loss":
                        float(final["loss"]),
                    "final_train_acc":
                        float(
                            final["train_acc"]
                        ),
                    "final_val_acc":
                        float(
                            final["val_acc"]
                        ),
                    "final_test_acc":
                        final_test_accuracy,
                    "best_epoch":
                        int(best["epoch"]),
                    "best_val_acc":
                        float(best["val_acc"]),
                    "test_acc_at_best_val":
                        float(best["test_acc"]),
                    "first_train_90":
                        first_epoch_at_least(
                            history,
                            "train_acc",
                            0.90,
                        ),
                    "first_val_90":
                        first_epoch_at_least(
                            history,
                            "val_acc",
                            0.90,
                        ),
                    "first_test_90":
                        first_epoch_at_least(
                            history,
                            "test_acc",
                            0.90,
                        ),
                    "outcome":
                        classify_outcome(
                            final_test_accuracy
                        ),
                    "collapsed":
                        final_test_accuracy
                        <= 0.52,
                    "recovered":
                        final_test_accuracy
                        >= 0.90,
                }
            )

            hidden_channels = int(
                summary["hidden_channels"]
            )

            found_epochs = sorted(
                int(value)
                for value in metrics[
                    "epoch"
                ].unique()
            )

            if found_epochs != METRIC_EPOCHS:
                raise RuntimeError(
                    f"{metric_path.name}: "
                    f"expected {METRIC_EPOCHS}, "
                    f"found {found_epochs}"
                )

            for epoch in METRIC_EPOCHS:
                row = select_final_hidden(
                    metrics[
                        metrics["epoch"]
                        == epoch
                    ],
                    hidden_channels,
                )

                metric_rows.append(
                    {
                        "graph_seed":
                            graph_seed,
                        "initialization_seed":
                            initialization_seed,
                        "learning_rate":
                            learning_rate,
                        "epoch":
                            epoch,
                        "pairwise_distance":
                            float(
                                row[
                                    "mean_pairwise_cosine_distance"
                                ]
                            ),
                        "class_margin":
                            float(
                                row[
                                    "class_separation_margin"
                                ]
                            ),
                        "fisher_ratio":
                            float(
                                row[
                                    "fisher_discriminant_ratio"
                                ]
                            ),
                        "rank_ratio":
                            float(
                                row[
                                    "effective_rank_ratio"
                                ]
                            ),
                    }
                )

    runs = pd.DataFrame(run_rows)
    metric_runs = pd.DataFrame(
        metric_rows
    )

    keys = [
        "graph_seed",
        "initialization_seed",
        "learning_rate",
    ]

    if runs.duplicated(keys).any():
        raise RuntimeError(
            "Duplicate graph/init/lr runs."
        )

    if len(runs) != 100:
        raise RuntimeError(
            f"Expected 100 runs, "
            f"found {len(runs)}"
        )

    observed_rates = sorted(
        runs["learning_rate"].unique()
    )

    if not np.allclose(
        observed_rates,
        LEARNING_RATES,
    ):
        raise RuntimeError(
            f"Unexpected learning rates: "
            f"{observed_rates}"
        )

    return (
        runs.sort_values(keys),
        metric_runs.sort_values(
            keys + ["epoch"]
        ),
    )


def summarize_runs(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for learning_rate, group in runs.groupby(
        "learning_rate"
    ):
        recovered_count = int(
            group["recovered"].sum()
        )

        collapsed_count = int(
            group["collapsed"].sum()
        )

        lower, upper = wilson_interval(
            recovered_count,
            len(group),
        )

        rows.append(
            {
                "learning_rate":
                    learning_rate,
                "n":
                    len(group),
                "recovered_runs":
                    recovered_count,
                "collapsed_runs":
                    collapsed_count,
                "partial_runs":
                    len(group)
                    - recovered_count
                    - collapsed_count,
                "recovery_rate":
                    recovered_count
                    / len(group),
                "recovery_ci95_lower":
                    lower,
                "recovery_ci95_upper":
                    upper,
                "final_test_acc_mean":
                    float(
                        group[
                            "final_test_acc"
                        ].mean()
                    ),
                "final_test_acc_std":
                    float(
                        group[
                            "final_test_acc"
                        ].std(ddof=1)
                    ),
                "test_at_best_val_mean":
                    float(
                        group[
                            "test_acc_at_best_val"
                        ].mean()
                    ),
                "first_test_90_mean":
                    float(
                        group[
                            "first_test_90"
                        ].mean()
                    ),
                "first_test_90_median":
                    float(
                        group[
                            "first_test_90"
                        ].median()
                    ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "learning_rate"
    )


def summarize_metrics(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for keys, group in metrics.groupby(
        [
            "learning_rate",
            "epoch",
        ]
    ):
        row = {
            "learning_rate": keys[0],
            "epoch": keys[1],
            "n": len(group),
        }

        for column in [
            "pairwise_distance",
            "class_margin",
            "fisher_ratio",
            "rank_ratio",
        ]:
            values = pd.to_numeric(
                group[column],
                errors="coerce",
            ).dropna()

            mean = float(values.mean())

            if len(values) >= 2:
                std = float(
                    values.std(ddof=1)
                )

                critical = float(
                    stats.t.ppf(
                        0.975,
                        df=len(values) - 1,
                    )
                )

                ci95 = (
                    critical
                    * std
                    / math.sqrt(len(values))
                )
            else:
                std = 0.0
                ci95 = 0.0

            row[f"{column}_mean"] = mean
            row[f"{column}_std"] = std
            row[f"{column}_ci95"] = ci95

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        [
            "learning_rate",
            "epoch",
        ]
    )


def cochran_q(
    matrix: np.ndarray,
) -> tuple[float, float]:
    matrix = matrix.astype(float)

    number_conditions = matrix.shape[1]

    column_sums = matrix.sum(axis=0)
    row_sums = matrix.sum(axis=1)
    total = matrix.sum()

    numerator = (
        (number_conditions - 1)
        * (
            number_conditions
            * np.square(
                column_sums
            ).sum()
            - total**2
        )
    )

    denominator = (
        number_conditions * total
        - np.square(row_sums).sum()
    )

    if denominator <= 0:
        return math.nan, math.nan

    statistic = float(
        numerator / denominator
    )

    pvalue = float(
        stats.chi2.sf(
            statistic,
            df=number_conditions - 1,
        )
    )

    return statistic, pvalue


def pairwise_tests(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "graph_seed",
        "initialization_seed",
    ]

    reference = runs[
        np.isclose(
            runs["learning_rate"],
            0.01,
        )
    ][
        keys
        + [
            "final_test_acc",
            "recovered",
        ]
    ].rename(
        columns={
            "final_test_acc":
                "reference_accuracy",
            "recovered":
                "reference_recovered",
        }
    )

    rows = []

    for learning_rate in [
        0.001,
        0.003,
        0.03,
    ]:
        target = runs[
            np.isclose(
                runs["learning_rate"],
                learning_rate,
            )
        ][
            keys
            + [
                "final_test_acc",
                "recovered",
            ]
        ].rename(
            columns={
                "final_test_acc":
                    "target_accuracy",
                "recovered":
                    "target_recovered",
            }
        )

        paired = reference.merge(
            target,
            on=keys,
            validate="one_to_one",
        )

        differences = (
            paired["target_accuracy"]
            - paired[
                "reference_accuracy"
            ]
        )

        try:
            wilcoxon = stats.wilcoxon(
                differences,
                alternative="two-sided",
            )

            wilcoxon_pvalue = float(
                wilcoxon.pvalue
            )
        except ValueError:
            wilcoxon_pvalue = math.nan

        target_only = int(
            (
                paired["target_recovered"]
                & ~paired[
                    "reference_recovered"
                ]
            ).sum()
        )

        reference_only = int(
            (
                paired[
                    "reference_recovered"
                ]
                & ~paired[
                    "target_recovered"
                ]
            ).sum()
        )

        discordant = (
            target_only
            + reference_only
        )

        if discordant > 0:
            mcnemar_pvalue = float(
                stats.binomtest(
                    target_only,
                    n=discordant,
                    p=0.5,
                    alternative="two-sided",
                ).pvalue
            )
        else:
            mcnemar_pvalue = 1.0

        rows.append(
            {
                "reference_lr":
                    0.01,
                "target_lr":
                    learning_rate,
                "n":
                    len(paired),
                "mean_accuracy_difference":
                    float(
                        differences.mean()
                    ),
                "accuracy_difference_std":
                    float(
                        differences.std(
                            ddof=1
                        )
                    ),
                "positive_combinations":
                    int(
                        (differences > 0).sum()
                    ),
                "negative_combinations":
                    int(
                        (differences < 0).sum()
                    ),
                "ties":
                    int(
                        (differences == 0).sum()
                    ),
                "wilcoxon_pvalue":
                    wilcoxon_pvalue,
                "target_recovers_reference_not":
                    target_only,
                "reference_recovers_target_not":
                    reference_only,
                "mcnemar_exact_pvalue":
                    mcnemar_pvalue,
            }
        )

    result = pd.DataFrame(rows)

    result[
        "wilcoxon_holm_pvalue"
    ] = holm_adjust(
        result[
            "wilcoxon_pvalue"
        ].tolist()
    )

    result[
        "mcnemar_holm_pvalue"
    ] = holm_adjust(
        result[
            "mcnemar_exact_pvalue"
        ].tolist()
    )

    return result.sort_values(
        "target_lr"
    )


def global_tests(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "graph_seed",
        "initialization_seed",
    ]

    accuracy = runs.pivot(
        index=keys,
        columns="learning_rate",
        values="final_test_acc",
    ).sort_index(axis=1)

    recovery = runs.pivot(
        index=keys,
        columns="learning_rate",
        values="recovered",
    ).sort_index(axis=1)

    friedman = stats.friedmanchisquare(
        *[
            accuracy[column].to_numpy()
            for column in accuracy.columns
        ]
    )

    q_statistic, q_pvalue = cochran_q(
        recovery.to_numpy(
            dtype=float
        )
    )

    return pd.DataFrame(
        [
            {
                "test":
                    "Friedman final accuracy",
                "statistic":
                    float(
                        friedman.statistic
                    ),
                "degrees_of_freedom":
                    len(
                        accuracy.columns
                    ) - 1,
                "pvalue":
                    float(
                        friedman.pvalue
                    ),
            },
            {
                "test":
                    "Cochran Q recovery",
                "statistic":
                    q_statistic,
                "degrees_of_freedom":
                    len(
                        recovery.columns
                    ) - 1,
                "pvalue":
                    q_pvalue,
            },
        ]
    )


def print_recovery_matrix(
    runs: pd.DataFrame,
) -> None:
    matrix = runs.pivot(
        index=[
            "graph_seed",
            "initialization_seed",
        ],
        columns="learning_rate",
        values="recovered",
    )

    matrix = matrix.sort_index().sort_index(
        axis=1
    )

    print(
        "\n=== Recovery matrix ==="
    )

    print(matrix.to_string())


def main() -> None:
    runs, metric_runs = load_runs()

    summary = summarize_runs(runs)
    metric_summary = summarize_metrics(
        metric_runs
    )

    pairwise = pairwise_tests(runs)
    global_test_frame = global_tests(runs)

    runs.to_csv(
        RUN_OUTPUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    pairwise.to_csv(
        PAIRWISE_OUTPUT,
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

    global_test_frame.to_csv(
        GLOBAL_TEST_OUTPUT,
        index=False,
    )

    print(
        "\n=== Recovery by learning rate ==="
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.8f}",
        )
    )

    print_recovery_matrix(runs)

    print(
        "\n=== Pairwise comparisons "
        "against lr=0.01 ==="
    )

    print(
        pairwise.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.10f}",
        )
    )

    print(
        "\n=== Global repeated-measures tests ==="
    )

    print(
        global_test_frame.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.10f}",
        )
    )

    print(
        "\n=== Temporal representation metrics ==="
    )

    print(
        metric_summary[
            [
                "learning_rate",
                "epoch",
                "pairwise_distance_mean",
                "class_margin_mean",
                "fisher_ratio_mean",
                "rank_ratio_mean",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.10f}",
        )
    )

    print("\nSaved:", RUN_OUTPUT)
    print("Saved:", SUMMARY_OUTPUT)
    print("Saved:", PAIRWISE_OUTPUT)
    print("Saved:", METRIC_RUN_OUTPUT)
    print("Saved:", METRIC_SUMMARY_OUTPUT)
    print("Saved:", GLOBAL_TEST_OUTPUT)


if __name__ == "__main__":
    main()
