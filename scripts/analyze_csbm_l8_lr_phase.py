from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


OLD_RUN_PATH = Path(
    "runs/csbm_l8_lr_sensitivity_fs050/"
    "csbm_l8_lr_run_level.csv"
)

OLD_METRIC_PATH = Path(
    "runs/csbm_l8_lr_sensitivity_fs050/"
    "csbm_l8_lr_metric_run_level.csv"
)

NEW_DIR = Path(
    "runs/csbm_l8_lr_transition_fs050"
)

PLOT_DIR = Path(
    "plots/csbm_l8_lr_phase"
)

RUN_OUTPUT = (
    NEW_DIR
    / "csbm_l8_lr_phase_run_level.csv"
)

SUMMARY_OUTPUT = (
    NEW_DIR
    / "csbm_l8_lr_phase_summary.csv"
)

METRIC_OUTPUT = (
    NEW_DIR
    / "csbm_l8_lr_phase_metric_summary.csv"
)

ADJACENT_OUTPUT = (
    NEW_DIR
    / "csbm_l8_lr_phase_adjacent_tests.csv"
)

THRESHOLD_OUTPUT = (
    NEW_DIR
    / "csbm_l8_lr_phase_thresholds.csv"
)

RETENTION_OUTPUT = (
    NEW_DIR
    / "csbm_l8_lr_phase_retention_predictors.csv"
)

GLOBAL_OUTPUT = (
    NEW_DIR
    / "csbm_l8_lr_phase_global_tests.csv"
)

LEARNING_RATES = [
    0.001,
    0.003,
    0.005,
    0.007,
    0.009,
    0.010,
    0.030,
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
    frame: pd.DataFrame,
    column: str,
    threshold: float,
) -> float:
    selected = frame[
        frame[column] >= threshold
    ]

    if selected.empty:
        return math.nan

    return float(selected["epoch"].min())


def classify_outcome(
    accuracy: float,
) -> str:
    if accuracy >= 0.90:
        return "recovered"

    if accuracy <= 0.52:
        return "collapsed"

    return "partial"


def wilson_interval(
    successes: int,
    total: int,
) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan

    z = 1.959963984540054
    proportion = successes / total

    denominator = 1.0 + z**2 / total

    center = (
        proportion
        + z**2 / (2.0 * total)
    ) / denominator

    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z**2 / (4.0 * total**2)
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

    valid_indices = np.where(
        ~np.isnan(values)
    )[0]

    adjusted = np.full_like(
        values,
        np.nan,
    )

    if len(valid_indices) == 0:
        return adjusted.tolist()

    order = valid_indices[
        np.argsort(
            values[valid_indices]
        )
    ]

    running = 0.0
    number_tests = len(order)

    for position, index in enumerate(order):
        candidate = (
            number_tests - position
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
        f"Cannot resolve graph seed: {filename}"
    )


def resolve_learning_rate(
    summary: dict,
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
        r"TRL(\d+p\d+)",
        dataset,
    )

    if match:
        return float(
            match.group(1).replace(
                "p",
                ".",
            )
        )

    raise RuntimeError(
        f"Cannot resolve learning rate: {dataset}"
    )


def module_number(
    name: object,
) -> int:
    match = re.search(
        r"\.(\d+)(?:#\d+)?$",
        str(name),
    )

    return int(match.group(1)) if match else -1


def select_final_hidden(
    frame: pd.DataFrame,
    hidden_channels: int,
) -> pd.Series:
    selected = frame[
        frame["layer_name"]
        .astype(str)
        .str.startswith("convs.")
        & (
            frame["embedding_dim"]
            == hidden_channels
        )
    ].copy()

    if selected.empty:
        raise RuntimeError(
            "No final hidden representation found."
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


def load_new_runs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    summary_files = sorted(
        NEW_DIR.glob("*_summary.json")
    )

    if len(summary_files) != 75:
        raise RuntimeError(
            f"Expected 75 new runs, "
            f"found {len(summary_files)}"
        )

    run_rows = []
    metric_rows = []

    for summary_path in summary_files:
        with summary_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            summary = json.load(handle)

        if summary["model"] != "GraphSAGE":
            raise RuntimeError(
                f"Unexpected model: "
                f"{summary_path.name}"
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

        graph_seed = resolve_graph_seed(
            summary,
            summary_path.name,
        )

        initialization_seed = int(
            summary["seed"]
        )

        learning_rate = resolve_learning_rate(
            summary
        )

        final = history.iloc[-1]

        best = history.loc[
            history["val_acc"].idxmax()
        ]

        final_test = float(
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
                    float(final["train_acc"]),
                "final_val_acc":
                    float(final["val_acc"]),
                "final_test_acc":
                    final_test,
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
                        final_test
                    ),
                "collapsed":
                    final_test <= 0.52,
                "recovered":
                    final_test >= 0.90,
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
                f"found epochs {found_epochs}"
            )

        for epoch in METRIC_EPOCHS:
            row = select_final_hidden(
                metrics[
                    metrics["epoch"] == epoch
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

    return (
        pd.DataFrame(run_rows),
        pd.DataFrame(metric_rows),
    )


def load_all_runs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    old_runs = pd.read_csv(
        OLD_RUN_PATH
    )

    old_metrics = pd.read_csv(
        OLD_METRIC_PATH
    )

    new_runs, new_metrics = load_new_runs()

    runs = pd.concat(
        [
            old_runs,
            new_runs,
        ],
        ignore_index=True,
    )

    metrics = pd.concat(
        [
            old_metrics,
            new_metrics,
        ],
        ignore_index=True,
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

    if len(runs) != 175:
        raise RuntimeError(
            f"Expected 175 runs, "
            f"found {len(runs)}"
        )

    for learning_rate, group in runs.groupby(
        "learning_rate"
    ):
        if len(group) != 25:
            raise RuntimeError(
                f"lr={learning_rate}: "
                f"expected 25, found {len(group)}"
            )

    observed = sorted(
        runs["learning_rate"].unique()
    )

    if not np.allclose(
        observed,
        LEARNING_RATES,
    ):
        raise RuntimeError(
            f"Unexpected rates: {observed}"
        )

    return (
        runs.sort_values(keys),
        metrics.sort_values(
            keys + ["epoch"]
        ),
    )


def add_epoch1_retention(
    runs: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "graph_seed",
        "initialization_seed",
        "learning_rate",
    ]

    pairwise = metrics.pivot(
        index=keys,
        columns="epoch",
        values="pairwise_distance",
    )

    pairwise = pairwise.rename(
        columns={
            0: "pairwise_epoch0",
            1: "pairwise_epoch1",
            4: "pairwise_epoch4",
            8: "pairwise_epoch8",
            32: "pairwise_epoch32",
            200: "pairwise_epoch200",
        }
    ).reset_index()

    pairwise[
        "epoch1_pairwise_retention"
    ] = (
        pairwise["pairwise_epoch1"]
        / pairwise["pairwise_epoch0"]
    )

    return runs.merge(
        pairwise,
        on=keys,
        validate="one_to_one",
    )


def summarize_runs(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for learning_rate, group in runs.groupby(
        "learning_rate"
    ):
        recovered = int(
            group["recovered"].sum()
        )

        collapsed = int(
            group["collapsed"].sum()
        )

        lower, upper = wilson_interval(
            recovered,
            len(group),
        )

        rows.append(
            {
                "learning_rate":
                    learning_rate,
                "n":
                    len(group),
                "recovered_runs":
                    recovered,
                "collapsed_runs":
                    collapsed,
                "partial_runs":
                    len(group)
                    - recovered
                    - collapsed,
                "recovery_rate":
                    recovered / len(group),
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
                "epoch1_pairwise_mean":
                    float(
                        group[
                            "pairwise_epoch1"
                        ].mean()
                    ),
                "epoch1_retention_mean":
                    float(
                        group[
                            "epoch1_pairwise_retention"
                        ].mean()
                    ),
                "epoch1_retention_median":
                    float(
                        group[
                            "epoch1_pairwise_retention"
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


def adjacent_tests(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "graph_seed",
        "initialization_seed",
    ]

    rows = []

    for lower, upper in zip(
        LEARNING_RATES[:-1],
        LEARNING_RATES[1:],
    ):
        left = runs[
            np.isclose(
                runs["learning_rate"],
                lower,
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
                    "lower_accuracy",
                "recovered":
                    "lower_recovered",
            }
        )

        right = runs[
            np.isclose(
                runs["learning_rate"],
                upper,
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
                    "upper_accuracy",
                "recovered":
                    "upper_recovered",
            }
        )

        paired = left.merge(
            right,
            on=keys,
            validate="one_to_one",
        )

        differences = (
            paired["upper_accuracy"]
            - paired["lower_accuracy"]
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

        upper_only = int(
            (
                paired["upper_recovered"]
                & ~paired["lower_recovered"]
            ).sum()
        )

        lower_only = int(
            (
                paired["lower_recovered"]
                & ~paired["upper_recovered"]
            ).sum()
        )

        discordant = (
            upper_only
            + lower_only
        )

        mcnemar_pvalue = (
            float(
                stats.binomtest(
                    upper_only,
                    n=discordant,
                    p=0.5,
                    alternative="two-sided",
                ).pvalue
            )
            if discordant
            else 1.0
        )

        rows.append(
            {
                "lower_lr": lower,
                "upper_lr": upper,
                "mean_accuracy_change":
                    float(
                        differences.mean()
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
                "upper_only_recovered":
                    upper_only,
                "lower_only_recovered":
                    lower_only,
                "mcnemar_exact_pvalue":
                    mcnemar_pvalue,
            }
        )

    result = pd.DataFrame(rows)

    result[
        "wilcoxon_holm_pvalue"
    ] = holm_adjust(
        result["wilcoxon_pvalue"].tolist()
    )

    result[
        "mcnemar_holm_pvalue"
    ] = holm_adjust(
        result[
            "mcnemar_exact_pvalue"
        ].tolist()
    )

    return result


def recovery_thresholds(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for keys, group in runs.groupby(
        [
            "graph_seed",
            "initialization_seed",
        ]
    ):
        group = group.sort_values(
            "learning_rate"
        )

        successful_rates = group.loc[
            group["recovered"],
            "learning_rate",
        ].tolist()

        pattern = group[
            "recovered"
        ].astype(bool).tolist()

        first_failure = math.nan
        contiguous_limit = math.nan

        for learning_rate, recovered in zip(
            group["learning_rate"],
            pattern,
        ):
            if recovered:
                contiguous_limit = float(
                    learning_rate
                )
            else:
                first_failure = float(
                    learning_rate
                )
                break

        monotonic = all(
            not pattern[index]
            or all(
                pattern[: index + 1]
            )
            for index in range(
                len(pattern)
            )
        )

        rows.append(
            {
                "graph_seed": keys[0],
                "initialization_seed": keys[1],
                "successful_lr_count":
                    len(successful_rates),
                "successful_learning_rates":
                    ",".join(
                        f"{value:.3f}"
                        for value in successful_rates
                    ),
                "largest_contiguous_success_lr":
                    contiguous_limit,
                "first_failure_lr":
                    first_failure,
                "monotonic_success_pattern":
                    monotonic,
            }
        )

    return pd.DataFrame(rows).sort_values(
        [
            "graph_seed",
            "initialization_seed",
        ]
    )


def retention_predictors(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for learning_rate, group in runs.groupby(
        "learning_rate"
    ):
        recovered = group[
            "recovered"
        ].astype(bool)

        number_positive = int(
            recovered.sum()
        )

        number_negative = int(
            (~recovered).sum()
        )

        if (
            number_positive == 0
            or number_negative == 0
        ):
            continue

        positive_values = group.loc[
            recovered,
            "epoch1_pairwise_retention",
        ]

        negative_values = group.loc[
            ~recovered,
            "epoch1_pairwise_retention",
        ]

        test = stats.mannwhitneyu(
            positive_values,
            negative_values,
            alternative="two-sided",
            method="asymptotic",
        )

        auc = float(
            test.statistic
            / (
                number_positive
                * number_negative
            )
        )

        rows.append(
            {
                "learning_rate":
                    learning_rate,
                "recovered_n":
                    number_positive,
                "failed_n":
                    number_negative,
                "recovered_retention_mean":
                    float(
                        positive_values.mean()
                    ),
                "failed_retention_mean":
                    float(
                        negative_values.mean()
                    ),
                "auc_recovered_higher":
                    auc,
                "direction_free_auc":
                    max(
                        auc,
                        1.0 - auc,
                    ),
                "mannwhitney_pvalue":
                    float(test.pvalue),
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        result[
            "holm_pvalue"
        ] = holm_adjust(
            result[
                "mannwhitney_pvalue"
            ].tolist()
        )

    return result


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
            * np.square(column_sums).sum()
            - total**2
        )
    )

    denominator = (
        number_conditions * total
        - np.square(row_sums).sum()
    )

    if denominator <= 0.0:
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
            accuracy[column]
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
                    len(LEARNING_RATES) - 1,
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
                    len(LEARNING_RATES) - 1,
                "pvalue":
                    q_pvalue,
            },
        ]
    )


def make_plots(
    summary: pd.DataFrame,
) -> None:
    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(9, 6))

    plt.plot(
        summary["learning_rate"],
        summary["recovery_rate"],
        marker="o",
    )

    plt.fill_between(
        summary["learning_rate"],
        summary["recovery_ci95_lower"],
        summary["recovery_ci95_upper"],
        alpha=0.2,
    )

    plt.xscale("log")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Learning rate")
    plt.ylabel("Recovery rate")
    plt.title(
        "GraphSAGE L8 recovery phase curve"
    )
    plt.tight_layout()

    path = (
        PLOT_DIR
        / "recovery_rate_by_lr.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("Saved:", path)

    plt.figure(figsize=(9, 6))

    plt.plot(
        summary["learning_rate"],
        summary[
            "epoch1_retention_mean"
        ],
        marker="o",
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Learning rate")
    plt.ylabel(
        "Epoch-1 pairwise retention"
    )
    plt.title(
        "First-step representation retention"
    )
    plt.tight_layout()

    path = (
        PLOT_DIR
        / "epoch1_retention_by_lr.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("Saved:", path)

    plt.figure(figsize=(9, 6))

    plt.plot(
        summary["learning_rate"],
        summary["first_test_90_mean"],
        marker="o",
    )

    plt.xscale("log")
    plt.xlabel("Learning rate")
    plt.ylabel(
        "Mean epoch reaching 90% test accuracy"
    )
    plt.title(
        "Recovery speed among successful runs"
    )
    plt.tight_layout()

    path = (
        PLOT_DIR
        / "recovery_speed_by_lr.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("Saved:", path)


def main() -> None:
    runs, metrics = load_all_runs()

    runs = add_epoch1_retention(
        runs,
        metrics,
    )

    summary = summarize_runs(runs)
    metric_summary = summarize_metrics(
        metrics
    )

    adjacent = adjacent_tests(runs)

    thresholds = recovery_thresholds(
        runs
    )

    retention = retention_predictors(
        runs
    )

    global_frame = global_tests(runs)

    runs.to_csv(
        RUN_OUTPUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    metric_summary.to_csv(
        METRIC_OUTPUT,
        index=False,
    )

    adjacent.to_csv(
        ADJACENT_OUTPUT,
        index=False,
    )

    thresholds.to_csv(
        THRESHOLD_OUTPUT,
        index=False,
    )

    retention.to_csv(
        RETENTION_OUTPUT,
        index=False,
    )

    global_frame.to_csv(
        GLOBAL_OUTPUT,
        index=False,
    )

    make_plots(summary)

    print(
        "\n=== Learning-rate phase curve ==="
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.10f}",
        )
    )

    print(
        "\n=== Adjacent learning-rate tests ==="
    )

    print(
        adjacent.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.10f}",
        )
    )

    print(
        "\n=== First-step retention predictors ==="
    )

    if retention.empty:
        print(
            "No learning rate contained "
            "both successful and failed runs."
        )
    else:
        print(
            retention.to_string(
                index=False,
                float_format=lambda value:
                    f"{value:.10f}",
            )
        )

    print(
        "\n=== Empirical stability thresholds ==="
    )

    print(
        thresholds.to_string(
            index=False,
        )
    )

    print(
        "\n=== Global repeated-measures tests ==="
    )

    print(
        global_frame.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.10f}",
        )
    )

    print("\nSaved:", RUN_OUTPUT)
    print("Saved:", SUMMARY_OUTPUT)
    print("Saved:", METRIC_OUTPUT)
    print("Saved:", ADJACENT_OUTPUT)
    print("Saved:", THRESHOLD_OUTPUT)
    print("Saved:", RETENTION_OUTPUT)
    print("Saved:", GLOBAL_OUTPUT)


if __name__ == "__main__":
    main()
