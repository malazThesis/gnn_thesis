from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


FACTORIAL_DIR = Path(
    "runs/csbm_l8_graph_init_factorial_fs050"
)

LOWER_DIR = Path(
    "runs/csbm_l8_pairnorm_lr_sensitivity_fs050"
)

UPPER_DIR = Path(
    "runs/csbm_l8_pairnorm_upper_lr_fs050"
)

EXISTING_RUN_PATH = (
    LOWER_DIR
    / "csbm_l8_model_lr_run_level.csv"
)

BASELINE_PHASE_PATH = Path(
    "runs/csbm_l8_lr_transition_fs050/"
    "csbm_l8_lr_phase_summary.csv"
)

PLOT_DIR = Path(
    "plots/csbm_l8_pairnorm_lr_phase"
)

RUN_OUTPUT = (
    UPPER_DIR
    / "csbm_l8_pairnorm_lr_phase_run_level.csv"
)

SUMMARY_OUTPUT = (
    UPPER_DIR
    / "csbm_l8_pairnorm_lr_phase_summary.csv"
)

ADJACENT_OUTPUT = (
    UPPER_DIR
    / "csbm_l8_pairnorm_lr_phase_adjacent_tests.csv"
)

THRESHOLD_OUTPUT = (
    UPPER_DIR
    / "csbm_l8_pairnorm_lr_phase_thresholds.csv"
)

COMPARISON_OUTPUT = (
    UPPER_DIR
    / "csbm_l8_pairnorm_vs_baseline_stability.csv"
)

LEARNING_RATES = [
    0.001,
    0.003,
    0.010,
    0.030,
    0.050,
    0.070,
    0.100,
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
    forced_rate: float | None = None,
) -> float:
    if forced_rate is not None:
        return forced_rate

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
        r"(?:PNLR|PNUP)(\d+p\d+)",
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


def final_pairnorm_hidden(
    frame: pd.DataFrame,
) -> pd.Series:
    selected = frame[
        frame["layer_name"]
        .astype(str)
        .str.startswith("pns.")
    ].copy()

    if selected.empty:
        raise RuntimeError(
            "No PairNorm hidden rows found."
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


def wilson_interval(
    successes: int,
    total: int,
) -> tuple[float, float]:
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

    adjusted = np.full_like(
        values,
        np.nan,
    )

    valid = np.where(
        ~np.isnan(values)
    )[0]

    order = valid[
        np.argsort(values[valid])
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


def load_upper_runs() -> pd.DataFrame:
    summary_files = sorted(
        UPPER_DIR.glob("*_summary.json")
    )

    if len(summary_files) != 75:
        raise RuntimeError(
            f"Expected 75 upper-LR runs, "
            f"found {len(summary_files)}"
        )

    rows = []

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
        ).sort_values("epoch")

        final = history.iloc[-1]
        best = history.loc[
            history["val_acc"].idxmax()
        ]

        final_test = float(
            final["test_acc"]
        )

        checkpoint_test = float(
            best["test_acc"]
        )

        rows.append(
            {
                "model":
                    "GraphSAGEPairNorm",
                "graph_seed":
                    resolve_graph_seed(
                        summary,
                        summary_path.name,
                    ),
                "initialization_seed":
                    int(summary["seed"]),
                "learning_rate":
                    resolve_learning_rate(
                        summary
                    ),
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
                    checkpoint_test,
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
                "final_recovered":
                    final_test >= 0.90,
                "checkpoint_recovered":
                    checkpoint_test >= 0.90,
                "collapsed":
                    final_test <= 0.52,
            }
        )

    return pd.DataFrame(rows)


def load_all_runs() -> pd.DataFrame:
    existing = pd.read_csv(
        EXISTING_RUN_PATH
    )

    existing = existing[
        existing["model"]
        == "GraphSAGEPairNorm"
    ].copy()

    existing[
        "final_recovered"
    ] = (
        existing["final_test_acc"]
        >= 0.90
    )

    existing[
        "checkpoint_recovered"
    ] = (
        existing["test_acc_at_best_val"]
        >= 0.90
    )

    upper = load_upper_runs()

    required = [
        "model",
        "graph_seed",
        "initialization_seed",
        "learning_rate",
        "final_loss",
        "final_train_acc",
        "final_val_acc",
        "final_test_acc",
        "best_epoch",
        "best_val_acc",
        "test_acc_at_best_val",
        "first_train_90",
        "first_val_90",
        "first_test_90",
        "final_recovered",
        "checkpoint_recovered",
        "collapsed",
    ]

    runs = pd.concat(
        [
            existing[required],
            upper[required],
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
            "Duplicate graph/init/lr combinations."
        )

    if len(runs) != 175:
        raise RuntimeError(
            f"Expected 175 PairNorm runs, "
            f"found {len(runs)}"
        )

    observed = sorted(
        runs["learning_rate"].unique()
    )

    if not np.allclose(
        observed,
        LEARNING_RATES,
    ):
        raise RuntimeError(
            f"Unexpected learning rates: "
            f"{observed}"
        )

    return runs.sort_values(keys)


def load_epoch1_retention() -> pd.DataFrame:
    specifications = [
        (
            FACTORIAL_DIR,
            0.01,
        ),
        (
            LOWER_DIR,
            None,
        ),
        (
            UPPER_DIR,
            None,
        ),
    ]

    rows = []

    for directory, forced_rate in specifications:
        for summary_path in sorted(
            directory.glob("*_summary.json")
        ):
            with summary_path.open(
                "r",
                encoding="utf-8",
            ) as handle:
                summary = json.load(handle)

            if (
                summary["model"]
                != "GraphSAGEPairNorm"
            ):
                continue

            learning_rate = (
                resolve_learning_rate(
                    summary,
                    forced_rate=forced_rate,
                )
            )

            metric_path = summary_path.with_name(
                summary_path.name.replace(
                    "_summary.json",
                    "_oversmoothing.csv",
                )
            )

            metrics = pd.read_csv(metric_path)

            epoch0 = final_pairnorm_hidden(
                metrics[
                    metrics["epoch"] == 0
                ]
            )

            epoch1 = final_pairnorm_hidden(
                metrics[
                    metrics["epoch"] == 1
                ]
            )

            distance0 = float(
                epoch0[
                    "mean_pairwise_cosine_distance"
                ]
            )

            distance1 = float(
                epoch1[
                    "mean_pairwise_cosine_distance"
                ]
            )

            rows.append(
                {
                    "graph_seed":
                        resolve_graph_seed(
                            summary,
                            summary_path.name,
                        ),
                    "initialization_seed":
                        int(summary["seed"]),
                    "learning_rate":
                        learning_rate,
                    "pairwise_epoch0":
                        distance0,
                    "pairwise_epoch1":
                        distance1,
                    "epoch1_pairwise_retention":
                        (
                            distance1 / distance0
                            if distance0 > 0.0
                            else math.nan
                        ),
                }
            )

    result = pd.DataFrame(rows)

    keys = [
        "graph_seed",
        "initialization_seed",
        "learning_rate",
    ]

    if len(result) != 175:
        raise RuntimeError(
            f"Expected 175 metric rows, "
            f"found {len(result)}"
        )

    if result.duplicated(keys).any():
        raise RuntimeError(
            "Duplicate retention rows."
        )

    return result


def summarize(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for learning_rate, group in runs.groupby(
        "learning_rate"
    ):
        final_successes = int(
            group["final_recovered"].sum()
        )

        checkpoint_successes = int(
            group[
                "checkpoint_recovered"
            ].sum()
        )

        lower, upper = wilson_interval(
            final_successes,
            len(group),
        )

        rows.append(
            {
                "learning_rate":
                    learning_rate,
                "n":
                    len(group),
                "final_recovered_runs":
                    final_successes,
                "checkpoint_recovered_runs":
                    checkpoint_successes,
                "collapsed_runs":
                    int(
                        group["collapsed"].sum()
                    ),
                "final_recovery_rate":
                    final_successes
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
            }
        )

    return pd.DataFrame(rows).sort_values(
        "learning_rate"
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
                "final_recovered",
            ]
        ].rename(
            columns={
                "final_test_acc":
                    "lower_accuracy",
                "final_recovered":
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
                "final_recovered",
            ]
        ].rename(
            columns={
                "final_test_acc":
                    "upper_accuracy",
                "final_recovered":
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
            wilcoxon_pvalue = float(
                stats.wilcoxon(
                    differences,
                    alternative="two-sided",
                ).pvalue
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
            if discordant > 0
            else 1.0
        )

        rows.append(
            {
                "lower_lr":
                    lower,
                "upper_lr":
                    upper,
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
                "upper_only_recovered":
                    upper_only,
                "lower_only_recovered":
                    lower_only,
                "wilcoxon_pvalue":
                    wilcoxon_pvalue,
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


def thresholds(
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

        pattern = group[
            "final_recovered"
        ].astype(bool).tolist()

        rates = group[
            "learning_rate"
        ].tolist()

        contiguous_limit = math.nan
        first_failure = math.nan

        for learning_rate, recovered in zip(
            rates,
            pattern,
        ):
            if recovered:
                contiguous_limit = learning_rate
            else:
                first_failure = learning_rate
                break

        monotonic = all(
            not pattern[index]
            or all(pattern[: index + 1])
            for index in range(len(pattern))
        )

        rows.append(
            {
                "graph_seed": keys[0],
                "initialization_seed":
                    keys[1],
                "successful_lr_count":
                    int(sum(pattern)),
                "largest_contiguous_success_lr":
                    contiguous_limit,
                "first_failure_lr":
                    first_failure,
                "monotonic_success_pattern":
                    monotonic,
                "successful_learning_rates":
                    ",".join(
                        f"{rate:.3f}"
                        for rate, recovered in zip(
                            rates,
                            pattern,
                        )
                        if recovered
                    ),
            }
        )

    return pd.DataFrame(rows)


def stability_comparison(
    pairnorm_summary: pd.DataFrame,
) -> pd.DataFrame:
    baseline = pd.read_csv(
        BASELINE_PHASE_PATH
    )

    baseline_stable = baseline[
        baseline["recovered_runs"] == 25
    ]["learning_rate"]

    pairnorm_stable = pairnorm_summary[
        pairnorm_summary[
            "final_recovered_runs"
        ] == 25
    ]["learning_rate"]

    baseline_limit = float(
        baseline_stable.max()
    )

    pairnorm_limit = (
        float(pairnorm_stable.max())
        if not pairnorm_stable.empty
        else math.nan
    )

    return pd.DataFrame(
        [
            {
                "model":
                    "GraphSAGE",
                "largest_fully_stable_lr":
                    baseline_limit,
                "reference":
                    "25/25 final recovery",
            },
            {
                "model":
                    "GraphSAGEPairNorm",
                "largest_fully_stable_lr":
                    pairnorm_limit,
                "reference":
                    "25/25 final recovery",
            },
            {
                "model":
                    "PairNorm stability factor",
                "largest_fully_stable_lr":
                    (
                        pairnorm_limit
                        / baseline_limit
                        if not math.isnan(
                            pairnorm_limit
                        )
                        else math.nan
                    ),
                "reference":
                    "PairNorm limit / baseline limit",
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
        summary["final_recovery_rate"],
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
    plt.ylabel("Final recovery rate")
    plt.title(
        "PairNorm L8 recovery phase curve"
    )
    plt.tight_layout()

    path = (
        PLOT_DIR
        / "pairnorm_recovery_rate_by_lr.png"
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
        "PairNorm optimization speed"
    )
    plt.tight_layout()

    path = (
        PLOT_DIR
        / "pairnorm_recovery_speed_by_lr.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("Saved:", path)


def main() -> None:
    runs = load_all_runs()

    retention = load_epoch1_retention()

    runs = runs.merge(
        retention,
        on=[
            "graph_seed",
            "initialization_seed",
            "learning_rate",
        ],
        validate="one_to_one",
    )

    summary = summarize(runs)
    adjacent = adjacent_tests(runs)
    threshold_frame = thresholds(runs)

    comparison = stability_comparison(
        summary
    )

    runs.to_csv(
        RUN_OUTPUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    adjacent.to_csv(
        ADJACENT_OUTPUT,
        index=False,
    )

    threshold_frame.to_csv(
        THRESHOLD_OUTPUT,
        index=False,
    )

    comparison.to_csv(
        COMPARISON_OUTPUT,
        index=False,
    )

    make_plots(summary)

    print(
        "\n=== PairNorm learning-rate phase ==="
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
        "\n=== Stability comparison ==="
    )

    print(
        comparison.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.10f}",
        )
    )

    print(
        "\n=== Individual thresholds ==="
    )

    print(
        threshold_frame.to_string(
            index=False,
        )
    )

    print("\nSaved:", RUN_OUTPUT)
    print("Saved:", SUMMARY_OUTPUT)
    print("Saved:", ADJACENT_OUTPUT)
    print("Saved:", THRESHOLD_OUTPUT)
    print("Saved:", COMPARISON_OUTPUT)


if __name__ == "__main__":
    main()
