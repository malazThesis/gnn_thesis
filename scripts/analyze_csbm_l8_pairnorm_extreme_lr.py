from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


EXTREME_DIR = Path(
    "runs/csbm_l8_pairnorm_extreme_lr_fs050"
)

PREVIOUS_RUNS = Path(
    "runs/csbm_l8_pairnorm_upper_lr_fs050/"
    "csbm_l8_pairnorm_lr_phase_run_level.csv"
)

RUN_OUTPUT = (
    EXTREME_DIR
    / "csbm_l8_pairnorm_extreme_lr_run_level.csv"
)

SUMMARY_OUTPUT = (
    EXTREME_DIR
    / "csbm_l8_pairnorm_extreme_lr_summary.csv"
)

ADJACENT_OUTPUT = (
    EXTREME_DIR
    / "csbm_l8_pairnorm_extreme_lr_adjacent_tests.csv"
)

STABILITY_OUTPUT = (
    EXTREME_DIR
    / "csbm_l8_pairnorm_extreme_lr_stability.csv"
)

EXTREME_RATES = [0.2, 0.3, 0.5]
BASELINE_STABLE_LIMIT = 0.005


def parse_graph_seed(summary: dict) -> int:
    dataset = str(summary.get("dataset", ""))

    match = re.search(
        r"-G(\d+)(?:-|$)",
        dataset,
    )

    if not match:
        raise RuntimeError(
            f"Cannot parse graph seed from {dataset}"
        )

    return int(match.group(1))


def parse_learning_rate(summary: dict) -> float:
    for key in ["lr", "learning_rate"]:
        if summary.get(key) is not None:
            return float(summary[key])

    dataset = str(summary.get("dataset", ""))

    match = re.search(
        r"PNEXT(\d+p\d+)",
        dataset,
    )

    if not match:
        raise RuntimeError(
            f"Cannot parse learning rate from {dataset}"
        )

    return float(
        match.group(1).replace("p", ".")
    )


def first_epoch_at_least(
    history: pd.DataFrame,
    column: str,
    threshold: float,
) -> float:
    selected = history[
        np.isfinite(history[column])
        & (history[column] >= threshold)
    ]

    if selected.empty:
        return math.nan

    return float(selected["epoch"].min())


def holm_adjust(pvalues: list[float]) -> list[float]:
    values = np.asarray(pvalues, dtype=float)
    adjusted = np.full(len(values), np.nan)

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


def load_extreme_runs() -> pd.DataFrame:
    summary_files = sorted(
        EXTREME_DIR.glob("*_summary.json")
    )

    if len(summary_files) != 75:
        raise RuntimeError(
            f"Expected 75 summary files, "
            f"found {len(summary_files)}"
        )

    rows = []

    numeric_columns = [
        "loss",
        "train_acc",
        "val_acc",
        "test_acc",
    ]

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

        if history.empty:
            raise RuntimeError(
                f"Empty history: {history_path}"
            )

        finite_matrix = np.isfinite(
            history[numeric_columns]
            .to_numpy(dtype=float)
        )

        numerical_failure = bool(
            not finite_matrix.all()
        )

        final = history.iloc[-1]

        valid_checkpoints = history[
            np.isfinite(history["val_acc"])
            & np.isfinite(history["test_acc"])
        ]

        if valid_checkpoints.empty:
            best_epoch = math.nan
            best_val = math.nan
            test_at_best = math.nan
        else:
            best = valid_checkpoints.loc[
                valid_checkpoints[
                    "val_acc"
                ].idxmax()
            ]

            best_epoch = float(best["epoch"])
            best_val = float(best["val_acc"])
            test_at_best = float(
                best["test_acc"]
            )

        final_test = float(
            final["test_acc"]
        )

        final_recovered = bool(
            np.isfinite(final_test)
            and final_test >= 0.90
        )

        checkpoint_recovered = bool(
            np.isfinite(test_at_best)
            and test_at_best >= 0.90
        )

        collapsed = bool(
            np.isfinite(final_test)
            and final_test <= 0.52
        )

        partial = bool(
            not numerical_failure
            and not final_recovered
            and not collapsed
        )

        finite_losses = history.loc[
            np.isfinite(history["loss"]),
            "loss",
        ]

        rows.append(
            {
                "graph_seed":
                    parse_graph_seed(summary),
                "initialization_seed":
                    int(summary["seed"]),
                "learning_rate":
                    parse_learning_rate(summary),
                "final_loss":
                    float(final["loss"]),
                "maximum_finite_loss":
                    (
                        float(finite_losses.max())
                        if not finite_losses.empty
                        else math.nan
                    ),
                "final_train_acc":
                    float(final["train_acc"]),
                "final_val_acc":
                    float(final["val_acc"]),
                "final_test_acc":
                    final_test,
                "best_epoch":
                    best_epoch,
                "best_val_acc":
                    best_val,
                "test_acc_at_best_val":
                    test_at_best,
                "first_test_90":
                    first_epoch_at_least(
                        history,
                        "test_acc",
                        0.90,
                    ),
                "final_recovered":
                    final_recovered,
                "checkpoint_recovered":
                    checkpoint_recovered,
                "collapsed":
                    collapsed,
                "partial":
                    partial,
                "numerical_failure":
                    numerical_failure,
            }
        )

    runs = pd.DataFrame(rows)

    keys = [
        "graph_seed",
        "initialization_seed",
        "learning_rate",
    ]

    if runs.duplicated(keys).any():
        raise RuntimeError(
            "Duplicate graph/init/lr combinations."
        )

    observed_rates = sorted(
        runs["learning_rate"].unique()
    )

    if not np.allclose(
        observed_rates,
        EXTREME_RATES,
    ):
        raise RuntimeError(
            f"Unexpected learning rates: "
            f"{observed_rates}"
        )

    return runs.sort_values(keys)


def summarize(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for learning_rate, group in runs.groupby(
        "learning_rate"
    ):
        rows.append(
            {
                "learning_rate":
                    learning_rate,
                "n":
                    len(group),
                "final_recovered_runs":
                    int(
                        group[
                            "final_recovered"
                        ].sum()
                    ),
                "checkpoint_recovered_runs":
                    int(
                        group[
                            "checkpoint_recovered"
                        ].sum()
                    ),
                "collapsed_runs":
                    int(group["collapsed"].sum()),
                "partial_runs":
                    int(group["partial"].sum()),
                "numerical_failure_runs":
                    int(
                        group[
                            "numerical_failure"
                        ].sum()
                    ),
                "final_recovery_rate":
                    float(
                        group[
                            "final_recovered"
                        ].mean()
                    ),
                "final_test_acc_mean":
                    float(
                        group[
                            "final_test_acc"
                        ].mean()
                    ),
                "final_test_acc_min":
                    float(
                        group[
                            "final_test_acc"
                        ].min()
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
                "maximum_loss_median":
                    float(
                        group[
                            "maximum_finite_loss"
                        ].median()
                    ),
                "maximum_loss_max":
                    float(
                        group[
                            "maximum_finite_loss"
                        ].max()
                    ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "learning_rate"
    )


def adjacent_tests(
    extreme: pd.DataFrame,
) -> pd.DataFrame:
    previous = pd.read_csv(PREVIOUS_RUNS)

    previous = previous[
        np.isclose(
            previous["learning_rate"],
            0.1,
        )
    ][
        [
            "graph_seed",
            "initialization_seed",
            "learning_rate",
            "final_test_acc",
            "final_recovered",
        ]
    ].copy()

    combined = pd.concat(
        [
            previous,
            extreme[
                [
                    "graph_seed",
                    "initialization_seed",
                    "learning_rate",
                    "final_test_acc",
                    "final_recovered",
                ]
            ],
        ],
        ignore_index=True,
    )

    keys = [
        "graph_seed",
        "initialization_seed",
    ]

    rates = [0.1, 0.2, 0.3, 0.5]
    rows = []

    for lower, upper in zip(
        rates[:-1],
        rates[1:],
    ):
        left = combined[
            np.isclose(
                combined["learning_rate"],
                lower,
            )
        ].rename(
            columns={
                "final_test_acc":
                    "lower_accuracy",
                "final_recovered":
                    "lower_recovered",
            }
        )

        right = combined[
            np.isclose(
                combined["learning_rate"],
                upper,
            )
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

        finite = differences[
            np.isfinite(differences)
        ]

        try:
            wilcoxon_pvalue = float(
                stats.wilcoxon(
                    finite,
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
            upper_only + lower_only
        )

        mcnemar_pvalue = (
            float(
                stats.binomtest(
                    upper_only,
                    n=discordant,
                    p=0.5,
                ).pvalue
            )
            if discordant
            else 1.0
        )

        rows.append(
            {
                "lower_lr":
                    lower,
                "upper_lr":
                    upper,
                "mean_accuracy_change":
                    float(finite.mean()),
                "positive_combinations":
                    int((finite > 0).sum()),
                "negative_combinations":
                    int((finite < 0).sum()),
                "ties":
                    int((finite == 0).sum()),
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


def stability_result(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    fully_stable = summary[
        summary["final_recovered_runs"] == 25
    ]

    largest_stable = (
        float(
            fully_stable[
                "learning_rate"
            ].max()
        )
        if not fully_stable.empty
        else math.nan
    )

    highest_tested = float(
        summary["learning_rate"].max()
    )

    censored_above_range = bool(
        np.isclose(
            largest_stable,
            highest_tested,
        )
    )

    first_unstable = summary[
        summary["final_recovered_runs"] < 25
    ]["learning_rate"]

    return pd.DataFrame(
        [
            {
                "baseline_fully_stable_lr":
                    BASELINE_STABLE_LIMIT,
                "pairnorm_largest_fully_stable_lr":
                    largest_stable,
                "stability_factor_vs_baseline":
                    (
                        largest_stable
                        / BASELINE_STABLE_LIMIT
                    ),
                "highest_pairnorm_lr_tested":
                    highest_tested,
                "first_not_fully_stable_lr":
                    (
                        float(first_unstable.min())
                        if not first_unstable.empty
                        else math.nan
                    ),
                "pairnorm_boundary_above_tested_range":
                    censored_above_range,
            }
        ]
    )


def main() -> None:
    runs = load_extreme_runs()
    summary = summarize(runs)
    adjacent = adjacent_tests(runs)
    stability = stability_result(summary)

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

    stability.to_csv(
        STABILITY_OUTPUT,
        index=False,
    )

    print("\n=== EXTREME PAIRNORM-LERNRATEN ===")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.10f}",
        )
    )

    print("\n=== BENACHBARTE LERNRATEN ===")

    print(
        adjacent.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.10f}",
        )
    )

    print("\n=== STABILITÄTSGRENZE ===")

    print(
        stability.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.10f}",
        )
    )

    print("\nSaved:", RUN_OUTPUT)
    print("Saved:", SUMMARY_OUTPUT)
    print("Saved:", ADJACENT_OUTPUT)
    print("Saved:", STABILITY_OUTPUT)


if __name__ == "__main__":
    main()
