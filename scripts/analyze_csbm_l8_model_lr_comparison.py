from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


BASELINE_PATH = Path(
    "runs/csbm_l8_lr_sensitivity_fs050/"
    "csbm_l8_lr_run_level.csv"
)

FACTORIAL_PATH = Path(
    "runs/csbm_l8_graph_init_factorial_fs050/"
    "csbm_l8_graph_init_run_level.csv"
)

PAIRNORM_DIR = Path(
    "runs/csbm_l8_pairnorm_lr_sensitivity_fs050"
)

OUTPUT_DIR = PAIRNORM_DIR

RUN_OUTPUT = (
    OUTPUT_DIR
    / "csbm_l8_model_lr_run_level.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "csbm_l8_model_lr_summary.csv"
)

MATCHED_OUTPUT = (
    OUTPUT_DIR
    / "csbm_l8_model_lr_matched_effects.csv"
)

GLOBAL_TUNING_OUTPUT = (
    OUTPUT_DIR
    / "csbm_l8_model_lr_global_validation_tuning.csv"
)

PER_RUN_TUNING_OUTPUT = (
    OUTPUT_DIR
    / "csbm_l8_model_lr_per_run_validation_tuning.csv"
)

ROBUSTNESS_OUTPUT = (
    OUTPUT_DIR
    / "csbm_l8_model_lr_robustness.csv"
)

LEARNING_RATES = [
    0.001,
    0.003,
    0.01,
    0.03,
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
) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan

    z = 1.959963984540054
    proportion = successes / total

    denominator = (
        1.0
        + z**2 / total
    )

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
        r"PNLR(\d+p\d+)",
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


def load_new_pairnorm_runs() -> pd.DataFrame:
    files = sorted(
        PAIRNORM_DIR.glob("*_summary.json")
    )

    if len(files) != 75:
        raise RuntimeError(
            f"Expected 75 new PairNorm runs, "
            f"found {len(files)}"
        )

    rows = []

    for summary_path in files:
        with summary_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            summary = json.load(handle)

        if (
            summary["model"]
            != "GraphSAGEPairNorm"
        ):
            raise RuntimeError(
                f"Unexpected model in "
                f"{summary_path.name}"
            )

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
                    float(
                        final["train_acc"]
                    ),
                "final_val_acc":
                    float(
                        final["val_acc"]
                    ),
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

    return pd.DataFrame(rows)


def load_all_runs() -> pd.DataFrame:
    baseline = pd.read_csv(
        BASELINE_PATH
    )

    baseline["model"] = "GraphSAGE"

    pairnorm_001 = pd.read_csv(
        FACTORIAL_PATH
    )

    pairnorm_001 = pairnorm_001[
        pairnorm_001["model"]
        == "GraphSAGEPairNorm"
    ].copy()

    pairnorm_001[
        "learning_rate"
    ] = 0.01

    new_pairnorm = (
        load_new_pairnorm_runs()
    )

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
        "outcome",
        "collapsed",
        "recovered",
    ]

    runs = pd.concat(
        [
            baseline[required],
            pairnorm_001[required],
            new_pairnorm[required],
        ],
        ignore_index=True,
    )

    keys = [
        "model",
        "graph_seed",
        "initialization_seed",
        "learning_rate",
    ]

    if runs.duplicated(keys).any():
        duplicates = runs[
            runs.duplicated(
                keys,
                keep=False,
            )
        ]

        raise RuntimeError(
            "Duplicate model/graph/init/lr rows:\n"
            + duplicates[
                keys
            ].to_string(index=False)
        )

    if len(runs) != 200:
        raise RuntimeError(
            f"Expected 200 total runs, "
            f"found {len(runs)}"
        )

    for model, group in runs.groupby(
        "model"
    ):
        if len(group) != 100:
            raise RuntimeError(
                f"{model}: expected 100 runs, "
                f"found {len(group)}"
            )

        observed = sorted(
            group[
                "learning_rate"
            ].unique()
        )

        if not np.allclose(
            observed,
            LEARNING_RATES,
        ):
            raise RuntimeError(
                f"{model}: unexpected rates "
                f"{observed}"
            )

    return runs.sort_values(
        keys
    ).reset_index(drop=True)


def summarize_runs(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for keys, group in runs.groupby(
        [
            "model",
            "learning_rate",
        ]
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
                "model": keys[0],
                "learning_rate": keys[1],
                "n": len(group),
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
                "best_val_acc_mean":
                    float(
                        group[
                            "best_val_acc"
                        ].mean()
                    ),
                "test_at_best_val_mean":
                    float(
                        group[
                            "test_acc_at_best_val"
                        ].mean()
                    ),
                "first_val_90_mean":
                    float(
                        group[
                            "first_val_90"
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
        [
            "model",
            "learning_rate",
        ]
    )


def matched_effects(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "graph_seed",
        "initialization_seed",
        "learning_rate",
    ]

    baseline = runs[
        runs["model"] == "GraphSAGE"
    ][
        keys
        + [
            "final_test_acc",
            "test_acc_at_best_val",
            "first_test_90",
            "recovered",
        ]
    ].rename(
        columns={
            "final_test_acc":
                "baseline_final_test",
            "test_acc_at_best_val":
                "baseline_test_at_best_val",
            "first_test_90":
                "baseline_first_test_90",
            "recovered":
                "baseline_recovered",
        }
    )

    pairnorm = runs[
        runs["model"]
        == "GraphSAGEPairNorm"
    ][
        keys
        + [
            "final_test_acc",
            "test_acc_at_best_val",
            "first_test_90",
            "recovered",
        ]
    ].rename(
        columns={
            "final_test_acc":
                "pairnorm_final_test",
            "test_acc_at_best_val":
                "pairnorm_test_at_best_val",
            "first_test_90":
                "pairnorm_first_test_90",
            "recovered":
                "pairnorm_recovered",
        }
    )

    paired = baseline.merge(
        pairnorm,
        on=keys,
        validate="one_to_one",
    )

    rows = []

    for learning_rate, group in paired.groupby(
        "learning_rate"
    ):
        differences = (
            group["pairnorm_final_test"]
            - group["baseline_final_test"]
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

        pn_only = int(
            (
                group["pairnorm_recovered"]
                & ~group[
                    "baseline_recovered"
                ]
            ).sum()
        )

        baseline_only = int(
            (
                group[
                    "baseline_recovered"
                ]
                & ~group[
                    "pairnorm_recovered"
                ]
            ).sum()
        )

        discordant = (
            pn_only
            + baseline_only
        )

        mcnemar_pvalue = (
            float(
                stats.binomtest(
                    pn_only,
                    n=discordant,
                    p=0.5,
                    alternative="two-sided",
                ).pvalue
            )
            if discordant > 0
            else 1.0
        )

        both_recovered = group[
            group["baseline_recovered"]
            & group["pairnorm_recovered"]
        ].copy()

        speed_difference = (
            both_recovered[
                "pairnorm_first_test_90"
            ]
            - both_recovered[
                "baseline_first_test_90"
            ]
        )

        rows.append(
            {
                "learning_rate":
                    learning_rate,
                "n":
                    len(group),
                "mean_final_test_difference":
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
                "pairnorm_only_recovered":
                    pn_only,
                "baseline_only_recovered":
                    baseline_only,
                "mcnemar_exact_pvalue":
                    mcnemar_pvalue,
                "both_recovered_n":
                    len(both_recovered),
                "mean_epoch90_difference_pairnorm_minus_baseline":
                    float(
                        speed_difference.mean()
                    )
                    if len(speed_difference)
                    else math.nan,
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
        "learning_rate"
    )


def global_validation_tuning(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for model, group in summary.groupby(
        "model"
    ):
        selected = group.sort_values(
            [
                "best_val_acc_mean",
                "recovery_rate",
                "first_val_90_mean",
                "learning_rate",
            ],
            ascending=[
                False,
                False,
                True,
                True,
            ],
        ).iloc[0]

        rows.append(
            {
                "model": model,
                "selected_learning_rate":
                    selected[
                        "learning_rate"
                    ],
                "selection_rule":
                    "highest mean best-val accuracy",
                "best_val_acc_mean":
                    selected[
                        "best_val_acc_mean"
                    ],
                "test_at_best_val_mean":
                    selected[
                        "test_at_best_val_mean"
                    ],
                "final_test_acc_mean":
                    selected[
                        "final_test_acc_mean"
                    ],
                "recovery_rate":
                    selected[
                        "recovery_rate"
                    ],
                "first_test_90_mean":
                    selected[
                        "first_test_90_mean"
                    ],
            }
        )

    return pd.DataFrame(rows)


def per_run_validation_tuning(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    selected_rows = []

    for keys, group in runs.groupby(
        [
            "model",
            "graph_seed",
            "initialization_seed",
        ]
    ):
        selected = group.sort_values(
            [
                "best_val_acc",
                "first_val_90",
                "best_epoch",
                "learning_rate",
            ],
            ascending=[
                False,
                True,
                True,
                True,
            ],
            na_position="last",
        ).iloc[0]

        selected_rows.append(
            {
                "model": keys[0],
                "graph_seed": keys[1],
                "initialization_seed":
                    keys[2],
                "selected_learning_rate":
                    selected[
                        "learning_rate"
                    ],
                "selected_best_val_acc":
                    selected[
                        "best_val_acc"
                    ],
                "selected_test_at_best_val":
                    selected[
                        "test_acc_at_best_val"
                    ],
                "selected_final_test_acc":
                    selected[
                        "final_test_acc"
                    ],
                "selected_first_test_90":
                    selected[
                        "first_test_90"
                    ],
                "selected_recovered":
                    selected[
                        "recovered"
                    ],
            }
        )

    selected = pd.DataFrame(
        selected_rows
    )

    rows = []

    for model, group in selected.groupby(
        "model"
    ):
        rows.append(
            {
                "model": model,
                "n": len(group),
                "recovered_runs": int(
                    group[
                        "selected_recovered"
                    ].sum()
                ),
                "test_at_best_val_mean":
                    float(
                        group[
                            "selected_test_at_best_val"
                        ].mean()
                    ),
                "final_test_acc_mean":
                    float(
                        group[
                            "selected_final_test_acc"
                        ].mean()
                    ),
                "first_test_90_mean":
                    float(
                        group[
                            "selected_first_test_90"
                        ].mean()
                    ),
                "lr_0p001_selected":
                    int(
                        np.isclose(
                            group[
                                "selected_learning_rate"
                            ],
                            0.001,
                        ).sum()
                    ),
                "lr_0p003_selected":
                    int(
                        np.isclose(
                            group[
                                "selected_learning_rate"
                            ],
                            0.003,
                        ).sum()
                    ),
                "lr_0p010_selected":
                    int(
                        np.isclose(
                            group[
                                "selected_learning_rate"
                            ],
                            0.01,
                        ).sum()
                    ),
                "lr_0p030_selected":
                    int(
                        np.isclose(
                            group[
                                "selected_learning_rate"
                            ],
                            0.03,
                        ).sum()
                    ),
            }
        )

    return pd.DataFrame(rows)


def robustness_summary(
    runs: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for model, group in runs.groupby(
        "model"
    ):
        matrix = group.pivot(
            index=[
                "graph_seed",
                "initialization_seed",
            ],
            columns="learning_rate",
            values="final_test_acc",
        )

        recovery_rates = summary[
            summary["model"] == model
        ]

        fully_stable_lrs = int(
            (
                recovery_rates[
                    "recovered_runs"
                ]
                == 25
            ).sum()
        )

        rows.append(
            {
                "model": model,
                "fully_stable_learning_rates":
                    fully_stable_lrs,
                "mean_worst_lr_accuracy":
                    float(
                        matrix.min(axis=1).mean()
                    ),
                "mean_best_lr_accuracy":
                    float(
                        matrix.max(axis=1).mean()
                    ),
                "mean_accuracy_range_across_lr":
                    float(
                        (
                            matrix.max(axis=1)
                            - matrix.min(axis=1)
                        ).mean()
                    ),
                "minimum_accuracy_any_combination":
                    float(
                        matrix.min().min()
                    ),
                "maximum_accuracy_any_combination":
                    float(
                        matrix.max().max()
                    ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    runs = load_all_runs()
    summary = summarize_runs(runs)

    matched = matched_effects(runs)

    global_tuning = (
        global_validation_tuning(
            summary
        )
    )

    per_run_tuning = (
        per_run_validation_tuning(
            runs
        )
    )

    robustness = robustness_summary(
        runs,
        summary,
    )

    runs.to_csv(
        RUN_OUTPUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    matched.to_csv(
        MATCHED_OUTPUT,
        index=False,
    )

    global_tuning.to_csv(
        GLOBAL_TUNING_OUTPUT,
        index=False,
    )

    per_run_tuning.to_csv(
        PER_RUN_TUNING_OUTPUT,
        index=False,
    )

    robustness.to_csv(
        ROBUSTNESS_OUTPUT,
        index=False,
    )

    print(
        "\n=== Model × learning rate ==="
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.8f}",
        )
    )

    print(
        "\n=== Matched PairNorm effects ==="
    )

    print(
        matched.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.10f}",
        )
    )

    print(
        "\n=== Global validation tuning ==="
    )

    print(
        global_tuning.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.10f}",
        )
    )

    print(
        "\n=== Per-run validation tuning ==="
    )

    print(
        per_run_tuning.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.10f}",
        )
    )

    print(
        "\n=== Learning-rate robustness ==="
    )

    print(
        robustness.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.10f}",
        )
    )

    print("\nSaved:", RUN_OUTPUT)
    print("Saved:", SUMMARY_OUTPUT)
    print("Saved:", MATCHED_OUTPUT)
    print("Saved:", GLOBAL_TUNING_OUTPUT)
    print("Saved:", PER_RUN_TUNING_OUTPUT)
    print("Saved:", ROBUSTNESS_OUTPUT)


if __name__ == "__main__":
    main()
