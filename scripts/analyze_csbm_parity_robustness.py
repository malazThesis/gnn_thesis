from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


RUN_DIR = Path("runs/csbm_parity_replicated")
PLOT_DIR = Path("plots/csbm_parity_replicated")

PAIR_SUMMARY_PATH = (
    RUN_DIR / "csbm_parity_pairnorm_summary.csv"
)
PAIRED_PATH = (
    RUN_DIR / "csbm_parity_pairnorm_paired.csv"
)
RUN_LEVEL_PATH = (
    RUN_DIR / "csbm_parity_run_level.csv"
)

CORRECTED_PATH = (
    RUN_DIR
    / "csbm_parity_pairnorm_summary_corrected.csv"
)
PARITY_DEPTH_PATH = (
    RUN_DIR
    / "csbm_parity_parity_by_depth.csv"
)


def holm_adjust(
    values: pd.Series,
) -> np.ndarray:
    pvalues = pd.to_numeric(
        values,
        errors="coerce",
    ).to_numpy(dtype=float)

    adjusted = np.full(
        len(pvalues),
        np.nan,
    )

    valid = np.where(
        np.isfinite(pvalues)
    )[0]

    if len(valid) == 0:
        return adjusted

    order = valid[
        np.argsort(pvalues[valid])
    ]

    number = len(order)
    running_maximum = 0.0

    for rank, index in enumerate(order):
        corrected = (
            number - rank
        ) * pvalues[index]

        running_maximum = max(
            running_maximum,
            corrected,
        )

        adjusted[index] = min(
            1.0,
            running_maximum,
        )

    return adjusted


def confidence_interval(
    values: pd.Series,
) -> tuple[float, float, float]:
    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    mean = float(values.mean())

    if len(values) < 2:
        return mean, 0.0, 0.0

    std = float(values.std(ddof=1))
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

    return mean, std, ci95


def test_against_zero(
    values: pd.Series,
) -> tuple[float, float]:
    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    array = values.to_numpy()

    if len(array) < 2:
        return math.nan, math.nan

    if np.allclose(array, 0.0):
        return 1.0, 1.0

    t_pvalue = float(
        stats.ttest_1samp(
            array,
            popmean=0.0,
        ).pvalue
    )

    try:
        wilcoxon_pvalue = float(
            stats.wilcoxon(
                array,
                zero_method="wilcox",
                alternative="two-sided",
            ).pvalue
        )
    except ValueError:
        wilcoxon_pvalue = 1.0

    return t_pvalue, wilcoxon_pvalue


def correct_pairnorm_tests() -> pd.DataFrame:
    summary = pd.read_csv(
        PAIR_SUMMARY_PATH
    )

    summary[
        "paired_t_pvalue_holm"
    ] = holm_adjust(
        summary["paired_t_pvalue"]
    )

    summary[
        "wilcoxon_pvalue_holm"
    ] = holm_adjust(
        summary["wilcoxon_pvalue"]
    )

    summary[
        "significant_t_holm_005"
    ] = (
        summary["paired_t_pvalue_holm"]
        < 0.05
    )

    summary[
        "significant_wilcoxon_holm_005"
    ] = (
        summary["wilcoxon_pvalue_holm"]
        < 0.05
    )

    summary.to_csv(
        CORRECTED_PATH,
        index=False,
    )

    return summary


def calculate_parity_by_depth(
    runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows = []

    group_columns = [
        "target_homophily",
        "graph_seed",
        "model",
    ]

    for keys, group in runs.groupby(
        group_columns
    ):
        by_depth = group.set_index(
            "num_layers"
        )

        for odd_depth in (3, 5, 7):
            odd_accuracy = float(
                by_depth.loc[
                    odd_depth,
                    "best_test_acc",
                ]
            )

            lower_even = float(
                by_depth.loc[
                    odd_depth - 1,
                    "best_test_acc",
                ]
            )

            upper_even = float(
                by_depth.loc[
                    odd_depth + 1,
                    "best_test_acc",
                ]
            )

            residual = (
                odd_accuracy
                - (
                    lower_even
                    + upper_even
                )
                / 2.0
            ) * 100.0

            seed_rows.append(
                {
                    "target_homophily":
                        keys[0],
                    "graph_seed": keys[1],
                    "model": keys[2],
                    "odd_depth": odd_depth,
                    "odd_accuracy_pct":
                        odd_accuracy * 100.0,
                    "lower_even_accuracy_pct":
                        lower_even * 100.0,
                    "upper_even_accuracy_pct":
                        upper_even * 100.0,
                    "parity_residual_pp":
                        residual,
                }
            )

    seed_frame = pd.DataFrame(seed_rows)
    summary_rows = []

    for keys, group in seed_frame.groupby(
        [
            "target_homophily",
            "model",
            "odd_depth",
        ]
    ):
        mean, std, ci95 = confidence_interval(
            group["parity_residual_pp"]
        )

        t_pvalue, wilcoxon_pvalue = (
            test_against_zero(
                group["parity_residual_pp"]
            )
        )

        summary_rows.append(
            {
                "target_homophily":
                    keys[0],
                "model": keys[1],
                "odd_depth": keys[2],
                "n": len(group),
                "parity_residual_mean_pp":
                    mean,
                "parity_residual_std_pp":
                    std,
                "parity_residual_ci95_pp":
                    ci95,
                "negative_seeds": int(
                    (
                        group[
                            "parity_residual_pp"
                        ]
                        < 0
                    ).sum()
                ),
                "positive_seeds": int(
                    (
                        group[
                            "parity_residual_pp"
                        ]
                        > 0
                    ).sum()
                ),
                "paired_t_pvalue":
                    t_pvalue,
                "wilcoxon_pvalue":
                    wilcoxon_pvalue,
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary[
        "paired_t_pvalue_holm"
    ] = holm_adjust(
        summary["paired_t_pvalue"]
    )

    summary[
        "wilcoxon_pvalue_holm"
    ] = holm_adjust(
        summary["wilcoxon_pvalue"]
    )

    summary.to_csv(
        PARITY_DEPTH_PATH,
        index=False,
    )

    seed_frame.to_csv(
        RUN_DIR
        / "csbm_parity_parity_by_depth_seed.csv",
        index=False,
    )

    return seed_frame, summary


def print_selected_seed_results(
    paired: pd.DataFrame,
) -> None:
    selected_conditions = [
        (0.1, "GCN", 8),
        (0.1, "GraphSAGE", 8),
        (0.5, "GraphSAGE", 7),
        (0.5, "GraphSAGE", 8),
    ]

    frames = []

    for homophily, architecture, depth in (
        selected_conditions
    ):
        frame = paired[
            (
                np.isclose(
                    paired[
                        "target_homophily"
                    ],
                    homophily,
                )
            )
            & (
                paired["architecture"]
                == architecture
            )
            & (
                paired["num_layers"]
                == depth
            )
        ][
            [
                "target_homophily",
                "architecture",
                "num_layers",
                "graph_seed",
                "baseline_best_test_acc",
                "pairnorm_best_test_acc",
                "accuracy_delta_pp",
            ]
        ].copy()

        frame[
            "baseline_best_test_acc"
        ] *= 100.0
        frame[
            "pairnorm_best_test_acc"
        ] *= 100.0

        frames.append(frame)

    selected = pd.concat(
        frames,
        ignore_index=True,
    )

    print(
        "\n=== Selected seed-level effects ==="
    )
    print(
        selected.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.2f}",
        )
    )


def plot_parity_depth(
    summary: pd.DataFrame,
) -> None:
    for homophily, frame in summary.groupby(
        "target_homophily"
    ):
        plt.figure(figsize=(9, 6))

        for model, model_frame in frame.groupby(
            "model"
        ):
            model_frame = model_frame.sort_values(
                "odd_depth"
            )

            plt.errorbar(
                model_frame["odd_depth"],
                model_frame[
                    "parity_residual_mean_pp"
                ],
                yerr=model_frame[
                    "parity_residual_ci95_pp"
                ],
                marker="o",
                capsize=3,
                label=model,
            )

        plt.axhline(
            0.0,
            linewidth=1,
        )
        plt.xlabel("Odd number of layers")
        plt.ylabel(
            "Odd accuracy minus neighboring "
            "even-depth mean (pp)"
        )
        plt.title(
            "Depth-specific parity residual "
            f"(homophily={homophily:.1f})"
        )
        plt.xticks([3, 5, 7])
        plt.legend()
        plt.tight_layout()

        path = (
            PLOT_DIR
            / (
                "parity_residual_by_depth_"
                f"h{int(homophily * 10):02d}.png"
            )
        )

        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print("Saved:", path)


def main() -> None:
    corrected = correct_pairnorm_tests()

    runs = pd.read_csv(
        RUN_LEVEL_PATH
    )
    paired = pd.read_csv(
        PAIRED_PATH
    )

    _, parity_summary = (
        calculate_parity_by_depth(runs)
    )

    plot_parity_depth(
        parity_summary
    )

    print(
        "\n=== PairNorm effects after Holm correction ==="
    )

    columns = [
        "target_homophily",
        "architecture",
        "num_layers",
        "accuracy_delta_mean_pp",
        "accuracy_delta_ci95_pp",
        "positive_seeds",
        "negative_seeds",
        "paired_t_pvalue",
        "paired_t_pvalue_holm",
        "wilcoxon_pvalue_holm",
        "significant_t_holm_005",
    ]

    print(
        corrected[columns].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.5f}",
        )
    )

    print(
        "\n=== Depth-specific parity effects ==="
    )

    print(
        parity_summary[
            [
                "target_homophily",
                "model",
                "odd_depth",
                "parity_residual_mean_pp",
                "parity_residual_ci95_pp",
                "negative_seeds",
                "positive_seeds",
                "paired_t_pvalue_holm",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.5f}",
        )
    )

    print_selected_seed_results(
        paired
    )

    print("\nSaved:")
    print(CORRECTED_PATH)
    print(PARITY_DEPTH_PATH)


if __name__ == "__main__":
    main()
