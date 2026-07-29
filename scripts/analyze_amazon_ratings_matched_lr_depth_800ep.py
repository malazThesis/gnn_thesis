from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(
    "runs/amazon_ratings_fair_lr_800ep"
)

RUN_FILE = (
    ROOT
    / "amazon_ratings_fair_lr_800ep_run_level.csv"
)

DETAIL_OUTPUT = (
    ROOT
    / "amazon_ratings_matched_lr_depth_800ep_detail.csv"
)

SUMMARY_OUTPUT = (
    ROOT
    / "amazon_ratings_matched_lr_depth_800ep_summary.csv"
)

LEARNING_RATE = 0.003


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
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    standard_error = std / math.sqrt(n)

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
        "mean_difference":
            mean,
        "mean_difference_percentage_points":
            100.0 * mean,
        "std_difference":
            std,
        "median_difference":
            float(np.median(values)),
        "minimum_difference":
            float(values.min()),
        "maximum_difference":
            float(values.max()),
        "positive_pairs":
            int(np.sum(values > 0)),
        "negative_pairs":
            int(np.sum(values < 0)),
        "zero_pairs":
            int(np.sum(values == 0)),
        "ci95_low":
            mean - critical * standard_error,
        "ci95_high":
            mean + critical * standard_error,
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
            exact_sign_flip_pvalue(values),
        "cohen_dz":
            (
                mean / std
                if std > 0
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

    ordered = valid[
        np.argsort(values[valid])
    ]

    running = 0.0
    total = len(ordered)

    for position, index in enumerate(
        ordered
    ):
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


def main() -> None:
    runs = pd.read_csv(RUN_FILE)

    selected = runs[
        np.isclose(
            runs["learning_rate"],
            LEARNING_RATE,
        )
        & runs["model"].isin(
            [
                "GraphSAGE",
                "GraphSAGEPairNorm",
            ]
        )
        & runs["num_layers"].isin(
            [4, 8]
        )
    ].copy()

    if len(selected) != 40:
        raise RuntimeError(
            f"Expected 40 rows at lr={LEARNING_RATE}, "
            f"found {len(selected)}"
        )

    counts = (
        selected.groupby(
            [
                "model",
                "num_layers",
            ]
        )
        .size()
    )

    if not (counts == 10).all():
        raise RuntimeError(
            "Expected ten splits per combination:\n"
            + counts.to_string()
        )

    pivot = selected.pivot(
        index=[
            "seed",
            "split_idx",
        ],
        columns=[
            "model",
            "num_layers",
        ],
        values="test_acc_at_best_val",
    )

    detail = pivot.reset_index()[
        [
            "seed",
            "split_idx",
        ]
    ].copy()

    detail["graphsage_l4"] = (
        pivot[
            ("GraphSAGE", 4)
        ].to_numpy()
    )

    detail["graphsage_l8"] = (
        pivot[
            ("GraphSAGE", 8)
        ].to_numpy()
    )

    detail["pairnorm_l4"] = (
        pivot[
            ("GraphSAGEPairNorm", 4)
        ].to_numpy()
    )

    detail["pairnorm_l8"] = (
        pivot[
            ("GraphSAGEPairNorm", 8)
        ].to_numpy()
    )

    detail["graphsage_depth_change"] = (
        detail["graphsage_l8"]
        - detail["graphsage_l4"]
    )

    detail["pairnorm_depth_change"] = (
        detail["pairnorm_l8"]
        - detail["pairnorm_l4"]
    )

    detail["depth_interaction"] = (
        detail["pairnorm_depth_change"]
        - detail["graphsage_depth_change"]
    )

    detail["graphsage_depth_loss"] = (
        -detail["graphsage_depth_change"]
    )

    detail["pairnorm_depth_loss"] = (
        -detail["pairnorm_depth_change"]
    )

    comparisons = [
        (
            "GraphSAGE L8 minus L4",
            detail["graphsage_depth_change"],
        ),
        (
            "PairNorm L8 minus L4",
            detail["pairnorm_depth_change"],
        ),
        (
            "PairNorm depth interaction",
            detail["depth_interaction"],
        ),
    ]

    rows = []

    for name, differences in comparisons:
        rows.append(
            {
                "comparison": name,
                "learning_rate":
                    LEARNING_RATE,
                **paired_statistics(
                    differences
                ),
            }
        )

    summary = pd.DataFrame(rows)

    for column in [
        "paired_t_pvalue",
        "wilcoxon_pvalue",
        "exact_sign_flip_pvalue",
    ]:
        summary[
            column.replace(
                "_pvalue",
                "_holm_pvalue",
            )
        ] = holm_adjust(
            summary[column].tolist()
        )

    baseline_loss = float(
        detail[
            "graphsage_depth_loss"
        ].mean()
    )

    pairnorm_loss = float(
        detail[
            "pairnorm_depth_loss"
        ].mean()
    )

    reduction = float(
        detail[
            "depth_interaction"
        ].mean()
    )

    relative_reduction = (
        reduction / baseline_loss
        if baseline_loss != 0
        else math.nan
    )

    summary[
        "baseline_mean_depth_loss"
    ] = baseline_loss

    summary[
        "pairnorm_mean_depth_loss"
    ] = pairnorm_loss

    summary[
        "absolute_depth_loss_reduction"
    ] = reduction

    summary[
        "relative_depth_loss_reduction"
    ] = relative_reduction

    detail.to_csv(
        DETAIL_OUTPUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print(
        "\n=== MATCHED-LR SPLIT EFFECTS ==="
    )

    print(
        detail[
            [
                "split_idx",
                "graphsage_depth_change",
                "pairnorm_depth_change",
                "depth_interaction",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.10f}",
        )
    )

    print(
        "\n=== MATCHED-LR TESTS ==="
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.10f}",
        )
    )

    print(
        "\nGraphSAGE depth loss: "
        f"{100 * baseline_loss:.4f} pp"
    )

    print(
        "PairNorm depth loss: "
        f"{100 * pairnorm_loss:.4f} pp"
    )

    print(
        "Absolute reduction: "
        f"{100 * reduction:.4f} pp"
    )

    print(
        "Relative reduction: "
        f"{100 * relative_reduction:.2f}%"
    )

    print("\nSaved:", DETAIL_OUTPUT)
    print("Saved:", SUMMARY_OUTPUT)


if __name__ == "__main__":
    main()
