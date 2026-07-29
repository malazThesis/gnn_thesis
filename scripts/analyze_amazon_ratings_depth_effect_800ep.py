from __future__ import annotations

import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(
    "runs/amazon_ratings_fair_lr_800ep"
)

PLOT_DIR = Path(
    "plots/amazon_ratings_fair_lr_800ep"
)

RUN_FILE = (
    ROOT
    / "amazon_ratings_fair_lr_800ep_run_level.csv"
)

SELECTED_FILE = (
    ROOT
    / "amazon_ratings_fair_lr_800ep_selected_configs.csv"
)

DETAIL_OUTPUT = (
    ROOT
    / "amazon_ratings_depth_effect_800ep_detail.csv"
)

SUMMARY_OUTPUT = (
    ROOT
    / "amazon_ratings_depth_effect_800ep_summary.csv"
)


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

    observed = abs(values.mean())
    extreme = 0
    total = 0

    for signs in itertools.product(
        [-1.0, 1.0],
        repeat=len(values),
    ):
        statistic = abs(
            np.mean(
                values
                * np.asarray(signs)
            )
        )

        if statistic >= observed - 1e-15:
            extreme += 1

        total += 1

    return extreme / total


def paired_statistics(
    values: pd.Series,
) -> dict[str, float | int]:
    differences = pd.to_numeric(
        values,
        errors="coerce",
    ).to_numpy(dtype=float)

    differences = differences[
        np.isfinite(differences)
    ]

    n = len(differences)
    mean = float(differences.mean())
    std = float(differences.std(ddof=1))
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
                differences,
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
            float(
                np.median(differences)
            ),
        "minimum_difference":
            float(
                differences.min()
            ),
        "maximum_difference":
            float(
                differences.max()
            ),
        "positive_pairs":
            int(
                np.sum(differences > 0)
            ),
        "negative_pairs":
            int(
                np.sum(differences < 0)
            ),
        "zero_pairs":
            int(
                np.sum(differences == 0)
            ),
        "ci95_low":
            mean
            - critical * standard_error,
        "ci95_high":
            mean
            + critical * standard_error,
        "paired_t_pvalue":
            float(
                stats.ttest_1samp(
                    differences,
                    popmean=0.0,
                ).pvalue
            ),
        "wilcoxon_pvalue":
            wilcoxon_pvalue,
        "exact_sign_flip_pvalue":
            exact_sign_flip_pvalue(
                differences
            ),
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
    runs = pd.read_csv(
        RUN_FILE
    )

    selected = pd.read_csv(
        SELECTED_FILE
    )

    selected_rates = {
        (
            row.model,
            int(row.num_layers),
        ): float(row.learning_rate)
        for row in selected.itertuples()
    }

    selected_runs = []

    for (
        model,
        depth,
    ), learning_rate in selected_rates.items():
        rows = runs[
            (
                runs["model"]
                == model
            )
            & (
                runs["num_layers"]
                == depth
            )
            & np.isclose(
                runs["learning_rate"],
                learning_rate,
            )
        ].copy()

        if len(rows) != 10:
            raise RuntimeError(
                f"{model} L{depth}: "
                f"expected 10 rows, found {len(rows)}"
            )

        selected_runs.append(rows)

    selected_runs = pd.concat(
        selected_runs,
        ignore_index=True,
    )

    pivot = selected_runs.pivot(
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

    required = [
        ("GraphSAGE", 4),
        ("GraphSAGE", 8),
        ("GraphSAGEPairNorm", 4),
        ("GraphSAGEPairNorm", 8),
    ]

    missing = [
        column
        for column in required
        if column not in pivot.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing pivot columns: {missing}"
        )

    index_frame = pivot.reset_index()[
        [
            "seed",
            "split_idx",
        ]
    ]

    detail = index_frame.copy()

    # Explizite Auswahl über die MultiIndex-Tupel.
    # Keine positionsabhängige Umbenennung verwenden.
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

    detail[
        "graphsage_depth_change"
    ] = (
        detail["graphsage_l8"]
        - detail["graphsage_l4"]
    )

    detail[
        "pairnorm_depth_change"
    ] = (
        detail["pairnorm_l8"]
        - detail["pairnorm_l4"]
    )

    # Positiv bedeutet:
    # PairNorm reduziert den L4→L8-Leistungsabfall.
    detail[
        "depth_interaction"
    ] = (
        detail["pairnorm_depth_change"]
        - detail["graphsage_depth_change"]
    )

    detail[
        "graphsage_depth_loss"
    ] = -detail[
        "graphsage_depth_change"
    ]

    detail[
        "pairnorm_depth_loss"
    ] = -detail[
        "pairnorm_depth_change"
    ]

    detail[
        "relative_depth_loss_reduction"
    ] = (
        detail[
            "graphsage_depth_loss"
        ]
        - detail[
            "pairnorm_depth_loss"
        ]
    ) / detail[
        "graphsage_depth_loss"
    ].replace(0, np.nan)

    tests = [
        {
            "comparison":
                "GraphSAGE L8 minus L4",
            "interpretation":
                "Negative values indicate depth degradation.",
            **paired_statistics(
                detail[
                    "graphsage_depth_change"
                ]
            ),
        },
        {
            "comparison":
                "PairNorm L8 minus L4",
            "interpretation":
                "Negative values indicate depth degradation.",
            **paired_statistics(
                detail[
                    "pairnorm_depth_change"
                ]
            ),
        },
        {
            "comparison":
                "PairNorm depth interaction",
            "interpretation":
                (
                    "Positive values indicate that PairNorm "
                    "reduces the L4-to-L8 depth penalty."
                ),
            **paired_statistics(
                detail[
                    "depth_interaction"
                ]
            ),
        },
    ]

    summary = pd.DataFrame(tests)

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

    baseline_mean_loss = float(
        detail[
            "graphsage_depth_loss"
        ].mean()
    )

    pairnorm_mean_loss = float(
        detail[
            "pairnorm_depth_loss"
        ].mean()
    )

    interaction_mean = float(
        detail[
            "depth_interaction"
        ].mean()
    )

    relative_reduction = (
        interaction_mean
        / baseline_mean_loss
        if baseline_mean_loss != 0
        else math.nan
    )

    summary[
        "baseline_mean_depth_loss"
    ] = baseline_mean_loss

    summary[
        "pairnorm_mean_depth_loss"
    ] = pairnorm_mean_loss

    summary[
        "absolute_depth_loss_reduction"
    ] = interaction_mean

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

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_frame = pd.DataFrame(
        {
            "split":
                detail["split_idx"],
            "GraphSAGE L4":
                detail["graphsage_l4"],
            "GraphSAGE L8":
                detail["graphsage_l8"],
            "PairNorm L4":
                detail["pairnorm_l4"],
            "PairNorm L8":
                detail["pairnorm_l8"],
        }
    )

    plt.figure(figsize=(9, 6))

    positions = {
        "GraphSAGE L4": 0,
        "GraphSAGE L8": 1,
        "PairNorm L4": 3,
        "PairNorm L8": 4,
    }

    for row in plot_frame.itertuples():
        plt.plot(
            [
                positions["GraphSAGE L4"],
                positions["GraphSAGE L8"],
            ],
            [
                row._2,
                row._3,
            ],
            marker="o",
            alpha=0.45,
        )

        plt.plot(
            [
                positions["PairNorm L4"],
                positions["PairNorm L8"],
            ],
            [
                row._4,
                row._5,
            ],
            marker="o",
            alpha=0.45,
        )

    plt.xticks(
        list(positions.values()),
        list(positions.keys()),
        rotation=15,
    )

    plt.ylabel(
        "Test accuracy at best validation epoch"
    )

    plt.title(
        "Amazon-Ratings paired depth effects"
    )

    plt.tight_layout()

    plot_path = (
        PLOT_DIR
        / "paired_depth_effect_800ep.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\n=== SPLIT-LEVEL DEPTH EFFECTS ===")

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

    print("\n=== STATISTICAL TESTS ===")

    print(
        summary.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.10f}",
        )
    )

    print(
        "\nMean GraphSAGE depth loss: "
        f"{100 * baseline_mean_loss:.4f} pp"
    )

    print(
        "Mean PairNorm depth loss: "
        f"{100 * pairnorm_mean_loss:.4f} pp"
    )

    print(
        "Absolute reduction: "
        f"{100 * interaction_mean:.4f} pp"
    )

    print(
        "Relative reduction: "
        f"{100 * relative_reduction:.2f}%"
    )

    print("\nSaved:", DETAIL_OUTPUT)
    print("Saved:", SUMMARY_OUTPUT)
    print("Saved:", plot_path)


if __name__ == "__main__":
    main()
