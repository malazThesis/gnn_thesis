from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(
    "runs/amazon_ratings_matched_lr_oversmoothing_800ep"
)

INPUT = (
    ROOT
    / "amazon_ratings_oversmoothing_penultimate_run_level.csv"
)

OUTPUT = (
    ROOT
    / "amazon_ratings_oversmoothing_fixed_epoch_effects.csv"
)

EPOCHS = [
    400,
    600,
    800,
]

DEPTHS = [
    4,
    8,
]

METRICS = [
    "mean_pairwise_cosine_distance",
    "mean_edge_cosine_distance",
    "normalized_dirichlet_energy",
    "effective_rank_ratio",
]


def exact_sign_flip_pvalue(
    values: np.ndarray,
) -> float:
    values = np.asarray(
        values,
        dtype=float,
    )

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
) -> dict:
    values = pd.to_numeric(
        differences,
        errors="coerce",
    ).dropna().to_numpy(dtype=float)

    n = len(values)
    mean = float(values.mean())
    std = float(values.std(ddof=1))

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
        "mean_difference":
            mean,
        "ci95_low":
            mean
            - critical * standard_error,
        "ci95_high":
            mean
            + critical * standard_error,
        "positive_pairs":
            int(np.sum(values > 0)),
        "negative_pairs":
            int(np.sum(values < 0)),
        "zero_pairs":
            int(np.sum(values == 0)),
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

    valid = np.where(
        np.isfinite(pvalues)
    )[0]

    ordered = valid[
        np.argsort(
            pvalues[valid]
        )
    ]

    running = 0.0
    total = len(ordered)

    for position, index in enumerate(
        ordered
    ):
        candidate = (
            total - position
        ) * pvalues[index]

        running = max(
            running,
            candidate,
        )

        adjusted[index] = min(
            running,
            1.0,
        )

    return adjusted.tolist()


frame = pd.read_csv(INPUT)

selected = frame[
    frame["epoch"].isin(EPOCHS)
].copy()

rows = []

for epoch in EPOCHS:
    for depth in DEPTHS:
        baseline = selected[
            (
                selected["epoch"]
                == epoch
            )
            & (
                selected["num_layers"]
                == depth
            )
            & (
                selected["model"]
                == "GraphSAGE"
            )
        ]

        pairnorm = selected[
            (
                selected["epoch"]
                == epoch
            )
            & (
                selected["num_layers"]
                == depth
            )
            & (
                selected["model"]
                == "GraphSAGEPairNorm"
            )
        ]

        if len(baseline) != 10:
            raise RuntimeError(
                f"Baseline epoch={epoch}, "
                f"L{depth}: {len(baseline)} rows"
            )

        if len(pairnorm) != 10:
            raise RuntimeError(
                f"PairNorm epoch={epoch}, "
                f"L{depth}: {len(pairnorm)} rows"
            )

        for metric in METRICS:
            paired = baseline[
                [
                    "seed",
                    "split_idx",
                    metric,
                ]
            ].rename(
                columns={
                    metric:
                        "baseline_value",
                }
            ).merge(
                pairnorm[
                    [
                        "seed",
                        "split_idx",
                        metric,
                    ]
                ].rename(
                    columns={
                        metric:
                            "pairnorm_value",
                    }
                ),
                on=[
                    "seed",
                    "split_idx",
                ],
                validate="one_to_one",
            )

            paired["difference"] = (
                paired["pairnorm_value"]
                - paired["baseline_value"]
            )

            baseline_mean = float(
                paired[
                    "baseline_value"
                ].mean()
            )

            pairnorm_mean = float(
                paired[
                    "pairnorm_value"
                ].mean()
            )

            rows.append(
                {
                    "epoch":
                        epoch,
                    "num_layers":
                        depth,
                    "metric":
                        metric,
                    "baseline_mean":
                        baseline_mean,
                    "pairnorm_mean":
                        pairnorm_mean,
                    "pairnorm_to_baseline_ratio":
                        (
                            pairnorm_mean
                            / baseline_mean
                            if baseline_mean != 0
                            else math.nan
                        ),
                    **paired_statistics(
                        paired["difference"]
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
    ] = (
        result.groupby(
            [
                "epoch",
                "num_layers",
            ]
        )[column]
        .transform(
            lambda series:
                holm_adjust(
                    series.tolist()
                )
        )
    )

result.to_csv(
    OUTPUT,
    index=False,
)

print(
    "\n=== FIXED-EPOCH PAIRNORM EFFECTS ==="
)

print(
    result[
        [
            "epoch",
            "num_layers",
            "metric",
            "baseline_mean",
            "pairnorm_mean",
            "pairnorm_to_baseline_ratio",
            "mean_difference",
            "ci95_low",
            "ci95_high",
            "positive_pairs",
            "negative_pairs",
            "exact_sign_flip_holm_pvalue",
            "cohen_dz",
        ]
    ].to_string(
        index=False,
        float_format=lambda value:
            f"{value:.10f}",
    )
)

print(
    "\n=== SIGN CONSISTENCY ==="
)

print(
    result.groupby(
        [
            "num_layers",
            "metric",
        ]
    ).agg(
        epochs_tested=(
            "epoch",
            "size",
        ),
        epochs_all_positive=(
            "positive_pairs",
            lambda values:
                int(
                    np.sum(
                        np.asarray(values)
                        == 10
                    )
                ),
        ),
        maximum_holm_pvalue=(
            "exact_sign_flip_holm_pvalue",
            "max",
        ),
    )
)

print("\nSaved:", OUTPUT)
