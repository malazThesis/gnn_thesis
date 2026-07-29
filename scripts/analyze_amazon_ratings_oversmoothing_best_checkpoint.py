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

METRIC_FILE = (
    ROOT
    / "amazon_ratings_oversmoothing_penultimate_run_level.csv"
)

ACCURACY_FILE = (
    ROOT
    / "amazon_ratings_oversmoothing_accuracy_run_level.csv"
)

OUTPUT = (
    ROOT
    / "amazon_ratings_oversmoothing_best_checkpoint_effects.csv"
)

DETAIL_OUTPUT = (
    ROOT
    / "amazon_ratings_oversmoothing_best_checkpoint_detail.csv"
)

METRICS = [
    "mean_pairwise_cosine_distance",
    "mean_edge_cosine_distance",
    "normalized_dirichlet_energy",
    "effective_rank_ratio",
]


def exact_sign_flip(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(values.mean())

    statistics = [
        abs(
            np.mean(
                values * np.asarray(signs)
            )
        )
        for signs in itertools.product(
            [-1.0, 1.0],
            repeat=len(values),
        )
    ]

    return float(
        np.mean(
            np.asarray(statistics)
            >= observed - 1e-15
        )
    )


def statistics(values: pd.Series) -> dict:
    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna().to_numpy(dtype=float)

    n = len(values)
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    critical = float(
        stats.t.ppf(0.975, n - 1)
    )

    standard_error = std / math.sqrt(n)

    return {
        "n_pairs": n,
        "mean_difference": mean,
        "ci95_low":
            mean - critical * standard_error,
        "ci95_high":
            mean + critical * standard_error,
        "positive_pairs":
            int(np.sum(values > 0)),
        "negative_pairs":
            int(np.sum(values < 0)),
        "exact_sign_flip_pvalue":
            exact_sign_flip(values),
        "cohen_dz":
            mean / std
            if std > 0
            else math.nan,
    }


def holm(values: list[float]) -> list[float]:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    running = 0.0

    for position, index in enumerate(order):
        candidate = (
            len(values) - position
        ) * values[index]

        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)

    return adjusted.tolist()


metrics = pd.read_csv(METRIC_FILE)
accuracy = pd.read_csv(ACCURACY_FILE)

aligned_rows = []

for run in accuracy.itertuples():
    candidates = metrics[
        (metrics["model"] == run.model)
        & (
            metrics["num_layers"]
            == run.num_layers
        )
        & (metrics["seed"] == run.seed)
        & (
            metrics["split_idx"]
            == run.split_idx
        )
        & (
            metrics["epoch"]
            <= run.best_epoch
        )
    ].sort_values("epoch")

    if candidates.empty:
        raise RuntimeError(
            f"No metric checkpoint for {run}"
        )

    selected = candidates.iloc[-1].to_dict()

    selected["best_epoch"] = int(
        run.best_epoch
    )

    selected[
        "test_acc_at_best_val"
    ] = float(
        run.test_acc_at_best_val
    )

    selected[
        "metric_epoch_gap"
    ] = (
        int(run.best_epoch)
        - int(selected["epoch"])
    )

    aligned_rows.append(selected)

aligned = pd.DataFrame(aligned_rows)
aligned.to_csv(DETAIL_OUTPUT, index=False)

rows = []

for depth in [4, 8]:
    baseline = aligned[
        (aligned["model"] == "GraphSAGE")
        & (aligned["num_layers"] == depth)
    ]

    pairnorm = aligned[
        (
            aligned["model"]
            == "GraphSAGEPairNorm"
        )
        & (aligned["num_layers"] == depth)
    ]

    for metric in METRICS:
        paired = baseline[
            ["seed", "split_idx", metric]
        ].rename(
            columns={
                metric: "baseline",
            }
        ).merge(
            pairnorm[
                ["seed", "split_idx", metric]
            ].rename(
                columns={
                    metric: "pairnorm",
                }
            ),
            on=["seed", "split_idx"],
            validate="one_to_one",
        )

        paired["difference"] = (
            paired["pairnorm"]
            - paired["baseline"]
        )

        rows.append(
            {
                "num_layers": depth,
                "metric": metric,
                "baseline_mean":
                    paired["baseline"].mean(),
                "pairnorm_mean":
                    paired["pairnorm"].mean(),
                **statistics(
                    paired["difference"]
                ),
            }
        )

result = pd.DataFrame(rows)

result[
    "exact_sign_flip_holm_pvalue"
] = result.groupby("num_layers")[
    "exact_sign_flip_pvalue"
].transform(
    lambda series:
        holm(series.tolist())
)

result.to_csv(OUTPUT, index=False)

print("\n=== CHECKPOINT DISTRIBUTION ===")

print(
    aligned.groupby(
        [
            "model",
            "num_layers",
            "epoch",
        ]
    ).size()
)

print(
    "\nMean checkpoint gap:",
    aligned["metric_epoch_gap"].mean(),
)

print(
    "\n=== PAIRNORM EFFECTS NEAR "
    "BEST VALIDATION EPOCH ==="
)

print(
    result.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.10f}",
    )
)

print("\nSaved:", DETAIL_OUTPUT)
print("Saved:", OUTPUT)
