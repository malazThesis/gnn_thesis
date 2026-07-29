from __future__ import annotations

import itertools
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(
    "runs/amazon_ratings_matched_lr_oversmoothing_800ep"
)

PLOT_DIR = Path(
    "plots/amazon_ratings_matched_lr_oversmoothing_800ep"
)

PENULTIMATE_OUTPUT = (
    ROOT
    / "amazon_ratings_oversmoothing_penultimate_run_level.csv"
)

SUMMARY_OUTPUT = (
    ROOT
    / "amazon_ratings_oversmoothing_penultimate_summary.csv"
)

PAIRNORM_OUTPUT = (
    ROOT
    / "amazon_ratings_oversmoothing_pairnorm_effects_epoch800.csv"
)

DEPTH_OUTPUT = (
    ROOT
    / "amazon_ratings_oversmoothing_depth_interactions_epoch800.csv"
)

CORRELATION_OUTPUT = (
    ROOT
    / "amazon_ratings_oversmoothing_accuracy_correlations.csv"
)

ACCURACY_OUTPUT = (
    ROOT
    / "amazon_ratings_oversmoothing_accuracy_run_level.csv"
)

METRICS = [
    "mean_pairwise_cosine_distance",
    "mean_edge_cosine_distance",
    "normalized_dirichlet_energy",
    "effective_rank",
    "effective_rank_ratio",
    "mean_embedding_norm",
    "mean_feature_variance",
    "gradient_norm",
]

PRIMARY_METRICS = [
    "mean_pairwise_cosine_distance",
    "mean_edge_cosine_distance",
    "normalized_dirichlet_energy",
    "effective_rank_ratio",
]

MODELS = [
    "GraphSAGE",
    "GraphSAGEPairNorm",
]

DEPTHS = [
    4,
    8,
]

EPOCHS = [
    1,
    10,
    25,
    50,
    100,
    200,
    400,
    600,
    800,
]


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

    if len(values) == 0:
        return math.nan

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

    if n < 2:
        raise RuntimeError(
            "Insufficient paired observations."
        )

    mean = float(
        values.mean()
    )

    standard_deviation = float(
        values.std(ddof=1)
    )

    standard_error = (
        standard_deviation
        / math.sqrt(n)
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
        "std_difference":
            standard_deviation,
        "median_difference":
            float(
                np.median(values)
            ),
        "minimum_difference":
            float(
                values.min()
            ),
        "maximum_difference":
            float(
                values.max()
            ),
        "positive_pairs":
            int(
                np.sum(values > 0)
            ),
        "negative_pairs":
            int(
                np.sum(values < 0)
            ),
        "zero_pairs":
            int(
                np.sum(values == 0)
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
                mean / standard_deviation
                if standard_deviation > 0
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


def resolve_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str:
    for column in candidates:
        if column in frame.columns:
            return column

    raise RuntimeError(
        f"None of {candidates} found in "
        f"{frame.columns.tolist()}"
    )


def target_layer_name(
    model: str,
    depth: int,
) -> str:
    hidden_index = depth - 2

    if model == "GraphSAGE":
        return f"convs.{hidden_index}"

    if model == "GraphSAGEPairNorm":
        return f"pns.{hidden_index}"

    raise ValueError(model)


def load_oversmoothing() -> pd.DataFrame:
    files = sorted(
        ROOT.glob("*_oversmoothing.csv")
    )

    if len(files) != 40:
        raise RuntimeError(
            f"Expected 40 files, found {len(files)}"
        )

    selected_rows = []

    for path in files:
        frame = pd.read_csv(path)

        model = str(
            frame["model"].iloc[0]
        )

        depth = int(
            frame["num_layers"].iloc[0]
        )

        target = target_layer_name(
            model,
            depth,
        )

        selected = frame[
            frame["layer_name"] == target
        ].copy()

        if len(selected) != len(EPOCHS):
            raise RuntimeError(
                f"{path.name}: target={target}, "
                f"expected {len(EPOCHS)} rows, "
                f"found {len(selected)}"
            )

        if set(selected["epoch"]) != set(EPOCHS):
            raise RuntimeError(
                f"{path.name}: unexpected epochs "
                f"{sorted(selected['epoch'].tolist())}"
            )

        for metric in METRICS:
            selected[metric] = pd.to_numeric(
                selected[metric],
                errors="coerce",
            )

        selected[
            "selected_representation"
        ] = target

        selected[
            "source_file"
        ] = str(path)

        selected_rows.append(selected)

    result = pd.concat(
        selected_rows,
        ignore_index=True,
    )

    keys = [
        "model",
        "num_layers",
        "seed",
        "split_idx",
        "epoch",
    ]

    if result.duplicated(keys).any():
        raise RuntimeError(
            "Duplicate penultimate rows."
        )

    expected = (
        result.groupby(
            [
                "model",
                "num_layers",
                "epoch",
            ]
        )
        .size()
    )

    if not (expected == 10).all():
        raise RuntimeError(
            "Expected ten splits per group:\n"
            + expected.to_string()
        )

    return result.sort_values(keys)


def load_accuracy() -> pd.DataFrame:
    files = sorted(
        ROOT.glob("*_history.csv")
    )

    if len(files) != 40:
        raise RuntimeError(
            f"Expected 40 histories, "
            f"found {len(files)}"
        )

    rows = []

    for path in files:
        frame = pd.read_csv(path)

        epoch_column = resolve_column(
            frame,
            ["epoch"],
        )

        val_column = resolve_column(
            frame,
            [
                "val_acc",
                "validation_acc",
                "val_accuracy",
                "val",
            ],
        )

        test_column = resolve_column(
            frame,
            [
                "test_acc",
                "test_accuracy",
                "test",
            ],
        )

        valid = frame[
            np.isfinite(
                pd.to_numeric(
                    frame[val_column],
                    errors="coerce",
                )
            )
        ].copy()

        valid[val_column] = pd.to_numeric(
            valid[val_column],
            errors="coerce",
        )

        valid[test_column] = pd.to_numeric(
            valid[test_column],
            errors="coerce",
        )

        best = valid.loc[
            valid[val_column].idxmax()
        ]

        match = re.match(
            r"Amazon-Ratings_(?P<model>.+)_"
            r"L(?P<depth>\d+)_H(?P<hidden>\d+)_"
            r"seed(?P<seed>\d+)_split(?P<split>\d+)_"
            r"history\.csv$",
            path.name,
        )

        if match is None:
            raise RuntimeError(
                f"Could not parse {path.name}"
            )

        rows.append(
            {
                "model":
                    match.group("model"),
                "num_layers":
                    int(
                        match.group("depth")
                    ),
                "seed":
                    int(
                        match.group("seed")
                    ),
                "split_idx":
                    int(
                        match.group("split")
                    ),
                "best_epoch":
                    int(
                        best[epoch_column]
                    ),
                "best_val_acc":
                    float(
                        best[val_column]
                    ),
                "test_acc_at_best_val":
                    float(
                        best[test_column]
                    ),
            }
        )

    result = pd.DataFrame(rows)

    if len(result) != 40:
        raise RuntimeError(
            f"Expected 40 accuracy rows, "
            f"found {len(result)}"
        )

    return result


def summarize(
    penultimate: pd.DataFrame,
) -> pd.DataFrame:
    aggregations = {}

    for metric in METRICS:
        aggregations[
            f"mean_{metric}"
        ] = (
            metric,
            "mean",
        )

        aggregations[
            f"std_{metric}"
        ] = (
            metric,
            "std",
        )

    return (
        penultimate.groupby(
            [
                "model",
                "num_layers",
                "epoch",
                "selected_representation",
            ],
            as_index=False,
        )
        .agg(**aggregations)
        .sort_values(
            [
                "num_layers",
                "model",
                "epoch",
            ]
        )
    )


def pairnorm_effects(
    penultimate: pd.DataFrame,
) -> pd.DataFrame:
    final = penultimate[
        penultimate["epoch"] == 800
    ]

    rows = []

    for depth in DEPTHS:
        baseline = final[
            (
                final["model"]
                == "GraphSAGE"
            )
            & (
                final["num_layers"]
                == depth
            )
        ]

        pairnorm = final[
            (
                final["model"]
                == "GraphSAGEPairNorm"
            )
            & (
                final["num_layers"]
                == depth
            )
        ]

        for metric in METRICS:
            left = baseline[
                [
                    "seed",
                    "split_idx",
                    metric,
                ]
            ].rename(
                columns={
                    metric: "baseline_value",
                }
            )

            right = pairnorm[
                [
                    "seed",
                    "split_idx",
                    metric,
                ]
            ].rename(
                columns={
                    metric: "pairnorm_value",
                }
            )

            paired = left.merge(
                right,
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

            rows.append(
                {
                    "num_layers":
                        depth,
                    "metric":
                        metric,
                    "baseline_mean":
                        float(
                            paired[
                                "baseline_value"
                            ].mean()
                        ),
                    "pairnorm_mean":
                        float(
                            paired[
                                "pairnorm_value"
                            ].mean()
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
            result.groupby("num_layers")[
                column
            ]
            .transform(
                lambda series:
                    holm_adjust(
                        series.tolist()
                    )
            )
        )

    return result


def depth_interactions(
    penultimate: pd.DataFrame,
) -> pd.DataFrame:
    final = penultimate[
        penultimate["epoch"] == 800
    ]

    rows = []

    for metric in METRICS:
        pieces = {}

        for model in MODELS:
            for depth in DEPTHS:
                key = (
                    "baseline"
                    if model == "GraphSAGE"
                    else "pairnorm"
                )

                key = f"{key}_l{depth}"

                pieces[key] = final[
                    (
                        final["model"]
                        == model
                    )
                    & (
                        final["num_layers"]
                        == depth
                    )
                ][
                    [
                        "seed",
                        "split_idx",
                        metric,
                    ]
                ].rename(
                    columns={
                        metric: key,
                    }
                )

        merged = (
            pieces["baseline_l4"]
            .merge(
                pieces["baseline_l8"],
                on=[
                    "seed",
                    "split_idx",
                ],
                validate="one_to_one",
            )
            .merge(
                pieces["pairnorm_l4"],
                on=[
                    "seed",
                    "split_idx",
                ],
                validate="one_to_one",
            )
            .merge(
                pieces["pairnorm_l8"],
                on=[
                    "seed",
                    "split_idx",
                ],
                validate="one_to_one",
            )
        )

        merged["baseline_depth_change"] = (
            merged["baseline_l8"]
            - merged["baseline_l4"]
        )

        merged["pairnorm_depth_change"] = (
            merged["pairnorm_l8"]
            - merged["pairnorm_l4"]
        )

        merged["interaction"] = (
            merged["pairnorm_depth_change"]
            - merged["baseline_depth_change"]
        )

        rows.append(
            {
                "metric":
                    metric,
                "baseline_l4_mean":
                    float(
                        merged[
                            "baseline_l4"
                        ].mean()
                    ),
                "baseline_l8_mean":
                    float(
                        merged[
                            "baseline_l8"
                        ].mean()
                    ),
                "pairnorm_l4_mean":
                    float(
                        merged[
                            "pairnorm_l4"
                        ].mean()
                    ),
                "pairnorm_l8_mean":
                    float(
                        merged[
                            "pairnorm_l8"
                        ].mean()
                    ),
                "baseline_mean_depth_change":
                    float(
                        merged[
                            "baseline_depth_change"
                        ].mean()
                    ),
                "pairnorm_mean_depth_change":
                    float(
                        merged[
                            "pairnorm_depth_change"
                        ].mean()
                    ),
                **paired_statistics(
                    merged["interaction"]
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
        ] = holm_adjust(
            result[column].tolist()
        )

    return result


def accuracy_correlations(
    penultimate: pd.DataFrame,
    accuracy: pd.DataFrame,
) -> pd.DataFrame:
    final = penultimate[
        penultimate["epoch"] == 800
    ]

    rows = []

    for depth in DEPTHS:
        baseline_accuracy = accuracy[
            (
                accuracy["model"]
                == "GraphSAGE"
            )
            & (
                accuracy["num_layers"]
                == depth
            )
        ][
            [
                "seed",
                "split_idx",
                "test_acc_at_best_val",
            ]
        ].rename(
            columns={
                "test_acc_at_best_val":
                    "baseline_test",
            }
        )

        pairnorm_accuracy = accuracy[
            (
                accuracy["model"]
                == "GraphSAGEPairNorm"
            )
            & (
                accuracy["num_layers"]
                == depth
            )
        ][
            [
                "seed",
                "split_idx",
                "test_acc_at_best_val",
            ]
        ].rename(
            columns={
                "test_acc_at_best_val":
                    "pairnorm_test",
            }
        )

        base = baseline_accuracy.merge(
            pairnorm_accuracy,
            on=[
                "seed",
                "split_idx",
            ],
            validate="one_to_one",
        )

        base["accuracy_gain"] = (
            base["pairnorm_test"]
            - base["baseline_test"]
        )

        for metric in PRIMARY_METRICS:
            baseline_metric = final[
                (
                    final["model"]
                    == "GraphSAGE"
                )
                & (
                    final["num_layers"]
                    == depth
                )
            ][
                [
                    "seed",
                    "split_idx",
                    metric,
                ]
            ].rename(
                columns={
                    metric: "baseline_metric",
                }
            )

            pairnorm_metric = final[
                (
                    final["model"]
                    == "GraphSAGEPairNorm"
                )
                & (
                    final["num_layers"]
                    == depth
                )
            ][
                [
                    "seed",
                    "split_idx",
                    metric,
                ]
            ].rename(
                columns={
                    metric: "pairnorm_metric",
                }
            )

            merged = (
                base
                .merge(
                    baseline_metric,
                    on=[
                        "seed",
                        "split_idx",
                    ],
                    validate="one_to_one",
                )
                .merge(
                    pairnorm_metric,
                    on=[
                        "seed",
                        "split_idx",
                    ],
                    validate="one_to_one",
                )
            )

            merged["metric_gain"] = (
                merged["pairnorm_metric"]
                - merged["baseline_metric"]
            )

            correlation = stats.spearmanr(
                merged["metric_gain"],
                merged["accuracy_gain"],
            )

            rows.append(
                {
                    "num_layers":
                        depth,
                    "metric":
                        metric,
                    "n":
                        len(merged),
                    "spearman_rho":
                        float(
                            correlation.statistic
                        ),
                    "spearman_pvalue":
                        float(
                            correlation.pvalue
                        ),
                }
            )

    result = pd.DataFrame(rows)

    result[
        "spearman_holm_pvalue"
    ] = (
        result.groupby("num_layers")[
            "spearman_pvalue"
        ]
        .transform(
            lambda series:
                holm_adjust(
                    series.tolist()
                )
        )
    )

    return result


def make_trajectory_plots(
    summary: pd.DataFrame,
) -> None:
    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels = {
        (
            "GraphSAGE",
            4,
        ): "GraphSAGE L4",

        (
            "GraphSAGEPairNorm",
            4,
        ): "PairNorm L4",

        (
            "GraphSAGE",
            8,
        ): "GraphSAGE L8",

        (
            "GraphSAGEPairNorm",
            8,
        ): "PairNorm L8",
    }

    for metric in PRIMARY_METRICS:
        mean_column = f"mean_{metric}"

        plt.figure(
            figsize=(9, 6)
        )

        for (
            model,
            depth,
        ), label in labels.items():
            rows = summary[
                (
                    summary["model"]
                    == model
                )
                & (
                    summary["num_layers"]
                    == depth
                )
            ].sort_values("epoch")

            plt.plot(
                rows["epoch"],
                rows[mean_column],
                marker="o",
                label=label,
            )

        plt.xlabel("Epoch")
        plt.ylabel(
            metric.replace("_", " ")
        )

        plt.title(
            "Amazon-Ratings penultimate "
            "hidden representation"
        )

        plt.legend()
        plt.tight_layout()

        path = (
            PLOT_DIR
            / f"{metric}_trajectory.png"
        )

        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()
        print("Saved:", path)


def main() -> None:
    penultimate = load_oversmoothing()
    accuracy = load_accuracy()

    summary = summarize(
        penultimate
    )

    effects = pairnorm_effects(
        penultimate
    )

    interactions = depth_interactions(
        penultimate
    )

    correlations = accuracy_correlations(
        penultimate,
        accuracy,
    )

    penultimate.to_csv(
        PENULTIMATE_OUTPUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    effects.to_csv(
        PAIRNORM_OUTPUT,
        index=False,
    )

    interactions.to_csv(
        DEPTH_OUTPUT,
        index=False,
    )

    correlations.to_csv(
        CORRELATION_OUTPUT,
        index=False,
    )

    accuracy.to_csv(
        ACCURACY_OUTPUT,
        index=False,
    )

    make_trajectory_plots(
        summary
    )

    print(
        "\n=== PENULTIMATE REPRESENTATIONS "
        "AT EPOCH 800 ==="
    )

    final_summary = summary[
        summary["epoch"] == 800
    ]

    columns = [
        "model",
        "num_layers",
        "selected_representation",
    ]

    columns += [
        f"mean_{metric}"
        for metric in METRICS
    ]

    print(
        final_summary[
            columns
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.10f}",
        )
    )

    print(
        "\n=== PAIRNORM MINUS BASELINE "
        "AT EPOCH 800 ==="
    )

    print(
        effects[
            [
                "num_layers",
                "metric",
                "baseline_mean",
                "pairnorm_mean",
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
        "\n=== DEPTH INTERACTIONS "
        "AT EPOCH 800 ==="
    )

    print(
        interactions[
            [
                "metric",
                "baseline_mean_depth_change",
                "pairnorm_mean_depth_change",
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
        "\n=== EXPLORATORY ACCURACY "
        "CORRELATIONS ==="
    )

    print(
        correlations.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.10f}",
        )
    )

    print("\nSaved:", PENULTIMATE_OUTPUT)
    print("Saved:", SUMMARY_OUTPUT)
    print("Saved:", PAIRNORM_OUTPUT)
    print("Saved:", DEPTH_OUTPUT)
    print("Saved:", CORRELATION_OUTPUT)
    print("Saved:", ACCURACY_OUTPUT)


if __name__ == "__main__":
    main()
