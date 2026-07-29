from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


RUN_DIR = Path(
    "runs/csbm_lowlabel_h01_parity"
)
PLOT_DIR = Path(
    "plots/csbm_lowlabel_h01_parity"
)

EXPECTED_RUNS = 120
EXPECTED_PAIRS = 60
DEPTHS = [5, 6, 7, 8, 9, 10]


def module_number(name: object) -> int:
    match = re.search(
        r"\.(\d+)(?:#\d+)?$",
        str(name),
    )
    return int(match.group(1)) if match else -1


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

    running_maximum = 0.0
    count = len(order)

    for rank, index in enumerate(order):
        corrected = (
            count - rank
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


def summarize(
    values: pd.Series,
) -> dict[str, float]:
    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    count = len(values)

    if count == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "std": math.nan,
            "ci95": math.nan,
        }

    mean = float(values.mean())

    if count == 1:
        return {
            "n": 1,
            "mean": mean,
            "std": 0.0,
            "ci95": 0.0,
        }

    std = float(values.std(ddof=1))

    critical = float(
        stats.t.ppf(
            0.975,
            df=count - 1,
        )
    )

    ci95 = (
        critical
        * std
        / math.sqrt(count)
    )

    return {
        "n": count,
        "mean": mean,
        "std": std,
        "ci95": ci95,
    }


def test_against_zero(
    values: pd.Series,
) -> tuple[float, float]:
    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    array = values.to_numpy(dtype=float)

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


def select_hidden(
    frame: pd.DataFrame,
    *,
    model: str,
    hidden_channels: int,
) -> pd.Series:
    frame = frame[
        frame["layer_name"].notna()
    ].copy()

    if model.endswith("PairNorm"):
        candidates = frame[
            frame["layer_name"]
            .astype(str)
            .str.startswith("pns.")
        ].copy()
    else:
        candidates = frame[
            (
                frame["layer_name"]
                .astype(str)
                .str.startswith("convs.")
            )
            & (
                frame["embedding_dim"]
                == hidden_channels
            )
        ].copy()

    if candidates.empty:
        raise RuntimeError(
            f"No hidden representation for {model}"
        )

    candidates["module_number"] = (
        candidates["layer_name"]
        .map(module_number)
    )

    return candidates.sort_values(
        [
            "module_number",
            "layer_index",
        ]
    ).iloc[-1]


def load_runs() -> pd.DataFrame:
    summary_files = sorted(
        RUN_DIR.glob("*_summary.json")
    )

    if len(summary_files) != EXPECTED_RUNS:
        raise RuntimeError(
            f"Expected {EXPECTED_RUNS} summaries, "
            f"found {len(summary_files)}"
        )

    rows = []

    for summary_path in summary_files:
        with summary_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            summary = json.load(handle)

        metrics_path = summary_path.with_name(
            summary_path.name.replace(
                "_summary.json",
                "_oversmoothing.csv",
            )
        )

        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)

        metrics = pd.read_csv(metrics_path)
        metrics = metrics[
            metrics["layer_name"].notna()
        ].copy()

        recorded_epochs = sorted(
            int(value)
            for value in metrics["epoch"].unique()
        )

        best_epoch = int(
            summary["best_epoch"]
        )

        nearest_best_epoch = min(
            recorded_epochs,
            key=lambda epoch: abs(
                epoch - best_epoch
            ),
        )

        final_epoch = max(recorded_epochs)

        model = str(summary["model"])
        hidden_channels = int(
            summary["hidden_channels"]
        )

        row = {
            "dataset": summary["dataset"],
            "model": model,
            "uses_pairnorm": model.endswith(
                "PairNorm"
            ),
            "num_layers": int(
                summary["num_layers"]
            ),
            "hidden_channels":
                hidden_channels,
            "seed": int(summary["seed"]),
            "graph_seed": int(
                summary["graph_seed"]
            ),
            "split_idx": int(
                summary["split_idx"]
            ),
            "target_homophily": float(
                summary["target_homophily"]
            ),
            "realized_homophily": float(
                summary["realized_homophily"]
            ),
            "feature_signal": float(
                summary["feature_signal"]
            ),
            "best_epoch": best_epoch,
            "nearest_best_metric_epoch":
                nearest_best_epoch,
            "best_train_acc": float(
                summary["best_train_acc"]
            ),
            "best_val_acc": float(
                summary["best_val_acc"]
            ),
            "best_test_acc": float(
                summary[
                    "best_test_acc_at_best_val"
                ]
            ),
        }

        for checkpoint, epoch in (
            ("best", nearest_best_epoch),
            ("final", final_epoch),
        ):
            epoch_frame = metrics[
                metrics["epoch"] == epoch
            ]

            hidden = select_hidden(
                epoch_frame,
                model=model,
                hidden_channels=hidden_channels,
            )

            row[
                f"{checkpoint}_pairwise_distance"
            ] = float(
                hidden[
                    "mean_pairwise_cosine_distance"
                ]
            )
            row[
                f"{checkpoint}_edge_distance"
            ] = float(
                hidden[
                    "mean_edge_cosine_distance"
                ]
            )
            row[
                f"{checkpoint}_rank_ratio"
            ] = float(
                hidden[
                    "effective_rank_ratio"
                ]
            )
            row[
                f"{checkpoint}_normalized_dirichlet"
            ] = float(
                hidden[
                    "normalized_dirichlet_energy"
                ]
            )

        rows.append(row)

    runs = pd.DataFrame(rows)

    keys = [
        "feature_signal",
        "graph_seed",
        "model",
        "num_layers",
    ]

    duplicates = runs.duplicated(
        keys,
        keep=False,
    )

    if duplicates.any():
        raise RuntimeError(
            "Duplicate configurations:\n"
            + runs.loc[
                duplicates,
                keys,
            ].to_string(index=False)
        )

    found_depths = sorted(
        runs["num_layers"].unique().tolist()
    )

    if found_depths != DEPTHS:
        raise RuntimeError(
            f"Expected depths {DEPTHS}, "
            f"found {found_depths}"
        )

    return runs.sort_values(
        keys
    ).reset_index(drop=True)


def build_pairs(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "target_homophily",
        "realized_homophily",
        "feature_signal",
        "graph_seed",
        "seed",
        "split_idx",
        "num_layers",
        "hidden_channels",
    ]

    values = [
        "best_epoch",
        "best_train_acc",
        "best_val_acc",
        "best_test_acc",
        "best_pairwise_distance",
        "best_edge_distance",
        "best_rank_ratio",
        "best_normalized_dirichlet",
        "final_pairwise_distance",
        "final_edge_distance",
        "final_rank_ratio",
        "final_normalized_dirichlet",
    ]

    baseline = runs[
        ~runs["uses_pairnorm"]
    ][keys + values].copy()

    pairnorm = runs[
        runs["uses_pairnorm"]
    ][keys + values].copy()

    baseline = baseline.rename(
        columns={
            column: f"baseline_{column}"
            for column in values
        }
    )

    pairnorm = pairnorm.rename(
        columns={
            column: f"pairnorm_{column}"
            for column in values
        }
    )

    paired = baseline.merge(
        pairnorm,
        on=keys,
        validate="one_to_one",
    )

    if len(paired) != EXPECTED_PAIRS:
        raise RuntimeError(
            f"Expected {EXPECTED_PAIRS} pairs, "
            f"found {len(paired)}"
        )

    for column in values:
        paired[f"delta_{column}"] = (
            paired[f"pairnorm_{column}"]
            - paired[f"baseline_{column}"]
        )

    paired["accuracy_delta_pp"] = (
        paired["delta_best_test_acc"]
        * 100.0
    )

    paired["baseline_chance"] = (
        paired["baseline_best_test_acc"]
        <= 0.52
    )
    paired["pairnorm_chance"] = (
        paired["pairnorm_best_test_acc"]
        <= 0.52
    )
    paired["collapse_prevented"] = (
        paired["baseline_chance"]
        & ~paired["pairnorm_chance"]
    )

    return paired.sort_values(
        [
            "feature_signal",
            "num_layers",
            "graph_seed",
        ]
    ).reset_index(drop=True)


def summarize_pairs(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for keys, group in paired.groupby(
        [
            "feature_signal",
            "num_layers",
        ]
    ):
        accuracy = summarize(
            group["accuracy_delta_pp"]
        )
        pairwise = summarize(
            group[
                "delta_final_pairwise_distance"
            ]
        )
        edge = summarize(
            group[
                "delta_final_edge_distance"
            ]
        )
        rank = summarize(
            group[
                "delta_final_rank_ratio"
            ]
        )

        t_pvalue, wilcoxon_pvalue = (
            test_against_zero(
                group["accuracy_delta_pp"]
            )
        )

        rows.append(
            {
                "feature_signal": keys[0],
                "num_layers": keys[1],
                "n": accuracy["n"],
                "baseline_accuracy_mean_pct":
                    group[
                        "baseline_best_test_acc"
                    ].mean()
                    * 100.0,
                "baseline_accuracy_std_pct":
                    group[
                        "baseline_best_test_acc"
                    ].std(ddof=1)
                    * 100.0,
                "pairnorm_accuracy_mean_pct":
                    group[
                        "pairnorm_best_test_acc"
                    ].mean()
                    * 100.0,
                "pairnorm_accuracy_std_pct":
                    group[
                        "pairnorm_best_test_acc"
                    ].std(ddof=1)
                    * 100.0,
                "accuracy_delta_mean_pp":
                    accuracy["mean"],
                "accuracy_delta_std_pp":
                    accuracy["std"],
                "accuracy_delta_ci95_pp":
                    accuracy["ci95"],
                "positive_seeds": int(
                    (
                        group["accuracy_delta_pp"]
                        > 0
                    ).sum()
                ),
                "negative_seeds": int(
                    (
                        group["accuracy_delta_pp"]
                        < 0
                    ).sum()
                ),
                "tie_seeds": int(
                    np.isclose(
                        group["accuracy_delta_pp"],
                        0.0,
                    ).sum()
                ),
                "baseline_chance_seeds": int(
                    group["baseline_chance"].sum()
                ),
                "pairnorm_chance_seeds": int(
                    group["pairnorm_chance"].sum()
                ),
                "collapse_prevented_seeds": int(
                    group[
                        "collapse_prevented"
                    ].sum()
                ),
                "paired_t_pvalue":
                    t_pvalue,
                "wilcoxon_pvalue":
                    wilcoxon_pvalue,
                "final_pairwise_delta_mean":
                    pairwise["mean"],
                "final_pairwise_delta_ci95":
                    pairwise["ci95"],
                "final_edge_delta_mean":
                    edge["mean"],
                "final_edge_delta_ci95":
                    edge["ci95"],
                "final_rank_delta_mean":
                    rank["mean"],
                "final_rank_delta_ci95":
                    rank["ci95"],
            }
        )

    result = pd.DataFrame(rows).sort_values(
        [
            "feature_signal",
            "num_layers",
        ]
    ).reset_index(drop=True)

    result[
        "paired_t_pvalue_holm"
    ] = holm_adjust(
        result["paired_t_pvalue"]
    )

    result[
        "wilcoxon_pvalue_holm"
    ] = holm_adjust(
        result["wilcoxon_pvalue"]
    )

    result[
        "significant_t_holm_005"
    ] = (
        result["paired_t_pvalue_holm"]
        < 0.05
    )

    return result


def summarize_models(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for keys, group in runs.groupby(
        [
            "feature_signal",
            "model",
            "num_layers",
        ]
    ):
        accuracy = summarize(
            group["best_test_acc"] * 100.0
        )
        pairwise = summarize(
            group["final_pairwise_distance"]
        )

        rows.append(
            {
                "feature_signal": keys[0],
                "model": keys[1],
                "num_layers": keys[2],
                "n": accuracy["n"],
                "accuracy_mean_pct":
                    accuracy["mean"],
                "accuracy_std_pct":
                    accuracy["std"],
                "accuracy_ci95_pct":
                    accuracy["ci95"],
                "final_pairwise_mean":
                    pairwise["mean"],
                "final_pairwise_ci95":
                    pairwise["ci95"],
                "chance_seeds": int(
                    (
                        group["best_test_acc"]
                        <= 0.52
                    ).sum()
                ),
                "ceiling_seeds": int(
                    (
                        group["best_test_acc"]
                        >= 0.95
                    ).sum()
                ),
                "best_epoch_mean": float(
                    group["best_epoch"].mean()
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        [
            "feature_signal",
            "model",
            "num_layers",
        ]
    ).reset_index(drop=True)


def analyze_adjacent_depths(
    runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows = []

    for keys, group in runs.groupby(
        [
            "feature_signal",
            "graph_seed",
            "model",
        ]
    ):
        by_depth = group.set_index(
            "num_layers"
        )

        for depth_from, depth_to in zip(
            DEPTHS[:-1],
            DEPTHS[1:],
        ):
            accuracy_from = float(
                by_depth.loc[
                    depth_from,
                    "best_test_acc",
                ]
            )
            accuracy_to = float(
                by_depth.loc[
                    depth_to,
                    "best_test_acc",
                ]
            )

            seed_rows.append(
                {
                    "feature_signal":
                        keys[0],
                    "graph_seed":
                        keys[1],
                    "model":
                        keys[2],
                    "uses_pairnorm":
                        keys[2].endswith(
                            "PairNorm"
                        ),
                    "depth_from":
                        depth_from,
                    "depth_to":
                        depth_to,
                    "accuracy_from_pct":
                        accuracy_from * 100.0,
                    "accuracy_to_pct":
                        accuracy_to * 100.0,
                    "adjacent_change_pp":
                        (
                            accuracy_to
                            - accuracy_from
                        )
                        * 100.0,
                }
            )

    seed_level = pd.DataFrame(seed_rows)
    summary_rows = []

    for keys, group in seed_level.groupby(
        [
            "feature_signal",
            "model",
            "depth_from",
            "depth_to",
        ]
    ):
        change = summarize(
            group["adjacent_change_pp"]
        )

        t_pvalue, wilcoxon_pvalue = (
            test_against_zero(
                group["adjacent_change_pp"]
            )
        )

        summary_rows.append(
            {
                "feature_signal":
                    keys[0],
                "model":
                    keys[1],
                "depth_from":
                    keys[2],
                "depth_to":
                    keys[3],
                "n":
                    change["n"],
                "adjacent_change_mean_pp":
                    change["mean"],
                "adjacent_change_std_pp":
                    change["std"],
                "adjacent_change_ci95_pp":
                    change["ci95"],
                "positive_seeds": int(
                    (
                        group[
                            "adjacent_change_pp"
                        ]
                        > 0
                    ).sum()
                ),
                "negative_seeds": int(
                    (
                        group[
                            "adjacent_change_pp"
                        ]
                        < 0
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
    ).sort_values(
        [
            "feature_signal",
            "model",
            "depth_from",
        ]
    ).reset_index(drop=True)

    summary[
        "paired_t_pvalue_holm"
    ] = holm_adjust(
        summary["paired_t_pvalue"]
    )

    return seed_level, summary


def analyze_local_parity(
    runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows = []

    for keys, group in runs.groupby(
        [
            "feature_signal",
            "graph_seed",
            "model",
        ]
    ):
        by_depth = group.set_index(
            "num_layers"
        )

        for odd_depth in (7, 9):
            odd_accuracy = float(
                by_depth.loc[
                    odd_depth,
                    "best_test_acc",
                ]
            )

            even_mean = float(
                (
                    by_depth.loc[
                        odd_depth - 1,
                        "best_test_acc",
                    ]
                    + by_depth.loc[
                        odd_depth + 1,
                        "best_test_acc",
                    ]
                )
                / 2.0
            )

            seed_rows.append(
                {
                    "feature_signal":
                        keys[0],
                    "graph_seed":
                        keys[1],
                    "model":
                        keys[2],
                    "odd_depth":
                        odd_depth,
                    "parity_residual_pp":
                        (
                            odd_accuracy
                            - even_mean
                        )
                        * 100.0,
                }
            )

    seed_level = pd.DataFrame(seed_rows)
    summary_rows = []

    for keys, group in seed_level.groupby(
        [
            "feature_signal",
            "model",
            "odd_depth",
        ]
    ):
        residual = summarize(
            group["parity_residual_pp"]
        )

        t_pvalue, wilcoxon_pvalue = (
            test_against_zero(
                group["parity_residual_pp"]
            )
        )

        summary_rows.append(
            {
                "feature_signal":
                    keys[0],
                "model":
                    keys[1],
                "odd_depth":
                    keys[2],
                "n":
                    residual["n"],
                "parity_residual_mean_pp":
                    residual["mean"],
                "parity_residual_std_pp":
                    residual["std"],
                "parity_residual_ci95_pp":
                    residual["ci95"],
                "positive_seeds": int(
                    (
                        group[
                            "parity_residual_pp"
                        ]
                        > 0
                    ).sum()
                ),
                "negative_seeds": int(
                    (
                        group[
                            "parity_residual_pp"
                        ]
                        < 0
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
    ).sort_values(
        [
            "feature_signal",
            "model",
            "odd_depth",
        ]
    ).reset_index(drop=True)

    summary[
        "paired_t_pvalue_holm"
    ] = holm_adjust(
        summary["paired_t_pvalue"]
    )

    return seed_level, summary


def build_correlations(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    metrics = {
        "final_pairwise_delta":
            "delta_final_pairwise_distance",
        "final_edge_delta":
            "delta_final_edge_distance",
        "final_rank_delta":
            "delta_final_rank_ratio",
        "best_pairwise_delta":
            "delta_best_pairwise_distance",
        "best_edge_delta":
            "delta_best_edge_distance",
    }

    groups = [
        ("All paired runs", paired),
    ]

    for signal, frame in paired.groupby(
        "feature_signal"
    ):
        groups.append(
            (
                f"Feature signal {signal:.2f}",
                frame,
            )
        )

    for depth, frame in paired.groupby(
        "num_layers"
    ):
        groups.append(
            (
                f"Depth L{depth}",
                frame,
            )
        )

    rows = []

    for group_name, frame in groups:
        for metric_name, column in metrics.items():
            correlation = (
                frame[
                    [
                        "accuracy_delta_pp",
                        column,
                    ]
                ]
                .corr(method="spearman")
                .iloc[0, 1]
            )

            rows.append(
                {
                    "group": group_name,
                    "n": len(frame),
                    "metric": metric_name,
                    "spearman_correlation":
                        correlation,
                }
            )

    return pd.DataFrame(rows)


def plot_accuracy(
    model_summary: pd.DataFrame,
) -> None:
    for signal, frame in model_summary.groupby(
        "feature_signal"
    ):
        plt.figure(figsize=(9, 6))

        for model, model_frame in frame.groupby(
            "model"
        ):
            model_frame = model_frame.sort_values(
                "num_layers"
            )

            plt.errorbar(
                model_frame["num_layers"],
                model_frame["accuracy_mean_pct"],
                yerr=model_frame[
                    "accuracy_ci95_pct"
                ],
                marker="o",
                capsize=3,
                label=model,
            )

        plt.xlabel("Number of layers")
        plt.ylabel("Best test accuracy (%)")
        plt.title(
            "Low-label heterophilic CSBM "
            f"(feature signal={signal:.2f})"
        )
        plt.xticks(DEPTHS)
        plt.legend()
        plt.tight_layout()

        path = (
            PLOT_DIR
            / (
                "accuracy_vs_depth_"
                f"fs{int(signal * 100):03d}.png"
            )
        )

        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print("Saved:", path)


def plot_pairnorm_delta(
    pair_summary: pd.DataFrame,
) -> None:
    for signal, frame in pair_summary.groupby(
        "feature_signal"
    ):
        frame = frame.sort_values(
            "num_layers"
        )

        plt.figure(figsize=(9, 6))

        plt.errorbar(
            frame["num_layers"],
            frame["accuracy_delta_mean_pp"],
            yerr=frame[
                "accuracy_delta_ci95_pp"
            ],
            marker="o",
            capsize=3,
        )

        plt.axhline(
            0.0,
            linewidth=1,
        )
        plt.xlabel("Number of layers")
        plt.ylabel(
            "PairNorm accuracy effect (pp)"
        )
        plt.title(
            "Paired PairNorm effect "
            f"(feature signal={signal:.2f})"
        )
        plt.xticks(DEPTHS)
        plt.tight_layout()

        path = (
            PLOT_DIR
            / (
                "pairnorm_delta_vs_depth_"
                f"fs{int(signal * 100):03d}.png"
            )
        )

        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print("Saved:", path)


def plot_pairwise(
    model_summary: pd.DataFrame,
) -> None:
    for signal, frame in model_summary.groupby(
        "feature_signal"
    ):
        plt.figure(figsize=(9, 6))

        for model, model_frame in frame.groupby(
            "model"
        ):
            model_frame = model_frame.sort_values(
                "num_layers"
            )

            plt.errorbar(
                model_frame["num_layers"],
                model_frame[
                    "final_pairwise_mean"
                ],
                yerr=model_frame[
                    "final_pairwise_ci95"
                ],
                marker="o",
                capsize=3,
                label=model,
            )

        plt.xlabel("Number of layers")
        plt.ylabel(
            "Final hidden pairwise cosine distance"
        )
        plt.title(
            "Representation separation "
            f"(feature signal={signal:.2f})"
        )
        plt.xticks(DEPTHS)
        plt.legend()
        plt.tight_layout()

        path = (
            PLOT_DIR
            / (
                "hidden_pairwise_vs_depth_"
                f"fs{int(signal * 100):03d}.png"
            )
        )

        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print("Saved:", path)


def plot_adjacent_changes(
    adjacent_summary: pd.DataFrame,
) -> None:
    for signal, frame in adjacent_summary.groupby(
        "feature_signal"
    ):
        plt.figure(figsize=(10, 6))

        for model, model_frame in frame.groupby(
            "model"
        ):
            model_frame = model_frame.sort_values(
                "depth_from"
            )

            labels = [
                (
                    f"{left}→{right}"
                )
                for left, right in zip(
                    model_frame["depth_from"],
                    model_frame["depth_to"],
                )
            ]

            plt.errorbar(
                labels,
                model_frame[
                    "adjacent_change_mean_pp"
                ],
                yerr=model_frame[
                    "adjacent_change_ci95_pp"
                ],
                marker="o",
                capsize=3,
                label=model,
            )

        plt.axhline(
            0.0,
            linewidth=1,
        )
        plt.xlabel("Depth transition")
        plt.ylabel(
            "Accuracy change at next depth (pp)"
        )
        plt.title(
            "Adjacent depth transitions "
            f"(feature signal={signal:.2f})"
        )
        plt.legend()
        plt.tight_layout()

        path = (
            PLOT_DIR
            / (
                "adjacent_depth_changes_"
                f"fs{int(signal * 100):03d}.png"
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
    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    runs = load_runs()
    paired = build_pairs(runs)

    pair_summary = summarize_pairs(
        paired
    )
    model_summary = summarize_models(
        runs
    )

    adjacent_seed, adjacent_summary = (
        analyze_adjacent_depths(runs)
    )

    parity_seed, parity_summary = (
        analyze_local_parity(runs)
    )

    correlations = build_correlations(
        paired
    )

    outputs = {
        "csbm_h01_parity_run_level.csv":
            runs,
        "csbm_h01_parity_paired.csv":
            paired,
        "csbm_h01_parity_pairnorm_summary.csv":
            pair_summary,
        "csbm_h01_parity_model_summary.csv":
            model_summary,
        "csbm_h01_parity_adjacent_seed.csv":
            adjacent_seed,
        "csbm_h01_parity_adjacent_summary.csv":
            adjacent_summary,
        "csbm_h01_parity_local_seed.csv":
            parity_seed,
        "csbm_h01_parity_local_summary.csv":
            parity_summary,
        "csbm_h01_parity_correlations.csv":
            correlations,
    }

    for filename, frame in outputs.items():
        path = RUN_DIR / filename
        frame.to_csv(
            path,
            index=False,
        )
        print("Saved:", path)

    plot_accuracy(model_summary)
    plot_pairnorm_delta(pair_summary)
    plot_pairwise(model_summary)
    plot_adjacent_changes(
        adjacent_summary
    )

    print(
        "\n=== PairNorm effects by depth ==="
    )
    print(
        pair_summary[
            [
                "feature_signal",
                "num_layers",
                "baseline_accuracy_mean_pct",
                "pairnorm_accuracy_mean_pct",
                "accuracy_delta_mean_pp",
                "accuracy_delta_ci95_pp",
                "positive_seeds",
                "negative_seeds",
                "baseline_chance_seeds",
                "pairnorm_chance_seeds",
                "collapse_prevented_seeds",
                "paired_t_pvalue_holm",
                "significant_t_holm_005",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    print(
        "\n=== Adjacent depth changes ==="
    )
    print(
        adjacent_summary[
            [
                "feature_signal",
                "model",
                "depth_from",
                "depth_to",
                "adjacent_change_mean_pp",
                "adjacent_change_ci95_pp",
                "positive_seeds",
                "negative_seeds",
                "paired_t_pvalue_holm",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    print(
        "\n=== Local parity residuals ==="
    )
    print(
        parity_summary[
            [
                "feature_signal",
                "model",
                "odd_depth",
                "parity_residual_mean_pp",
                "parity_residual_ci95_pp",
                "positive_seeds",
                "negative_seeds",
                "paired_t_pvalue_holm",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    print(
        "\n=== Mechanism correlations ==="
    )
    print(
        correlations[
            correlations["group"].isin(
                [
                    "All paired runs",
                    "Feature signal 0.25",
                    "Feature signal 0.50",
                ]
            )
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    print(
        "\n=== Model-level collapse diagnostics ==="
    )
    print(
        model_summary[
            [
                "feature_signal",
                "model",
                "num_layers",
                "accuracy_mean_pct",
                "accuracy_ci95_pct",
                "chance_seeds",
                "ceiling_seeds",
                "final_pairwise_mean",
                "best_epoch_mean",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )


if __name__ == "__main__":
    main()
