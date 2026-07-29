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
    "runs/csbm_lowlabel_graphsage"
)
PLOT_DIR = Path(
    "plots/csbm_lowlabel_graphsage"
)

EXPECTED_RUNS = 180
EXPECTED_PAIRS = 90


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

    order = valid[
        np.argsort(pvalues[valid])
    ]

    running_maximum = 0.0
    number = len(order)

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


def summarize(
    values: pd.Series,
) -> dict:
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
            f"No hidden representation found for {model}"
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

        metric_path = summary_path.with_name(
            summary_path.name.replace(
                "_summary.json",
                "_oversmoothing.csv",
            )
        )

        if not metric_path.exists():
            raise FileNotFoundError(metric_path)

        metrics = pd.read_csv(metric_path)
        metrics = metrics[
            metrics["layer_name"].notna()
        ].copy()

        recorded_epochs = sorted(
            int(value)
            for value in metrics["epoch"].unique()
        )

        best_epoch = int(summary["best_epoch"])

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
            "final_metric_epoch":
                final_epoch,
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
        "target_homophily",
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
        how="inner",
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

    return paired.sort_values(
        [
            "target_homophily",
            "feature_signal",
            "num_layers",
            "graph_seed",
        ]
    ).reset_index(drop=True)


def summarize_pairs(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    group_columns = [
        "target_homophily",
        "feature_signal",
        "num_layers",
    ]

    for keys, group in paired.groupby(
        group_columns
    ):
        accuracy = summarize(
            group["accuracy_delta_pp"]
        )
        final_pairwise = summarize(
            group[
                "delta_final_pairwise_distance"
            ]
        )
        final_edge = summarize(
            group[
                "delta_final_edge_distance"
            ]
        )
        final_rank = summarize(
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
                "target_homophily":
                    keys[0],
                "feature_signal":
                    keys[1],
                "num_layers":
                    keys[2],
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
                "paired_t_pvalue":
                    t_pvalue,
                "wilcoxon_pvalue":
                    wilcoxon_pvalue,
                "final_pairwise_delta_mean":
                    final_pairwise["mean"],
                "final_pairwise_delta_ci95":
                    final_pairwise["ci95"],
                "final_edge_delta_mean":
                    final_edge["mean"],
                "final_edge_delta_ci95":
                    final_edge["ci95"],
                "final_rank_delta_mean":
                    final_rank["mean"],
                "final_rank_delta_ci95":
                    final_rank["ci95"],
            }
        )

    result = pd.DataFrame(rows).sort_values(
        group_columns
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
            "target_homophily",
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
                "target_homophily":
                    keys[0],
                "feature_signal":
                    keys[1],
                "model":
                    keys[2],
                "num_layers":
                    keys[3],
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
                "ceiling_runs_ge_95pct":
                    int(
                        (
                            group["best_test_acc"]
                            >= 0.95
                        ).sum()
                    ),
                "chance_runs_le_52pct":
                    int(
                        (
                            group["best_test_acc"]
                            <= 0.52
                        ).sum()
                    ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        [
            "target_homophily",
            "feature_signal",
            "model",
            "num_layers",
        ]
    ).reset_index(drop=True)


def analyze_depth(
    runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows = []

    for keys, group in runs.groupby(
        [
            "target_homophily",
            "feature_signal",
            "graph_seed",
            "model",
            "uses_pairnorm",
        ]
    ):
        by_depth = group.set_index(
            "num_layers"
        )

        if not {2, 7, 8}.issubset(
            set(by_depth.index)
        ):
            raise RuntimeError(
                f"Missing depth for group {keys}"
            )

        seed_rows.append(
            {
                "target_homophily":
                    keys[0],
                "feature_signal":
                    keys[1],
                "graph_seed":
                    keys[2],
                "model":
                    keys[3],
                "uses_pairnorm":
                    keys[4],
                "accuracy_l2_pct":
                    float(
                        by_depth.loc[
                            2,
                            "best_test_acc",
                        ]
                    )
                    * 100.0,
                "accuracy_l7_pct":
                    float(
                        by_depth.loc[
                            7,
                            "best_test_acc",
                        ]
                    )
                    * 100.0,
                "accuracy_l8_pct":
                    float(
                        by_depth.loc[
                            8,
                            "best_test_acc",
                        ]
                    )
                    * 100.0,
            }
        )

    model_depth = pd.DataFrame(seed_rows)

    model_depth["drop_l2_l7_pp"] = (
        model_depth["accuracy_l2_pct"]
        - model_depth["accuracy_l7_pct"]
    )
    model_depth["drop_l2_l8_pp"] = (
        model_depth["accuracy_l2_pct"]
        - model_depth["accuracy_l8_pct"]
    )

    keys = [
        "target_homophily",
        "feature_signal",
        "graph_seed",
    ]

    baseline = model_depth[
        ~model_depth["uses_pairnorm"]
    ].drop(
        columns=[
            "model",
            "uses_pairnorm",
        ]
    )

    pairnorm = model_depth[
        model_depth["uses_pairnorm"]
    ].drop(
        columns=[
            "model",
            "uses_pairnorm",
        ]
    )

    baseline = baseline.rename(
        columns={
            column: f"baseline_{column}"
            for column in baseline.columns
            if column not in keys
        }
    )

    pairnorm = pairnorm.rename(
        columns={
            column: f"pairnorm_{column}"
            for column in pairnorm.columns
            if column not in keys
        }
    )

    seed_level = baseline.merge(
        pairnorm,
        on=keys,
        validate="one_to_one",
    )

    for depth in (7, 8):
        seed_level[
            f"drop_reduction_l2_l{depth}_pp"
        ] = (
            seed_level[
                f"baseline_drop_l2_l{depth}_pp"
            ]
            - seed_level[
                f"pairnorm_drop_l2_l{depth}_pp"
            ]
        )

    summary_rows = []

    for keys, group in seed_level.groupby(
        [
            "target_homophily",
            "feature_signal",
        ]
    ):
        for depth in (7, 8):
            column = (
                f"drop_reduction_l2_l{depth}_pp"
            )

            reduction = summarize(
                group[column]
            )

            t_pvalue, wilcoxon_pvalue = (
                test_against_zero(
                    group[column]
                )
            )

            summary_rows.append(
                {
                    "target_homophily":
                        keys[0],
                    "feature_signal":
                        keys[1],
                    "target_depth":
                        depth,
                    "n":
                        reduction["n"],
                    "baseline_depth_drop_mean_pp":
                        group[
                            (
                                "baseline_drop_"
                                f"l2_l{depth}_pp"
                            )
                        ].mean(),
                    "pairnorm_depth_drop_mean_pp":
                        group[
                            (
                                "pairnorm_drop_"
                                f"l2_l{depth}_pp"
                            )
                        ].mean(),
                    "depth_drop_reduction_mean_pp":
                        reduction["mean"],
                    "depth_drop_reduction_ci95_pp":
                        reduction["ci95"],
                    "positive_reduction_seeds":
                        int(
                            (
                                group[column] > 0
                            ).sum()
                        ),
                    "negative_reduction_seeds":
                        int(
                            (
                                group[column] < 0
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
            "target_homophily",
            "feature_signal",
            "target_depth",
        ]
    )

    summary[
        "paired_t_pvalue_holm"
    ] = holm_adjust(
        summary["paired_t_pvalue"]
    )

    return seed_level, summary


def build_correlations(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = {
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

    for homophily, frame in paired.groupby(
        "target_homophily"
    ):
        groups.append(
            (
                f"Homophily {homophily:.1f}",
                frame,
            )
        )

    for keys, frame in paired.groupby(
        [
            "target_homophily",
            "feature_signal",
        ]
    ):
        groups.append(
            (
                (
                    f"Homophily {keys[0]:.1f}, "
                    f"signal {keys[1]:.2f}"
                ),
                frame,
            )
        )

    rows = []

    for group_name, frame in groups:
        for metric_name, column in (
            metric_columns.items()
        ):
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
                    "group":
                        group_name,
                    "n":
                        len(frame),
                    "metric":
                        metric_name,
                    "spearman_correlation":
                        correlation,
                }
            )

    return pd.DataFrame(rows)


def plot_accuracy(
    model_summary: pd.DataFrame,
) -> None:
    for keys, frame in model_summary.groupby(
        [
            "target_homophily",
            "feature_signal",
        ]
    ):
        homophily, signal = keys

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
                    "accuracy_mean_pct"
                ],
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
            "Low-label CSBM accuracy "
            f"(h={homophily:.1f}, "
            f"feature signal={signal:.2f})"
        )
        plt.xticks([2, 7, 8])
        plt.legend()
        plt.tight_layout()

        path = (
            PLOT_DIR
            / (
                "accuracy_vs_depth_"
                f"h{int(homophily * 10):02d}_"
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
    for keys, frame in pair_summary.groupby(
        [
            "target_homophily",
            "feature_signal",
        ]
    ):
        homophily, signal = keys
        frame = frame.sort_values(
            "num_layers"
        )

        plt.figure(figsize=(8, 6))

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
            "Low-label paired PairNorm effect "
            f"(h={homophily:.1f}, "
            f"feature signal={signal:.2f})"
        )
        plt.xticks([2, 7, 8])
        plt.tight_layout()

        path = (
            PLOT_DIR
            / (
                "pairnorm_delta_"
                f"h{int(homophily * 10):02d}_"
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
    for keys, frame in model_summary.groupby(
        [
            "target_homophily",
            "feature_signal",
        ]
    ):
        homophily, signal = keys

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
            "Low-label representation separation "
            f"(h={homophily:.1f}, "
            f"feature signal={signal:.2f})"
        )
        plt.xticks([2, 7, 8])
        plt.legend()
        plt.tight_layout()

        path = (
            PLOT_DIR
            / (
                "hidden_pairwise_"
                f"h{int(homophily * 10):02d}_"
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


def plot_delta_scatter(
    paired: pd.DataFrame,
) -> None:
    plt.figure(figsize=(9, 6))

    for keys, frame in paired.groupby(
        [
            "target_homophily",
            "feature_signal",
        ]
    ):
        label = (
            f"h={keys[0]:.1f}, "
            f"FS={keys[1]:.2f}"
        )

        plt.scatter(
            frame[
                "delta_final_pairwise_distance"
            ],
            frame["accuracy_delta_pp"],
            label=label,
        )

    plt.axhline(
        0.0,
        linewidth=1,
    )
    plt.xlabel(
        "PairNorm delta in final pairwise distance"
    )
    plt.ylabel(
        "PairNorm accuracy effect (pp)"
    )
    plt.title(
        "Accuracy improvement and "
        "representation separation"
    )
    plt.legend()
    plt.tight_layout()

    path = (
        PLOT_DIR
        / "accuracy_delta_vs_pairwise_delta.png"
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

    depth_seed, depth_summary = (
        analyze_depth(runs)
    )

    correlations = build_correlations(
        paired
    )

    outputs = {
        "csbm_lowlabel_run_level.csv":
            runs,
        "csbm_lowlabel_pairnorm_paired.csv":
            paired,
        "csbm_lowlabel_pairnorm_summary.csv":
            pair_summary,
        "csbm_lowlabel_model_summary.csv":
            model_summary,
        "csbm_lowlabel_depth_seed.csv":
            depth_seed,
        "csbm_lowlabel_depth_summary.csv":
            depth_summary,
        "csbm_lowlabel_correlations.csv":
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
    plot_delta_scatter(paired)

    print(
        "\n=== PairNorm effects ==="
    )

    print(
        pair_summary[
            [
                "target_homophily",
                "feature_signal",
                "num_layers",
                "baseline_accuracy_mean_pct",
                "pairnorm_accuracy_mean_pct",
                "accuracy_delta_mean_pp",
                "accuracy_delta_ci95_pp",
                "positive_seeds",
                "negative_seeds",
                "paired_t_pvalue",
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
        "\n=== Depth-drop reduction ==="
    )

    print(
        depth_summary[
            [
                "target_homophily",
                "feature_signal",
                "target_depth",
                "baseline_depth_drop_mean_pp",
                "pairnorm_depth_drop_mean_pp",
                "depth_drop_reduction_mean_pp",
                "depth_drop_reduction_ci95_pp",
                "positive_reduction_seeds",
                "paired_t_pvalue_holm",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    print(
        "\n=== Main mechanism correlations ==="
    )

    selected_groups = [
        "All paired runs",
        "Feature signal 0.25",
        "Feature signal 0.50",
        "Homophily 0.1",
        "Homophily 0.5",
        "Homophily 0.9",
    ]

    print(
        correlations[
            correlations["group"].isin(
                selected_groups
            )
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    print(
        "\n=== Ceiling and chance-level diagnostics ==="
    )

    print(
        model_summary[
            [
                "target_homophily",
                "feature_signal",
                "model",
                "num_layers",
                "accuracy_mean_pct",
                "ceiling_runs_ge_95pct",
                "chance_runs_le_52pct",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )


if __name__ == "__main__":
    main()
