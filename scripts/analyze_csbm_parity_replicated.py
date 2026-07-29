from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:
    stats = None


RUN_DIR = Path("runs/csbm_parity_replicated")
PLOT_DIR = Path("plots/csbm_parity_replicated")

EXPECTED_RUNS = 280
EXPECTED_PAIRS = 140


def module_number(name: object) -> int:
    match = re.search(
        r"\.(\d+)(?:#\d+)?$",
        str(name),
    )
    return int(match.group(1)) if match else -1


def summarize(values: pd.Series) -> dict:
    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    n = len(values)

    if n == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "std": math.nan,
            "ci95": math.nan,
        }

    mean = float(values.mean())

    if n == 1:
        return {
            "n": 1,
            "mean": mean,
            "std": 0.0,
            "ci95": 0.0,
        }

    std = float(values.std(ddof=1))
    sem = std / math.sqrt(n)

    if stats is not None:
        critical = float(
            stats.t.ppf(
                0.975,
                df=n - 1,
            )
        )
    else:
        critical = 1.96

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95": critical * sem,
    }


def significance(values: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if stats is None or len(values) < 2:
        return math.nan, math.nan

    array = values.to_numpy()

    if np.allclose(array, 0.0):
        return 1.0, 1.0

    t_test = stats.ttest_1samp(
        array,
        popmean=0.0,
    )

    try:
        wilcoxon = stats.wilcoxon(
            array,
            zero_method="wilcox",
            alternative="two-sided",
        )
        wilcoxon_p = float(
            wilcoxon.pvalue
        )
    except ValueError:
        wilcoxon_p = 1.0

    return (
        float(t_test.pvalue),
        wilcoxon_p,
    )


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
            f"No hidden layer found for {model}"
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
            raise FileNotFoundError(
                metric_path
            )

        metrics = pd.read_csv(metric_path)
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
            "architecture": model.replace(
                "PairNorm",
                "",
            ),
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
            "best_epoch": best_epoch,
            "nearest_best_metric_epoch":
                nearest_best_epoch,
            "best_test_acc": float(
                summary[
                    "best_test_acc_at_best_val"
                ]
            ),
            "best_val_acc": float(
                summary["best_val_acc"]
            ),
        }

        for checkpoint, epoch in (
            ("best", nearest_best_epoch),
            ("final", final_epoch),
        ):
            checkpoint_frame = metrics[
                metrics["epoch"] == epoch
            ]

            hidden = select_hidden(
                checkpoint_frame,
                model=model,
                hidden_channels=hidden_channels,
            )

            prefix = f"{checkpoint}_hidden"

            row[
                f"{prefix}_pairwise_distance"
            ] = float(
                hidden[
                    "mean_pairwise_cosine_distance"
                ]
            )
            row[
                f"{prefix}_edge_distance"
            ] = float(
                hidden[
                    "mean_edge_cosine_distance"
                ]
            )
            row[
                f"{prefix}_rank_ratio"
            ] = float(
                hidden[
                    "effective_rank_ratio"
                ]
            )
            row[
                f"{prefix}_normalized_dirichlet"
            ] = float(
                hidden[
                    "normalized_dirichlet_energy"
                ]
            )

        rows.append(row)

    runs = pd.DataFrame(rows)

    keys = [
        "target_homophily",
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
        "dataset",
        "architecture",
        "num_layers",
        "hidden_channels",
        "seed",
        "graph_seed",
        "split_idx",
        "target_homophily",
        "realized_homophily",
    ]

    values = [
        "best_epoch",
        "best_test_acc",
        "best_val_acc",
        "best_hidden_pairwise_distance",
        "best_hidden_edge_distance",
        "best_hidden_rank_ratio",
        "best_hidden_normalized_dirichlet",
        "final_hidden_pairwise_distance",
        "final_hidden_edge_distance",
        "final_hidden_rank_ratio",
        "final_hidden_normalized_dirichlet",
    ]

    baseline = runs[
        ~runs["uses_pairnorm"]
    ][keys + values].copy()

    pairnorm = runs[
        runs["uses_pairnorm"]
    ][keys + values].copy()

    baseline = baseline.rename(
        columns={
            column:
                f"baseline_{column}"
            for column in values
        }
    )

    pairnorm = pairnorm.rename(
        columns={
            column:
                f"pairnorm_{column}"
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
            - paired[
                f"baseline_{column}"
            ]
        )

    paired["accuracy_delta_pp"] = (
        paired["delta_best_test_acc"]
        * 100.0
    )

    return paired.sort_values(
        [
            "target_homophily",
            "architecture",
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
        "architecture",
        "num_layers",
    ]

    for keys, group in paired.groupby(
        group_columns
    ):
        accuracy = summarize(
            group["accuracy_delta_pp"]
        )
        pairwise = summarize(
            group[
                "delta_final_hidden_"
                "pairwise_distance"
            ]
        )
        rank_ratio = summarize(
            group[
                "delta_final_hidden_"
                "rank_ratio"
            ]
        )
        best_pairwise = summarize(
            group[
                "delta_best_hidden_"
                "pairwise_distance"
            ]
        )

        t_p, wilcoxon_p = significance(
            group["accuracy_delta_pp"]
        )

        rows.append(
            {
                "target_homophily":
                    keys[0],
                "architecture": keys[1],
                "num_layers": keys[2],
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
                        group[
                            "accuracy_delta_pp"
                        ]
                        > 0
                    ).sum()
                ),
                "negative_seeds": int(
                    (
                        group[
                            "accuracy_delta_pp"
                        ]
                        < 0
                    ).sum()
                ),
                "tie_seeds": int(
                    np.isclose(
                        group[
                            "accuracy_delta_pp"
                        ],
                        0.0,
                    ).sum()
                ),
                "paired_t_pvalue": t_p,
                "wilcoxon_pvalue":
                    wilcoxon_p,
                "final_pairwise_delta_mean":
                    pairwise["mean"],
                "final_pairwise_delta_ci95":
                    pairwise["ci95"],
                "final_rank_ratio_delta_mean":
                    rank_ratio["mean"],
                "final_rank_ratio_delta_ci95":
                    rank_ratio["ci95"],
                "best_pairwise_delta_mean":
                    best_pairwise["mean"],
                "best_pairwise_delta_ci95":
                    best_pairwise["ci95"],
            }
        )

    return pd.DataFrame(rows).sort_values(
        group_columns
    ).reset_index(drop=True)


def summarize_models(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for keys, group in runs.groupby(
        [
            "target_homophily",
            "model",
            "num_layers",
        ]
    ):
        accuracy = summarize(
            group["best_test_acc"] * 100.0
        )
        pairwise = summarize(
            group[
                "final_hidden_pairwise_distance"
            ]
        )
        rank_ratio = summarize(
            group[
                "final_hidden_rank_ratio"
            ]
        )

        rows.append(
            {
                "target_homophily":
                    keys[0],
                "model": keys[1],
                "num_layers": keys[2],
                "n": accuracy["n"],
                "accuracy_mean_pct":
                    accuracy["mean"],
                "accuracy_std_pct":
                    accuracy["std"],
                "accuracy_ci95_pct":
                    accuracy["ci95"],
                "hidden_pairwise_mean":
                    pairwise["mean"],
                "hidden_pairwise_ci95":
                    pairwise["ci95"],
                "hidden_rank_ratio_mean":
                    rank_ratio["mean"],
                "hidden_rank_ratio_ci95":
                    rank_ratio["ci95"],
            }
        )

    return pd.DataFrame(rows).sort_values(
        [
            "target_homophily",
            "model",
            "num_layers",
        ]
    ).reset_index(drop=True)


def analyze_depth_drop(
    runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_rows = []

    for keys, group in runs.groupby(
        [
            "target_homophily",
            "graph_seed",
            "architecture",
            "uses_pairnorm",
        ]
    ):
        by_depth = group.set_index(
            "num_layers"
        )

        model_rows.append(
            {
                "target_homophily":
                    keys[0],
                "graph_seed": keys[1],
                "architecture": keys[2],
                "uses_pairnorm": keys[3],
                "accuracy_l2_pct":
                    by_depth.loc[
                        2,
                        "best_test_acc",
                    ]
                    * 100.0,
                "accuracy_l8_pct":
                    by_depth.loc[
                        8,
                        "best_test_acc",
                    ]
                    * 100.0,
                "depth_drop_pp": (
                    by_depth.loc[
                        2,
                        "best_test_acc",
                    ]
                    - by_depth.loc[
                        8,
                        "best_test_acc",
                    ]
                )
                * 100.0,
            }
        )

    effects = pd.DataFrame(model_rows)

    merge_keys = [
        "target_homophily",
        "graph_seed",
        "architecture",
    ]

    baseline = effects[
        ~effects["uses_pairnorm"]
    ].drop(
        columns="uses_pairnorm"
    )

    pairnorm = effects[
        effects["uses_pairnorm"]
    ].drop(
        columns="uses_pairnorm"
    )

    baseline = baseline.rename(
        columns={
            "accuracy_l2_pct":
                "baseline_accuracy_l2_pct",
            "accuracy_l8_pct":
                "baseline_accuracy_l8_pct",
            "depth_drop_pp":
                "baseline_depth_drop_pp",
        }
    )

    pairnorm = pairnorm.rename(
        columns={
            "accuracy_l2_pct":
                "pairnorm_accuracy_l2_pct",
            "accuracy_l8_pct":
                "pairnorm_accuracy_l8_pct",
            "depth_drop_pp":
                "pairnorm_depth_drop_pp",
        }
    )

    seed_level = baseline.merge(
        pairnorm,
        on=merge_keys,
        validate="one_to_one",
    )

    seed_level[
        "depth_drop_reduction_pp"
    ] = (
        seed_level[
            "baseline_depth_drop_pp"
        ]
        - seed_level[
            "pairnorm_depth_drop_pp"
        ]
    )

    rows = []

    for keys, group in seed_level.groupby(
        [
            "target_homophily",
            "architecture",
        ]
    ):
        baseline_stats = summarize(
            group["baseline_depth_drop_pp"]
        )
        pairnorm_stats = summarize(
            group["pairnorm_depth_drop_pp"]
        )
        reduction = summarize(
            group["depth_drop_reduction_pp"]
        )

        t_p, wilcoxon_p = significance(
            group["depth_drop_reduction_pp"]
        )

        rows.append(
            {
                "target_homophily":
                    keys[0],
                "architecture": keys[1],
                "n": reduction["n"],
                "baseline_depth_drop_mean_pp":
                    baseline_stats["mean"],
                "baseline_depth_drop_std_pp":
                    baseline_stats["std"],
                "pairnorm_depth_drop_mean_pp":
                    pairnorm_stats["mean"],
                "pairnorm_depth_drop_std_pp":
                    pairnorm_stats["std"],
                "depth_drop_reduction_mean_pp":
                    reduction["mean"],
                "depth_drop_reduction_ci95_pp":
                    reduction["ci95"],
                "positive_reduction_seeds":
                    int(
                        (
                            group[
                                "depth_drop_reduction_pp"
                            ]
                            > 0
                        ).sum()
                    ),
                "paired_t_pvalue": t_p,
                "wilcoxon_pvalue":
                    wilcoxon_p,
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        [
            "target_homophily",
            "architecture",
        ]
    ).reset_index(drop=True)

    return seed_level, summary


def analyze_parity(
    runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows = []

    for keys, group in runs.groupby(
        [
            "target_homophily",
            "graph_seed",
            "model",
        ]
    ):
        by_depth = group.set_index(
            "num_layers"
        )

        residuals = []

        for odd_depth in (3, 5, 7):
            odd_accuracy = float(
                by_depth.loc[
                    odd_depth,
                    "best_test_acc",
                ]
            )

            neighboring_even_mean = float(
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

            residuals.append(
                (
                    odd_accuracy
                    - neighboring_even_mean
                )
                * 100.0
            )

        model = keys[2]

        seed_rows.append(
            {
                "target_homophily":
                    keys[0],
                "graph_seed": keys[1],
                "model": model,
                "architecture":
                    model.replace(
                        "PairNorm",
                        "",
                    ),
                "uses_pairnorm":
                    model.endswith(
                        "PairNorm"
                    ),
                "odd_minus_neighbor_even_pp":
                    float(
                        np.mean(residuals)
                    ),
            }
        )

    seed_level = pd.DataFrame(seed_rows)
    rows = []

    for keys, group in seed_level.groupby(
        [
            "target_homophily",
            "model",
            "architecture",
            "uses_pairnorm",
        ]
    ):
        residual = summarize(
            group[
                "odd_minus_neighbor_even_pp"
            ]
        )

        t_p, wilcoxon_p = significance(
            group[
                "odd_minus_neighbor_even_pp"
            ]
        )

        rows.append(
            {
                "target_homophily":
                    keys[0],
                "model": keys[1],
                "architecture": keys[2],
                "uses_pairnorm": keys[3],
                "n": residual["n"],
                "parity_residual_mean_pp":
                    residual["mean"],
                "parity_residual_std_pp":
                    residual["std"],
                "parity_residual_ci95_pp":
                    residual["ci95"],
                "negative_residual_seeds":
                    int(
                        (
                            group[
                                "odd_minus_neighbor_even_pp"
                            ]
                            < 0
                        ).sum()
                    ),
                "paired_t_pvalue": t_p,
                "wilcoxon_pvalue":
                    wilcoxon_p,
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        [
            "target_homophily",
            "model",
        ]
    ).reset_index(drop=True)

    return seed_level, summary


def analyze_correlations(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    metrics = {
        "final_pairwise_delta":
            "delta_final_hidden_"
            "pairwise_distance",
        "final_edge_delta":
            "delta_final_hidden_"
            "edge_distance",
        "final_rank_ratio_delta":
            "delta_final_hidden_"
            "rank_ratio",
        "best_pairwise_delta":
            "delta_best_hidden_"
            "pairwise_distance",
        "best_rank_ratio_delta":
            "delta_best_hidden_"
            "rank_ratio",
    }

    groups = [
        ("All paired runs", paired),
    ]

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
            "architecture",
        ]
    ):
        groups.append(
            (
                (
                    f"Homophily {keys[0]:.1f}, "
                    f"{keys[1]}"
                ),
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
    for homophily, frame in model_summary.groupby(
        "target_homophily"
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
            "Replicated CSBM accuracy "
            f"(homophily={homophily:.1f})"
        )
        plt.xticks(range(2, 9))
        plt.legend()
        plt.tight_layout()

        output = (
            PLOT_DIR
            / (
                "accuracy_vs_depth_"
                f"h{int(homophily * 10):02d}.png"
            )
        )

        plt.savefig(
            output,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print("Saved:", output)


def plot_pairnorm(
    pair_summary: pd.DataFrame,
) -> None:
    for homophily, frame in pair_summary.groupby(
        "target_homophily"
    ):
        plt.figure(figsize=(9, 6))

        for architecture, architecture_frame in (
            frame.groupby("architecture")
        ):
            architecture_frame = (
                architecture_frame.sort_values(
                    "num_layers"
                )
            )

            plt.errorbar(
                architecture_frame[
                    "num_layers"
                ],
                architecture_frame[
                    "accuracy_delta_mean_pp"
                ],
                yerr=architecture_frame[
                    "accuracy_delta_ci95_pp"
                ],
                marker="o",
                capsize=3,
                label=architecture,
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
            f"(homophily={homophily:.1f})"
        )
        plt.xticks(range(2, 9))
        plt.legend()
        plt.tight_layout()

        output = (
            PLOT_DIR
            / (
                "pairnorm_delta_vs_depth_"
                f"h{int(homophily * 10):02d}.png"
            )
        )

        plt.savefig(
            output,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print("Saved:", output)


def plot_pairwise(
    model_summary: pd.DataFrame,
) -> None:
    for homophily, frame in model_summary.groupby(
        "target_homophily"
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
                    "hidden_pairwise_mean"
                ],
                yerr=model_frame[
                    "hidden_pairwise_ci95"
                ],
                marker="o",
                capsize=3,
                label=model,
            )

        plt.xlabel("Number of layers")
        plt.ylabel(
            "Final hidden pairwise "
            "cosine distance"
        )
        plt.title(
            "Representation separation "
            f"(homophily={homophily:.1f})"
        )
        plt.xticks(range(2, 9))
        plt.legend()
        plt.tight_layout()

        output = (
            PLOT_DIR
            / (
                "hidden_pairwise_vs_depth_"
                f"h{int(homophily * 10):02d}.png"
            )
        )

        plt.savefig(
            output,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print("Saved:", output)


def plot_parity(
    parity_summary: pd.DataFrame,
) -> None:
    for homophily, frame in parity_summary.groupby(
        "target_homophily"
    ):
        frame = frame.sort_values("model")

        plt.figure(figsize=(10, 6))

        positions = np.arange(len(frame))

        plt.bar(
            positions,
            frame[
                "parity_residual_mean_pp"
            ],
            yerr=frame[
                "parity_residual_ci95_pp"
            ],
            capsize=4,
        )

        plt.axhline(
            0.0,
            linewidth=1,
        )
        plt.xticks(
            positions,
            frame["model"],
            rotation=20,
        )
        plt.ylabel(
            "Odd depth minus neighboring "
            "even depths (pp)"
        )
        plt.title(
            "Local depth-parity effect "
            f"(homophily={homophily:.1f})"
        )
        plt.tight_layout()

        output = (
            PLOT_DIR
            / (
                "parity_residual_"
                f"h{int(homophily * 10):02d}.png"
            )
        )

        plt.savefig(
            output,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print("Saved:", output)


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
        analyze_depth_drop(runs)
    )
    parity_seed, parity_summary = (
        analyze_parity(runs)
    )
    correlations = analyze_correlations(
        paired
    )

    outputs = {
        "csbm_parity_run_level.csv":
            runs,
        "csbm_parity_pairnorm_paired.csv":
            paired,
        "csbm_parity_pairnorm_summary.csv":
            pair_summary,
        "csbm_parity_model_depth_summary.csv":
            model_summary,
        "csbm_parity_depth_seed.csv":
            depth_seed,
        "csbm_parity_depth_summary.csv":
            depth_summary,
        "csbm_parity_parity_seed.csv":
            parity_seed,
        "csbm_parity_parity_summary.csv":
            parity_summary,
        "csbm_parity_correlations.csv":
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
    plot_pairnorm(pair_summary)
    plot_pairwise(model_summary)
    plot_parity(parity_summary)

    print(
        "\n=== Paired PairNorm effects ==="
    )
    print(
        pair_summary[
            [
                "target_homophily",
                "architecture",
                "num_layers",
                "baseline_accuracy_mean_pct",
                "pairnorm_accuracy_mean_pct",
                "accuracy_delta_mean_pp",
                "accuracy_delta_ci95_pp",
                "positive_seeds",
                "negative_seeds",
                "paired_t_pvalue",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    print(
        "\n=== L2-to-L8 depth effects ==="
    )
    print(
        depth_summary[
            [
                "target_homophily",
                "architecture",
                "baseline_depth_drop_mean_pp",
                "pairnorm_depth_drop_mean_pp",
                "depth_drop_reduction_mean_pp",
                "depth_drop_reduction_ci95_pp",
                "positive_reduction_seeds",
                "paired_t_pvalue",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    print(
        "\n=== Odd-depth parity effects ==="
    )
    print(
        parity_summary[
            [
                "target_homophily",
                "model",
                "parity_residual_mean_pp",
                "parity_residual_ci95_pp",
                "negative_residual_seeds",
                "paired_t_pvalue",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    print(
        "\n=== Delta correlations ==="
    )
    print(
        correlations[
            correlations["group"].isin(
                [
                    "All paired runs",
                    "Homophily 0.1",
                    "Homophily 0.5",
                ]
            )
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )


if __name__ == "__main__":
    main()
