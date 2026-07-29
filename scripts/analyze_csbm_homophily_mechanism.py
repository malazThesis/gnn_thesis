from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RUN_DIR = Path(
    "runs/csbm_homophily_mechanism"
)
PLOT_DIR = Path(
    "plots/csbm_homophily_mechanism"
)

RUN_SUMMARY_CSV = (
    RUN_DIR / "csbm_homophily_run_summary.csv"
)
PAIRNORM_CSV = (
    RUN_DIR / "csbm_homophily_pairnorm_deltas.csv"
)
DEPTH_CSV = (
    RUN_DIR / "csbm_homophily_depth_effects.csv"
)
CORRELATION_CSV = (
    RUN_DIR / "csbm_homophily_correlations.csv"
)


METRIC_COLUMNS = [
    "mean_pairwise_cosine_distance",
    "mean_edge_cosine_distance",
    "normalized_dirichlet_energy",
    "effective_rank",
    "effective_rank_ratio",
    "mean_embedding_norm",
    "mean_feature_variance",
]


def module_number(name: object) -> int:
    match = re.search(
        r"\.(\d+)(?:#\d+)?$",
        str(name),
    )

    return int(match.group(1)) if match else -1


def read_summary(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def select_hidden_row(
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


def select_output_row(
    frame: pd.DataFrame,
) -> pd.Series:
    candidates = frame[
        (
            frame["layer_name"].notna()
        )
        & (
            frame["layer_name"]
            .astype(str)
            .str.startswith("convs.")
        )
    ].copy()

    if candidates.empty:
        raise RuntimeError(
            "No output convolution found."
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


def add_stage_metrics(
    target: dict,
    *,
    prefix: str,
    row: pd.Series,
) -> None:
    target[f"{prefix}_layer_name"] = (
        row["layer_name"]
    )
    target[f"{prefix}_embedding_dim"] = int(
        row["embedding_dim"]
    )

    for column in METRIC_COLUMNS:
        target[f"{prefix}_{column}"] = float(
            row[column]
        )


def load_run_table() -> pd.DataFrame:
    summary_files = sorted(
        RUN_DIR.glob("*_summary.json")
    )

    if len(summary_files) != 36:
        raise RuntimeError(
            f"Expected 36 summaries, "
            f"found {len(summary_files)}"
        )

    rows = []

    for summary_path in summary_files:
        summary = read_summary(summary_path)

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

        best_epoch = int(summary["best_epoch"])
        final_epoch = max(recorded_epochs)

        nearest_best_epoch = min(
            recorded_epochs,
            key=lambda epoch: abs(
                epoch - best_epoch
            ),
        )

        row = {
            "dataset": summary["dataset"],
            "model": summary["model"],
            "architecture": (
                summary["model"]
                .replace("PairNorm", "")
            ),
            "uses_pairnorm": (
                summary["model"]
                .endswith("PairNorm")
            ),
            "num_layers": int(
                summary["num_layers"]
            ),
            "hidden_channels": int(
                summary["hidden_channels"]
            ),
            "seed": int(summary["seed"]),
            "split_idx": int(
                summary["split_idx"]
            ),
            "graph_seed": int(
                summary["graph_seed"]
            ),
            "target_homophily": float(
                summary["target_homophily"]
            ),
            "realized_homophily": float(
                summary["realized_homophily"]
            ),
            "realized_average_degree": float(
                summary[
                    "realized_average_degree"
                ]
            ),
            "feature_signal": float(
                summary["feature_signal"]
            ),
            "best_epoch": best_epoch,
            "nearest_best_metric_epoch":
                nearest_best_epoch,
            "final_metric_epoch": final_epoch,
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

        for checkpoint_name, epoch in (
            ("best", nearest_best_epoch),
            ("final", final_epoch),
        ):
            checkpoint = metrics[
                metrics["epoch"] == epoch
            ].copy()

            hidden = select_hidden_row(
                checkpoint,
                model=summary["model"],
                hidden_channels=int(
                    summary["hidden_channels"]
                ),
            )
            output = select_output_row(
                checkpoint
            )

            add_stage_metrics(
                row,
                prefix=(
                    f"{checkpoint_name}_hidden"
                ),
                row=hidden,
            )
            add_stage_metrics(
                row,
                prefix=(
                    f"{checkpoint_name}_output"
                ),
                row=output,
            )

        rows.append(row)

    result = pd.DataFrame(rows)

    return result.sort_values(
        [
            "target_homophily",
            "architecture",
            "uses_pairnorm",
            "num_layers",
        ]
    ).reset_index(drop=True)


def build_pairnorm_table(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "dataset",
        "architecture",
        "num_layers",
        "hidden_channels",
        "seed",
        "split_idx",
        "graph_seed",
        "target_homophily",
        "realized_homophily",
    ]

    value_columns = [
        "best_epoch",
        "best_test_acc",
        "best_val_acc",
        "best_hidden_mean_pairwise_cosine_distance",
        "best_hidden_mean_edge_cosine_distance",
        "best_hidden_normalized_dirichlet_energy",
        "best_hidden_effective_rank_ratio",
        "final_hidden_mean_pairwise_cosine_distance",
        "final_hidden_mean_edge_cosine_distance",
        "final_hidden_normalized_dirichlet_energy",
        "final_hidden_effective_rank_ratio",
        "final_output_mean_pairwise_cosine_distance",
        "final_output_effective_rank_ratio",
    ]

    baseline = runs[
        ~runs["uses_pairnorm"]
    ][keys + value_columns].copy()

    pairnorm = runs[
        runs["uses_pairnorm"]
    ][keys + value_columns].copy()

    baseline = baseline.rename(
        columns={
            column: f"baseline_{column}"
            for column in value_columns
        }
    )
    pairnorm = pairnorm.rename(
        columns={
            column: f"pairnorm_{column}"
            for column in value_columns
        }
    )

    paired = baseline.merge(
        pairnorm,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    if len(paired) != 18:
        raise RuntimeError(
            f"Expected 18 paired rows, "
            f"found {len(paired)}"
        )

    for column in value_columns:
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
            "architecture",
            "num_layers",
        ]
    ).reset_index(drop=True)


def build_depth_table(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    group_columns = [
        "target_homophily",
        "realized_homophily",
        "architecture",
        "uses_pairnorm",
    ]

    for keys, group in runs.groupby(
        group_columns
    ):
        by_depth = group.set_index(
            "num_layers"
        )

        if not {2, 4, 8}.issubset(
            by_depth.index
        ):
            raise RuntimeError(
                f"Missing depth for group {keys}"
            )

        accuracy_l2 = float(
            by_depth.loc[2, "best_test_acc"]
        )
        accuracy_l4 = float(
            by_depth.loc[4, "best_test_acc"]
        )
        accuracy_l8 = float(
            by_depth.loc[8, "best_test_acc"]
        )

        rows.append(
            {
                "target_homophily": keys[0],
                "realized_homophily": keys[1],
                "architecture": keys[2],
                "uses_pairnorm": keys[3],
                "accuracy_l2": accuracy_l2,
                "accuracy_l4": accuracy_l4,
                "accuracy_l8": accuracy_l8,
                "depth_drop_l2_to_l4_pp": (
                    accuracy_l2
                    - accuracy_l4
                ) * 100.0,
                "depth_drop_l4_to_l8_pp": (
                    accuracy_l4
                    - accuracy_l8
                ) * 100.0,
                "depth_drop_l2_to_l8_pp": (
                    accuracy_l2
                    - accuracy_l8
                ) * 100.0,
                "final_hidden_rank_ratio_l2":
                    float(
                        by_depth.loc[
                            2,
                            (
                                "final_hidden_"
                                "effective_rank_ratio"
                            ),
                        ]
                    ),
                "final_hidden_rank_ratio_l8":
                    float(
                        by_depth.loc[
                            8,
                            (
                                "final_hidden_"
                                "effective_rank_ratio"
                            ),
                        ]
                    ),
            }
        )

    depth = pd.DataFrame(rows)

    keys = [
        "target_homophily",
        "realized_homophily",
        "architecture",
    ]

    baseline = depth[
        ~depth["uses_pairnorm"]
    ].drop(
        columns=["uses_pairnorm"]
    )

    pairnorm = depth[
        depth["uses_pairnorm"]
    ].drop(
        columns=["uses_pairnorm"]
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

    result = baseline.merge(
        pairnorm,
        on=keys,
        validate="one_to_one",
    )

    result[
        "pairnorm_depth_drop_reduction_pp"
    ] = (
        result[
            "baseline_depth_drop_l2_to_l8_pp"
        ]
        - result[
            "pairnorm_depth_drop_l2_to_l8_pp"
        ]
    )

    return result.sort_values(
        [
            "target_homophily",
            "architecture",
        ]
    ).reset_index(drop=True)


def build_correlations(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    variables = {
        "final_hidden_pairwise_distance": (
            "final_hidden_"
            "mean_pairwise_cosine_distance"
        ),
        "final_hidden_edge_distance": (
            "final_hidden_"
            "mean_edge_cosine_distance"
        ),
        "final_hidden_normalized_dirichlet": (
            "final_hidden_"
            "normalized_dirichlet_energy"
        ),
        "final_hidden_effective_rank_ratio": (
            "final_hidden_"
            "effective_rank_ratio"
        ),
        "best_hidden_effective_rank_ratio": (
            "best_hidden_"
            "effective_rank_ratio"
        ),
    }

    rows = []

    groups = [
        ("All runs", runs),
    ]

    for model, frame in runs.groupby("model"):
        groups.append((model, frame))

    for architecture, frame in runs.groupby(
        "architecture"
    ):
        groups.append(
            (f"Architecture: {architecture}", frame)
        )

    for label, frame in groups:
        for metric_label, column in variables.items():
            coefficient = (
                frame[
                    ["best_test_acc", column]
                ]
                .corr(method="spearman")
                .iloc[0, 1]
            )

            rows.append(
                {
                    "group": label,
                    "n": len(frame),
                    "metric": metric_label,
                    "spearman_correlation":
                        coefficient,
                }
            )

    return pd.DataFrame(rows)


def plot_accuracy_by_homophily(
    runs: pd.DataFrame,
) -> None:
    for homophily, frame in runs.groupby(
        "target_homophily"
    ):
        plt.figure(figsize=(9, 6))

        for model, model_frame in frame.groupby(
            "model"
        ):
            model_frame = model_frame.sort_values(
                "num_layers"
            )

            plt.plot(
                model_frame["num_layers"],
                model_frame["best_test_acc"]
                * 100.0,
                marker="o",
                label=model,
            )

        plt.xlabel("Number of layers")
        plt.ylabel("Best test accuracy (%)")
        plt.title(
            "CSBM accuracy by depth "
            f"(target homophily={homophily:.1f})"
        )
        plt.xticks([2, 4, 8])
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


def plot_pairnorm_delta(
    paired: pd.DataFrame,
) -> None:
    for homophily, frame in paired.groupby(
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

            plt.plot(
                architecture_frame["num_layers"],
                architecture_frame[
                    "accuracy_delta_pp"
                ],
                marker="o",
                label=architecture,
            )

        plt.axhline(
            0.0,
            linewidth=1,
        )
        plt.xlabel("Number of layers")
        plt.ylabel(
            "PairNorm accuracy effect "
            "(percentage points)"
        )
        plt.title(
            "Paired PairNorm effect "
            f"(target homophily={homophily:.1f})"
        )
        plt.xticks([2, 4, 8])
        plt.legend()
        plt.tight_layout()

        output = (
            PLOT_DIR
            / (
                "pairnorm_accuracy_delta_"
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


def plot_hidden_rank(
    runs: pd.DataFrame,
) -> None:
    for homophily, frame in runs.groupby(
        "target_homophily"
    ):
        plt.figure(figsize=(9, 6))

        for model, model_frame in frame.groupby(
            "model"
        ):
            model_frame = model_frame.sort_values(
                "num_layers"
            )

            plt.plot(
                model_frame["num_layers"],
                model_frame[
                    (
                        "final_hidden_"
                        "effective_rank_ratio"
                    )
                ],
                marker="o",
                label=model,
            )

        plt.xlabel("Number of layers")
        plt.ylabel(
            "Final hidden effective-rank ratio"
        )
        plt.title(
            "Hidden dimensional diversity "
            f"(target homophily={homophily:.1f})"
        )
        plt.xticks([2, 4, 8])
        plt.legend()
        plt.tight_layout()

        output = (
            PLOT_DIR
            / (
                "hidden_rank_ratio_vs_depth_"
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


def plot_accuracy_rank_scatter(
    runs: pd.DataFrame,
) -> None:
    plt.figure(figsize=(9, 6))

    for model, frame in runs.groupby("model"):
        plt.scatter(
            frame[
                (
                    "final_hidden_"
                    "effective_rank_ratio"
                )
            ],
            frame["best_test_acc"] * 100.0,
            label=model,
        )

    plt.xlabel(
        "Final hidden effective-rank ratio"
    )
    plt.ylabel("Best test accuracy (%)")
    plt.title(
        "Accuracy and hidden representation rank"
    )
    plt.legend()
    plt.tight_layout()

    output = (
        PLOT_DIR
        / "accuracy_vs_hidden_rank_ratio.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("Saved:", output)


def print_key_results(
    paired: pd.DataFrame,
    depth: pd.DataFrame,
    correlations: pd.DataFrame,
) -> None:
    display_columns = [
        "target_homophily",
        "architecture",
        "num_layers",
        "baseline_best_test_acc",
        "pairnorm_best_test_acc",
        "accuracy_delta_pp",
        (
            "delta_final_hidden_"
            "effective_rank_ratio"
        ),
        (
            "delta_final_hidden_"
            "mean_pairwise_cosine_distance"
        ),
    ]

    print("\nPaired PairNorm effects:")
    print(
        paired[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    depth_columns = [
        "target_homophily",
        "architecture",
        "baseline_depth_drop_l2_to_l8_pp",
        "pairnorm_depth_drop_l2_to_l8_pp",
        "pairnorm_depth_drop_reduction_pp",
    ]

    print("\nDepth effects:")
    print(
        depth[depth_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nAll-run Spearman correlations:")
    print(
        correlations[
            correlations["group"] == "All runs"
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    positive = int(
        (paired["accuracy_delta_pp"] > 0).sum()
    )
    total = len(paired)

    print(
        "\nPairNorm accuracy improvements:",
        f"{positive}/{total}",
    )

    best = paired.nlargest(
        5,
        "accuracy_delta_pp",
    )[
        [
            "target_homophily",
            "architecture",
            "num_layers",
            "accuracy_delta_pp",
        ]
    ]

    worst = paired.nsmallest(
        5,
        "accuracy_delta_pp",
    )[
        [
            "target_homophily",
            "architecture",
            "num_layers",
            "accuracy_delta_pp",
        ]
    ]

    print("\nLargest PairNorm improvements:")
    print(
        best.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nLargest PairNorm degradations:")
    print(
        worst.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


def main() -> None:
    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    runs = load_run_table()
    paired = build_pairnorm_table(runs)
    depth = build_depth_table(runs)
    correlations = build_correlations(runs)

    runs.to_csv(
        RUN_SUMMARY_CSV,
        index=False,
    )
    paired.to_csv(
        PAIRNORM_CSV,
        index=False,
    )
    depth.to_csv(
        DEPTH_CSV,
        index=False,
    )
    correlations.to_csv(
        CORRELATION_CSV,
        index=False,
    )

    plot_accuracy_by_homophily(runs)
    plot_pairnorm_delta(paired)
    plot_hidden_rank(runs)
    plot_accuracy_rank_scatter(runs)

    print_key_results(
        paired,
        depth,
        correlations,
    )

    print("\nSaved:")
    print(RUN_SUMMARY_CSV)
    print(PAIRNORM_CSV)
    print(DEPTH_CSV)
    print(CORRELATION_CSV)


if __name__ == "__main__":
    main()
