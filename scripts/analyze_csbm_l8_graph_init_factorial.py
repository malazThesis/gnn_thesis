from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


RUN_DIR = Path(
    "runs/csbm_l8_graph_init_factorial_fs050"
)

RUN_OUTPUT = (
    RUN_DIR
    / "csbm_l8_graph_init_run_level.csv"
)

GRAPH_OUTPUT = (
    RUN_DIR
    / "csbm_l8_graph_effects.csv"
)

INIT_OUTPUT = (
    RUN_DIR
    / "csbm_l8_initialization_effects.csv"
)

PAIRED_OUTPUT = (
    RUN_DIR
    / "csbm_l8_pairnorm_paired_effects.csv"
)

ANOVA_OUTPUT = (
    RUN_DIR
    / "csbm_l8_graph_init_variance_decomposition.csv"
)


def first_epoch_at_least(
    history: pd.DataFrame,
    column: str,
    threshold: float,
) -> float:
    selected = history[
        history[column] >= threshold
    ]

    if selected.empty:
        return math.nan

    return float(selected["epoch"].min())


def classify_outcome(
    final_test_acc: float,
) -> str:
    if final_test_acc >= 0.90:
        return "recovered"

    if final_test_acc <= 0.52:
        return "collapsed"

    return "partial"


def resolve_graph_seed(
    summary: dict,
    filename: str,
) -> int:
    if summary.get("graph_seed") is not None:
        return int(summary["graph_seed"])

    match = re.search(
        r"-G(\d+)_",
        filename,
    )

    if match:
        return int(match.group(1))

    data_path = str(
        summary.get(
            "data_path",
            summary.get("dataset_path", ""),
        )
    )

    match = re.search(
        r"seed(\d+)",
        data_path,
    )

    if match:
        return int(match.group(1))

    raise RuntimeError(
        f"Could not determine graph seed for {filename}"
    )


def load_runs() -> pd.DataFrame:
    summary_files = sorted(
        RUN_DIR.glob("*_summary.json")
    )

    if len(summary_files) != 50:
        raise RuntimeError(
            f"Expected 50 summaries, "
            f"found {len(summary_files)}"
        )

    rows = []

    for summary_path in summary_files:
        with summary_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            summary = json.load(handle)

        history_path = summary_path.with_name(
            summary_path.name.replace(
                "_summary.json",
                "_history.csv",
            )
        )

        history = pd.read_csv(history_path)
        history = history.sort_values("epoch")

        final = history.iloc[-1]
        best = history.loc[
            history["val_acc"].idxmax()
        ]

        model = str(summary["model"])
        graph_seed = resolve_graph_seed(
            summary,
            summary_path.name,
        )
        initialization_seed = int(
            summary["seed"]
        )

        final_test_acc = float(
            final["test_acc"]
        )

        rows.append(
            {
                "model": model,
                "uses_pairnorm":
                    model.endswith("PairNorm"),
                "graph_seed": graph_seed,
                "initialization_seed":
                    initialization_seed,
                "final_loss":
                    float(final["loss"]),
                "final_train_acc":
                    float(final["train_acc"]),
                "final_val_acc":
                    float(final["val_acc"]),
                "final_test_acc":
                    final_test_acc,
                "best_epoch":
                    int(best["epoch"]),
                "best_val_acc":
                    float(best["val_acc"]),
                "test_acc_at_best_val":
                    float(best["test_acc"]),
                "first_train_90":
                    first_epoch_at_least(
                        history,
                        "train_acc",
                        0.90,
                    ),
                "first_val_90":
                    first_epoch_at_least(
                        history,
                        "val_acc",
                        0.90,
                    ),
                "first_test_90":
                    first_epoch_at_least(
                        history,
                        "test_acc",
                        0.90,
                    ),
                "outcome":
                    classify_outcome(
                        final_test_acc
                    ),
                "collapsed":
                    final_test_acc <= 0.52,
                "recovered":
                    final_test_acc >= 0.90,
            }
        )

    frame = pd.DataFrame(rows)

    keys = [
        "model",
        "graph_seed",
        "initialization_seed",
    ]

    if frame.duplicated(keys).any():
        raise RuntimeError(
            "Duplicate graph/init/model combinations."
        )

    expected_models = {
        "GraphSAGE",
        "GraphSAGEPairNorm",
    }

    if set(frame["model"]) != expected_models:
        raise RuntimeError(
            f"Unexpected models: "
            f"{sorted(frame['model'].unique())}"
        )

    return frame.sort_values(keys).reset_index(
        drop=True
    )


def grouped_summary(
    runs: pd.DataFrame,
    grouping: str,
) -> pd.DataFrame:
    rows = []

    for keys, group in runs.groupby(
        ["model", grouping]
    ):
        rows.append(
            {
                "model": keys[0],
                grouping: keys[1],
                "n": len(group),
                "collapsed_runs": int(
                    group["collapsed"].sum()
                ),
                "recovered_runs": int(
                    group["recovered"].sum()
                ),
                "collapse_rate": float(
                    group["collapsed"].mean()
                ),
                "recovery_rate": float(
                    group["recovered"].mean()
                ),
                "final_test_acc_mean": float(
                    group["final_test_acc"].mean()
                ),
                "final_test_acc_std": float(
                    group["final_test_acc"].std(
                        ddof=1
                    )
                ),
                "test_at_best_val_mean": float(
                    group[
                        "test_acc_at_best_val"
                    ].mean()
                ),
                "first_test_90_mean": float(
                    group["first_test_90"].mean()
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["model", grouping]
    )


def paired_effects(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "graph_seed",
        "initialization_seed",
    ]

    baseline = runs[
        runs["model"] == "GraphSAGE"
    ][
        keys
        + [
            "final_test_acc",
            "test_acc_at_best_val",
            "recovered",
        ]
    ].rename(
        columns={
            "final_test_acc":
                "baseline_final_test_acc",
            "test_acc_at_best_val":
                "baseline_test_at_best_val",
            "recovered":
                "baseline_recovered",
        }
    )

    pairnorm = runs[
        runs["model"]
        == "GraphSAGEPairNorm"
    ][
        keys
        + [
            "final_test_acc",
            "test_acc_at_best_val",
            "recovered",
        ]
    ].rename(
        columns={
            "final_test_acc":
                "pairnorm_final_test_acc",
            "test_acc_at_best_val":
                "pairnorm_test_at_best_val",
            "recovered":
                "pairnorm_recovered",
        }
    )

    paired = baseline.merge(
        pairnorm,
        on=keys,
        validate="one_to_one",
    )

    paired["final_test_difference"] = (
        paired["pairnorm_final_test_acc"]
        - paired["baseline_final_test_acc"]
    )

    paired["best_test_difference"] = (
        paired["pairnorm_test_at_best_val"]
        - paired["baseline_test_at_best_val"]
    )

    return paired.sort_values(keys)


def additive_two_way(
    frame: pd.DataFrame,
    value_column: str,
) -> dict[str, float | str]:
    matrix = frame.pivot(
        index="graph_seed",
        columns="initialization_seed",
        values=value_column,
    )

    if matrix.isna().any().any():
        raise RuntimeError(
            f"Incomplete matrix for {value_column}"
        )

    values = matrix.to_numpy(
        dtype=float
    )

    number_graphs, number_inits = (
        values.shape
    )

    grand_mean = float(values.mean())
    graph_means = values.mean(axis=1)
    init_means = values.mean(axis=0)

    ss_graph = float(
        number_inits
        * np.square(
            graph_means - grand_mean
        ).sum()
    )

    ss_init = float(
        number_graphs
        * np.square(
            init_means - grand_mean
        ).sum()
    )

    residual = (
        values
        - graph_means[:, None]
        - init_means[None, :]
        + grand_mean
    )

    ss_residual = float(
        np.square(residual).sum()
    )

    df_graph = number_graphs - 1
    df_init = number_inits - 1
    df_residual = (
        df_graph
        * df_init
    )

    ms_graph = ss_graph / df_graph
    ms_init = ss_init / df_init
    ms_residual = (
        ss_residual / df_residual
    )

    if ms_residual > 0.0:
        f_graph = ms_graph / ms_residual
        f_init = ms_init / ms_residual

        p_graph = float(
            stats.f.sf(
                f_graph,
                df_graph,
                df_residual,
            )
        )

        p_init = float(
            stats.f.sf(
                f_init,
                df_init,
                df_residual,
            )
        )
    else:
        f_graph = math.nan
        f_init = math.nan
        p_graph = math.nan
        p_init = math.nan

    ss_total = (
        ss_graph
        + ss_init
        + ss_residual
    )

    return {
        "metric": value_column,
        "grand_mean": grand_mean,
        "ss_graph": ss_graph,
        "ss_initialization": ss_init,
        "ss_residual_interaction":
            ss_residual,
        "variance_fraction_graph":
            ss_graph / ss_total,
        "variance_fraction_initialization":
            ss_init / ss_total,
        "variance_fraction_interaction":
            ss_residual / ss_total,
        "f_graph": f_graph,
        "p_graph": p_graph,
        "f_initialization": f_init,
        "p_initialization": p_init,
    }


def print_matrix(
    runs: pd.DataFrame,
    model: str,
    value_column: str,
    title: str,
) -> None:
    matrix = (
        runs[
            runs["model"] == model
        ]
        .pivot(
            index="graph_seed",
            columns="initialization_seed",
            values=value_column,
        )
        .sort_index()
        .sort_index(axis=1)
    )

    print(f"\n=== {title} ===")

    print(
        matrix.to_string(
            float_format=lambda value:
                f"{value:.4f}",
        )
    )


def main() -> None:
    runs = load_runs()

    runs.to_csv(
        RUN_OUTPUT,
        index=False,
    )

    graph_summary = grouped_summary(
        runs,
        "graph_seed",
    )

    init_summary = grouped_summary(
        runs,
        "initialization_seed",
    )

    graph_summary.to_csv(
        GRAPH_OUTPUT,
        index=False,
    )

    init_summary.to_csv(
        INIT_OUTPUT,
        index=False,
    )

    paired = paired_effects(runs)

    paired.to_csv(
        PAIRED_OUTPUT,
        index=False,
    )

    baseline = runs[
        runs["model"] == "GraphSAGE"
    ].copy()

    anova = pd.DataFrame(
        [
            additive_two_way(
                baseline,
                "final_test_acc",
            ),
            additive_two_way(
                baseline,
                "test_acc_at_best_val",
            ),
            additive_two_way(
                baseline.assign(
                    recovered_numeric=(
                        baseline["recovered"]
                        .astype(float)
                    )
                ),
                "recovered_numeric",
            ),
        ]
    )

    anova.to_csv(
        ANOVA_OUTPUT,
        index=False,
    )

    print_matrix(
        runs,
        "GraphSAGE",
        "final_test_acc",
        "GraphSAGE final test accuracy",
    )

    print_matrix(
        runs,
        "GraphSAGEPairNorm",
        "final_test_acc",
        "PairNorm final test accuracy",
    )

    print_matrix(
        runs,
        "GraphSAGE",
        "recovered",
        "GraphSAGE recovery matrix",
    )

    print(
        "\n=== Effects by graph ==="
    )

    print(
        graph_summary.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print(
        "\n=== Effects by initialization ==="
    )

    print(
        init_summary.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    differences = paired[
        "final_test_difference"
    ]

    t_result = stats.ttest_1samp(
        differences,
        popmean=0.0,
    )

    try:
        wilcoxon = stats.wilcoxon(
            differences,
            alternative="two-sided",
        )

        wilcoxon_pvalue = float(
            wilcoxon.pvalue
        )
    except ValueError:
        wilcoxon_pvalue = math.nan

    print(
        "\n=== Paired PairNorm effect ==="
    )

    print(
        "Mean final-test difference:",
        f"{differences.mean():.8f}",
    )

    print(
        "Positive combinations:",
        int((differences > 0).sum()),
        "/",
        len(differences),
    )

    print(
        "Paired t-test p:",
        f"{float(t_result.pvalue):.10g}",
    )

    print(
        "Wilcoxon p:",
        f"{wilcoxon_pvalue:.10g}",
    )

    print(
        "\n=== Additive variance decomposition ==="
    )

    print(
        anova.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.8f}",
        )
    )

    print("\nSaved:", RUN_OUTPUT)
    print("Saved:", GRAPH_OUTPUT)
    print("Saved:", INIT_OUTPUT)
    print("Saved:", PAIRED_OUTPUT)
    print("Saved:", ANOVA_OUTPUT)


if __name__ == "__main__":
    main()
