from __future__ import annotations

import itertools
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ABLATION_DIR = Path(
    "runs/csbm_l8_pairnorm_component_ablation_fs050"
)

BASELINE_DIR = Path(
    "runs/csbm_l8_lr_sensitivity_fs050"
)

PAIRNORM_DIR = Path(
    "runs/csbm_l8_pairnorm_lr_sensitivity_fs050"
)

PAIRNORM_EXTREME_DIR = Path(
    "runs/csbm_l8_pairnorm_extreme_lr_fs050"
)

PLOT_DIR = Path(
    "plots/csbm_l8_pairnorm_component_ablation"
)

RUN_OUTPUT = (
    ABLATION_DIR
    / "csbm_l8_component_ablation_run_level.csv"
)

SUMMARY_OUTPUT = (
    ABLATION_DIR
    / "csbm_l8_component_ablation_summary.csv"
)

PAIRED_OUTPUT = (
    ABLATION_DIR
    / "csbm_l8_component_ablation_paired_tests.csv"
)

METRIC_RUN_OUTPUT = (
    ABLATION_DIR
    / "csbm_l8_component_ablation_metric_run_level.csv"
)

METRIC_SUMMARY_OUTPUT = (
    ABLATION_DIR
    / "csbm_l8_component_ablation_metric_summary.csv"
)

METRIC_EPOCHS = {
    0,
    1,
    8,
    32,
    200,
}

MODEL_ORDER = [
    "GraphSAGE",
    "GraphSAGECenterNorm",
    "GraphSAGEScaleNorm",
    "GraphSAGEPairNorm",
]


def graph_seed_from_summary(
    summary: dict,
    filename: str,
) -> int:
    dataset = str(
        summary.get("dataset", "")
    )

    match = re.search(
        r"-G(\d+)(?:-|$)",
        dataset,
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

    match = re.search(
        r"-G(\d+)_",
        filename,
    )

    if match:
        return int(match.group(1))

    raise RuntimeError(
        f"Cannot resolve graph seed: {filename}"
    )


def learning_rate_from_summary(
    summary: dict,
) -> float:
    for key in [
        "lr",
        "learning_rate",
    ]:
        if summary.get(key) is not None:
            return float(summary[key])

    dataset = str(
        summary.get("dataset", "")
    )

    patterns = [
        r"LR(\d+p\d+)",
        r"PNLR(\d+p\d+)",
        r"PNEXT(\d+p\d+)",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            dataset,
        )

        if match:
            return float(
                match.group(1).replace(
                    "p",
                    ".",
                )
            )

    raise RuntimeError(
        f"Cannot resolve learning rate: {dataset}"
    )


def first_epoch_at_least(
    history: pd.DataFrame,
    column: str,
    threshold: float,
) -> float:
    selected = history[
        np.isfinite(history[column])
        & (
            history[column]
            >= threshold
        )
    ]

    if selected.empty:
        return math.nan

    return float(
        selected["epoch"].min()
    )


def module_number(
    layer_name: object,
) -> int:
    match = re.search(
        r"\.(\d+)(?:#\d+)?$",
        str(layer_name),
    )

    return (
        int(match.group(1))
        if match
        else -1
    )


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


def load_runs_from_directory(
    directory: Path,
    *,
    allowed_models: set[str],
    allowed_rates: set[float],
) -> pd.DataFrame:
    rows = []

    for summary_path in sorted(
        directory.glob("*_summary.json")
    ):
        with summary_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            summary = json.load(handle)

        model = str(
            summary.get("model", "")
        )

        if model not in allowed_models:
            continue

        learning_rate = (
            learning_rate_from_summary(
                summary
            )
        )

        if not any(
            np.isclose(
                learning_rate,
                candidate,
            )
            for candidate in allowed_rates
        ):
            continue

        history_path = summary_path.with_name(
            summary_path.name.replace(
                "_summary.json",
                "_history.csv",
            )
        )

        metric_path = summary_path.with_name(
            summary_path.name.replace(
                "_summary.json",
                "_oversmoothing.csv",
            )
        )

        history = pd.read_csv(
            history_path
        ).sort_values("epoch")

        numeric_columns = [
            "loss",
            "train_acc",
            "val_acc",
            "test_acc",
        ]

        numerical_failure = bool(
            not np.isfinite(
                history[numeric_columns]
                .to_numpy(dtype=float)
            ).all()
        )

        final = history.iloc[-1]

        valid_checkpoints = history[
            np.isfinite(
                history["val_acc"]
            )
            & np.isfinite(
                history["test_acc"]
            )
        ]

        if valid_checkpoints.empty:
            best_epoch = math.nan
            best_val = math.nan
            test_at_best = math.nan
        else:
            best = valid_checkpoints.loc[
                valid_checkpoints[
                    "val_acc"
                ].idxmax()
            ]

            best_epoch = float(
                best["epoch"]
            )

            best_val = float(
                best["val_acc"]
            )

            test_at_best = float(
                best["test_acc"]
            )

        final_test = float(
            final["test_acc"]
        )

        final_recovered = bool(
            np.isfinite(final_test)
            and final_test >= 0.90
        )

        checkpoint_recovered = bool(
            np.isfinite(test_at_best)
            and test_at_best >= 0.90
        )

        collapsed = bool(
            np.isfinite(final_test)
            and final_test <= 0.52
        )

        partial = bool(
            not numerical_failure
            and not final_recovered
            and not collapsed
        )

        rows.append(
            {
                "model":
                    model,
                "graph_seed":
                    graph_seed_from_summary(
                        summary,
                        summary_path.name,
                    ),
                "initialization_seed":
                    int(summary["seed"]),
                "learning_rate":
                    learning_rate,
                "final_loss":
                    float(final["loss"]),
                "final_train_acc":
                    float(final["train_acc"]),
                "final_val_acc":
                    float(final["val_acc"]),
                "final_test_acc":
                    final_test,
                "best_epoch":
                    best_epoch,
                "best_val_acc":
                    best_val,
                "test_acc_at_best_val":
                    test_at_best,
                "first_test_90":
                    first_epoch_at_least(
                        history,
                        "test_acc",
                        0.90,
                    ),
                "final_recovered":
                    final_recovered,
                "checkpoint_recovered":
                    checkpoint_recovered,
                "collapsed":
                    collapsed,
                "partial":
                    partial,
                "numerical_failure":
                    numerical_failure,
                "summary_path":
                    str(summary_path),
                "metric_path":
                    str(metric_path),
            }
        )

    return pd.DataFrame(rows)


def load_all_runs() -> pd.DataFrame:
    frames = [
        load_runs_from_directory(
            ABLATION_DIR,
            allowed_models={
                "GraphSAGECenterNorm",
                "GraphSAGEScaleNorm",
            },
            allowed_rates={
                0.03,
                0.50,
            },
        ),
        load_runs_from_directory(
            BASELINE_DIR,
            allowed_models={
                "GraphSAGE",
            },
            allowed_rates={
                0.03,
            },
        ),
        load_runs_from_directory(
            PAIRNORM_DIR,
            allowed_models={
                "GraphSAGEPairNorm",
            },
            allowed_rates={
                0.03,
            },
        ),
        load_runs_from_directory(
            PAIRNORM_EXTREME_DIR,
            allowed_models={
                "GraphSAGEPairNorm",
            },
            allowed_rates={
                0.50,
            },
        ),
    ]

    runs = pd.concat(
        frames,
        ignore_index=True,
    )

    expected = {
        ("GraphSAGE", 0.03): 25,
        ("GraphSAGECenterNorm", 0.03): 25,
        ("GraphSAGEScaleNorm", 0.03): 25,
        ("GraphSAGEPairNorm", 0.03): 25,
        ("GraphSAGECenterNorm", 0.50): 25,
        ("GraphSAGEScaleNorm", 0.50): 25,
        ("GraphSAGEPairNorm", 0.50): 25,
    }

    observed = (
        runs.groupby(
            [
                "model",
                "learning_rate",
            ]
        )
        .size()
        .to_dict()
    )

    for key, expected_count in expected.items():
        count = int(
            observed.get(key, 0)
        )

        if count != expected_count:
            raise RuntimeError(
                f"{key}: expected "
                f"{expected_count}, found {count}"
            )

    keys = [
        "model",
        "graph_seed",
        "initialization_seed",
        "learning_rate",
    ]

    if runs.duplicated(keys).any():
        raise RuntimeError(
            "Duplicate model/graph/init/lr rows."
        )

    return runs.sort_values(keys)


def summarize_runs(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        model,
        learning_rate,
    ), group in runs.groupby(
        [
            "model",
            "learning_rate",
        ]
    ):
        rows.append(
            {
                "model":
                    model,
                "learning_rate":
                    learning_rate,
                "n":
                    len(group),
                "final_recovered_runs":
                    int(
                        group[
                            "final_recovered"
                        ].sum()
                    ),
                "checkpoint_recovered_runs":
                    int(
                        group[
                            "checkpoint_recovered"
                        ].sum()
                    ),
                "collapsed_runs":
                    int(
                        group["collapsed"].sum()
                    ),
                "partial_runs":
                    int(
                        group["partial"].sum()
                    ),
                "numerical_failure_runs":
                    int(
                        group[
                            "numerical_failure"
                        ].sum()
                    ),
                "final_test_acc_mean":
                    float(
                        group[
                            "final_test_acc"
                        ].mean()
                    ),
                "final_test_acc_std":
                    float(
                        group[
                            "final_test_acc"
                        ].std(ddof=1)
                    ),
                "final_test_acc_min":
                    float(
                        group[
                            "final_test_acc"
                        ].min()
                    ),
                "test_at_best_val_mean":
                    float(
                        group[
                            "test_acc_at_best_val"
                        ].mean()
                    ),
                "first_test_90_mean":
                    float(
                        group[
                            "first_test_90"
                        ].mean()
                    ),
                "first_test_90_median":
                    float(
                        group[
                            "first_test_90"
                        ].median()
                    ),
            }
        )

    result = pd.DataFrame(rows)

    result["model_order"] = result[
        "model"
    ].map(
        {
            model: index
            for index, model in enumerate(
                MODEL_ORDER
            )
        }
    )

    return (
        result.sort_values(
            [
                "learning_rate",
                "model_order",
            ]
        )
        .drop(
            columns="model_order"
        )
    )


def paired_tests(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "graph_seed",
        "initialization_seed",
    ]

    rows = []

    for learning_rate, rate_group in runs.groupby(
        "learning_rate"
    ):
        available_models = [
            model
            for model in MODEL_ORDER
            if model in set(
                rate_group["model"]
            )
        ]

        for model_a, model_b in itertools.combinations(
            available_models,
            2,
        ):
            left = rate_group[
                rate_group["model"] == model_a
            ][
                keys
                + [
                    "final_test_acc",
                    "final_recovered",
                    "first_test_90",
                ]
            ].rename(
                columns={
                    "final_test_acc":
                        "accuracy_a",
                    "final_recovered":
                        "recovered_a",
                    "first_test_90":
                        "first90_a",
                }
            )

            right = rate_group[
                rate_group["model"] == model_b
            ][
                keys
                + [
                    "final_test_acc",
                    "final_recovered",
                    "first_test_90",
                ]
            ].rename(
                columns={
                    "final_test_acc":
                        "accuracy_b",
                    "final_recovered":
                        "recovered_b",
                    "first_test_90":
                        "first90_b",
                }
            )

            paired = left.merge(
                right,
                on=keys,
                validate="one_to_one",
            )

            differences = (
                paired["accuracy_b"]
                - paired["accuracy_a"]
            )

            finite = differences[
                np.isfinite(differences)
            ]

            try:
                wilcoxon_pvalue = float(
                    stats.wilcoxon(
                        finite,
                        alternative="two-sided",
                    ).pvalue
                )
            except ValueError:
                wilcoxon_pvalue = math.nan

            b_only = int(
                (
                    paired["recovered_b"]
                    & ~paired["recovered_a"]
                ).sum()
            )

            a_only = int(
                (
                    paired["recovered_a"]
                    & ~paired["recovered_b"]
                ).sum()
            )

            discordant = (
                b_only + a_only
            )

            mcnemar_pvalue = (
                float(
                    stats.binomtest(
                        b_only,
                        n=discordant,
                        p=0.5,
                    ).pvalue
                )
                if discordant
                else 1.0
            )

            both_recovered = paired[
                paired["recovered_a"]
                & paired["recovered_b"]
                & np.isfinite(
                    paired["first90_a"]
                )
                & np.isfinite(
                    paired["first90_b"]
                )
            ]

            speed_change = (
                both_recovered["first90_b"]
                - both_recovered["first90_a"]
            )

            rows.append(
                {
                    "learning_rate":
                        learning_rate,
                    "model_a":
                        model_a,
                    "model_b":
                        model_b,
                    "mean_accuracy_change_b_minus_a":
                        float(
                            finite.mean()
                        ),
                    "b_only_recovered":
                        b_only,
                    "a_only_recovered":
                        a_only,
                    "both_recovered_n":
                        len(both_recovered),
                    "mean_first90_change_b_minus_a":
                        (
                            float(
                                speed_change.mean()
                            )
                            if len(speed_change)
                            else math.nan
                        ),
                    "wilcoxon_pvalue":
                        wilcoxon_pvalue,
                    "mcnemar_exact_pvalue":
                        mcnemar_pvalue,
                }
            )

    result = pd.DataFrame(rows)

    result[
        "wilcoxon_holm_pvalue"
    ] = holm_adjust(
        result[
            "wilcoxon_pvalue"
        ].tolist()
    )

    result[
        "mcnemar_holm_pvalue"
    ] = holm_adjust(
        result[
            "mcnemar_exact_pvalue"
        ].tolist()
    )

    return result


def select_final_hidden(
    frame: pd.DataFrame,
    model: str,
    epoch: int,
) -> pd.Series:
    selected = frame[
        frame["epoch"] == epoch
    ].copy()

    selected["module_number"] = (
        selected[
            "layer_name"
        ].map(module_number)
    )

    if model == "GraphSAGE":
        convs = selected[
            selected[
                "layer_name"
            ]
            .astype(str)
            .str.startswith("convs.")
        ].sort_values(
            "module_number"
        )

        if len(convs) < 2:
            raise RuntimeError(
                "Baseline hidden layer not found."
            )

        return convs.iloc[-2]

    normalizers = selected[
        selected[
            "layer_name"
        ]
        .astype(str)
        .str.startswith("pns.")
    ].sort_values(
        "module_number"
    )

    if normalizers.empty:
        raise RuntimeError(
            f"No normalization layer for {model}"
        )

    return normalizers.iloc[-1]


def extract_metrics(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    metric_columns = [
        "mean_pairwise_cosine_distance",
        "mean_edge_cosine_distance",
        "normalized_dirichlet_energy",
        "effective_rank_ratio",
        "mean_embedding_norm",
        "class_separation_margin",
        "class_centroid_cosine_distance",
        "fisher_discriminant_ratio",
    ]

    for record in runs.to_dict(
        orient="records"
    ):
        frame = pd.read_csv(
            record["metric_path"]
        )

        available_epochs = set(
            frame["epoch"]
            .astype(int)
            .tolist()
        )

        for epoch in sorted(
            METRIC_EPOCHS
            & available_epochs
        ):
            hidden = select_final_hidden(
                frame,
                record["model"],
                epoch,
            )

            row = {
                "model":
                    record["model"],
                "graph_seed":
                    record["graph_seed"],
                "initialization_seed":
                    record[
                        "initialization_seed"
                    ],
                "learning_rate":
                    record["learning_rate"],
                "epoch":
                    epoch,
                "layer_name":
                    hidden["layer_name"],
            }

            for column in metric_columns:
                row[column] = float(
                    hidden[column]
                )

            rows.append(row)

    return pd.DataFrame(rows)


def summarize_metrics(
    metric_runs: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = [
        "mean_pairwise_cosine_distance",
        "mean_edge_cosine_distance",
        "normalized_dirichlet_energy",
        "effective_rank_ratio",
        "mean_embedding_norm",
        "class_separation_margin",
        "class_centroid_cosine_distance",
        "fisher_discriminant_ratio",
    ]

    return (
        metric_runs.groupby(
            [
                "model",
                "learning_rate",
                "epoch",
            ],
            as_index=False,
        )[metric_columns]
        .agg(
            ["mean", "std"]
        )
        .reset_index()
    )


def make_plots(
    summary: pd.DataFrame,
) -> None:
    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for learning_rate in sorted(
        summary["learning_rate"].unique()
    ):
        selected = summary[
            np.isclose(
                summary["learning_rate"],
                learning_rate,
            )
        ].copy()

        selected["model_order"] = (
            selected["model"].map(
                {
                    model: index
                    for index, model
                    in enumerate(MODEL_ORDER)
                }
            )
        )

        selected = selected.sort_values(
            "model_order"
        )

        labels = [
            model.replace(
                "GraphSAGE",
                "",
            )
            or "Baseline"
            for model in selected["model"]
        ]

        plt.figure(
            figsize=(9, 6)
        )

        plt.bar(
            labels,
            selected[
                "final_recovered_runs"
            ] / selected["n"],
        )

        plt.ylim(0.0, 1.05)
        plt.ylabel("Final recovery rate")
        plt.title(
            "Component ablation, "
            f"learning rate={learning_rate:g}"
        )
        plt.tight_layout()

        path = (
            PLOT_DIR
            / (
                "recovery_rate_lr_"
                f"{learning_rate:g}.png"
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
    runs = load_all_runs()
    summary = summarize_runs(runs)
    paired = paired_tests(runs)

    metric_runs = extract_metrics(
        runs
    )

    metric_summary = summarize_metrics(
        metric_runs
    )

    runs.to_csv(
        RUN_OUTPUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    paired.to_csv(
        PAIRED_OUTPUT,
        index=False,
    )

    metric_runs.to_csv(
        METRIC_RUN_OUTPUT,
        index=False,
    )

    metric_summary.to_csv(
        METRIC_SUMMARY_OUTPUT,
        index=False,
    )

    make_plots(summary)

    print("\n=== ABLATIONSERGEBNISSE ===")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.10f}",
        )
    )

    print("\n=== GEPAARTE VERGLEICHE ===")

    print(
        paired.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.10f}",
        )
    )

    print(
        "\n=== REPRÄSENTATIONSMETRIKEN "
        "EPOCH 1 UND 200 ==="
    )

    compact_metrics = (
        metric_runs[
            metric_runs["epoch"].isin(
                [1, 200]
            )
        ]
        .groupby(
            [
                "model",
                "learning_rate",
                "epoch",
            ],
            as_index=False,
        )
        .agg(
            pairwise_mean=(
                "mean_pairwise_cosine_distance",
                "mean",
            ),
            class_margin_mean=(
                "class_separation_margin",
                "mean",
            ),
            rank_ratio_mean=(
                "effective_rank_ratio",
                "mean",
            ),
            embedding_norm_mean=(
                "mean_embedding_norm",
                "mean",
            ),
        )
    )

    print(
        compact_metrics.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.10f}",
        )
    )

    print("\nSaved:", RUN_OUTPUT)
    print("Saved:", SUMMARY_OUTPUT)
    print("Saved:", PAIRED_OUTPUT)
    print("Saved:", METRIC_RUN_OUTPUT)
    print("Saved:", METRIC_SUMMARY_OUTPUT)


if __name__ == "__main__":
    main()
