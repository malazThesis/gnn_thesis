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
    "runs/csbm_classmetrics_temporal_fs050"
)

PLOT_DIR = Path(
    "plots/csbm_classmetrics_temporal_fs050"
)

METRIC_EPOCHS = [0, 1, 4, 8, 32, 200]

METRICS = [
    "mean_pairwise_cosine_distance",
    "within_class_cosine_distance",
    "between_class_cosine_distance",
    "class_separation_margin",
    "fisher_discriminant_ratio",
    "effective_rank_ratio",
]

METRIC_RUN_OUTPUT = (
    RUN_DIR
    / "csbm_temporal_metric_run_level.csv"
)

METRIC_SUMMARY_OUTPUT = (
    RUN_DIR
    / "csbm_temporal_metric_summary.csv"
)

HISTORY_RUN_OUTPUT = (
    RUN_DIR
    / "csbm_temporal_history_run_level.csv"
)

HISTORY_SUMMARY_OUTPUT = (
    RUN_DIR
    / "csbm_temporal_history_summary.csv"
)

TRAINING_SUMMARY_OUTPUT = (
    RUN_DIR
    / "csbm_temporal_training_summary.csv"
)

EFFECT_OUTPUT = (
    RUN_DIR
    / "csbm_temporal_pairnorm_effects_epoch200.csv"
)


def module_number(name: object) -> int:
    match = re.search(
        r"\.(\d+)(?:#\d+)?$",
        str(name),
    )

    return int(match.group(1)) if match else -1


def mean_ci95(
    values: pd.Series,
) -> tuple[int, float, float, float]:
    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    n = len(values)

    if n == 0:
        return (
            0,
            math.nan,
            math.nan,
            math.nan,
        )

    mean = float(values.mean())

    if n == 1:
        return (
            1,
            mean,
            0.0,
            0.0,
        )

    std = float(
        values.std(ddof=1)
    )

    critical = float(
        stats.t.ppf(
            0.975,
            df=n - 1,
        )
    )

    ci95 = (
        critical
        * std
        / math.sqrt(n)
    )

    return n, mean, std, ci95


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

    return float(
        selected["epoch"].min()
    )


def first_epoch_below(
    history: pd.DataFrame,
    column: str,
    threshold: float,
) -> float:
    selected = history[
        history[column] <= threshold
    ]

    if selected.empty:
        return math.nan

    return float(
        selected["epoch"].min()
    )


def select_representations(
    frame: pd.DataFrame,
    *,
    model: str,
    hidden_channels: int,
) -> dict[str, pd.Series]:
    frame = frame[
        frame["layer_name"].notna()
    ].copy()

    frame["module_number"] = (
        frame["layer_name"].map(
            module_number
        )
    )

    convolution_rows = frame[
        frame["layer_name"]
        .astype(str)
        .str.startswith("convs.")
    ].copy()

    if convolution_rows.empty:
        raise RuntimeError(
            f"No convolution rows found for {model}"
        )

    logits = convolution_rows.sort_values(
        [
            "module_number",
            "layer_index",
        ]
    ).iloc[-1]

    if model.endswith("PairNorm"):
        hidden_rows = frame[
            frame["layer_name"]
            .astype(str)
            .str.startswith("pns.")
        ].copy()
    else:
        hidden_rows = convolution_rows[
            convolution_rows["embedding_dim"]
            == hidden_channels
        ].copy()

    if hidden_rows.empty:
        raise RuntimeError(
            f"No final hidden row found for {model}"
        )

    final_hidden = hidden_rows.sort_values(
        [
            "module_number",
            "layer_index",
        ]
    ).iloc[-1]

    return {
        "final_hidden": final_hidden,
        "logits": logits,
    }


def load_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    summary_files = sorted(
        RUN_DIR.glob("*_summary.json")
    )

    if len(summary_files) != 20:
        raise RuntimeError(
            f"Expected 20 summary files, "
            f"found {len(summary_files)}"
        )

    metric_rows = []
    history_frames = []
    training_rows = []

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

        history_path = summary_path.with_name(
            summary_path.name.replace(
                "_summary.json",
                "_history.csv",
            )
        )

        if not metric_path.exists():
            raise RuntimeError(
                f"Missing metrics: {metric_path}"
            )

        if not history_path.exists():
            raise RuntimeError(
                f"Missing history: {history_path}"
            )

        metrics = pd.read_csv(metric_path)
        history = pd.read_csv(history_path)

        model = str(summary["model"])
        depth = int(summary["num_layers"])
        seed = int(summary["graph_seed"])
        split_idx = int(summary["split_idx"])
        hidden_channels = int(
            summary["hidden_channels"]
        )
        feature_signal = float(
            summary["feature_signal"]
        )

        found_epochs = sorted(
            int(value)
            for value in metrics[
                "epoch"
            ].unique()
        )

        if found_epochs != METRIC_EPOCHS:
            raise RuntimeError(
                f"{metric_path.name}: "
                f"expected {METRIC_EPOCHS}, "
                f"found {found_epochs}"
            )

        history = history.copy()

        history["feature_signal"] = (
            feature_signal
        )
        history["model"] = model
        history["num_layers"] = depth
        history["graph_seed"] = seed
        history["split_idx"] = split_idx
        history["uses_pairnorm"] = (
            model.endswith("PairNorm")
        )

        history_frames.append(history)

        final_row = history.sort_values(
            "epoch"
        ).iloc[-1]

        best_index = history[
            "val_acc"
        ].idxmax()

        best_row = history.loc[
            best_index
        ]

        training_rows.append(
            {
                "feature_signal":
                    feature_signal,
                "model":
                    model,
                "num_layers":
                    depth,
                "graph_seed":
                    seed,
                "split_idx":
                    split_idx,
                "final_loss":
                    float(final_row["loss"]),
                "final_train_acc":
                    float(final_row["train_acc"]),
                "final_val_acc":
                    float(final_row["val_acc"]),
                "final_test_acc":
                    float(final_row["test_acc"]),
                "best_epoch":
                    int(best_row["epoch"]),
                "best_val_acc":
                    float(best_row["val_acc"]),
                "test_acc_at_best_val":
                    float(best_row["test_acc"]),
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
                "first_loss_below_065":
                    first_epoch_below(
                        history,
                        "loss",
                        0.65,
                    ),
                "collapsed_final":
                    bool(
                        final_row["test_acc"]
                        <= 0.52
                    ),
            }
        )

        history_indexed = (
            history.set_index("epoch")
        )

        for epoch in METRIC_EPOCHS:
            epoch_frame = metrics[
                metrics["epoch"] == epoch
            ].copy()

            selected = select_representations(
                epoch_frame,
                model=model,
                hidden_channels=hidden_channels,
            )

            for representation, row in (
                selected.items()
            ):
                output = {
                    "feature_signal":
                        feature_signal,
                    "model":
                        model,
                    "uses_pairnorm":
                        model.endswith(
                            "PairNorm"
                        ),
                    "num_layers":
                        depth,
                    "graph_seed":
                        seed,
                    "split_idx":
                        split_idx,
                    "epoch":
                        epoch,
                    "representation":
                        representation,
                    "layer_name":
                        row["layer_name"],
                    "embedding_dim":
                        int(
                            row["embedding_dim"]
                        ),
                }

                for metric in METRICS:
                    output[metric] = float(
                        row[metric]
                    )

                if epoch in history_indexed.index:
                    history_row = (
                        history_indexed.loc[epoch]
                    )

                    output["loss"] = float(
                        history_row["loss"]
                    )
                    output["train_acc"] = float(
                        history_row["train_acc"]
                    )
                    output["val_acc"] = float(
                        history_row["val_acc"]
                    )
                    output["test_acc"] = float(
                        history_row["test_acc"]
                    )
                else:
                    output["loss"] = math.nan
                    output["train_acc"] = math.nan
                    output["val_acc"] = math.nan
                    output["test_acc"] = math.nan

                within = output[
                    "within_class_cosine_distance"
                ]
                between = output[
                    "between_class_cosine_distance"
                ]
                global_distance = output[
                    "mean_pairwise_cosine_distance"
                ]

                denominator = (
                    within
                    + between
                )

                output[
                    "normalized_class_margin"
                ] = (
                    (
                        between
                        - within
                    )
                    / denominator
                    if denominator > 0.0
                    else math.nan
                )

                output[
                    "margin_to_global_distance"
                ] = (
                    output[
                        "class_separation_margin"
                    ]
                    / global_distance
                    if global_distance > 0.0
                    else math.nan
                )

                metric_rows.append(output)

    metric_runs = pd.DataFrame(
        metric_rows
    )

    history_runs = pd.concat(
        history_frames,
        ignore_index=True,
    )

    training_runs = pd.DataFrame(
        training_rows
    )

    expected_metric_rows = (
        20
        * len(METRIC_EPOCHS)
        * 2
    )

    if len(metric_runs) != expected_metric_rows:
        raise RuntimeError(
            f"Expected {expected_metric_rows} "
            f"metric rows, "
            f"found {len(metric_runs)}"
        )

    return (
        metric_runs,
        history_runs,
        training_runs,
    )


def summarize_metrics(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    value_columns = (
        METRICS
        + [
            "normalized_class_margin",
            "margin_to_global_distance",
            "loss",
            "train_acc",
            "val_acc",
            "test_acc",
        ]
    )

    group_columns = [
        "feature_signal",
        "model",
        "num_layers",
        "epoch",
        "representation",
    ]

    rows = []

    for keys, group in runs.groupby(
        group_columns
    ):
        row = {
            "feature_signal": keys[0],
            "model": keys[1],
            "num_layers": keys[2],
            "epoch": keys[3],
            "representation": keys[4],
        }

        for column in value_columns:
            n, mean, std, ci95 = mean_ci95(
                group[column]
            )

            row[f"{column}_n"] = n
            row[f"{column}_mean"] = mean
            row[f"{column}_std"] = std
            row[f"{column}_ci95"] = ci95

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        group_columns
    ).reset_index(drop=True)


def summarize_history(
    history: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for keys, group in history.groupby(
        [
            "feature_signal",
            "model",
            "num_layers",
            "epoch",
        ]
    ):
        row = {
            "feature_signal": keys[0],
            "model": keys[1],
            "num_layers": keys[2],
            "epoch": keys[3],
        }

        for column in [
            "loss",
            "train_acc",
            "val_acc",
            "test_acc",
            "gradient_norm",
        ]:
            n, mean, std, ci95 = mean_ci95(
                group[column]
            )

            row[f"{column}_n"] = n
            row[f"{column}_mean"] = mean
            row[f"{column}_std"] = std
            row[f"{column}_ci95"] = ci95

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        [
            "feature_signal",
            "model",
            "num_layers",
            "epoch",
        ]
    ).reset_index(drop=True)


def summarize_training(
    training: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for keys, group in training.groupby(
        [
            "feature_signal",
            "model",
            "num_layers",
        ]
    ):
        row = {
            "feature_signal": keys[0],
            "model": keys[1],
            "num_layers": keys[2],
            "n": len(group),
            "collapsed_seeds": int(
                group[
                    "collapsed_final"
                ].sum()
            ),
        }

        for column in [
            "final_loss",
            "final_train_acc",
            "final_val_acc",
            "final_test_acc",
            "best_epoch",
            "best_val_acc",
            "test_acc_at_best_val",
            "first_train_90",
            "first_val_90",
            "first_test_90",
            "first_loss_below_065",
        ]:
            _, mean, std, ci95 = mean_ci95(
                group[column]
            )

            row[f"{column}_mean"] = mean
            row[f"{column}_std"] = std
            row[f"{column}_ci95"] = ci95

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        [
            "feature_signal",
            "model",
            "num_layers",
        ]
    ).reset_index(drop=True)


def paired_effects_epoch200(
    metric_runs: pd.DataFrame,
) -> pd.DataFrame:
    selected = metric_runs[
        metric_runs["epoch"] == 200
    ].copy()

    baseline = selected[
        selected["model"] == "GraphSAGE"
    ].copy()

    pairnorm = selected[
        selected["model"]
        == "GraphSAGEPairNorm"
    ].copy()

    keys = [
        "feature_signal",
        "num_layers",
        "graph_seed",
        "split_idx",
        "epoch",
        "representation",
    ]

    value_columns = [
        "mean_pairwise_cosine_distance",
        "class_separation_margin",
        "fisher_discriminant_ratio",
        "effective_rank_ratio",
        "test_acc",
    ]

    baseline = baseline[
        keys + value_columns
    ].rename(
        columns={
            column: f"baseline_{column}"
            for column in value_columns
        }
    )

    pairnorm = pairnorm[
        keys + value_columns
    ].rename(
        columns={
            column: f"pairnorm_{column}"
            for column in value_columns
        }
    )

    paired = baseline.merge(
        pairnorm,
        on=keys,
        validate="one_to_one",
    )

    rows = []

    for keys_value, group in paired.groupby(
        [
            "feature_signal",
            "num_layers",
            "representation",
        ]
    ):
        for metric in value_columns:
            differences = (
                group[
                    f"pairnorm_{metric}"
                ]
                - group[
                    f"baseline_{metric}"
                ]
            )

            n, mean, std, ci95 = mean_ci95(
                differences
            )

            if n >= 2 and std > 0.0:
                t_result = stats.ttest_1samp(
                    differences,
                    popmean=0.0,
                )

                t_statistic = float(
                    t_result.statistic
                )
                t_pvalue = float(
                    t_result.pvalue
                )
            else:
                t_statistic = math.nan
                t_pvalue = math.nan

            try:
                wilcoxon_result = stats.wilcoxon(
                    differences,
                    alternative="two-sided",
                    zero_method="wilcox",
                )

                wilcoxon_pvalue = float(
                    wilcoxon_result.pvalue
                )
            except ValueError:
                wilcoxon_pvalue = math.nan

            rows.append(
                {
                    "feature_signal":
                        keys_value[0],
                    "num_layers":
                        keys_value[1],
                    "representation":
                        keys_value[2],
                    "metric":
                        metric,
                    "n":
                        n,
                    "mean_difference_pairnorm_minus_baseline":
                        mean,
                    "std_difference":
                        std,
                    "ci95_difference":
                        ci95,
                    "positive_seeds":
                        int(
                            (differences > 0).sum()
                        ),
                    "negative_seeds":
                        int(
                            (differences < 0).sum()
                        ),
                    "t_statistic":
                        t_statistic,
                    "t_pvalue":
                        t_pvalue,
                    "wilcoxon_pvalue":
                        wilcoxon_pvalue,
                }
            )

    return pd.DataFrame(rows).sort_values(
        [
            "num_layers",
            "representation",
            "metric",
        ]
    ).reset_index(drop=True)


def plot_metric(
    summary: pd.DataFrame,
    *,
    representation: str,
    mean_column: str,
    ci_column: str,
    ylabel: str,
    filename: str,
    scale: str = "linear",
) -> None:
    selected = summary[
        summary["representation"]
        == representation
    ].copy()

    plt.figure(figsize=(10, 6))

    for keys, group in selected.groupby(
        [
            "model",
            "num_layers",
        ]
    ):
        model, depth = keys
        group = group.sort_values("epoch")

        x = group["epoch"].to_numpy(
            dtype=float
        )
        y = group[mean_column].to_numpy(
            dtype=float
        )
        ci = group[ci_column].to_numpy(
            dtype=float
        )

        label = f"{model}, L{depth}"

        plt.plot(
            x,
            y,
            marker="o",
            label=label,
        )

        plt.fill_between(
            x,
            y - ci,
            y + ci,
            alpha=0.2,
        )

    if scale == "log":
        plt.yscale("log")
    elif scale == "symlog":
        plt.yscale(
            "symlog",
            linthresh=1e-8,
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(
        f"{ylabel} over training "
        f"({representation})"
    )
    plt.legend()
    plt.tight_layout()

    path = PLOT_DIR / filename

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("Saved:", path)


def plot_history(
    summary: pd.DataFrame,
    *,
    mean_column: str,
    ci_column: str,
    ylabel: str,
    filename: str,
) -> None:
    plt.figure(figsize=(10, 6))

    for keys, group in summary.groupby(
        [
            "model",
            "num_layers",
        ]
    ):
        model, depth = keys
        group = group.sort_values("epoch")

        x = group["epoch"].to_numpy(
            dtype=float
        )
        y = group[mean_column].to_numpy(
            dtype=float
        )
        ci = group[ci_column].to_numpy(
            dtype=float
        )

        plt.plot(
            x,
            y,
            label=f"{model}, L{depth}",
        )

        plt.fill_between(
            x,
            y - ci,
            y + ci,
            alpha=0.2,
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} over training")
    plt.legend()
    plt.tight_layout()

    path = PLOT_DIR / filename

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

    (
        metric_runs,
        history_runs,
        training_runs,
    ) = load_data()

    metric_summary = summarize_metrics(
        metric_runs
    )

    history_summary = summarize_history(
        history_runs
    )

    training_summary = summarize_training(
        training_runs
    )

    effects = paired_effects_epoch200(
        metric_runs
    )

    metric_runs.to_csv(
        METRIC_RUN_OUTPUT,
        index=False,
    )

    metric_summary.to_csv(
        METRIC_SUMMARY_OUTPUT,
        index=False,
    )

    history_runs.to_csv(
        HISTORY_RUN_OUTPUT,
        index=False,
    )

    history_summary.to_csv(
        HISTORY_SUMMARY_OUTPUT,
        index=False,
    )

    training_summary.to_csv(
        TRAINING_SUMMARY_OUTPUT,
        index=False,
    )

    effects.to_csv(
        EFFECT_OUTPUT,
        index=False,
    )

    plot_metric(
        metric_summary,
        representation="final_hidden",
        mean_column=(
            "mean_pairwise_cosine_distance_mean"
        ),
        ci_column=(
            "mean_pairwise_cosine_distance_ci95"
        ),
        ylabel=(
            "Mean pairwise cosine distance"
        ),
        filename=(
            "final_hidden_pairwise_distance.png"
        ),
        scale="log",
    )

    plot_metric(
        metric_summary,
        representation="final_hidden",
        mean_column=(
            "class_separation_margin_mean"
        ),
        ci_column=(
            "class_separation_margin_ci95"
        ),
        ylabel="Class separation margin",
        filename=(
            "final_hidden_class_margin.png"
        ),
        scale="symlog",
    )

    plot_metric(
        metric_summary,
        representation="logits",
        mean_column=(
            "class_separation_margin_mean"
        ),
        ci_column=(
            "class_separation_margin_ci95"
        ),
        ylabel="Logit class separation margin",
        filename="logit_class_margin.png",
        scale="symlog",
    )

    plot_metric(
        metric_summary,
        representation="final_hidden",
        mean_column=(
            "fisher_discriminant_ratio_mean"
        ),
        ci_column=(
            "fisher_discriminant_ratio_ci95"
        ),
        ylabel="Fisher discriminant ratio",
        filename=(
            "final_hidden_fisher_ratio.png"
        ),
        scale="symlog",
    )

    plot_history(
        history_summary,
        mean_column="test_acc_mean",
        ci_column="test_acc_ci95",
        ylabel="Test accuracy",
        filename="test_accuracy.png",
    )

    plot_history(
        history_summary,
        mean_column="loss_mean",
        ci_column="loss_ci95",
        ylabel="Training loss",
        filename="training_loss.png",
    )

    print(
        "\n=== Training outcomes ==="
    )

    print(
        training_summary[
            [
                "model",
                "num_layers",
                "n",
                "collapsed_seeds",
                "final_loss_mean",
                "final_train_acc_mean",
                "final_val_acc_mean",
                "final_test_acc_mean",
                "best_epoch_mean",
                "test_acc_at_best_val_mean",
                "first_train_90_mean",
                "first_val_90_mean",
                "first_loss_below_065_mean",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.8f}",
        )
    )

    print(
        "\n=== Temporal final-hidden metrics ==="
    )

    final_hidden = metric_summary[
        metric_summary["representation"]
        == "final_hidden"
    ]

    print(
        final_hidden[
            [
                "model",
                "num_layers",
                "epoch",
                "mean_pairwise_cosine_distance_mean",
                "class_separation_margin_mean",
                "fisher_discriminant_ratio_mean",
                "effective_rank_ratio_mean",
                "test_acc_mean",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.10f}",
        )
    )

    print(
        "\n=== Temporal logit metrics ==="
    )

    logits = metric_summary[
        metric_summary["representation"]
        == "logits"
    ]

    print(
        logits[
            [
                "model",
                "num_layers",
                "epoch",
                "mean_pairwise_cosine_distance_mean",
                "class_separation_margin_mean",
                "fisher_discriminant_ratio_mean",
                "test_acc_mean",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.10f}",
        )
    )

    print(
        "\n=== PairNorm effects at epoch 200 ==="
    )

    print(
        effects.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.10f}",
        )
    )

    print("\nSaved:", METRIC_RUN_OUTPUT)
    print("Saved:", METRIC_SUMMARY_OUTPUT)
    print("Saved:", HISTORY_RUN_OUTPUT)
    print("Saved:", HISTORY_SUMMARY_OUTPUT)
    print("Saved:", TRAINING_SUMMARY_OUTPUT)
    print("Saved:", EFFECT_OUTPUT)


if __name__ == "__main__":
    main()
