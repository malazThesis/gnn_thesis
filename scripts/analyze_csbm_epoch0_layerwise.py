from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


RUN_DIR = Path("runs/csbm_epoch0_check_v2")
PLOT_DIR = Path("plots/csbm_epoch0_layerwise")

RUN_OUTPUT = (
    RUN_DIR / "csbm_epoch0_layerwise_run.csv"
)
SUMMARY_OUTPUT = (
    RUN_DIR / "csbm_epoch0_layerwise_summary.csv"
)
CHANGE_OUTPUT = (
    RUN_DIR / "csbm_epoch0_layerwise_change.csv"
)


def module_number(name: object) -> int:
    match = re.search(
        r"\.(\d+)(?:#\d+)?$",
        str(name),
    )
    return int(match.group(1)) if match else -1


def confidence_interval(
    values: pd.Series,
) -> tuple[float, float, float]:
    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    mean = float(values.mean())

    if len(values) < 2:
        return mean, 0.0, 0.0

    std = float(values.std(ddof=1))
    critical = float(
        stats.t.ppf(
            0.975,
            df=len(values) - 1,
        )
    )

    ci95 = critical * std / math.sqrt(len(values))

    return mean, std, ci95


def load_rows() -> pd.DataFrame:
    rows = []

    summaries = sorted(
        RUN_DIR.glob("*_summary.json")
    )

    if len(summaries) != 40:
        raise RuntimeError(
            f"Expected 40 summaries, found {len(summaries)}"
        )

    for summary_path in summaries:
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

        frame = pd.read_csv(metric_path)
        frame = frame[
            frame["epoch"].isin([0, 1])
            & frame["layer_name"].notna()
        ].copy()

        model = str(summary["model"])
        hidden_channels = int(
            summary["hidden_channels"]
        )

        if model.endswith("PairNorm"):
            frame = frame[
                frame["layer_name"]
                .astype(str)
                .str.startswith("pns.")
            ].copy()
        else:
            frame = frame[
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

        frame["stage"] = (
            frame["layer_name"]
            .map(module_number)
            + 1
        )

        for row in frame.itertuples(index=False):
            rows.append(
                {
                    "feature_signal": float(
                        summary["feature_signal"]
                    ),
                    "model": model,
                    "num_layers": int(
                        summary["num_layers"]
                    ),
                    "graph_seed": int(
                        summary["graph_seed"]
                    ),
                    "epoch": int(row.epoch),
                    "stage": int(row.stage),
                    "pairwise_distance": float(
                        row.mean_pairwise_cosine_distance
                    ),
                    "edge_distance": float(
                        row.mean_edge_cosine_distance
                    ),
                    "rank_ratio": float(
                        row.effective_rank_ratio
                    ),
                }
            )

    result = pd.DataFrame(rows)

    expected_groups = (
        result.groupby(
            [
                "feature_signal",
                "model",
                "num_layers",
                "graph_seed",
                "epoch",
            ]
        )
        .size()
    )

    print(
        "Layer counts per run:",
        sorted(expected_groups.unique()),
    )

    return result


def summarize_layers(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for keys, group in runs.groupby(
        [
            "feature_signal",
            "model",
            "num_layers",
            "epoch",
            "stage",
        ]
    ):
        pairwise_mean, pairwise_std, pairwise_ci95 = (
            confidence_interval(
                group["pairwise_distance"]
            )
        )

        edge_mean, _, edge_ci95 = (
            confidence_interval(
                group["edge_distance"]
            )
        )

        rank_mean, _, rank_ci95 = (
            confidence_interval(
                group["rank_ratio"]
            )
        )

        rows.append(
            {
                "feature_signal": keys[0],
                "model": keys[1],
                "num_layers": keys[2],
                "epoch": keys[3],
                "stage": keys[4],
                "n": len(group),
                "pairwise_mean":
                    pairwise_mean,
                "pairwise_std":
                    pairwise_std,
                "pairwise_ci95":
                    pairwise_ci95,
                "edge_mean":
                    edge_mean,
                "edge_ci95":
                    edge_ci95,
                "rank_ratio_mean":
                    rank_mean,
                "rank_ratio_ci95":
                    rank_ci95,
            }
        )

    result = pd.DataFrame(rows).sort_values(
        [
            "feature_signal",
            "model",
            "num_layers",
            "epoch",
            "stage",
        ]
    )

    result["previous_pairwise"] = (
        result.groupby(
            [
                "feature_signal",
                "model",
                "num_layers",
                "epoch",
            ]
        )["pairwise_mean"]
        .shift(1)
    )

    result["within_forward_retention"] = (
        result["pairwise_mean"]
        / result["previous_pairwise"]
    )

    return result


def build_epoch_change(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "feature_signal",
        "model",
        "num_layers",
        "stage",
    ]

    values = [
        "pairwise_mean",
        "edge_mean",
        "rank_ratio_mean",
    ]

    epoch0 = summary[
        summary["epoch"] == 0
    ][keys + values].copy()

    epoch1 = summary[
        summary["epoch"] == 1
    ][keys + values].copy()

    epoch0 = epoch0.rename(
        columns={
            column: f"epoch0_{column}"
            for column in values
        }
    )

    epoch1 = epoch1.rename(
        columns={
            column: f"epoch1_{column}"
            for column in values
        }
    )

    result = epoch0.merge(
        epoch1,
        on=keys,
        validate="one_to_one",
    )

    result["optimizer_pairwise_ratio"] = (
        result["epoch1_pairwise_mean"]
        / result["epoch0_pairwise_mean"]
    )

    result["optimizer_pairwise_change"] = (
        result["epoch1_pairwise_mean"]
        - result["epoch0_pairwise_mean"]
    )

    result["optimizer_edge_ratio"] = (
        result["epoch1_edge_mean"]
        / result["epoch0_edge_mean"]
    )

    result["optimizer_rank_ratio"] = (
        result["epoch1_rank_ratio_mean"]
        / result["epoch0_rank_ratio_mean"]
    )

    return result.sort_values(keys)


def plot_profiles(
    summary: pd.DataFrame,
) -> None:
    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for keys, frame in summary.groupby(
        [
            "feature_signal",
            "num_layers",
        ]
    ):
        signal, depth = keys

        plt.figure(figsize=(10, 6))

        for line_keys, line in frame.groupby(
            [
                "model",
                "epoch",
            ]
        ):
            model, epoch = line_keys
            line = line.sort_values("stage")

            plt.errorbar(
                line["stage"],
                line["pairwise_mean"],
                yerr=line["pairwise_ci95"],
                marker="o",
                capsize=3,
                label=f"{model}, epoch {epoch}",
            )

        plt.yscale("log")
        plt.xlabel("Hidden message-passing stage")
        plt.ylabel(
            "Mean pairwise cosine distance"
        )
        plt.title(
            "Layerwise pre-compression and "
            "first-step effect "
            f"(FS={signal:.2f}, L{depth})"
        )
        plt.legend()
        plt.tight_layout()

        path = (
            PLOT_DIR
            / (
                f"epoch0_epoch1_fs"
                f"{int(signal * 100):03d}_"
                f"L{depth}.png"
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
    runs = load_rows()
    summary = summarize_layers(runs)
    changes = build_epoch_change(summary)

    runs.to_csv(
        RUN_OUTPUT,
        index=False,
    )
    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )
    changes.to_csv(
        CHANGE_OUTPUT,
        index=False,
    )

    plot_profiles(summary)

    print(
        "\n=== First optimizer-step effect by layer ==="
    )

    print(
        changes[
            [
                "feature_signal",
                "model",
                "num_layers",
                "stage",
                "epoch0_pairwise_mean",
                "epoch1_pairwise_mean",
                "optimizer_pairwise_ratio",
                "optimizer_edge_ratio",
                "optimizer_rank_ratio",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.8f}",
        )
    )

    print("\nSaved:", RUN_OUTPUT)
    print("Saved:", SUMMARY_OUTPUT)
    print("Saved:", CHANGE_OUTPUT)


if __name__ == "__main__":
    main()
