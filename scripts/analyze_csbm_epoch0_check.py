from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd
from scipy import stats


RUN_DIR = Path("runs/csbm_epoch0_check_v2")

RUN_OUTPUT = RUN_DIR / "csbm_epoch0_run_level.csv"
SUMMARY_OUTPUT = RUN_DIR / "csbm_epoch0_summary.csv"
CHANGE_OUTPUT = RUN_DIR / "csbm_epoch0_to_epoch1_change.csv"


def module_number(name: object) -> int:
    match = re.search(
        r"\.(\d+)(?:#\d+)?$",
        str(name),
    )
    return int(match.group(1)) if match else -1


def metric_value(
    row: pd.Series,
    *candidate_names: str,
) -> float:
    for name in candidate_names:
        if (
            name in row.index
            and pd.notna(row[name])
        ):
            return float(row[name])

    return float("nan")


def summarize(values: pd.Series) -> dict[str, float]:
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
    critical = float(
        stats.t.ppf(
            0.975,
            df=n - 1,
        )
    )

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95": critical * std / math.sqrt(n),
    }


def select_last_hidden(
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


def load_run_level() -> pd.DataFrame:
    summary_files = sorted(
        RUN_DIR.glob("*_summary.json")
    )

    if len(summary_files) != 40:
        raise RuntimeError(
            f"Expected 40 summaries, "
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

        metrics = pd.read_csv(metric_path)

        epochs = sorted(
            int(value)
            for value in metrics["epoch"].unique()
        )

        if epochs != [0, 1]:
            raise RuntimeError(
                f"{metric_path.name}: "
                f"expected [0, 1], found {epochs}"
            )

        model = str(summary["model"])
        hidden_channels = int(
            summary["hidden_channels"]
        )

        for epoch in (0, 1):
            epoch_frame = metrics[
                metrics["epoch"] == epoch
            ]

            hidden = select_last_hidden(
                epoch_frame,
                model=model,
                hidden_channels=hidden_channels,
            )

            rows.append(
                {
                    "feature_signal": float(
                        summary["feature_signal"]
                    ),
                    "model": model,
                    "uses_pairnorm":
                        model.endswith("PairNorm"),
                    "num_layers": int(
                        summary["num_layers"]
                    ),
                    "graph_seed": int(
                        summary["graph_seed"]
                    ),
                    "epoch": epoch,
                    "layer_name":
                        hidden["layer_name"],
                    "pairwise_distance": float(
                        hidden[
                            "mean_pairwise_cosine_distance"
                        ]
                    ),
                    "edge_distance": float(
                        hidden[
                            "mean_edge_cosine_distance"
                        ]
                    ),
                    "rank_ratio": float(
                        hidden[
                            "effective_rank_ratio"
                        ]
                    ),
                    "normalized_dirichlet": float(
                        hidden[
                            "normalized_dirichlet_energy"
                        ]
                    ),
                    "embedding_norm": metric_value(
                        hidden,
                        "embedding_norm",
                        "mean_embedding_norm",
                        "embedding_norm_mean",
                    ),
                    "feature_variance": metric_value(
                        hidden,
                        "feature_variance",
                        "mean_feature_variance",
                        "feature_variance_mean",
                    ),
                }
            )

    frame = pd.DataFrame(rows)

    expected_rows = 40 * 2

    if len(frame) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows, "
            f"found {len(frame)}"
        )

    return frame.sort_values(
        [
            "feature_signal",
            "model",
            "num_layers",
            "graph_seed",
            "epoch",
        ]
    ).reset_index(drop=True)


def build_summary(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    group_columns = [
        "feature_signal",
        "model",
        "num_layers",
        "epoch",
    ]

    for keys, group in runs.groupby(
        group_columns
    ):
        pairwise = summarize(
            group["pairwise_distance"]
        )
        edge = summarize(
            group["edge_distance"]
        )
        rank = summarize(
            group["rank_ratio"]
        )

        rows.append(
            {
                "feature_signal": keys[0],
                "model": keys[1],
                "num_layers": keys[2],
                "epoch": keys[3],
                "n": pairwise["n"],
                "pairwise_mean":
                    pairwise["mean"],
                "pairwise_std":
                    pairwise["std"],
                "pairwise_ci95":
                    pairwise["ci95"],
                "edge_mean":
                    edge["mean"],
                "edge_ci95":
                    edge["ci95"],
                "rank_ratio_mean":
                    rank["mean"],
                "rank_ratio_ci95":
                    rank["ci95"],
                "below_1e_3_seeds": int(
                    (
                        group["pairwise_distance"]
                        < 1e-3
                    ).sum()
                ),
                "below_1e_4_seeds": int(
                    (
                        group["pairwise_distance"]
                        < 1e-4
                    ).sum()
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        group_columns
    ).reset_index(drop=True)


def build_changes(
    runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = [
        "feature_signal",
        "model",
        "uses_pairnorm",
        "num_layers",
        "graph_seed",
    ]

    values = [
        "pairwise_distance",
        "edge_distance",
        "rank_ratio",
        "normalized_dirichlet",
        "embedding_norm",
        "feature_variance",
    ]

    epoch0 = runs[
        runs["epoch"] == 0
    ][keys + values].copy()

    epoch1 = runs[
        runs["epoch"] == 1
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

    paired = epoch0.merge(
        epoch1,
        on=keys,
        validate="one_to_one",
    )

    for column in values:
        paired[f"change_{column}"] = (
            paired[f"epoch1_{column}"]
            - paired[f"epoch0_{column}"]
        )

    paired["pairwise_ratio_epoch1_epoch0"] = (
        paired["epoch1_pairwise_distance"]
        / paired["epoch0_pairwise_distance"]
    )

    summary_rows = []

    for keys_value, group in paired.groupby(
        [
            "feature_signal",
            "model",
            "num_layers",
        ]
    ):
        change = summarize(
            group["change_pairwise_distance"]
        )
        ratio = summarize(
            group["pairwise_ratio_epoch1_epoch0"]
        )

        summary_rows.append(
            {
                "feature_signal":
                    keys_value[0],
                "model":
                    keys_value[1],
                "num_layers":
                    keys_value[2],
                "n":
                    change["n"],
                "epoch0_pairwise_mean":
                    group[
                        "epoch0_pairwise_distance"
                    ].mean(),
                "epoch1_pairwise_mean":
                    group[
                        "epoch1_pairwise_distance"
                    ].mean(),
                "pairwise_change_mean":
                    change["mean"],
                "pairwise_change_ci95":
                    change["ci95"],
                "epoch1_epoch0_ratio_mean":
                    ratio["mean"],
                "epoch0_below_1e_4_seeds":
                    int(
                        (
                            group[
                                "epoch0_pairwise_distance"
                            ]
                            < 1e-4
                        ).sum()
                    ),
                "epoch1_below_1e_4_seeds":
                    int(
                        (
                            group[
                                "epoch1_pairwise_distance"
                            ]
                            < 1e-4
                        ).sum()
                    ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        [
            "feature_signal",
            "model",
            "num_layers",
        ]
    ).reset_index(drop=True)

    return paired, summary


def main() -> None:
    runs = load_run_level()
    summary = build_summary(runs)
    paired, changes = build_changes(runs)

    runs.to_csv(
        RUN_OUTPUT,
        index=False,
    )
    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )
    paired.to_csv(
        CHANGE_OUTPUT,
        index=False,
    )

    print(
        "\n=== Epoch 0 and Epoch 1 ==="
    )

    print(
        summary[
            [
                "feature_signal",
                "model",
                "num_layers",
                "epoch",
                "pairwise_mean",
                "pairwise_ci95",
                "edge_mean",
                "rank_ratio_mean",
                "below_1e_3_seeds",
                "below_1e_4_seeds",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.8f}",
        )
    )

    print(
        "\n=== Change after first optimizer step ==="
    )

    print(
        changes.to_string(
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
