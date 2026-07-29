from __future__ import annotations

import itertools
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


RUN_DIR = Path(
    "runs/csbm_l8_graph_init_factorial_fs050"
)

RUN_PATH = (
    RUN_DIR / "csbm_l8_graph_init_run_level.csv"
)

OUTPUT_PATH = (
    RUN_DIR
    / "csbm_l8_early_predictors_exact.csv"
)

EARLY_RUN_OUTPUT = (
    RUN_DIR
    / "csbm_l8_early_metrics_run_level.csv"
)

EPOCHS = [0, 1, 4, 8, 32]

METRICS = [
    "mean_pairwise_cosine_distance",
    "class_separation_margin",
    "fisher_discriminant_ratio",
    "effective_rank_ratio",
]


def module_number(name: object) -> int:
    match = re.search(
        r"\.(\d+)(?:#\d+)?$",
        str(name),
    )

    return int(match.group(1)) if match else -1


def resolve_graph_seed(
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

    graph_seed = summary.get(
        "graph_seed"
    )

    if graph_seed is not None:
        return int(graph_seed)

    raise RuntimeError(
        f"Cannot resolve graph seed: {filename}"
    )


def select_rows(
    frame: pd.DataFrame,
    hidden_channels: int,
) -> dict[str, pd.Series]:
    frame = frame.copy()

    frame["module_number"] = (
        frame["layer_name"].map(
            module_number
        )
    )

    convolutions = frame[
        frame["layer_name"]
        .astype(str)
        .str.startswith("convs.")
    ].copy()

    logits = convolutions.sort_values(
        [
            "module_number",
            "layer_index",
        ]
    ).iloc[-1]

    hidden = convolutions[
        convolutions["embedding_dim"]
        == hidden_channels
    ].sort_values(
        [
            "module_number",
            "layer_index",
        ]
    ).iloc[-1]

    return {
        "final_hidden": hidden,
        "logits": logits,
    }


def holm_adjust(
    pvalues: list[float],
) -> list[float]:
    values = np.asarray(
        pvalues,
        dtype=float,
    )

    order = np.argsort(values)
    adjusted = np.empty_like(values)

    running = 0.0
    number_tests = len(values)

    for position, index in enumerate(order):
        candidate = (
            number_tests - position
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


def exact_auc_test(
    values: np.ndarray,
    recovered: np.ndarray,
    combination_matrix: np.ndarray,
) -> tuple[float, float, float]:
    ranks = rankdata(
        values,
        method="average",
    )

    number_positive = int(
        recovered.sum()
    )

    number_negative = (
        len(values)
        - number_positive
    )

    rank_sum = float(
        ranks[recovered].sum()
    )

    auc = (
        rank_sum
        - number_positive
        * (number_positive + 1)
        / 2
    ) / (
        number_positive
        * number_negative
    )

    observed_statistic = abs(
        auc - 0.5
    )

    all_rank_sums = (
        combination_matrix @ ranks
    )

    all_aucs = (
        all_rank_sums
        - number_positive
        * (number_positive + 1)
        / 2
    ) / (
        number_positive
        * number_negative
    )

    exact_pvalue = float(
        np.mean(
            np.abs(all_aucs - 0.5)
            >= observed_statistic - 1e-12
        )
    )

    direction_free_auc = max(
        auc,
        1.0 - auc,
    )

    cliff_delta = (
        2.0 * auc
        - 1.0
    )

    return (
        float(auc),
        float(direction_free_auc),
        exact_pvalue,
        float(cliff_delta),
    )


def main() -> None:
    outcomes = pd.read_csv(RUN_PATH)

    outcomes = outcomes[
        outcomes["model"] == "GraphSAGE"
    ].copy()

    if len(outcomes) != 25:
        raise RuntimeError(
            f"Expected 25 baseline runs, "
            f"found {len(outcomes)}"
        )

    metric_rows = []

    summary_files = sorted(
        RUN_DIR.glob("*_summary.json")
    )

    for summary_path in summary_files:
        with summary_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            summary = json.load(handle)

        if summary["model"] != "GraphSAGE":
            continue

        graph_seed = resolve_graph_seed(
            summary,
            summary_path.name,
        )

        initialization_seed = int(
            summary["seed"]
        )

        metric_path = summary_path.with_name(
            summary_path.name.replace(
                "_summary.json",
                "_oversmoothing.csv",
            )
        )

        frame = pd.read_csv(metric_path)

        for epoch in EPOCHS:
            selected = select_rows(
                frame[
                    frame["epoch"] == epoch
                ],
                hidden_channels=int(
                    summary["hidden_channels"]
                ),
            )

            for representation, row in (
                selected.items()
            ):
                result = {
                    "graph_seed":
                        graph_seed,
                    "initialization_seed":
                        initialization_seed,
                    "epoch":
                        epoch,
                    "representation":
                        representation,
                }

                for metric in METRICS:
                    result[metric] = float(
                        row[metric]
                    )

                metric_rows.append(result)

    metrics = pd.DataFrame(
        metric_rows
    )

    metrics = metrics.merge(
        outcomes[
            [
                "graph_seed",
                "initialization_seed",
                "recovered",
                "final_test_acc",
            ]
        ],
        on=[
            "graph_seed",
            "initialization_seed",
        ],
        validate="many_to_one",
    )

    metrics.to_csv(
        EARLY_RUN_OUTPUT,
        index=False,
    )

    base_pairs = (
        outcomes[
            [
                "graph_seed",
                "initialization_seed",
                "recovered",
            ]
        ]
        .sort_values(
            [
                "graph_seed",
                "initialization_seed",
            ]
        )
        .reset_index(drop=True)
    )

    number_positive = int(
        base_pairs["recovered"].sum()
    )

    combinations = list(
        itertools.combinations(
            range(len(base_pairs)),
            number_positive,
        )
    )

    combination_matrix = np.zeros(
        (
            len(combinations),
            len(base_pairs),
        ),
        dtype=float,
    )

    for row_index, indices in enumerate(
        combinations
    ):
        combination_matrix[
            row_index,
            list(indices),
        ] = 1.0

    rows = []

    for keys, group in metrics.groupby(
        [
            "epoch",
            "representation",
        ]
    ):
        aligned = base_pairs.merge(
            group,
            on=[
                "graph_seed",
                "initialization_seed",
                "recovered",
            ],
            validate="one_to_one",
        )

        recovered = aligned[
            "recovered"
        ].astype(bool).to_numpy()

        for metric in METRICS:
            values = aligned[
                metric
            ].to_numpy(
                dtype=float
            )

            (
                auc,
                direction_free_auc,
                exact_pvalue,
                cliff_delta,
            ) = exact_auc_test(
                values,
                recovered,
                combination_matrix,
            )

            rows.append(
                {
                    "epoch": keys[0],
                    "representation": keys[1],
                    "metric": metric,
                    "recovered_n": int(
                        recovered.sum()
                    ),
                    "collapsed_n": int(
                        (~recovered).sum()
                    ),
                    "recovered_mean": float(
                        values[
                            recovered
                        ].mean()
                    ),
                    "collapsed_mean": float(
                        values[
                            ~recovered
                        ].mean()
                    ),
                    "auc_recovered_higher":
                        auc,
                    "direction_free_auc":
                        direction_free_auc,
                    "cliff_delta":
                        cliff_delta,
                    "exact_pvalue":
                        exact_pvalue,
                }
            )

    result = pd.DataFrame(rows)

    result["holm_pvalue"] = holm_adjust(
        result["exact_pvalue"].tolist()
    )

    result = result.sort_values(
        [
            "holm_pvalue",
            "exact_pvalue",
            "direction_free_auc",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\n=== Exact early-predictor tests ==="
    )

    print(
        result.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.10f}",
        )
    )

    print(
        "\nCombinations enumerated:",
        len(combinations),
    )

    print("\nSaved:", EARLY_RUN_OUTPUT)
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
