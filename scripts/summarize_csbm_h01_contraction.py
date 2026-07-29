from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "runs/csbm_lowlabel_h01_parity/"
    "csbm_h01_layerwise_collapse.csv"
)

SUMMARY_OUTPUT = Path(
    "runs/csbm_lowlabel_h01_parity/"
    "csbm_h01_layerwise_contraction_summary.csv"
)

RATIO_OUTPUT = Path(
    "runs/csbm_lowlabel_h01_parity/"
    "csbm_h01_layerwise_contraction_ratios.csv"
)


def first_threshold_stage(
    frame: pd.DataFrame,
    threshold: float,
):
    matches = frame[
        frame["pairwise_mean"] <= threshold
    ]

    if matches.empty:
        return np.nan

    return int(
        matches.iloc[0][
            "message_passing_stage"
        ]
    )


def main():
    frame = pd.read_csv(INPUT)

    frame = frame[
        frame["epoch"] == 1
    ].copy()

    summary_rows = []
    ratio_rows = []

    group_columns = [
        "feature_signal",
        "model",
        "num_layers",
    ]

    for keys, group in frame.groupby(
        group_columns
    ):
        group = group.sort_values(
            "message_passing_stage"
        ).copy()

        group["previous_pairwise"] = (
            group["pairwise_mean"].shift(1)
        )

        group["contraction_ratio"] = (
            group["pairwise_mean"]
            / group["previous_pairwise"]
        )

        for row in group.iloc[1:].itertuples(
            index=False
        ):
            ratio_rows.append(
                {
                    "feature_signal": keys[0],
                    "model": keys[1],
                    "num_layers": keys[2],
                    "message_passing_stage":
                        row.message_passing_stage,
                    "previous_pairwise":
                        row.previous_pairwise,
                    "pairwise_mean":
                        row.pairwise_mean,
                    "contraction_ratio":
                        row.contraction_ratio,
                    "contraction_percent":
                        (
                            1.0
                            - row.contraction_ratio
                        )
                        * 100.0,
                }
            )

        first_value = float(
            group.iloc[0]["pairwise_mean"]
        )
        last_value = float(
            group.iloc[-1]["pairwise_mean"]
        )

        number_transitions = len(group) - 1

        if (
            first_value > 0
            and last_value > 0
            and number_transitions > 0
        ):
            geometric_factor = (
                last_value / first_value
            ) ** (
                1.0 / number_transitions
            )
        else:
            geometric_factor = np.nan

        summary_rows.append(
            {
                "feature_signal": keys[0],
                "model": keys[1],
                "num_layers": keys[2],
                "num_hidden_stages": len(group),
                "first_pairwise":
                    first_value,
                "last_pairwise":
                    last_value,
                "total_retained_fraction":
                    (
                        last_value / first_value
                        if first_value > 0
                        else np.nan
                    ),
                "geometric_retention_per_stage":
                    geometric_factor,
                "geometric_contraction_percent":
                    (
                        1.0 - geometric_factor
                    )
                    * 100.0,
                "first_stage_below_1e_3":
                    first_threshold_stage(
                        group,
                        1e-3,
                    ),
                "first_stage_below_1e_4":
                    first_threshold_stage(
                        group,
                        1e-4,
                    ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    ).sort_values(group_columns)

    ratios = pd.DataFrame(
        ratio_rows
    ).sort_values(
        group_columns
        + ["message_passing_stage"]
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    ratios.to_csv(
        RATIO_OUTPUT,
        index=False,
    )

    print(
        "\n=== Geometric contraction ==="
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.8f}",
        )
    )

    print("\nSaved:", SUMMARY_OUTPUT)
    print("Saved:", RATIO_OUTPUT)


if __name__ == "__main__":
    main()
