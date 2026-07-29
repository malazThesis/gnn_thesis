from pathlib import Path

import pandas as pd


DATASETS = {
    "Amazon-Computers": "amazon_computers",
    "Coauthor-CS": "coauthor_cs",
    "Coauthor-Physics": "coauthor_physics",
    "Actor": "actor",
    "Chameleon": "chameleon",
    "Squirrel": "squirrel",
    "Cornell": "cornell",
    "Texas": "texas",
    "Wisconsin": "wisconsin",
    "Amazon-Ratings": "amazon_ratings",
}

PHASES = {
    "Phase 1": "phase1",
    "Phase 2": "phase2",
}

OUTPUT_PATH = Path(
    "runs/realworld_extension_pairnorm_effects.csv"
)


def get_best(df):
    return (
        df.sort_values(
            [
                "mean_best_test_acc",
                "mean_best_val_acc",
            ],
            ascending=[False, False],
        )
        .iloc[0]
    )


def main():
    rows = []

    for dataset_name, dataset_slug in DATASETS.items():
        for phase_name, phase_slug in PHASES.items():
            baseline_path = Path(
                f"runs/{dataset_slug}_{phase_slug}_baselines/"
                f"{dataset_slug}_{phase_slug}_baselines_grouped.csv"
            )

            full_path = Path(
                f"runs/{dataset_slug}_{phase_slug}_full/"
                f"{dataset_slug}_{phase_slug}_full_grouped.csv"
            )

            if not baseline_path.exists():
                raise FileNotFoundError(baseline_path)

            if not full_path.exists():
                raise FileNotFoundError(full_path)

            baseline_df = pd.read_csv(baseline_path)
            full_df = pd.read_csv(full_path)

            pairnorm_df = full_df[
                full_df["model"].str.endswith(
                    "PairNorm",
                    na=False,
                )
            ].copy()

            if pairnorm_df.empty:
                raise RuntimeError(
                    f"No PairNorm rows found in {full_path}"
                )

            best_baseline = get_best(baseline_df)
            best_pairnorm = get_best(pairnorm_df)
            best_overall = get_best(full_df)

            pairnorm_gain = (
                best_pairnorm["mean_best_test_acc"]
                - best_baseline["mean_best_test_acc"]
            )

            rows.append(
                {
                    "dataset": dataset_name,
                    "phase": phase_name,
                    "baseline_model": best_baseline["model"],
                    "baseline_layers": int(
                        best_baseline["num_layers"]
                    ),
                    "baseline_hidden": int(
                        best_baseline["hidden_channels"]
                    ),
                    "baseline_mean_test": best_baseline[
                        "mean_best_test_acc"
                    ],
                    "baseline_std_test": best_baseline[
                        "std_best_test_acc"
                    ],
                    "pairnorm_model": best_pairnorm["model"],
                    "pairnorm_layers": int(
                        best_pairnorm["num_layers"]
                    ),
                    "pairnorm_hidden": int(
                        best_pairnorm["hidden_channels"]
                    ),
                    "pairnorm_mean_test": best_pairnorm[
                        "mean_best_test_acc"
                    ],
                    "pairnorm_std_test": best_pairnorm[
                        "std_best_test_acc"
                    ],
                    "pairnorm_gain": pairnorm_gain,
                    "pairnorm_gain_percentage_points": (
                        100.0 * pairnorm_gain
                    ),
                    "overall_winner": best_overall["model"],
                    "pairnorm_is_overall_winner": bool(
                        str(best_overall["model"]).endswith(
                            "PairNorm"
                        )
                    ),
                }
            )

    result = pd.DataFrame(rows)

    result = result.sort_values(
        ["phase", "pairnorm_gain"],
        ascending=[True, False],
    ).reset_index(drop=True)

    result.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")

    display_columns = [
        "dataset",
        "phase",
        "baseline_model",
        "pairnorm_model",
        "baseline_mean_test",
        "pairnorm_mean_test",
        "pairnorm_gain_percentage_points",
        "pairnorm_is_overall_winner",
    ]

    print(
        result[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


if __name__ == "__main__":
    main()
