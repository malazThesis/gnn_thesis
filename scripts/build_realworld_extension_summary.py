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

BLOCKS = {
    "Phase 1 Baselines": "phase1_baselines",
    "Phase 2 Baselines": "phase2_baselines",
    "Phase 1 Full": "phase1_full",
    "Phase 2 Full": "phase2_full",
}

OUTPUT_BEST = Path(
    "runs/realworld_extension_best_summary.csv"
)
OUTPUT_COMPARISON = Path(
    "runs/realworld_extension_comparison_summary.csv"
)


def main():
    rows = []
    missing = []

    for dataset_name, dataset_slug in DATASETS.items():
        for block_name, block_slug in BLOCKS.items():
            csv_path = Path(
                f"runs/{dataset_slug}_{block_slug}/"
                f"{dataset_slug}_{block_slug}_best_grouped.csv"
            )

            if not csv_path.exists():
                missing.append(str(csv_path))
                continue

            df = pd.read_csv(csv_path)

            if len(df) != 1:
                raise ValueError(
                    f"Expected exactly one row in {csv_path}, "
                    f"found {len(df)}"
                )

            row = df.iloc[0].to_dict()
            row["dataset_display"] = dataset_name
            row["block"] = block_name
            rows.append(row)

    if missing:
        print("Missing files:")
        for path in missing:
            print(f"  {path}")
        raise SystemExit(1)

    best_df = pd.DataFrame(rows)

    column_order = [
        "dataset_display",
        "block",
        "model",
        "num_layers",
        "hidden_channels",
        "mean_best_test_acc",
        "std_best_test_acc",
        "mean_best_val_acc",
        "std_best_val_acc",
        "mean_final_test_acc",
        "std_final_test_acc",
        "mean_best_epoch",
        "std_best_epoch",
        "num_splits",
    ]

    best_df = best_df[column_order]
    best_df.to_csv(OUTPUT_BEST, index=False)

    score_table = best_df.pivot(
        index="dataset_display",
        columns="block",
        values="mean_best_test_acc",
    )

    comparison = pd.DataFrame(
        {
            "dataset": score_table.index,
            "phase1_baseline": score_table[
                "Phase 1 Baselines"
            ].values,
            "phase2_baseline": score_table[
                "Phase 2 Baselines"
            ].values,
            "phase1_full": score_table[
                "Phase 1 Full"
            ].values,
            "phase2_full": score_table[
                "Phase 2 Full"
            ].values,
        }
    )

    comparison["baseline_depth_drop"] = (
        comparison["phase1_baseline"]
        - comparison["phase2_baseline"]
    )

    comparison["full_depth_drop"] = (
        comparison["phase1_full"]
        - comparison["phase2_full"]
    )

    comparison["pairnorm_gain_phase1"] = (
        comparison["phase1_full"]
        - comparison["phase1_baseline"]
    )

    comparison["pairnorm_gain_phase2"] = (
        comparison["phase2_full"]
        - comparison["phase2_baseline"]
    )

    comparison = comparison.sort_values(
        "dataset"
    ).reset_index(drop=True)

    comparison.to_csv(
        OUTPUT_COMPARISON,
        index=False,
    )

    print(f"Saved: {OUTPUT_BEST}")
    print(f"Saved: {OUTPUT_COMPARISON}")

    print("\n=== Cross-dataset comparison ===")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


if __name__ == "__main__":
    main()
