import json
import os

import pandas as pd


RUN_DIR = "runs/actor_phase2_baselines"

FULL_CSV = os.path.join(
    RUN_DIR,
    "actor_phase2_baselines_summary.csv",
)
GROUPED_CSV = os.path.join(
    RUN_DIR,
    "actor_phase2_baselines_grouped.csv",
)
BEST_CSV = os.path.join(
    RUN_DIR,
    "actor_phase2_baselines_best_grouped.csv",
)


def main():
    rows = []

    for filename in os.listdir(RUN_DIR):
        if not filename.endswith("_summary.json"):
            continue

        path = os.path.join(RUN_DIR, filename)

        with open(path, "r", encoding="utf-8") as file:
            rows.append(json.load(file))

    if not rows:
        raise RuntimeError(
            f"No *_summary.json files found in {RUN_DIR}"
        )

    df = pd.DataFrame(rows)

    required_columns = {
        "dataset",
        "model",
        "num_layers",
        "hidden_channels",
        "split_idx",
        "best_test_acc_at_best_val",
        "best_val_acc",
        "final_test_acc",
        "best_epoch",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df = df.sort_values(
        [
            "dataset",
            "model",
            "num_layers",
            "hidden_channels",
            "split_idx",
        ]
    ).reset_index(drop=True)

    df.to_csv(FULL_CSV, index=False)

    grouped = (
        df.groupby(
            [
                "dataset",
                "model",
                "num_layers",
                "hidden_channels",
            ],
            as_index=False,
        )
        .agg(
            mean_best_test_acc=(
                "best_test_acc_at_best_val",
                "mean",
            ),
            std_best_test_acc=(
                "best_test_acc_at_best_val",
                "std",
            ),
            mean_best_val_acc=("best_val_acc", "mean"),
            std_best_val_acc=("best_val_acc", "std"),
            mean_final_test_acc=("final_test_acc", "mean"),
            std_final_test_acc=("final_test_acc", "std"),
            mean_best_epoch=("best_epoch", "mean"),
            std_best_epoch=("best_epoch", "std"),
            num_splits=("split_idx", "count"),
        )
    )

    std_columns = [
        "std_best_test_acc",
        "std_best_val_acc",
        "std_final_test_acc",
        "std_best_epoch",
    ]

    grouped[std_columns] = grouped[std_columns].fillna(0.0)

    grouped = grouped.sort_values(
        ["mean_best_test_acc", "mean_best_val_acc"],
        ascending=[False, False],
    ).reset_index(drop=True)

    grouped.to_csv(GROUPED_CSV, index=False)

    best_df = grouped.head(1).copy()
    best_df.to_csv(BEST_CSV, index=False)

    print(f"Loaded runs: {len(df)}")
    print(f"Grouped configurations: {len(grouped)}")
    print(f"Saved: {FULL_CSV}")
    print(f"Saved: {GROUPED_CSV}")
    print(f"Saved: {BEST_CSV}")

    print("\n=== Best grouped config ===")
    print(best_df.to_string(index=False))


if __name__ == "__main__":
    main()
