#!/usr/bin/env python3
"""Rebuild the 15-dataset/60-block broad-screening selection from existing grouped CSV files.

Select on mean validation accuracy only. At exact ties prefer fewer layers,
then fewer hidden channels, then alphabetically earlier model names.
No training and no test-accuracy-based selection.
Run from the gnn_thesis_cluster repository root.
"""
from pathlib import Path
import pandas as pd

DATASETS = (
    "Cora", "CiteSeer", "PubMed", "Roman-Empire", "Amazon-Photo",
    "Amazon-Computers", "Coauthor-CS", "Coauthor-Physics", "Actor",
    "Chameleon", "Squirrel", "Cornell", "Texas", "Wisconsin",
    "Amazon-Ratings",
)
BLOCKS = (
    ("phase1_baselines", 1, "baseline"),
    ("phase1_full", 1, "full"),
    ("phase2_baselines", 2, "baseline"),
    ("phase2_full", 2, "full"),
)

def main():
    rows = []
    for dataset in DATASETS:
        slug = dataset.lower().replace("-", "_")
        for block, phase, search in BLOCKS:
            path = Path("runs") / f"{slug}_{block}" / f"{slug}_{block}_grouped.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Missing grouped summary: {path}")
            df = pd.read_csv(path)
            required = {
                "model", "num_layers", "hidden_channels",
                "mean_best_val_acc", "mean_best_test_acc", "std_best_test_acc",
            }
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"{path}: missing columns: {sorted(missing)}")

            df = df.sort_values(
                ["mean_best_val_acc", "num_layers", "hidden_channels", "model"],
                ascending=[False, True, True, True],
                kind="stable",
            )
            chosen = df.iloc[0]
            max_val = chosen["mean_best_val_acc"]
            tied = df[df["mean_best_val_acc"] == max_val]
            rows.append({
                "dataset": dataset,
                "phase": phase,
                "search": search,
                "model": chosen["model"],
                "num_layers": int(chosen["num_layers"]),
                "hidden_channels": int(chosen["hidden_channels"]),
                "mean_val_pct": 100.0 * chosen["mean_best_val_acc"],
                "mean_test_pct": 100.0 * chosen["mean_best_test_acc"],
                "sd_test_pct": 100.0 * chosen["std_best_test_acc"],
                "num_tied_for_best_val": len(tied),
                "source": str(path),
            })

    out = Path("runs/broad_screening_validation_selected_rebuilt.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    if len(result) != 60:
        raise AssertionError(f"Expected 60 entries, found {len(result)}")
    result.to_csv(out, index=False)
    print(f"Saved {len(result)} validation-selected rows to {out}")
    print("Validation ties:")
    print(result[result.num_tied_for_best_val > 1][
        ["dataset", "phase", "search", "model", "num_layers", "hidden_channels"]
    ].to_string(index=False))

if __name__ == "__main__":
    main()
