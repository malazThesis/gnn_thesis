import os
import json
import pandas as pd

run_dir = os.path.expanduser("~/gnn_thesis_cluster/runs/realworld_phase1")
out_csv = os.path.join(run_dir, "realworld_phase1_summary.csv")
out_best_csv = os.path.join(run_dir, "realworld_phase1_best_by_dataset.csv")

rows = []

for fname in os.listdir(run_dir):
    if fname.endswith("_summary.json"):
        path = os.path.join(run_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            rows.append(json.load(f))

df = pd.DataFrame(rows)

# sort for readability
df = df.sort_values(
    by=["dataset", "model", "num_layers", "hidden_channels", "seed"]
).reset_index(drop=True)

df.to_csv(out_csv, index=False)

# best config per dataset according to best_test_acc_at_best_val
best_df = (
    df.sort_values(
        by=["dataset", "best_test_acc_at_best_val", "best_val_acc"],
        ascending=[True, False, False]
    )
    .groupby("dataset", as_index=False)
    .first()
)

best_df.to_csv(out_best_csv, index=False)

print(f"Saved full summary to: {out_csv}")
print(f"Saved best-per-dataset summary to: {out_best_csv}")

print("\n=== Full table head ===")
print(df.head(20).to_string(index=False))

print("\n=== Best config per dataset ===")
print(best_df.to_string(index=False))
