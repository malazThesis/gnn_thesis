import os
import json
import pandas as pd

run_dir = os.path.expanduser("~/gnn_thesis_cluster/runs/roman_empire_phase2_baselines")
out_full_csv = os.path.join(run_dir, "roman_empire_phase2_baselines_summary.csv")
out_grouped_csv = os.path.join(run_dir, "roman_empire_phase2_baselines_grouped.csv")
out_best_csv = os.path.join(run_dir, "roman_empire_phase2_baselines_best_grouped.csv")

rows = []

for fname in os.listdir(run_dir):
    if fname.endswith("_summary.json"):
        path = os.path.join(run_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            rows.append(json.load(f))

df = pd.DataFrame(rows)

df = df.sort_values(
    by=["dataset", "model", "num_layers", "hidden_channels", "split_idx"]
).reset_index(drop=True)
df.to_csv(out_full_csv, index=False)

grouped = (
    df.groupby(["dataset", "model", "num_layers", "hidden_channels"], as_index=False)
      .agg(
          mean_best_test_acc=("best_test_acc_at_best_val", "mean"),
          std_best_test_acc=("best_test_acc_at_best_val", "std"),
          mean_best_val_acc=("best_val_acc", "mean"),
          std_best_val_acc=("best_val_acc", "std"),
          mean_final_test_acc=("final_test_acc", "mean"),
          std_final_test_acc=("final_test_acc", "std"),
          mean_best_epoch=("best_epoch", "mean"),
          std_best_epoch=("best_epoch", "std"),
          num_splits=("split_idx", "count"),
      )
)

grouped["std_best_test_acc"] = grouped["std_best_test_acc"].fillna(0.0)
grouped["std_best_val_acc"] = grouped["std_best_val_acc"].fillna(0.0)
grouped["std_final_test_acc"] = grouped["std_final_test_acc"].fillna(0.0)
grouped["std_best_epoch"] = grouped["std_best_epoch"].fillna(0.0)

grouped = grouped.sort_values(
    by=["mean_best_test_acc", "mean_best_val_acc"],
    ascending=[False, False]
).reset_index(drop=True)

grouped.to_csv(out_grouped_csv, index=False)

best_df = (
    grouped.sort_values(
        by=["mean_best_test_acc", "mean_best_val_acc"],
        ascending=[False, False]
    )
    .groupby("dataset", as_index=False)
    .first()
)

best_df.to_csv(out_best_csv, index=False)

print(f"Saved full summary to: {out_full_csv}")
print(f"Saved grouped summary to: {out_grouped_csv}")
print(f"Saved best grouped summary to: {out_best_csv}")
print("\n=== Best grouped config ===")
print(best_df.to_string(index=False))
