import os
import pandas as pd
import matplotlib.pyplot as plt

plot_dir = os.path.expanduser("plots/amazon_computers_phase1_vs_phase2")
os.makedirs(plot_dir, exist_ok=True)

p1_base = pd.read_csv("runs/amazon_computers_phase1_baselines/amazon_computers_phase1_baselines_grouped.csv")
p2_base = pd.read_csv("runs/amazon_computers_phase2_baselines/amazon_computers_phase2_baselines_grouped.csv")
p1_full = pd.read_csv("runs/amazon_computers_phase1_full/amazon_computers_phase1_full_grouped.csv")
p2_full = pd.read_csv("runs/amazon_computers_phase2_full/amazon_computers_phase2_full_grouped.csv")

baseline_models = ["GCN", "GAT", "GraphSAGE"]

# Baselines: Phase 1 vs Phase 2
plt.figure(figsize=(12, 6))
rows = []
for phase_name, df in [("Phase1", p1_base), ("Phase2", p2_base)]:
    for model_name in baseline_models:
        best_row = df[df["model"] == model_name].sort_values(
            by=["mean_best_test_acc", "mean_best_val_acc"],
            ascending=[False, False]
        ).iloc[0]
        rows.append({
            "label": f"{model_name}\n{phase_name}\nL={int(best_row['num_layers'])}, H={int(best_row['hidden_channels'])}",
            "score": best_row["mean_best_test_acc"],
            "std": best_row["std_best_test_acc"],
        })

plot_df = pd.DataFrame(rows)
plt.bar(plot_df["label"], plot_df["score"], yerr=plot_df["std"], capsize=4)
plt.ylabel("Mean Best Test Accuracy")
plt.title("Amazon Computers - Baselines: Phase 1 vs Phase 2")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "amazon_computers_baselines_phase1_vs_phase2_bar.png"))
plt.close()

# Full: Phase 1 vs Phase 2
plt.figure(figsize=(13, 6))
rows = []
for phase_name, df in [("Phase1", p1_full), ("Phase2", p2_full)]:
    for model_name in sorted(df["model"].unique()):
        best_row = df[df["model"] == model_name].sort_values(
            by=["mean_best_test_acc", "mean_best_val_acc"],
            ascending=[False, False]
        ).iloc[0]
        rows.append({
            "label": f"{model_name}\n{phase_name}\nL={int(best_row['num_layers'])}, H={int(best_row['hidden_channels'])}",
            "score": best_row["mean_best_test_acc"],
            "std": best_row["std_best_test_acc"],
        })

plot_df = pd.DataFrame(rows)
plt.bar(plot_df["label"], plot_df["score"], yerr=plot_df["std"], capsize=4)
plt.ylabel("Mean Best Test Accuracy")
plt.title("Amazon Computers - Full: Phase 1 vs Phase 2")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "amazon_computers_full_phase1_vs_phase2_bar.png"))
plt.close()

print(f"Saved plots to: {plot_dir}")
