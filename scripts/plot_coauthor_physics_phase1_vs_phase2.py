import os
import pandas as pd
import matplotlib.pyplot as plt

plot_dir = os.path.expanduser("plots/coauthor_physics_phase1_vs_phase2")
os.makedirs(plot_dir, exist_ok=True)

p1_base = pd.read_csv("runs/coauthor_physics_phase1_baselines/coauthor_physics_phase1_baselines_grouped.csv")
p2_base = pd.read_csv("runs/coauthor_physics_phase2_baselines/coauthor_physics_phase2_baselines_grouped.csv")
p1_full = pd.read_csv("runs/coauthor_physics_phase1_full/coauthor_physics_phase1_full_grouped.csv")
p2_full = pd.read_csv("runs/coauthor_physics_phase2_full/coauthor_physics_phase2_full_grouped.csv")

baseline_models = ["GCN", "GAT", "GraphSAGE"]

plt.figure(figsize=(12, 6))
rows = []
for phase_name, df in [("Phase1", p1_base), ("Phase2", p2_base)]:
    for model_name in baseline_models:
        best = df[df["model"] == model_name].sort_values(
            by=["mean_best_test_acc", "mean_best_val_acc"],
            ascending=[False, False]
        ).iloc[0]
        rows.append({
            "label": f"{model_name}\n{phase_name}\nL={int(best['num_layers'])}, H={int(best['hidden_channels'])}",
            "score": best["mean_best_test_acc"],
            "std": best["std_best_test_acc"],
        })

plot_df = pd.DataFrame(rows)
plt.bar(plot_df["label"], plot_df["score"], yerr=plot_df["std"], capsize=4)
plt.ylabel("Mean Best Test Accuracy")
plt.title("Coauthor Physics - Baselines: Phase 1 vs Phase 2")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "coauthor_physics_baselines_phase1_vs_phase2_bar.png"))
plt.close()

plt.figure(figsize=(13, 6))
rows = []
for phase_name, df in [("Phase1", p1_full), ("Phase2", p2_full)]:
    for model_name in sorted(df["model"].unique()):
        best = df[df["model"] == model_name].sort_values(
            by=["mean_best_test_acc", "mean_best_val_acc"],
            ascending=[False, False]
        ).iloc[0]
        rows.append({
            "label": f"{model_name}\n{phase_name}\nL={int(best['num_layers'])}, H={int(best['hidden_channels'])}",
            "score": best["mean_best_test_acc"],
            "std": best["std_best_test_acc"],
        })

plot_df = pd.DataFrame(rows)
plt.bar(plot_df["label"], plot_df["score"], yerr=plot_df["std"], capsize=4)
plt.ylabel("Mean Best Test Accuracy")
plt.title("Coauthor Physics - Full: Phase 1 vs Phase 2")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "coauthor_physics_full_phase1_vs_phase2_bar.png"))
plt.close()

print(f"Saved plots to: {plot_dir}")
