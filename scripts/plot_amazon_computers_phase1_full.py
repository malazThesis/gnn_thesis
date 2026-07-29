import os
import pandas as pd
import matplotlib.pyplot as plt

run_dir = os.path.expanduser("runs/amazon_computers_phase1_full")
plot_dir = os.path.expanduser("plots/amazon_computers_phase1_full")
os.makedirs(plot_dir, exist_ok=True)

csv_path = os.path.join(run_dir, "amazon_computers_phase1_full_grouped.csv")
df = pd.read_csv(csv_path)

plt.figure(figsize=(12, 7))
for model_name in sorted(df["model"].unique()):
    for hidden_channels in sorted(df["hidden_channels"].unique()):
        line_subset = df[
            (df["model"] == model_name) &
            (df["hidden_channels"] == hidden_channels)
        ].sort_values("num_layers")

        if len(line_subset) == 0:
            continue

        plt.errorbar(
            line_subset["num_layers"],
            line_subset["mean_best_test_acc"],
            yerr=line_subset["std_best_test_acc"],
            marker="o",
            capsize=4,
            label=f"{model_name} (H={hidden_channels})",
        )

plt.xlabel("Number of Layers")
plt.ylabel("Mean Best Test Accuracy")
plt.title("Amazon Computers Phase 1 Full")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "amazon_computers_phase1_full_depth.png"))
plt.close()

plt.figure(figsize=(12, 6))
best_rows = []
for model_name in sorted(df["model"].unique()):
    best_row = df[df["model"] == model_name].sort_values(
        by=["mean_best_test_acc", "mean_best_val_acc"],
        ascending=[False, False]
    ).iloc[0]

    best_rows.append({
        "label": f"{model_name}\nL={int(best_row['num_layers'])}, H={int(best_row['hidden_channels'])}",
        "score": best_row["mean_best_test_acc"],
        "std": best_row["std_best_test_acc"],
    })

plot_df = pd.DataFrame(best_rows)

plt.bar(plot_df["label"], plot_df["score"], yerr=plot_df["std"], capsize=4)
plt.ylabel("Mean Best Test Accuracy")
plt.title("Amazon Computers Phase 1 Full - Best Config per Model")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "amazon_computers_phase1_full_best_bar.png"))
plt.close()

baseline_models = ["GCN", "GAT", "GraphSAGE"]
plt.figure(figsize=(11, 6))
for model_name in baseline_models:
    for hidden_channels in sorted(df["hidden_channels"].unique()):
        line_subset = df[
            (df["model"] == model_name) &
            (df["hidden_channels"] == hidden_channels)
        ].sort_values("num_layers")

        if len(line_subset) == 0:
            continue

        plt.errorbar(
            line_subset["num_layers"],
            line_subset["mean_best_test_acc"],
            yerr=line_subset["std_best_test_acc"],
            marker="o",
            capsize=4,
            label=f"{model_name} (H={hidden_channels})",
        )

plt.xlabel("Number of Layers")
plt.ylabel("Mean Best Test Accuracy")
plt.title("Amazon Computers Phase 1 Baselines within Full Block")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "amazon_computers_phase1_full_baselines_only.png"))
plt.close()

print(f"Saved plots to: {plot_dir}")
