import os
import pandas as pd
import matplotlib.pyplot as plt

run_dir = os.path.expanduser("~/gnn_thesis_cluster/runs/realworld_phase2")
plot_dir = os.path.expanduser("~/gnn_thesis_cluster/plots/realworld_phase2_grouped")
os.makedirs(plot_dir, exist_ok=True)

csv_path = os.path.join(run_dir, "realworld_phase2_grouped.csv")
df = pd.read_csv(csv_path)

# Plot 1: mean best test acc vs layers, grouped by model + hidden_channels
for dataset_name in sorted(df["dataset"].unique()):
    plt.figure(figsize=(11, 7))
    subset = df[df["dataset"] == dataset_name]

    for model_name in sorted(subset["model"].unique()):
        for hidden_channels in sorted(subset["hidden_channels"].unique()):
            line_subset = subset[
                (subset["model"] == model_name) &
                (subset["hidden_channels"] == hidden_channels)
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
    plt.title(f"Phase 2 Grouped Depth Comparison - {dataset_name}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{dataset_name}_phase2_grouped_depth_comparison.png"))
    plt.close()

# Plot 2: mean best validation acc vs layers, grouped by model + hidden_channels
for dataset_name in sorted(df["dataset"].unique()):
    plt.figure(figsize=(11, 7))
    subset = df[df["dataset"] == dataset_name]

    for model_name in sorted(subset["model"].unique()):
        for hidden_channels in sorted(subset["hidden_channels"].unique()):
            line_subset = subset[
                (subset["model"] == model_name) &
                (subset["hidden_channels"] == hidden_channels)
            ].sort_values("num_layers")

            if len(line_subset) == 0:
                continue

            plt.errorbar(
                line_subset["num_layers"],
                line_subset["mean_best_val_acc"],
                yerr=line_subset["std_best_val_acc"],
                marker="o",
                capsize=4,
                label=f"{model_name} (H={hidden_channels})",
            )

    plt.xlabel("Number of Layers")
    plt.ylabel("Mean Best Validation Accuracy")
    plt.title(f"Phase 2 Grouped Validation Comparison - {dataset_name}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{dataset_name}_phase2_grouped_val_comparison.png"))
    plt.close()

# Plot 3: best grouped configuration bar plot per dataset with L and H in labels
for dataset_name in sorted(df["dataset"].unique()):
    plt.figure(figsize=(12, 7))
    subset = df[df["dataset"] == dataset_name]

    best_rows = []
    for model_name in sorted(subset["model"].unique()):
        best_row = subset[subset["model"] == model_name].sort_values(
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
    plt.title(f"Best Grouped Model Configurations - {dataset_name}")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{dataset_name}_phase2_grouped_best_models_bar.png"))
    plt.close()

print(f"Saved plots to: {plot_dir}")
