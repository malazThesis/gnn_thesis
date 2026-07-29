import os
import pandas as pd
import matplotlib.pyplot as plt

run_dir = os.path.expanduser("~/gnn_thesis_cluster/runs/realworld_phase2")
plot_dir = os.path.expanduser("~/gnn_thesis_cluster/plots/realworld_phase2_grouped_no_hidden")
os.makedirs(plot_dir, exist_ok=True)

csv_path = os.path.join(run_dir, "realworld_phase2_grouped.csv")
df = pd.read_csv(csv_path)

# Für jeden Datensatz, Modell und Layer die beste Hidden-Dimension wählen
best_hidden_df = (
    df.sort_values(
        by=["dataset", "model", "num_layers", "mean_best_test_acc", "mean_best_val_acc"],
        ascending=[True, True, True, False, False]
    )
    .groupby(["dataset", "model", "num_layers"], as_index=False)
    .first()
)

# Plot 1: Mean best test accuracy vs layers
for dataset_name in sorted(best_hidden_df["dataset"].unique()):
    plt.figure(figsize=(10, 6))
    subset = best_hidden_df[best_hidden_df["dataset"] == dataset_name]

    for model_name in sorted(subset["model"].unique()):
        model_subset = subset[subset["model"] == model_name].sort_values("num_layers")

        plt.errorbar(
            model_subset["num_layers"],
            model_subset["mean_best_test_acc"],
            yerr=model_subset["std_best_test_acc"],
            marker="o",
            capsize=4,
            label=model_name,
        )

    plt.xlabel("Number of Layers")
    plt.ylabel("Mean Best Test Accuracy")
    plt.title(f"Phase 2 Grouped Depth Comparison (best hidden size) - {dataset_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{dataset_name}_phase2_grouped_depth_comparison_no_hidden.png"))
    plt.close()

# Plot 2: Mean best validation accuracy vs layers
for dataset_name in sorted(best_hidden_df["dataset"].unique()):
    plt.figure(figsize=(10, 6))
    subset = best_hidden_df[best_hidden_df["dataset"] == dataset_name]

    for model_name in sorted(subset["model"].unique()):
        model_subset = subset[subset["model"] == model_name].sort_values("num_layers")

        plt.errorbar(
            model_subset["num_layers"],
            model_subset["mean_best_val_acc"],
            yerr=model_subset["std_best_val_acc"],
            marker="o",
            capsize=4,
            label=model_name,
        )

    plt.xlabel("Number of Layers")
    plt.ylabel("Mean Best Validation Accuracy")
    plt.title(f"Phase 2 Grouped Validation Comparison (best hidden size) - {dataset_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{dataset_name}_phase2_grouped_val_comparison_no_hidden.png"))
    plt.close()

# Plot 3: best model config per dataset
for dataset_name in sorted(df["dataset"].unique()):
    plt.figure(figsize=(11, 6))
    subset = df[df["dataset"] == dataset_name]

    best_rows = []
    for model_name in sorted(subset["model"].unique()):
        best_row = subset[subset["model"] == model_name].sort_values(
            by=["mean_best_test_acc", "mean_best_val_acc"],
            ascending=[False, False]
        ).iloc[0]

        best_rows.append({
            "label": model_name,
            "score": best_row["mean_best_test_acc"],
            "std": best_row["std_best_test_acc"],
        })

    plot_df = pd.DataFrame(best_rows)

    plt.bar(plot_df["label"], plot_df["score"], yerr=plot_df["std"], capsize=4)
    plt.ylabel("Mean Best Test Accuracy")
    plt.title(f"Best Grouped Model Configurations - {dataset_name}")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{dataset_name}_phase2_grouped_best_models_bar_no_hidden.png"))
    plt.close()

print(f"Saved plots to: {plot_dir}")
