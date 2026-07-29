import os
import pandas as pd
import matplotlib.pyplot as plt

run_dir = os.path.expanduser("~/gnn_thesis_cluster/runs/realworld_phase1_gs")
plot_dir = os.path.expanduser("~/gnn_thesis_cluster/plots/realworld_phase1_gs")
os.makedirs(plot_dir, exist_ok=True)

csv_path = os.path.join(run_dir, "realworld_phase1_gs_summary.csv")
df = pd.read_csv(csv_path)

for dataset_name in sorted(df["dataset"].unique()):
    plt.figure(figsize=(10, 6))
    subset = df[df["dataset"] == dataset_name]

    for model_name in sorted(subset["model"].unique()):
        model_subset = (
            subset[subset["model"] == model_name]
            .groupby("num_layers", as_index=False)["best_test_acc_at_best_val"]
            .max()
            .sort_values("num_layers")
        )

        plt.plot(
            model_subset["num_layers"],
            model_subset["best_test_acc_at_best_val"],
            marker="o",
            label=model_name,
        )

    plt.xlabel("Number of Layers")
    plt.ylabel("Best Test Accuracy at Best Validation")
    plt.title(f"Phase 1 Depth Comparison - {dataset_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{dataset_name}_phase1_depth_comparison.png"))
    plt.close()

for dataset_name in sorted(df["dataset"].unique()):
    plt.figure(figsize=(11, 6))
    subset = df[df["dataset"] == dataset_name]

    best_rows = []
    for model_name in sorted(subset["model"].unique()):
        best_row = subset[subset["model"] == model_name].sort_values(
            by=["best_test_acc_at_best_val", "best_val_acc"],
            ascending=[False, False]
        ).iloc[0]
        best_rows.append({
            "model": model_name,
            "score": best_row["best_test_acc_at_best_val"],
            "hidden_channels": best_row["hidden_channels"],
            "num_layers": best_row["num_layers"],
        })

    plot_df = pd.DataFrame(best_rows)

    plt.bar(plot_df["model"], plot_df["score"])
    plt.ylabel("Best Test Accuracy at Best Validation")
    plt.title(f"Best Model Configurations - {dataset_name}")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{dataset_name}_phase1_best_models_bar.png"))
    plt.close()

print(f"Saved plots to: {plot_dir}")
