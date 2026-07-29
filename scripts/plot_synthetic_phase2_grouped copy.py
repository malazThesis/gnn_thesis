import os
import pandas as pd
import matplotlib.pyplot as plt

csv_path = os.path.expanduser("~/gnn_thesis_cluster/runs/synthetic_phase2_grouped.csv")
plot_dir = os.path.expanduser("~/gnn_thesis_cluster/plots/synthetic_phase2_grouped")
os.makedirs(plot_dir, exist_ok=True)

df = pd.read_csv(csv_path)

# Pro Topologie und Modell: Accuracy vs num_layers, Linien nach distance
for topology in sorted(df["topology"].unique()):
    for model_name in sorted(df["model"].unique()):
        plt.figure(figsize=(10, 6))
        subset = df[(df["topology"] == topology) & (df["model"] == model_name)]

        for distance in sorted(subset["distance"].unique()):
            line_subset = subset[subset["distance"] == distance].sort_values("num_layers")

            plt.errorbar(
                line_subset["num_layers"],
                line_subset["mean_best_test_acc"],
                yerr=line_subset["std_best_test_acc"],
                marker="o",
                capsize=4,
                label=f"D={distance}",
            )

        plt.axhline(y=1.0 / 8.0, linestyle="--", label="Chance (0.125)")
        plt.xlabel("Number of Layers")
        plt.ylabel("Mean Best Test Accuracy")
        plt.title(f"Synthetic Phase 2 Depth Sweep - {topology} - {model_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{topology}_{model_name}_depth_sweep.png"))
        plt.close()

print(f"Saved plots to: {plot_dir}")
