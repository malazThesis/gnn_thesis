import os
import pandas as pd
import matplotlib.pyplot as plt

run_dir = os.path.expanduser("~/gnn_thesis_cluster/runs/synthetic_crossed_ring_phase1")
plot_dir = os.path.expanduser("~/gnn_thesis_cluster/plots/synthetic_crossed_ring_phase1_grouped")
os.makedirs(plot_dir, exist_ok=True)

csv_path = os.path.join(run_dir, "synthetic_crossed_ring_phase1_grouped.csv")
df = pd.read_csv(csv_path)

# Plot 1: per model, accuracy vs distance, separate lines for hidden size
for model_name in sorted(df["model"].unique()):
    plt.figure(figsize=(10, 6))
    subset = df[df["model"] == model_name]

    for hidden_channels in sorted(subset["hidden_channels"].unique()):
        line_subset = subset[subset["hidden_channels"] == hidden_channels].sort_values("distance")

        plt.errorbar(
            line_subset["distance"],
            line_subset["mean_best_test_acc"],
            yerr=line_subset["std_best_test_acc"],
            marker="o",
            capsize=4,
            label=f"H={hidden_channels}",
        )

    chance_level = 1.0 / 8.0
    plt.axhline(y=chance_level, linestyle="--", label="Chance (0.125)")
    plt.xlabel("Distance")
    plt.ylabel("Mean Best Test Accuracy")
    plt.title(f"CrossedRing Phase 1 - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"CrossedRing_phase1_{model_name}_distance_width.png"))
    plt.close()

# Plot 2: best hidden size per model as bar plot
for model_name in sorted(df["model"].unique()):
    plt.figure(figsize=(8, 5))
    subset = df[df["model"] == model_name]

    best_rows = []
    for distance in sorted(subset["distance"].unique()):
        best_row = subset[subset["distance"] == distance].sort_values(
            by=["mean_best_test_acc", "mean_final_test_acc"],
            ascending=[False, False]
        ).iloc[0]

        best_rows.append({
            "label": f"D={int(best_row['distance'])}\nH={int(best_row['hidden_channels'])}",
            "score": best_row["mean_best_test_acc"],
            "std": best_row["std_best_test_acc"],
        })

    plot_df = pd.DataFrame(best_rows)

    plt.bar(plot_df["label"], plot_df["score"], yerr=plot_df["std"], capsize=4)
    plt.ylabel("Mean Best Test Accuracy")
    plt.title(f"CrossedRing Phase 1 Best Width by Distance - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"CrossedRing_phase1_{model_name}_best_width_bar.png"))
    plt.close()

print(f"Saved plots to: {plot_dir}")
