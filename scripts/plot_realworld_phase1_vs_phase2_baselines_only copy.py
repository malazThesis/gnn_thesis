import os
import pandas as pd
import matplotlib.pyplot as plt

phase1_path = os.path.expanduser(
    "~/gnn_thesis_cluster/runs/realworld_phase1_gs/realworld_phase1_gs_grouped.csv"
)
phase2_path = os.path.expanduser(
    "~/gnn_thesis_cluster/runs/realworld_phase2/realworld_phase2_grouped.csv"
)

plot_dir = os.path.expanduser(
    "~/gnn_thesis_cluster/plots/realworld_phase1_vs_phase2_baselines_only"
)
os.makedirs(plot_dir, exist_ok=True)

phase1_df = pd.read_csv(phase1_path)
phase2_df = pd.read_csv(phase2_path)

phase1_df["phase"] = "Phase1"
phase2_df["phase"] = "Phase2"

df = pd.concat([phase1_df, phase2_df], ignore_index=True)

baseline_models = ["GCN", "GAT", "GraphSAGE"]
df = df[df["model"].isin(baseline_models)].copy()

# Für jede Kombination beste hidden dimension wählen
best_hidden_df = (
    df.sort_values(
        by=[
            "dataset",
            "phase",
            "model",
            "num_layers",
            "mean_best_test_acc",
            "mean_best_val_acc",
        ],
        ascending=[True, True, True, True, False, False],
    )
    .groupby(["dataset", "phase", "model", "num_layers"], as_index=False)
    .first()
)

# Plot 1: Depth comparison, nur Baselines
for dataset_name in sorted(best_hidden_df["dataset"].unique()):
    plt.figure(figsize=(11, 7))
    subset = best_hidden_df[best_hidden_df["dataset"] == dataset_name]

    for model_name in baseline_models:
        model_subset = subset[subset["model"] == model_name].sort_values("num_layers")

        if len(model_subset) == 0:
            continue

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
    plt.title(f"Phase 1 + Phase 2 Baseline Models Only - {dataset_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(plot_dir, f"{dataset_name}_phase1_vs_phase2_baselines_only.png")
    )
    plt.close()

# Plot 2: Best config comparison, nur Baselines
for dataset_name in sorted(df["dataset"].unique()):
    plt.figure(figsize=(12, 7))
    subset = df[df["dataset"] == dataset_name]

    best_rows = []
    for model_name in baseline_models:
        model_subset = subset[subset["model"] == model_name]
        if len(model_subset) == 0:
            continue

        best_row = model_subset.sort_values(
            by=["mean_best_test_acc", "mean_best_val_acc"],
            ascending=[False, False]
        ).iloc[0]

        best_rows.append({
            "label": (
                f"{model_name}\n"
                f"{best_row['phase']}, "
                f"L={int(best_row['num_layers'])}, "
                f"H={int(best_row['hidden_channels'])}"
            ),
            "score": best_row["mean_best_test_acc"],
            "std": best_row["std_best_test_acc"],
        })

    plot_df = pd.DataFrame(best_rows)

    plt.bar(plot_df["label"], plot_df["score"], yerr=plot_df["std"], capsize=4)
    plt.ylabel("Mean Best Test Accuracy")
    plt.title(f"Best Baseline Model Configurations - {dataset_name}")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(
        os.path.join(plot_dir, f"{dataset_name}_phase1_vs_phase2_baselines_best_bar.png")
    )
    plt.close()

print(f"Saved plots to: {plot_dir}")
