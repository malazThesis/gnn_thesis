import os
import pandas as pd
import matplotlib.pyplot as plt

DATASETS = {
    "Roman-Empire": {
        "phase1_baselines": os.path.expanduser("~/gnn_thesis_cluster/runs/roman_empire_phase1_baselines/roman_empire_phase1_baselines_grouped.csv"),
        "phase2_baselines": os.path.expanduser("~/gnn_thesis_cluster/runs/roman_empire_phase2_baselines/roman_empire_phase2_baselines_grouped.csv"),
        "phase1_full": os.path.expanduser("~/gnn_thesis_cluster/runs/roman_empire_phase1_full/roman_empire_phase1_full_grouped.csv"),
        "phase2_full": os.path.expanduser("~/gnn_thesis_cluster/runs/roman_empire_phase2_full/roman_empire_phase2_full_grouped.csv"),
    },
    "Amazon-Photo": {
        "phase1_baselines": os.path.expanduser("~/gnn_thesis_cluster/runs/amazon_photo_phase1_baselines/amazon_photo_phase1_baselines_grouped.csv"),
        "phase2_baselines": os.path.expanduser("~/gnn_thesis_cluster/runs/amazon_photo_phase2_baselines/amazon_photo_phase2_baselines_grouped.csv"),
        "phase1_full": os.path.expanduser("~/gnn_thesis_cluster/runs/amazon_photo_phase1_full/amazon_photo_phase1_full_grouped.csv"),
        "phase2_full": os.path.expanduser("~/gnn_thesis_cluster/runs/amazon_photo_phase2_full/amazon_photo_phase2_full_grouped.csv"),
    },
}

plot_dir = os.path.expanduser("~/gnn_thesis_cluster/plots/roman_amazon_phase1_vs_phase2")
os.makedirs(plot_dir, exist_ok=True)

baseline_models = ["GCN", "GAT", "GraphSAGE"]

for dataset_name, paths in DATASETS.items():
    p1_base = pd.read_csv(paths["phase1_baselines"])
    p2_base = pd.read_csv(paths["phase2_baselines"])
    p1_full = pd.read_csv(paths["phase1_full"])
    p2_full = pd.read_csv(paths["phase2_full"])

    p1_base["phase"] = "Phase1"
    p2_base["phase"] = "Phase2"
    p1_full["phase"] = "Phase1"
    p2_full["phase"] = "Phase2"

    # ----------------------------
    # Baselines only: Phase 1 vs Phase 2
    # ----------------------------
    plt.figure(figsize=(12, 7))
    for phase_name, df in [("Phase1", p1_base), ("Phase2", p2_base)]:
        for model_name in baseline_models:
            best_row = df[df["model"] == model_name].sort_values(
                by=["mean_best_test_acc", "mean_best_val_acc"],
                ascending=[False, False]
            ).iloc[0]

            plt.scatter(
                [f"{model_name}\n{phase_name}"],
                [best_row["mean_best_test_acc"]],
                s=90,
                label=f"{model_name} {phase_name}" if phase_name == "Phase1" else None,
            )

    # cleaner grouped bar plot
    plt.close()

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
    plt.title(f"{dataset_name} - Baselines: Phase 1 vs Phase 2")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{dataset_name}_baselines_phase1_vs_phase2_bar.png"))
    plt.close()

    # ----------------------------
    # Full models: Phase 1 vs Phase 2
    # ----------------------------
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
    plt.title(f"{dataset_name} - Full: Phase 1 vs Phase 2")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{dataset_name}_full_phase1_vs_phase2_bar.png"))
    plt.close()

print(f"Saved plots to: {plot_dir}")
