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

plot_dir = os.path.expanduser("~/gnn_thesis_cluster/plots/roman_amazon_baselines_vs_full")
os.makedirs(plot_dir, exist_ok=True)

for dataset_name, paths in DATASETS.items():
    for phase_name, base_key, full_key in [
        ("Phase1", "phase1_baselines", "phase1_full"),
        ("Phase2", "phase2_baselines", "phase2_full"),
    ]:
        base_df = pd.read_csv(paths[base_key])
        full_df = pd.read_csv(paths[full_key])

        best_base = base_df.sort_values(
            by=["mean_best_test_acc", "mean_best_val_acc"],
            ascending=[False, False]
        ).iloc[0]

        best_full = full_df.sort_values(
            by=["mean_best_test_acc", "mean_best_val_acc"],
            ascending=[False, False]
        ).iloc[0]

        plot_df = pd.DataFrame([
            {
                "label": f"Baselines\n{best_base['model']}\nL={int(best_base['num_layers'])}, H={int(best_base['hidden_channels'])}",
                "score": best_base["mean_best_test_acc"],
                "std": best_base["std_best_test_acc"],
            },
            {
                "label": f"Full\n{best_full['model']}\nL={int(best_full['num_layers'])}, H={int(best_full['hidden_channels'])}",
                "score": best_full["mean_best_test_acc"],
                "std": best_full["std_best_test_acc"],
            },
        ])

        plt.figure(figsize=(9, 6))
        plt.bar(plot_df["label"], plot_df["score"], yerr=plot_df["std"], capsize=4)
        plt.ylabel("Mean Best Test Accuracy")
        plt.title(f"{dataset_name} - {phase_name}: Baselines vs Full")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{dataset_name}_{phase_name}_baselines_vs_full.png"))
        plt.close()

print(f"Saved plots to: {plot_dir}")
