import os
import pandas as pd
import matplotlib.pyplot as plt

plot_dir = os.path.expanduser("~/gnn_thesis_cluster/plots/all_realworld_full_old_vs_new")
os.makedirs(plot_dir, exist_ok=True)

dataset_order = ["Cora", "CiteSeer", "PubMed", "Roman-empire", "Amazon-Photo"]

# Old grouped files
old_phase1 = pd.read_csv(
    os.path.expanduser("~/gnn_thesis_cluster/runs/realworld_phase1_gs/realworld_phase1_gs_grouped.csv")
)
old_phase2 = pd.read_csv(
    os.path.expanduser("~/gnn_thesis_cluster/runs/realworld_phase2/realworld_phase2_grouped.csv")
)

# New grouped files
roman_phase1 = pd.read_csv(
    os.path.expanduser("~/gnn_thesis_cluster/runs/roman_empire_phase1_full/roman_empire_phase1_full_grouped.csv")
)
roman_phase2 = pd.read_csv(
    os.path.expanduser("~/gnn_thesis_cluster/runs/roman_empire_phase2_full/roman_empire_phase2_full_grouped.csv")
)
amazon_phase1 = pd.read_csv(
    os.path.expanduser("~/gnn_thesis_cluster/runs/amazon_photo_phase1_full/amazon_photo_phase1_full_grouped.csv")
)
amazon_phase2 = pd.read_csv(
    os.path.expanduser("~/gnn_thesis_cluster/runs/amazon_photo_phase2_full/amazon_photo_phase2_full_grouped.csv")
)

phase1_df = pd.concat([old_phase1, roman_phase1, amazon_phase1], ignore_index=True)
phase2_df = pd.concat([old_phase2, roman_phase2, amazon_phase2], ignore_index=True)

def best_by_dataset(df):
    rows = []
    for dataset_name in dataset_order:
        subset = df[df["dataset"] == dataset_name]
        if len(subset) == 0:
            continue
        best_row = subset.sort_values(
            by=["mean_best_test_acc", "mean_best_val_acc"],
            ascending=[False, False]
        ).iloc[0]
        rows.append(best_row.to_dict())
    return pd.DataFrame(rows)

phase1_best = best_by_dataset(phase1_df)
phase2_best = best_by_dataset(phase2_df)

phase1_best.to_csv(
    os.path.expanduser("~/gnn_thesis_cluster/runs/all_realworld_full_phase1_best_by_dataset.csv"),
    index=False
)
phase2_best.to_csv(
    os.path.expanduser("~/gnn_thesis_cluster/runs/all_realworld_full_phase2_best_by_dataset.csv"),
    index=False
)

def make_plot(df, title, out_name):
    labels = [
        f"{row['dataset']}\n{row['model']}\nL={int(row['num_layers'])}, H={int(row['hidden_channels'])}"
        for _, row in df.iterrows()
    ]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, df["mean_best_test_acc"], yerr=df["std_best_test_acc"], capsize=4)
    plt.ylabel("Mean Best Test Accuracy")
    plt.title(title)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, out_name))
    plt.close()

make_plot(
    phase1_best,
    "All Real-World Full Models - Phase 1 Best by Dataset",
    "all_realworld_full_phase1_best_by_dataset.png",
)

make_plot(
    phase2_best,
    "All Real-World Full Models - Phase 2 Best by Dataset",
    "all_realworld_full_phase2_best_by_dataset.png",
)

print(f"Saved plots to: {plot_dir}")
print("\nPhase 1 best by dataset:")
print(phase1_best.to_string(index=False))
print("\nPhase 2 best by dataset:")
print(phase2_best.to_string(index=False))
