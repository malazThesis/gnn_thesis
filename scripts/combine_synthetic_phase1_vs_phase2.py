import os
import pandas as pd

phase1_files = [
    ("Ring", os.path.expanduser("~/gnn_thesis_cluster/runs/synthetic_ring_phase1/synthetic_ring_phase1_grouped.csv")),
    ("CrossedRing", os.path.expanduser("~/gnn_thesis_cluster/runs/synthetic_crossed_ring_phase1/synthetic_crossed_ring_phase1_grouped.csv")),
    ("CliquePath", os.path.expanduser("~/gnn_thesis_cluster/runs/synthetic_cliquepath_phase1/synthetic_cliquepath_phase1_grouped.csv")),
]

phase2_file = os.path.expanduser("~/gnn_thesis_cluster/runs/synthetic_phase2_grouped.csv")
out_path = os.path.expanduser("~/gnn_thesis_cluster/runs/synthetic_phase1_vs_phase2_grouped.csv")

dfs = []

for topology, path in phase1_files:
    df = pd.read_csv(path)
    df["phase"] = "Phase1"
    dfs.append(df)

phase2_df = pd.read_csv(phase2_file)
phase2_df["phase"] = "Phase2"
dfs.append(phase2_df)

df = pd.concat(dfs, ignore_index=True)
df.to_csv(out_path, index=False)

print(f"Saved combined CSV to: {out_path}")
print(df.head(20).to_string(index=False))
