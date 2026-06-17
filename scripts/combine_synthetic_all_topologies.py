import os
import pandas as pd

ring_path = os.path.expanduser("~/gnn_thesis_cluster/runs/synthetic_ring_phase1/synthetic_ring_phase1_grouped.csv")
crossed_path = os.path.expanduser("~/gnn_thesis_cluster/runs/synthetic_crossed_ring_phase1/synthetic_crossed_ring_phase1_grouped.csv")
clique_path = os.path.expanduser("~/gnn_thesis_cluster/runs/synthetic_cliquepath_phase1/synthetic_cliquepath_phase1_grouped.csv")

out_path = os.path.expanduser("~/gnn_thesis_cluster/runs/synthetic_all_topologies_grouped.csv")

ring_df = pd.read_csv(ring_path)
crossed_df = pd.read_csv(crossed_path)
clique_df = pd.read_csv(clique_path)

df = pd.concat([ring_df, crossed_df, clique_df], ignore_index=True)
df.to_csv(out_path, index=False)

print(f"Saved combined grouped CSV to: {out_path}")
print(df.head(20).to_string(index=False))
