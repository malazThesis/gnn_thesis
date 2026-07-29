import argparse
import csv
import subprocess
import sys
import os

SCRIPT_MAP = {
    "Ring": "scripts/run_synthetic_ring.py",
    "CrossedRing": "scripts/run_synthetic_crossed_ring.py",
    "CliquePath": "scripts/run_synthetic_cliquepath.py",
}

def load_grid_row(grid_path: str, task_id: int):
    with open(grid_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[task_id]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=str, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--out-base-dir", type=str, default="runs/synthetic_phase2")
    args = parser.parse_args()

    row = load_grid_row(args.grid, args.task_id)
    topology = row["topology"]

    if topology not in SCRIPT_MAP:
        raise ValueError(f"Unknown topology: {topology}")

    topo_out_dir = os.path.join(args.out_base_dir, topology)

    cmd = [
        sys.executable,
        SCRIPT_MAP[topology],
        "--grid", args.grid,
        "--task-id", str(args.task_id),
        "--epochs", str(args.epochs),
        "--out-dir", topo_out_dir,
    ]

    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
