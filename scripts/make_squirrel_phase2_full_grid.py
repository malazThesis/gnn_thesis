import csv
import os
from itertools import product


DATASET = "Squirrel"

MODELS = [
    "GCN",
    "GAT",
    "GraphSAGE",
    "GCNPairNorm",
    "GATPairNorm",
    "GraphSAGEPairNorm",
]

NUM_LAYERS = [8, 16, 32]
HIDDEN_CHANNELS = [64, 128]
SPLIT_INDICES = range(10)
SEED = 1

OUTPUT_PATH = "configs/squirrel_phase2_full.csv"


def main():
    rows = []

    combinations = product(
        MODELS,
        NUM_LAYERS,
        HIDDEN_CHANNELS,
        SPLIT_INDICES,
    )

    for experiment_id, values in enumerate(combinations):
        model, num_layers, hidden_channels, split_idx = values

        rows.append(
            {
                "experiment_id": experiment_id,
                "dataset": DATASET,
                "model": model,
                "num_layers": num_layers,
                "hidden_channels": hidden_channels,
                "seed": SEED,
                "split_idx": split_idx,
            }
        )

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "experiment_id",
                "dataset",
                "model",
                "num_layers",
                "hidden_channels",
                "seed",
                "split_idx",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Wrote {len(rows)} experiments to "
        f"{os.path.abspath(OUTPUT_PATH)}"
    )


if __name__ == "__main__":
    main()
