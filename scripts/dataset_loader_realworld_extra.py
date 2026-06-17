import os
from copy import deepcopy

import torch
from torch_geometric.datasets import Amazon, HeterophilousGraphDataset
from torch_geometric.transforms import RandomNodeSplit


def _select_split_mask(mask: torch.Tensor, split_idx: int) -> torch.Tensor:
    if mask.dim() == 1:
        return mask
    return mask[:, split_idx]


def _build_amazon_photo_splits(data, num_splits: int = 10):
    train_masks = []
    val_masks = []
    test_masks = []

    base_data = deepcopy(data)

    for split_idx in range(num_splits):
        torch.manual_seed(12345 + split_idx)

        split_data = RandomNodeSplit(
            split="test_rest",
            num_splits=1,
            num_train_per_class=20,
            num_val=500,
            num_test=1000,
        )(deepcopy(base_data))

        train_masks.append(split_data.train_mask)
        val_masks.append(split_data.val_mask)
        test_masks.append(split_data.test_mask)

    data.train_mask = torch.stack(train_masks, dim=1)
    data.val_mask = torch.stack(val_masks, dim=1)
    data.test_mask = torch.stack(test_masks, dim=1)
    return data


def load_extra_dataset(name: str, root: str = "data", split_idx: int = 0):
    name = name.lower()

    if name == "roman-empire":
        dataset = HeterophilousGraphDataset(
            root=os.path.join(root, "Heterophilous"),
            name="Roman-empire",
        )
        data = dataset[0]

    elif name == "amazon-photo":
        dataset = Amazon(
            root=os.path.join(root, "Amazon"),
            name="Photo",
        )
        data = dataset[0]
        data = _build_amazon_photo_splits(data, num_splits=10)

    else:
        raise ValueError(f"Unsupported dataset: {name}")

    data = deepcopy(data)

    data.train_mask = _select_split_mask(data.train_mask, split_idx)
    data.val_mask = _select_split_mask(data.val_mask, split_idx)
    data.test_mask = _select_split_mask(data.test_mask, split_idx)

    return dataset, data
