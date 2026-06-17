# Repository Notes

## 1. Environment and package notes

This repository was developed and executed on the RWTH cluster using a Python virtual environment.

### Main PyTorch / PyG stack used
- torch: `2.11.0+cu128`
- torch_geometric: `2.7.0`
- pyg_lib: `0.6.0+pt211cu128`
- torch_scatter: `2.1.2+pt211cu128`
- torch_sparse: `0.6.18+pt211cu128`
- numpy: `2.2.6`
- scipy: `1.15.3`


## 2. Repository structure

### `scripts/`
- experiment runners
- grid generators
- aggregation scripts
- plotting scripts
- helper dataset loaders

### `configs/`
Contains experiment grids and configuration CSV/YAML files.

### `slurm/`
Contains Slurm job scripts for cluster execution.

### `runs/`
Contains generated experiment outputs:
- `*_summary.json`
- `*_history.csv`
- grouped CSV summaries
- best-by-dataset CSV summaries

### `plots/`
Contains generated figures for the thesis.


### Real-world Phase 1

Typical settings:
- layers: 2, 4
- hidden sizes: 64, 128


### Real-world Phase 2

Typical settings:
- layers: 8, 16, 32
- hidden sizes: 64, 128

### Baselines blocks
Models:
- GCN
- GAT
- GraphSAGE

### Full blocks
Models:
- GCN
- GAT
- GraphSAGE
- GCNPairNorm
- GATPairNorm
- GraphSAGEPairNorm


### Synthetic Phase 1

- compare topology, distance, and hidden dimension under approximately matched depth

Topologies:
- Ring
- CrossedRing
- CliquePath


### Synthetic Phase 2

- depth sweep at fixed representative distances

## 6. Recommended plots order 

For real-world results:
1. baseline Phase 1
2. baseline Phase 2
3. full Phase 1
4. full Phase 2
5. Phase 1 vs Phase 2 comparison
6. baselines vs full comparison

For synthetic results:
1. Phase 1 per topology
2. topology comparison plots
3. Phase 2 depth sweeps
4. Phase 1 vs Phase 2 comparison
