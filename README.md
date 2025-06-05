# DiTango: Efficient Video Generation with selective attention state caching

DiTango is a video generation acceleration system based on distributed feature reuse. By caching and efficiently reusing key attention computation features, it significantly reduces communication overhead while maintaining high-quality video generation.

## Environment Requirements

- Distributed Computing Environment
- CUDA 12.4
- Python 3.10+
- PyTorch 2.1+

## Setup

```bash
git clone https://github.com/[your-username]/ditango
cd ditango
pip install -r requirements.txt
```

## Quick Start

### 1. Model Preparation

- Download weights for your video generation model
- Complete the original model's environment setup and dependencies
- Currently supported models:
  - CogVideoX
  - HunyuanVideo
  - Stepvideo-T2V

### 2. System Configuration

1. Move the sample script to model directory:
```bash
cp ditango/executor/sample_video.py /path/to/model/
```

2. Configure system parameters:
```bash
# Edit config.yaml to adjust parallelization strategy and other parameters
vim ditango/configs/<model_name>/config.yaml
```

### 3. Redundancy Analysis (First-time Only)

Generate redundancy map for parallel inference:

```bash
# Torchrun
torchrun --nproc_per_node=N --nnodes=M \
    --node_rank=0 --master_addr="localhost" --master_port=29500 \
    ditango/executor/<model_name>/preprocess.py

# Slurm
srun --nodes=N --ntasks-per-node=M python ditango/executor/<model_name>/preprocess.py
```

The redundancy map will be saved at `ditango/configs/<model_name>/redundancy_map_<steps>_<layers>.pt`

### 4. Video Generation

Run distributed inference:

```bash
# Torchrun
torchrun --nproc_per_node=N --nnodes=M \
    --node_rank=0 --master_addr="localhost" --master_port=29500 \
    sample_video.py

# Slurm
srun --nodes=N --ntasks-per-node=M python sample_video.py
```

Generated videos and parallelization strategy visualizations will be saved to the directory specified in `config.yaml`.

## Directory Structure

```
ditango/
├── core/
├── configs/
│   └── <model_name>/
│       ├── config.yaml
│       ├── redundancy_map_<steps>_<layers>.pt
│       └── redundancy_visualization/
├── executor/
│   └── <model_name>/
│       ├── preprocess.py
│       └── sample_video.py
├── timer.py
├── logger.py
├── utils.py
└── README.md
```

## Citation

If you find this work useful, please cite our paper:
```
[Citation placeholder]
```
