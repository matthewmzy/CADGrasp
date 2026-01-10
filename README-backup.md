# CADGrasp: Interaction Bisector Surface for Dexterous Grasping

Contact- and collision-aware dexterous grasping pipeline that introduces **Interaction Bisector Surface (IBS)** as an intermediate representation. IBS captures equidistant points between hand and objects, enabling LASDiffusion to predict contact-aware voxels and PoseOptimize to recover feasible hand poses.

![image](./figure/teaser.png)

## Installation

### 1. Clone with submodules

```bash
git clone --recursive https://github.com/matthewmzy/CADGrasp.git
cd CADGrasp

# If you already cloned without --recursive:
git submodule update --init --recursive
```

### 2. Create conda environment

```bash
conda create -n cad python=3.11
conda activate cad
```

### 3. Install PyTorch

```bash
# Install PyTorch with CUDA support (adjust for your CUDA version)
# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or use Chinese mirror for faster download:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4. Install CADGrasp package

```bash
pip install -e .
```

### 5. Install PyTorch3D

```bash
# Option 1: Install via conda (recommended)
conda install pytorch3d -c pytorch3d

# Option 2: Build from source
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

### 6. Install TorchSDF (for SDF computation)

TorchSDF requires CUDA compilation. Make sure `CUDA_HOME` is set correctly:

```bash
cd thirdparty/TorchSDF
export CUDA_HOME=/usr/local/cuda  # Adjust to your CUDA path
pip install -e . --no-build-isolation
cd ../..
```

### 7. Install MinkowskiEngine (for sparse convolutions)

For **CUDA 12.x** compatibility, we use a patched fork:

```bash
cd thirdparty/MinkowskiEngine
mkdir -p MinkowskiEngineBackend  # Required directory
export CUDA_HOME=/usr/local/cuda  # Adjust to your CUDA path
pip install . --no-build-isolation  # Do not add -e because Backend will fail to compile
cd ../..
```

> **Note**: The MinkowskiEngine submodule points to [MinkowskiEngineCuda13](https://github.com/AzharSindhi/MinkowskiEngineCuda13), which includes fixes for CUDA 12/13 compatibility.

### 8. Install IsaacGym (optional, for simulation)

Download IsaacGym from [NVIDIA](https://developer.nvidia.com/isaac-gym) and install:

```bash
cd isaacgym/python && pip install -e .
```

### Troubleshooting

**TorchSDF/MinkowskiEngine compilation fails:**
- Ensure `CUDA_HOME` points to your CUDA installation (e.g., `/usr/local/cuda-12.4`)
- Use `--no-build-isolation` flag to use existing PyTorch
- Check that `nvcc --version` matches your PyTorch CUDA version

**RTX 50 series (sm_120) not supported:**
- Current PyTorch only supports up to sm_90 (RTX 40 series)
- The code will still compile and work on these GPUs using PTX fallback

## Project Structure

```
CADGrasp/
├── src/cadgrasp/              # Main Python package
│   ├── baseline/              # DexGraspNet 2.0 baseline models
│   │   ├── network/           # Neural network architectures
│   │   ├── eval/              # Evaluation scripts
│   │   ├── preprocess/        # Data preprocessing
│   │   └── utils/             # Utilities (robot model, visualization, etc.)
│   ├── ibs/                   # IBS data processing
│   │   ├── scripts/           # IBS computation scripts
│   │   └── utils/             # IBS utilities
│   ├── optimizer/             # Pose optimization with IBS
│   └── evaluator/             # End-to-end evaluation pipeline
├── thirdparty/
│   └── LASDiffusion/          # IBS voxel diffusion model (git submodule)
├── configs/                   # Hydra/YAML configuration files
├── assets/                    # Robot URDF models and meshes
├── data/                      # Datasets (download separately)
│   ├── scenes/                # GraspNet-1Billion scenes
│   ├── meshdata/              # Object meshes
│   └── ibs/                   # Generated IBS data
├── experiments/               # Experiment scripts and examples
│   ├── scripts/               # Bash scripts for experiments
│   ├── tests/                 # Visualization and test scripts
│   └── examples/              # Example code
└── pyproject.toml             # Package configuration
```

## Quick Start

### Train IBS Diffusion Model

```bash
cd thirdparty/LASDiffusion
python train.py train_from_folder \
    --name LEAP_dif \
    --ibs_path ../../data/ibs/ibsdata \
    --scene_pc_path ../../data/scenes \
    --batch_size 64 --training_epoch 200000
```

### Run End-to-End Evaluation

```bash
python -m cadgrasp.evaluator.predict
```

### Visualize Results

```bash
python experiments/tests/visualize_dex_pred.py --ckpt_path path/to/checkpoint.pth
```

### CADGrasp Prediction Pipeline

Run the full CADGrasp prediction pipeline (DexGraspNet2.0 → IBS → Optimization):

```bash
# Predict on GraspNet scenes (starting from scene_0100)
python -m cadgrasp.baseline.eval.predict_dexterous \
    --ckpt_path data/DexGraspNet2.0/DexGraspNet2.0-ckpts/CAD/ckpt/ckpt_50000.pth \
    --las_exp_name LEAP_dif \
    --scene_id scene_0100 \
    --scene_num 10 \
    --top_n 5 \
    --dataset graspnet
```

**Key parameters:**
- `--ckpt_path`: DexGraspNet2.0 checkpoint for grasp point and rotation prediction
- `--las_exp_name`: LASDiffusion experiment name (model in `thirdparty/LASDiffusion/results/{name}/recent/last.ckpt`)
- `--top_n`: Number of top grasp candidates per view for IBS prediction (default: 5)
- `--diffusion_steps`: Number of diffusion steps for IBS generation (default: 50)
- `--max_iters`: Maximum pose optimization iterations (default: 200)
- `--parallel_num`: Number of parallel optimizations per IBS (default: 10)

Results will be saved to `data/DexGraspNet2.0/DexGraspNet2.0-ckpts/CAD/results/`.

## Data Preparation

### Download DexGraspNet 2.0 Data

```bash
# Download from HuggingFace
huggingface-cli download lhrlhr/DexGraspNet2.0 --local-dir data/
```

### Setup CADGrasp Checkpoint Directory

After downloading the data, create a separate checkpoint directory for CADGrasp predictions:

```bash
# Copy the OURS checkpoint to CAD directory
cp -r data/DexGraspNet2.0/DexGraspNet2.0-ckpts/OURS data/DexGraspNet2.0/DexGraspNet2.0-ckpts/CAD
```

This ensures CADGrasp results are saved separately from baseline results.

### Generate IBS Data

**快速批量生成 (推荐)**：只需三条命令即可完成从原始抓取数据到IBS训练数据的全流程：

```bash
# 1. 仿真筛选 - 使用IsaacGym筛选成功抓取 (可选，需安装IsaacGym)
python src/cadgrasp/ibs/scripts/batch_filter_grasps.py --scene_start 0 --scene_end 100 --gpu_ids 0,1,2,3

# 2. FPS采样 - 对每个场景的抓取进行FPS降采样 (跳过第1步会默认所有抓取成功)
python src/cadgrasp/ibs/scripts/batch_fps_sample_grasps.py --scene_start 0 --scene_end 100

# 3. IBS计算 - 生成IBS体素数据到data/ibsdata/
python src/cadgrasp/ibs/scripts/batch_calculate_ibs.py --scene_start 0 --scene_end 100
```

> 详细参数说明和单场景调试请参考 [src/cadgrasp/ibs/scripts/README.md](src/cadgrasp/ibs/scripts/README.md)

## IBS Data Format

| File | Shape | Description |
|------|-------|-------------|
| `data/ibs/ibsdata/ibs/scene_xxxx.npy` | `(N, 40, 40, 40, 3)` bool | IBS voxels: occupancy, contact, thumb_contact |
| `data/ibs/ibsdata/w2h_trans/scene_xxxx.npy` | `(N, 4, 4)` float32 | World-to-hand transforms |
| `data/ibs/scene_valid_ids/scene_xxxx/view_yyyy.npy` | `(N,)` bool | Per-view visibility masks |

## Module Reference

### cadgrasp.baseline
DexGraspNet 2.0 baseline implementation with graspness prediction and grasp generation.

### cadgrasp.ibs
IBS (Interaction Bisector Surface) computation from grasp data.

### cadgrasp.optimizer
Adam-based pose optimization using IBS energy functions.

### cadgrasp.evaluator
End-to-end evaluation combining LASDiffusion prediction and pose optimization.

### thirdparty.LASDiffusion
3D voxel diffusion model for IBS prediction from scene point clouds.

## Citation

```bibtex
@article{cadgrasp2024,
  title={CADGrasp: Contact and Collision Aware Dexterous Grasping with Interaction Bisector Surface},
  author={...},
  year={2024}
}
```

## License

MIT License
