# CADGrasp - AI Coding Instructions

## Project Overview
**CADGrasp** (Contact and Collision Aware Dexterous Grasping) is a dexterous grasping framework built on DexGraspNet 2.0's codebase. The core innovation is using **IBS (Interaction Bisector Surface)** as an intermediate representation—equidistant points within a certain range between hand and objects in cluttered multi-object scenes.

### Pipeline Overview
```
Scene Point Cloud → LASDiffusion (IBS voxel prediction) → IBS point cloud → PoseOptimize (hand pose optimization) → Final Grasp
```

## Architecture

### CADGrasp Core Pipeline
1. **`IBSGrasp/`** - IBS data extraction from cluttered scenes
   - `scripts/` - Scripts to extract IBS from DexGraspNet 2.0 grasp data
   - `ibsdata/` - Processed IBS voxel data
   - `conf/` - Hydra configs for scene processing
2. **`LASDiffusion/`** - Voxel diffusion model for IBS prediction
   - `train.py` - Train IBS voxel diffusion model
   - `generate.py` - `generate_based_on_scene_obs()` predicts IBS voxels from scene point cloud
   - `network/` - 3D UNet-based diffusion architecture
3. **`PoseOptimize/`** - Adam-based pose optimization from predicted IBS
   - `utils/AdamOpt.py` - `AdamOptimizer` optimizes hand pose to match IBS
   - `utils/HandModel.py` - Hand model for optimization
4. **`Dex2Evaluator/`** - End-to-end evaluation integrating IBS policy
   - `new_predict.py` - `IBSPolicy()` combines LASDiffusion + PoseOptimize

### DexGraspNet 2.0 Baseline (inherited)
- **`src/network/`** - Baseline neural network models (graspness + diffusion/CVAE/ISA)
- **`src/utils/`** - Dataset, robot models, config utilities
- **`src/eval/`** - Evaluation pipeline (predict → simulate → print)
- **`src/preprocess/`** - Data preprocessing

## Key Patterns

### IBS (Interaction Bisector Surface)
IBS represents equidistant points between hand surface and object surface within a local region. It captures contact-aware geometric relationships:
```python
# IBS voxelization in LASDiffusion/generate.py
def voxelize_scene_pc(scene_pcs, hand_pose):
    # Transform scene to hand frame, voxelize in [-0.1, 0.1]³ at 0.005 resolution
    # Returns (B, 40, 40, 40) voxel grid
```

### Config System
Configs use hierarchical YAML with `DotDict` for `config.model.type` access:
```python
from src.utils.config import load_config
config = load_config('configs/network/train_dex_ours.yaml')
print(config.model.backbone)  # "sparseconv"
```

### Robot Models
Robots defined in `robot_models/urdf/` with metadata in `robot_models/meta/`:
```python
robot = RobotModel('robot_models/urdf/leap_hand.urdf', 'robot_models/meta/leap_hand/meta.yaml')
```

## Developer Workflows

### CADGrasp Pipeline

#### 1. IBS Data Preparation (IBSGrasp)
Extract IBS from DexGraspNet 2.0 grasp data for training LASDiffusion.

#### 2. Train IBS Diffusion Model (LASDiffusion)
```bash
cd LASDiffusion
python train.py --name my_exp --ibs_path ../IBSGrasp/ibsdata --scene_pc_path ../data/scenes
```

#### 3. IBS-based Grasp Prediction (Dex2Evaluator)
```bash
python Dex2Evaluator/new_predict.py  # Uses IBSPolicy: LASDiffusion → PoseOptimize
```

### DexGraspNet 2.0 Baseline

#### Training
```bash
python src/train.py --exp_name my_exp --yaml configs/network/train_dex_ours.yaml
```

#### Evaluation (3-step pipeline)
```bash
# 1. Predict grasps
python src/eval/predict_dexterous_all.py --ckpt path/to/ckpt.pth --dataset graspnet --scene_id_start 100 --scene_id_end 190

# 2. Simulate in IsaacGym
python src/eval/evaluate_dexterous_all.py --dataset graspnet --split dense --ckpt_path_list path/to/ckpt.pth

# 3. Print results
python src/eval/print_dexterous_results.py --ckpt path/to/ckpt.pth
```

### Visualization
```bash
python tests/visualize_scene.py          # Scene point clouds
python tests/visualize_dex_pred.py --ckpt_path=path/to/ckpt.pth  # Predicted grasps
```

## Critical Dependencies
- **CUDA 11.7** - Required for PyTorch3D and MinkowskiEngine
- **IsaacGym** - Physics simulation for evaluation (download from NVIDIA)
- **MinkowskiEngine** - Sparse convolutions, requires `libopenblas-dev`
- **TorchSDF/torchprimitivesdf** - SDF computation for collision checking
- **PyTorch Lightning** - Used by LASDiffusion for training

## Data Splits
- `train`: scenes 0-99, `val`: 90-99
- `test_seen`: 100-129, `test_similar`: 130-159, `test_novel`: 160-189
- Acronym dataset: `scene_dense_*`, `scene_random_*`, `scene_loose_*`

## Code Conventions
1. **Working directory**: Scripts use `os.chdir()` to project root - always run from repo root
2. **Multi-GPU evaluation**: Uses `multiprocessing.Pool` with `--gpu_list` argument
3. **Checkpoints**: Saved to `experiments/{exp_name}/ckpt/ckpt_{iter}.pth`
4. **Logging**: TensorBoard logs in `experiments/{exp_name}/log/`, optional W&B support

## Common Pitfalls
- Missing `CUDA_HOME=/usr/local/cuda-11.7` breaks MinkowskiEngine install
- NumPy version conflicts: use `numpy==1.23.0`, update to 1.24.1 for gripper experiments
- Python path issues: run `export PATH="/path/to/conda/envs/DexGrasp/bin:$PATH"`
