# IBS Data Processing Pipeline

This directory contains the complete processing pipeline from DexGraspNet2.0 raw grasp data to IBS training data.

## 🔄 Complete Processing Flow

```
dex_grasps_new/ ──► IsaacGym Sim ──► dex_grasps_success_indices/ ──► FPS Sampling ──► fps_sampled_indices/ ──► IBS Compute ──► ibsdata/
                   (Stage 1)                                         (Stage 2)                             (Stage 3)
```

---

## 📁 Script Description

### Stage 1: Grasp Simulation Filtering

| Script | Function | Command |
|--------|----------|---------|
| `filter_grasps_by_sim.py` | Single scene IsaacGym simulation filtering | `python filter_grasps_by_sim.py --scene_id scene_0055` |
| `batch_filter_grasps.py` | Multi-scene batch filtering | `python batch_filter_grasps.py --scene_start 0 --scene_end 100` |
| `load_success_grasps.py` | Utility module: load filtered successful grasps | (import as module) |

**Input**: `data/DexGraspNet2.0/dex_grasps_new/scene_XXXX/leap_hand/XXX.npz`

**Output**: `data/DexGraspNet2.0/dex_grasps_success_indices/scene_XXXX/leap_hand/XXX.npz`

**Notes**: 
- IsaacGym installation required for Stage 1
- If Stage 1 is skipped, subsequent stages assume all grasps are successful

---

### Stage 2: FPS Sampling

| Script | Function | Command |
|--------|----------|---------|
| `fps_sample_grasps.py` | Single scene FPS sampling | `python fps_sample_grasps.py --scene_id 55 --max_grasps 5000` |
| `batch_fps_sample_grasps.py` | Multi-scene batch sampling | `python batch_fps_sample_grasps.py --scene_start 0 --scene_end 100` |
| `load_fps_grasps.py` | Utility module: load FPS-sampled grasps | (import as module) |

**Input**: 
- `data/DexGraspNet2.0/dex_grasps_new/scene_XXXX/leap_hand/XXX.npz`
- `data/DexGraspNet2.0/dex_grasps_success_indices/scene_XXXX/leap_hand/XXX.npz` (optional)

**Output**: `data/DexGraspNet2.0/fps_sampled_indices/scene_XXXX/leap_hand/XXX.npz`

**Key Parameters**:
- `--max_grasps`: Maximum grasps per scene (default: 5000)
- `--perturbation`: Grasp point random perturbation scale (default: 0.02m), for handling multiple grasps at the same point

---

### Stage 3: IBS Computation

| Script | Function | Command |
|--------|----------|---------|
| `calculate_ibs_new.py` | Single scene IBS computation | `python calculate_ibs_new.py --scene_id 55` |
| `batch_calculate_ibs.py` | Multi-scene batch computation | `python batch_calculate_ibs.py --scene_start 0 --scene_end 100` |

**Input**:
- `data/DexGraspNet2.0/fps_sampled_indices/scene_XXXX/leap_hand/XXX.npz`
- Scene data and object meshes

**Output** (saved to `data/ibsdata/`):
- `ibs/scene_XXXX.npy`: IBS voxel data `(N, 40, 40, 40, 3)`
- `w2h_trans/scene_XXXX.npy`: World-to-hand transformation matrices `(N, 4, 4)`

---

### Stage 4: View Annotation (LASDiffusion Training Prerequisite)

| Script | Function | Command |
|--------|----------|---------|
| `annotate_ibs_for_view.py` | Annotate visible IBS for each view | `python annotate_ibs_for_view.py --scene_start 0 --scene_end 100` |

**Input**: 
- `data/ibsdata/` (IBS data)
- `data/DexGraspNet2.0/scenes/` (scene point clouds)

**Output**: `data/ibsdata/scene_valid_ids/scene_XXXX/view_YYYY.npy`

**Notes**:
- This step is a **prerequisite for LASDiffusion training**
- Purpose: filter IBS visible from each camera view (grasp point within 1cm of scene point cloud)
- If view filtering is not needed, set `use_view_filter=False` in `IBS_Dataset`

---

### Utility Scripts

| Script | Function |
|--------|----------|
| `scene.py` | Scene data loading class, used by multiple scripts |

---

## 📦 IBS Data Format

IBS voxel is a 4D array of shape `(40, 40, 40, 3)` with three channels:

| Channel | Name | Description | Value Range |
|---------|------|-------------|-------------|
| 0 | `occupancy` | IBS occupied voxels | -1 (empty) / 1 (occupied) |
| 1 | `contact` | Finger contact regions | 0 (no contact) / 1 (contact) |
| 2 | `thumb_contact` | Thumb contact regions | 0 (no contact) / 2 (contact) |

**Voxel Parameters**:
- Spatial range: `[-0.1, 0.1]^3` (hand coordinate frame)
- Resolution: `0.005m` (5mm)
- Grid size: `40 × 40 × 40`

---

## 🚀 Quick Start

### Full Pipeline (Including Training Preparation)

```bash
# 1. Simulation filtering (requires IsaacGym)
python batch_filter_grasps.py --scene_start 0 --scene_end 100 --gpu_ids 0,1,2,3

# 2. FPS sampling
python batch_fps_sample_grasps.py --scene_start 0 --scene_end 100

# 3. IBS computation
python batch_calculate_ibs.py --scene_start 0 --scene_end 100

# 4. View annotation (required before LASDiffusion training)
python annotate_ibs_for_view.py --scene_start 0 --scene_end 100 --gpu_ids 0,1,2,3
```

### Skip Simulation Filtering

If IsaacGym is not installed or you want to skip simulation filtering:

```bash
# Run FPS sampling directly (assumes all grasps successful)
python batch_fps_sample_grasps.py --scene_start 0 --scene_end 100

# Then run IBS computation
python batch_calculate_ibs.py --scene_start 0 --scene_end 100
```

### Single Scene Test

```bash
# Process single scene (scene_0055)
python fps_sample_grasps.py --scene_id 55
python calculate_ibs_new.py --scene_id 55 --visualize
```

---

## ⚙️ Default Configuration

All scripts use built-in default configurations, no external config files required.

### Scene Class Default Parameters
```python
SceneConfig(
    scene_id=0,
    robot_name='leap_hand',
    urdf_path='robot_models/urdf/leap_hand_simplified.urdf',
    meta_path='robot_models/meta/leap_hand/meta.yaml',
    camera='realsense',
    table_size=[0.6, 0.6, 0.0],
    device='cuda:0',
    num_samples=4096,
    scene_base_path='data/DexGraspNet2.0/scenes',
    mesh_base_path='data/DexGraspNet2.0/meshdata'
)
```

### IBS Computation Default Parameters
```python
IBSConfig(
    bound=0.1,           # Spatial range [-0.1, 0.1]
    resolution=0.005,    # Voxel resolution 5mm
    delta=0.005,         # IBS thickness threshold
    epsilon=1e-5,        # Iteration convergence threshold
    max_iteration=20,    # Maximum iterations
    voxel_size=40,       # Voxel grid size
    contact_delta=0.0075,    # Contact point threshold
    thumb_contact_delta=0.0085  # Thumb contact threshold
)
```

---

## 📚 Related Modules

- `src/cadgrasp/ibs/utils/ibs_repr.py`: IBS data classes (`IBS`, `IBSBatch`, `IBSConfig`)
- `src/cadgrasp/ibs/utils/transforms.py`: Coordinate transformation utilities
- `thirdparty/LASDiffusion/network/data_loader.py`: IBS dataset loader
- `thirdparty/LASDiffusion/generate.py`: IBS generation/inference

