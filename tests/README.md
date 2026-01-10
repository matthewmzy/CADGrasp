# Tests - Visualization and Data Tools

This directory contains visualization scripts and data utility tools for the CADGrasp project.

## Scripts Overview

### Data Visualization

| Script | Description | Usage |
|--------|-------------|-------|
| `visualize_scene.py` | Visualize scene meshes with object poses | `python tests/visualize_scene.py --scene_id 55` |
| `visualize_dex_grasp.py` | Visualize DexGraspNet grasp annotations | `python tests/visualize_dex_grasp.py --scene scene_0055` |
| `vis_ibs.py` | Visualize IBS voxel data | `python tests/vis_ibs.py --scene_id 55` |
| `vis_fps_grasps.py` | Visualize FPS-sampled grasp points | `python tests/vis_fps_grasps.py --scene_id 55` |

### Data Statistics

| Script | Description | Usage |
|--------|-------------|-------|
| `count_data.py` | Count data at each pipeline stage | `python tests/count_data.py` |

### Network Prediction Visualization

| Script | Description | Usage |
|--------|-------------|-------|
| `visualize_dex_pred.py` | Visualize trained network predictions | `python tests/visualize_dex_pred.py --ckpt_path ... --scene scene_0055` |

## Data Paths

All scripts use the following default data paths:

- **Scene data**: `data/DexGraspNet2.0/scenes/`
- **Mesh data**: `data/DexGraspNet2.0/meshdata/`
- **Raw grasps**: `data/DexGraspNet2.0/dex_grasps_new/`
- **Success indices**: `data/DexGraspNet2.0/dex_grasps_success_indices/`
- **FPS indices**: `data/DexGraspNet2.0/fps_sampled_indices/`
- **IBS data**: `data/ibsdata/`

## Example Commands

### Visualize Scene 55

```bash
# Scene with object meshes
python tests/visualize_scene.py --scene_id 55

# Grasp annotations (raw)
python tests/visualize_dex_grasp.py --scene scene_0055 --grasp_num 5

# Grasp annotations (FPS filtered)
python tests/visualize_dex_grasp.py --scene scene_0055 --use_fps --grasp_num 10

# IBS voxels
python tests/vis_ibs.py --scene_id 55 --max_grasps 3

# IBS with scene point cloud
python tests/vis_ibs.py --scene_id 55 --with_scene_pc

# FPS sampling visualization
python tests/vis_fps_grasps.py --scene_id 55 --compare_raw
```

### Data Statistics

```bash
# All scenes
python tests/count_data.py

# Specific scene range with details
python tests/count_data.py --scene_start 100 --scene_end 130 --detailed
```

## Dependencies

- `plotly`: Interactive 3D visualization
- `trimesh`: Mesh loading
- `transforms3d`: Coordinate transformations
- `tabulate`: Table formatting (for count_data.py)
- `numpy`, `torch`: Numerical computation

Install with:
```bash
pip install plotly trimesh transforms3d tabulate
```
