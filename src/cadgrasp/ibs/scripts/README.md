# IBS数据处理流水线

本目录包含从DexGraspNet2.0原始抓取数据到IBS训练数据的完整处理流水线。

## 🔄 完整处理流程

```
dex_grasps_new/ ──► IsaacGym仿真 ──► dex_grasps_success_indices/ ──► FPS采样 ──► fps_sampled_indices/ ──► IBS计算 ──► ibsdata/
                   (Stage 1)                                         (Stage 2)                        (Stage 3)
```

---

## 📁 脚本文件说明

### Stage 1: 抓取数据仿真筛选

| 脚本 | 功能 | 运行命令 |
|------|------|----------|
| `filter_grasps_by_sim.py` | 单场景IsaacGym仿真筛选 | `python filter_grasps_by_sim.py --scene_id scene_0055` |
| `batch_filter_grasps.py` | 多场景批量筛选 | `python batch_filter_grasps.py --scene_start 0 --scene_end 100` |
| `load_success_grasps.py` | 工具模块：加载筛选后的成功抓取 | (作为模块导入使用) |

**输入**: `data/DexGraspNet2.0/dex_grasps_new/scene_XXXX/leap_hand/XXX.npz`

**输出**: `data/DexGraspNet2.0/dex_grasps_success_indices/scene_XXXX/leap_hand/XXX.npz`

**注意**: 
- 需要安装IsaacGym才能运行Stage 1
- 如果跳过Stage 1，后续阶段会默认所有抓取都成功

---

### Stage 2: FPS采样

| 脚本 | 功能 | 运行命令 |
|------|------|----------|
| `fps_sample_grasps.py` | 单场景FPS采样 | `python fps_sample_grasps.py --scene_id 55 --max_grasps 5000` |
| `batch_fps_sample_grasps.py` | 多场景批量采样 | `python batch_fps_sample_grasps.py --scene_start 0 --scene_end 100` |
| `load_fps_grasps.py` | 工具模块：加载FPS采样后的抓取 | (作为模块导入使用) |

**输入**: 
- `data/DexGraspNet2.0/dex_grasps_new/scene_XXXX/leap_hand/XXX.npz`
- `data/DexGraspNet2.0/dex_grasps_success_indices/scene_XXXX/leap_hand/XXX.npz` (可选)

**输出**: `data/DexGraspNet2.0/fps_sampled_indices/scene_XXXX/leap_hand/XXX.npz`

**关键参数**:
- `--max_grasps`: 每场景最大抓取数 (默认5000)
- `--perturbation`: 抓取点随机扰动尺度 (默认0.02m), 用于处理同一抓取点的多个抓取

---

### Stage 3: IBS计算

| 脚本 | 功能 | 运行命令 |
|------|------|----------|
| `calculate_ibs_new.py` | 单场景IBS计算 | `python calculate_ibs_new.py --scene_id 55` |
| `batch_calculate_ibs.py` | 多场景批量计算 | `python batch_calculate_ibs.py --scene_start 0 --scene_end 100` |

**输入**:
- `data/DexGraspNet2.0/fps_sampled_indices/scene_XXXX/leap_hand/XXX.npz`
- 场景数据和物体mesh

**输出** (存储到 `data/ibsdata/`):
- `ibs/scene_XXXX.npy`: IBS体素数据 `(N, 40, 40, 40, 3)`
- `w2h_trans/scene_XXXX.npy`: 世界到手坐标系变换矩阵 `(N, 4, 4)`
- `hand_dis/scene_XXXX.npy`: 手到IBS点的距离 `(N, 40, 40, 40)`

---

### Stage 4: 视角标注（LASDiffusion训练前置）

| 脚本 | 功能 | 运行命令 |
|------|------|----------|
| `annotate_ibs_for_view.py` | 为每个视角标注可见IBS | `python annotate_ibs_for_view.py --scene_start 0 --scene_end 100` |

**输入**: 
- `data/ibsdata/` (IBS数据)
- `data/DexGraspNet2.0/scenes/` (场景点云)

**输出**: `data/ibsdata/scene_valid_ids/scene_XXXX/view_YYYY.npy`

**说明**:
- 此步骤是 **LASDiffusion训练的前置条件**
- 目的是筛选出从每个相机视角可见的IBS（grasp点在场景点云1cm范围内）
- 如果不需要视角过滤，可以在 `IBS_Dataset` 中设置 `use_view_filter=False`

---

### 工具脚本

| 脚本 | 功能 |
|------|------|
| `scene.py` | 场景数据加载类，被多个脚本使用 |

---

## 📦 IBS数据格式

IBS体素为 `(40, 40, 40, 3)` 的4维数组，三个通道含义：

| 通道 | 名称 | 描述 | 值范围 |
|------|------|------|--------|
| 0 | `occupancy` | IBS占用体素 | -1 (空) / 1 (占用) |
| 1 | `contact` | 手指接触区域 | 0 (非接触) / 1 (接触) |
| 2 | `thumb_contact` | 大拇指接触区域 | 0 (非接触) / 2 (接触) |

**体素参数**:
- 空间范围: `[-0.1, 0.1]^3` (手坐标系)
- 分辨率: `0.005m` (5mm)
- 网格大小: `40 × 40 × 40`

---

## 🚀 快速开始

### 完整流水线运行（包括训练准备）

```bash
# 1. 仿真筛选 (需要IsaacGym)
python batch_filter_grasps.py --scene_start 0 --scene_end 100 --gpu_ids 0,1,2,3

# 2. FPS采样
python batch_fps_sample_grasps.py --scene_start 0 --scene_end 100

# 3. IBS计算
python batch_calculate_ibs.py --scene_start 0 --scene_end 100

# 4. 视角标注 (LASDiffusion训练前需要)
python annotate_ibs_for_view.py --scene_start 0 --scene_end 100 --gpu_ids 0,1,2,3
```

### 跳过仿真筛选

如果没有安装IsaacGym或者想跳过仿真筛选步骤：

```bash
# 直接运行FPS采样 (会默认所有抓取成功)
python batch_fps_sample_grasps.py --scene_start 0 --scene_end 100

# 然后运行IBS计算
python batch_calculate_ibs.py --scene_start 0 --scene_end 100
```

### 单场景测试

```bash
# 处理单个场景 (scene_0055)
python fps_sample_grasps.py --scene_id 55
python calculate_ibs_new.py --scene_id 55 --visualize
```

---

## ⚙️ 默认配置

所有脚本都使用代码内置的默认配置，无需外部配置文件。

### Scene类默认参数
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

### IBS计算默认参数
```python
IBSConfig(
    bound=0.1,           # 空间范围 [-0.1, 0.1]
    resolution=0.005,    # 体素分辨率 5mm
    delta=0.005,         # IBS厚度阈值
    epsilon=1e-5,        # 迭代收敛阈值
    max_iteration=20,    # 最大迭代次数
    voxel_size=40,       # 体素网格大小
    contact_delta=0.0075,    # 接触点阈值
    thumb_contact_delta=0.0085  # 大拇指接触阈值
)
```

---

## 📚 相关模块

- `src/cadgrasp/ibs/utils/ibs_repr.py`: IBS数据类 (`IBS`, `IBSBatch`, `IBSConfig`)
- `src/cadgrasp/ibs/utils/transforms.py`: 坐标变换工具
- `thirdparty/LASDiffusion/network/data_loader.py`: IBS数据集加载器
- `thirdparty/LASDiffusion/generate.py`: IBS生成/推理
