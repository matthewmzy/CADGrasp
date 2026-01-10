# CADGrasp - Copilot Instructions

## 项目概述

**CADGrasp** (论文标题: *CADGrasp: Learning Contact and Collision Aware General Dexterous Grasping in Cluttered Scenes*) 是一个面向杂乱场景的接触感知和碰撞感知的灵巧手抓取流水线项目。

### 核心创新: Interaction Bisector Surface (IBS)

IBS是本项目提出的关键中间抓取表征，定义为：**手和物体之间距离相等的点组成的集合**（即手-物体距离的等值面）。

IBS表示形式：
- **点云形式**: 用于可视化和优化器输入
- **体素形式**: 用于Diffusion网络的输入输出（40×40×40体素网格）

IBS体素包含三个通道：
1. `occupancy`: IBS占用体素 (值: -1/1)
2. `contact`: 手指接触区域 (值: 1)
3. `thumb_contact`: 大拇指接触区域 (值: 2)

---

## 项目工作流程

### 完整Pipeline

```
DexGraspNet2.0数据 → 预处理(深度图转点云) → IsaacGym仿真筛选 → IBS数据计算 → LASDiffusion训练 → IBS预测 → 位姿优化 → IsaacGym评测
```

### 阶段详解

#### Stage 1: 数据准备与预处理
- **数据来源**: DexGraspNet2.0 (Baseline项目)
- **预处理脚本**: `src/cadgrasp/baseline/preprocess/compute_network_input_all.py`
- **主要任务**: 深度图转点云，生成网络输入数据

#### Stage 2: 抓取数据筛选
- **工具**: IsaacGym仿真器
- **新脚本** (推荐):
  - `src/cadgrasp/ibs/scripts/filter_grasps_by_sim.py`: 单场景筛选，输出成功索引
  - `src/cadgrasp/ibs/scripts/batch_filter_grasps.py`: 多场景批量筛选
  - `src/cadgrasp/ibs/scripts/load_success_grasps.py`: 工具模块，加载成功抓取数据
- **旧脚本** (已弃用): `batch_eval_grasps.py`, `evaluate_grasps.py`
- **数据流**:
  - 输入: `data/DexGraspNet2.0/dex_grasps_new/scene_XXXX/leap_hand/XXX.npz`
  - 输出: `data/DexGraspNet2.0/dex_grasps_success_indices/scene_XXXX/leap_hand/XXX.npz` (仅含 `success_indices`)
- **目的**: 筛选稳健的抓取数据用于后续IBS计算

#### Stage 2.5: FPS抓取采样
- **目的**: 对成功抓取进行FPS采样，控制每场景抓取数量上界
- **核心脚本**:
  - `src/cadgrasp/ibs/scripts/fps_sample_grasps.py`: 单场景FPS采样
  - `src/cadgrasp/ibs/scripts/batch_fps_sample_grasps.py`: 多场景批量采样
  - `src/cadgrasp/ibs/scripts/load_fps_grasps.py`: 工具模块，加载FPS采样后的抓取
- **数据流**:
  - 输入: `dex_grasps_new` + `dex_grasps_success_indices` (可选，不存在则默认全部成功)
  - 输出: `data/DexGraspNet2.0/fps_sampled_indices/scene_XXXX/leap_hand/XXX.npz` (仅含 `fps_indices`)
- **关键参数**:
  - `max_grasps`: 每场景最大抓取数 (默认5000)
  - `perturbation`: 随机扰动尺度 (默认0.02m/2cm)，用于处理同一grasp point上的多个抓取

#### Stage 3: IBS数据集生成
- **新脚本** (推荐):
  - `src/cadgrasp/ibs/scripts/calculate_ibs_new.py`: 从FPS索引加载抓取并计算IBS
  - `src/cadgrasp/ibs/scripts/batch_calculate_ibs.py`: 多场景批量计算
- **IBS数据类**: `src/cadgrasp/ibs/utils/ibs_repr.py`
  - `IBS`: 单个IBS数据表示，支持点云/体素转换、接触点提取等
  - `IBSBatch`: 批量IBS数据处理
  - `IBSConfig`: IBS计算配置参数
- **旧脚本**: `calculate_ibs.py`, `batch_cal_ibs.py` (从success_grasps.npz加载)
- **输入**: FPS采样后的抓取数据 + 场景点云 + 手模型
- **输出**: 
  - `ibs/scene_XXXX.npy`: IBS体素数据 (N, 40, 40, 40, 3)
  - `w2h_trans/scene_XXXX.npy`: 世界坐标系到手坐标系变换矩阵
  - `hand_dis/scene_XXXX.npy`: 手到IBS点的距离
- **IBS体素通道**:
  - 通道0 (occupancy): IBS占用体素
  - 通道1 (contact): 接触区域
  - 通道2 (thumb_contact): 大拇指接触区域

#### Stage 4: LASDiffusion网络训练
- **代码位置**: `LASDiffusion/`
- **模型**: 基于UNet的3D Diffusion模型
- **训练脚本**: `LASDiffusion/train.py`
- **数据加载**: `LASDiffusion/network/data_loader.py`
- **输入**: 场景点云体素 (B, 1, 40, 40, 40)
- **输出**: IBS体素 (B, 2, 40, 40, 40)


#### Stage 5: IBS预测与位姿优化
- **预测入口**: `LASDiffusion/generate.py` → `generate_based_on_scene_obs()`
- **优化器**: `src/cadgrasp/optimizer/`
  - `AdamOpt.py`: 主优化器封装
  - `IBSAdam.py`: IBS约束的Adam优化
  - `HandModel.py`: 手模型正向运动学与碰撞检测
- **能量函数**:
  - 接触匹配能量 (contact_match_energy)
  - 关节限制能量 (joint_limits_energy)
  - 自穿透能量 (self_penetration_energy)

#### Stage 6: 仿真评测
- **评测脚本**: `src/cadgrasp/baseline/eval/evaluate_dexterous.py`
- **预测脚本**: `src/cadgrasp/evaluator/predict.py`, `eval/new_predict.py`
- **仿真器**: IsaacGym (`src/cadgrasp/baseline/utils/simulator/`)
- **评测器**: `src/cadgrasp/baseline/utils/data_evaluator/`

---

## 目录结构说明

```
CADGrasp/
├── src/cadgrasp/                    # 主Python包
│   ├── baseline/                    # DexGraspNet 2.0 baseline代码
│   │   ├── network/                 # 神经网络架构(CVAE, Diffusion等)
│   │   ├── eval/                    # 评测脚本
│   │   ├── preprocess/              # 数据预处理
│   │   └── utils/                   # 工具类(robot_model, simulator等)
│   ├── ibs/                         # IBS数据处理
│   │   ├── scripts/                 # IBS计算脚本
│   │   │   ├── calculate_ibs.py     # 核心IBS计算
│   │   │   ├── batch_cal_ibs.py     # 批量计算
│   │   │   └── annotate_ibs_for_view.py  # 视角标注
│   │   └── utils/                   # 变换工具
│   ├── optimizer/                   # 位姿优化器
│   │   ├── AdamOpt.py              # Adam优化封装
│   │   ├── IBSAdam.py              # IBS约束优化
│   │   ├── HandModel.py            # 手模型
│   │   └── ibs_func.py             # IBS相关函数
│   └── evaluator/                   # 端到端评测
│       ├── predict.py              # 预测入口
│       └── eval/                   # 评测脚本
│
├── LASDiffusion/                    # IBS体素Diffusion模型
│   ├── train.py                    # 训练入口
│   ├── generate.py                 # 生成/推理
│   ├── network/
│   │   ├── model.py                # OccupancyDiffusion模型
│   │   ├── model_trainer.py        # PyTorch Lightning训练器
│   │   ├── data_loader.py          # IBS数据集
│   │   └── unet.py                 # 3D UNet backbone
│   └── utils/
│       └── visualize_ibs_vox.py    # IBS体素可视化
│
├── robot_models/                    # 机器人URDF和mesh
│   ├── urdf/                       # URDF文件
│   ├── meshes/                     # 碰撞和可视化mesh
│   └── meta/                       # 元数据(关节限制等)
│
├── configs/                         # Hydra/YAML配置
│   ├── evaluator_predict.yaml      # 评测配置
│   ├── optimizer/                  # 优化器配置
│   └── network/                    # 网络训练配置
│
├── data/                            # 数据目录
│   ├── DexGraspNet2.0/             # 下载的数据集
│   │   ├── scenes/                 # 场景数据
│   │   ├── meshdata/               # 物体mesh
│   │   ├── dex_grasps_new/         # 抓取标注
│   │   ├── dex_grasps_success_indices/  # 仿真筛选后的成功索引
│   │   └── fps_sampled_indices/    # FPS采样后的索引
│   ├── ibsdata/                    # 生成的IBS数据(ibs/, w2h_trans/, hand_dis/)
│   └── hfd.sh                      # 数据下载脚本
│
├── experiment_bashes/               # 实验脚本
│   ├── main_result.sh              # 主实验
│   └── ablation_*.sh               # 消融实验
│
├── thirdparty/                      # 第三方依赖(git submodule)
│   ├── MinkowskiEngine/            # 稀疏卷积
│   └── TorchSDF/                   # SDF计算
│
└── tests/                           # 可视化和数据工具脚本
    ├── vis_ibs.py                  # IBS体素可视化
    ├── vis_fps_grasps.py           # FPS采样抓取点可视化
    ├── visualize_dex_grasp.py      # 抓取标注可视化
    ├── visualize_scene.py          # 场景mesh可视化
    ├── visualize_dex_pred.py       # 网络预测结果可视化
    └── count_data.py               # 数据统计工具
```

---

## 关键类和函数

### IBS计算 (`calculate_ibs.py`)
```python
def calculate_IBS(cfg):
    # 核心IBS计算逻辑
    # 1. 加载场景和抓取数据
    # 2. 计算手表面点云和物体表面点云
    # 3. 对每个体素点计算到手和物体的距离
    # 4. 使用迭代法求解等距面
    # 5. 标注接触区域和大拇指接触区域
```

### Diffusion模型 (`LASDiffusion/network/model.py`)
```python
class OccupancyDiffusion(nn.Module):
    def training_loss(self, ibs, scene):
        # ibs: (B, 2, 40, 40, 40) - 目标IBS体素
        # scene: (B, 1, 40, 40, 40) - 条件场景体素
        # 返回: MSE损失 + 接触损失 + 大拇指接触损失
    
    def sample_based_on_scene(self, batch_size, scene_voxel, steps):
        # DDPM采样，基于场景体素条件生成IBS体素
```

### 位姿优化 (`IBSAdam.py`)
```python
class IBSAdam:
    def reset(self, ibs_triplets, ...):
        # 初始化优化，加载IBS数据(occupancy, contact, thumb_contact)
    
    def contact_match_energy(self):
        # 计算接触点与手掌内侧的匹配能量
    
    def joint_limits_energy(self):
        # 关节限制惩罚
    
    def self_penetration_energy(self):
        # 自穿透惩罚
```

### 手模型 (`HandModel.py`)
```python
class HandModel:
    # 基于pytorch_kinematics的手正向运动学
    # 支持LEAP Hand等灵巧手
    # 提供表面点采样、碰撞检测等功能
```

---

## 常用命令

### 数据预处理
```bash
python src/cadgrasp/baseline/preprocess/compute_network_input_all.py \
    --dataset graspnet --scene_id_start 100 --scene_id_end 190
```

### IBS数据生成
```bash
python src/cadgrasp/ibs/scripts/batch_cal_ibs.py
```

### 训练LASDiffusion
```bash
python LASDiffusion/train.py \
    --ibs_path IBSGrasp/ibsdata \
    --scene_pc_path data/scenes \
    --name LEAP_dif \
    --training_epoch 200000 \
    --batch_size 64
```

### 评测
```bash
# 预测抓取
python src/cadgrasp/baseline/eval/predict_dexterous_all.py \
    --ckpt data/DexGraspNet2.0-ckpts/.../ckpt_50000.pth \
    --dataset graspnet

# 仿真评测
python src/cadgrasp/baseline/eval/evaluate_dexterous_all.py \
    --dataset graspnet \
    --ckpt_path_list ...
```

---

## 坐标系约定

- **世界坐标系 (World Frame)**: GraspNet场景的全局坐标系
- **相机坐标系 (Camera Frame)**: 深度相机坐标系，由`extrinsics`变换到世界坐标系
- **手坐标系 (Hand Frame / IBS Frame)**: 以抓取点为原点，手掌朝向为z轴的局部坐标系
  - IBS体素在手坐标系下计算，范围: [-0.1, 0.1]^3，分辨率: 0.005

### 关键变换
- `c2w_trans`: 相机到世界变换
- `w2h_trans`: 世界到手坐标系变换
- `pred_ibs2cam`: 预测的IBS坐标系到相机坐标系变换

---

## 依赖说明

### 核心依赖
- PyTorch >= 2.0
- PyTorch Lightning >= 2.0
- PyTorch3D
- MinkowskiEngine (稀疏卷积，需从源码编译)
- TorchSDF (SDF计算，需从源码编译)
- IsaacGym (仿真评测，需NVIDIA账号下载)

### 安装顺序
1. PyTorch + CUDA
2. `pip install -e .` (安装cadgrasp包)
3. PyTorch3D
4. TorchSDF (`thirdparty/TorchSDF`)
5. MinkowskiEngine (`thirdparty/MinkowskiEngine`)
6. IsaacGym (可选)

---

## 配置文件说明

### `configs/evaluator_predict.yaml`
- 场景配置、手模型配置、三个阶段(初始位姿、IBS预测、位姿优化)的参数

### `IBSGrasp/conf/ibs.yaml`
- IBS计算参数：分辨率、体素范围、每场景最大IBS数量等

### `configs/network/train_dex_ours.yaml`
- 网络训练配置

---

## 注意事项

1. **GPU内存**: IBS计算和Diffusion训练需要较大显存，建议16GB+
2. **多GPU训练**: LASDiffusion支持DDP训练
3. **数据路径**: 注意`data/`目录下的符号链接或实际数据位置
4. **URDF路径**: 手模型URDF在`robot_models/urdf/`，需确保路径正确
5. **IsaacGym版本**: 仿真评测需要对应CUDA版本的IsaacGym
