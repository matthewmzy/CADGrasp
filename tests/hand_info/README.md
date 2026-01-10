# LEAP Hand Information Tools

这个目录包含用于可视化和编辑 LEAP Hand 相关信息的工具脚本。

## 文件说明

### 1. visualize_keypoints.py - 基于 Open3D 的交互式编辑器

使用 Open3D 的 `VisualizerWithKeyCallback` 创建的轻量级可视化工具。

**功能：**
- 加载 LEAP Hand URDF 模型
- 显示 penetration keypoints 和对应的透明球体（半径 0.025m）
- 支持键盘快捷键操作

**使用方法：**
```bash
cd /home/ubuntu/Documents/CADGrasp
python tests/hand_info/visualize_keypoints.py
```

**快捷键：**
| 按键 | 功能 |
|------|------|
| H | 显示帮助 |
| S | 保存 keypoints |
| Q | 退出 |
| R | 重置关节角度 |
| J | 修改关节角度 |
| A | 添加 keypoint |
| D | 删除 keypoint |
| E | 编辑 keypoint offset |
| L/K | 切换 link |
| N/P | 切换 keypoint |

---

### 2. visualize_keypoints_gui.py - 完整 GUI 编辑器

使用 Open3D GUI 模块创建的完整图形界面编辑器。

**功能：**
- 左侧控制面板：选择 link、管理 keypoints、调整 offset
- 滑动条实时调整关节角度
- 滑动条实时调整 keypoint 的 X/Y/Z offset
- 红色球体表示选中的 keypoint，蓝色表示未选中

**使用方法：**
```bash
cd /home/ubuntu/Documents/CADGrasp
python tests/hand_info/visualize_keypoints_gui.py
```

**依赖：**
```bash
pip install open3d numpy trimesh pytorch-kinematics transforms3d
```

---

### 3. visualize_keypoints_plotly.py - 浏览器可视化

使用 Plotly 在浏览器中展示交互式 3D 可视化。

**功能：**
- 在浏览器中显示手模型和 keypoints
- 每个 link 的 keypoints 使用不同颜色
- 鼠标悬停显示 keypoint 详细信息
- 支持导出 HTML 文件

**使用方法：**
```bash
cd /home/ubuntu/Documents/CADGrasp

# 基本用法
python tests/hand_info/visualize_keypoints_plotly.py

# 指定关节角度（弧度，逗号分隔）
python tests/hand_info/visualize_keypoints_plotly.py --joint_angles "0.5,0.3,0.2,..."

# 导出 HTML
python tests/hand_info/visualize_keypoints_plotly.py --output output.html
```

**依赖：**
```bash
pip install plotly numpy trimesh pytorch-kinematics transforms3d
```

---

## Penetration Keypoints 文件格式

keypoints 存储在 `robot_models/meta/leap_hand/penetration_keypoints.json`：

```json
{
    "link_name": [
        [x1, y1, z1],  // keypoint 1 在 link 局部坐标系下的 offset
        [x2, y2, z2],  // keypoint 2
        ...
    ],
    "another_link": [
        ...
    ]
}
```

**坐标系说明：**
- offset 是相对于 link 的局部坐标系
- 单位是米 (m)
- 球体半径 0.025m 是自碰撞检测的阈值（两个 keypoint 之间距离小于此值则产生惩罚）

---

## LEAP Hand Links 列表

| Link 名称 | 描述 |
|-----------|------|
| hand_base_link | 手掌基座 |
| mcp_joint | 食指 MCP 关节 |
| pip | 食指 PIP 关节 |
| dip | 食指 DIP 关节 |
| fingertip | 食指指尖 |
| mcp_joint_2 | 中指 MCP 关节 |
| pip_2 | 中指 PIP 关节 |
| dip_2 | 中指 DIP 关节 |
| fingertip_2 | 中指指尖 |
| mcp_joint_3 | 无名指 MCP 关节 |
| pip_3 | 无名指 PIP 关节 |
| dip_3 | 无名指 DIP 关节 |
| fingertip_3 | 无名指指尖 |
| pip_4 | 大拇指基座 |
| thumb_pip | 大拇指 PIP |
| thumb_dip | 大拇指 DIP |
| thumb_fingertip | 大拇指指尖 |

---

## 注意事项

1. **球体半径**: 固定为 0.025m，这是 `HandModel.get_self_penetration()` 中使用的自碰撞检测阈值

2. **坐标系**: keypoints 的 offset 是在 link 的局部坐标系下定义的，不是世界坐标系

3. **保存**: 修改后记得保存（按 S 键或点击保存按钮）

4. **文件路径**: 默认使用 `robot_models/urdf/leap_hand.urdf` 和 `robot_models/meta/leap_hand/penetration_keypoints.json`
