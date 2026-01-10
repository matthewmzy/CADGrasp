#!/usr/bin/env python3
"""
LEAP Hand Penetration Keypoints 可视化工具 - Plotly 版本

使用 Plotly 在浏览器中交互式可视化 LEAP Hand 和 Penetration Keypoints
支持：
1. 鼠标拖拽旋转视角
2. 滚轮缩放
3. 显示手模型和透明球体

用法：
    python visualize_keypoints_plotly.py [--joint_angles "0,0,0,..."]
"""

import json
import os
import sys
import argparse
import numpy as np
import plotly.graph_objects as go
import trimesh as tm
import transforms3d
from typing import Dict, List, Optional
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytorch_kinematics as pk
import torch
import urdf_parser_py.urdf as URDF_PARSER


def load_urdf_and_keypoints(urdf_path: str, keypoints_path: str):
    """加载 URDF 和 keypoints"""
    robot_urdf = URDF_PARSER.URDF.from_xml_file(urdf_path)
    robot_pk = pk.build_chain_from_urdf(open(urdf_path).read())
    
    keypoints = {}
    if os.path.exists(keypoints_path):
        with open(keypoints_path, 'r') as f:
            keypoints = json.load(f)
    
    return robot_urdf, robot_pk, keypoints


def get_link_transform(robot_pk, joint_angles: np.ndarray, link_name: str) -> np.ndarray:
    """获取 link 变换矩阵"""
    q = torch.tensor(joint_angles, dtype=torch.float).unsqueeze(0)
    fk = robot_pk.forward_kinematics(q)
    
    if link_name in fk:
        return fk[link_name].get_matrix()[0].numpy()
    return np.eye(4)


def load_link_mesh(robot_urdf, link_name: str) -> Optional[tm.Trimesh]:
    """加载 link mesh"""
    link = None
    for l in robot_urdf.links:
        if l.name == link_name:
            link = l
            break
    
    if link is None or len(link.visuals) == 0:
        return None
    
    meshes = []
    for visual in link.visuals:
        if visual.geometry.filename:
            filename = visual.geometry.filename
            if filename.startswith('package://'):
                filename = filename.replace('package://', '')
            mesh_path = str(PROJECT_ROOT / filename)
            
            if not os.path.exists(mesh_path):
                continue
            
            mesh = tm.load(mesh_path, force='mesh', process=False)
            
            try:
                scale = np.array(visual.geometry.scale).reshape([1, 3])
            except:
                scale = np.array([[1, 1, 1]])
            
            try:
                rotation = transforms3d.euler.euler2mat(*visual.origin.rpy)
                translation = np.array(visual.origin.xyz)
            except AttributeError:
                rotation = np.eye(3)
                translation = np.zeros(3)
            
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = translation
            
            mesh.apply_scale(scale)
            mesh.apply_transform(transform)
            meshes.append(mesh)
    
    if not meshes:
        return None
    
    return tm.util.concatenate(meshes)


def create_sphere_mesh(center: np.ndarray, radius: float, resolution: int = 20) -> tuple:
    """创建球体 mesh 数据用于 Plotly"""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z


def visualize_hand_with_keypoints(
    robot_urdf, 
    robot_pk, 
    keypoints: Dict[str, List[List[float]]],
    joint_angles: np.ndarray,
    sphere_radius: float = 0.025 / 2,
    output_html: str = None
):
    """创建 Plotly 可视化"""
    fig = go.Figure()
    
    link_names = [link.name for link in robot_urdf.links]
    
    # 添加手模型
    for link_name in link_names:
        mesh = load_link_mesh(robot_urdf, link_name)
        if mesh is None:
            continue
        
        transform = get_link_transform(robot_pk, joint_angles, link_name)
        mesh.apply_transform(transform)
        
        # 添加 mesh
        fig.add_trace(go.Mesh3d(
            x=mesh.vertices[:, 0],
            y=mesh.vertices[:, 1],
            z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0],
            j=mesh.faces[:, 1],
            k=mesh.faces[:, 2],
            color='lightgray',
            opacity=0.8,
            name=link_name,
            hoverinfo='name',
            showlegend=True,
            legendgroup='hand'
        ))
    
    # 添加 keypoints 球体
    colors = [
        'blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow',
        'pink', 'brown', 'gray', 'olive', 'navy', 'teal', 'maroon', 'lime'
    ]
    
    color_idx = 0
    for link_name, kps in keypoints.items():
        if not kps:
            continue
        
        transform = get_link_transform(robot_pk, joint_angles, link_name)
        color = colors[color_idx % len(colors)]
        color_idx += 1
        
        for kp_idx, kp_offset in enumerate(kps):
            kp_local = np.array([*kp_offset, 1.0])
            kp_world = transform @ kp_local
            
            # 创建球体
            x, y, z = create_sphere_mesh(kp_world[:3], sphere_radius)
            
            fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                opacity=0.5,
                showscale=False,
                colorscale=[[0, color], [1, color]],
                name=f"{link_name}[{kp_idx}]: {kp_offset}",
                hoverinfo='name',
                showlegend=True,
                legendgroup=link_name
            ))
            
            # 添加中心点标记
            fig.add_trace(go.Scatter3d(
                x=[kp_world[0]],
                y=[kp_world[1]],
                z=[kp_world[2]],
                mode='markers',
                marker=dict(size=5, color=color),
                name=f"center_{link_name}[{kp_idx}]",
                hoverinfo='text',
                hovertext=f"{link_name}[{kp_idx}]<br>offset: {kp_offset}<br>world: [{kp_world[0]:.4f}, {kp_world[1]:.4f}, {kp_world[2]:.4f}]",
                showlegend=False
            ))
    
    # 添加坐标系
    axis_length = 0.05
    fig.add_trace(go.Scatter3d(
        x=[0, axis_length], y=[0, 0], z=[0, 0],
        mode='lines', line=dict(color='red', width=5),
        name='X axis', showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, axis_length], z=[0, 0],
        mode='lines', line=dict(color='green', width=5),
        name='Y axis', showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, axis_length],
        mode='lines', line=dict(color='blue', width=5),
        name='Z axis', showlegend=False
    ))
    
    # 设置布局
    fig.update_layout(
        title={
            'text': 'LEAP Hand Penetration Keypoints Visualization<br><sup>球体半径: 0.025 / 2m (自碰撞检测阈值)</sup>',
            'x': 0.5,
            'xanchor': 'center'
        },
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=0, r=0, t=60, b=0)
    )
    
    if output_html:
        fig.write_html(output_html)
        print(f"[INFO] 可视化已保存到: {output_html}")
    
    fig.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description="LEAP Hand Keypoints 可视化")
    parser.add_argument('--urdf', type=str, 
                       default=str(PROJECT_ROOT / "robot_models/urdf/leap_hand.urdf"),
                       help="URDF 文件路径")
    parser.add_argument('--keypoints', type=str,
                       default=str(PROJECT_ROOT / "robot_models/meta/leap_hand/penetration_keypoints.json"),
                       help="keypoints JSON 文件路径")
    parser.add_argument('--joint_angles', type=str, default=None,
                       help="关节角度 (逗号分隔，弧度)")
    parser.add_argument('--output', type=str, default=None,
                       help="输出 HTML 文件路径")
    parser.add_argument('--sphere_radius', type=float, default=0.025 / 2,
                       help="球体半径")
    
    args = parser.parse_args()
    
    # 加载数据
    robot_urdf, robot_pk, keypoints = load_urdf_and_keypoints(args.urdf, args.keypoints)
    
    # 解析关节角度
    revolute_joints = [j for j in robot_urdf.joints if j.joint_type == 'revolute']
    n_dofs = len(revolute_joints)
    
    if args.joint_angles:
        joint_angles = np.array([float(x) for x in args.joint_angles.split(',')])
        if len(joint_angles) != n_dofs:
            print(f"[WARNING] 关节角度数量 ({len(joint_angles)}) != 关节数量 ({n_dofs})，使用零位姿")
            joint_angles = np.zeros(n_dofs)
    else:
        joint_angles = np.zeros(n_dofs)
    
    print(f"[INFO] URDF: {args.urdf}")
    print(f"[INFO] Keypoints: {args.keypoints}")
    print(f"[INFO] 关节数量: {n_dofs}")
    print(f"[INFO] Keypoints 数量: {sum(len(v) for v in keypoints.values())}")
    
    # 显示 keypoints 详情
    print("\n[INFO] Keypoints 详情:")
    for link, kps in keypoints.items():
        if kps:
            print(f"  {link}: {len(kps)} 个点")
            for i, kp in enumerate(kps):
                print(f"    [{i}]: {kp}")
    
    # 可视化
    visualize_hand_with_keypoints(
        robot_urdf, robot_pk, keypoints, joint_angles,
        sphere_radius=args.sphere_radius,
        output_html=args.output
    )


if __name__ == "__main__":
    main()
