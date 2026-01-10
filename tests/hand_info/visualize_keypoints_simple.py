#!/usr/bin/env python3
"""
交互式 LEAP Hand Penetration Keypoints 可视化和编辑工具

功能：
1. 加载 LEAP Hand URDF 并可视化
2. 实时显示 penetration keypoints（关节局部坐标系下的点）
3. 以 keypoints 为中心显示半径 0.025 的透明球
4. 支持实时修改 keypoints 并保存
5. 支持拖拽视角

用法：
    python visualize_keypoints.py

快捷键：
    - 鼠标左键拖拽：旋转视角
    - 鼠标滚轮：缩放
    - Q：退出
    - S：保存当前 keypoints 到 JSON
    - R：重置到默认关节角度
    - J：修改关节角度
    - A：添加新 keypoint
    - D：删除选中的 keypoint
    - E：编辑选中 keypoint 的 offset
    - N：选择下一个 keypoint
    - P：选择上一个 keypoint
    - L：选择下一个 link
    - K：选择上一个 link
    - H：显示帮助
"""

import json
import os
import sys
import numpy as np
import open3d as o3d
import trimesh as tm
import transforms3d
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytorch_kinematics as pk
import torch
import urdf_parser_py.urdf as URDF_PARSER


class LeapHandKeypointEditor:
    """LEAP Hand Penetration Keypoints 可视化和编辑器"""
    
    def __init__(self, urdf_path: str, keypoints_path: str, sphere_radius: float = 0.025):
        """
        初始化编辑器
        
        Args:
            urdf_path: URDF 文件路径
            keypoints_path: penetration_keypoints.json 路径
            sphere_radius: 透明球半径
        """
        self.urdf_path = urdf_path
        self.keypoints_path = keypoints_path
        self.sphere_radius = sphere_radius
        
        # 加载 URDF
        self.robot_urdf = URDF_PARSER.URDF.from_xml_file(urdf_path)
        self.robot_pk = pk.build_chain_from_urdf(open(urdf_path).read())
        
        # 获取所有 link 名称
        self.link_names = [link.name for link in self.robot_urdf.links]
        
        # 获取所有 revolute joints
        self.revolute_joints = [joint for joint in self.robot_urdf.joints if joint.joint_type == 'revolute']
        self.joint_names = [joint.name for joint in self.revolute_joints]
        self.n_dofs = len(self.revolute_joints)
        
        # 当前关节角度（零位姿）
        self.joint_angles = np.zeros(self.n_dofs)
        
        # 加载或初始化 keypoints
        self.keypoints = self._load_keypoints()
        
        # 当前选中状态
        self.selected_link_idx = 0
        self.selected_keypoint_idx = 0
        
        # 可视化元素
        self.vis = None
        self.mesh_geometries = {}  # link_name -> o3d.geometry.TriangleMesh
        self.sphere_geometries = {}  # (link_name, kp_idx) -> o3d.geometry.TriangleMesh
        self.coordinate_frames = {}  # link_name -> o3d.geometry.TriangleMesh
        
        # 颜色配置
        self.mesh_color = [0.7, 0.7, 0.7]  # 灰色手模型
        self.sphere_color_normal = [0.2, 0.6, 1.0]  # 蓝色正常球
        self.sphere_color_selected = [1.0, 0.3, 0.3]  # 红色选中球
        self.sphere_alpha = 0.5  # 透明度
        
    def _load_keypoints(self) -> Dict[str, List[List[float]]]:
        """加载 keypoints JSON 文件"""
        if os.path.exists(self.keypoints_path):
            with open(self.keypoints_path, 'r') as f:
                return json.load(f)
        else:
            # 初始化空的 keypoints 字典
            print(f"[WARNING] {self.keypoints_path} 不存在，创建空的 keypoints")
            return {link: [] for link in self.link_names}
    
    def save_keypoints(self):
        """保存 keypoints 到 JSON 文件"""
        # 确保目录存在
        os.makedirs(os.path.dirname(self.keypoints_path), exist_ok=True)
        
        # 只保存非空的 link
        keypoints_to_save = {k: v for k, v in self.keypoints.items() if v}
        
        with open(self.keypoints_path, 'w') as f:
            json.dump(keypoints_to_save, f, indent=4)
        print(f"[INFO] Keypoints 已保存到 {self.keypoints_path}")
    
    def _load_link_mesh(self, link_name: str) -> Optional[o3d.geometry.TriangleMesh]:
        """加载单个 link 的 mesh"""
        link = None
        for l in self.robot_urdf.links:
            if l.name == link_name:
                link = l
                break
        
        if link is None or len(link.visuals) == 0:
            return None
        
        meshes = []
        for visual in link.visuals:
            if visual.geometry.filename:
                filename = visual.geometry.filename
                # 处理 package:// 前缀
                if filename.startswith('package://'):
                    filename = filename.replace('package://', '')
                
                # 使用绝对路径
                mesh_path = str(PROJECT_ROOT / filename)
                
                if not os.path.exists(mesh_path):
                    print(f"[WARNING] Mesh 文件不存在: {mesh_path}")
                    continue
                
                # 加载 mesh
                mesh = tm.load(mesh_path, force='mesh', process=False)
                
                # 应用 scale
                try:
                    scale = np.array(visual.geometry.scale).reshape([1, 3])
                except:
                    scale = np.array([[1, 1, 1]])
                
                # 应用 origin transform
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
        
        # 合并所有 mesh
        combined_mesh = tm.util.concatenate(meshes)
        
        # 转换为 Open3D mesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(combined_mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(combined_mesh.faces)
        o3d_mesh.compute_vertex_normals()
        
        return o3d_mesh
    
    def _get_link_transform(self, link_name: str) -> np.ndarray:
        """获取 link 的世界坐标系变换矩阵"""
        # 使用 pytorch_kinematics 计算 FK
        q = torch.tensor(self.joint_angles, dtype=torch.float).unsqueeze(0)
        fk = self.robot_pk.forward_kinematics(q)
        
        if link_name in fk:
            transform = fk[link_name].get_matrix()[0].numpy()
            return transform
        else:
            return np.eye(4)
    
    def _create_sphere(self, center: np.ndarray, radius: float, color: List[float], alpha: float = 0.5) -> o3d.geometry.TriangleMesh:
        """创建透明球"""
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
        sphere.translate(center)
        sphere.paint_uniform_color(color)
        # Open3D 不直接支持透明度，但可以设置较浅的颜色来模拟
        # 对于真正的透明效果，可能需要使用其他渲染方式
        sphere.compute_vertex_normals()
        return sphere
    
    def _create_coordinate_frame(self, transform: np.ndarray, size: float = 0.03) -> o3d.geometry.TriangleMesh:
        """创建坐标系可视化"""
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        frame.transform(transform)
        return frame
    
    def update_visualization(self):
        """更新可视化"""
        if self.vis is None:
            return
        
        # 清除所有几何体
        self.vis.clear_geometries()
        
        # 更新每个 link 的 mesh
        for link_name in self.link_names:
            mesh = self._load_link_mesh(link_name)
            if mesh is not None:
                transform = self._get_link_transform(link_name)
                mesh.transform(transform)
                mesh.paint_uniform_color(self.mesh_color)
                self.vis.add_geometry(mesh)
        
        # 获取当前选中的 link
        links_with_keypoints = [ln for ln in self.link_names if ln in self.keypoints and self.keypoints[ln]]
        if not links_with_keypoints:
            links_with_keypoints = self.link_names
        
        if self.selected_link_idx >= len(links_with_keypoints):
            self.selected_link_idx = 0
        
        # 更新 keypoints 球体
        for link_name, kps in self.keypoints.items():
            if not kps:
                continue
            
            transform = self._get_link_transform(link_name)
            
            for kp_idx, kp_offset in enumerate(kps):
                # 将 keypoint 从局部坐标转换到世界坐标
                kp_local = np.array([*kp_offset, 1.0])
                kp_world = transform @ kp_local
                
                # 判断是否选中
                is_selected = (link_name == links_with_keypoints[self.selected_link_idx] if links_with_keypoints else False) and \
                             kp_idx == self.selected_keypoint_idx
                
                color = self.sphere_color_selected if is_selected else self.sphere_color_normal
                
                # 创建球体
                sphere = self._create_sphere(kp_world[:3], self.sphere_radius, color)
                self.vis.add_geometry(sphere)
                
                # 如果选中，显示坐标系
                if is_selected:
                    kp_transform = transform.copy()
                    kp_transform[:3, 3] = kp_world[:3]
                    frame = self._create_coordinate_frame(kp_transform, size=0.02)
                    self.vis.add_geometry(frame)
        
        # 添加世界坐标系
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        self.vis.add_geometry(world_frame)
        
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def _print_status(self):
        """打印当前状态"""
        links_with_keypoints = [ln for ln in self.link_names if ln in self.keypoints and self.keypoints[ln]]
        
        if links_with_keypoints and self.selected_link_idx < len(links_with_keypoints):
            selected_link = links_with_keypoints[self.selected_link_idx]
            kps = self.keypoints.get(selected_link, [])
            print(f"\n[STATUS] 选中 Link: {selected_link}")
            print(f"         Keypoints 数量: {len(kps)}")
            if kps and self.selected_keypoint_idx < len(kps):
                print(f"         选中 Keypoint [{self.selected_keypoint_idx}]: {kps[self.selected_keypoint_idx]}")
        else:
            print("\n[STATUS] 没有 keypoints")
    
    def _key_callback_help(self, vis):
        """显示帮助"""
        print("\n" + "="*60)
        print("LEAP Hand Penetration Keypoints 编辑器 - 快捷键")
        print("="*60)
        print("视角控制:")
        print("  鼠标左键拖拽 - 旋转视角")
        print("  鼠标滚轮     - 缩放")
        print("  Ctrl+左键    - 平移")
        print("-"*60)
        print("选择操作:")
        print("  L - 选择下一个 link")
        print("  K - 选择上一个 link")
        print("  N - 选择下一个 keypoint")
        print("  P - 选择上一个 keypoint")
        print("-"*60)
        print("编辑操作:")
        print("  A - 添加新 keypoint")
        print("  D - 删除选中的 keypoint")
        print("  E - 编辑选中 keypoint 的 offset")
        print("-"*60)
        print("关节控制:")
        print("  J - 修改关节角度")
        print("  R - 重置到零位姿")
        print("-"*60)
        print("其他:")
        print("  S - 保存 keypoints 到 JSON")
        print("  H - 显示此帮助")
        print("  Q - 退出")
        print("="*60 + "\n")
        return False
    
    def _key_callback_save(self, vis):
        """保存 keypoints"""
        self.save_keypoints()
        return False
    
    def _key_callback_quit(self, vis):
        """退出"""
        print("[INFO] 退出编辑器")
        vis.destroy_window()
        return True
    
    def _key_callback_reset_joints(self, vis):
        """重置关节角度"""
        self.joint_angles = np.zeros(self.n_dofs)
        print("[INFO] 关节角度已重置到零位姿")
        self.update_visualization()
        return False
    
    def _key_callback_modify_joints(self, vis):
        """修改关节角度"""
        print("\n[INPUT] 修改关节角度")
        print("当前关节角度:")
        for i, (name, angle) in enumerate(zip(self.joint_names, self.joint_angles)):
            print(f"  [{i}] {name}: {np.degrees(angle):.1f}°")
        
        try:
            idx_str = input("输入要修改的关节索引 (或 'all' 设置所有): ").strip()
            if idx_str.lower() == 'all':
                angle_str = input("输入角度 (度): ").strip()
                angle = np.radians(float(angle_str))
                self.joint_angles[:] = angle
            else:
                idx = int(idx_str)
                if 0 <= idx < self.n_dofs:
                    angle_str = input(f"输入 {self.joint_names[idx]} 的角度 (度): ").strip()
                    self.joint_angles[idx] = np.radians(float(angle_str))
                else:
                    print("[ERROR] 无效的关节索引")
        except ValueError:
            print("[ERROR] 输入无效")
        
        self.update_visualization()
        return False
    
    def _key_callback_next_link(self, vis):
        """选择下一个 link"""
        links_with_keypoints = [ln for ln in self.link_names if ln in self.keypoints and self.keypoints[ln]]
        if not links_with_keypoints:
            links_with_keypoints = self.link_names
        
        self.selected_link_idx = (self.selected_link_idx + 1) % len(links_with_keypoints)
        self.selected_keypoint_idx = 0
        self._print_status()
        self.update_visualization()
        return False
    
    def _key_callback_prev_link(self, vis):
        """选择上一个 link"""
        links_with_keypoints = [ln for ln in self.link_names if ln in self.keypoints and self.keypoints[ln]]
        if not links_with_keypoints:
            links_with_keypoints = self.link_names
        
        self.selected_link_idx = (self.selected_link_idx - 1) % len(links_with_keypoints)
        self.selected_keypoint_idx = 0
        self._print_status()
        self.update_visualization()
        return False
    
    def _key_callback_next_keypoint(self, vis):
        """选择下一个 keypoint"""
        links_with_keypoints = [ln for ln in self.link_names if ln in self.keypoints and self.keypoints[ln]]
        if links_with_keypoints and self.selected_link_idx < len(links_with_keypoints):
            selected_link = links_with_keypoints[self.selected_link_idx]
            kps = self.keypoints.get(selected_link, [])
            if kps:
                self.selected_keypoint_idx = (self.selected_keypoint_idx + 1) % len(kps)
                self._print_status()
                self.update_visualization()
        return False
    
    def _key_callback_prev_keypoint(self, vis):
        """选择上一个 keypoint"""
        links_with_keypoints = [ln for ln in self.link_names if ln in self.keypoints and self.keypoints[ln]]
        if links_with_keypoints and self.selected_link_idx < len(links_with_keypoints):
            selected_link = links_with_keypoints[self.selected_link_idx]
            kps = self.keypoints.get(selected_link, [])
            if kps:
                self.selected_keypoint_idx = (self.selected_keypoint_idx - 1) % len(kps)
                self._print_status()
                self.update_visualization()
        return False
    
    def _key_callback_add_keypoint(self, vis):
        """添加新 keypoint"""
        print("\n[INPUT] 添加新 keypoint")
        print("可用的 link:")
        for i, name in enumerate(self.link_names):
            print(f"  [{i}] {name}")
        
        try:
            idx_str = input("输入 link 索引: ").strip()
            idx = int(idx_str)
            if 0 <= idx < len(self.link_names):
                link_name = self.link_names[idx]
                offset_str = input("输入 offset [x, y, z] (逗号分隔): ").strip()
                offset = [float(x.strip()) for x in offset_str.split(',')]
                if len(offset) == 3:
                    if link_name not in self.keypoints:
                        self.keypoints[link_name] = []
                    self.keypoints[link_name].append(offset)
                    print(f"[INFO] 已添加 keypoint 到 {link_name}: {offset}")
                    
                    # 更新选中状态
                    links_with_keypoints = [ln for ln in self.link_names if ln in self.keypoints and self.keypoints[ln]]
                    self.selected_link_idx = links_with_keypoints.index(link_name) if link_name in links_with_keypoints else 0
                    self.selected_keypoint_idx = len(self.keypoints[link_name]) - 1
                else:
                    print("[ERROR] offset 必须是 3 个数值")
            else:
                print("[ERROR] 无效的 link 索引")
        except ValueError:
            print("[ERROR] 输入无效")
        
        self.update_visualization()
        return False
    
    def _key_callback_delete_keypoint(self, vis):
        """删除选中的 keypoint"""
        links_with_keypoints = [ln for ln in self.link_names if ln in self.keypoints and self.keypoints[ln]]
        if links_with_keypoints and self.selected_link_idx < len(links_with_keypoints):
            selected_link = links_with_keypoints[self.selected_link_idx]
            kps = self.keypoints.get(selected_link, [])
            if kps and self.selected_keypoint_idx < len(kps):
                deleted = kps.pop(self.selected_keypoint_idx)
                print(f"[INFO] 已删除 {selected_link} 的 keypoint [{self.selected_keypoint_idx}]: {deleted}")
                
                # 调整选中索引
                if self.selected_keypoint_idx >= len(kps) and len(kps) > 0:
                    self.selected_keypoint_idx = len(kps) - 1
                elif len(kps) == 0:
                    self.selected_keypoint_idx = 0
                
                self.update_visualization()
        return False
    
    def _key_callback_edit_keypoint(self, vis):
        """编辑选中 keypoint 的 offset"""
        links_with_keypoints = [ln for ln in self.link_names if ln in self.keypoints and self.keypoints[ln]]
        if links_with_keypoints and self.selected_link_idx < len(links_with_keypoints):
            selected_link = links_with_keypoints[self.selected_link_idx]
            kps = self.keypoints.get(selected_link, [])
            if kps and self.selected_keypoint_idx < len(kps):
                current = kps[self.selected_keypoint_idx]
                print(f"\n[INPUT] 编辑 {selected_link} 的 keypoint [{self.selected_keypoint_idx}]")
                print(f"当前值: {current}")
                
                try:
                    offset_str = input("输入新的 offset [x, y, z] (逗号分隔，直接回车保持不变): ").strip()
                    if offset_str:
                        offset = [float(x.strip()) for x in offset_str.split(',')]
                        if len(offset) == 3:
                            kps[self.selected_keypoint_idx] = offset
                            print(f"[INFO] 已更新为: {offset}")
                        else:
                            print("[ERROR] offset 必须是 3 个数值")
                except ValueError:
                    print("[ERROR] 输入无效")
                
                self.update_visualization()
        return False
    
    def run(self):
        """运行可视化编辑器"""
        print("\n" + "="*60)
        print("LEAP Hand Penetration Keypoints 编辑器")
        print("="*60)
        print(f"URDF: {self.urdf_path}")
        print(f"Keypoints: {self.keypoints_path}")
        print(f"球体半径: {self.sphere_radius}")
        print(f"关节数量: {self.n_dofs}")
        print("按 H 显示快捷键帮助")
        print("="*60 + "\n")
        
        # 创建可视化窗口
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="LEAP Hand Keypoints Editor", width=1280, height=720)
        
        # 注册按键回调
        self.vis.register_key_callback(ord('H'), self._key_callback_help)
        self.vis.register_key_callback(ord('S'), self._key_callback_save)
        self.vis.register_key_callback(ord('Q'), self._key_callback_quit)
        self.vis.register_key_callback(ord('R'), self._key_callback_reset_joints)
        self.vis.register_key_callback(ord('J'), self._key_callback_modify_joints)
        self.vis.register_key_callback(ord('L'), self._key_callback_next_link)
        self.vis.register_key_callback(ord('K'), self._key_callback_prev_link)
        self.vis.register_key_callback(ord('N'), self._key_callback_next_keypoint)
        self.vis.register_key_callback(ord('P'), self._key_callback_prev_keypoint)
        self.vis.register_key_callback(ord('A'), self._key_callback_add_keypoint)
        self.vis.register_key_callback(ord('D'), self._key_callback_delete_keypoint)
        self.vis.register_key_callback(ord('E'), self._key_callback_edit_keypoint)
        
        # 初始化显示
        self.update_visualization()
        self._print_status()
        
        # 设置渲染选项
        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.1])
        opt.mesh_show_wireframe = False
        opt.mesh_show_back_face = True
        
        # 设置视角
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.5)
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 1, 0])
        
        # 运行
        self.vis.run()
        self.vis.destroy_window()


def main():
    # 默认路径
    urdf_path = str(PROJECT_ROOT / "robot_models/urdf/leap_hand.urdf")
    keypoints_path = str(PROJECT_ROOT / "robot_models/meta/leap_hand/penetration_keypoints.json")
    
    # 检查文件是否存在
    if not os.path.exists(urdf_path):
        print(f"[ERROR] URDF 文件不存在: {urdf_path}")
        sys.exit(1)
    
    # 创建并运行编辑器
    editor = LeapHandKeypointEditor(
        urdf_path=urdf_path,
        keypoints_path=keypoints_path,
        sphere_radius=0.025
    )
    editor.run()


if __name__ == "__main__":
    main()
