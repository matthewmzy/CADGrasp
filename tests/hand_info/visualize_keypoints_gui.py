#!/usr/bin/env python3
"""
LEAP Hand Penetration Keypoints 交互式 GUI 编辑器

功能：
1. 加载 LEAP Hand URDF 并实时可视化
2. 通过滑动条实时调整关节角度
3. 通过滑动条实时调整 keypoint offset
4. 以 keypoints 为中心显示半径 0.025 的透明球
5. 支持拖拽视角、添加/删除 keypoints
6. 实时更新显示

依赖：
    pip install open3d numpy trimesh pytorch-kinematics

用法：
    python visualize_keypoints_gui.py
"""

import json
import os
import sys
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
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


class LeapHandKeypointGUIEditor:
    """LEAP Hand Penetration Keypoints GUI 编辑器"""
    
    MENU_OPEN = 1
    MENU_SAVE = 2
    MENU_QUIT = 3
    MENU_RESET_JOINTS = 11
    MENU_ADD_KEYPOINT = 21
    MENU_DELETE_KEYPOINT = 22
    MENU_HELP = 31
    
    def __init__(self, urdf_path: str, keypoints_path: str, sphere_radius: float = 0.025):
        """初始化 GUI 编辑器"""
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
        self.joint_limits = [(joint.limit.lower, joint.limit.upper) for joint in self.revolute_joints]
        self.n_dofs = len(self.revolute_joints)
        
        # 当前关节角度
        self.joint_angles = np.zeros(self.n_dofs)
        
        # 加载 keypoints
        self.keypoints = self._load_keypoints()
        
        # 当前选中状态
        self.selected_link = None
        self.selected_keypoint_idx = -1
        
        # GUI 组件
        self.window = None
        self.scene_widget = None
        self.joint_sliders = {}
        self.keypoint_sliders = {}
        self.link_dropdown = None
        self.keypoint_list = None
        
        # 材质
        self.hand_material = None
        self.sphere_material_normal = None
        self.sphere_material_selected = None
        
    def _load_keypoints(self) -> Dict[str, List[List[float]]]:
        """加载 keypoints JSON"""
        if os.path.exists(self.keypoints_path):
            with open(self.keypoints_path, 'r') as f:
                return json.load(f)
        else:
            print(f"[WARNING] {self.keypoints_path} 不存在，创建空的 keypoints")
            return {link: [] for link in self.link_names}
    
    def _save_keypoints(self):
        """保存 keypoints"""
        os.makedirs(os.path.dirname(self.keypoints_path), exist_ok=True)
        keypoints_to_save = {k: v for k, v in self.keypoints.items() if v}
        with open(self.keypoints_path, 'w') as f:
            json.dump(keypoints_to_save, f, indent=4)
        print(f"[INFO] Keypoints 已保存到 {self.keypoints_path}")
    
    def _load_link_mesh(self, link_name: str) -> Optional[o3d.geometry.TriangleMesh]:
        """加载 link mesh"""
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
        
        combined = tm.util.concatenate(meshes)
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(combined.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(combined.faces)
        o3d_mesh.compute_vertex_normals()
        return o3d_mesh
    
    def _get_link_transform(self, link_name: str) -> np.ndarray:
        """获取 link 变换矩阵"""
        q = torch.tensor(self.joint_angles, dtype=torch.float).unsqueeze(0)
        fk = self.robot_pk.forward_kinematics(q)
        
        if link_name in fk:
            return fk[link_name].get_matrix()[0].numpy()
        return np.eye(4)
    
    def _update_scene(self):
        """更新 3D 场景"""
        if self.scene_widget is None:
            return
        
        scene = self.scene_widget.scene
        
        # 清除现有几何体
        scene.clear_geometry()
        
        # 添加坐标系
        scene.add_geometry("world_frame", o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05), 
                          rendering.MaterialRecord())
        
        # 添加手模型的每个 link
        for link_name in self.link_names:
            mesh = self._load_link_mesh(link_name)
            if mesh is not None:
                transform = self._get_link_transform(link_name)
                mesh.transform(transform)
                
                # Use special color for selected link
                if link_name == self.selected_link:
                    selected_mat = rendering.MaterialRecord()
                    selected_mat.shader = "defaultLit"
                    selected_mat.base_color = [1.0, 0.6, 0.2, 0.9]  # Orange for selected
                    scene.add_geometry(f"link_{link_name}", mesh, selected_mat)
                else:
                    scene.add_geometry(f"link_{link_name}", mesh, self.hand_material)
        
        # Add large coordinate frame at selected link's origin
        if self.selected_link:
            transform = self._get_link_transform(self.selected_link)
            link_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
            link_frame.transform(transform)
            scene.add_geometry("selected_link_frame", link_frame, rendering.MaterialRecord())
        
        # 添加 keypoints 球体
        for link_name, kps in self.keypoints.items():
            if not kps:
                continue
            
            transform = self._get_link_transform(link_name)
            
            for kp_idx, kp_offset in enumerate(kps):
                kp_local = np.array([*kp_offset, 1.0])
                kp_world = transform @ kp_local
                
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.sphere_radius, resolution=16)
                sphere.translate(kp_world[:3])
                sphere.compute_vertex_normals()
                
                is_selected = (link_name == self.selected_link and kp_idx == self.selected_keypoint_idx)
                mat = self.sphere_material_selected if is_selected else self.sphere_material_normal
                
                scene.add_geometry(f"sphere_{link_name}_{kp_idx}", sphere, mat)
                
                # 为选中的 keypoint 添加局部坐标系
                if is_selected:
                    kp_frame_transform = transform.copy()
                    kp_frame_transform[:3, 3] = kp_world[:3]
                    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
                    frame.transform(kp_frame_transform)
                    scene.add_geometry("selected_frame", frame, rendering.MaterialRecord())
    
    def _update_keypoint_list(self):
        """更新 keypoint 列表显示"""
        if self.keypoint_list is None or self.selected_link is None:
            return
        
        kps = self.keypoints.get(self.selected_link, [])
        items = [f"[{i}] ({kp[0]:.4f}, {kp[1]:.4f}, {kp[2]:.4f})" for i, kp in enumerate(kps)]
        self.keypoint_list.set_items(items)
        
        if self.selected_keypoint_idx >= 0 and self.selected_keypoint_idx < len(kps):
            self.keypoint_list.selected_index = self.selected_keypoint_idx
    
    def _update_keypoint_sliders(self):
        """更新 keypoint offset 滑动条"""
        if self.selected_link is None or self.selected_keypoint_idx < 0:
            return
        
        kps = self.keypoints.get(self.selected_link, [])
        if self.selected_keypoint_idx >= len(kps):
            return
        
        offset = kps[self.selected_keypoint_idx]
        
        if 'x' in self.keypoint_sliders:
            self.keypoint_sliders['x'].double_value = offset[0]
        if 'y' in self.keypoint_sliders:
            self.keypoint_sliders['y'].double_value = offset[1]
        if 'z' in self.keypoint_sliders:
            self.keypoint_sliders['z'].double_value = offset[2]
    
    def _on_joint_slider_changed(self, idx: int, value: float):
        """关节滑动条变化回调"""
        self.joint_angles[idx] = value
        self._update_scene()
    
    def _on_keypoint_slider_changed(self, axis: str, value: float):
        """Keypoint offset 滑动条变化回调"""
        if self.selected_link is None or self.selected_keypoint_idx < 0:
            return
        
        kps = self.keypoints.get(self.selected_link, [])
        if self.selected_keypoint_idx >= len(kps):
            return
        
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
        kps[self.selected_keypoint_idx][axis_idx] = value
        
        self._update_keypoint_list()
        self._update_scene()
    
    def _on_link_dropdown_changed(self, new_val: str, idx: int):
        """Link 下拉框变化回调"""
        self.selected_link = new_val
        self.selected_keypoint_idx = 0 if self.keypoints.get(new_val, []) else -1
        self._update_keypoint_list()
        self._update_keypoint_sliders()
        self._update_scene()
    
    def _on_keypoint_list_selected(self, new_val: str, is_double_click: bool):
        """Keypoint 列表选择回调"""
        if self.keypoint_list is not None:
            self.selected_keypoint_idx = self.keypoint_list.selected_index
            self._update_keypoint_sliders()
            self._update_scene()
    
    def _on_add_keypoint(self):
        """添加 keypoint"""
        if self.selected_link is None:
            return
        
        if self.selected_link not in self.keypoints:
            self.keypoints[self.selected_link] = []
        
        self.keypoints[self.selected_link].append([0.0, 0.0, 0.0])
        self.selected_keypoint_idx = len(self.keypoints[self.selected_link]) - 1
        
        self._update_keypoint_list()
        self._update_keypoint_sliders()
        self._update_scene()
    
    def _on_delete_keypoint(self):
        """删除 keypoint"""
        if self.selected_link is None or self.selected_keypoint_idx < 0:
            return
        
        kps = self.keypoints.get(self.selected_link, [])
        if self.selected_keypoint_idx < len(kps):
            kps.pop(self.selected_keypoint_idx)
            if self.selected_keypoint_idx >= len(kps):
                self.selected_keypoint_idx = len(kps) - 1 if kps else -1
        
        self._update_keypoint_list()
        self._update_keypoint_sliders()
        self._update_scene()
    
    def _on_save(self):
        """保存回调"""
        self._save_keypoints()
    
    def _on_reset_joints(self):
        """重置关节角度"""
        self.joint_angles = np.zeros(self.n_dofs)
        for i, slider in enumerate(self.joint_sliders.values()):
            slider.double_value = 0.0
        self._update_scene()
    
    def _on_menu_clicked(self, menu_id: int):
        """菜单点击回调"""
        if menu_id == self.MENU_SAVE:
            self._on_save()
        elif menu_id == self.MENU_QUIT:
            gui.Application.instance.quit()
        elif menu_id == self.MENU_RESET_JOINTS:
            self._on_reset_joints()
        elif menu_id == self.MENU_ADD_KEYPOINT:
            self._on_add_keypoint()
        elif menu_id == self.MENU_DELETE_KEYPOINT:
            self._on_delete_keypoint()
        elif menu_id == self.MENU_HELP:
            self._show_help()
    
    def _show_help(self):
        """显示帮助对话框"""
        em = self.window.theme.font_size
        dlg = gui.Dialog("Help")
        
        layout = gui.Vert(0, gui.Margins(em, em, em, em))
        layout.add_child(gui.Label("LEAP Hand Penetration Keypoints Editor"))
        layout.add_child(gui.Label(""))
        layout.add_child(gui.Label("Instructions:"))
        layout.add_child(gui.Label("1. Select link from dropdown on the left"))
        layout.add_child(gui.Label("2. Use sliders to adjust joint angles"))
        layout.add_child(gui.Label("3. Click keypoint list to select point"))
        layout.add_child(gui.Label("4. Use X/Y/Z sliders to adjust offset"))
        layout.add_child(gui.Label("5. Blue=normal, Red=selected keypoint"))
        layout.add_child(gui.Label("6. Sphere radius: 0.025m (collision threshold)"))
        layout.add_child(gui.Label(""))
        layout.add_child(gui.Label("View: Left-drag=rotate, Scroll=zoom, Right=pan"))
        
        ok_btn = gui.Button("OK")
        ok_btn.set_on_clicked(lambda: self.window.close_dialog())
        layout.add_child(ok_btn)
        
        dlg.add_child(layout)
        self.window.show_dialog(dlg)
    
    def run(self):
        """运行 GUI 应用"""
        app = gui.Application.instance
        app.initialize()
        
        # 创建窗口
        self.window = app.create_window("LEAP Hand Keypoints Editor", 1600, 900)
        em = self.window.theme.font_size
        
        # 创建材质
        self.hand_material = rendering.MaterialRecord()
        self.hand_material.shader = "defaultLit"
        self.hand_material.base_color = [0.7, 0.7, 0.7, 1.0]
        
        self.sphere_material_normal = rendering.MaterialRecord()
        self.sphere_material_normal.shader = "defaultLit"
        self.sphere_material_normal.base_color = [0.2, 0.6, 1.0, 0.6]
        
        self.sphere_material_selected = rendering.MaterialRecord()
        self.sphere_material_selected.shader = "defaultLit"
        self.sphere_material_selected.base_color = [1.0, 0.3, 0.3, 0.8]
        
        # 创建 3D 场景
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.1, 0.1, 0.15, 1.0])
        self.scene_widget.scene.scene.set_sun_light(
            [0.577, -0.577, -0.577], [1.0, 1.0, 1.0], 100000)
        self.scene_widget.scene.scene.enable_sun_light(True)
        
        # 设置相机
        bounds = o3d.geometry.AxisAlignedBoundingBox([-0.15, -0.15, -0.15], [0.15, 0.15, 0.15])
        self.scene_widget.setup_camera(60, bounds, [0, 0, 0])
        self.scene_widget.look_at([0, 0, 0], [0.3, 0.3, 0.3], [0, 1, 0])
        
        # ====== 创建左侧控制面板 ======
        panel = gui.Vert(0, gui.Margins(0.5*em, 0.5*em, 0.5*em, 0.5*em))
        
        # --- Link Selection ---
        panel.add_child(gui.Label("Select Link:"))
        self.link_dropdown = gui.Combobox()
        links_with_kp = [ln for ln in self.link_names if ln in self.keypoints and self.keypoints[ln]]
        if not links_with_kp:
            links_with_kp = self.link_names
        for ln in self.link_names:
            self.link_dropdown.add_item(ln)
        self.link_dropdown.selected_index = 0
        self.selected_link = self.link_names[0]
        self.link_dropdown.set_on_selection_changed(self._on_link_dropdown_changed)
        panel.add_child(self.link_dropdown)
        
        panel.add_fixed(0.5 * em)
        
        # --- Keypoints 列表 ---
        panel.add_child(gui.Label("Keypoints:"))
        self.keypoint_list = gui.ListView()
        self.keypoint_list.set_max_visible_items(5)
        self.keypoint_list.set_on_selection_changed(self._on_keypoint_list_selected)
        panel.add_child(self.keypoint_list)
        
        # Keypoint buttons
        kp_btn_layout = gui.Horiz(0.5 * em)
        add_btn = gui.Button("Add")
        add_btn.set_on_clicked(self._on_add_keypoint)
        del_btn = gui.Button("Delete")
        del_btn.set_on_clicked(self._on_delete_keypoint)
        kp_btn_layout.add_child(add_btn)
        kp_btn_layout.add_child(del_btn)
        panel.add_child(kp_btn_layout)
        
        panel.add_fixed(0.5 * em)
        
        # --- Keypoint Offset 滑动条 ---
        panel.add_child(gui.Label("Keypoint Offset:"))
        
        for axis in ['x', 'y', 'z']:
            row = gui.Horiz(0.5 * em)
            row.add_child(gui.Label(f"{axis.upper()}:"))
            slider = gui.Slider(gui.Slider.DOUBLE)
            slider.set_limits(-0.1, 0.1)
            slider.double_value = 0.0
            slider.set_on_value_changed(lambda val, a=axis: self._on_keypoint_slider_changed(a, val))
            self.keypoint_sliders[axis] = slider
            row.add_child(slider)
            panel.add_child(row)
        
        panel.add_fixed(em)
        panel.add_child(gui.Label("="*30))
        panel.add_fixed(0.5 * em)
        
        # --- Joint Angle Sliders ---
        panel.add_child(gui.Label("Joint Angles (rad):"))
        
        # 创建滚动区域容纳所有关节滑动条
        joint_scroll = gui.ScrollableVert(0, gui.Margins(0, 0, 0, 0))
        
        for i, (name, limits) in enumerate(zip(self.joint_names, self.joint_limits)):
            row = gui.Horiz(0.25 * em)
            # 截断长名称
            short_name = name[:12] + ".." if len(name) > 14 else name
            row.add_child(gui.Label(f"{short_name}:"))
            slider = gui.Slider(gui.Slider.DOUBLE)
            slider.set_limits(limits[0], limits[1])
            slider.double_value = 0.0
            slider.set_on_value_changed(lambda val, idx=i: self._on_joint_slider_changed(idx, val))
            self.joint_sliders[i] = slider
            row.add_child(slider)
            joint_scroll.add_child(row)
        
        panel.add_child(joint_scroll)
        
        panel.add_fixed(em)
        
        # --- Action Buttons ---
        btn_layout = gui.Horiz(0.5 * em)
        save_btn = gui.Button("Save")
        save_btn.set_on_clicked(self._on_save)
        reset_btn = gui.Button("Reset Joints")
        reset_btn.set_on_clicked(self._on_reset_joints)
        btn_layout.add_child(save_btn)
        btn_layout.add_child(reset_btn)
        panel.add_child(btn_layout)
        
        # ====== Create Menu ======
        if gui.Application.instance.menubar is None:
            menu = gui.Menu()
            file_menu = gui.Menu()
            file_menu.add_item("Save", self.MENU_SAVE)
            file_menu.add_separator()
            file_menu.add_item("Quit", self.MENU_QUIT)
            menu.add_menu("File", file_menu)
            
            edit_menu = gui.Menu()
            edit_menu.add_item("Reset Joints", self.MENU_RESET_JOINTS)
            edit_menu.add_separator()
            edit_menu.add_item("Add Keypoint", self.MENU_ADD_KEYPOINT)
            edit_menu.add_item("Delete Keypoint", self.MENU_DELETE_KEYPOINT)
            menu.add_menu("Edit", edit_menu)
            
            help_menu = gui.Menu()
            help_menu.add_item("Help", self.MENU_HELP)
            menu.add_menu("Help", help_menu)
            
            gui.Application.instance.menubar = menu
        
        # 为每个菜单项单独注册回调
        self.window.set_on_menu_item_activated(self.MENU_SAVE, self._on_save)
        self.window.set_on_menu_item_activated(self.MENU_QUIT, lambda: gui.Application.instance.quit())
        self.window.set_on_menu_item_activated(self.MENU_RESET_JOINTS, self._on_reset_joints)
        self.window.set_on_menu_item_activated(self.MENU_ADD_KEYPOINT, self._on_add_keypoint)
        self.window.set_on_menu_item_activated(self.MENU_DELETE_KEYPOINT, self._on_delete_keypoint)
        self.window.set_on_menu_item_activated(self.MENU_HELP, self._show_help)
        
        # ====== 布局 ======
        self.window.add_child(self.scene_widget)
        self.window.add_child(panel)
        
        # 设置布局回调
        def on_layout(layout_context):
            content_rect = self.window.content_rect
            panel_width = 22 * em
            panel_height = content_rect.height
            
            # 3D 场景占据右侧
            self.scene_widget.frame = gui.Rect(
                panel_width, content_rect.y,
                content_rect.width - panel_width, content_rect.height)
            
            # 控制面板在左侧
            panel.frame = gui.Rect(
                content_rect.x, content_rect.y,
                panel_width, panel_height)
        
        self.window.set_on_layout(on_layout)
        
        # 初始化显示
        if self.keypoints.get(self.selected_link, []):
            self.selected_keypoint_idx = 0
        else:
            self.selected_keypoint_idx = -1
        
        self._update_keypoint_list()
        self._update_keypoint_sliders()
        self._update_scene()
        
        # 运行
        app.run()


def main():
    urdf_path = str(PROJECT_ROOT / "robot_models/urdf/leap_hand.urdf")
    keypoints_path = str(PROJECT_ROOT / "robot_models/meta/leap_hand/penetration_keypoints.json")
    
    if not os.path.exists(urdf_path):
        print(f"[ERROR] URDF 文件不存在: {urdf_path}")
        sys.exit(1)
    
    editor = LeapHandKeypointGUIEditor(
        urdf_path=urdf_path,
        keypoints_path=keypoints_path,
        sphere_radius=0.025 / 2
    )
    editor.run()


if __name__ == "__main__":
    main()
