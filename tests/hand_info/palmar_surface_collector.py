#!/usr/bin/env python3
"""
LEAP Hand Palmar Surface Point Cloud Collector

A GUI tool for collecting palm-side surface points from LEAP Hand mesh.

Algorithm:
1. Sample surface points and normals from each link's mesh
2. Filter points based on normal direction (palm-facing)
3. Allow manual adjustment via GUI
4. Export to XML format compatible with palmar_surface_points.xml

Usage:
    python palmar_surface_collector.py
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
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from xml.dom.minidom import Document
import xml.etree.ElementTree as ET

# Add project path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytorch_kinematics as pk
import torch
import urdf_parser_py.urdf as URDF_PARSER


class PalmarSurfaceCollector:
    """GUI tool for collecting palmar surface points"""
    
    MENU_SAVE = 1
    MENU_EXPORT_XML = 2
    MENU_QUIT = 3
    MENU_SAMPLE_ALL = 11
    MENU_CLEAR_LINK = 12
    MENU_HELP = 21
    
    def __init__(self, urdf_path: str, output_xml_path: str):
        self.urdf_path = urdf_path
        self.output_xml_path = output_xml_path
        
        # Load URDF
        self.robot_urdf = URDF_PARSER.URDF.from_xml_file(urdf_path)
        self.robot_pk = pk.build_chain_from_urdf(open(urdf_path).read())
        
        # Get all link names
        self.link_names = [link.name for link in self.robot_urdf.links]
        
        # Get revolute joints
        self.revolute_joints = [j for j in self.robot_urdf.joints if j.joint_type == 'revolute']
        self.joint_names = [j.name for j in self.revolute_joints]
        self.joint_limits = [(j.limit.lower, j.limit.upper) for j in self.revolute_joints]
        self.n_dofs = len(self.revolute_joints)
        
        # Current joint angles
        self.joint_angles = np.zeros(self.n_dofs)
        
        # Mesh data for each link (local coordinates)
        self.link_meshes: Dict[str, tm.Trimesh] = {}
        self.link_surface_points: Dict[str, np.ndarray] = {}  # (N, 3)
        self.link_surface_normals: Dict[str, np.ndarray] = {}  # (N, 3)
        
        # Selected palmar points for each link (local coordinates)
        self.palmar_points: Dict[str, List[np.ndarray]] = {ln: [] for ln in self.link_names}
        self.palmar_normals: Dict[str, List[np.ndarray]] = {ln: [] for ln in self.link_names}
        
        # Dense sampled points (before FPS) - cached for re-sampling
        self._dense_points: Dict[str, np.ndarray] = {}
        self._dense_normals: Dict[str, np.ndarray] = {}
        
        # Current selection state
        self.selected_link = None
        self.selected_point_idx = -1
        
        # Parameters for automatic selection
        self.palmar_direction = np.array([0.0, 0.0, -1.0])  # Default: -Z is palm facing
        self.angle_threshold = 60.0  # degrees
        self.point_spacing = 0.005  # target spacing between points (5mm)
        self.samples_per_face = 10  # samples per selected face for dense sampling
        
        # Face data for each link (local coordinates)
        self.link_face_centers: Dict[str, np.ndarray] = {}  # (N_faces, 3)
        self.link_face_normals: Dict[str, np.ndarray] = {}  # (N_faces, 3)
        self.link_face_areas: Dict[str, np.ndarray] = {}  # (N_faces,)
        
        # GUI components
        self.window = None
        self.scene_widget = None
        self.link_dropdown = None
        self.point_list = None
        self.angle_slider = None
        self.dir_sliders = {}
        self.joint_sliders = {}
        
        # Materials
        self.mesh_material = None
        self.point_material_normal = None
        self.point_material_selected = None
        self.point_material_palmar = None
        
        # Load meshes
        self._load_all_meshes()
    
    def _load_all_meshes(self):
        """Load mesh for each link"""
        for link in self.robot_urdf.links:
            mesh = self._load_link_mesh(link.name)
            if mesh is not None:
                self.link_meshes[link.name] = mesh
                # Store face information (centers, normals, areas)
                if len(mesh.faces) > 0:
                    # Compute face centers
                    face_centers = mesh.vertices[mesh.faces].mean(axis=1)  # (N_faces, 3)
                    face_normals = mesh.face_normals  # (N_faces, 3)
                    face_areas = mesh.area_faces  # (N_faces,)
                    
                    self.link_face_centers[link.name] = face_centers
                    self.link_face_normals[link.name] = face_normals
                    self.link_face_areas[link.name] = face_areas
                    
                    # Also keep a sparse preview sample for visualization
                    pts, face_idx = tm.sample.sample_surface(mesh, 2000)
                    normals = mesh.face_normals[face_idx]
                    self.link_surface_points[link.name] = pts
                    self.link_surface_normals[link.name] = normals
                    
                    print(f"[INFO] Loaded {link.name}: {len(mesh.faces)} faces")
    
    def _load_link_mesh(self, link_name: str) -> Optional[tm.Trimesh]:
        """Load mesh for a single link in local coordinates"""
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
        
        return tm.util.concatenate(meshes)
    
    def _get_link_transform(self, link_name: str) -> np.ndarray:
        """Get link transform matrix"""
        q = torch.tensor(self.joint_angles, dtype=torch.float).unsqueeze(0)
        fk = self.robot_pk.forward_kinematics(q)
        
        if link_name in fk:
            return fk[link_name].get_matrix()[0].numpy()
        return np.eye(4)
    
    def _auto_select_palmar_points(self, link_name: str):
        """Automatically select palmar points based on normal direction
        
        New algorithm:
        1. Filter faces by normal direction
        2. Densely sample points on selected faces (weighted by area)
        3. FPS to ensure uniform spacing (~0.005m)
        """
        if link_name not in self.link_face_centers:
            return
        
        mesh = self.link_meshes[link_name]
        face_centers = self.link_face_centers[link_name]
        face_normals = self.link_face_normals[link_name]
        face_areas = self.link_face_areas[link_name]
        
        # Step 1: Filter faces by normal direction
        palmar_dir = self.palmar_direction / (np.linalg.norm(self.palmar_direction) + 1e-8)
        cos_angles = np.dot(face_normals, palmar_dir)
        angles = np.arccos(np.clip(cos_angles, -1, 1)) * 180 / np.pi
        
        # Get indices of palmar-facing faces
        palmar_face_mask = angles < self.angle_threshold
        palmar_face_indices = np.where(palmar_face_mask)[0]
        
        if len(palmar_face_indices) == 0:
            print(f"[WARN] No palmar-facing faces found for {link_name}")
            self.palmar_points[link_name] = []
            self.palmar_normals[link_name] = []
            return
        
        print(f"[INFO] {link_name}: {len(palmar_face_indices)}/{len(face_normals)} faces are palmar-facing")
        
        # Step 2: Create submesh with only palmar faces and sample densely
        palmar_faces = mesh.faces[palmar_face_indices]
        palmar_mesh = tm.Trimesh(vertices=mesh.vertices, faces=palmar_faces, process=False)
        
        # Estimate number of points needed based on area and spacing
        total_area = palmar_mesh.area
        # Approximate: each point covers a circle of radius = spacing/2
        points_per_unit_area = 1.0 / (np.pi * (self.point_spacing / 2) ** 2)
        estimated_points = int(total_area * points_per_unit_area)
        
        # Sample at least 10x more points than needed for FPS
        n_samples = max(estimated_points * 10, 1000)
        n_samples = min(n_samples, 50000)  # Cap to avoid memory issues
        
        sampled_pts, face_idx = tm.sample.sample_surface(palmar_mesh, n_samples)
        sampled_normals = palmar_mesh.face_normals[face_idx]
        
        print(f"[INFO] {link_name}: Sampled {len(sampled_pts)} points on palmar faces (area={total_area:.6f}m²)")
        
        # Cache dense samples for re-sampling with different spacing
        self._dense_points[link_name] = sampled_pts
        self._dense_normals[link_name] = sampled_normals
        
        # Store for FPS (will be done in _subsample_palmar_points)
        self.palmar_points[link_name] = [sampled_pts[i] for i in range(len(sampled_pts))]
        self.palmar_normals[link_name] = [sampled_normals[i] for i in range(len(sampled_normals))]
    
    def _subsample_palmar_points(self, link_name: str, target_spacing: float = None):
        """Subsample palmar points using FPS based on target spacing
        
        Args:
            link_name: Name of the link
            target_spacing: Target spacing between points (default: self.point_spacing)
        """
        if link_name not in self.palmar_points or len(self.palmar_points[link_name]) == 0:
            return
        
        if target_spacing is None:
            target_spacing = self.point_spacing
        
        points = np.array(self.palmar_points[link_name])
        normals = np.array(self.palmar_normals[link_name])
        
        if len(points) <= 1:
            return
        
        # Farthest Point Sampling with distance threshold
        selected_indices = [0]
        distances = np.full(len(points), np.inf)
        
        while True:
            last_idx = selected_indices[-1]
            dist_to_last = np.linalg.norm(points - points[last_idx], axis=1)
            distances = np.minimum(distances, dist_to_last)
            
            # Find farthest point
            max_dist = np.max(distances)
            
            # Stop if max distance is less than target spacing
            # (all remaining points are already within spacing of selected points)
            if max_dist < target_spacing:
                break
            
            next_idx = np.argmax(distances)
            selected_indices.append(next_idx)
            
            # Safety: limit max points
            if len(selected_indices) >= 5000:
                print(f"[WARN] Reached max 5000 points for {link_name}")
                break
        
        self.palmar_points[link_name] = [points[i] for i in selected_indices]
        self.palmar_normals[link_name] = [normals[i] for i in selected_indices]
        
        print(f"[INFO] {link_name}: FPS selected {len(selected_indices)} points (spacing={target_spacing}m)")
    
    def _update_scene(self):
        """Update 3D scene"""
        if self.scene_widget is None:
            return
        
        scene = self.scene_widget.scene
        scene.clear_geometry()
        
        # Add world frame
        scene.add_geometry("world_frame", 
                          o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03),
                          rendering.MaterialRecord())
        
        # Add hand meshes
        for link_name, mesh in self.link_meshes.items():
            transform = self._get_link_transform(link_name)
            
            # Create Open3D mesh
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
            o3d_mesh.compute_vertex_normals()
            o3d_mesh.transform(transform)
            
            # Use special color for selected link
            if link_name == self.selected_link:
                mat = rendering.MaterialRecord()
                mat.shader = "defaultLit"
                mat.base_color = [1.0, 0.6, 0.2, 0.9]  # Orange for selected
                scene.add_geometry(f"mesh_{link_name}", o3d_mesh, mat)
            else:
                scene.add_geometry(f"mesh_{link_name}", o3d_mesh, self.mesh_material)
        
        # Add all surface points for selected link (small gray points)
        if self.selected_link and self.selected_link in self.link_surface_points:
            transform = self._get_link_transform(self.selected_link)
            pts_local = self.link_surface_points[self.selected_link]
            pts_world = (transform[:3, :3] @ pts_local.T).T + transform[:3, 3]
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_world)
            pcd.paint_uniform_color([0.5, 0.5, 0.5])
            
            mat = rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            mat.point_size = 3.0
            scene.add_geometry("surface_points", pcd, mat)
        
        # Add palmar points for all links
        for link_name, pts_list in self.palmar_points.items():
            if not pts_list:
                continue
            
            transform = self._get_link_transform(link_name)
            pts_local = np.array(pts_list)
            pts_world = (transform[:3, :3] @ pts_local.T).T + transform[:3, 3]
            
            # Different color for selected link
            is_selected = (link_name == self.selected_link)
            color = [0.0, 1.0, 0.3] if is_selected else [0.2, 0.6, 1.0]
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_world)
            pcd.paint_uniform_color(color)
            
            mat = rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            mat.point_size = 8.0 if is_selected else 5.0
            scene.add_geometry(f"palmar_{link_name}", pcd, mat)
        
        # Highlight selected point
        if (self.selected_link and 
            self.selected_point_idx >= 0 and 
            self.selected_point_idx < len(self.palmar_points.get(self.selected_link, []))):
            
            transform = self._get_link_transform(self.selected_link)
            pt_local = self.palmar_points[self.selected_link][self.selected_point_idx]
            pt_world = transform[:3, :3] @ pt_local + transform[:3, 3]
            
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
            sphere.translate(pt_world)
            sphere.paint_uniform_color([1.0, 0.0, 0.0])
            sphere.compute_vertex_normals()
            
            scene.add_geometry("selected_point", sphere, rendering.MaterialRecord())
        
        # Add large coordinate frame at selected link's origin
        if self.selected_link:
            transform = self._get_link_transform(self.selected_link)
            link_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
            link_frame.transform(transform)
            scene.add_geometry("selected_link_frame", link_frame, rendering.MaterialRecord())
    
    def _update_point_list(self):
        """Update point list display"""
        if self.point_list is None or self.selected_link is None:
            return
        
        pts = self.palmar_points.get(self.selected_link, [])
        items = [f"[{i}] ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})" for i, p in enumerate(pts)]
        self.point_list.set_items(items)
        
        if self.selected_point_idx >= 0 and self.selected_point_idx < len(pts):
            self.point_list.selected_index = self.selected_point_idx
    
    def _on_link_changed(self, name: str, idx: int):
        """Link dropdown changed"""
        self.selected_link = name
        self.selected_point_idx = 0 if self.palmar_points.get(name, []) else -1
        self._update_point_list()
        self._update_scene()
    
    def _on_point_selected(self, name: str, is_double: bool):
        """Point list selection changed"""
        if self.point_list:
            self.selected_point_idx = self.point_list.selected_index
            self._update_scene()
    
    def _on_joint_changed(self, idx: int, val: float):
        """Joint slider changed"""
        self.joint_angles[idx] = val
        self._update_scene()
    
    def _on_angle_threshold_changed(self, val: float):
        """Angle threshold slider changed"""
        self.angle_threshold = val
    
    def _on_spacing_changed(self, val: float):
        """Point spacing slider changed (val is in mm)"""
        self.point_spacing = val / 1000.0  # Convert to meters
    
    def _on_palmar_dir_changed(self, axis: str, val: float):
        """Palmar direction changed"""
        idx = {'x': 0, 'y': 1, 'z': 2}[axis]
        self.palmar_direction[idx] = val
    
    def _on_auto_select(self):
        """Auto select palmar points for current link"""
        if self.selected_link:
            self._auto_select_palmar_points(self.selected_link)
            self._subsample_palmar_points(self.selected_link)
            self._update_point_list()
            self._update_scene()
    
    def _on_auto_select_all(self):
        """Auto select palmar points for all links"""
        total_points = 0
        for link_name in self.link_meshes.keys():
            self._auto_select_palmar_points(link_name)
            self._subsample_palmar_points(link_name)
            total_points += len(self.palmar_points.get(link_name, []))
        self._update_point_list()
        self._update_scene()
        print(f"[INFO] Auto-selected palmar points for all links, total: {total_points} points")
    
    def _on_resample_fps(self):
        """Re-apply FPS with current spacing to all links that have cached dense samples"""
        total_points = 0
        resampled_count = 0
        
        for link_name in self.link_meshes.keys():
            if link_name in self._dense_points and len(self._dense_points[link_name]) > 0:
                # Restore dense samples
                pts = self._dense_points[link_name]
                normals = self._dense_normals[link_name]
                self.palmar_points[link_name] = [pts[i] for i in range(len(pts))]
                self.palmar_normals[link_name] = [normals[i] for i in range(len(normals))]
                
                # Re-apply FPS
                self._subsample_palmar_points(link_name)
                total_points += len(self.palmar_points.get(link_name, []))
                resampled_count += 1
        
        self._update_point_list()
        self._update_scene()
        print(f"[INFO] Re-sampled {resampled_count} links with spacing={self.point_spacing*1000:.1f}mm, total: {total_points} points")
    
    def _on_clear_link(self):
        """Clear palmar points for current link"""
        if self.selected_link:
            self.palmar_points[self.selected_link] = []
            self.palmar_normals[self.selected_link] = []
            self.selected_point_idx = -1
            self._update_point_list()
            self._update_scene()
    
    def _on_delete_point(self):
        """Delete selected point"""
        if (self.selected_link and 
            self.selected_point_idx >= 0 and 
            self.selected_point_idx < len(self.palmar_points.get(self.selected_link, []))):
            
            self.palmar_points[self.selected_link].pop(self.selected_point_idx)
            self.palmar_normals[self.selected_link].pop(self.selected_point_idx)
            
            if self.selected_point_idx >= len(self.palmar_points[self.selected_link]):
                self.selected_point_idx = len(self.palmar_points[self.selected_link]) - 1
            
            self._update_point_list()
            self._update_scene()
    
    def _on_add_point(self):
        """Add a point manually (at origin of link)"""
        if self.selected_link:
            self.palmar_points[self.selected_link].append(np.array([0.0, 0.0, 0.0]))
            self.palmar_normals[self.selected_link].append(self.palmar_direction.copy())
            self.selected_point_idx = len(self.palmar_points[self.selected_link]) - 1
            self._update_point_list()
            self._update_scene()
    
    def _export_xml(self):
        """Export palmar points to XML file"""
        doc = Document()
        root = doc.createElement('Robot')
        doc.appendChild(root)
        
        # Add comment
        comment = doc.createComment(' Palmar surface points for LEAP Hand ')
        root.appendChild(comment)
        
        for link_name, pts_list in self.palmar_points.items():
            if not pts_list:
                continue
            
            normals_list = self.palmar_normals.get(link_name, [])
            
            link_data = doc.createElement('PointCloudLinkData')
            root.appendChild(link_data)
            
            # Link name
            name_elem = doc.createElement('linkName')
            name_elem.appendChild(doc.createTextNode(link_name))
            link_data.appendChild(name_elem)
            
            # Points
            points_elem = doc.createElement('points')
            for pt in pts_list:
                vec = doc.createElement('Vector3')
                for axis, val in zip(['x', 'y', 'z'], pt):
                    axis_elem = doc.createElement(axis)
                    axis_elem.appendChild(doc.createTextNode(f'{val:.6f}'))
                    vec.appendChild(axis_elem)
                points_elem.appendChild(vec)
            link_data.appendChild(points_elem)
            
            # Normals
            normals_elem = doc.createElement('normal')
            for i, pt in enumerate(pts_list):
                vec = doc.createElement('Vector3')
                normal = normals_list[i] if i < len(normals_list) else self.palmar_direction
                for axis, val in zip(['x', 'y', 'z'], normal):
                    axis_elem = doc.createElement(axis)
                    axis_elem.appendChild(doc.createTextNode(f'{val:.6f}'))
                    vec.appendChild(axis_elem)
                normals_elem.appendChild(vec)
            link_data.appendChild(normals_elem)
        
        # Write to file
        os.makedirs(os.path.dirname(self.output_xml_path), exist_ok=True)
        with open(self.output_xml_path, 'w') as f:
            f.write(doc.toprettyxml(indent='  '))
        
        print(f"[INFO] Exported palmar points to {self.output_xml_path}")
        
        # Show summary
        total = sum(len(pts) for pts in self.palmar_points.values())
        print(f"[INFO] Total points: {total}")
        for link_name, pts in self.palmar_points.items():
            if pts:
                print(f"  {link_name}: {len(pts)} points")
    
    def _show_help(self):
        """Show help dialog"""
        em = self.window.theme.font_size
        dlg = gui.Dialog("Help")
        
        layout = gui.Vert(0, gui.Margins(em, em, em, em))
        layout.add_child(gui.Label("Palmar Surface Point Collector"))
        layout.add_child(gui.Label(""))
        layout.add_child(gui.Label("Algorithm:"))
        layout.add_child(gui.Label("1. Set palmar direction (normal facing palm)"))
        layout.add_child(gui.Label("2. Set angle threshold for auto-selection"))
        layout.add_child(gui.Label("3. Click 'Auto Select' to filter points"))
        layout.add_child(gui.Label("4. Manually add/delete points as needed"))
        layout.add_child(gui.Label("5. Export to XML when done"))
        layout.add_child(gui.Label(""))
        layout.add_child(gui.Label("Tips:"))
        layout.add_child(gui.Label("- Green points = current link's palmar points"))
        layout.add_child(gui.Label("- Blue points = other links' palmar points"))
        layout.add_child(gui.Label("- Gray points = all surface samples"))
        layout.add_child(gui.Label("- Red sphere = selected point"))
        
        ok_btn = gui.Button("OK")
        ok_btn.set_on_clicked(lambda: self.window.close_dialog())
        layout.add_child(ok_btn)
        
        dlg.add_child(layout)
        self.window.show_dialog(dlg)
    
    def run(self):
        """Run the GUI application"""
        app = gui.Application.instance
        app.initialize()
        
        self.window = app.create_window("Palmar Surface Collector", 1600, 900)
        em = self.window.theme.font_size
        
        # Materials
        self.mesh_material = rendering.MaterialRecord()
        self.mesh_material.shader = "defaultLit"
        self.mesh_material.base_color = [0.7, 0.7, 0.7, 0.6]
        
        # Scene
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.1, 0.1, 0.15, 1.0])
        self.scene_widget.scene.scene.set_sun_light([0.577, -0.577, -0.577], [1.0, 1.0, 1.0], 100000)
        self.scene_widget.scene.scene.enable_sun_light(True)
        
        bounds = o3d.geometry.AxisAlignedBoundingBox([-0.15, -0.15, -0.15], [0.15, 0.15, 0.15])
        self.scene_widget.setup_camera(60, bounds, [0, 0, 0])
        self.scene_widget.look_at([0, 0, 0], [0.3, 0.3, 0.3], [0, 1, 0])
        
        # Left panel
        panel = gui.Vert(0, gui.Margins(0.5*em, 0.5*em, 0.5*em, 0.5*em))
        
        # --- Link Selection ---
        panel.add_child(gui.Label("Select Link:"))
        self.link_dropdown = gui.Combobox()
        for ln in self.link_names:
            self.link_dropdown.add_item(ln)
        self.link_dropdown.selected_index = 0
        self.selected_link = self.link_names[0]
        self.link_dropdown.set_on_selection_changed(self._on_link_changed)
        panel.add_child(self.link_dropdown)
        
        panel.add_fixed(0.5 * em)
        
        # --- Palmar Direction ---
        panel.add_child(gui.Label("Palmar Direction:"))
        for axis in ['x', 'y', 'z']:
            row = gui.Horiz(0.5 * em)
            row.add_child(gui.Label(f"{axis.upper()}:"))
            slider = gui.Slider(gui.Slider.DOUBLE)
            slider.set_limits(-1.0, 1.0)
            slider.double_value = self.palmar_direction[{'x': 0, 'y': 1, 'z': 2}[axis]]
            slider.set_on_value_changed(lambda v, a=axis: self._on_palmar_dir_changed(a, v))
            self.dir_sliders[axis] = slider
            row.add_child(slider)
            panel.add_child(row)
        
        # --- Angle Threshold ---
        panel.add_child(gui.Label("Angle Threshold (deg):"))
        self.angle_slider = gui.Slider(gui.Slider.DOUBLE)
        self.angle_slider.set_limits(10.0, 90.0)
        self.angle_slider.double_value = self.angle_threshold
        self.angle_slider.set_on_value_changed(self._on_angle_threshold_changed)
        panel.add_child(self.angle_slider)
        
        # --- Point Spacing ---
        panel.add_child(gui.Label("Point Spacing (mm):"))
        self.spacing_slider = gui.Slider(gui.Slider.DOUBLE)
        self.spacing_slider.set_limits(1.0, 20.0)  # 1mm to 20mm
        self.spacing_slider.double_value = self.point_spacing * 1000  # Convert to mm
        self.spacing_slider.set_on_value_changed(self._on_spacing_changed)
        panel.add_child(self.spacing_slider)
        
        # --- Auto Select Buttons ---
        btn_row = gui.Horiz(0.5 * em)
        auto_btn = gui.Button("Auto Select")
        auto_btn.set_on_clicked(self._on_auto_select)
        auto_all_btn = gui.Button("Select All Links")
        auto_all_btn.set_on_clicked(self._on_auto_select_all)
        btn_row.add_child(auto_btn)
        btn_row.add_child(auto_all_btn)
        panel.add_child(btn_row)
        
        # Re-sample button (use new spacing on existing selection)
        resample_btn = gui.Button("Re-sample FPS (apply new spacing)")
        resample_btn.set_on_clicked(self._on_resample_fps)
        panel.add_child(resample_btn)
        
        panel.add_fixed(em)
        panel.add_child(gui.Label("="*30))
        panel.add_fixed(0.5 * em)
        
        # --- Point List ---
        panel.add_child(gui.Label("Palmar Points:"))
        self.point_list = gui.ListView()
        self.point_list.set_max_visible_items(8)
        self.point_list.set_on_selection_changed(self._on_point_selected)
        panel.add_child(self.point_list)
        
        # Point buttons
        pt_btn_row = gui.Horiz(0.5 * em)
        add_btn = gui.Button("Add")
        add_btn.set_on_clicked(self._on_add_point)
        del_btn = gui.Button("Delete")
        del_btn.set_on_clicked(self._on_delete_point)
        clear_btn = gui.Button("Clear Link")
        clear_btn.set_on_clicked(self._on_clear_link)
        pt_btn_row.add_child(add_btn)
        pt_btn_row.add_child(del_btn)
        pt_btn_row.add_child(clear_btn)
        panel.add_child(pt_btn_row)
        
        panel.add_fixed(em)
        panel.add_child(gui.Label("="*30))
        panel.add_fixed(0.5 * em)
        
        # --- Joint Sliders ---
        panel.add_child(gui.Label("Joint Angles (rad):"))
        joint_scroll = gui.ScrollableVert(0, gui.Margins(0, 0, 0, 0))
        
        for i, (name, limits) in enumerate(zip(self.joint_names, self.joint_limits)):
            row = gui.Horiz(0.25 * em)
            short_name = name[:12] + ".." if len(name) > 14 else name
            row.add_child(gui.Label(f"{short_name}:"))
            slider = gui.Slider(gui.Slider.DOUBLE)
            slider.set_limits(limits[0], limits[1])
            slider.double_value = 0.0
            slider.set_on_value_changed(lambda v, idx=i: self._on_joint_changed(idx, v))
            self.joint_sliders[i] = slider
            row.add_child(slider)
            joint_scroll.add_child(row)
        
        panel.add_child(joint_scroll)
        
        panel.add_fixed(em)
        
        # --- Export Button ---
        export_btn = gui.Button("Export to XML")
        export_btn.set_on_clicked(self._export_xml)
        panel.add_child(export_btn)
        
        # --- Menu ---
        if gui.Application.instance.menubar is None:
            menu = gui.Menu()
            file_menu = gui.Menu()
            file_menu.add_item("Export XML", self.MENU_EXPORT_XML)
            file_menu.add_separator()
            file_menu.add_item("Quit", self.MENU_QUIT)
            menu.add_menu("File", file_menu)
            
            edit_menu = gui.Menu()
            edit_menu.add_item("Auto Select All", self.MENU_SAMPLE_ALL)
            edit_menu.add_item("Clear Current Link", self.MENU_CLEAR_LINK)
            menu.add_menu("Edit", edit_menu)
            
            help_menu = gui.Menu()
            help_menu.add_item("Help", self.MENU_HELP)
            menu.add_menu("Help", help_menu)
            
            gui.Application.instance.menubar = menu
        
        self.window.set_on_menu_item_activated(self.MENU_EXPORT_XML, self._export_xml)
        self.window.set_on_menu_item_activated(self.MENU_QUIT, lambda: gui.Application.instance.quit())
        self.window.set_on_menu_item_activated(self.MENU_SAMPLE_ALL, self._on_auto_select_all)
        self.window.set_on_menu_item_activated(self.MENU_CLEAR_LINK, self._on_clear_link)
        self.window.set_on_menu_item_activated(self.MENU_HELP, self._show_help)
        
        # Layout
        self.window.add_child(self.scene_widget)
        self.window.add_child(panel)
        
        def on_layout(ctx):
            rect = self.window.content_rect
            panel_width = 22 * em
            self.scene_widget.frame = gui.Rect(panel_width, rect.y, rect.width - panel_width, rect.height)
            panel.frame = gui.Rect(rect.x, rect.y, panel_width, rect.height)
        
        self.window.set_on_layout(on_layout)
        
        # Initial update
        self._update_point_list()
        self._update_scene()
        
        app.run()


def main():
    urdf_path = str(PROJECT_ROOT / "robot_models/urdf/leap_hand_simplified.urdf")
    output_xml = str(PROJECT_ROOT / "robot_models/meta/leap_hand/palmar_surface_points_new.xml")
    
    if not os.path.exists(urdf_path):
        print(f"[ERROR] URDF not found: {urdf_path}")
        sys.exit(1)
    
    print(f"[INFO] URDF: {urdf_path}")
    print(f"[INFO] Output: {output_xml}")
    
    collector = PalmarSurfaceCollector(urdf_path, output_xml)
    collector.run()


if __name__ == "__main__":
    main()
