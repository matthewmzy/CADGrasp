"""
IBS Visualization with Open3D.

Visualize IBS voxels (occupancy, contact, thumb_contact) with different colors,
along with scene mesh, point cloud, hand mesh, and coordinate frames.

All elements are transformed to the TABLE coordinate system (world frame).

Color Legend:
- Blue: IBS Occupancy (non-contact)
- Green: Contact region  
- Red: Thumb contact region
- Grey/colored: Scene point cloud (segmented by object)
- Colored meshes: Scene objects
- Orange: Hand mesh

Coordinate Frames:
- Large frame: World origin (table frame)
- Small frame: Hand frame (grasp pose)

Usage:
    python tests/vis_ibs_o3d.py --scene_id 0 --grasp_idx 0
    python tests/vis_ibs_o3d.py --scene_id 0 --grasp_idx 0 --with_mesh --with_hand
    conda run -n cad python tests/vis_ibs_o3d.py --scene_id 0 --grasp_idx 0 --with_mesh --with_hand
    
"""

import argparse
import os
import numpy as np
import open3d as o3d
import torch
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
from typing import Optional, List, Tuple, Dict

from cadgrasp.paths import project_path
from cadgrasp.ibs.utils.ibs_repr import IBS, IBSConfig
from cadgrasp.baseline.utils.robot_model import RobotModel


def load_ibs_data(scene_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load IBS voxel data and w2h transforms for a scene."""
    scene_name = f'scene_{str(scene_id).zfill(4)}'
    ibs_path = project_path('data/ibsdata')
    
    ibs_file = os.path.join(ibs_path, 'ibs', f'{scene_name}.npy')
    w2h_file = os.path.join(ibs_path, 'w2h_trans', f'{scene_name}.npy')
    
    ibs_voxels = np.load(ibs_file)  # (N, 40, 40, 40, 3)
    w2h_trans = np.load(w2h_file)   # (N, 4, 4)
    
    print(f"Loaded IBS data: {ibs_voxels.shape[0]} grasps")
    return ibs_voxels, w2h_trans


def load_scene_extrinsics(scene_id: int, camera: str = 'realsense') -> np.ndarray:
    """Load cam0_wrt_table (world to table transform)."""
    scene_name = f'scene_{str(scene_id).zfill(4)}'
    extrinsics_path = project_path(
        'data/DexGraspNet2.0/scenes', scene_name, camera, 'cam0_wrt_table.npy'
    )
    return np.load(extrinsics_path)  # (4, 4)


def load_scene_pc(scene_id: int, view: int = 0, camera: str = 'realsense') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load scene point cloud from network_input.npz.
    
    Returns:
        pc: (N, 3) points in camera frame
        seg: (N,) segmentation labels
        extrinsics: (4, 4) camera-to-table transform
    """
    scene_name = f'scene_{str(scene_id).zfill(4)}'
    pc_file = project_path('data/DexGraspNet2.0/scenes', scene_name, camera, 'network_input.npz')
    
    data = np.load(pc_file)
    pc = data['pc'][view]           # (40000, 3) in camera frame
    seg = data['seg'][view]         # (40000,)
    extrinsics = data['extrinsics'][view]  # (4, 4) cam2table
    
    return pc, seg, extrinsics


def load_grasp_data_for_scene(scene_id: int) -> Dict[str, np.ndarray]:
    """Load grasp data using FPS indices."""
    from cadgrasp.ibs.scripts.load_fps_grasps import load_fps_grasps_for_scene
    
    scene_name = f'scene_{str(scene_id).zfill(4)}'
    grasp_data, _ = load_fps_grasps_for_scene(
        scene_name, 'leap_hand',
        grasp_base_path=project_path('data/DexGraspNet2.0/dex_grasps_new'),
        fps_base_path=project_path('data/DexGraspNet2.0/fps_sampled_indices'),
        return_object_ids=True
    )
    return grasp_data


def load_scene_meshes(scene_id: int, camera: str = 'realsense', view: int = 0) -> List[o3d.geometry.TriangleMesh]:
    """
    Load scene object meshes with their poses in TABLE coordinate system.
    
    The annotation XML contains poses in WORLD coordinate system.
    We need to apply extrinsics (cam0_wrt_table) to transform to TABLE coords.
    """
    scene_name = f'scene_{str(scene_id).zfill(4)}'
    annotation_file = project_path(
        'data/DexGraspNet2.0/scenes', scene_name, camera, 
        'annotations', f'{str(view).zfill(4)}.xml'
    )
    
    # Load extrinsics (world to table transform)
    extrinsics = load_scene_extrinsics(scene_id, camera)
    R_w2t = extrinsics[:3, :3]
    t_w2t = extrinsics[:3, 3]
    
    # Parse annotation XML
    tree = ET.parse(annotation_file)
    root = tree.getroot()
    
    meshes = []
    for obj in root.findall('obj'):
        obj_id = int(obj.find('obj_id').text)
        pos_str = obj.find('pos_in_world').text
        ori_str = obj.find('ori_in_world').text
        
        pos_world = np.array([float(x) for x in pos_str.split()])
        ori_world = np.array([float(x) for x in ori_str.split()])  # quaternion (w, x, y, z)
        
        # Load mesh
        mesh_path = project_path(
            'data/DexGraspNet2.0/meshdata', 
            str(obj_id).zfill(3), 
            'nontextured.ply'
        )
        
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
        
        # Object pose in world frame
        rot_world = R.from_quat([ori_world[1], ori_world[2], ori_world[3], ori_world[0]])  # scipy uses (x, y, z, w)
        rot_matrix_world = rot_world.as_matrix()
        
        # Transform object from world to table frame
        # pos_table = R_w2t @ pos_world + t_w2t
        # rot_table = R_w2t @ rot_world
        pos_table = R_w2t @ pos_world + t_w2t
        rot_matrix_table = R_w2t @ rot_matrix_world
        
        transform = np.eye(4)
        transform[:3, :3] = rot_matrix_table
        transform[:3, 3] = pos_table
        mesh.transform(transform)
        
        # Assign random color
        np.random.seed(obj_id)
        color = np.random.rand(3) * 0.5 + 0.3  # Muted colors
        mesh.paint_uniform_color(color)
        
        meshes.append(mesh)
    
    print(f"Loaded {len(meshes)} object meshes (in table frame)")
    return meshes


def get_hand_mesh(
    robot_model: RobotModel,
    translation: np.ndarray,  # (3,) in table frame
    rotation: np.ndarray,     # (3, 3) in table frame
    qpos_dict: Dict[str, torch.Tensor]  # joint angles
) -> o3d.geometry.TriangleMesh:
    """
    Get hand mesh in table coordinate system.
    """
    # Compute forward kinematics
    link_trans, link_rots = robot_model.forward_kinematics(qpos_dict)
    
    # Apply global transform (translation and rotation)
    trans_tensor = torch.from_numpy(translation).float().unsqueeze(0)
    rot_tensor = torch.from_numpy(rotation).float().unsqueeze(0)
    
    link_trans = {k: torch.einsum('nab,nb->na', rot_tensor, v) + trans_tensor 
                  for k, v in link_trans.items()}
    link_rots = {k: torch.einsum('nab,nbc->nac', rot_tensor, v) 
                 for k, v in link_rots.items()}
    
    # Create combined mesh
    combined_mesh = o3d.geometry.TriangleMesh()
    
    for link_name in robot_model._geometry:
        if 'visual_vertices' not in robot_model._geometry[link_name]:
            continue
            
        vertices = robot_model._geometry[link_name]['visual_vertices']
        faces = robot_model._geometry[link_name]['visual_faces']
        
        if len(vertices) == 0:
            continue
        
        # Get link transform
        link_t = link_trans[link_name][0].numpy()
        link_r = link_rots[link_name][0].numpy()
        
        # Transform vertices
        vertices_transformed = (link_r @ vertices.T).T + link_t
        
        # Create mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices_transformed)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        combined_mesh += mesh
    
    combined_mesh.compute_vertex_normals()
    combined_mesh.paint_uniform_color([1.0, 0.6, 0.2])  # Orange
    
    return combined_mesh


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply 4x4 transformation to points."""
    if points.shape[0] == 0:
        return points
    points_hom = np.hstack([points, np.ones((points.shape[0], 1))])
    return (transform @ points_hom.T).T[:, :3]


def create_coordinate_frame(transform: np.ndarray, size: float = 0.05) -> o3d.geometry.TriangleMesh:
    """Create a coordinate frame at the given transform."""
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(transform)
    return frame


def create_table_mesh(size: float = 0.6, height: float = -0.01) -> o3d.geometry.TriangleMesh:
    """Create a table mesh as a flat box."""
    table = o3d.geometry.TriangleMesh.create_box(
        width=size, height=size, depth=0.01
    )
    table.translate([-size/2, -size/2, height])
    table.paint_uniform_color([0.6, 0.5, 0.4])  # Brown
    table.compute_vertex_normals()
    return table


def visualize(
    scene_id: int,
    grasp_idx: int,
    view: int = 0,
    camera: str = 'realsense',
    with_mesh: bool = False,
    with_pc: bool = True,
    with_hand: bool = False,
    with_table: bool = True,
    show_hand_frame: bool = True,
    show_world_frame: bool = True,
):
    """Main visualization function."""
    
    # Load IBS data
    ibs_voxels, w2h_trans_all = load_ibs_data(scene_id)
    
    if grasp_idx >= len(ibs_voxels):
        print(f"Error: grasp_idx {grasp_idx} out of range (max {len(ibs_voxels)-1})")
        return
    
    # Create IBS object using ibs_repr.py
    ibs = IBS.from_voxel(
        voxel=ibs_voxels[grasp_idx],
        w2h_trans=w2h_trans_all[grasp_idx]
    )
    
    # h2w_trans: hand frame to table/world frame
    h2w = np.linalg.inv(w2h_trans_all[grasp_idx])
    
    print(f"\nVisualizing scene {scene_id}, grasp {grasp_idx}")
    print(f"w2h_trans:\n{w2h_trans_all[grasp_idx]}")
    
    # Get point clouds from IBS using ibs_repr methods
    # All in hand frame initially
    occ_points_hand = ibs.get_non_contact_ibs_points().numpy()  # IBS without contact
    contact_points_hand = ibs.get_contact_points().numpy()
    thumb_points_hand = ibs.get_thumb_contact_points().numpy()
    
    print(f"IBS points (in hand frame):")
    print(f"  Occupancy (non-contact): {len(occ_points_hand)}")
    print(f"  Contact: {len(contact_points_hand)}")
    print(f"  Thumb contact: {len(thumb_points_hand)}")
    
    # Transform IBS points from hand frame to table frame
    occ_points_table = transform_points(occ_points_hand, h2w)
    contact_points_table = transform_points(contact_points_hand, h2w)
    thumb_points_table = transform_points(thumb_points_hand, h2w)
    
    # Create geometries list
    geometries = []
    
    # Create IBS point clouds (now in table frame)
    # Blue: Occupancy (non-contact)
    if len(occ_points_table) > 0:
        occ_pcd = o3d.geometry.PointCloud()
        occ_pcd.points = o3d.utility.Vector3dVector(occ_points_table)
        occ_pcd.paint_uniform_color([0.3, 0.3, 0.9])  # Blue
        geometries.append(occ_pcd)
    
    # Green: Contact
    if len(contact_points_table) > 0:
        contact_pcd = o3d.geometry.PointCloud()
        contact_pcd.points = o3d.utility.Vector3dVector(contact_points_table)
        contact_pcd.paint_uniform_color([0.2, 0.9, 0.2])  # Green
        geometries.append(contact_pcd)
    
    # Red: Thumb contact
    if len(thumb_points_table) > 0:
        thumb_pcd = o3d.geometry.PointCloud()
        thumb_pcd.points = o3d.utility.Vector3dVector(thumb_points_table)
        thumb_pcd.paint_uniform_color([0.9, 0.2, 0.2])  # Red
        geometries.append(thumb_pcd)
    
    # Add scene point cloud (transform from camera to table frame)
    if with_pc:
        try:
            pc_cam, seg, cam2table = load_scene_pc(scene_id, view, camera)
            # Transform from camera frame to table frame
            pc_table = transform_points(pc_cam, cam2table)
            
            scene_pcd = o3d.geometry.PointCloud()
            scene_pcd.points = o3d.utility.Vector3dVector(pc_table)
            
            # Color by segmentation
            unique_segs = np.unique(seg)
            colors = np.zeros((len(pc_table), 3))
            for s in unique_segs:
                mask = seg == s
                np.random.seed(int(s) + 1000)
                color = np.random.rand(3) * 0.5 + 0.3
                colors[mask] = color
            scene_pcd.colors = o3d.utility.Vector3dVector(colors)
            
            geometries.append(scene_pcd)
            print(f"Scene point cloud: {len(pc_table)} points (in table frame)")
        except Exception as e:
            print(f"Warning: Could not load scene point cloud: {e}")
    
    # Add scene meshes (already in table frame from load_scene_meshes)
    if with_mesh:
        try:
            meshes = load_scene_meshes(scene_id, camera, view)
            geometries.extend(meshes)
        except Exception as e:
            print(f"Warning: Could not load scene meshes: {e}")
    
    # Add hand mesh
    if with_hand:
        try:
            # Load grasp data
            grasp_data = load_grasp_data_for_scene(scene_id)
            
            # Load extrinsics for coordinate transform
            extrinsics = load_scene_extrinsics(scene_id, camera)
            R_w2t = extrinsics[:3, :3]
            t_w2t = extrinsics[:3, 3]
            
            # Get grasp parameters (in world frame originally)
            trans_world = grasp_data['translation'][grasp_idx]
            rot_world = grasp_data['rotation'][grasp_idx]
            
            # Transform to table frame
            trans_table = R_w2t @ trans_world + t_w2t
            rot_table = R_w2t @ rot_world
            
            # Get joint positions for this grasp
            qpos_dict = {}
            robot_model = RobotModel(
                urdf_path=project_path('robot_models/urdf/leap_hand_simplified.urdf'),
                meta_path=project_path('robot_models/meta/leap_hand/meta.yaml')
            )
            for joint_name in robot_model.joint_names:
                if joint_name in grasp_data:
                    qpos_dict[joint_name] = torch.from_numpy(
                        grasp_data[joint_name][grasp_idx:grasp_idx+1]
                    ).float()
            
            hand_mesh = get_hand_mesh(robot_model, trans_table, rot_table, qpos_dict)
            geometries.append(hand_mesh)
            print("Hand mesh added (in table frame)")
        except Exception as e:
            print(f"Warning: Could not load hand mesh: {e}")
            import traceback
            traceback.print_exc()
    
    # Add table
    if with_table:
        table = create_table_mesh()
        geometries.append(table)
    
    # Add coordinate frames
    if show_world_frame:
        world_frame = create_coordinate_frame(np.eye(4), size=0.1)
        geometries.append(world_frame)
    
    if show_hand_frame:
        # Hand frame in table coordinates
        hand_frame = create_coordinate_frame(h2w, size=0.05)
        geometries.append(hand_frame)
    
    # Print legend
    print("\n--- Visualization Legend ---")
    print("Blue points: IBS Occupancy (non-contact)")
    print("Green points: Contact region")
    print("Red points: Thumb contact region")
    print("Large coordinate frame (0.1m): World/Table origin")
    print("Small coordinate frame (0.05m): Hand frame")
    if with_pc:
        print("Colored points: Scene point cloud")
    if with_mesh:
        print("Colored meshes: Scene objects")
    if with_hand:
        print("Orange mesh: Hand")
    print("Brown box: Table surface")
    print("----------------------------\n")
    
    # Visualize
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"IBS Visualization - Scene {scene_id}, Grasp {grasp_idx}",
        width=1280,
        height=720,
        point_show_normal=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize IBS with Open3D")
    parser.add_argument('--scene_id', type=int, default=0, help='Scene ID')
    parser.add_argument('--grasp_idx', type=int, default=0, help='Grasp index')
    parser.add_argument('--view', type=int, default=0, help='Camera view index')
    parser.add_argument('--camera', type=str, default='realsense', help='Camera type')
    parser.add_argument('--with_mesh', action='store_true', help='Include scene meshes')
    parser.add_argument('--with_hand', action='store_true', help='Include hand mesh')
    parser.add_argument('--no_pc', action='store_true', help='Exclude point cloud')
    parser.add_argument('--no_table', action='store_true', help='Exclude table')
    parser.add_argument('--no_hand_frame', action='store_true', help='Hide hand coordinate frame')
    parser.add_argument('--no_world_frame', action='store_true', help='Hide world coordinate frame')
    
    args = parser.parse_args()
    
    visualize(
        scene_id=args.scene_id,
        grasp_idx=args.grasp_idx,
        view=args.view,
        camera=args.camera,
        with_mesh=args.with_mesh,
        with_hand=args.with_hand,
        with_pc=not args.no_pc,
        with_table=not args.no_table,
        show_hand_frame=not args.no_hand_frame,
        show_world_frame=not args.no_world_frame,
    )


if __name__ == '__main__':
    main()
