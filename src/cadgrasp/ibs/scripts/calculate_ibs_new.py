"""
Calculate IBS (Interaction Bisector Surface) from FPS-sampled grasps.

This script computes IBS voxels for grasps that have been filtered by 
FPS sampling. It loads grasp data using fps_sampled_indices.

The IBS computation algorithm:
1. For each grasp, transform all points to hand coordinate system
2. For each voxel point, compute distance to scene and hand surface
3. Find points where |d_scene - d_hand| < delta (bisector surface)
4. Iteratively refine these points to lie exactly on the bisector
5. Mark contact regions where both distances are small
6. Mark thumb contact regions separately

Input:
    - data/DexGraspNet2.0/dex_grasps_new/scene_XXXX/leap_hand/XXX.npz
    - data/DexGraspNet2.0/fps_sampled_indices/scene_XXXX/leap_hand/XXX.npz

Output:
    - data/ibsdata/ibs/scene_XXXX.npy: (N, 40, 40, 40, 3) IBS voxels
    - data/ibsdata/w2h_trans/scene_XXXX.npy: (N, 4, 4) transformations
    - data/ibsdata/hand_dis/scene_XXXX.npy: (N, 40, 40, 40) hand distances

Usage:
    python calculate_ibs_new.py scene_id=100 device=cuda:0
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from termcolor import cprint

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cadgrasp.ibs.utils.transforms import batch_transform_points, transform_points
from cadgrasp.ibs.utils.ibs_repr import IBS, IBSBatch, IBSConfig
from cadgrasp.ibs.scripts.scene import Scene
from cadgrasp.ibs.scripts.load_fps_grasps import load_fps_grasps_for_scene
from cadgrasp.baseline.utils.robot_model import RobotModel


def points_query(pc: torch.Tensor, points: torch.Tensor):
    """
    Query nearest points and distances from a point cloud.
    
    Args:
        pc: (M, 3) reference point cloud
        points: (N, 3) query points
    
    Returns:
        min_distances: (N,) distances to nearest points
        unit_vectors: (N, 3) unit vectors from query to nearest points
        min_indices: (N,) indices of nearest points in pc
    """
    distances = torch.cdist(points, pc)  # (N, M)
    min_distances, min_indices = torch.min(distances, dim=1)  # (N,)
    nearest_points = pc[min_indices]  # (N, 3)
    vectors = nearest_points - points  # (N, 3)
    unit_vectors = vectors / (torch.norm(vectors, dim=1, keepdim=True) + 1e-4)
    return min_distances, unit_vectors, min_indices


def compute_single_ibs(
    grasp_id: int,
    batch_points_tensor: torch.Tensor,
    scene_surface_points: torch.Tensor,
    target_object_points: torch.Tensor,
    hand_surface_points: torch.Tensor,
    hand_thumb_points: torch.Tensor,
    world_to_hand_coord_transforms: torch.Tensor,
    table_height: float,
    config: IBSConfig,
    device: str,
    use_iterative_refinement: bool = True
) -> tuple:
    """
    Compute IBS for a single grasp.
    
    Args:
        grasp_id: Index of the grasp in the batch
        batch_points_tensor: (B, num_voxels, 3) grid points
        scene_surface_points: (B, M, 3) scene points in hand coord
        target_object_points: (B, M, 3) target object points in hand coord
        hand_surface_points: (B, M, 3) hand surface points in hand coord
        hand_thumb_points: (B, M, 3) hand thumb points in hand coord
        world_to_hand_coord_transforms: (B, 4, 4) transformations
        table_height: Height of table surface
        config: IBSConfig
        device: torch device
        use_iterative_refinement: Whether to use iterative refinement for IBS points
    
    Returns:
        ibs_voxel: (40, 40, 40, 3) IBS voxel
        hand_dis_voxel: (40, 40, 40) hand distance voxel
    """
    bound = config.bound
    resolution = config.resolution
    delta = config.delta
    epsilon = config.epsilon
    max_iteration = config.max_iteration
    voxel_size = config.voxel_size
    
    # Query distances
    scene_dis, _, _ = points_query(scene_surface_points[grasp_id], batch_points_tensor[grasp_id])
    target_obj_dis, _, _ = points_query(target_object_points[grasp_id], batch_points_tensor[grasp_id])
    hand_dis, _, _ = points_query(hand_surface_points[grasp_id], batch_points_tensor[grasp_id])
    thumb_dis, _, _ = points_query(hand_thumb_points[grasp_id], batch_points_tensor[grasp_id])
    
    if use_iterative_refinement:
        # Initial IBS mask: points where scene and hand distances are close
        ibs_mask = torch.abs(scene_dis - hand_dis) < delta * 2
        ibs_points = batch_points_tensor[grasp_id][ibs_mask]
        
        # Iteratively refine IBS points to lie exactly on bisector
        for i in range(max_iteration):
            if len(ibs_points) == 0:
                break
            
            scene_dis_, scene_uv, _ = points_query(scene_surface_points[grasp_id], ibs_points)
            hand_dis_, hand_uv, _ = points_query(hand_surface_points[grasp_id], ibs_points)
            dis_diff = scene_dis_ - hand_dis_
            
            if torch.max(torch.abs(dis_diff)) < epsilon:
                break
            
            # Compute adjustment direction
            adjustments = torch.where(dis_diff[:, None] > 0, scene_uv, -hand_uv)
            
            # Compute step size (Newton-like step, clamped for stability)
            # The denominator (1 - cos_angle) prevents division issues when vectors are parallel
            cos_angle = torch.sum(scene_uv * hand_uv, dim=1)
            step_size = dis_diff / (1 - cos_angle + 1e-6)
            step_size = torch.clamp(step_size, min=-0.01, max=0.01)
            
            # Update points
            ibs_points = ibs_points + step_size[:, None] * adjustments
        
        # Convert refined points to voxel coordinates
        ibs_points = (ibs_points - torch.tensor([-bound, -bound, -bound], device=device)) / resolution
        ibs_points = torch.floor(ibs_points).long()
        ibs_points = torch.clamp(ibs_points, 0, voxel_size - 1)
        
        # Create IBS occupancy mask
        ibs_mask = torch.zeros((voxel_size, voxel_size, voxel_size), dtype=torch.bool, device=device)
        if len(ibs_points) > 0:
            ibs_mask[ibs_points[:, 0], ibs_points[:, 1], ibs_points[:, 2]] = True
        ibs_mask = ibs_mask.ravel()
    else:
        # Simple threshold-based IBS (without refinement)
        ibs_mask = torch.abs(scene_dis - hand_dis) < delta
    
    # Contact mask: points close to both target object and hand
    contact_delta = config.contact_delta
    thumb_delta = config.thumb_contact_delta
    
    contact_mask = (target_obj_dis < contact_delta) & (hand_dis < contact_delta) & ibs_mask
    thumb_contact_mask = contact_mask & (thumb_dis < thumb_delta)
    
    # Remove points under table (transform back to world to check z)
    world_points = transform_points(
        batch_points_tensor[grasp_id], 
        torch.inverse(world_to_hand_coord_transforms[grasp_id])
    )
    under_table_mask = world_points[:, 2] < table_height
    ibs_mask = ibs_mask & ~under_table_mask
    
    # Reshape to voxel grid
    ibs_mask = ibs_mask.reshape(voxel_size, voxel_size, voxel_size, 1)
    contact_mask = contact_mask.reshape(voxel_size, voxel_size, voxel_size, 1)
    thumb_contact_mask = thumb_contact_mask.reshape(voxel_size, voxel_size, voxel_size, 1)
    
    # Combine channels: [occupancy, contact, thumb_contact]
    ibs_voxel = torch.cat([ibs_mask, contact_mask, thumb_contact_mask], dim=3)
    hand_dis_voxel = hand_dis.reshape(voxel_size, voxel_size, voxel_size)
    
    return ibs_voxel, hand_dis_voxel


def calculate_ibs_for_scene(
    scene_id: int,
    grasp_base_path: str = 'data/DexGraspNet2.0/dex_grasps_new',
    fps_indices_path: str = 'data/DexGraspNet2.0/fps_sampled_indices',
    output_base_path: str = 'data/ibsdata',
    scene_base_path: str = 'data/DexGraspNet2.0/scenes',
    mesh_base_path: str = 'data/DexGraspNet2.0/meshdata',
    urdf_path: str = 'robot_models/urdf/leap_hand_simplified.urdf',
    meta_path: str = 'robot_models/meta/leap_hand/meta.yaml',
    device: str = 'cuda:0',
    use_iterative_refinement: bool = True,
    n_points_each_link: int = 300,
    skip_existing: bool = True,
    visualize: bool = False
) -> dict:
    """
    Calculate IBS for all FPS-sampled grasps in a scene.
    
    Args:
        scene_id: Scene identifier
        grasp_base_path: Path to original grasp data
        fps_indices_path: Path to FPS indices
        output_base_path: Path to save IBS data
        scene_base_path: Path to scene data
        mesh_base_path: Path to mesh data
        urdf_path: Path to robot URDF
        meta_path: Path to robot meta
        device: Torch device
        use_iterative_refinement: Whether to use iterative IBS refinement
        n_points_each_link: Number of surface points per robot link
        skip_existing: Skip if output already exists
        visualize: Whether to visualize (first few grasps)
    
    Returns:
        Statistics dict
    """
    scene_name = f'scene_{str(scene_id).zfill(4)}'
    
    # Check if already processed
    ibs_output_path = os.path.join(output_base_path, 'ibs', f'{scene_name}.npy')
    if skip_existing and os.path.exists(ibs_output_path):
        cprint(f"Skipping {scene_name} (already processed)", 'yellow')
        return {'status': 'skipped'}
    
    # Load FPS-sampled grasps
    cprint(f"Loading FPS-sampled grasps for {scene_name}...", 'cyan')
    grasp_data, object_indices = load_fps_grasps_for_scene(
        scene_name, 'leap_hand',
        grasp_base_path=grasp_base_path,
        fps_base_path=fps_indices_path,
        return_object_ids=True
    )
    
    if not grasp_data or 'point' not in grasp_data:
        cprint(f"No FPS grasps found for {scene_name}", 'yellow')
        return {'status': 'no_data'}
    
    batch_size = len(grasp_data['point'])
    if batch_size == 0:
        cprint(f"Empty grasp data for {scene_name}", 'yellow')
        return {'status': 'empty'}
    
    cprint(f"Processing {scene_name}: {batch_size} grasps", 'green')
    
    # Load scene using new-style constructor
    scene = Scene(
        scene_id=scene_id,
        device=device,
        visualize=False,
        grasp_num=0,
        scene_base_path=scene_base_path,
        mesh_base_path=mesh_base_path
    )
    
    # Load extrinsics to transform grasp data from world to table coordinate system
    # Scene class applies this transform to mesh/points, so we need to do the same for grasps
    extrinsics_path = os.path.join(scene_base_path, scene_name, 'realsense/cam0_wrt_table.npy')
    extrinsics = np.load(extrinsics_path)  # (4, 4) world-to-table transform
    
    # Initialize robot model
    robot_model = RobotModel(urdf_path=urdf_path, meta_path=meta_path)
    
    # Extract grasp parameters
    # Note: In DexGraspNet2.0 data, grasp point is stored as 'point'
    trans_world = grasp_data['translation']  # (B, 3) in world coord
    rot_world = grasp_data['rotation']       # (B, 3, 3) in world coord
    gp_world = grasp_data['point']           # (B, 3) - grasp points in world coord
    
    # Transform grasp data from world to table coordinate system
    # trans_table = R @ trans_world + t
    # rot_table = R @ rot_world
    # gp_table = R @ gp_world + t
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    trans = ((R @ trans_world.T).T + t).astype(np.float32)  # (B, 3) in table coord
    rot = np.einsum('ij,bjk->bik', R, rot_world).astype(np.float32)  # (B, 3, 3) in table coord
    gp = ((R @ gp_world.T).T + t).astype(np.float32)  # (B, 3) in table coord
    
    # Get joint positions
    qpos_dict = {
        k: torch.from_numpy(v).to(device=device)
        for k, v in grasp_data.items()
        if k not in ['translation', 'rotation', 'point']
    }
    
    # Get hand surface points (including thumb points)
    hand_thumb_points, _, hand_whole_points = robot_model.get_finger_surface_points(
        torch.from_numpy(trans).to(device=device),
        torch.from_numpy(rot).to(device=device),
        qpos_dict,
        n_points_each_link=n_points_each_link
    )
    
    # Compute world-to-hand coordinate transformations
    # Hand coordinate: origin at grasp point, rotation aligned with hand orientation
    local_frame = np.eye(4)[None].repeat(batch_size, axis=0)
    local_frame[:, :3, 3] = gp
    local_frame[:, :3, :3] = rot
    world_to_hand_coord_transforms = torch.from_numpy(
        np.linalg.inv(local_frame)
    ).to(dtype=torch.float, device=device)
    
    # Get scene and target object surface points
    scene_surface_points = scene.surface_points  # (M, 3) in world coord
    target_object_points = scene.get_target_object_points_from_grasp_points(gp)  # (B, M, 3)
    
    # Transform all points to hand coordinate system
    hand_surface_points = batch_transform_points(hand_whole_points, world_to_hand_coord_transforms)
    hand_thumb_points = batch_transform_points(hand_thumb_points, world_to_hand_coord_transforms)
    scene_surface_points_batch = batch_transform_points(
        scene_surface_points.unsqueeze(0).repeat(batch_size, 1, 1),
        world_to_hand_coord_transforms
    )
    target_object_points = batch_transform_points(target_object_points, world_to_hand_coord_transforms)
    
    # Create IBS configuration
    config = IBSConfig()
    
    # Generate voxel grid points
    bound = config.bound
    resolution = config.resolution
    grid_x, grid_y, grid_z = np.mgrid[
        -bound:bound:resolution,
        -bound:bound:resolution,
        -bound:bound:resolution
    ]
    points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
    batch_points = points[np.newaxis, :, :].repeat(batch_size, axis=0)
    batch_points_tensor = torch.tensor(batch_points, dtype=torch.float, device=device)
    
    # Compute IBS for each grasp
    ibs_voxels = []
    hand_dis_voxels = []
    
    for grasp_id in tqdm(range(batch_size), desc=f"Computing IBS for {scene_name}"):
        ibs_voxel, hand_dis_voxel = compute_single_ibs(
            grasp_id=grasp_id,
            batch_points_tensor=batch_points_tensor,
            scene_surface_points=scene_surface_points_batch,
            target_object_points=target_object_points,
            hand_surface_points=hand_surface_points,
            hand_thumb_points=hand_thumb_points,
            world_to_hand_coord_transforms=world_to_hand_coord_transforms,
            table_height=scene.table_size[2],
            config=config,
            device=device,
            use_iterative_refinement=use_iterative_refinement
        )
        
        ibs_voxels.append(ibs_voxel)
        hand_dis_voxels.append(hand_dis_voxel)
    
    # Stack results
    ibs_voxels = torch.stack(ibs_voxels)  # (B, 40, 40, 40, 3)
    hand_dis_voxels = torch.stack(hand_dis_voxels)  # (B, 40, 40, 40)
    
    # Free GPU memory
    torch.cuda.empty_cache()
    
    # Create output directories
    os.makedirs(os.path.join(output_base_path, 'ibs'), exist_ok=True)
    os.makedirs(os.path.join(output_base_path, 'w2h_trans'), exist_ok=True)
    os.makedirs(os.path.join(output_base_path, 'hand_dis'), exist_ok=True)
    
    # Save results
    np.save(
        os.path.join(output_base_path, 'ibs', f'{scene_name}.npy'),
        ibs_voxels.cpu().numpy()
    )
    np.save(
        os.path.join(output_base_path, 'w2h_trans', f'{scene_name}.npy'),
        world_to_hand_coord_transforms.cpu().numpy()
    )
    np.save(
        os.path.join(output_base_path, 'hand_dis', f'{scene_name}.npy'),
        hand_dis_voxels.cpu().numpy()
    )
    
    # Compute statistics
    ibs_batch = IBSBatch(voxels=ibs_voxels, w2h_trans=world_to_hand_coord_transforms, device='cpu')
    stats = ibs_batch.get_statistics()
    stats['status'] = 'success'
    stats['scene_name'] = scene_name
    
    cprint(f"  {scene_name}: {batch_size} grasps, avg {stats['avg_ibs_points']:.1f} IBS points, "
           f"avg {stats['avg_contact_points']:.1f} contact points", 'green')
    
    return stats


def main():
    """Main entry point with argparse."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate IBS for a scene')
    parser.add_argument('--scene_id', type=int, default=55, help='Scene ID')
    parser.add_argument('--grasp_base_path', type=str, 
                        default='data/DexGraspNet2.0/dex_grasps_new',
                        help='Path to grasp data')
    parser.add_argument('--fps_indices_path', type=str,
                        default='data/DexGraspNet2.0/fps_sampled_indices',
                        help='Path to FPS indices')
    parser.add_argument('--output_base_path', type=str,
                        default='data/ibsdata',
                        help='Path to save IBS data')
    parser.add_argument('--scene_base_path', type=str,
                        default='data/DexGraspNet2.0/scenes',
                        help='Path to scene data')
    parser.add_argument('--mesh_base_path', type=str,
                        default='data/DexGraspNet2.0/meshdata',
                        help='Path to mesh data')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--no_refinement', action='store_true',
                        help='Disable iterative IBS refinement')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing output')
    parser.add_argument('--visualize', action='store_true',
                        help='Enable visualization')
    args = parser.parse_args()
    
    stats = calculate_ibs_for_scene(
        scene_id=args.scene_id,
        grasp_base_path=args.grasp_base_path,
        fps_indices_path=args.fps_indices_path,
        output_base_path=args.output_base_path,
        scene_base_path=args.scene_base_path,
        mesh_base_path=args.mesh_base_path,
        device=args.device,
        use_iterative_refinement=not args.no_refinement,
        skip_existing=not args.force,
        visualize=args.visualize
    )
    
    cprint(f"\nResult: {stats}", 'cyan')


if __name__ == '__main__':
    main()
