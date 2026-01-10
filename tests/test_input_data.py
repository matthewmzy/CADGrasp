"""
Visualize training data loaded by IBS_Dataset.

This script visualizes how data is loaded during LASDiffusion training:
- Scene voxel (input to the network)
- IBS voxel (target output: occupancy, contact, thumb)

Usage:
    python tests/test_input_data.py [--num_samples 4] [--scene_id 0] [--batch_size 4]
"""

import os
import sys
import argparse
import numpy as np
import torch
import open3d as o3d
from typing import List, Tuple

# Add thirdparty to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'thirdparty'))

from cadgrasp.paths import project_path
from LASDiffusion.network.data_loader import IBS_Dataset, VOXEL_BOUND, VOXEL_RESOLUTION, VOXEL_SIZE


def voxel_to_points(voxel: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert voxel grid to physical point coordinates."""
    indices = np.where(voxel > threshold)
    if len(indices[0]) == 0:
        return np.zeros((0, 3))
    voxel_coords = np.stack(indices, axis=-1)
    physical_coords = voxel_coords * VOXEL_RESOLUTION - VOXEL_BOUND
    return physical_coords


def decode_ibs_voxel(ibs_voxel: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode IBS voxel from network format to occupancy/contact/thumb masks.
    
    Args:
        ibs_voxel: (2, 40, 40, 40) network format
        
    Returns:
        occupancy: (40, 40, 40) bool mask
        contact: (40, 40, 40) bool mask  
        thumb: (40, 40, 40) bool mask
    """
    occupancy = ibs_voxel[0] > 0  # Channel 0 > 0 means occupied
    contact = (ibs_voxel[1] > 0.5) & (ibs_voxel[1] < 1.5)  # Channel 1 ~ 1
    thumb = ibs_voxel[1] > 1.5  # Channel 1 ~ 2
    return occupancy, contact, thumb


def create_coordinate_frame(size: float = 0.05, origin: np.ndarray = None):
    """Create coordinate frame at origin."""
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    if origin is not None:
        frame.translate(origin)
    return frame


def create_voxel_bounds_box():
    """Create wireframe box showing voxel bounds."""
    points = [
        [-VOXEL_BOUND, -VOXEL_BOUND, -VOXEL_BOUND],
        [VOXEL_BOUND, -VOXEL_BOUND, -VOXEL_BOUND],
        [VOXEL_BOUND, VOXEL_BOUND, -VOXEL_BOUND],
        [-VOXEL_BOUND, VOXEL_BOUND, -VOXEL_BOUND],
        [-VOXEL_BOUND, -VOXEL_BOUND, VOXEL_BOUND],
        [VOXEL_BOUND, -VOXEL_BOUND, VOXEL_BOUND],
        [VOXEL_BOUND, VOXEL_BOUND, VOXEL_BOUND],
        [-VOXEL_BOUND, VOXEL_BOUND, VOXEL_BOUND],
    ]
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]
    colors = [[0.5, 0.5, 0.5] for _ in range(len(lines))]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def visualize_single_sample(
    scene_voxel: np.ndarray,
    ibs_voxel: np.ndarray,
    sample_idx: int,
    show_separate: bool = False
):
    """
    Visualize a single sample from the dataset.
    
    Args:
        scene_voxel: (1, 40, 40, 40) scene voxel
        ibs_voxel: (2, 40, 40, 40) IBS voxel in network format
        sample_idx: Sample index for title
        show_separate: If True, show scene and IBS in separate windows
    """
    geometries = []
    
    # Decode IBS voxel
    occupancy, contact, thumb = decode_ibs_voxel(ibs_voxel)
    
    # Scene points (gray)
    scene_pts = voxel_to_points(scene_voxel[0], threshold=0.5)
    if len(scene_pts) > 0:
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(scene_pts)
        scene_pcd.paint_uniform_color([0.6, 0.6, 0.6])
        geometries.append(scene_pcd)
    
    # IBS occupancy (excluding contact) - blue
    occ_only = occupancy & ~contact & ~thumb
    occ_pts = voxel_to_points(occ_only.astype(float))
    if len(occ_pts) > 0:
        occ_pcd = o3d.geometry.PointCloud()
        occ_pcd.points = o3d.utility.Vector3dVector(occ_pts)
        occ_pcd.paint_uniform_color([0.2, 0.4, 0.9])  # Blue
        geometries.append(occ_pcd)
    
    # Contact points - green
    contact_pts = voxel_to_points(contact.astype(float))
    if len(contact_pts) > 0:
        contact_pcd = o3d.geometry.PointCloud()
        contact_pcd.points = o3d.utility.Vector3dVector(contact_pts)
        contact_pcd.paint_uniform_color([0.2, 0.9, 0.2])  # Green
        geometries.append(contact_pcd)
    
    # Thumb contact - yellow
    thumb_pts = voxel_to_points(thumb.astype(float))
    if len(thumb_pts) > 0:
        thumb_pcd = o3d.geometry.PointCloud()
        thumb_pcd.points = o3d.utility.Vector3dVector(thumb_pts)
        thumb_pcd.paint_uniform_color([0.9, 0.9, 0.2])  # Yellow
        geometries.append(thumb_pcd)
    
    # Add coordinate frame and bounds box
    geometries.append(create_coordinate_frame(size=0.03))
    geometries.append(create_voxel_bounds_box())
    
    # Print statistics
    print(f"\n--- Sample {sample_idx} ---")
    print(f"Scene voxels: {np.sum(scene_voxel > 0.5)}")
    print(f"IBS occupancy: {np.sum(occupancy)}")
    print(f"  - Contact: {np.sum(contact)}")
    print(f"  - Thumb: {np.sum(thumb)}")
    print(f"  - Other: {np.sum(occ_only)}")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Sample {sample_idx} - Gray:Scene, Blue:IBS, Green:Contact, Yellow:Thumb",
        width=1280,
        height=720,
    )


def visualize_batch_grid(
    scene_voxels: np.ndarray,
    ibs_voxels: np.ndarray,
    grid_size: Tuple[int, int] = (2, 2)
):
    """
    Visualize multiple samples in a grid layout.
    
    Args:
        scene_voxels: (B, 1, 40, 40, 40) batch of scene voxels
        ibs_voxels: (B, 2, 40, 40, 40) batch of IBS voxels
        grid_size: (rows, cols) grid layout
    """
    rows, cols = grid_size
    spacing = 0.3  # Space between samples
    
    geometries = []
    
    batch_size = min(scene_voxels.shape[0], rows * cols)
    
    for i in range(batch_size):
        row = i // cols
        col = i % cols
        offset = np.array([col * spacing, row * spacing, 0])
        
        # Decode
        occupancy, contact, thumb = decode_ibs_voxel(ibs_voxels[i])
        
        # Scene (gray)
        scene_pts = voxel_to_points(scene_voxels[i, 0], threshold=0.5)
        if len(scene_pts) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(scene_pts + offset)
            pcd.paint_uniform_color([0.6, 0.6, 0.6])
            geometries.append(pcd)
        
        # IBS occupancy only (blue)
        occ_only = occupancy & ~contact & ~thumb
        occ_pts = voxel_to_points(occ_only.astype(float))
        if len(occ_pts) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(occ_pts + offset)
            pcd.paint_uniform_color([0.2, 0.4, 0.9])
            geometries.append(pcd)
        
        # Contact (green)
        contact_pts = voxel_to_points(contact.astype(float))
        if len(contact_pts) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(contact_pts + offset)
            pcd.paint_uniform_color([0.2, 0.9, 0.2])
            geometries.append(pcd)
        
        # Thumb (yellow)
        thumb_pts = voxel_to_points(thumb.astype(float))
        if len(thumb_pts) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(thumb_pts + offset)
            pcd.paint_uniform_color([0.9, 0.9, 0.2])
            geometries.append(pcd)
        
        # Add coordinate frame for each sample
        frame = create_coordinate_frame(size=0.02)
        frame.translate(offset)
        geometries.append(frame)
        
        # Print stats
        print(f"Sample {i}: Scene={np.sum(scene_voxels[i] > 0.5)}, "
              f"IBS={np.sum(occupancy)}, Contact={np.sum(contact)}, Thumb={np.sum(thumb)}")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Batch of {batch_size} samples - Grid {rows}x{cols}",
        width=1600,
        height=900,
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize IBS training data")
    parser.add_argument('--scene_id', type=int, default=0, 
                        help='Scene ID to visualize (default: 0)')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of samples to visualize (default: 4)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for dataset (default: 8)')
    parser.add_argument('--view_id', type=int, default=None,
                        help='Specific view ID (default: random)')
    parser.add_argument('--grid', action='store_true',
                        help='Show samples in grid layout')
    parser.add_argument('--ibs_path', type=str, default=None,
                        help='Path to IBS data')
    parser.add_argument('--scene_path', type=str, default=None,
                        help='Path to scene data')
    args = parser.parse_args()
    
    # Setup paths
    ibs_path = args.ibs_path or project_path('data/ibsdata')
    scene_path = args.scene_path or project_path('data/DexGraspNet2.0/scenes')
    
    print(f"Loading data from:")
    print(f"  IBS path: {ibs_path}")
    print(f"  Scene path: {scene_path}")
    print(f"  Scene ID: {args.scene_id}")
    
    # Create dataset - same way as training
    dataset = IBS_Dataset(
        path=ibs_path,
        scene_path=scene_path,
        ibs_load_per_scene=args.batch_size,
        split='train',
        use_view_filter=False,  # Don't require view annotations
        scene_ids=[args.scene_id],
    )
    
    if len(dataset) == 0:
        print(f"Error: No data found for scene {args.scene_id}")
        print(f"Available IBS files:")
        ibs_dir = os.path.join(ibs_path, 'ibs')
        if os.path.exists(ibs_dir):
            for f in sorted(os.listdir(ibs_dir))[:10]:
                print(f"  {f}")
        return
    
    print(f"\nDataset contains {len(dataset)} items (scene-view pairs)")
    print(f"Each item has {args.batch_size} IBS samples")
    
    # Get samples
    num_items = min(args.num_samples, len(dataset))
    
    if args.grid and num_items > 1:
        # Collect multiple items for grid visualization
        all_scene = []
        all_ibs = []
        
        for i in range(num_items):
            idx = i if args.view_id is None else args.view_id
            idx = idx % len(dataset)
            
            data = dataset[idx]
            # Take first sample from each batch
            all_scene.append(data['scene'][0:1].numpy())  # (1, 1, 40, 40, 40)
            all_ibs.append(data['ibs'][0:1].numpy())      # (1, 2, 40, 40, 40)
        
        scene_batch = np.concatenate(all_scene, axis=0)  # (N, 1, 40, 40, 40)
        ibs_batch = np.concatenate(all_ibs, axis=0)      # (N, 2, 40, 40, 40)
        
        grid_rows = int(np.ceil(np.sqrt(num_items)))
        grid_cols = int(np.ceil(num_items / grid_rows))
        
        print(f"\n=== Grid Visualization ({grid_rows}x{grid_cols}) ===")
        visualize_batch_grid(scene_batch, ibs_batch, (grid_rows, grid_cols))
        
    else:
        # Visualize samples one by one
        for i in range(num_items):
            idx = i if args.view_id is None else args.view_id
            idx = idx % len(dataset)
            
            print(f"\n=== Loading item {idx} ===")
            data = dataset[idx]
            
            scene_voxel = data['scene'].numpy()  # (B, 1, 40, 40, 40)
            ibs_voxel = data['ibs'].numpy()      # (B, 2, 40, 40, 40)
            
            print(f"Scene voxel shape: {scene_voxel.shape}")
            print(f"IBS voxel shape: {ibs_voxel.shape}")
            
            # Visualize first sample in batch
            visualize_single_sample(
                scene_voxel[0],  # (1, 40, 40, 40)
                ibs_voxel[0],    # (2, 40, 40, 40)
                sample_idx=i
            )
            
            if i < num_items - 1:
                input("Press Enter to see next sample...")


if __name__ == '__main__':
    main()
