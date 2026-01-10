"""
FPS (Farthest Point Sampling) for grasp selection.

This script performs FPS on successful grasps based on grasp points,
with random perturbation to handle multiple grasps at the same point.

Input: 
    - data/DexGraspNet2.0/dex_grasps_new/scene_XXXX/leap_hand/XXX.npz
    - data/DexGraspNet2.0/dex_grasps_success_indices/scene_XXXX/leap_hand/XXX.npz (optional)

Output:
    - data/DexGraspNet2.0/fps_sampled_indices/scene_XXXX/leap_hand/XXX.npz
      (contains 'fps_indices' key - indices into success_indices or original grasps)

Usage:
    # Single scene
    python fps_sample_grasps.py --scene_id 100 --max_grasps 5000
    
    # Batch processing (see batch_fps_sample_grasps.py)
"""

import os
import sys
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from termcolor import cprint
from typing import Dict, List, Optional, Tuple

import torch
from pytorch3d.ops import sample_farthest_points

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cadgrasp.ibs.scripts.load_success_grasps import (
    load_success_indices,
    load_success_grasps_for_object
)


def add_point_perturbation(
    points: np.ndarray,
    perturbation_scale: float = 0.02
) -> np.ndarray:
    """
    Add random perturbation to grasp points for FPS diversity.
    
    This is necessary because multiple grasps may share the same grasp point,
    and FPS would not distinguish them without perturbation.
    
    Args:
        points: (N, 3) array of grasp points
        perturbation_scale: Scale of random perturbation (default: 2cm)
    
    Returns:
        (N, 3) array of perturbed points
    """
    # Uniform perturbation in [-scale/2, scale/2]
    noise = np.random.rand(*points.shape) * perturbation_scale - perturbation_scale / 2
    return points + noise


def fps_select_grasps(
    grasp_points: np.ndarray,
    max_grasps: int,
    perturbation_scale: float = 0.02,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Select grasps using Farthest Point Sampling on grasp points.
    
    Args:
        grasp_points: (N, 3) array of grasp points
        max_grasps: Maximum number of grasps to select
        perturbation_scale: Scale of random perturbation for FPS diversity
        device: Device for pytorch3d FPS computation
    
    Returns:
        Array of selected indices
    """
    num_grasps = len(grasp_points)
    
    if num_grasps <= max_grasps:
        # No sampling needed, return all indices
        return np.arange(num_grasps)
    
    # Add perturbation to handle duplicate grasp points
    perturbed_points = add_point_perturbation(grasp_points, perturbation_scale)
    
    # Convert to torch tensor for FPS
    points_tensor = torch.tensor(perturbed_points, dtype=torch.float32, device=device)
    points_tensor = points_tensor.unsqueeze(0)  # (1, N, 3)
    
    # FPS selection
    _, fps_indices = sample_farthest_points(points_tensor, K=max_grasps)
    
    return fps_indices[0].cpu().numpy()


def process_scene(
    scene_id: int,
    robot_name: str = 'leap_hand',
    grasp_base_path: str = 'data/DexGraspNet2.0/dex_grasps_new',
    success_indices_path: str = 'data/DexGraspNet2.0/dex_grasps_success_indices',
    output_base_path: str = 'data/DexGraspNet2.0/fps_sampled_indices',
    max_grasps_per_scene: int = 5000,
    perturbation_scale: float = 0.02,
    device: str = 'cuda',
    skip_existing: bool = True
) -> Dict[str, int]:
    """
    Process a single scene: FPS sample successful grasps and save indices.
    
    This function:
    1. Loads success indices (or defaults to all if not available)
    2. Collects all grasp points across objects
    3. Performs FPS sampling with perturbation
    4. Maps FPS results back to per-object indices
    5. Saves per-object FPS indices
    
    Args:
        scene_id: Scene identifier (integer)
        robot_name: Robot name
        grasp_base_path: Path to original grasp data
        success_indices_path: Path to success indices (optional)
        output_base_path: Path to save FPS indices
        max_grasps_per_scene: Maximum grasps to select per scene
        perturbation_scale: Scale of perturbation for FPS
        device: Device for computation
        skip_existing: Skip if output already exists
    
    Returns:
        Dict with statistics (original_count, fps_count, per_object_counts)
    """
    scene_name = f'scene_{str(scene_id).zfill(4)}'
    output_dir = os.path.join(output_base_path, scene_name, robot_name)
    grasp_dir = os.path.join(grasp_base_path, scene_name, robot_name)
    
    # Check if scene has grasp data
    if not os.path.exists(grasp_dir):
        cprint(f"No grasp data for {scene_name}", 'yellow')
        return {'status': 'no_data'}
    
    # Check if already processed
    if skip_existing and os.path.exists(output_dir):
        existing_files = glob(os.path.join(output_dir, '*.npz'))
        if len(existing_files) > 0:
            cprint(f"Skipping {scene_name} (already processed)", 'yellow')
            return {'status': 'skipped'}
    
    # Get all object files
    object_files = sorted(glob(os.path.join(grasp_dir, '*.npz')))
    if len(object_files) == 0:
        cprint(f"No grasp files in {scene_name}", 'yellow')
        return {'status': 'no_files'}
    
    # Load success indices for all objects (defaults to all if not available)
    success_indices_dict = load_success_indices(
        scene_name, robot_name,
        base_path=success_indices_path,
        grasp_base_path=grasp_base_path,
        default_all_success=True
    )
    
    # Collect all grasp points across objects
    all_grasp_points = []
    all_object_ids = []
    all_local_indices = []  # Index within each object's success_indices
    object_success_indices = {}  # Store success indices per object
    
    for obj_file in object_files:
        object_id = os.path.basename(obj_file)
        
        # Get success indices for this object
        if object_id in success_indices_dict:
            obj_success_indices = success_indices_dict[object_id]
        else:
            # Default to all indices
            grasp_data = np.load(obj_file)
            num_grasps = len(grasp_data['point'])
            obj_success_indices = np.arange(num_grasps)
        
        if len(obj_success_indices) == 0:
            continue
        
        # Load grasp data
        grasp_data = np.load(obj_file)
        grasp_points = grasp_data['point'][obj_success_indices]
        
        # Store for later mapping
        object_success_indices[object_id] = obj_success_indices
        
        # Collect grasp points
        num_success = len(grasp_points)
        all_grasp_points.append(grasp_points)
        all_object_ids.extend([object_id] * num_success)
        all_local_indices.extend(range(num_success))
    
    if len(all_grasp_points) == 0:
        cprint(f"No successful grasps in {scene_name}", 'yellow')
        return {'status': 'no_success'}
    
    # Concatenate all grasp points
    all_grasp_points = np.concatenate(all_grasp_points, axis=0)
    all_object_ids = np.array(all_object_ids)
    all_local_indices = np.array(all_local_indices)
    
    total_grasps = len(all_grasp_points)
    cprint(f"Processing {scene_name}: {total_grasps} grasps -> max {max_grasps_per_scene}", 'cyan')
    
    # FPS selection
    fps_global_indices = fps_select_grasps(
        all_grasp_points,
        max_grasps=max_grasps_per_scene,
        perturbation_scale=perturbation_scale,
        device=device
    )
    
    # Map FPS indices back to per-object indices
    # fps_indices[object_id] = indices into that object's success_indices
    fps_per_object = {}
    for global_idx in fps_global_indices:
        obj_id = all_object_ids[global_idx]
        local_idx = all_local_indices[global_idx]
        
        if obj_id not in fps_per_object:
            fps_per_object[obj_id] = []
        fps_per_object[obj_id].append(local_idx)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save per-object FPS indices
    stats = {
        'status': 'success',
        'total_original': total_grasps,
        'total_fps': len(fps_global_indices),
        'objects': {}
    }
    
    for object_id in object_success_indices.keys():
        # Get FPS indices for this object (indices into success_indices)
        if object_id in fps_per_object:
            fps_local_indices = np.array(sorted(fps_per_object[object_id]))
            # Convert to indices into original grasp data
            obj_success_idx = object_success_indices[object_id]
            fps_original_indices = obj_success_idx[fps_local_indices]
        else:
            fps_original_indices = np.array([], dtype=np.int64)
        
        # Save
        output_file = os.path.join(output_dir, object_id)
        np.savez(output_file, fps_indices=fps_original_indices)
        
        stats['objects'][object_id] = {
            'success_count': len(object_success_indices[object_id]),
            'fps_count': len(fps_original_indices)
        }
    
    cprint(f"  {scene_name}: {total_grasps} -> {len(fps_global_indices)} grasps", 'green')
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='FPS sample grasps for a scene')
    parser.add_argument('--scene_id', type=int, required=True, help='Scene ID to process')
    parser.add_argument('--robot_name', type=str, default='leap_hand', help='Robot name')
    parser.add_argument('--grasp_path', type=str, 
                        default='data/DexGraspNet2.0/dex_grasps_new',
                        help='Path to original grasp data')
    parser.add_argument('--success_indices_path', type=str,
                        default='data/DexGraspNet2.0/dex_grasps_success_indices',
                        help='Path to success indices')
    parser.add_argument('--output_path', type=str,
                        default='data/DexGraspNet2.0/fps_sampled_indices',
                        help='Path to save FPS indices')
    parser.add_argument('--max_grasps', type=int, default=5000,
                        help='Maximum grasps per scene')
    parser.add_argument('--perturbation', type=float, default=0.02,
                        help='Perturbation scale for FPS (default: 2cm)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for computation')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocessing even if output exists')
    
    args = parser.parse_args()
    
    stats = process_scene(
        scene_id=args.scene_id,
        robot_name=args.robot_name,
        grasp_base_path=args.grasp_path,
        success_indices_path=args.success_indices_path,
        output_base_path=args.output_path,
        max_grasps_per_scene=args.max_grasps,
        perturbation_scale=args.perturbation,
        device=args.device,
        skip_existing=not args.force
    )
    
    cprint(f"\nResult: {stats}", 'cyan')


if __name__ == '__main__':
    main()
