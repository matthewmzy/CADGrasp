"""
Batch FPS sampling for multiple scenes.

This script processes multiple scenes in parallel using multiprocessing.

Usage:
    # Process scenes 0-1000 using 4 processes
    python batch_fps_sample_grasps.py --scene_start 0 --scene_end 1000 --num_workers 4
    
    # Process specific scene range with custom max grasps
    python batch_fps_sample_grasps.py --scene_start 100 --scene_end 200 --max_grasps 3000
"""

import os
import sys
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from termcolor import cprint
from multiprocessing import Pool, cpu_count
from functools import partial

from cadgrasp.ibs.scripts.fps_sample_grasps import process_scene
from cadgrasp.paths import project_path


def get_scene_ids_with_grasps(
    grasp_base_path: str = 'data/DexGraspNet2.0/dex_grasps_new',
    robot_name: str = 'leap_hand',
    scene_start: int = 0,
    scene_end: int = 10000
) -> list:
    """
    Get list of scene IDs that have grasp data.
    
    Args:
        grasp_base_path: Path to grasp data
        robot_name: Robot name
        scene_start: Start scene ID (inclusive)
        scene_end: End scene ID (exclusive)
    
    Returns:
        List of scene IDs with grasp data
    """
    scene_ids = []
    
    for scene_id in range(scene_start, scene_end):
        scene_name = f'scene_{str(scene_id).zfill(4)}'
        grasp_dir = os.path.join(grasp_base_path, scene_name, robot_name)
        
        if os.path.exists(grasp_dir):
            files = glob(os.path.join(grasp_dir, '*.npz'))
            if len(files) > 0:
                scene_ids.append(scene_id)
    
    return scene_ids


def process_single_scene(
    scene_id: int,
    robot_name: str,
    grasp_base_path: str,
    success_indices_path: str,
    output_base_path: str,
    max_grasps_per_scene: int,
    perturbation_scale: float,
    device: str,
    skip_existing: bool
) -> dict:
    """
    Wrapper function for multiprocessing.
    """
    try:
        return process_scene(
            scene_id=scene_id,
            robot_name=robot_name,
            grasp_base_path=grasp_base_path,
            success_indices_path=success_indices_path,
            output_base_path=output_base_path,
            max_grasps_per_scene=max_grasps_per_scene,
            perturbation_scale=perturbation_scale,
            device=device,
            skip_existing=skip_existing
        )
    except Exception as e:
        cprint(f"Error processing scene {scene_id}: {e}", 'red')
        return {'status': 'error', 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Batch FPS sample grasps for multiple scenes')
    parser.add_argument('--scene_start', type=int, default=0, help='Start scene ID (inclusive)')
    parser.add_argument('--scene_end', type=int, default=10000, help='End scene ID (exclusive)')
    parser.add_argument('--robot_name', type=str, default='leap_hand', help='Robot name')
    parser.add_argument('--grasp_path', type=str,
                        default=project_path('data/DexGraspNet2.0/dex_grasps_new'),
                        help='Path to original grasp data')
    parser.add_argument('--success_indices_path', type=str,
                        default=project_path('data/DexGraspNet2.0/dex_grasps_success_indices'),
                        help='Path to success indices')
    parser.add_argument('--output_path', type=str,
                        default=project_path('data/DexGraspNet2.0/fps_sampled_indices'),
                        help='Path to save FPS indices')
    parser.add_argument('--max_grasps', type=int, default=5000,
                        help='Maximum grasps per scene')
    parser.add_argument('--perturbation', type=float, default=0.02,
                        help='Perturbation scale for FPS (default: 2cm)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for computation')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of parallel workers')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocessing even if output exists')
    
    args = parser.parse_args()
    
    # Get scenes with grasp data
    cprint(f"Scanning for scenes with grasp data in {args.grasp_path}...", 'cyan')
    scene_ids = get_scene_ids_with_grasps(
        args.grasp_path, args.robot_name,
        args.scene_start, args.scene_end
    )
    
    cprint(f"Found {len(scene_ids)} scenes with grasp data", 'green')
    
    if len(scene_ids) == 0:
        cprint("No scenes to process!", 'red')
        return
    
    # Process scenes
    success_count = 0
    skip_count = 0
    error_count = 0
    total_original = 0
    total_fps = 0
    
    if args.num_workers <= 1:
        # Sequential processing
        for scene_id in tqdm(scene_ids, desc="Processing scenes"):
            stats = process_single_scene(
                scene_id=scene_id,
                robot_name=args.robot_name,
                grasp_base_path=args.grasp_path,
                success_indices_path=args.success_indices_path,
                output_base_path=args.output_path,
                max_grasps_per_scene=args.max_grasps,
                perturbation_scale=args.perturbation,
                device=args.device,
                skip_existing=not args.force
            )
            
            if stats.get('status') == 'success':
                success_count += 1
                total_original += stats.get('total_original', 0)
                total_fps += stats.get('total_fps', 0)
            elif stats.get('status') == 'skipped':
                skip_count += 1
            elif stats.get('status') == 'error':
                error_count += 1
    else:
        # Parallel processing
        # Note: For GPU operations, parallel processing may not work well
        # Consider using device='cpu' or single worker for GPU operations
        cprint(f"Using {args.num_workers} workers (note: GPU operations may conflict)", 'yellow')
        
        func = partial(
            process_single_scene,
            robot_name=args.robot_name,
            grasp_base_path=args.grasp_path,
            success_indices_path=args.success_indices_path,
            output_base_path=args.output_path,
            max_grasps_per_scene=args.max_grasps,
            perturbation_scale=args.perturbation,
            device='cpu',  # Use CPU for parallel processing
            skip_existing=not args.force
        )
        
        with Pool(args.num_workers) as pool:
            results = list(tqdm(
                pool.imap(func, scene_ids),
                total=len(scene_ids),
                desc="Processing scenes"
            ))
        
        for stats in results:
            if stats.get('status') == 'success':
                success_count += 1
                total_original += stats.get('total_original', 0)
                total_fps += stats.get('total_fps', 0)
            elif stats.get('status') == 'skipped':
                skip_count += 1
            elif stats.get('status') == 'error':
                error_count += 1
    
    # Print summary
    cprint("\n" + "=" * 60, 'cyan')
    cprint("FPS Sampling Summary", 'cyan')
    cprint("=" * 60, 'cyan')
    cprint(f"Total scenes found: {len(scene_ids)}", 'white')
    cprint(f"Successfully processed: {success_count}", 'green')
    cprint(f"Skipped (already exists): {skip_count}", 'yellow')
    cprint(f"Errors: {error_count}", 'red')
    
    if success_count > 0:
        cprint(f"\nTotal grasps: {total_original} -> {total_fps} after FPS", 'green')
        cprint(f"Average reduction: {total_original / max(success_count, 1):.1f} -> {total_fps / max(success_count, 1):.1f} per scene", 'green')


if __name__ == '__main__':
    main()
