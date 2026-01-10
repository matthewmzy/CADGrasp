"""
Batch IBS calculation for multiple scenes.

This script processes multiple scenes to compute IBS data.

Usage:
    # Process scenes 0-1000
    python batch_calculate_ibs.py --scene_start 0 --scene_end 1000
    
    # Process with specific GPU
    python batch_calculate_ibs.py --scene_start 0 --scene_end 100 --device cuda:1
"""

import os
import sys
import argparse
from glob import glob
from tqdm import tqdm
from termcolor import cprint

from cadgrasp.ibs.scripts.calculate_ibs_new import calculate_ibs_for_scene
from cadgrasp.paths import project_path


def get_scene_ids_with_fps_indices(
    fps_base_path: str = 'data/DexGraspNet2.0/fps_sampled_indices',
    robot_name: str = 'leap_hand',
    scene_start: int = 0,
    scene_end: int = 10000
) -> list:
    """
    Get list of scene IDs that have FPS indices.
    
    Args:
        fps_base_path: Path to FPS indices
        robot_name: Robot name
        scene_start: Start scene ID (inclusive)
        scene_end: End scene ID (exclusive)
    
    Returns:
        List of scene IDs with FPS indices
    """
    scene_ids = []
    
    for scene_id in range(scene_start, scene_end):
        scene_name = f'scene_{str(scene_id).zfill(4)}'
        fps_dir = os.path.join(fps_base_path, scene_name, robot_name)
        
        if os.path.exists(fps_dir):
            files = glob(os.path.join(fps_dir, '*.npz'))
            if len(files) > 0:
                scene_ids.append(scene_id)
    
    return scene_ids


def main():
    parser = argparse.ArgumentParser(description='Batch IBS calculation')
    parser.add_argument('--scene_start', type=int, default=0, help='Start scene ID (inclusive)')
    parser.add_argument('--scene_end', type=int, default=10000, help='End scene ID (exclusive)')
    parser.add_argument('--grasp_path', type=str,
                        default=project_path('data/DexGraspNet2.0/dex_grasps_new'),
                        help='Path to original grasp data')
    parser.add_argument('--fps_path', type=str,
                        default=project_path('data/DexGraspNet2.0/fps_sampled_indices'),
                        help='Path to FPS indices')
    parser.add_argument('--output_path', type=str,
                        default=project_path('data/ibsdata'),
                        help='Path to save IBS data')
    parser.add_argument('--scene_base_path', type=str,
                        default=project_path('data/DexGraspNet2.0/scenes'),
                        help='Path to scene data')
    parser.add_argument('--mesh_base_path', type=str,
                        default=project_path('data/DexGraspNet2.0/meshdata'),
                        help='Path to mesh data')
    parser.add_argument('--urdf_path', type=str,
                        default=project_path('robot_models/urdf/leap_hand_simplified.urdf'),
                        help='Path to robot URDF')
    parser.add_argument('--meta_path', type=str,
                        default=project_path('robot_models/meta/leap_hand/meta.yaml'),
                        help='Path to robot meta')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for computation')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocessing even if output exists')
    parser.add_argument('--no_iterative', action='store_true',
                        help='Disable iterative IBS refinement')
    
    args = parser.parse_args()
    
    # Get scenes with FPS indices
    cprint(f"Scanning for scenes with FPS indices in {args.fps_path}...", 'cyan')
    scene_ids = get_scene_ids_with_fps_indices(
        args.fps_path, 'leap_hand',
        args.scene_start, args.scene_end
    )
    
    cprint(f"Found {len(scene_ids)} scenes with FPS indices", 'green')
    
    if len(scene_ids) == 0:
        cprint("No scenes to process!", 'red')
        return
    
    # Process scenes
    success_count = 0
    skip_count = 0
    error_count = 0
    total_grasps = 0
    total_ibs_points = 0
    
    for scene_id in tqdm(scene_ids, desc="Processing scenes"):
        try:
            stats = calculate_ibs_for_scene(
                scene_id=scene_id,
                grasp_base_path=args.grasp_path,
                fps_indices_path=args.fps_path,
                output_base_path=args.output_path,
                scene_base_path=args.scene_base_path,
                mesh_base_path=args.mesh_base_path,
                urdf_path=args.urdf_path,
                meta_path=args.meta_path,
                device=args.device,
                use_iterative_refinement=not args.no_iterative,
                skip_existing=not args.force
            )
            
            if stats.get('status') == 'success':
                success_count += 1
                total_grasps += stats.get('batch_size', 0)
                total_ibs_points += stats.get('total_ibs_points', 0)
            elif stats.get('status') == 'skipped':
                skip_count += 1
            else:
                error_count += 1
                
        except Exception as e:
            cprint(f"Error processing scene {scene_id}: {e}", 'red')
            error_count += 1
            import traceback
            traceback.print_exc()
    
    # Print summary
    cprint("\n" + "=" * 60, 'cyan')
    cprint("IBS Calculation Summary", 'cyan')
    cprint("=" * 60, 'cyan')
    cprint(f"Total scenes found: {len(scene_ids)}", 'white')
    cprint(f"Successfully processed: {success_count}", 'green')
    cprint(f"Skipped (already exists): {skip_count}", 'yellow')
    cprint(f"Errors: {error_count}", 'red')
    
    if success_count > 0:
        cprint(f"\nTotal grasps processed: {total_grasps}", 'green')
        cprint(f"Total IBS points: {total_ibs_points}", 'green')
        cprint(f"Average IBS points per grasp: {total_ibs_points / max(total_grasps, 1):.1f}", 'green')


if __name__ == '__main__':
    main()
