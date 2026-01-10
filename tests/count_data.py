"""
Count and Statistics Data for CADGrasp Pipeline.

Count data at each stage of the pipeline:
1. Raw grasps (dex_grasps_new)
2. Success-filtered grasps (dex_grasps_success_indices)
3. FPS-sampled grasps (fps_sampled_indices)
4. IBS data (ibsdata)

Usage:
    python tests/count_data.py
    python tests/count_data.py --scene_start 100 --scene_end 130
    python tests/count_data.py --detailed
"""

import os
import sys

import argparse
import numpy as np
from glob import glob
from collections import defaultdict
from tabulate import tabulate

from cadgrasp.paths import project_path

# Default paths
DEFAULT_GRASP_PATH = project_path('data/DexGraspNet2.0/dex_grasps_new')
DEFAULT_SUCCESS_PATH = project_path('data/DexGraspNet2.0/dex_grasps_success_indices')
DEFAULT_FPS_PATH = project_path('data/DexGraspNet2.0/fps_sampled_indices')
DEFAULT_IBS_PATH = project_path('data/ibsdata')


def count_raw_grasps(scene_id: int, robot_name: str, grasp_path: str):
    """Count raw grasps for a scene."""
    scene_name = f'scene_{str(scene_id).zfill(4)}'
    scene_dir = os.path.join(grasp_path, scene_name, robot_name)
    
    if not os.path.exists(scene_dir):
        return None
    
    total = 0
    n_objects = 0
    for npz_file in glob(os.path.join(scene_dir, '*.npz')):
        data = np.load(npz_file)
        total += len(data['translation'])
        n_objects += 1
    
    return {'count': total, 'objects': n_objects}


def count_success_grasps(scene_id: int, robot_name: str, grasp_path: str, success_path: str):
    """Count success-filtered grasps for a scene."""
    scene_name = f'scene_{str(scene_id).zfill(4)}'
    scene_grasp_dir = os.path.join(grasp_path, scene_name, robot_name)
    scene_success_dir = os.path.join(success_path, scene_name, robot_name)
    
    if not os.path.exists(scene_grasp_dir):
        return None
    
    total = 0
    n_objects = 0
    
    for npz_file in glob(os.path.join(scene_grasp_dir, '*.npz')):
        obj_name = os.path.basename(npz_file)
        success_file = os.path.join(scene_success_dir, obj_name)
        
        if os.path.exists(success_file):
            data = np.load(success_file)
            total += len(data['success_indices'])
        else:
            # No success file means not filtered yet
            data = np.load(npz_file)
            total += len(data['translation'])
        n_objects += 1
    
    return {'count': total, 'objects': n_objects}


def count_fps_grasps(scene_id: int, robot_name: str, grasp_path: str, success_path: str, fps_path: str):
    """Count FPS-sampled grasps for a scene."""
    scene_name = f'scene_{str(scene_id).zfill(4)}'
    scene_fps_dir = os.path.join(fps_path, scene_name, robot_name)
    
    if not os.path.exists(scene_fps_dir):
        return None
    
    total = 0
    n_objects = 0
    
    for npz_file in glob(os.path.join(scene_fps_dir, '*.npz')):
        data = np.load(npz_file)
        total += len(data['fps_indices'])
        n_objects += 1
    
    return {'count': total, 'objects': n_objects}


def count_ibs_data(scene_id: int, ibs_path: str):
    """Count IBS data for a scene."""
    scene_name = f'scene_{str(scene_id).zfill(4)}'
    ibs_file = os.path.join(ibs_path, 'ibs', f'{scene_name}.npy')
    
    if not os.path.exists(ibs_file):
        return None
    
    data = np.load(ibs_file)
    return {'count': data.shape[0]}


def main():
    parser = argparse.ArgumentParser(description="Count data at each pipeline stage")
    parser.add_argument('--scene_start', type=int, default=0, help='Start scene ID')
    parser.add_argument('--scene_end', type=int, default=190, help='End scene ID (exclusive)')
    parser.add_argument('--robot_name', type=str, default='leap_hand')
    parser.add_argument('--grasp_path', type=str, default=DEFAULT_GRASP_PATH)
    parser.add_argument('--success_path', type=str, default=DEFAULT_SUCCESS_PATH)
    parser.add_argument('--fps_path', type=str, default=DEFAULT_FPS_PATH)
    parser.add_argument('--ibs_path', type=str, default=DEFAULT_IBS_PATH)
    parser.add_argument('--detailed', action='store_true', help='Show per-scene details')
    
    args = parser.parse_args()
    
    # Collect statistics
    stats = {
        'raw': {'scenes': 0, 'total': 0, 'objects': 0},
        'success': {'scenes': 0, 'total': 0, 'objects': 0},
        'fps': {'scenes': 0, 'total': 0, 'objects': 0},
        'ibs': {'scenes': 0, 'total': 0},
    }
    
    detailed_data = []
    
    for scene_id in range(args.scene_start, args.scene_end):
        row = {'scene': f'scene_{str(scene_id).zfill(4)}'}
        
        # Raw grasps
        raw = count_raw_grasps(scene_id, args.robot_name, args.grasp_path)
        if raw:
            stats['raw']['scenes'] += 1
            stats['raw']['total'] += raw['count']
            stats['raw']['objects'] += raw['objects']
            row['raw'] = raw['count']
        else:
            row['raw'] = '-'
        
        # Success-filtered grasps
        success = count_success_grasps(scene_id, args.robot_name, args.grasp_path, args.success_path)
        if success:
            stats['success']['scenes'] += 1
            stats['success']['total'] += success['count']
            stats['success']['objects'] += success['objects']
            row['success'] = success['count']
        else:
            row['success'] = '-'
        
        # FPS-sampled grasps
        fps = count_fps_grasps(scene_id, args.robot_name, args.grasp_path, args.success_path, args.fps_path)
        if fps:
            stats['fps']['scenes'] += 1
            stats['fps']['total'] += fps['count']
            stats['fps']['objects'] += fps['objects']
            row['fps'] = fps['count']
        else:
            row['fps'] = '-'
        
        # IBS data
        ibs = count_ibs_data(scene_id, args.ibs_path)
        if ibs:
            stats['ibs']['scenes'] += 1
            stats['ibs']['total'] += ibs['count']
            row['ibs'] = ibs['count']
        else:
            row['ibs'] = '-'
        
        detailed_data.append(row)
    
    # Print summary
    print("\n" + "=" * 70)
    print("CADGrasp Pipeline Data Statistics")
    print("=" * 70)
    print(f"Scene range: {args.scene_start} - {args.scene_end - 1}")
    print(f"Robot: {args.robot_name}")
    print()
    
    summary_table = [
        ['Stage', 'Scenes', 'Total Grasps/IBS', 'Avg per Scene'],
        ['Raw Grasps', stats['raw']['scenes'], stats['raw']['total'], 
         f"{stats['raw']['total']/max(1,stats['raw']['scenes']):.0f}"],
        ['Success Filtered', stats['success']['scenes'], stats['success']['total'],
         f"{stats['success']['total']/max(1,stats['success']['scenes']):.0f}"],
        ['FPS Sampled', stats['fps']['scenes'], stats['fps']['total'],
         f"{stats['fps']['total']/max(1,stats['fps']['scenes']):.0f}"],
        ['IBS Data', stats['ibs']['scenes'], stats['ibs']['total'],
         f"{stats['ibs']['total']/max(1,stats['ibs']['scenes']):.0f}"],
    ]
    
    print(tabulate(summary_table, headers='firstrow', tablefmt='grid'))
    
    # Print ratios
    if stats['raw']['total'] > 0:
        print(f"\nSuccess rate: {stats['success']['total']/stats['raw']['total']*100:.1f}%")
    if stats['success']['total'] > 0:
        print(f"FPS sampling ratio: {stats['fps']['total']/stats['success']['total']*100:.1f}%")
    
    # Detailed per-scene output
    if args.detailed:
        print("\n" + "-" * 70)
        print("Per-Scene Details")
        print("-" * 70)
        
        # Filter to only show scenes with data
        detailed_data = [d for d in detailed_data if any(v != '-' for k, v in d.items() if k != 'scene')]
        
        if detailed_data:
            print(tabulate(detailed_data, headers='keys', tablefmt='simple'))
        else:
            print("No data found in specified scene range.")


if __name__ == '__main__':
    main()
