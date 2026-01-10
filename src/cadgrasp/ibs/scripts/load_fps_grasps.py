"""
Utility functions to load FPS-sampled grasps.

This module provides functions to load grasp data using FPS indices
from data/DexGraspNet2.0/fps_sampled_indices.

Example usage:
    from cadgrasp.ibs.scripts.load_fps_grasps import load_fps_grasps_for_scene
    
    grasps = load_fps_grasps_for_scene('scene_0000', 'leap_hand')
    # grasps is a dict with keys like 'translation', 'rotation', 'point', joint angles...
"""

import os
import numpy as np
from glob import glob
from typing import Dict, List, Optional, Tuple
from termcolor import cprint


def load_fps_indices(
    scene_id: str,
    robot_name: str = 'leap_hand',
    fps_base_path: str = 'data/DexGraspNet2.0/fps_sampled_indices'
) -> Dict[str, np.ndarray]:
    """
    Load FPS indices for all objects in a scene.
    
    Args:
        scene_id: Scene identifier (e.g., 'scene_0000')
        robot_name: Robot name (e.g., 'leap_hand')
        fps_base_path: Base path to FPS indices
    
    Returns:
        Dict mapping object_id (e.g., '046.npz') to FPS indices array
    """
    indices_path = os.path.join(fps_base_path, scene_id, robot_name)
    
    if not os.path.exists(indices_path):
        cprint(f"FPS indices not found: {indices_path}", 'red')
        return {}
    
    result = {}
    for npz_file in sorted(glob(os.path.join(indices_path, '*.npz'))):
        obj_id = os.path.basename(npz_file)
        data = np.load(npz_file)
        result[obj_id] = data['fps_indices']
    
    return result


def load_fps_grasps_for_object(
    scene_id: str,
    object_id: str,
    robot_name: str = 'leap_hand',
    grasp_base_path: str = 'data/DexGraspNet2.0/dex_grasps_new',
    fps_base_path: str = 'data/DexGraspNet2.0/fps_sampled_indices'
) -> Optional[Dict[str, np.ndarray]]:
    """
    Load FPS-sampled grasps for a single object.
    
    Args:
        scene_id: Scene identifier (e.g., 'scene_0000')
        object_id: Object identifier (e.g., '046.npz' or '046')
        robot_name: Robot name (e.g., 'leap_hand')
        grasp_base_path: Base path to original grasp data
        fps_base_path: Base path to FPS indices
    
    Returns:
        Dict with grasp data arrays (filtered by FPS), or None if not found
    """
    # Normalize object_id
    if not object_id.endswith('.npz'):
        object_id = f'{object_id}.npz'
    
    # Load original grasps
    grasp_file = os.path.join(grasp_base_path, scene_id, robot_name, object_id)
    if not os.path.exists(grasp_file):
        cprint(f"Grasp file not found: {grasp_file}", 'red')
        return None
    
    # Load FPS indices
    fps_file = os.path.join(fps_base_path, scene_id, robot_name, object_id)
    if not os.path.exists(fps_file):
        cprint(f"FPS indices not found: {fps_file}", 'red')
        return None
    
    # Load data
    grasp_data = np.load(grasp_file)
    fps_data = np.load(fps_file)
    fps_indices = fps_data['fps_indices']
    
    if len(fps_indices) == 0:
        cprint(f"No FPS grasps for {object_id}", 'yellow')
        return None
    
    # Filter grasps by FPS indices
    result = {}
    for key in grasp_data.files:
        result[key] = grasp_data[key][fps_indices]
    
    return result


def load_fps_grasps_for_scene(
    scene_id: str,
    robot_name: str = 'leap_hand',
    grasp_base_path: str = 'data/DexGraspNet2.0/dex_grasps_new',
    fps_base_path: str = 'data/DexGraspNet2.0/fps_sampled_indices',
    return_object_ids: bool = False
) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, List[int]]]]:
    """
    Load all FPS-sampled grasps for a scene, concatenated across objects.
    
    Args:
        scene_id: Scene identifier (e.g., 'scene_0000')
        robot_name: Robot name (e.g., 'leap_hand')
        grasp_base_path: Base path to original grasp data
        fps_base_path: Base path to FPS indices
        return_object_ids: If True, also return mapping of grasp index to object_id
    
    Returns:
        Tuple of (grasps_dict, object_ids_dict) where:
            - grasps_dict: Dict with concatenated grasp data
            - object_ids_dict: Dict mapping object_id to list of indices in grasps_dict
                              (only if return_object_ids=True)
    """
    # Get FPS indices
    fps_indices_dict = load_fps_indices(scene_id, robot_name, fps_base_path)
    
    if not fps_indices_dict:
        cprint(f"No FPS indices for {scene_id}", 'red')
        return ({}, None) if return_object_ids else {}
    
    # Collect data from all objects
    all_data = {}
    object_indices = {}
    current_idx = 0
    
    for object_id, fps_indices in fps_indices_dict.items():
        if len(fps_indices) == 0:
            continue
        
        obj_grasps = load_fps_grasps_for_object(
            scene_id, object_id, robot_name,
            grasp_base_path, fps_base_path
        )
        
        if obj_grasps is None or len(obj_grasps.get('point', [])) == 0:
            continue
        
        num_grasps = len(obj_grasps['point'])
        
        # Track indices for this object
        if return_object_ids:
            object_indices[object_id] = list(range(current_idx, current_idx + num_grasps))
        current_idx += num_grasps
        
        # Concatenate data
        for key, value in obj_grasps.items():
            if key not in all_data:
                all_data[key] = []
            all_data[key].append(value)
    
    # Concatenate all arrays
    result = {}
    for key, arrays in all_data.items():
        if len(arrays) > 0:
            result[key] = np.concatenate(arrays, axis=0)
    
    if return_object_ids:
        return result, object_indices
    return result


def get_fps_statistics(
    scene_id: str,
    robot_name: str = 'leap_hand',
    grasp_base_path: str = 'data/DexGraspNet2.0/dex_grasps_new',
    fps_base_path: str = 'data/DexGraspNet2.0/fps_sampled_indices'
) -> Dict[str, Dict[str, int]]:
    """
    Get statistics about FPS-sampled grasps for a scene.
    
    Args:
        scene_id: Scene identifier
        robot_name: Robot name
        grasp_base_path: Base path to original grasp data
        fps_base_path: Base path to FPS indices
    
    Returns:
        Dict with statistics per object and total
    """
    stats = {}
    total_original = 0
    total_fps = 0
    
    grasp_path = os.path.join(grasp_base_path, scene_id, robot_name)
    fps_path = os.path.join(fps_base_path, scene_id, robot_name)
    
    if not os.path.exists(grasp_path):
        cprint(f"Grasp path not found: {grasp_path}", 'red')
        return {}
    
    for obj_file in sorted(glob(os.path.join(grasp_path, '*.npz'))):
        object_id = os.path.basename(obj_file)
        
        # Count original grasps
        grasp_data = np.load(obj_file)
        num_original = len(grasp_data['point'])
        
        # Count FPS grasps
        fps_file = os.path.join(fps_path, object_id)
        if os.path.exists(fps_file):
            fps_data = np.load(fps_file)
            num_fps = len(fps_data['fps_indices'])
        else:
            num_fps = 0
        
        stats[object_id] = {
            'original': num_original,
            'fps': num_fps,
            'rate': num_fps / num_original * 100 if num_original > 0 else 0
        }
        
        total_original += num_original
        total_fps += num_fps
    
    stats['_total'] = {
        'original': total_original,
        'fps': total_fps,
        'rate': total_fps / total_original * 100 if total_original > 0 else 0
    }
    
    return stats


if __name__ == '__main__':
    # Test the module
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_id', type=str, default='scene_0000')
    parser.add_argument('--robot_name', type=str, default='leap_hand')
    args = parser.parse_args()
    
    # Test loading
    cprint(f"Testing FPS load for {args.scene_id}...", 'cyan')
    
    grasps, obj_indices = load_fps_grasps_for_scene(
        args.scene_id, args.robot_name, return_object_ids=True
    )
    
    if grasps:
        cprint(f"Loaded {len(grasps.get('point', []))} FPS-sampled grasps", 'green')
        cprint(f"Keys: {list(grasps.keys())}", 'green')
        cprint(f"Objects: {list(obj_indices.keys())}", 'green')
    else:
        cprint("No FPS data found", 'yellow')
    
    # Test statistics
    stats = get_fps_statistics(args.scene_id, args.robot_name)
    if stats:
        cprint(f"\nStatistics for {args.scene_id}:", 'cyan')
        for obj_id, obj_stats in stats.items():
            if obj_id == '_total':
                cprint(f"  TOTAL: {obj_stats['fps']}/{obj_stats['original']} ({obj_stats['rate']:.1f}%)", 'green')
            else:
                cprint(f"  {obj_id}: {obj_stats['fps']}/{obj_stats['original']} ({obj_stats['rate']:.1f}%)", 'white')
