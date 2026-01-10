"""
Utility functions to load filtered successful grasps.

This module provides functions to load grasp data using success indices
from data/DexGraspNet2.0/dex_grasps_success_indices.

Example usage:
    from cadgrasp.ibs.scripts.load_success_grasps import load_success_grasps_for_scene
    
    grasps = load_success_grasps_for_scene('scene_0000', 'leap_hand')
    # grasps is a dict with keys like 'translation', 'rotation', 'point', joint angles...
"""

import os
import numpy as np
from glob import glob
from typing import Dict, List, Optional, Tuple
from termcolor import cprint


def load_success_indices(
    scene_id: str,
    robot_name: str = 'leap_hand',
    base_path: str = 'data/DexGraspNet2.0/dex_grasps_success_indices',
    grasp_base_path: str = 'data/DexGraspNet2.0/dex_grasps_new',
    default_all_success: bool = True
) -> Dict[str, np.ndarray]:
    """
    Load success indices for all objects in a scene.
    
    If success_indices folder doesn't exist and default_all_success=True,
    returns all indices (assuming all grasps are successful).
    
    Args:
        scene_id: Scene identifier (e.g., 'scene_0000')
        robot_name: Robot name (e.g., 'leap_hand')
        base_path: Base path to success indices
        grasp_base_path: Base path to original grasp data (used when defaulting to all)
        default_all_success: If True, return all indices when success_indices not found
    
    Returns:
        Dict mapping object_id (e.g., '046.npz') to success indices array
    """
    indices_path = os.path.join(base_path, scene_id, robot_name)
    grasp_path = os.path.join(grasp_base_path, scene_id, robot_name)
    
    # Check if success_indices folder exists
    if os.path.exists(indices_path):
        # Load from success indices
        result = {}
        for npz_file in sorted(glob(os.path.join(indices_path, '*.npz'))):
            obj_id = os.path.basename(npz_file)
            data = np.load(npz_file)
            result[obj_id] = data['success_indices']
        return result
    
    # Success indices not found
    if default_all_success:
        # Default: return all indices from original grasp data
        if not os.path.exists(grasp_path):
            cprint(f"Neither success indices nor grasp data found for {scene_id}", 'red')
            return {}
        
        cprint(f"Success indices not found, defaulting to all grasps for {scene_id}", 'yellow')
        result = {}
        for npz_file in sorted(glob(os.path.join(grasp_path, '*.npz'))):
            obj_id = os.path.basename(npz_file)
            data = np.load(npz_file)
            num_grasps = len(data['point']) if 'point' in data.files else len(data[data.files[0]])
            result[obj_id] = np.arange(num_grasps)
        return result
    else:
        cprint(f"Success indices not found: {indices_path}", 'red')
        return {}


def load_success_grasps_for_object(
    scene_id: str,
    object_id: str,
    robot_name: str = 'leap_hand',
    grasp_base_path: str = 'data/DexGraspNet2.0/dex_grasps_new',
    indices_base_path: str = 'data/DexGraspNet2.0/dex_grasps_success_indices',
    default_all_success: bool = True
) -> Optional[Dict[str, np.ndarray]]:
    """
    Load successful grasps for a single object.
    
    If success_indices file doesn't exist and default_all_success=True,
    returns all grasps (assuming all are successful).
    
    Args:
        scene_id: Scene identifier (e.g., 'scene_0000')
        object_id: Object identifier (e.g., '046.npz' or '046')
        robot_name: Robot name (e.g., 'leap_hand')
        grasp_base_path: Base path to original grasp data
        indices_base_path: Base path to success indices
        default_all_success: If True, return all grasps when success_indices not found
    
    Returns:
        Dict with grasp data arrays (filtered by success), or None if not found
    """
    # Normalize object_id
    if not object_id.endswith('.npz'):
        object_id = f'{object_id}.npz'
    
    # Load original grasps
    grasp_file = os.path.join(grasp_base_path, scene_id, robot_name, object_id)
    if not os.path.exists(grasp_file):
        cprint(f"Grasp file not found: {grasp_file}", 'red')
        return None
    
    grasp_data = np.load(grasp_file)
    
    # Load success indices
    indices_file = os.path.join(indices_base_path, scene_id, robot_name, object_id)
    if os.path.exists(indices_file):
        indices_data = np.load(indices_file)
        success_indices = indices_data['success_indices']
    elif default_all_success:
        # Default to all indices
        num_grasps = len(grasp_data['point']) if 'point' in grasp_data.files else len(grasp_data[grasp_data.files[0]])
        success_indices = np.arange(num_grasps)
    else:
        cprint(f"Success indices not found: {indices_file}", 'red')
        return None
    
    if len(success_indices) == 0:
        cprint(f"No successful grasps for {object_id}", 'yellow')
        return None
    
    # Filter grasps by success indices
    result = {}
    for key in grasp_data.files:
        result[key] = grasp_data[key][success_indices]
    
    return result


def load_success_grasps_for_scene(
    scene_id: str,
    robot_name: str = 'leap_hand',
    grasp_base_path: str = 'data/DexGraspNet2.0/dex_grasps_new',
    indices_base_path: str = 'data/DexGraspNet2.0/dex_grasps_success_indices',
    return_object_ids: bool = False,
    default_all_success: bool = True
) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, List[int]]]]:
    """
    Load all successful grasps for a scene, concatenated across objects.
    
    If success_indices folder doesn't exist and default_all_success=True,
    returns all grasps (assuming all are successful).
    
    Args:
        scene_id: Scene identifier (e.g., 'scene_0000')
        robot_name: Robot name (e.g., 'leap_hand')
        grasp_base_path: Base path to original grasp data
        indices_base_path: Base path to success indices
        return_object_ids: If True, also return mapping of grasp index to object_id
        default_all_success: If True, return all grasps when success_indices not found
    
    Returns:
        Tuple of (grasps_dict, object_ids_dict) where:
            - grasps_dict: Dict with concatenated grasp data
            - object_ids_dict: Dict mapping object_id to list of indices in grasps_dict
                              (only if return_object_ids=True)
    """
    # Get all object files
    grasp_path = os.path.join(grasp_base_path, scene_id, robot_name)
    if not os.path.exists(grasp_path):
        cprint(f"Grasp path not found: {grasp_path}", 'red')
        return ({}, None) if return_object_ids else {}
    
    object_files = sorted(glob(os.path.join(grasp_path, '*.npz')))
    if len(object_files) == 0:
        cprint(f"No grasp files in {grasp_path}", 'red')
        return ({}, None) if return_object_ids else {}
    
    # Collect data from all objects
    all_data = {}
    object_indices = {}
    current_idx = 0
    
    for obj_file in object_files:
        object_id = os.path.basename(obj_file)
        obj_grasps = load_success_grasps_for_object(
            scene_id, object_id, robot_name,
            grasp_base_path, indices_base_path,
            default_all_success=default_all_success
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


def get_success_statistics(
    scene_id: str,
    robot_name: str = 'leap_hand',
    grasp_base_path: str = 'data/DexGraspNet2.0/dex_grasps_new',
    indices_base_path: str = 'data/DexGraspNet2.0/dex_grasps_success_indices'
) -> Dict[str, Dict[str, int]]:
    """
    Get statistics about successful grasps for a scene.
    
    Args:
        scene_id: Scene identifier
        robot_name: Robot name
        grasp_base_path: Base path to original grasp data
        indices_base_path: Base path to success indices
    
    Returns:
        Dict with statistics per object and total
    """
    stats = {}
    total_original = 0
    total_success = 0
    
    grasp_path = os.path.join(grasp_base_path, scene_id, robot_name)
    indices_path = os.path.join(indices_base_path, scene_id, robot_name)
    
    if not os.path.exists(grasp_path) or not os.path.exists(indices_path):
        return {}
    
    for obj_file in sorted(glob(os.path.join(grasp_path, '*.npz'))):
        object_id = os.path.basename(obj_file)
        
        # Count original grasps
        grasp_data = np.load(obj_file)
        num_original = len(grasp_data['point'])
        
        # Count successful grasps
        indices_file = os.path.join(indices_path, object_id)
        if os.path.exists(indices_file):
            indices_data = np.load(indices_file)
            num_success = len(indices_data['success_indices'])
        else:
            num_success = 0
        
        stats[object_id] = {
            'original': num_original,
            'success': num_success,
            'rate': num_success / num_original * 100 if num_original > 0 else 0
        }
        
        total_original += num_original
        total_success += num_success
    
    stats['_total'] = {
        'original': total_original,
        'success': total_success,
        'rate': total_success / total_original * 100 if total_original > 0 else 0
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
    cprint(f"Testing load for {args.scene_id}...", 'cyan')
    
    grasps, obj_indices = load_success_grasps_for_scene(
        args.scene_id, args.robot_name, return_object_ids=True
    )
    
    if grasps:
        cprint(f"Loaded {len(grasps.get('point', []))} successful grasps", 'green')
        cprint(f"Keys: {list(grasps.keys())}", 'green')
        cprint(f"Objects: {list(obj_indices.keys())}", 'green')
    
    # Test statistics
    stats = get_success_statistics(args.scene_id, args.robot_name)
    if stats:
        cprint(f"\nStatistics for {args.scene_id}:", 'cyan')
        for obj_id, obj_stats in stats.items():
            if obj_id == '_total':
                cprint(f"  TOTAL: {obj_stats['success']}/{obj_stats['original']} ({obj_stats['rate']:.1f}%)", 'green')
            else:
                cprint(f"  {obj_id}: {obj_stats['success']}/{obj_stats['original']} ({obj_stats['rate']:.1f}%)", 'white')
