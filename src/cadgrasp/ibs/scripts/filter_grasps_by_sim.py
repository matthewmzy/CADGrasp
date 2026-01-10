"""
Filter successful grasps using IsaacGym simulation.

This script reads grasp data from data/DexGraspNet2.0/dex_grasps_new,
evaluates them in IsaacGym simulation, and saves the indices of successful
grasps to data/DexGraspNet2.0/dex_grasps_success_indices with the same
directory structure.

Usage:
    python src/cadgrasp/ibs/scripts/filter_grasps_by_sim.py scene_id=scene_0000

Directory structure:
    Input:  data/DexGraspNet2.0/dex_grasps_new/scene_XXXX/leap_hand/046.npz
    Output: data/DexGraspNet2.0/dex_grasps_success_indices/scene_XXXX/leap_hand/046.npz
            (contains only 'success_indices' key with indices of successful grasps)
"""

import os
import sys

from isaacgym import gymapi, gymtorch
from cadgrasp.baseline.utils.data_evaluator.simulation_evaluator import SimulationEvaluator
import yaml
import numpy as np
import transforms3d
import torch
import xml.etree.ElementTree as ET
from pytorch3d.transforms import matrix_to_euler_angles
from pytorch3d import transforms as pttf
from typing import Union
from termcolor import cprint
from glob import glob

from cadgrasp.baseline.utils.util import set_seed


def load_scene_annotation(scene_id: str):
    """Load scene annotation and object poses."""
    scene_path = os.path.join('data/DexGraspNet2.0/scenes', scene_id)
    extrinsics_path = os.path.join(scene_path, 'realsense/cam0_wrt_table.npy')
    extrinsics = np.load(extrinsics_path)
    annotation_path = os.path.join(scene_path, 'realsense/annotations/0000.xml')
    annotation = ET.parse(annotation_path)

    # Parse scene annotation
    object_pose_dict = {}
    for obj in annotation.findall('obj'):
        object_code = str(int(obj.find('obj_id').text)).zfill(3)
        translation = np.array([float(x) for x in obj.find('pos_in_world').text.split()])
        rotation = np.array([float(x) for x in obj.find('ori_in_world').text.split()])
        rotation = transforms3d.quaternions.quat2mat(rotation)
        object_pose = np.eye(4)
        object_pose[:3, :3] = rotation
        object_pose[:3, 3] = translation
        object_pose = extrinsics @ object_pose
        object_pose_dict[object_code] = object_pose

    # Load object surface points
    object_surface_points_dict = {}
    for object_code in object_pose_dict:
        object_surface_points_path = os.path.join('data/DexGraspNet2.0/meshdata', 
            object_code, f'surface_points_1000.npy')
        object_surface_points = np.load(object_surface_points_path)
        object_pose = object_pose_dict[object_code]
        object_surface_points = object_surface_points @ object_pose[:3, :3].T + object_pose[:3, 3]
        object_surface_points_dict[object_code] = object_surface_points

    return object_pose_dict, object_surface_points_dict, extrinsics


def load_grasp_data_for_object(grasp_file_path: str, extrinsics: np.ndarray):
    """
    Load grasp data for a single object and transform to world frame.
    
    Args:
        grasp_file_path: Path to the .npz file containing grasp data
        extrinsics: Camera extrinsics (cam0_wrt_table)
    
    Returns:
        dict: Grasp data with keys 'translation', 'rotation', and joint angles
    """
    data = np.load(grasp_file_path)
    
    # Get basic grasp info
    points = data['point']  # [N, 3] - in camera frame relative to cam0
    translations = data['translation']  # [N, 3]
    rotations = data['rotation']  # [N, 3, 3]
    
    # Get joint angle keys (everything except point, translation, rotation)
    joint_keys = [k for k in data.files if k not in ['point', 'translation', 'rotation']]
    
    # Transform from cam0 frame to world frame
    world_rot = extrinsics[:3, :3]
    world_trans = extrinsics[:3, 3]
    
    # Transform translations and rotations to world frame
    trans_world = (world_rot @ translations.T).T + world_trans  # [N, 3]
    rot_world = np.einsum('ab,nbc->nac', world_rot, rotations)  # [N, 3, 3]
    
    # Build grasp dict
    grasps = {
        'translation': trans_world,
        'rotation': rot_world,
    }
    
    # Add joint angles
    for key in joint_keys:
        grasps[key] = data[key]
    
    return grasps


def evaluate_grasps_for_object(
    grasps: dict,
    data_evaluator: SimulationEvaluator,
    batch_size: int = 100
) -> np.ndarray:
    """
    Evaluate grasps using simulation and return success indices.
    
    Args:
        grasps: Dict with 'translation', 'rotation', and joint angles
        data_evaluator: SimulationEvaluator instance
        batch_size: Batch size for evaluation
    
    Returns:
        np.ndarray: Boolean array indicating success for each grasp
    """
    num_grasps = len(grasps['translation'])
    successes = []
    
    for i in range(0, num_grasps, batch_size):
        end = min(i + batch_size, num_grasps)
        grasps_batch = {joint: grasps[joint][i:end] for joint in grasps}
        
        # Pad the first grasp to avoid IsaacGym bug
        grasps_batch = {
            joint: np.concatenate([grasps_batch[joint][:1], grasps_batch[joint]]) 
            for joint in grasps_batch
        }
        
        successes_batch = data_evaluator.evaluate_data(grasps_batch)
        successes.append(successes_batch[1:])  # Remove padded grasp
    
    return np.concatenate(successes)


def filter_grasps_for_scene(
    scene_id: str,
    robot_name: str = 'leap_hand',
    input_base: str = 'data/DexGraspNet2.0/dex_grasps_new',
    output_base: str = 'data/DexGraspNet2.0/dex_grasps_success_indices',
    evaluator_config_path: str = 'configs/data_evaluator/leap_hand/SimulationEvaluator.yaml',
    device: str = 'cuda:0',
    batch_size: int = 100,
    headless: bool = True,
    overwrite: bool = False,
    seed: int = 2
):
    """
    Filter successful grasps for a scene using IsaacGym simulation.
    
    Args:
        scene_id: Scene identifier (e.g., "scene_0000")
        robot_name: Robot name (default: 'leap_hand')
        input_base: Path to input grasp data
        output_base: Path to save success indices
        evaluator_config_path: Path to evaluator config
        device: Torch device
        batch_size: Batch size for simulation
        headless: Run simulation headless
        overwrite: Overwrite existing results
        seed: Random seed
    """
    set_seed(seed)
    torch_device = torch.device(device)
    
    # Paths
    input_scene_path = os.path.join(input_base, scene_id, robot_name)
    output_scene_path = os.path.join(output_base, scene_id, robot_name)
    
    # Check if input exists
    if not os.path.exists(input_scene_path):
        cprint(f"Input path does not exist: {input_scene_path}", 'red')
        return
    
    # Get all grasp files for this scene
    grasp_files = sorted(glob(os.path.join(input_scene_path, '*.npz')))
    if len(grasp_files) == 0:
        cprint(f"No grasp files found in {input_scene_path}", 'red')
        return
    
    # Check if already processed (all files exist)
    all_processed = True
    for grasp_file in grasp_files:
        obj_id = os.path.basename(grasp_file)
        output_file = os.path.join(output_scene_path, obj_id)
        if not os.path.exists(output_file):
            all_processed = False
            break
    
    if all_processed and not overwrite:
        cprint(f'{scene_id} already processed, skipping', 'yellow')
        return
    
    cprint(f"Processing {scene_id} with {len(grasp_files)} objects", 'green')
    
    # Load scene annotation
    object_pose_dict, object_surface_points_dict, extrinsics = load_scene_annotation(scene_id)
    
    # Create data evaluator
    evaluator_config = yaml.safe_load(open(evaluator_config_path, 'r'))
    evaluator_config['headless'] = headless
    data_evaluator = SimulationEvaluator(evaluator_config, torch_device)
    
    # Set environments
    data_evaluator.set_environments(
        object_pose_dict, 
        object_surface_points_dict, 
        batch_size + 1, 
        'graspnet'
    )
    
    # Process each object's grasp file
    os.makedirs(output_scene_path, exist_ok=True)
    
    for grasp_file in grasp_files:
        obj_id = os.path.basename(grasp_file)  # e.g., "046.npz"
        output_file = os.path.join(output_scene_path, obj_id)
        
        # Skip if already processed
        if os.path.exists(output_file) and not overwrite:
            cprint(f"  {obj_id} already processed, skipping", 'yellow')
            continue
        
        cprint(f"  Processing {obj_id}...", 'cyan')
        
        # Load grasp data
        grasps = load_grasp_data_for_object(grasp_file, extrinsics)
        num_grasps = len(grasps['translation'])
        
        if num_grasps == 0:
            cprint(f"    No grasps found, skipping", 'yellow')
            continue
        
        # Evaluate grasps
        successes = evaluate_grasps_for_object(
            grasps, 
            data_evaluator, 
            batch_size
        )
        
        # Get success indices
        success_indices = np.where(successes)[0]
        success_rate = len(success_indices) / num_grasps * 100
        
        cprint(f"    {len(success_indices)}/{num_grasps} successful ({success_rate:.1f}%)", 'green')
        
        # Save success indices
        np.savez(output_file, success_indices=success_indices)
    
    cprint(f"Finished processing {scene_id}", 'green')


def main():
    """Main entry point with argparse."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Filter successful grasps using IsaacGym simulation')
    parser.add_argument('--scene_id', type=str, default='scene_0000', help='Scene ID')
    parser.add_argument('--robot_name', type=str, default='leap_hand', help='Robot name')
    parser.add_argument('--input_base', type=str, 
                        default='data/DexGraspNet2.0/dex_grasps_new',
                        help='Path to input grasp data')
    parser.add_argument('--output_base', type=str,
                        default='data/DexGraspNet2.0/dex_grasps_success_indices',
                        help='Path to save success indices')
    parser.add_argument('--evaluator_config', type=str,
                        default='configs/data_evaluator/leap_hand/SimulationEvaluator.yaml',
                        help='Path to evaluator config')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--headless', action='store_true', default=True, help='Run headless')
    parser.add_argument('--no_headless', action='store_false', dest='headless', help='Show GUI')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing results')
    parser.add_argument('--seed', type=int, default=2, help='Random seed')
    args = parser.parse_args()
    
    filter_grasps_for_scene(
        scene_id=args.scene_id,
        robot_name=args.robot_name,
        input_base=args.input_base,
        output_base=args.output_base,
        evaluator_config_path=args.evaluator_config,
        device=args.device,
        batch_size=args.batch_size,
        headless=args.headless,
        overwrite=args.overwrite,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
