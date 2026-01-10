"""
Annotate IBS Data for Camera Views.

This script generates per-view valid IBS indices based on visibility
from each camera viewpoint. It is a PREPROCESSING step required before
training the LASDiffusion model.

Purpose:
    For each (scene, view) pair, determine which IBS data points are "visible"
    from that view by checking if the grasp point is near the observed scene 
    point cloud. This enables view-dependent training.

When to use:
    - Run AFTER Stage 3 (IBS calculation) and BEFORE LASDiffusion training
    - Required if training LASDiffusion with view-based filtering
    - Not required for inference/evaluation

Output:
    data/ibsdata/scene_valid_ids/scene_XXXX/view_YYYY.npy
    Each file contains a boolean array of shape (N,) where N is the number
    of IBS samples for that scene.

Alternative:
    If you don't want view-based filtering during training, you can set 
    `use_view_filter=False` in IBS_Dataset. This will use all IBS data
    regardless of view, but may result in training on IBS that are not
    visible from the input view.

Usage:
    # Single GPU
    python annotate_ibs_for_view.py --scene_start 0 --scene_end 100 --gpu_ids 0
    
    # Multi-GPU (parallel processing)
    python annotate_ibs_for_view.py --scene_start 0 --scene_end 100 --gpu_ids 0,1,2,3
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from loguru import logger
from torch.multiprocessing import set_start_method, Pool, current_process

try:
    set_start_method('spawn')
except RuntimeError:
    pass

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
np.seterr(all='raise')


# ============ Default Paths ============
DEFAULT_IBS_PATH = 'data/ibsdata'
DEFAULT_SCENE_PATH = 'data/DexGraspNet2.0/scenes'
DEFAULT_OUTPUT_PATH = 'data/ibsdata/scene_valid_ids'

# ============ Global Config (set by main) ============
CONFIG = {
    'ibs_path': DEFAULT_IBS_PATH,
    'scene_path': DEFAULT_SCENE_PATH,
    'output_path': DEFAULT_OUTPUT_PATH,
    'scene_ids': list(range(100)),
    'view_ids': list(range(256)),
    'gpu_list': ['0'],
    'distance_threshold': 0.01,  # 1cm - max distance to consider grasp visible
    'enable_vis': False,
}


def transform_points(pc: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    """Transform points by 4x4 matrix."""
    homo = torch.cat((pc, torch.ones((pc.shape[0], 1), device=pc.device)), dim=1)
    homo = homo @ trans.T
    return homo[:, :3] / homo[:, 3:4]


def process_single_scene(scene_id: int):
    """Process all views for a single scene."""
    try:
        worker_id = current_process()._identity[0] if current_process()._identity else 0
        device = f"cuda:{CONFIG['gpu_list'][(worker_id - 1) % len(CONFIG['gpu_list'])]}"
        
        scene_name = f'scene_{str(scene_id).zfill(4)}'
        
        # Load IBS data
        ibs_file = os.path.join(CONFIG['ibs_path'], 'ibs', f'{scene_name}.npy')
        w2h_file = os.path.join(CONFIG['ibs_path'], 'w2h_trans', f'{scene_name}.npy')
        
        if not os.path.exists(ibs_file):
            logger.warning(f"IBS data not found for {scene_name}")
            return
        
        w2h_trans = np.load(w2h_file)
        # Grasp points are at origin in hand frame, so world position is inverse transform origin
        grasp_points = np.linalg.inv(w2h_trans)[:, :3, 3]  # (N, 3)
        grasp_points = torch.from_numpy(grasp_points).float().to(device)
        
        # Create output directory
        output_dir = os.path.join(CONFIG['output_path'], scene_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Load scene network input (contains point clouds for all views)
        scene_data_path = os.path.join(
            CONFIG['scene_path'], scene_name, 'realsense', 'network_input.npz'
        )
        if not os.path.exists(scene_data_path):
            logger.warning(f"Scene data not found: {scene_data_path}")
            return
            
        scene_data = np.load(scene_data_path)
        
        for view_id in tqdm(CONFIG['view_ids'], desc=f'{scene_name}', leave=False):
            output_file = os.path.join(output_dir, f'view_{str(view_id).zfill(4)}.npy')
            
            if os.path.exists(output_file):
                continue
            
            # Get scene point cloud for this view
            scene_pc_c = scene_data['pc'][view_id]  # (N, 3) in camera frame
            c2w_trans = scene_data['extrinsics'][view_id]  # (4, 4)
            
            scene_pc_c = torch.from_numpy(scene_pc_c).float().to(device)
            c2w_trans = torch.from_numpy(c2w_trans).float().to(device)
            
            # Transform to world frame
            scene_pc_w = transform_points(scene_pc_c, c2w_trans)
            
            # Compute distance from each grasp point to nearest scene point
            dist = torch.cdist(grasp_points, scene_pc_w)  # (N_grasps, N_scene)
            min_dist = torch.min(dist, dim=1)[0]  # (N_grasps,)
            
            # Grasp is valid if within threshold distance
            valid_mask = (min_dist < CONFIG['distance_threshold']).cpu().numpy()
            
            np.save(output_file, valid_mask)
        
        logger.info(f"Done: {scene_name}")
        
    except Exception as e:
        logger.error(f"Error processing scene {scene_id}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Annotate IBS for views (preprocessing for LASDiffusion training)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process scenes 0-99 with 4 GPUs
    python annotate_ibs_for_view.py --scene_start 0 --scene_end 100 --gpu_ids 0,1,2,3
    
    # Process scenes 100-190 with single GPU
    python annotate_ibs_for_view.py --scene_start 100 --scene_end 190 --gpu_ids 0
        """
    )
    parser.add_argument('--ibs_path', type=str, default=DEFAULT_IBS_PATH,
                        help='Path to IBS data directory')
    parser.add_argument('--scene_path', type=str, default=DEFAULT_SCENE_PATH,
                        help='Path to scene data')
    parser.add_argument('--output_path', type=str, default=DEFAULT_OUTPUT_PATH,
                        help='Path to save valid IDs')
    parser.add_argument('--scene_start', type=int, default=0, 
                        help='Start scene ID')
    parser.add_argument('--scene_end', type=int, default=100, 
                        help='End scene ID (exclusive)')
    parser.add_argument('--view_start', type=int, default=0, 
                        help='Start view ID')
    parser.add_argument('--view_end', type=int, default=256, 
                        help='End view ID (exclusive)')
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='Comma-separated GPU IDs for parallel processing')
    parser.add_argument('--distance_threshold', type=float, default=0.01,
                        help='Max distance (m) for grasp to be considered visible')
    parser.add_argument('--visualize', action='store_true', 
                        help='Enable visualization (debug)')
    
    args = parser.parse_args()
    
    # Update global config
    CONFIG['ibs_path'] = args.ibs_path
    CONFIG['scene_path'] = args.scene_path
    CONFIG['output_path'] = args.output_path
    CONFIG['scene_ids'] = list(range(args.scene_start, args.scene_end))
    CONFIG['view_ids'] = list(range(args.view_start, args.view_end))
    CONFIG['gpu_list'] = [x.strip() for x in args.gpu_ids.split(',')]
    CONFIG['distance_threshold'] = args.distance_threshold
    CONFIG['enable_vis'] = args.visualize
    
    n_gpus = len(CONFIG['gpu_list'])
    n_scenes = len(CONFIG['scene_ids'])
    
    print(f"Annotating IBS for views:")
    print(f"  Scenes: {args.scene_start} - {args.scene_end - 1} ({n_scenes} scenes)")
    print(f"  Views: {args.view_start} - {args.view_end - 1}")
    print(f"  GPUs: {CONFIG['gpu_list']}")
    print(f"  Output: {args.output_path}")
    
    if n_gpus > 1:
        with Pool(n_gpus) as pool:
            list(tqdm(
                pool.imap_unordered(process_single_scene, CONFIG['scene_ids'], chunksize=1),
                total=n_scenes,
                desc='Processing scenes'
            ))
    else:
        for scene_id in tqdm(CONFIG['scene_ids'], desc='Processing scenes'):
            process_single_scene(scene_id)
    
    print(f"\nDone! Valid IDs saved to: {args.output_path}")


if __name__ == "__main__":
    main()