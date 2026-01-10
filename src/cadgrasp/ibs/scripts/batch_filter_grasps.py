"""
Batch script to filter successful grasps using IsaacGym simulation.

This script processes multiple scenes in parallel using multiple GPUs.
It reads grasp data from data/DexGraspNet2.0/dex_grasps_new and saves
success indices to data/DexGraspNet2.0/dex_grasps_success_indices.

Usage:
    python src/cadgrasp/ibs/scripts/batch_filter_grasps.py --scene_start 0 --scene_end 100

Configuration:
    - Use --num_gpus to specify number of GPUs to use
    - Use --gpu_ids to specify specific GPU IDs (e.g., "0,1,2,3")
"""

import os
import sys
import argparse
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from loguru import logger
from torch.multiprocessing import set_start_method, Pool, current_process
from glob import glob

try:
    set_start_method('spawn')
except RuntimeError:
    pass

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
np.seterr(all='raise')


def get_scene_ids_with_grasps(
    base_path: str = 'data/DexGraspNet2.0/dex_grasps_new',
    scene_start: int = 0,
    scene_end: int = 10000
):
    """Get all scene IDs that have grasp data within the specified range."""
    scene_ids = []
    
    for scene_num in range(scene_start, scene_end):
        scene_id = f'scene_{str(scene_num).zfill(4)}'
        scene_dir = os.path.join(base_path, scene_id)
        
        if os.path.exists(scene_dir):
            # Check if there are any grasp files
            grasp_files = glob(os.path.join(scene_dir, '*/*.npz'))
            if len(grasp_files) > 0:
                scene_ids.append(scene_id)
    
    return scene_ids


# Global config for worker processes
_worker_config = {}


def init_worker(config):
    """Initialize worker with shared config."""
    global _worker_config
    _worker_config = config


def process_scene(scene_id: str):
    """Process a single scene."""
    global _worker_config
    
    try:
        # Import here to avoid issues with multiprocessing
        from cadgrasp.ibs.scripts.filter_grasps_by_sim import filter_grasps_for_scene
        
        worker = current_process()._identity[0]
        gpu_ids = _worker_config['gpu_ids']
        device = f"cuda:{gpu_ids[(worker - 1) % len(gpu_ids)]}"
        
        filter_grasps_for_scene(
            scene_id=scene_id,
            robot_name=_worker_config['robot_name'],
            input_base=_worker_config['input_base'],
            output_base=_worker_config['output_base'],
            evaluator_config_path=_worker_config['evaluator_config'],
            device=device,
            batch_size=_worker_config['batch_size'],
            headless=_worker_config['headless'],
            overwrite=_worker_config['overwrite'],
            seed=_worker_config['seed']
        )
        
        logger.info(f"Done for {scene_id}")
        return {'scene_id': scene_id, 'status': 'success'}
        
    except Exception as e:
        logger.error(f"Error processing {scene_id}: {e}")
        import traceback
        traceback.print_exc()
        return {'scene_id': scene_id, 'status': 'error', 'error': str(e)}


def main():
    """Main function to batch process scenes."""
    parser = argparse.ArgumentParser(description='Batch filter successful grasps')
    parser.add_argument('--scene_start', type=int, default=0, help='Start scene ID')
    parser.add_argument('--scene_end', type=int, default=10000, help='End scene ID')
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
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3,4,5,6,7',
                        help='Comma-separated GPU IDs to use')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--headless', action='store_true', default=True,
                        help='Run headless')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing')
    parser.add_argument('--seed', type=int, default=2, help='Random seed')
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    
    # Get all scene IDs with grasp data
    all_scene_ids = get_scene_ids_with_grasps(
        args.input_base, args.scene_start, args.scene_end
    )
    logger.info(f"Found {len(all_scene_ids)} scenes with grasp data")
    
    # Filter to only include scenes that haven't been processed
    scene_ids_to_process = []
    
    for scene_id in all_scene_ids:
        input_path = os.path.join(args.input_base, scene_id, args.robot_name)
        output_path = os.path.join(args.output_base, scene_id, args.robot_name)
        
        if not os.path.exists(input_path):
            continue
            
        # Check if all objects have been processed
        input_files = glob(os.path.join(input_path, '*.npz'))
        if len(input_files) == 0:
            continue
            
        all_processed = True
        for input_file in input_files:
            obj_id = os.path.basename(input_file)
            output_file = os.path.join(output_path, obj_id)
            if not os.path.exists(output_file):
                all_processed = False
                break
        
        if not all_processed or args.overwrite:
            scene_ids_to_process.append(scene_id)
    
    logger.info(f"Will process {len(scene_ids_to_process)} scenes")
    
    if len(scene_ids_to_process) == 0:
        logger.info("All scenes already processed!")
        return
    
    # Create worker config
    worker_config = {
        'gpu_ids': gpu_ids,
        'robot_name': args.robot_name,
        'input_base': args.input_base,
        'output_base': args.output_base,
        'evaluator_config': args.evaluator_config,
        'batch_size': args.batch_size,
        'headless': args.headless,
        'overwrite': args.overwrite,
        'seed': args.seed,
    }
    
    # Process in parallel
    num_workers = min(len(gpu_ids), len(scene_ids_to_process))
    logger.info(f"Using {num_workers} workers on GPUs: {gpu_ids[:num_workers]}")
    
    with Pool(num_workers, initializer=init_worker, initargs=(worker_config,)) as p:
        results = list(tqdm(
            p.imap_unordered(process_scene, scene_ids_to_process, chunksize=1),
            total=len(scene_ids_to_process),
            desc='Filtering grasps'
        ))
    
    # Summary
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = sum(1 for r in results if r['status'] == 'error')
    
    logger.info(f"\nBatch processing complete!")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Errors: {error_count}")


if __name__ == "__main__":
    main()
