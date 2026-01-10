import os
import sys

import argparse
import multiprocessing
from rich.progress import track
from cadgrasp.paths import project_path

def compute_network_input(scene_id):
    
    # command line
    command = ' '.join([
        'python -m cadgrasp.baseline.preprocess.compute_network_input',
        f'--robot_name {args.robot_name}',
        f'--urdf_path {args.urdf_path}',
        f'--meta_path {args.meta_path}',
        f'--camera {args.camera}',
        f'--scene_id {scene_id}',
        f'--seed {args.seed}',
        f'--overwrite {args.overwrite}',
        f'--dataset {args.dataset}',
        # ' > /dev/null 2>&1',
    ])
    
    # execute command
    os.system(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cpu', type=int, default=60)
    parser.add_argument('--robot_name', type=str,
        default='leap_hand', choices=['leap_hand'])
    parser.add_argument('--dataset', type=str, 
        default='acronym', choices=['graspnet', 'acronym'])
    parser.add_argument('--urdf_path', type=str, 
        default=project_path('robot_models/urdf/leap_hand.urdf'))
    parser.add_argument('--meta_path', type=str, 
        default=project_path('robot_models/meta/leap_hand/meta.yaml'))
    parser.add_argument('--camera', type=str, 
        default='realsense')
    parser.add_argument('--scene_id_start', type=int, default=200)
    parser.add_argument('--scene_id_end', type=int, default=380)
    parser.add_argument('--seed', type=int, 
        default=0)
    parser.add_argument('--overwrite', type=int, default=0)
    args = parser.parse_args()
    
    # get scene id list
    if args.dataset == 'graspnet':
        scene_id_list = [f'scene_{str(i).zfill(4)}' for i in range(args.scene_id_start, args.scene_id_end)]
        if args.scene_id_start > 8500:
            scene_id_list = [f'scene_{str(i).zfill(4)}' for i in range(args.scene_id_start, args.scene_id_end,5)]
    elif args.dataset == 'acronym':
        scene_id_list = []
        scene_id_list += [f'scene_dense_{i}' for i in range(100)]
        scene_id_list += [f'scene_random_{i}' for i in range(90)]
        scene_id_list += [f'scene_loose_{i}' for i in range(30)]
    
    # compose scenes in parallel
    with multiprocessing.Pool(args.n_cpu) as pool:
        it = track(
            pool.imap_unordered(compute_network_input, scene_id_list), 
            total=len(scene_id_list), 
            description='computing', 
        )
        list(it)
