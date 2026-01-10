import os
import sys

import argparse
import multiprocessing
from rich.progress import track

def evaluate_network(scene_id):
    
    # get worker id
    worker = multiprocessing.current_process()._identity[0]
    gpu = args.gpu_list[worker - 1]
    
    # command line
    command = ' '.join([
        'python src/eval/evaluate_dexterous.py',
        f'--ckpt_path_list {" ".join(args.ckpt_path_list)}',
        f'--device cuda:{gpu}',
        f'--robot_name {args.robot_name}',
        f'--scene_id {scene_id}',
        f'--seed {args.seed}',
        f'--evaluator {args.evaluator}',
        f'--headless 1',
        f'--batch_size {args.batch_size}',
        f'--overwrite {args.overwrite}',
        f'--dataset {args.dataset}',
        f'--split {args.split}',
        f'--strategy {args.strategy}',
        # ' > /dev/null 2>&1',
    ])
    
    # execute command
    os.system(command)
    
def gen_split_dict(args):
    test_scene_splits = {}
    test_graspnet = {
        'debug':[f'scene_{str(i).zfill(4)}' for i in range(100, 101)],
        'seen':[f'scene_{str(i).zfill(4)}' for i in range(100, 130)],
        'similar':[f'scene_{str(i).zfill(4)}' for i in range(130, 160)],
        'novel':[f'scene_{str(i).zfill(4)}' for i in range(160, 190)],
        'dense':[f'scene_{str(i).zfill(4)}' for i in range(100, 190)],
        'random':[f'scene_{str(i).zfill(4)}' for i in range(9000, 9900,5)], # Not synthesized yet
        'loose':[f'scene_{str(i).zfill(4)}' for i in range(200, 380)],
    }
    test_acronym = {
        'dense': [f'scene_dense_{i}' for i in range(100)],
        'random': [f'scene_random_{i}' for i in range(90)],
        'loose': [f'scene_loose_{i}' for i in range(30)],
    }
    test_scene_splits['graspnet'] = test_graspnet
    test_scene_splits['acronym'] = test_acronym
    
    return test_scene_splits


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input list to save setup time
    
    parser.add_argument('--ckpt_path_list', type=str, nargs='*', 
        default=[
            'experiments/dex_ours/ckpt/ckpt_50000.pth',
            'experiments/exp_dex_ours_final/ckpt/ckpt_50000.pth',
            'experiments/exp_dex_isagrasp_final/ckpt/ckpt_50000.pth',
            'experiments/exp_dex_grasptta_final/ckpt/ckpt_50000.pth',
            ])
    parser.add_argument('--gpu_list', type=int, nargs='+', default=[0,1,2,3,4,5,6,7])
    parser.add_argument('--robot_name', type=str,
        default='leap_hand', choices=['leap_hand'])
    parser.add_argument('--dataset', type=str,
        default='acronym', choices=['graspnet', 'acronym'])
    parser.add_argument('--split', type=str,
        default='dense', choices=['train', 'dense', 'random', 'loose', 'debug'])
    parser.add_argument('--seed', type=int, 
        default=0)
    parser.add_argument('--evaluator', type=str, default='SimulationEvaluator', 
        choices=['SimulationEvaluator'])
    parser.add_argument('--batch_size', type=int, default=100)  # >100 may cause blow up
    parser.add_argument('--overwrite', type=int, default=1)
    parser.add_argument('--strategy', type=str,
        default='ours', choices=['ours', 'top10', 'graspness', 'logprob', 'random'])
    args = parser.parse_args()
    
    # get scene id list
    test_scene_splits = gen_split_dict(args)
    scene_id_list = test_scene_splits[args.dataset][args.split]  
    
    # compose scenes in parallel
    with multiprocessing.Pool(len(args.gpu_list)) as pool:
        it = track(
            pool.imap_unordered(evaluate_network, scene_id_list), 
            total=len(scene_id_list), 
            description='evaluating', 
        )
        list(it)










