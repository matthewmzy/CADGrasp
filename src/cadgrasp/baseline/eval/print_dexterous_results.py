import os

import argparse
import numpy as np
import plotly.graph_objects as go


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, 
        default='experiments/dex_ours/ckpt/ckpt_50000.pth')
    args = parser.parse_args()

    for dataset in ['graspnet', 'acronym']:
        args.dataset = dataset

        result_path = os.path.join(os.path.dirname(os.path.dirname(args.ckpt_path)), 'results') if args.dataset == 'graspnet' else os.path.join(os.path.dirname(os.path.dirname(args.ckpt_path)), 'results_acronym')
        if args.dataset == 'graspnet':
            scene_id_list = [f'scene_{str(i).zfill(4)}' for i in range(100,190)] + [f'scene_{str(i).zfill(4)}' for i in range(200,380)] + [f'scene_{str(i).zfill(4)}' for i in range(9000, 9900, 5)]
            dense_list = [f'scene_{str(i).zfill(4)}' for i in range(100, 190)]
            loose_list = [f'scene_{str(i).zfill(4)}' for i in range(200, 380)]
            loose_list = [f for f in loose_list if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(args.ckpt_path)), 'results', f,'sim_success.npy'))]
            random_list = [f'scene_{str(i).zfill(4)}' for i in range(9000, 9900, 5)]
            random_list = [f for f in random_list if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(args.ckpt_path)), 'results', f))]
        
        if args.dataset == 'acronym':
            scene_id_list = sorted(os.listdir(os.path.join(os.path.dirname(os.path.dirname(args.ckpt_path)), 'results_acronym')))
            dense_list = [f for f in scene_id_list if 'dense' in f]
            random_list = [f for f in scene_id_list if 'random' in f]
            loose_list = [f for f in scene_id_list if 'loose' in f]
            dense_list = [f for f in dense_list if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(args.ckpt_path)), 'results_acronym',f,'sim_success.npy'))]

        success_rate_dict = {}
        for scene_id in scene_id_list:
            try:
                save_path = os.path.join(result_path, scene_id, 'sim_success.npy')
                sim_success = np.load(save_path)
                success_rate_dict[scene_id] = sim_success.mean().item()
            except: # skip scenes where object poses are out of camera coverage.
                # print(scene_id)
                pass

        average_success_rate = np.mean(list(success_rate_dict.values()))
        print(f'Average {dataset} success rate: {average_success_rate:.3f}')
        # for scene_list_name, scene_list in zip(['seen', 'similar', 'novel', 'dense', 'random', 'loose'], [seen_list, similar_list, novel_list, dense_list, random_list, loose_list]):
        for scene_list_name, scene_list in zip(['dense', 'random', 'loose'], [dense_list, random_list, loose_list]):
            try:
                success_rate = np.mean([success_rate_dict[scene_id] for scene_id in scene_list])
                print(f'{args.dataset} {scene_list_name} success rate: {success_rate:.3f}')
            except:
                pass