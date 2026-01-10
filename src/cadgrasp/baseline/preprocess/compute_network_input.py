import os
import sys

import argparse
import numpy as np
from rich.progress import track

from cadgrasp.baseline.utils.util import set_seed
from cadgrasp.baseline.utils.vis_plotly import Vis
from cadgrasp.paths import project_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_name', type=str, 
        default='leap_hand', choices=['leap_hand'])
    parser.add_argument('--dataset', type=str, 
        default='graspnet', choices=['graspnet', 'acronym'])
    parser.add_argument('--urdf_path', type=str, 
        default=project_path('robot_models/urdf/leap_hand.urdf'))
    parser.add_argument('--meta_path', type=str, 
        default=project_path('robot_models/meta/leap_hand/meta.yaml'))
    parser.add_argument('--camera', type=str, 
        default='realsense')
    parser.add_argument('--scene_id', type=str, 
        default='scene_0000')
    parser.add_argument('--seed', type=int, 
        default=0)
    parser.add_argument('--overwrite', type=int, default=0)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    if args.dataset == 'graspnet':
        # not overwrite
        save_path = path = project_path('data/DexGraspNet2.0/scenes', args.scene_id, args.camera, 'network_input.npz')
    elif args.dataset == 'acronym':
        if args.scene_id.split('_')[1] == 'dense':
            save_path = path = os.path.join('data/acronym_test_scenes/network_input_dense', args.scene_id, args.camera, 'network_input.npz')
        elif args.scene_id.split('_')[1] == 'random':
            save_path = path = os.path.join('data/acronym_test_scenes/network_input_random', args.scene_id, args.camera, 'network_input.npz')
        else:
            save_path = path = os.path.join('data/acronym_test_scenes/network_input_loose', args.scene_id, args.camera, 'network_input.npz')
    if os.path.exists(save_path) and not args.overwrite:
        quit()
    
    # scene loader
    vis = Vis(
        robot_name=args.robot_name,
        urdf_path=args.urdf_path,
        meta_path=args.meta_path,
    )
    
    view_id_list = [str(i).zfill(4) for i in range(256)]
    
    # load scene pc
    pc_all = []
    seg_all = []
    extrinsics_all = []
    edge_all = []
    for view_id in track(view_id_list, 'loading'):
        if args.dataset == 'graspnet':
            _, pc, extrinsics = vis.scene_plotly(args.scene_id, view_id, args.camera, 
            with_pc=True, with_extrinsics=True, mode='pc', num_points=40000)
        elif args.dataset == 'acronym':
             _, pc, extrinsics = vis.acronym_scene_plotly_test(args.scene_id, view_id, args.camera, 
            with_pc=True, with_extrinsics=True, mode='pc', num_points=40000)
            #  vis.show(vis.pc_plotly(pc[:, :3]))
        seg = pc[:, 3].int()
        edge = pc[:, 4].int()
        pc = pc[:, :3].float()
        pc_all.append(pc.numpy())
        seg_all.append(seg.numpy())
        extrinsics_all.append(extrinsics)
        edge_all.append(edge.numpy())
    pc_all = np.stack(pc_all)
    seg_all = np.stack(seg_all)
    extrinsics_all = np.stack(extrinsics_all)
    edge_all = np.stack(edge_all)
    
    # save network input
    network_input = {
        'pc': pc_all,
        'seg': seg_all,
        'extrinsics': extrinsics_all,
        'edge': edge_all,
    }
    np.savez(save_path, **network_input)
