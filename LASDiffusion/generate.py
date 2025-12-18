import torch
import numpy as np
import os, sys
sys.path.append('.')
from LASDiffusion.network.model_trainer import DiffusionModel
from LASDiffusion.utils.utils import str2bool, ensure_directory
from LASDiffusion.utils.visualize_ibs_vox import devoxelize
import argparse
import plotly.graph_objects as go
from tqdm import tqdm
from termcolor import cprint
from LASDiffusion.utils.transforms import transform_points, transform_points, pc_plotly, show


def voxelize_scene_pc(scene_pcs, hand_pose):
    """
    scene_pc : (B,N,3)
    hand_pose: (B,4,4)
    """
    scene_pcs = [transform_points(scene_pcs[i][:,:3], torch.inverse(hand_pose[i])) for i in range(len(scene_pcs))]
    x_min, x_max = -0.1, 0.1
    y_min, y_max = -0.1, 0.1
    z_min, z_max = -0.1, 0.1
    resolution = 0.005
    scene_voxels = []
    for scene_pc in scene_pcs:
        mask = (scene_pc[:, 0] > x_min) & (scene_pc[:, 0] < x_max) & (scene_pc[:, 1] > y_min) & (scene_pc[:, 1] < y_max) & (scene_pc[:, 2] > z_min) & (scene_pc[:, 2] < z_max)
        scene_pc = scene_pc[mask]
        scene_pc = (scene_pc - torch.tensor([x_min, y_min, z_min], dtype=scene_pc.dtype, device=scene_pc.device)) // resolution
        scene_pc = scene_pc.to(dtype=torch.long)
        scene_voxel_float = torch.zeros([int((x_max-x_min)/resolution), int((y_max-y_min)/resolution), int((z_max-z_min)/resolution)], dtype=torch.float, device='cuda:0')
        scene_voxel_float[scene_pc[:, 0], scene_pc[:, 1], scene_pc[:, 2]] = 1
        scene_voxels.append(scene_voxel_float.unsqueeze(0))
    return torch.cat(scene_voxels, dim=0)

def generate_based_on_scene_obs(
    model_path: str,
    output_path: str = "./outputs",
    ema: bool = True,
    steps: int = 50,
    truncated_time: float = 0.0,
    hand_pose = None,
    scene_pc = None,
    stride_size = 64,
):
    model_name, model_id = model_path.split('/')[-2], model_path.split('/')[-1]
    discrete_diffusion = DiffusionModel.load_from_checkpoint(model_path).cuda()
    postfix = f"{model_name}_{model_id}_{ema}_{steps}_{truncated_time}_conditional"
    root_dir = os.path.join(output_path, postfix)

    ensure_directory(root_dir)

    if scene_pc is None:
        test_dataloader = discrete_diffusion.test_dataloader()
        batch = next(iter(test_dataloader))
        ibs_ref_batch = batch["ibs"].reshape(-1,2,40,40,40)
        scene_voxel_batch = batch["scene"].reshape(-1,1,40,40,40)
        batch_size = ibs_ref_batch.shape[0]
    else:
        ibs_ref_batch = None
        scene_voxel_batch = voxelize_scene_pc(scene_pc, hand_pose.float()).unsqueeze(1)
        batch_size = scene_voxel_batch.shape[0]

    generator = discrete_diffusion.ema_model if ema else discrete_diffusion.model
    res_tensors = []
    for i in range(0, batch_size, stride_size):
        stride_size = min(stride_size, batch_size-i)
        res_tensors.append(generator.sample_based_on_scene(batch_size=stride_size, scene_voxel = scene_voxel_batch[i:i+stride_size], steps=steps, truncated_index=truncated_time))
    res_tensor = torch.cat(res_tensors, dim=0)
    # import pdb; pdb.set_trace()
    res_vox = res_tensor.cpu().numpy()
    # cprint(f"{np.mean(res_vox[:,0])}, {np.min(res_vox[:,0])}, {np.max(res_vox[:,0])}", color='green')
    # cprint(f"{np.mean(res_vox[:,1])}, {np.min(res_vox[:,1])}, {np.max(res_vox[:,1])}", color='green')
    ret_vox = np.zeros((batch_size,40,40,40,3))
    ret_vox[...,0][res_vox[:,0]>0] = 1
    ret_vox[...,1][np.logical_and(res_vox[:,1]>0.5,res_vox[:,1]<1.5)] = 1
    ret_vox[...,2][res_vox[:,1]>1.5] = 1
    return ret_vox

def pc_plotly(pc, size=3, color='green'):
    return go.Scatter3d(
                x=pc[:, 0] if isinstance(pc, np.ndarray) else pc[:, 0].numpy(),
                y=pc[:, 1] if isinstance(pc, np.ndarray) else pc[:, 1].numpy(),
                z=pc[:, 2] if isinstance(pc, np.ndarray) else pc[:, 2].numpy(),
                mode='markers',
                marker=dict(size=size, color=color),
            )

def generate_debug():
    test_model = DiffusionModel(ibs_path = "IBSGrasp/ibsdata",
                                scene_pc_path = "data/scenes",
                                ibs_load_per_scene=1)
    test_dataloader = test_model.test_dataloader()
    batch = next(iter(test_dataloader))
    ibs_ref_batch = batch["ibs"].reshape(-1,2,40,40,40)
    scene_voxel_batch = batch["scene"].reshape(-1,1,40,40,40)
    batch_size = ibs_ref_batch.shape[0]
    res_vox = ibs_ref_batch.cpu().numpy()
    scene_voxel_batch = scene_voxel_batch.cpu().numpy()
    ret_vox = np.zeros((batch_size,40,40,40,3))
    ret_vox[...,0][res_vox[:,0]>0] = 1
    ret_vox[...,1][res_vox[:,1]>0.5] = 1
    ret_vox[...,2][np.logical_and(res_vox[:,1]<0.5,res_vox[:,1]>-0.5)] = 1
    print(np.sum(ret_vox[...,0]), np.sum(ret_vox[...,1]), np.sum(ret_vox[...,2]))
    vis_cnt = 0
    for ibs, scene_pc in zip(ret_vox, scene_voxel_batch):
        scene_pc = scene_pc[0][...,None]
        pad = np.zeros_like(scene_pc).repeat(2,axis=-1)
        scene_pc = np.concatenate([scene_pc, pad], axis=-1)
        oc_pc, co_pc, th_pc = devoxelize(ibs)
        sc_pc, _, _ = devoxelize(scene_pc)
        show([
            pc_plotly(oc_pc, color='red'),
            pc_plotly(co_pc, color='yellow'),
            pc_plotly(th_pc, color='blue'),
            pc_plotly(sc_pc, color='green'),
        ])
        vis_cnt+=1
        if vis_cnt==3:break


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='generate something')
    parser.add_argument("--generate_method", type=str, default='debug',
                        help="please choose :\n \
                            1. 'generate_unconditional' \n \
                            2. 'generate_based_on_scene' \n \
                            3. 'debug' \n ")

    parser.add_argument("--model_path", type=str, default="LASDiffusion/results/LEAP_dif/recent/last.ckpt")
    parser.add_argument("--output_path", type=str, default="LASDiffusion/outputs")
    parser.add_argument("--ema", type=str2bool, default=True)
    parser.add_argument("--num_generate", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--truncated_time", type=float, default=0.0)


    args = parser.parse_args()
    method = (args.generate_method).lower()
    ensure_directory(args.output_path)
    if method == "generate_based_on_scene":
        generate_based_on_scene_obs(model_path=args.model_path,
                                    steps=args.steps,
                                    output_path=args.output_path, ema=args.ema, #start_index=args.start_index, 
                               
                               truncated_time=args.truncated_time)
    elif method == 'debug':
        generate_debug()
    else:
        raise NotImplementedError
