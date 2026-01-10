"""
CADGrasp Prediction Pipeline for Dexterous Grasping.

This script implements the CADGrasp prediction pipeline:
1. Use DexGraspNet2.0 checkpoint to predict grasp points and rotations
2. Compose w2h_trans from grasp point + rotation
3. Select top_n candidates per view based on score
4. Crop and voxelize scene point cloud in hand frame
5. Use LASDiffusion to predict IBS voxels
6. Filter IBS with IBSFilter (force-closure check)
7. Optimize hand pose with AdamOpt
8. Save results

Usage:
    python -m cadgrasp.baseline.eval.predict_dexterous \
        --ckpt_path data/DexGraspNet2.0/DexGraspNet2.0-ckpts/CAD/ckpt/ckpt_50000.pth \
        --las_exp_name LEAP_dif \
        --scene_id scene_0100 \
        --dataset graspnet
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import trange, tqdm
from termcolor import cprint
from torch.utils.tensorboard import SummaryWriter

# Baseline imports
from cadgrasp.baseline.utils.robot_model import RobotModel
from cadgrasp.baseline.utils.util import set_seed
from cadgrasp.baseline.utils.config import ckpt_to_config
from cadgrasp.baseline.utils.dataset import get_sparse_tensor
from cadgrasp.baseline.network.model import get_model

# CADGrasp imports
from cadgrasp.optimizer.AdamOpt import AdamOptimizer
from cadgrasp.optimizer.ibs_func import IBSFilter
from cadgrasp.optimizer.rot6d import robust_compute_rotation_matrix_from_ortho6d

# LASDiffusion imports
from LASDiffusion.network.model_trainer import DiffusionModel
from LASDiffusion.generate import voxelize_scene_pc, decode_ibs_voxel


def compose_w2h_trans(rotations: torch.Tensor, translations: torch.Tensor) -> torch.Tensor:
    """
    Compose world-to-hand transformation matrices from rotation and translation.
    
    Args:
        rotations: (B, K, 3, 3) rotation matrices
        translations: (B, K, 3) translation vectors (grasp points)
    
    Returns:
        w2h_trans: (B, K, 4, 4) transformation matrices
    """
    B, K = rotations.shape[:2]
    device = rotations.device
    
    # Create 4x4 transformation matrices
    w2h_trans = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(B, K, 1, 1)
    w2h_trans[:, :, :3, :3] = rotations.transpose(-1, -2)  # Inverse rotation
    w2h_trans[:, :, :3, 3] = -torch.einsum('bkij,bkj->bki', rotations.transpose(-1, -2), translations)
    
    return w2h_trans


def select_top_n_per_view(
    scores: torch.Tensor,
    rotations: torch.Tensor,
    translations: torch.Tensor,
    qposs: torch.Tensor,
    grasp_points: torch.Tensor,
    top_n: int
) -> tuple:
    """
    Select top_n candidates per view based on score.
    
    Args:
        scores: (B, K) scores
        rotations: (B, K, 3, 3)
        translations: (B, K, 3)
        qposs: (B, K, J)
        grasp_points: (B, K, 3) grasp points (seed points)
        top_n: number of candidates to select per view
    
    Returns:
        Tuple of selected tensors, each with shape (B, top_n, ...)
    """
    B, K = scores.shape
    
    # Get top_n indices per view
    _, top_indices = torch.topk(scores, min(top_n, K), dim=1)  # (B, top_n)
    
    # Gather selected candidates
    batch_idx = torch.arange(B, device=scores.device).unsqueeze(1).expand(-1, top_n)
    
    sel_rotations = rotations[batch_idx, top_indices]  # (B, top_n, 3, 3)
    sel_translations = translations[batch_idx, top_indices]  # (B, top_n, 3)
    sel_qposs = qposs[batch_idx, top_indices]  # (B, top_n, J)
    sel_scores = scores[batch_idx, top_indices]  # (B, top_n)
    sel_grasp_points = grasp_points[batch_idx, top_indices]  # (B, top_n, 3)
    
    return sel_rotations, sel_translations, sel_qposs, sel_scores, sel_grasp_points


def predict_ibs_batch(
    model: DiffusionModel,
    scene_pcs: torch.Tensor,
    w2h_trans: torch.Tensor,
    device: str,
    stride_size: int = 64,
    steps: int = 50,
    ema: bool = True
) -> np.ndarray:
    """
    Predict IBS voxels using LASDiffusion.
    
    Args:
        model: Loaded LASDiffusion model
        scene_pcs: (B, N, 3) scene point clouds in camera frame
        w2h_trans: (B, 4, 4) world-to-hand transformations
        device: torch device
        stride_size: batch size for generation
        steps: number of diffusion steps
        ema: whether to use EMA model
    
    Returns:
        ibs_voxels: (B, 2, 40, 40, 40) IBS voxels in network format
    """
    # Voxelize scene in hand frame
    # Note: w2h_trans transforms from world/camera to hand frame
    # voxelize_scene_pc expects hand_pose which is the inverse (hand-to-world)
    hand_pose = torch.inverse(w2h_trans)
    scene_voxels = voxelize_scene_pc(scene_pcs, hand_pose.float(), device)
    
    batch_size = scene_voxels.shape[0]
    generator = model.ema_model if ema else model.model
    
    res_tensors = []
    with torch.no_grad():
        for i in range(0, batch_size, stride_size):
            curr_stride = min(stride_size, batch_size - i)
            res = generator.sample_based_on_scene(
                batch_size=curr_stride,
                scene_voxel=scene_voxels[i:i+curr_stride],
                steps=steps,
                truncated_index=0.0
            )
            res_tensors.append(res)
    
    res_tensor = torch.cat(res_tensors, dim=0)
    return res_tensor.cpu().numpy()


def transform_pose_to_world(
    rotations: np.ndarray,
    translations: np.ndarray,
    extrinsics: np.ndarray
) -> tuple:
    """
    Transform poses from camera frame to world frame.
    
    Args:
        rotations: (B, 3, 3) rotation matrices in camera frame
        translations: (B, 3) translations in camera frame
        extrinsics: (B, 4, 4) camera-to-world transforms
    
    Returns:
        world_rotations: (B, 3, 3)
        world_translations: (B, 3)
    """
    world_rotations = extrinsics[:, :3, :3] @ rotations
    world_translations = (
        extrinsics[:, :3, :3] @ translations[:, :, None] + 
        extrinsics[:, :3, 3:]
    )[:, :, 0]
    return world_rotations, world_translations


def main():
    parser = argparse.ArgumentParser(description='CADGrasp Prediction Pipeline')
    
    # Checkpoint paths
    parser.add_argument('--ckpt_path', type=str, 
        default='data/DexGraspNet2.0/DexGraspNet2.0-ckpts/CAD/ckpt/ckpt_50000.pth',
        help='Path to DexGraspNet2.0 checkpoint')
    parser.add_argument('--las_exp_name', type=str, default='LEAP_dif',
        help='LASDiffusion experiment name')
    parser.add_argument('--las_ckpt', type=str, default=None,
        help='LASDiffusion checkpoint path (overrides las_exp_name)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda:0')
    
    # Robot model
    parser.add_argument('--urdf_path', type=str, 
        default='robot_models/urdf/leap_hand_simplified.urdf')
    parser.add_argument('--meta_path', type=str, 
        default='robot_models/meta/leap_hand/meta.yaml')
    parser.add_argument('--hand_name', type=str, default='leap_hand')
    
    # Data
    parser.add_argument('--camera', type=str, default='realsense')
    parser.add_argument('--scene_id', type=str, default='scene_0100')
    parser.add_argument('--dataset', type=str, default='graspnet', 
        choices=['graspnet', 'acronym'])
    parser.add_argument('--all_scene_ids_acronym', type=str, nargs='*', default=None)
    
    # Sampling parameters
    parser.add_argument('--grasp_num', type=int, default=1024,
        help='Number of grasp candidates from DexGraspNet2.0')
    parser.add_argument('--top_n', type=int, default=5,
        help='Number of top candidates per view for IBS prediction')
    parser.add_argument('--stride', type=int, default=32,
        help='Batch stride for DexGraspNet2.0 inference')
    
    # IBS generation
    parser.add_argument('--diffusion_steps', type=int, default=50,
        help='Number of diffusion steps for IBS generation')
    parser.add_argument('--ibs_stride', type=int, default=64,
        help='Batch stride for IBS generation')
    
    # Optimizer parameters
    parser.add_argument('--max_iters', type=int, default=200,
        help='Maximum optimization iterations')
    parser.add_argument('--lr', type=float, default=5e-3,
        help='Learning rate for optimization')
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--lr_decay_every', type=int, default=100)
    parser.add_argument('--parallel_num', type=int, default=10,
        help='Number of parallel optimizations per IBS')
    
    # IBSFilter parameters
    parser.add_argument('--cone_mu', type=float, default=1.0,
        help='Friction coefficient for force-closure filtering')
    parser.add_argument('--disable_filter', action='store_true',
        help='Disable IBSFilter force-closure filtering')
    
    # Misc
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--overwrite', type=int, default=1)
    parser.add_argument('--scene_num', type=int, default=10)
    parser.add_argument('--logs_path', type=str, default='logs/cadgrasp_predict')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device(args.device)
    
    # === Load Models ===
    cprint('[CADGrasp] Loading models...', 'cyan')
    
    # 1. Load DexGraspNet2.0 model
    robot_model = RobotModel(args.urdf_path, args.meta_path)
    config = ckpt_to_config(args.ckpt_path)
    dex_model = get_model(config.model)
    dex_model.config.voxel_size = config.data.voxel_size
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    dex_model.load_state_dict(ckpt['model'], strict=False)
    dex_model.to(device)
    dex_model.eval()
    
    # 2. Load LASDiffusion model
    if args.las_ckpt:
        las_ckpt_path = args.las_ckpt
    else:
        las_ckpt_path = f'thirdparty/LASDiffusion/results/{args.las_exp_name}/recent/last.ckpt'
    
    if not os.path.exists(las_ckpt_path):
        raise FileNotFoundError(f"LASDiffusion checkpoint not found: {las_ckpt_path}")
    
    las_model = DiffusionModel.load_from_checkpoint(las_ckpt_path).to(device)
    las_model.eval()
    cprint(f'[CADGrasp] Loaded LASDiffusion from {las_ckpt_path}', 'green')
    
    # 3. Initialize IBSFilter
    ibs_filter = IBSFilter(mu=args.cone_mu)
    
    # 4. Create tensorboard writer
    os.makedirs(args.logs_path, exist_ok=True)
    writer = SummaryWriter(args.logs_path)
    
    # === Process Scenes ===
    if args.dataset == 'graspnet':
        start_idx = int(args.scene_id.split('_')[-1])
    elif args.dataset == 'acronym':
        start_idx = int(args.scene_id)
    
    for scene_idx in trange(args.scene_num, desc='Processing scenes'):
        # Determine scene ID
        if args.dataset == 'graspnet':
            if start_idx < 190:
                if start_idx + scene_idx >= 190:
                    break
                scene_id = f'scene_{start_idx + scene_idx:04d}'
            elif start_idx < 380:
                if start_idx + scene_idx >= 380:
                    break
                scene_id = f'scene_{start_idx + scene_idx:04d}'
            elif start_idx > 8500:
                if start_idx + scene_idx * 5 >= 9900:
                    break
                scene_id = f'scene_{start_idx + scene_idx * 5:04d}'
            else:
                scene_id = f'scene_{start_idx + scene_idx:04d}'
            
            save_path = os.path.join(
                os.path.dirname(os.path.dirname(args.ckpt_path)), 
                'results', scene_id, 'grasps.npz'
            )
            load_path = os.path.join('data/DexGraspNet2.0/scenes', scene_id, args.camera, 'network_input.npz')
            
        elif args.dataset == 'acronym':
            all_scene_ids = args.all_scene_ids_acronym
            if start_idx + scene_idx >= len(all_scene_ids):
                break
            scene_id = all_scene_ids[start_idx + scene_idx].strip(',').strip('[').strip(']')
            split = scene_id.split('_')[1]
            load_path = os.path.join(
                f'data/acronym_test_scenes/network_input_{split}', 
                scene_id, args.camera, 'network_input.npz'
            )
            save_path = os.path.join(
                os.path.dirname(os.path.dirname(args.ckpt_path)), 
                'results_acronym', scene_id, 'grasps.npz'
            )
        
        # Skip if exists
        if os.path.exists(save_path) and not args.overwrite:
            continue
        
        cprint(f'[CADGrasp] Processing {scene_id}', 'cyan')
        
        try:
            # === Step 1: Load network input ===
            network_input = dict(np.load(load_path))
            pc_all = torch.tensor(network_input['pc'], dtype=torch.float)  # (256, N, 3)
            seg_all = torch.tensor(network_input['seg'], dtype=torch.long)
            edge_all = torch.tensor(network_input['edge'], dtype=torch.long)
            extrinsics_all = network_input['extrinsics']  # (256, 4, 4)
            
            num_views = pc_all.shape[0]
            
            # === Step 2: DexGraspNet2.0 prediction ===
            with torch.no_grad():
                rotations, translations, qposs, scores, grasp_points_list = [], [], [], [], []
                
                for i in range(0, num_views, args.stride):
                    data_part = get_sparse_tensor(pc_all[i:i+args.stride], config.data.voxel_size)
                    data_part['seg'] = seg_all[i:i+args.stride]
                    data_part = {k: v.to(device) for k, v in data_part.items()}
                    edge_part = edge_all[i:i+args.stride]
                    
                    # with_point=True to get grasp_points (seed_points)
                    result = dex_model.sample(
                        data_part, args.grasp_num, 
                        graspness_scale=5, allow_fail=True, 
                        cate=False, edge=edge_part.to(device), 
                        with_score_parts=True, with_point=True
                    )
                    # result: [rot, trans, joints, score, obj_indices, graspness, log_prob, seed_points]
                    rotation, translation, qpos, score = result[0].cpu(), result[1].cpu(), result[2].cpu(), result[3].cpu()
                    grasp_point = result[7].cpu()  # seed_points from with_point=True
                    
                    rotations.append(rotation)
                    translations.append(translation)
                    qposs.append(qpos)
                    scores.append(score)
                    # grasp_point is (B*K, 3), reshape to (B, K, 3)
                    batch_size = rotation.shape[0]
                    grasp_points_list.append(grasp_point.reshape(batch_size, -1, 3))
                
                rotations = torch.cat(rotations, dim=0)      # (256, K, 3, 3)
                translations = torch.cat(translations, dim=0)  # (256, K, 3)
                qposs = torch.cat(qposs, dim=0)              # (256, K, J)
                scores = torch.cat(scores, dim=0)            # (256, K)
                grasp_points = torch.cat(grasp_points_list, dim=0)  # (256, K, 3)
            
            # === Step 3: Select top_n per view ===
            sel_rot, sel_trans, sel_qpos, sel_scores, sel_grasp_pts = select_top_n_per_view(
                scores, rotations, translations, qposs, grasp_points, args.top_n
            )
            
            # === Step 4: Compose w2h_trans ===
            # Use grasp_points (not translations!) to compose w2h_trans
            # grasp_point is the IBS frame origin, rotation defines the frame orientation
            w2h_trans = compose_w2h_trans(sel_rot, sel_grasp_pts)  # (256, top_n, 4, 4)
            
            # Reshape for batch processing: (256 * top_n, ...)
            B, top_n = w2h_trans.shape[:2]
            w2h_trans_flat = w2h_trans.reshape(-1, 4, 4).to(device)  # (256*top_n, 4, 4)
            
            # Expand scene point clouds
            pc_expanded = pc_all.unsqueeze(1).expand(-1, top_n, -1, -1)  # (256, top_n, N, 3)
            pc_flat = pc_expanded.reshape(-1, pc_all.shape[1], 3).to(device)  # (256*top_n, N, 3)
            
            # === Step 5: Predict IBS with LASDiffusion ===
            cprint(f'[CADGrasp] Predicting IBS for {B * top_n} candidates...', 'cyan')
            ibs_voxels = predict_ibs_batch(
                model=las_model,
                scene_pcs=pc_flat,
                w2h_trans=w2h_trans_flat,
                device=str(device),
                stride_size=args.ibs_stride,
                steps=args.diffusion_steps,
                ema=True
            )  # (256*top_n, 2, 40, 40, 40)
            
            # === Step 6: Filter IBS with IBSFilter ===
            cprint(f'[CADGrasp] Filtering IBS...', 'cyan')
            filter_result = ibs_filter.filter_batch(
                ibs_voxels=ibs_voxels,
                num_particles=B,
                top_n=top_n,
                visualize_count=0,
                disable_force_closure=args.disable_filter,
                verbose=args.verbose
            )
            
            valid_triplets = filter_result.triplets
            valid_particle_indices = filter_result.particle_indices
            valid_global_indices = filter_result.global_indices
            
            if len(valid_triplets) == 0:
                cprint(f'[CADGrasp] No valid IBS found for {scene_id}, skipping...', 'yellow')
                continue
            
            cprint(f'[CADGrasp] {len(valid_triplets)}/{B} views have valid IBS', 'green')
            
            # === Step 7: Optimize with AdamOpt ===
            cprint(f'[CADGrasp] Optimizing hand poses...', 'cyan')
            optimizer = AdamOptimizer(
                hand_name=args.hand_name,
                writer=writer,
                max_iters=args.max_iters,
                learning_rate=args.lr,
                lr_decay=args.lr_decay,
                decay_every=args.lr_decay_every,
                parallel_num=args.parallel_num,
                device=str(device)
            )
            
            # Convert triplets to legacy format for AdamOpt
            ibs_triplets = [
                (t.ibs_occu, t.ibs_cont, t.ibs_thumb_cont) 
                for t in valid_triplets
            ]
            
            q_trajectory, energy_dict = optimizer.run_adam(
                ibs_triplets=ibs_triplets,
                running_name=f"{scene_id}",
                cone_viz_num=0,
                cone_mu=args.cone_mu,
                filt_or_not=False  # Already filtered
            )
            
            # Get final pose
            final_pose = q_trajectory[:, -1, :].cpu().numpy()  # (num_valid, 25)
            
            # === Step 8: Transform back to camera/world frame and save ===
            # Extract components from optimized pose
            opt_translations = final_pose[:, :3]
            opt_rotations = robust_compute_rotation_matrix_from_ortho6d(
                torch.from_numpy(final_pose[:, 3:9])
            ).numpy()
            opt_qpos = final_pose[:, 9:]
            
            # Build hand-to-IBS transforms
            hand2ibs = np.eye(4)[np.newaxis].repeat(len(final_pose), axis=0)
            hand2ibs[:, :3, 3] = opt_translations
            hand2ibs[:, :3, :3] = opt_rotations
            
            # Get corresponding w2h_trans and extrinsics
            valid_w2h = w2h_trans_flat[valid_global_indices].cpu().numpy()
            valid_extrinsics = extrinsics_all[valid_particle_indices]
            
            # hand_in_camera = cam_to_ibs^(-1) @ hand_in_ibs
            # where cam_to_ibs = w2h_trans^(-1)
            ibs2cam = np.linalg.inv(valid_w2h)
            hand_in_camera = ibs2cam @ hand2ibs
            
            camera_translations = hand_in_camera[:, :3, 3]
            camera_rotations = hand_in_camera[:, :3, :3]
            
            # Transform to world frame
            world_rotations, world_translations = transform_pose_to_world(
                camera_rotations, camera_translations, valid_extrinsics
            )
            
            # === Save results ===
            grasps = {
                'rotation': world_rotations,
                'translation': world_translations,
                'valid_view_indices': np.array(valid_particle_indices),
            }
            
            # Add joint positions
            for i, joint_name in enumerate(robot_model.joint_names):
                grasps[joint_name] = opt_qpos[:, i]
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(save_path, **grasps)
            cprint(f'[CADGrasp] Saved {len(valid_triplets)} grasps to {save_path}', 'green')
            
        except Exception as e:
            cprint(f'[CADGrasp] Error processing {scene_id}: {e}', 'red')
            if args.verbose:
                import traceback
                traceback.print_exc()
            continue
    
    writer.close()
    cprint('[CADGrasp] Done!', 'green')


if __name__ == '__main__':
    main()
