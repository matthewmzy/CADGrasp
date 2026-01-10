"""
Test script for hand pose optimization using ground truth IBS data.

This script tests the AdamOptimizer by:
1. Loading ground truth IBS voxels from ibsdata
2. Converting to optimizer input format (ibs_triplets)
3. Running parallel optimization (5 IBS, 5 parallel poses each)
4. Saving interactive HTML visualization with trajectory playback

Usage:
    python tests/test_optimizer.py --scene_id 0 --num_ibs 5
    python tests/test_optimizer.py --scene_id 0 --ibs_indices 0,10,20,30,40
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from termcolor import cprint
import k3d
import copy

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cadgrasp.optimizer.AdamOpt import AdamOptimizer
from cadgrasp.optimizer.ibs_func import IBSFilter, estimate_normals
from cadgrasp.optimizer.HandModel import get_handmodel
from cadgrasp.ibs.utils.ibs_repr import IBS, IBSConfig
from LASDiffusion.utils.visualize_ibs_vox import devoxelize


def load_gt_ibs_data(scene_id: int, ibs_indices: list, data_root: str = 'data/ibsdata'):
    """
    Load ground truth IBS data from ibsdata folder.
    
    Args:
        scene_id: Scene ID (0, 1, ...)
        ibs_indices: List of IBS indices to load
        data_root: Root path for IBS data
    
    Returns:
        ibs_voxels: (N, 40, 40, 40, 3) IBS voxel data (bool)
        w2h_trans: (N, 4, 4) world to hand transformation
    """
    scene_name = f'scene_{scene_id:04d}'
    ibs_path = os.path.join(data_root, 'ibs', f'{scene_name}.npy')
    w2h_path = os.path.join(data_root, 'w2h_trans', f'{scene_name}.npy')
    
    if not os.path.exists(ibs_path):
        raise FileNotFoundError(f"IBS file not found: {ibs_path}")
    if not os.path.exists(w2h_path):
        raise FileNotFoundError(f"w2h_trans file not found: {w2h_path}")
    
    ibs_all = np.load(ibs_path)  # (N, 40, 40, 40, 3) bool
    w2h_all = np.load(w2h_path)  # (N, 4, 4)
    
    cprint(f"[Test] Loaded IBS data: {ibs_all.shape}, w2h_trans: {w2h_all.shape}", 'cyan')
    
    # Select specified indices
    ibs_voxels = ibs_all[ibs_indices]
    w2h_trans = w2h_all[ibs_indices]
    
    return ibs_voxels, w2h_trans


def convert_gt_ibs_to_network_format(ibs_voxels: np.ndarray) -> np.ndarray:
    """
    Convert ground truth IBS voxels (40,40,40,3 bool) to network output format (2, 40, 40, 40).
    
    Ground truth format:
        - Channel 0: occupancy (True/False)
        - Channel 1: contact (True/False)  
        - Channel 2: thumb_contact (True/False)
    
    Network format:
        - Channel 0: -1 (empty), 1 (occupancy), 2 (contact)
        - Channel 1: 0 (not thumb), 2 (thumb contact)
    
    Args:
        ibs_voxels: (N, 40, 40, 40, 3) bool array
    
    Returns:
        network_voxels: (N, 2, 40, 40, 40) float array
    """
    N = ibs_voxels.shape[0]
    network_voxels = np.zeros((N, 2, 40, 40, 40), dtype=np.float32)
    
    for i in range(N):
        occu = ibs_voxels[i, :, :, :, 0]      # occupancy
        cont = ibs_voxels[i, :, :, :, 1]      # contact
        thumb = ibs_voxels[i, :, :, :, 2]     # thumb contact
        
        # Channel 0: empty=-1, occu=1, contact=2
        network_voxels[i, 0] = -1  # Default empty
        network_voxels[i, 0][occu] = 1
        network_voxels[i, 0][cont] = 2
        network_voxels[i, 0][thumb] = 2  # Thumb contact is also contact
        
        # Channel 1: no thumb=0, thumb=2
        network_voxels[i, 1][thumb] = 2
    
    return network_voxels


def process_ibs_for_optimizer(network_voxels: np.ndarray, mu: float = 1.0):
    """
    Process IBS voxels to get triplets for optimizer.
    
    Args:
        network_voxels: (N, 2, 40, 40, 40) network format
        mu: Friction coefficient
    
    Returns:
        ibs_triplets: List of (ibs_occu, ibs_cont, ibs_thumb) tuples
    """
    ibs_filter = IBSFilter(mu=mu)
    triplets = []
    
    for i in range(network_voxels.shape[0]):
        ibs_vox = network_voxels[i]
        
        # Devoxelize
        ibs_occu, ibs_cont, ibs_thumb = ibs_filter._devoxelize_network_output(ibs_vox)
        
        # Check minimum points
        if ibs_occu.shape[0] < 5 or ibs_cont.shape[0] < 5 or ibs_thumb.shape[0] < 5:
            cprint(f"[Test] IBS {i} has insufficient points, skipping", 'yellow')
            continue
        
        # Estimate normals
        all_points = np.concatenate([ibs_occu, ibs_cont, ibs_thumb], axis=0)
        all_normals, _ = estimate_normals(all_points)
        
        n1, n2 = ibs_occu.shape[0], ibs_cont.shape[0]
        ibs_occu = np.concatenate([ibs_occu, all_normals[:n1]], axis=1)
        ibs_cont = np.concatenate([ibs_cont, all_normals[n1:n1+n2]], axis=1)
        ibs_thumb = np.concatenate([ibs_thumb, all_normals[n1+n2:]], axis=1)
        
        triplets.append((ibs_occu, ibs_cont, ibs_thumb))
    
    return triplets


def visualize_k3d_optimization(
    ibs_triplets: list,
    q_trajectory: torch.Tensor,
    hand_name: str,
    save_path: str,
    device: str = 'cuda'
):
    """
    Create interactive k3d visualization of optimization trajectory.
    
    Args:
        ibs_triplets: List of (ibs_occu, ibs_cont, ibs_thumb) tuples
        q_trajectory: (num_ibs, traj_len, n_dofs) trajectory tensor
        hand_name: Hand model name
        save_path: Path to save HTML
        device: Device for hand model
    """
    cprint(f"[Test] Creating visualization for {len(ibs_triplets)} IBS...", 'cyan')
    
    hand_model = get_handmodel(hand_name, 1, device)
    num_ibs = len(ibs_triplets)
    traj_len = q_trajectory.shape[1]
    
    plot = k3d.plot()
    
    # Colors for different IBS
    ibs_colors = [0xff0000, 0x00ff00, 0x0000ff, 0xffff00, 0xff00ff]
    hand_colors = [0xff6666, 0x66ff66, 0x6666ff, 0xffff66, 0xff66ff]
    
    # Add static IBS point clouds
    for i, (ibs_occu, ibs_cont, ibs_thumb) in enumerate(ibs_triplets):
        offset = np.array([i * 0.3, 0, 0])  # Offset for visualization
        
        # IBS occupancy
        occu_pts = ibs_occu[:, :3] + offset
        plot += k3d.points(occu_pts.astype(np.float32), color=0x808080, point_size=0.002, name=f'IBS_{i}_occu')
        
        # Contact points
        cont_pts = ibs_cont[:, :3] + offset
        plot += k3d.points(cont_pts.astype(np.float32), color=ibs_colors[i % len(ibs_colors)], point_size=0.004, name=f'IBS_{i}_contact')
        
        # Thumb contact
        thumb_pts = ibs_thumb[:, :3] + offset
        plot += k3d.points(thumb_pts.astype(np.float32), color=0x00ffff, point_size=0.005, name=f'IBS_{i}_thumb')
    
    # Add animated hand meshes
    mesh_objects = []
    pts_objects = []
    thumb_pts_objects = []
    
    for i in range(num_ibs):
        offset = np.array([i * 0.3, 0, 0])
        
        # Get initial frame mesh
        q_init = q_trajectory[i, 0].unsqueeze(0)
        mesh_data = hand_model.get_k3d_data(q=q_init, i=0, opacity=0.7, color=hand_colors[i % len(hand_colors)], concat=True)
        
        # Build animation dict
        vertices_dict = {}
        for t in range(traj_len):
            q_t = q_trajectory[i, t].unsqueeze(0)
            mesh_t = hand_model.get_k3d_data(q=q_t, i=0, opacity=0.7, color=hand_colors[i % len(hand_colors)], concat=True)
            vertices_dict[str(t)] = mesh_t.vertices + offset.astype(np.float32)
        
        mesh_data.vertices = vertices_dict
        plot += mesh_data
        mesh_objects.append(mesh_data)
        
        # Contact points animation
        pts_data, thumb_data = hand_model.get_palmar_points_k3d(i=0)
        
        pts_dict = {}
        thumb_pts_dict = {}
        for t in range(traj_len):
            q_t = q_trajectory[i, t].unsqueeze(0)
            hand_model.update_kinematics(q=q_t)
            pts_t, thumb_t = hand_model.get_palmar_points_k3d(i=0)
            pts_dict[str(t)] = pts_t.positions + offset.astype(np.float32)
            thumb_pts_dict[str(t)] = thumb_t.positions + offset.astype(np.float32)
        
        pts_data.positions = pts_dict
        thumb_data.positions = thumb_pts_dict
        
        plot += pts_data
        plot += thumb_data
        pts_objects.append(pts_data)
        thumb_pts_objects.append(thumb_data)
    
    plot.fps = 30
    
    # Save HTML
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    html_snapshot = plot.get_snapshot()
    with open(save_path, 'w') as f:
        f.write(html_snapshot)
    
    cprint(f"[Test] Saved visualization to {save_path}", 'green')


def main():
    parser = argparse.ArgumentParser(description='Test hand pose optimizer with GT IBS')
    parser.add_argument('--scene_id', type=int, default=0, help='Scene ID')
    parser.add_argument('--num_ibs', type=int, default=5, help='Number of IBS to optimize')
    parser.add_argument('--ibs_indices', type=str, default=None, 
                        help='Comma-separated IBS indices (overrides num_ibs)')
    parser.add_argument('--hand_name', type=str, default='leap_hand')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_iters', type=int, default=200, help='Max optimization iterations')
    parser.add_argument('--parallel_num', type=int, default=5, help='Parallel poses per IBS')
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('--mu', type=float, default=1.0, help='Friction coefficient')
    parser.add_argument('--output_dir', type=str, default='logs/test_optimizer')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    
    # Determine IBS indices
    if args.ibs_indices:
        ibs_indices = [int(x) for x in args.ibs_indices.split(',')]
    else:
        # Load all IBS first to get total count
        ibs_path = f'data/ibsdata/ibs/scene_{args.scene_id:04d}.npy'
        if not os.path.exists(ibs_path):
            raise FileNotFoundError(f"IBS file not found: {ibs_path}")
        total_ibs = np.load(ibs_path).shape[0]
        
        # Sample evenly spaced indices
        step = max(1, total_ibs // args.num_ibs)
        ibs_indices = list(range(0, min(total_ibs, args.num_ibs * step), step))[:args.num_ibs]
    
    cprint(f"[Test] Using IBS indices: {ibs_indices}", 'cyan')
    
    # === Step 1: Load GT IBS data ===
    cprint("[Test] Loading ground truth IBS data...", 'cyan')
    ibs_voxels, w2h_trans = load_gt_ibs_data(args.scene_id, ibs_indices)
    
    # === Step 2: Convert to network format ===
    cprint("[Test] Converting to network format...", 'cyan')
    network_voxels = convert_gt_ibs_to_network_format(ibs_voxels)
    
    # === Step 3: Process for optimizer ===
    cprint("[Test] Processing IBS for optimizer...", 'cyan')
    ibs_triplets = process_ibs_for_optimizer(network_voxels, mu=args.mu)
    
    if len(ibs_triplets) == 0:
        cprint("[Test] No valid IBS triplets found!", 'red')
        return
    
    cprint(f"[Test] Got {len(ibs_triplets)} valid IBS triplets", 'green')
    
    # === Step 4: Run optimization ===
    cprint("[Test] Running optimization...", 'cyan')
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(args.output_dir)
    
    optimizer = AdamOptimizer(
        hand_name=args.hand_name,
        writer=writer,
        max_iters=args.max_iters,
        learning_rate=args.lr,
        parallel_num=args.parallel_num,
        device=str(device)
    )
    
    q_trajectory, energy_dict = optimizer.run_adam(
        ibs_triplets=ibs_triplets,
        running_name=f"scene_{args.scene_id:04d}_test",
        cone_viz_num=0,
        cone_mu=args.mu,
        filt_or_not=False
    )
    
    cprint(f"[Test] Optimization complete. Trajectory shape: {q_trajectory.shape}", 'green')
    
    # === Step 5: Visualize ===
    save_path = os.path.join(args.output_dir, f'scene_{args.scene_id:04d}_optimization.html')
    visualize_k3d_optimization(
        ibs_triplets=ibs_triplets,
        q_trajectory=q_trajectory,
        hand_name=args.hand_name,
        save_path=save_path,
        device=str(device)
    )
    
    # Save energy curves
    for i in range(len(ibs_triplets)):
        from cadgrasp.optimizer.visualize import save_energy_curve
        save_energy_curve(None, energy_dict, i, args.output_dir)
    
    writer.close()
    cprint("[Test] Done!", 'green')


if __name__ == '__main__':
    main()
