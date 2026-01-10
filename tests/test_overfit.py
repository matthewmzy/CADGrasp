"""
Test IBS Generation on Train vs Test Scenes.

This script evaluates the trained LASDiffusion model by:
1. Loading IBS data for scene 0 (train) and scene 1 (test)
2. Generating IBS predictions from scene observations
3. Visualizing the comparison between GT and predicted IBS

Usage:
    python tests/test_overfit.py --model_path thirdparty/LASDiffusion/results/overfit_scene0/recent/last.ckpt
"""

import os
import sys
import argparse
import numpy as np
import torch
import open3d as o3d
from typing import Tuple, Optional

# Add thirdparty to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'thirdparty'))

from cadgrasp.paths import project_path
from cadgrasp.ibs.utils.ibs_repr import IBS
from LASDiffusion.network.model_trainer import DiffusionModel
from LASDiffusion.network.data_loader import IBS_Dataset


def load_model(model_path: str, device: str = 'cuda:0'):
    """Load trained model."""
    model = DiffusionModel.load_from_checkpoint(model_path).to(device)
    model.eval()
    return model


def get_sample_from_scene(
    scene_id: int, 
    view_id: int = 0, 
    ibs_idx: int = 0,
    device: str = 'cuda:0'
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Get a sample (scene voxel, IBS GT) for a specific scene and view.
    
    Returns:
        scene_voxel: (1, 1, 40, 40, 40) scene voxel
        ibs_gt_voxel: (1, 2, 40, 40, 40) GT IBS voxel in network format
        ibs_gt_raw: (40, 40, 40, 3) GT IBS voxel in raw format
    """
    # Create dataset for this scene only
    ds = IBS_Dataset(
        path=project_path('data/ibsdata'),
        scene_path=project_path('data/DexGraspNet2.0/scenes'),
        ibs_load_per_scene=1,
        split='train',
        use_view_filter=False,
        scene_ids=[scene_id]
    )
    
    # Find item with matching view
    target_item = None
    for item in ds.items:
        if item[1] == view_id:
            target_item = item
            break
    
    if target_item is None:
        print(f"View {view_id} not found for scene {scene_id}, using first available")
        target_item = ds.items[0]
    
    scene_id, view_id, valid_ids = target_item
    scene_name = f'scene_{str(scene_id).zfill(4)}'
    
    # Load IBS data
    ibs_file = os.path.join(ds.path, 'ibs', f'{scene_name}.npy')
    w2h_file = os.path.join(ds.path, 'w2h_trans', f'{scene_name}.npy')
    
    ibs_all = np.load(ibs_file)  # (N, 40, 40, 40, 3)
    w2h_all = np.load(w2h_file)  # (N, 4, 4)
    
    # Select specific IBS
    valid_indices = np.where(valid_ids)[0]
    if ibs_idx >= len(valid_indices):
        ibs_idx = 0
    selected_idx = valid_indices[ibs_idx]
    
    ibs_raw = ibs_all[selected_idx]  # (40, 40, 40, 3)
    w2h = w2h_all[selected_idx]      # (4, 4)
    
    # Load scene point cloud
    scene_data = np.load(os.path.join(
        ds.scene_path, scene_name, 'realsense', 'network_input.npz'
    ))
    scene_pc = torch.from_numpy(scene_data['pc'][view_id]).float()  # (N, 3)
    c2w_trans = torch.from_numpy(scene_data['extrinsics'][view_id]).float()  # (4, 4)
    w2h_tensor = torch.from_numpy(w2h).float()
    
    # Transform scene PC: camera -> world -> hand
    from cadgrasp.ibs.utils.transforms import transform_points
    scene_pc_w = transform_points(scene_pc, c2w_trans)
    scene_pc_h = transform_points(scene_pc_w, w2h_tensor)
    
    # Voxelize scene
    scene_voxel = voxelize_scene(scene_pc_h, device)  # (1, 1, 40, 40, 40)
    
    # Convert IBS to network format
    ibs_tensor = torch.from_numpy(ibs_raw).to(dtype=torch.bool)
    ibs_voxel = torch.full((40, 40, 40, 2), -1.0, dtype=torch.float)
    ibs_voxel[..., 0] = torch.where(ibs_tensor[..., 0], 1.0, -1.0)
    ibs_voxel[..., 1] = torch.where(ibs_tensor[..., 1], 1.0, ibs_voxel[..., 1])
    ibs_voxel[..., 1] = torch.where(ibs_tensor[..., 2], 2.0, ibs_voxel[..., 1])
    ibs_voxel = ibs_voxel.permute(3, 0, 1, 2).unsqueeze(0)  # (1, 2, 40, 40, 40)
    
    return scene_voxel.to(device), ibs_voxel.to(device), ibs_raw


def voxelize_scene(scene_pc_h: torch.Tensor, device: str = 'cuda:0') -> torch.Tensor:
    """Voxelize scene point cloud in hand frame."""
    VOXEL_BOUND = 0.1
    VOXEL_RESOLUTION = 0.005
    VOXEL_SIZE = 40
    
    scene_voxel = torch.zeros((VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE), dtype=torch.float)
    
    # Filter points within bounds
    mask = (
        (scene_pc_h[:, 0] > -VOXEL_BOUND) & (scene_pc_h[:, 0] < VOXEL_BOUND) &
        (scene_pc_h[:, 1] > -VOXEL_BOUND) & (scene_pc_h[:, 1] < VOXEL_BOUND) &
        (scene_pc_h[:, 2] > -VOXEL_BOUND) & (scene_pc_h[:, 2] < VOXEL_BOUND)
    )
    scene_pc_h = scene_pc_h[mask]
    
    # Convert to voxel indices
    voxel_idx = ((scene_pc_h + VOXEL_BOUND) / VOXEL_RESOLUTION).to(dtype=torch.long)
    voxel_idx = torch.clamp(voxel_idx, 0, VOXEL_SIZE - 1)
    
    scene_voxel[voxel_idx[:, 0], voxel_idx[:, 1], voxel_idx[:, 2]] = 1.0
    
    return scene_voxel.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 40, 40, 40)


def decode_prediction(pred: torch.Tensor) -> np.ndarray:
    """Decode network output to (40, 40, 40, 3) format."""
    pred_np = pred.cpu().numpy()
    result = np.zeros((40, 40, 40, 3), dtype=np.float32)
    
    # Occupancy: threshold at 0
    result[..., 0] = (pred_np[0, 0] > 0).astype(np.float32)
    
    # Contact: values between 0.5 and 1.5
    result[..., 1] = ((pred_np[0, 1] > 0.5) & (pred_np[0, 1] < 1.5)).astype(np.float32)
    
    # Thumb: values > 1.5
    result[..., 2] = (pred_np[0, 1] > 1.5).astype(np.float32)
    
    return result


def voxel_to_points(voxel_mask: np.ndarray) -> np.ndarray:
    """Convert voxel mask to physical coordinates."""
    VOXEL_BOUND = 0.1
    VOXEL_RESOLUTION = 0.005
    
    indices = np.where(voxel_mask)
    if len(indices[0]) == 0:
        return np.zeros((0, 3))
    voxel_coords = np.stack(indices, axis=-1)
    physical_coords = voxel_coords * VOXEL_RESOLUTION + np.array([-VOXEL_BOUND, -VOXEL_BOUND, -VOXEL_BOUND])
    return physical_coords


def visualize_comparison(
    gt_ibs: np.ndarray, 
    pred_ibs: np.ndarray, 
    scene_voxel: np.ndarray,
    title: str = "GT vs Pred"
):
    """Visualize GT and predicted IBS side by side."""
    
    # Create point clouds
    geometries = []
    
    # Scene (grey)
    scene_pts = voxel_to_points(scene_voxel[0, 0] > 0.5)
    if len(scene_pts) > 0:
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(scene_pts)
        scene_pcd.paint_uniform_color([0.5, 0.5, 0.5])
        geometries.append(scene_pcd)
    
    # GT IBS - shifted left
    offset_left = np.array([-0.25, 0, 0])
    
    gt_occ = voxel_to_points(gt_ibs[..., 0].astype(bool) & ~gt_ibs[..., 1].astype(bool))
    gt_contact = voxel_to_points(gt_ibs[..., 1].astype(bool))
    gt_thumb = voxel_to_points(gt_ibs[..., 2].astype(bool))
    
    if len(gt_occ) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(gt_occ + offset_left)
        pcd.paint_uniform_color([0.3, 0.3, 0.9])  # Blue
        geometries.append(pcd)
    
    if len(gt_contact) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(gt_contact + offset_left)
        pcd.paint_uniform_color([0.2, 0.9, 0.2])  # Green
        geometries.append(pcd)
    
    if len(gt_thumb) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(gt_thumb + offset_left)
        pcd.paint_uniform_color([0.9, 0.9, 0.2])  # Yellow
        geometries.append(pcd)
    
    # Pred IBS - shifted right
    offset_right = np.array([0.25, 0, 0])
    
    pred_occ = voxel_to_points(pred_ibs[..., 0].astype(bool) & ~pred_ibs[..., 1].astype(bool))
    pred_contact = voxel_to_points(pred_ibs[..., 1].astype(bool))
    pred_thumb = voxel_to_points(pred_ibs[..., 2].astype(bool))
    
    if len(pred_occ) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pred_occ + offset_right)
        pcd.paint_uniform_color([0.9, 0.3, 0.3])  # Red
        geometries.append(pcd)
    
    if len(pred_contact) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pred_contact + offset_right)
        pcd.paint_uniform_color([0.2, 0.9, 0.9])  # Cyan
        geometries.append(pcd)
    
    if len(pred_thumb) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pred_thumb + offset_right)
        pcd.paint_uniform_color([0.9, 0.2, 0.9])  # Magenta
        geometries.append(pcd)
    
    # Add labels (coordinate frames)
    gt_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    gt_frame.translate(offset_left)
    geometries.append(gt_frame)
    
    pred_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    pred_frame.translate(offset_right)
    geometries.append(pred_frame)
    
    print(f"\n{title}")
    print(f"GT: Occ={len(gt_occ)}, Contact={len(gt_contact)}, Thumb={len(gt_thumb)}")
    print(f"Pred: Occ={len(pred_occ)}, Contact={len(pred_contact)}, Thumb={len(pred_thumb)}")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name=title,
        width=1280,
        height=720,
    )


def compute_iou(gt: np.ndarray, pred: np.ndarray) -> dict:
    """Compute IoU metrics between GT and predicted IBS."""
    metrics = {}
    
    for i, name in enumerate(['occupancy', 'contact', 'thumb']):
        gt_mask = gt[..., i] > 0.5
        pred_mask = pred[..., i] > 0.5
        
        intersection = np.sum(gt_mask & pred_mask)
        union = np.sum(gt_mask | pred_mask)
        
        iou = intersection / (union + 1e-6)
        precision = intersection / (np.sum(pred_mask) + 1e-6)
        recall = intersection / (np.sum(gt_mask) + 1e-6)
        
        metrics[name] = {
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'gt_count': np.sum(gt_mask),
            'pred_count': np.sum(pred_mask),
        }
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, 
                        default='thirdparty/LASDiffusion/results/overfit_scene0/recent/last.ckpt')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--visualize', action='store_true', default=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    
    # Check if model exists
    model_path = project_path(args.model_path)
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please wait for training to complete or provide a valid model path.")
        return
    
    print(f"Loading model from {model_path}")
    model = load_model(model_path, args.device)
    generator = model.ema_model
    
    # Test on train scene (scene 0)
    print("\n" + "="*60)
    print("Testing on TRAIN scene (scene_0000)")
    print("="*60)
    
    scene_voxel_0, ibs_gt_0, ibs_raw_0 = get_sample_from_scene(0, view_id=0, ibs_idx=0, device=args.device)
    
    with torch.no_grad():
        pred_0 = generator.sample_based_on_scene(
            batch_size=1,
            scene_voxel=scene_voxel_0,
            steps=args.steps,
            truncated_index=0.0
        )
    
    pred_ibs_0 = decode_prediction(pred_0)
    metrics_0 = compute_iou(ibs_raw_0, pred_ibs_0)
    
    print("\nMetrics for Scene 0 (Train):")
    for name, m in metrics_0.items():
        print(f"  {name}: IoU={m['iou']:.3f}, Prec={m['precision']:.3f}, Recall={m['recall']:.3f}")
    
    if args.visualize:
        visualize_comparison(
            ibs_raw_0, pred_ibs_0, scene_voxel_0.cpu().numpy(),
            title="Scene 0 (Train) - Left: GT, Right: Pred"
        )
    
    # Test on test scene (scene 1) - if IBS data exists
    print("\n" + "="*60)
    print("Testing on TEST scene (scene_0001)")
    print("="*60)
    
    ibs_file_1 = project_path('data/ibsdata/ibs/scene_0001.npy')
    network_input_1 = project_path('data/DexGraspNet2.0/scenes/scene_0001/realsense/network_input.npz')
    
    if os.path.exists(ibs_file_1) and os.path.exists(network_input_1):
        scene_voxel_1, ibs_gt_1, ibs_raw_1 = get_sample_from_scene(1, view_id=0, ibs_idx=0, device=args.device)
        
        with torch.no_grad():
            pred_1 = generator.sample_based_on_scene(
                batch_size=1,
                scene_voxel=scene_voxel_1,
                steps=args.steps,
                truncated_index=0.0
            )
        
        pred_ibs_1 = decode_prediction(pred_1)
        metrics_1 = compute_iou(ibs_raw_1, pred_ibs_1)
        
        print("\nMetrics for Scene 1 (Test):")
        for name, m in metrics_1.items():
            print(f"  {name}: IoU={m['iou']:.3f}, Prec={m['precision']:.3f}, Recall={m['recall']:.3f}")
        
        if args.visualize:
            visualize_comparison(
                ibs_raw_1, pred_ibs_1, scene_voxel_1.cpu().numpy(),
                title="Scene 1 (Test) - Left: GT, Right: Pred"
            )
    elif os.path.exists(ibs_file_1):
        print(f"IBS data for scene 1 not found. Generating IBS without GT comparison...")
        
        # Still try to generate IBS for scene 1 using the network_input
        scene_data = np.load(project_path('data/DexGraspNet2.0/scenes/scene_0001/realsense/network_input.npz'))
        scene_pc = torch.from_numpy(scene_data['pc'][0]).float()  # view 0
        
        # Use a random hand pose (from scene 0's w2h)
        w2h_0 = np.load(project_path('data/ibsdata/w2h_trans/scene_0000.npy'))[0]
        
        from cadgrasp.ibs.utils.transforms import transform_points
        c2w = torch.from_numpy(scene_data['extrinsics'][0]).float()
        scene_pc_w = transform_points(scene_pc, c2w)
        scene_pc_h = transform_points(scene_pc_w, torch.from_numpy(w2h_0).float())
        
        scene_voxel_1 = voxelize_scene(scene_pc_h, args.device)
        
        with torch.no_grad():
            pred_1 = generator.sample_based_on_scene(
                batch_size=1,
                scene_voxel=scene_voxel_1,
                steps=args.steps,
                truncated_index=0.0
            )
        
        pred_ibs_1 = decode_prediction(pred_1)
        
        print(f"\nPrediction for Scene 1 (no GT available):")
        print(f"  Occupancy: {np.sum(pred_ibs_1[..., 0])}")
        print(f"  Contact: {np.sum(pred_ibs_1[..., 1])}")
        print(f"  Thumb: {np.sum(pred_ibs_1[..., 2])}")


if __name__ == '__main__':
    main()
