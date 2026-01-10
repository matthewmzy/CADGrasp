"""
IBS Voxel Visualization Script.

Visualize IBS (Interaction Bisector Surface) voxels for a given scene.
Uses data from data/ibsdata directory.

Usage:
    python tests/vis_ibs.py --scene_id 55
    python tests/vis_ibs.py --scene_id 55 --grasp_indices 0 1 2 --save_plot
    python tests/vis_ibs.py --scene_id 55 --with_scene_pc
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional

from cadgrasp.paths import project_path

# Default paths
DEFAULT_IBS_PATH = project_path('data/ibsdata')
DEFAULT_SCENE_PATH = project_path('data/DexGraspNet2.0/scenes')

# IBS voxel parameters
BOUND = 0.1
RESOLUTION = 0.005
GRID_SIZE = 40


def load_ibs_data(ibs_path: str, scene_id: int):
    """
    Load IBS data for a scene.
    
    Args:
        ibs_path: Base path to IBS data
        scene_id: Scene ID
    
    Returns:
        Tuple of (ibs_voxels, w2h_trans, hand_dis) or (None, None, None) if not found
    """
    scene_name = f'scene_{str(scene_id).zfill(4)}'
    
    ibs_file = os.path.join(ibs_path, 'ibs', f'{scene_name}.npy')
    w2h_file = os.path.join(ibs_path, 'w2h_trans', f'{scene_name}.npy')
    hand_dis_file = os.path.join(ibs_path, 'hand_dis', f'{scene_name}.npy')
    
    files = [ibs_file, w2h_file]
    if not all(os.path.exists(f) for f in files):
        print(f"IBS data not found for scene {scene_id} at {ibs_path}")
        return None, None, None
    
    ibs_voxels = np.load(ibs_file)  # (N, 40, 40, 40, 3)
    w2h_trans = np.load(w2h_file)   # (N, 4, 4)
    hand_dis = np.load(hand_dis_file) if os.path.exists(hand_dis_file) else None
    
    return ibs_voxels, w2h_trans, hand_dis


def load_scene_pc(scene_path: str, scene_id: int, camera: str = 'realsense', view: str = '0000'):
    """Load scene point cloud."""
    scene_name = f'scene_{str(scene_id).zfill(4)}'
    pc_path = os.path.join(scene_path, scene_name, camera, 'points', f'{view}.npz')
    
    if not os.path.exists(pc_path):
        print(f"Scene point cloud not found: {pc_path}")
        return None
    
    data = np.load(pc_path)
    return data['xyz']


def voxel_to_points(voxel_mask: np.ndarray):
    """Convert voxel mask to physical coordinates."""
    indices = np.where(voxel_mask)
    voxel_coords = np.stack(indices, axis=-1)  # (N, 3)
    physical_coords = voxel_coords * RESOLUTION + np.array([-BOUND, -BOUND, -BOUND])
    return physical_coords


def transform_points(points: np.ndarray, transform: np.ndarray):
    """Apply 4x4 transformation to points."""
    if points.shape[0] == 0:
        return points
    points_hom = np.hstack([points, np.ones((points.shape[0], 1))])
    return (transform @ points_hom.T).T[:, :3]


def visualize_ibs(args):
    """Main visualization function."""
    # Load IBS data
    ibs_voxels, w2h_trans, hand_dis = load_ibs_data(args.ibs_path, args.scene_id)
    if ibs_voxels is None:
        return
    
    n_grasps = ibs_voxels.shape[0]
    print(f"Scene {args.scene_id}: {n_grasps} IBS data found")
    
    # Determine which grasps to visualize
    if args.grasp_indices:
        grasp_indices = [i for i in args.grasp_indices if i < n_grasps]
    else:
        # Visualize last N grasps by default
        start_idx = max(0, n_grasps - args.max_grasps)
        grasp_indices = list(range(start_idx, n_grasps))
    
    fig = go.Figure()
    
    # Add scene point cloud if requested
    if args.with_scene_pc:
        scene_pc = load_scene_pc(args.scene_path, args.scene_id, args.camera, args.view)
        if scene_pc is not None:
            fig.add_trace(go.Scatter3d(
                x=scene_pc[:, 0],
                y=scene_pc[:, 1],
                z=scene_pc[:, 2],
                mode='markers',
                marker=dict(size=1, color='gray', opacity=0.3),
                name='Scene PC'
            ))
    
    # Color palette
    colors = px.colors.qualitative.Set1
    
    for idx, grasp_id in enumerate(grasp_indices):
        voxel = ibs_voxels[grasp_id]  # (40, 40, 40, 3)
        w2h = w2h_trans[grasp_id]     # (4, 4)
        h2w = np.linalg.inv(w2h)      # Transform to world
        
        # Extract masks
        ibs_mask = voxel[..., 0].astype(bool)
        contact_mask = voxel[..., 1].astype(bool)
        thumb_mask = voxel[..., 2].astype(bool)
        
        # Get points in hand frame
        ibs_only = ibs_mask & ~contact_mask & ~thumb_mask
        contact_only = contact_mask & ~thumb_mask
        
        # Convert to world coordinates
        ibs_pts = transform_points(voxel_to_points(ibs_only), h2w)
        contact_pts = transform_points(voxel_to_points(contact_only), h2w)
        thumb_pts = transform_points(voxel_to_points(thumb_mask), h2w)
        
        color = colors[idx % len(colors)]
        
        # Add traces
        if len(ibs_pts) > 0:
            fig.add_trace(go.Scatter3d(
                x=ibs_pts[:, 0], y=ibs_pts[:, 1], z=ibs_pts[:, 2],
                mode='markers',
                marker=dict(size=3, color=color, opacity=0.5),
                name=f'IBS #{grasp_id}'
            ))
        
        if len(contact_pts) > 0:
            fig.add_trace(go.Scatter3d(
                x=contact_pts[:, 0], y=contact_pts[:, 1], z=contact_pts[:, 2],
                mode='markers',
                marker=dict(size=5, color='red', opacity=0.8),
                name=f'Contact #{grasp_id}'
            ))
        
        if len(thumb_pts) > 0:
            fig.add_trace(go.Scatter3d(
                x=thumb_pts[:, 0], y=thumb_pts[:, 1], z=thumb_pts[:, 2],
                mode='markers',
                marker=dict(size=5, color='blue', opacity=0.8),
                name=f'Thumb #{grasp_id}'
            ))
    
    # Layout
    fig.update_layout(
        title=f'IBS Visualization - Scene {args.scene_id} ({len(grasp_indices)} grasps)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        showlegend=True
    )
    
    # Save or show
    if args.save_plot:
        output_path = os.path.join(args.ibs_path, f'vis_scene_{args.scene_id}.html')
        fig.write_html(output_path)
        print(f"Saved to {output_path}")
    else:
        fig.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize IBS voxels for a scene")
    parser.add_argument('--scene_id', type=int, required=True, help='Scene ID (e.g., 55 for scene_0055)')
    parser.add_argument('--ibs_path', type=str, default=DEFAULT_IBS_PATH, help='Path to IBS data directory')
    parser.add_argument('--scene_path', type=str, default=DEFAULT_SCENE_PATH, help='Path to scene data')
    parser.add_argument('--grasp_indices', type=int, nargs='+', default=None, help='Specific grasp indices to visualize')
    parser.add_argument('--max_grasps', type=int, default=5, help='Max grasps to visualize (if indices not specified)')
    parser.add_argument('--with_scene_pc', action='store_true', help='Include scene point cloud')
    parser.add_argument('--camera', type=str, default='realsense', help='Camera type')
    parser.add_argument('--view', type=str, default='0000', help='View ID')
    parser.add_argument('--save_plot', action='store_true', help='Save as HTML instead of showing')
    
    args = parser.parse_args()
    visualize_ibs(args)


if __name__ == "__main__":
    main()