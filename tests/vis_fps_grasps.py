"""
Visualize FPS-Sampled Grasps.

Visualize grasp points after FPS sampling to verify the sampling quality.
Shows grasp points distribution for a scene.

Usage:
    python tests/vis_fps_grasps.py --scene_id 55
    python tests/vis_fps_grasps.py --scene_id 55 --compare_raw
    python tests/vis_fps_grasps.py --scene_id 55 --object_id 046
"""

import os
import sys

import argparse
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from glob import glob

from cadgrasp.paths import project_path

# Default paths
DEFAULT_GRASP_PATH = project_path('data/DexGraspNet2.0/dex_grasps_new')
DEFAULT_SUCCESS_PATH = project_path('data/DexGraspNet2.0/dex_grasps_success_indices')
DEFAULT_FPS_PATH = project_path('data/DexGraspNet2.0/fps_sampled_indices')


def load_grasp_points(scene_id: int, robot_name: str = 'leap_hand',
                      grasp_path: str = DEFAULT_GRASP_PATH,
                      success_path: str = DEFAULT_SUCCESS_PATH,
                      fps_path: str = DEFAULT_FPS_PATH,
                      use_success: bool = True,
                      use_fps: bool = True,
                      object_id: str = None):
    """
    Load grasp points with optional filtering.
    
    Returns:
        Dict mapping object_id to dict with 'raw', 'success', 'fps' point arrays
    """
    scene_name = f'scene_{str(scene_id).zfill(4)}'
    scene_grasp_dir = os.path.join(grasp_path, scene_name, robot_name)
    
    if not os.path.exists(scene_grasp_dir):
        print(f"Grasp directory not found: {scene_grasp_dir}")
        return {}
    
    result = {}
    
    for npz_file in sorted(os.listdir(scene_grasp_dir)):
        if not npz_file.endswith('.npz'):
            continue
        
        obj_id = npz_file.replace('.npz', '')
        if object_id and obj_id != object_id:
            continue
        
        # Load raw grasps
        raw_data = np.load(os.path.join(scene_grasp_dir, npz_file))
        raw_points = raw_data['point']
        
        result[obj_id] = {'raw': raw_points}
        
        # Load success indices
        success_file = os.path.join(success_path, scene_name, robot_name, npz_file)
        if os.path.exists(success_file):
            success_data = np.load(success_file)
            success_indices = success_data['success_indices']
            result[obj_id]['success'] = raw_points[success_indices]
            result[obj_id]['success_rate'] = len(success_indices) / len(raw_points)
        else:
            result[obj_id]['success'] = raw_points
            result[obj_id]['success_rate'] = 1.0
        
        # Load FPS indices
        fps_file = os.path.join(fps_path, scene_name, robot_name, npz_file)
        if os.path.exists(fps_file):
            fps_data = np.load(fps_file)
            fps_indices = fps_data['fps_indices']
            # FPS indices are relative to success-filtered grasps
            result[obj_id]['fps'] = result[obj_id]['success'][fps_indices]
            result[obj_id]['fps_ratio'] = len(fps_indices) / len(result[obj_id]['success']) if len(result[obj_id]['success']) > 0 else 0
        else:
            result[obj_id]['fps'] = None
            result[obj_id]['fps_ratio'] = 0
    
    return result


def visualize_fps_grasps(args):
    """Main visualization function."""
    data = load_grasp_points(
        args.scene_id, args.robot_name,
        args.grasp_path, args.success_path, args.fps_path,
        object_id=args.object_id
    )
    
    if not data:
        print("No data found")
        return
    
    # Print statistics
    print(f"\nScene {args.scene_id} Statistics:")
    print("-" * 60)
    total_raw = 0
    total_success = 0
    total_fps = 0
    
    for obj_id, obj_data in data.items():
        n_raw = len(obj_data['raw'])
        n_success = len(obj_data['success'])
        n_fps = len(obj_data['fps']) if obj_data['fps'] is not None else 0
        total_raw += n_raw
        total_success += n_success
        total_fps += n_fps
        
        print(f"Object {obj_id}: Raw={n_raw}, Success={n_success} ({obj_data['success_rate']*100:.1f}%), "
              f"FPS={n_fps} ({obj_data['fps_ratio']*100:.1f}%)")
    
    print("-" * 60)
    print(f"Total: Raw={total_raw}, Success={total_success}, FPS={total_fps}")
    
    # Create visualization
    fig = go.Figure()
    colors = px.colors.qualitative.Set1
    
    for idx, (obj_id, obj_data) in enumerate(data.items()):
        color = colors[idx % len(colors)]
        
        if args.compare_raw:
            # Show raw points (smaller, more transparent)
            fig.add_trace(go.Scatter3d(
                x=obj_data['raw'][:, 0],
                y=obj_data['raw'][:, 1],
                z=obj_data['raw'][:, 2],
                mode='markers',
                marker=dict(size=2, color='gray', opacity=0.2),
                name=f'{obj_id} Raw ({len(obj_data["raw"])})'
            ))
        
        if args.show_success and obj_data['success'] is not None:
            # Show success points
            fig.add_trace(go.Scatter3d(
                x=obj_data['success'][:, 0],
                y=obj_data['success'][:, 1],
                z=obj_data['success'][:, 2],
                mode='markers',
                marker=dict(size=3, color=color, opacity=0.4),
                name=f'{obj_id} Success ({len(obj_data["success"])})'
            ))
        
        if obj_data['fps'] is not None and len(obj_data['fps']) > 0:
            # Show FPS sampled points (larger, more visible)
            fig.add_trace(go.Scatter3d(
                x=obj_data['fps'][:, 0],
                y=obj_data['fps'][:, 1],
                z=obj_data['fps'][:, 2],
                mode='markers',
                marker=dict(size=6, color=color, opacity=0.9, 
                           line=dict(color='black', width=1)),
                name=f'{obj_id} FPS ({len(obj_data["fps"])})'
            ))
    
    fig.update_layout(
        title=f'FPS Grasp Sampling - Scene {args.scene_id}',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y', 
            zaxis_title='Z',
            aspectmode='data'
        ),
        showlegend=True
    )
    
    if args.output_path:
        fig.write_html(args.output_path)
        print(f"\nSaved to {args.output_path}")
    else:
        fig.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize FPS-sampled grasp points")
    parser.add_argument('--scene_id', type=int, required=True, help='Scene ID (e.g., 55)')
    parser.add_argument('--robot_name', type=str, default='leap_hand')
    parser.add_argument('--object_id', type=str, default=None, help='Specific object ID (e.g., 046)')
    parser.add_argument('--compare_raw', action='store_true', help='Also show raw grasp points')
    parser.add_argument('--show_success', action='store_true', help='Show success-filtered points')
    parser.add_argument('--grasp_path', type=str, default=DEFAULT_GRASP_PATH)
    parser.add_argument('--success_path', type=str, default=DEFAULT_SUCCESS_PATH)
    parser.add_argument('--fps_path', type=str, default=DEFAULT_FPS_PATH)
    parser.add_argument('--output_path', type=str, default=None, help='Output HTML file')
    
    args = parser.parse_args()
    visualize_fps_grasps(args)


if __name__ == '__main__':
    main()
