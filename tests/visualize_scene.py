"""
Visualize GraspNet Scene with Object Meshes.

Load scene annotation and visualize object meshes with their poses.
Uses data from data/DexGraspNet2.0/scenes and data/DexGraspNet2.0/meshdata.

Usage:
    python tests/visualize_scene.py --scene_id 55
    python tests/visualize_scene.py --scene_id 55 --camera kinect
"""

import os
import sys

import argparse
import numpy as np
import transforms3d
import trimesh as tm
import xml.etree.ElementTree as ET
import plotly.graph_objects as go
import plotly.express as px

from cadgrasp.paths import project_path

# Default paths (DexGraspNet2.0 structure)
DEFAULT_SCENE_PATH = project_path('data/DexGraspNet2.0/scenes')
DEFAULT_MESH_PATH = project_path('data/DexGraspNet2.0/meshdata')


def load_scene_objects(scene_path: str, scene_id: str, camera: str = 'kinect', view: str = '0000'):
    """
    Load object poses from scene annotation.
    
    Args:
        scene_path: Base path to scene data
        scene_id: Scene ID (e.g., '0055')
        camera: Camera type ('kinect' or 'realsense')
        view: View ID (e.g., '0000')
    
    Returns:
        Dict mapping object_code to pose dict with 'translation' and 'rotation'
    """
    scene_name = f'scene_{scene_id}'
    scene_dir = os.path.join(scene_path, scene_name)
    
    # Load extrinsics (camera to world transform)
    extrinsics_path = os.path.join(scene_dir, camera, 'cam0_wrt_table.npy')
    if not os.path.exists(extrinsics_path):
        print(f"Extrinsics not found: {extrinsics_path}")
        return {}
    extrinsics = np.load(extrinsics_path)
    
    # Load annotation
    annotation_path = os.path.join(scene_dir, camera, 'annotations', f'{view}.xml')
    if not os.path.exists(annotation_path):
        print(f"Annotation not found: {annotation_path}")
        return {}
    
    annotation = ET.parse(annotation_path)
    
    # Parse objects
    object_pose_dict = {}
    for obj in annotation.findall('obj'):
        object_code = str(int(obj.find('obj_id').text)).zfill(3)
        translation = np.array([float(x) for x in obj.find('pos_in_world').text.split()])
        rotation_quat = np.array([float(x) for x in obj.find('ori_in_world').text.split()])
        rotation = transforms3d.quaternions.quat2mat(rotation_quat)
        
        # Transform to camera frame
        object_pose_dict[object_code] = dict(
            translation=extrinsics[:3, :3] @ translation + extrinsics[:3, 3],
            rotation=extrinsics[:3, :3] @ rotation,
        )
    
    return dict(sorted(object_pose_dict.items()))


def load_object_meshes(mesh_path: str, object_codes: list):
    """Load object meshes for given object codes."""
    meshes = {}
    for code in object_codes:
        mesh_file = os.path.join(mesh_path, code, 'simplified.obj')
        if os.path.exists(mesh_file):
            meshes[code] = tm.load(mesh_file)
        else:
            # Try alternative mesh file
            mesh_file = os.path.join(mesh_path, code, 'nontextured.ply')
            if os.path.exists(mesh_file):
                meshes[code] = tm.load(mesh_file)
            else:
                print(f"Mesh not found for object {code}")
    return meshes


def visualize_scene(args):
    """Main visualization function."""
    # Load objects
    object_poses = load_scene_objects(args.scene_path, args.scene_id, args.camera, args.view)
    if not object_poses:
        return
    
    print(f"Scene {args.scene_id}: {len(object_poses)} objects")
    
    # Load meshes
    meshes = load_object_meshes(args.mesh_path, list(object_poses.keys()))
    
    # Create plotly figure
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    
    for idx, (object_code, pose) in enumerate(object_poses.items()):
        if object_code not in meshes:
            continue
        
        mesh = meshes[object_code]
        vertices = mesh.vertices @ pose['rotation'].T + pose['translation']
        faces = mesh.faces
        
        color = colors[idx % len(colors)]
        
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=color,
            opacity=0.9,
            hoverinfo='text',
            text=[f'Object {object_code}'] * len(faces),
            name=f'Object {object_code}'
        ))
    
    # Layout
    fig.update_layout(
        title=f'Scene {args.scene_id} ({args.camera}, view {args.view})',
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
        ),
        showlegend=True
    )
    
    if args.output_path:
        fig.write_html(args.output_path)
        print(f"Saved to {args.output_path}")
    else:
        fig.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize GraspNet scene with object meshes")
    parser.add_argument('--scene_id', type=str, required=True, help='Scene ID (e.g., 0055)')
    parser.add_argument('--camera', type=str, default='kinect', choices=['kinect', 'realsense'])
    parser.add_argument('--view', type=str, default='0000', help='View ID')
    parser.add_argument('--scene_path', type=str, default=DEFAULT_SCENE_PATH)
    parser.add_argument('--mesh_path', type=str, default=DEFAULT_MESH_PATH)
    parser.add_argument('--output_path', type=str, default=None, help='Output HTML file path')
    
    args = parser.parse_args()
    visualize_scene(args)


if __name__ == '__main__':
    main()
