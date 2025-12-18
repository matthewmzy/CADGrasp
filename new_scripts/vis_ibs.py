import os
import argparse
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch

def visualize_ibs_voxels(args):
    """
    Visualize IBS voxels for a given scene_id using Plotly.
    
    Args:
        args: Object containing ibs_save_path, scene_id, max_grasps_to_visualize, and save_plot.
    """
    # Paths to saved .npy files
    ibs_save_path = args.ibs_save_path
    scene_id = args.scene_id
    scene_name = f'scene_{str(scene_id).zfill(4)}'
    ibs_path = os.path.join(ibs_save_path, 'ibs', f'{scene_name}.npy')
    w2h_trans_path = os.path.join(ibs_save_path, 'w2h_trans', f'{scene_name}.npy')
    hand_dis_path = os.path.join(ibs_save_path, 'hand_dis', f'{scene_name}.npy')

    # Check if files exist
    if not all(os.path.exists(p) for p in [ibs_path, w2h_trans_path, hand_dis_path]):
        print(f"Data for scene {scene_id} not found at {ibs_save_path}")
        return

    # Load data
    ibs_con_voxels = np.load(ibs_path)  # Shape: (batch_size, 40, 40, 40, 3)
    w2h_trans = np.load(w2h_trans_path)  # Shape: (batch_size, 4, 4)
    hand_dis = np.load(hand_dis_path)    # Shape: (batch_size, 40, 40, 40)

    # Voxel grid parameters (from original script)
    bound = 0.1
    resolution = 0.005
    grid_size = 40

    # Create voxel grid coordinates
    x, y, z = np.mgrid[0:grid_size, 0:grid_size, 0:grid_size]
    voxel_coords = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T  # Shape: (40*40*40, 3)
    # Convert voxel indices to physical coordinates
    physical_coords = voxel_coords * resolution + np.array([-bound, -bound, -bound])

    # Initialize Plotly figure
    fig = go.Figure()

    # Process each grasp in the batch
    batch_size = ibs_con_voxels.shape[0]
    # for grasp_id in range(min(batch_size, args.max_grasps_to_visualize)):
    for grasp_id in range(batch_size-args.max_grasps_to_visualize-1, batch_size-1):
        # Extract masks
        ibs_mask = ibs_con_voxels[grasp_id, :, :, :, 0].ravel().astype(bool)  # IBS mask
        contact_mask = ibs_con_voxels[grasp_id, :, :, :, 1].ravel().astype(bool)  # Contact mask
        thumb_contact_mask = ibs_con_voxels[grasp_id, :, :, :, 2].ravel().astype(bool)  # Thumb contact mask

        # Get corresponding points
        ibs_points = physical_coords[ibs_mask & ~contact_mask]  # IBS points excluding contact
        contact_points = physical_coords[contact_mask & ~thumb_contact_mask]  # Contact points excluding thumb
        thumb_points = physical_coords[thumb_contact_mask]  # Thumb contact points

        # Transform points to world coordinates
        w2h_transform = w2h_trans[grasp_id]  # Shape: (4, 4)
        h2w_transform = np.linalg.inv(w2h_transform)  # Hand-to-world transform

        def transform_points(points, transform):
            if points.shape[0] == 0:
                return points
            points_hom = np.hstack([points, np.ones((points.shape[0], 1))])  # Homogeneous coordinates
            points_transformed = (transform @ points_hom.T).T[:, :3]  # Apply transform
            return points_transformed

        ibs_points_world = transform_points(ibs_points, h2w_transform)
        contact_points_world = transform_points(contact_points, h2w_transform)
        thumb_points_world = transform_points(thumb_points, h2w_transform)

        # Define colors
        colors = px.colors.sequential.Plasma
        np.random.seed(0)
        idxs = np.random.randint(0, len(colors), 3)
        ibs_color, contact_color, thumb_color = [colors[i] for i in idxs]

        # Add scatter plots to figure
        if ibs_points_world.shape[0] > 0:
            fig.add_trace(go.Scatter3d(
                x=ibs_points_world[:, 0],
                y=ibs_points_world[:, 1],
                z=ibs_points_world[:, 2],
                mode='markers',
                marker=dict(size=4, color=ibs_color, opacity=0.6),
                name=f'IBS (Grasp {grasp_id})'
            ))

        if contact_points_world.shape[0] > 0:
            fig.add_trace(go.Scatter3d(
                x=contact_points_world[:, 0],
                y=contact_points_world[:, 1],
                z=contact_points_world[:, 2],
                mode='markers',
                marker=dict(size=7, color=contact_color, opacity=0.8),
                name=f'Contact (Grasp {grasp_id})'
            ))

        if thumb_points_world.shape[0] > 0:
            fig.add_trace(go.Scatter3d(
                x=thumb_points_world[:, 0],
                y=thumb_points_world[:, 1],
                z=thumb_points_world[:, 2],
                mode='markers',
                marker=dict(size=7, color=thumb_color, opacity=0.8),
                name=f'Thumb Contact (Grasp {grasp_id})'
            ))

    # Update layout
    fig.update_layout(
        title=f'IBS Voxel Visualization for Scene {scene_id}',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        showlegend=True
    )

    # Show or save the figure
    if args.save_plot:
        output_path = os.path.join(ibs_save_path, f'ibs_visualization_scene_{scene_id}.html')
        fig.write_html(output_path)
        print(f"Visualization saved to {output_path}")
    else:
        fig.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize IBS voxels for a given scene using Plotly.")
    parser.add_argument(
        "--ibs_save_path",
        type=str,
        default='IBSGrasp/fpsTargetDataNew',
        help="Path to the directory containing saved IBS data (ibs, w2h_trans, hand_dis subdirectories)."
    )
    parser.add_argument(
        "--scene_id",
        type=int,
        required=True,
        help="Scene ID to visualize (e.g., 1 for scene_0001)."
    )
    parser.add_argument(
        "--max_grasps_to_visualize",
        type=int,
        default=5,
        help="Maximum number of grasps to visualize (default: 5)."
    )
    parser.add_argument(
        "--save_plot",
        action="store_true",
        help="If set, save the visualization as an HTML file instead of displaying it."
    )
    args = parser.parse_args()
    visualize_ibs_voxels(args)

if __name__ == "__main__":
    main()