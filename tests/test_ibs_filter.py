"""
Test script for IBSFilter visualization.

This script visualizes:
1. IBS point clouds with estimated normals
2. Contact point clusters with friction cones
3. Thumb contact friction cone (highlighted)
4. Force-closure relationships between clusters

Usage:
    python tests/test_ibs_filter.py --scene_id 0 --num_ibs 5
    python tests/test_ibs_filter.py --scene_id 0 --ibs_indices 0,10,20,30,40 --mu 0.8
"""

import os
import sys
import argparse
import numpy as np
import torch
from termcolor import cprint
from plotly import graph_objects as go
import plotly.express as px
from sklearn.cluster import DBSCAN

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cadgrasp.optimizer.ibs_func import (
    IBSFilter, 
    estimate_normals, 
    cluster_points,
    is_in_friction_cone,
    compute_force_closure_score,
    generate_friction_cone_vertices,
    FilterConfig
)
from cadgrasp.ibs.utils.ibs_repr import IBSConfig


def load_gt_ibs_data(scene_id: int, ibs_indices: list, data_root: str = 'data/ibsdata'):
    """Load ground truth IBS data."""
    scene_name = f'scene_{scene_id:04d}'
    ibs_path = os.path.join(data_root, 'ibs', f'{scene_name}.npy')
    
    if not os.path.exists(ibs_path):
        raise FileNotFoundError(f"IBS file not found: {ibs_path}")
    
    ibs_all = np.load(ibs_path)  # (N, 40, 40, 40, 3) bool
    cprint(f"[Test] Loaded IBS data: {ibs_all.shape}", 'cyan')
    
    ibs_voxels = ibs_all[ibs_indices]
    return ibs_voxels


def devoxelize_gt_ibs(ibs_voxel: np.ndarray) -> tuple:
    """
    Convert ground truth IBS voxel (40,40,40,3 bool) to point clouds.
    
    Returns:
        ibs_occu: (N1, 3) occupancy points
        ibs_cont: (N2, 3) contact points
        ibs_thumb: (N3, 3) thumb contact points
    """
    config = IBSConfig()
    bound = config.bound
    resolution = config.resolution
    
    # Create coordinate grid
    grid_x, grid_y, grid_z = np.mgrid[
        -bound:bound:resolution,
        -bound:bound:resolution,
        -bound:bound:resolution
    ]
    points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
    
    # Get masks
    occu = ibs_voxel[:, :, :, 0].ravel()
    cont = ibs_voxel[:, :, :, 1].ravel()
    thumb = ibs_voxel[:, :, :, 2].ravel()
    
    # Exclusive masks
    occu_only = occu & ~cont & ~thumb
    cont_only = cont & ~thumb
    
    ibs_occu = points[occu_only]
    ibs_cont = points[cont_only]
    ibs_thumb = points[thumb]
    
    return ibs_occu, ibs_cont, ibs_thumb


def visualize_ibs_with_friction_cones(
    ibs_voxels: np.ndarray,
    mu: float = 1.0,
    output_dir: str = 'logs/test_ibs_filter'
):
    """
    Create detailed visualization of IBS with friction cones.
    
    Args:
        ibs_voxels: (N, 40, 40, 40, 3) IBS voxels
        mu: Friction coefficient
        output_dir: Directory to save HTML files
    """
    os.makedirs(output_dir, exist_ok=True)
    num_ibs = ibs_voxels.shape[0]
    colors = px.colors.qualitative.Plotly
    
    for idx in range(num_ibs):
        cprint(f"[Test] Processing IBS {idx + 1}/{num_ibs}...", 'cyan')
        
        # Devoxelize
        ibs_occu, ibs_cont, ibs_thumb = devoxelize_gt_ibs(ibs_voxels[idx])
        
        if ibs_occu.shape[0] < 5 or ibs_cont.shape[0] < 5 or ibs_thumb.shape[0] < 5:
            cprint(f"[Test] IBS {idx} has insufficient points, skipping", 'yellow')
            continue
        
        # Estimate normals
        all_points = np.concatenate([ibs_occu, ibs_cont, ibs_thumb], axis=0)
        all_normals, avg_dist = estimate_normals(all_points)
        
        n1, n2 = ibs_occu.shape[0], ibs_cont.shape[0]
        ibs_occu_n = np.concatenate([ibs_occu, all_normals[:n1]], axis=1)
        ibs_cont_n = np.concatenate([ibs_cont, all_normals[n1:n1+n2]], axis=1)
        ibs_thumb_n = np.concatenate([ibs_thumb, all_normals[n1+n2:]], axis=1)
        
        # Cluster contact points
        eps = avg_dist * 2.0
        clusters, cluster_indices = cluster_points(ibs_cont, threshold=eps)
        
        # Thumb statistics
        thumb_mean_pos = np.mean(ibs_thumb, axis=0)
        thumb_mean_norm = -np.mean(ibs_thumb_n[:, 3:6], axis=0)
        thumb_mean_norm = thumb_mean_norm / (np.linalg.norm(thumb_mean_norm) + 1e-8)
        
        # Create figure
        fig = go.Figure()
        
        # === IBS Occupancy (gray, transparent) ===
        fig.add_trace(go.Scatter3d(
            x=ibs_occu[:, 0], y=ibs_occu[:, 1], z=ibs_occu[:, 2],
            mode="markers",
            marker=dict(size=2, color="gray", opacity=0.2),
            name="IBS Occupancy",
        ))
        
        # === Thumb Contact Points (blue, highlighted) ===
        fig.add_trace(go.Scatter3d(
            x=ibs_thumb[:, 0], y=ibs_thumb[:, 1], z=ibs_thumb[:, 2],
            mode="markers",
            marker=dict(size=5, color="blue"),
            name="Thumb Contact",
        ))
        
        # === Thumb Mean Point (large diamond) ===
        fig.add_trace(go.Scatter3d(
            x=[thumb_mean_pos[0]], y=[thumb_mean_pos[1]], z=[thumb_mean_pos[2]],
            mode="markers",
            marker=dict(size=15, color="blue", symbol="diamond", line=dict(width=2, color='white')),
            name="Thumb Center",
        ))
        
        # === Thumb Normal (thick line) ===
        normal_end = thumb_mean_pos + 0.08 * thumb_mean_norm
        fig.add_trace(go.Scatter3d(
            x=[thumb_mean_pos[0], normal_end[0]],
            y=[thumb_mean_pos[1], normal_end[1]],
            z=[thumb_mean_pos[2], normal_end[2]],
            mode="lines",
            line=dict(color="blue", width=12),
            name="Thumb Normal (Axis)",
        ))
        
        # === Thumb Friction Cone (highlighted, larger) ===
        thumb_cone_verts = generate_friction_cone_vertices(
            thumb_mean_pos, thumb_mean_norm, height=0.08, mu=mu, num_sides=30
        )
        triangles = [[0, i, i+1] for i in range(1, len(thumb_cone_verts)-1)]
        triangles.append([0, len(thumb_cone_verts)-1, 1])
        fig.add_trace(go.Mesh3d(
            x=thumb_cone_verts[:, 0], y=thumb_cone_verts[:, 1], z=thumb_cone_verts[:, 2],
            i=[t[0] for t in triangles],
            j=[t[1] for t in triangles],
            k=[t[2] for t in triangles],
            color="cyan", opacity=0.5,
            name="Thumb Friction Cone (μ={:.1f})".format(mu),
        ))
        
        # === Contact Clusters with Friction Cones ===
        cluster_info = []
        for c_idx, (cluster, indices) in enumerate(zip(clusters, cluster_indices)):
            if cluster.shape[0] == 0:
                continue
            
            color = colors[c_idx % len(colors)]
            
            cluster_mean_pos = np.mean(cluster, axis=0)
            cluster_mean_norm = -np.mean(ibs_cont_n[indices, 3:6], axis=0)
            cluster_mean_norm = cluster_mean_norm / (np.linalg.norm(cluster_mean_norm) + 1e-8)
            
            # Check force-closure
            in_thumb_cone = is_in_friction_cone(cluster_mean_pos, thumb_mean_norm, thumb_mean_pos, mu)
            in_cluster_cone = is_in_friction_cone(thumb_mean_pos, cluster_mean_norm, cluster_mean_pos, mu)
            fc_score = compute_force_closure_score(cluster_mean_pos, cluster_mean_norm, thumb_mean_pos, thumb_mean_norm)
            
            is_valid = in_thumb_cone and in_cluster_cone
            prefix = "✓" if is_valid else "✗"
            
            cluster_info.append({
                'idx': c_idx,
                'valid': is_valid,
                'score': fc_score,
                'pos': cluster_mean_pos,
                'norm': cluster_mean_norm
            })
            
            # Cluster points
            fig.add_trace(go.Scatter3d(
                x=cluster[:, 0], y=cluster[:, 1], z=cluster[:, 2],
                mode="markers",
                marker=dict(size=4, color=color),
                name=f"{prefix} Cluster {c_idx} (n={len(cluster)})",
            ))
            
            # Cluster center
            fig.add_trace(go.Scatter3d(
                x=[cluster_mean_pos[0]], y=[cluster_mean_pos[1]], z=[cluster_mean_pos[2]],
                mode="markers",
                marker=dict(size=10, color=color, symbol="diamond"),
                name=f"Cluster {c_idx} Center",
            ))
            
            # Cluster normal
            normal_end = cluster_mean_pos + 0.05 * cluster_mean_norm
            fig.add_trace(go.Scatter3d(
                x=[cluster_mean_pos[0], normal_end[0]],
                y=[cluster_mean_pos[1], normal_end[1]],
                z=[cluster_mean_pos[2], normal_end[2]],
                mode="lines",
                line=dict(color=color, width=8),
                name=f"Cluster {c_idx} Normal",
            ))
            
            # Cluster friction cone
            cone_verts = generate_friction_cone_vertices(
                cluster_mean_pos, cluster_mean_norm, height=0.05, mu=mu
            )
            triangles = [[0, i, i+1] for i in range(1, len(cone_verts)-1)]
            triangles.append([0, len(cone_verts)-1, 1])
            
            opacity = 0.4 if is_valid else 0.15
            fig.add_trace(go.Mesh3d(
                x=cone_verts[:, 0], y=cone_verts[:, 1], z=cone_verts[:, 2],
                i=[t[0] for t in triangles],
                j=[t[1] for t in triangles],
                k=[t[2] for t in triangles],
                color=color, opacity=opacity,
                name=f"Cluster {c_idx} Cone {'(valid)' if is_valid else ''}",
            ))
            
            # Draw connection line to thumb for valid clusters
            if is_valid:
                fig.add_trace(go.Scatter3d(
                    x=[cluster_mean_pos[0], thumb_mean_pos[0]],
                    y=[cluster_mean_pos[1], thumb_mean_pos[1]],
                    z=[cluster_mean_pos[2], thumb_mean_pos[2]],
                    mode="lines",
                    line=dict(color="green", width=4, dash='dot'),
                    name=f"Force-Closure Link {c_idx}",
                ))
        
        # === Add annotations ===
        valid_clusters = [c for c in cluster_info if c['valid']]
        invalid_clusters = [c for c in cluster_info if not c['valid']]
        
        title_text = f"IBS {idx} - μ={mu:.1f}<br>"
        title_text += f"<span style='color:green'>Valid clusters: {len(valid_clusters)}</span> | "
        title_text += f"<span style='color:red'>Invalid: {len(invalid_clusters)}</span><br>"
        if valid_clusters:
            best_score = max(c['score'] for c in valid_clusters)
            title_text += f"Best force-closure score: {best_score:.3f}"
        
        # === Layout ===
        fig.update_layout(
            title=dict(text=title_text, x=0.5),
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y", 
                zaxis_title="Z",
                aspectmode="data",
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.0)
                )
            ),
            legend=dict(
                yanchor="top", y=0.99, 
                xanchor="left", x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            ),
            showlegend=True,
            width=1400,
            height=900,
        )
        
        # Save HTML
        save_path = os.path.join(output_dir, f'ibs_{idx}_friction_cones.html')
        fig.write_html(save_path)
        cprint(f"[Test] Saved: {save_path}", 'green')
        
        # Print summary
        cprint(f"[Test] IBS {idx} Summary:", 'cyan')
        cprint(f"  - Occupancy points: {ibs_occu.shape[0]}", 'white')
        cprint(f"  - Contact points: {ibs_cont.shape[0]} ({len(clusters)} clusters)", 'white')
        cprint(f"  - Thumb points: {ibs_thumb.shape[0]}", 'white')
        cprint(f"  - Valid clusters: {len(valid_clusters)}/{len(clusters)}", 
               'green' if valid_clusters else 'red')
        for c in cluster_info:
            status = "✓" if c['valid'] else "✗"
            cprint(f"    {status} Cluster {c['idx']}: score={c['score']:.3f}", 
                   'green' if c['valid'] else 'yellow')


def create_combined_visualization(
    ibs_voxels: np.ndarray,
    mu: float = 1.0,
    output_dir: str = 'logs/test_ibs_filter'
):
    """
    Create a combined visualization showing all IBS side by side.
    """
    os.makedirs(output_dir, exist_ok=True)
    num_ibs = ibs_voxels.shape[0]
    colors = px.colors.qualitative.Plotly
    
    fig = go.Figure()
    
    for idx in range(num_ibs):
        offset = np.array([idx * 0.25, 0, 0])  # Horizontal offset
        
        # Devoxelize
        ibs_occu, ibs_cont, ibs_thumb = devoxelize_gt_ibs(ibs_voxels[idx])
        
        if ibs_occu.shape[0] < 5 or ibs_cont.shape[0] < 5 or ibs_thumb.shape[0] < 5:
            continue
        
        # Estimate normals
        all_points = np.concatenate([ibs_occu, ibs_cont, ibs_thumb], axis=0)
        all_normals, avg_dist = estimate_normals(all_points)
        
        n1, n2 = ibs_occu.shape[0], ibs_cont.shape[0]
        ibs_thumb_n = np.concatenate([ibs_thumb, all_normals[n1+n2:]], axis=1)
        
        # Thumb
        thumb_mean_pos = np.mean(ibs_thumb, axis=0) + offset
        thumb_mean_norm = -np.mean(ibs_thumb_n[:, 3:6], axis=0)
        thumb_mean_norm = thumb_mean_norm / (np.linalg.norm(thumb_mean_norm) + 1e-8)
        
        # Add IBS points
        fig.add_trace(go.Scatter3d(
            x=ibs_occu[:, 0] + offset[0], 
            y=ibs_occu[:, 1] + offset[1], 
            z=ibs_occu[:, 2] + offset[2],
            mode="markers",
            marker=dict(size=2, color="gray", opacity=0.15),
            name=f"IBS {idx} Occu",
            legendgroup=f"ibs_{idx}",
            showlegend=(idx == 0)
        ))
        
        fig.add_trace(go.Scatter3d(
            x=ibs_cont[:, 0] + offset[0], 
            y=ibs_cont[:, 1] + offset[1], 
            z=ibs_cont[:, 2] + offset[2],
            mode="markers",
            marker=dict(size=3, color=colors[idx % len(colors)]),
            name=f"IBS {idx} Contact",
            legendgroup=f"ibs_{idx}"
        ))
        
        fig.add_trace(go.Scatter3d(
            x=ibs_thumb[:, 0] + offset[0], 
            y=ibs_thumb[:, 1] + offset[1], 
            z=ibs_thumb[:, 2] + offset[2],
            mode="markers",
            marker=dict(size=5, color="blue"),
            name=f"IBS {idx} Thumb",
            legendgroup=f"ibs_{idx}"
        ))
        
        # Thumb friction cone
        thumb_cone_verts = generate_friction_cone_vertices(
            thumb_mean_pos, thumb_mean_norm, height=0.06, mu=mu
        )
        triangles = [[0, i, i+1] for i in range(1, len(thumb_cone_verts)-1)]
        triangles.append([0, len(thumb_cone_verts)-1, 1])
        fig.add_trace(go.Mesh3d(
            x=thumb_cone_verts[:, 0], 
            y=thumb_cone_verts[:, 1], 
            z=thumb_cone_verts[:, 2],
            i=[t[0] for t in triangles],
            j=[t[1] for t in triangles],
            k=[t[2] for t in triangles],
            color="cyan", opacity=0.4,
            name=f"IBS {idx} Thumb Cone",
            legendgroup=f"ibs_{idx}"
        ))
    
    fig.update_layout(
        title=f"IBSFilter Test - {num_ibs} IBS Samples (μ={mu})",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        width=1600,
        height=900
    )
    
    save_path = os.path.join(output_dir, 'combined_ibs_view.html')
    fig.write_html(save_path)
    cprint(f"[Test] Saved combined view: {save_path}", 'green')


def main():
    parser = argparse.ArgumentParser(description='Test IBSFilter visualization')
    parser.add_argument('--scene_id', type=int, default=0, help='Scene ID')
    parser.add_argument('--num_ibs', type=int, default=5, help='Number of IBS to visualize')
    parser.add_argument('--ibs_indices', type=str, default=None,
                        help='Comma-separated IBS indices (overrides num_ibs)')
    parser.add_argument('--mu', type=float, default=1.0, help='Friction coefficient')
    parser.add_argument('--output_dir', type=str, default='logs/test_ibs_filter')
    args = parser.parse_args()
    
    # Determine IBS indices
    if args.ibs_indices:
        ibs_indices = [int(x) for x in args.ibs_indices.split(',')]
    else:
        ibs_path = f'data/ibsdata/ibs/scene_{args.scene_id:04d}.npy'
        if not os.path.exists(ibs_path):
            raise FileNotFoundError(f"IBS file not found: {ibs_path}")
        total_ibs = np.load(ibs_path).shape[0]
        
        step = max(1, total_ibs // args.num_ibs)
        ibs_indices = list(range(0, min(total_ibs, args.num_ibs * step), step))[:args.num_ibs]
    
    cprint(f"[Test] Using IBS indices: {ibs_indices}", 'cyan')
    cprint(f"[Test] Friction coefficient μ = {args.mu}", 'cyan')
    
    # Load data
    ibs_voxels = load_gt_ibs_data(args.scene_id, ibs_indices)
    
    # Create individual visualizations
    visualize_ibs_with_friction_cones(
        ibs_voxels=ibs_voxels,
        mu=args.mu,
        output_dir=args.output_dir
    )
    
    # Create combined visualization
    create_combined_visualization(
        ibs_voxels=ibs_voxels,
        mu=args.mu,
        output_dir=args.output_dir
    )
    
    cprint("[Test] Done!", 'green')


if __name__ == '__main__':
    main()
