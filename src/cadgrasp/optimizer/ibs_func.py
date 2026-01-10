"""
IBS Filter for Force-Closure Grasp Selection.

This module filters and selects IBS predictions that satisfy force-closure constraints,
ensuring thumb and finger contact points can form stable grasps.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, NamedTuple
from sklearn.cluster import DBSCAN
import plotly.express as px
from plotly import graph_objects as go
import open3d as o3d
from termcolor import cprint

from cadgrasp.ibs.utils.ibs_repr import IBS, IBSBatch, IBSConfig


# ==================== Data Classes ====================

@dataclass
class FilterConfig:
    """Configuration for IBS filtering."""
    mu: float = 1.0                     # Friction coefficient
    min_contact_points: int = 10        # Minimum contact points required
    min_points_per_channel: int = 5     # Minimum points in each channel
    cluster_eps_multiplier: float = 2.0 # DBSCAN eps = avg_dist * multiplier


class IBSTriplet(NamedTuple):
    """Filtered IBS triplet with normals for optimizer."""
    ibs_occu: np.ndarray      # (N1, 6) [x, y, z, nx, ny, nz]
    ibs_cont: np.ndarray      # (N2, 6) filtered contact points with normals
    ibs_thumb_cont: np.ndarray  # (N3, 6) thumb contact points with normals


class FilterResult(NamedTuple):
    """Result of IBS filtering."""
    triplets: List[IBSTriplet]        # Valid IBS triplets
    particle_indices: List[int]        # Indices of valid particles
    global_indices: List[int]          # Global indices in original batch


# ==================== Utility Functions ====================

def cluster_points(points: np.ndarray, threshold: float = 0.01) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Cluster points using DBSCAN.
    
    Args:
        points: (N, 3) point coordinates
        threshold: DBSCAN eps parameter
    
    Returns:
        clustered_points: List of point arrays per cluster
        clustered_indices: List of index arrays per cluster
    """
    clustering = DBSCAN(eps=threshold, min_samples=1).fit(points)
    labels = clustering.labels_
    unique_labels = np.unique(labels[labels != -1])
    clustered_indices = [np.where(labels == label)[0] for label in unique_labels]
    clustered_points = [points[labels == label] for label in unique_labels]
    return clustered_points, clustered_indices


def is_in_friction_cone(point: np.ndarray, normal: np.ndarray, ref_point: np.ndarray, mu: float = 1.0) -> bool:
    """
    Check if point is within the friction cone centered at ref_point.
    
    Args:
        point: Target point position
        normal: Cone axis direction (pointing outward from surface)
        ref_point: Cone apex position
        mu: Friction coefficient
    
    Returns:
        True if point is within friction cone
    """
    vec = point - ref_point
    vec_norm = np.linalg.norm(vec)
    if vec_norm < 1e-8:
        return False
    vec = vec / vec_norm
    
    normal = normal / (np.linalg.norm(normal) + 1e-8)
    cos_theta = np.dot(vec, normal)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    theta_cone = np.arctan(mu)
    
    return theta <= theta_cone


def compute_force_closure_score(p1: np.ndarray, n1: np.ndarray, p2: np.ndarray, n2: np.ndarray) -> float:
    """
    Compute force-closure score between two contact points.
    Higher score indicates better force-closure condition.
    
    Args:
        p1, n1: Position and normal of contact 1
        p2, n2: Position and normal of contact 2
    
    Returns:
        Score in [-1, 1], higher is better
    """
    n1 = n1 / (np.linalg.norm(n1) + 1e-8)
    n2 = n2 / (np.linalg.norm(n2) + 1e-8)
    v12 = (p2 - p1)
    v12 = v12 / (np.linalg.norm(v12) + 1e-8)
    v21 = -v12
    
    cos_theta1 = np.dot(v12, n1)
    cos_theta2 = np.dot(v21, n2)
    
    return min(cos_theta1, cos_theta2)


def estimate_normals(points: np.ndarray, flip_to_positive_z: bool = True) -> Tuple[np.ndarray, float]:
    """
    Estimate normals for point cloud using Open3D.
    
    Args:
        points: (N, 3) point coordinates
        flip_to_positive_z: Flip normals to have positive z mean
    
    Returns:
        normals: (N, 3) estimated normals
        avg_dist: Average nearest neighbor distance
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    avg_dist = np.mean(pcd.compute_nearest_neighbor_distance())
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=avg_dist * 2.5, max_nn=15)
    )
    pcd.orient_normals_consistent_tangent_plane(k=30)
    
    normals = np.asarray(pcd.normals)
    
    if flip_to_positive_z:
        mean_normal = np.mean(normals, axis=0)
        if mean_normal[2] < 0:
            normals = -normals
    
    return normals, avg_dist


# ==================== IBSFilter Class ====================

class IBSFilter:
    """
    Filter IBS predictions for force-closure grasp selection.
    
    This class processes network-predicted IBS voxels and:
    1. Converts voxels to point clouds with normals
    2. Clusters contact points
    3. Filters clusters based on force-closure constraints
    4. Selects the best IBS from candidates for each particle
    
    Example usage:
        filter = IBSFilter(mu=1.0)
        
        # From network output (N*top_n, 2, 40, 40, 40)
        result = filter.filter_batch(
            ibs_voxels=network_output,
            num_particles=N,
            top_n=top_n
        )
        
        # Access results
        for triplet in result.triplets:
            print(triplet.ibs_occu.shape)  # (M, 6)
    """
    
    def __init__(self, mu: float = 1.0, config: Optional[FilterConfig] = None):
        """
        Initialize IBSFilter.
        
        Args:
            mu: Friction coefficient for force-closure check
            config: Filter configuration
        """
        self.mu = mu
        self.config = config or FilterConfig(mu=mu)
        self.ibs_config = IBSConfig()
    
    def filter_batch(
        self,
        ibs_voxels: np.ndarray,
        num_particles: int,
        top_n: int,
        visualize_count: int = 0,
        disable_force_closure: bool = False,
        verbose: bool = False
    ) -> FilterResult:
        """
        Filter a batch of IBS predictions.
        
        Args:
            ibs_voxels: (N*top_n, 2, 40, 40, 40) network output voxels
                       Channel 0: occupancy+contact (-1 empty, 1 occu, 2 contact)
                       Channel 1: thumb contact (2 for thumb)
            num_particles: Number of particles (N)
            top_n: Number of candidates per particle
            visualize_count: Number of particles to visualize
            disable_force_closure: Skip force-closure filtering
            verbose: Print debug info
        
        Returns:
            FilterResult containing valid triplets and indices
        """
        triplets = []
        particle_indices = []
        global_indices = []
        
        for i in range(num_particles):
            result = self._process_particle(
                ibs_voxels=ibs_voxels,
                particle_idx=i,
                top_n=top_n,
                visualize=(i < visualize_count),
                disable_force_closure=disable_force_closure,
                verbose=verbose
            )
            
            if result is not None:
                triplet, local_best = result
                triplets.append(triplet)
                particle_indices.append(i)
                global_indices.append(i * top_n + local_best)
        
        return FilterResult(
            triplets=triplets,
            particle_indices=particle_indices,
            global_indices=global_indices
        )
    
    def filter_from_ibs_batch(
        self,
        ibs_batch: IBSBatch,
        num_particles: int,
        top_n: int,
        visualize_count: int = 0,
        disable_force_closure: bool = False
    ) -> FilterResult:
        """
        Filter from IBSBatch object (40,40,40,3 format).
        
        Args:
            ibs_batch: IBSBatch with voxels in (N*top_n, 40, 40, 40, 3) format
            num_particles: Number of particles
            top_n: Candidates per particle
            visualize_count: Number to visualize
            disable_force_closure: Skip force-closure check
        
        Returns:
            FilterResult
        """
        # Convert IBSBatch to network format
        network_voxels = ibs_batch.to_network_input().cpu().numpy()
        return self.filter_batch(
            ibs_voxels=network_voxels,
            num_particles=num_particles,
            top_n=top_n,
            visualize_count=visualize_count,
            disable_force_closure=disable_force_closure
        )
    
    # ==================== Legacy API (Backward Compatible) ====================
    
    def filter_ibs(
        self,
        ibs_goal: np.ndarray,
        top_n: int,
        cone_viz_num: int = 0,
        disable: bool = False,
        verbose: bool = False,
        output_score: bool = True
    ) -> Tuple[List[Tuple], List[int], List[int]]:
        """
        Legacy API for backward compatibility.
        
        Args:
            ibs_goal: (N*top_n, 2, 40, 40, 40) network output
            top_n: Candidates per particle
            cone_viz_num: Number to visualize
            disable: Skip force-closure check
            verbose: Debug output
            output_score: Unused (kept for compatibility)
        
        Returns:
            ibs_triplets: List of (ibs_occu, ibs_cont, ibs_thumb) tuples
            valid_indices: Particle indices
            valid_global_indices: Global indices
        """
        num_particles = ibs_goal.shape[0] // top_n
        result = self.filter_batch(
            ibs_voxels=ibs_goal,
            num_particles=num_particles,
            top_n=top_n,
            visualize_count=cone_viz_num,
            disable_force_closure=disable,
            verbose=verbose
        )
        
        # Convert to legacy format
        legacy_triplets = [
            (t.ibs_occu, t.ibs_cont, t.ibs_thumb_cont) 
            for t in result.triplets
        ]
        
        return legacy_triplets, result.particle_indices, result.global_indices
    
    # ==================== Internal Methods ====================
    
    def _process_particle(
        self,
        ibs_voxels: np.ndarray,
        particle_idx: int,
        top_n: int,
        visualize: bool = False,
        disable_force_closure: bool = False,
        verbose: bool = False
    ) -> Optional[Tuple[IBSTriplet, int]]:
        """
        Process all candidates for a single particle and select the best.
        
        Returns:
            (best_triplet, local_index) or None if no valid candidate
        """
        candidates = []
        scores = []
        
        for j in range(top_n):
            global_idx = particle_idx * top_n + j
            ibs_vox = ibs_voxels[global_idx]
            
            result = self._process_single_ibs(
                ibs_vox=ibs_vox,
                idx=global_idx,
                visualize=visualize and (j == 0),
                disable_force_closure=disable_force_closure
            )
            
            if result is not None:
                triplet, score = result
                if triplet.ibs_cont.shape[0] >= self.config.min_contact_points:
                    candidates.append((triplet, j))
                    scores.append(score)
        
        if not candidates:
            if verbose:
                cprint(f"[IBSFilter] No valid candidates for particle {particle_idx}", 'yellow')
            return None
        
        # Select best by force-closure score
        best_idx = np.argmax(scores)
        return candidates[best_idx]
    
    def _process_single_ibs(
        self,
        ibs_vox: np.ndarray,
        idx: int,
        visualize: bool = False,
        disable_force_closure: bool = False
    ) -> Optional[Tuple[IBSTriplet, float]]:
        """
        Process a single IBS voxel.
        
        Args:
            ibs_vox: (2, 40, 40, 40) network format voxel
            idx: Global index for visualization
            visualize: Whether to show visualization
            disable_force_closure: Skip force-closure filtering
        
        Returns:
            (IBSTriplet, score) or None
        """
        # Devoxelize to point clouds
        ibs_occu, ibs_cont, ibs_thumb = self._devoxelize_network_output(ibs_vox)
        
        # Check minimum points
        min_pts = self.config.min_points_per_channel
        if ibs_occu.shape[0] < min_pts or ibs_cont.shape[0] < min_pts or ibs_thumb.shape[0] < min_pts:
            return None
        
        # Estimate normals
        all_points = np.concatenate([ibs_occu, ibs_cont, ibs_thumb], axis=0)
        all_normals, avg_dist = estimate_normals(all_points)
        
        # Split normals back
        n1 = ibs_occu.shape[0]
        n2 = ibs_cont.shape[0]
        
        ibs_occu = np.concatenate([ibs_occu, all_normals[:n1]], axis=1)
        ibs_cont = np.concatenate([ibs_cont, all_normals[n1:n1+n2]], axis=1)
        ibs_thumb = np.concatenate([ibs_thumb, all_normals[n1+n2:]], axis=1)
        
        ibs_cont_original = ibs_cont.copy()
        
        # Filter contact points by force-closure
        ibs_cont, cluster_indices, all_clusters, scores = self._filter_by_force_closure(
            ibs_cont=ibs_cont,
            ibs_thumb=ibs_thumb,
            avg_dist=avg_dist,
            disable=disable_force_closure
        )
        
        # Visualization
        if visualize:
            self._visualize(
                ibs_occu=ibs_occu,
                ibs_thumb=ibs_thumb,
                ibs_cont_original=ibs_cont_original,
                cluster_indices=cluster_indices,
                all_clusters=all_clusters,
                particle_idx=idx
            )
        
        if ibs_cont.shape[0] == 0:
            return None
        
        score = max(scores) if scores else 0.0
        triplet = IBSTriplet(ibs_occu=ibs_occu, ibs_cont=ibs_cont, ibs_thumb_cont=ibs_thumb)
        
        return triplet, score
    
    def _devoxelize_network_output(self, ibs_vox: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert network output voxel to point clouds.
        
        Network format (2, 40, 40, 40):
        - Channel 0: -1 (empty), 1 (occupancy), values near 2 indicate contact
        - Channel 1: 0 (no thumb), values > 1.5 indicate thumb
        
        Returns:
            ibs_occu: (N1, 3) general IBS points
            ibs_cont: (N2, 3) contact points
            ibs_thumb: (N3, 3) thumb contact points
        """
        bound = self.ibs_config.bound
        resolution = self.ibs_config.resolution
        
        # Create coordinate grid
        grid_x, grid_y, grid_z = np.mgrid[
            -bound:bound:resolution,
            -bound:bound:resolution,
            -bound:bound:resolution
        ]
        points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
        
        # Parse network output
        ch0 = ibs_vox[0].ravel()  # occupancy + contact
        ch1 = ibs_vox[1].ravel()  # thumb
        
        # Masks
        occu_mask = ch0 > 0.5           # Occupied voxels
        cont_mask = ch0 > 1.5           # Contact (value ~2)
        thumb_mask = ch1 > 1.5          # Thumb contact
        
        # Exclusive masks
        cont_only = cont_mask & ~thumb_mask
        occu_only = occu_mask & ~cont_mask & ~thumb_mask
        
        ibs_occu = points[occu_only]
        ibs_cont = points[cont_only]
        ibs_thumb = points[thumb_mask]
        
        return ibs_occu, ibs_cont, ibs_thumb
    
    def _filter_by_force_closure(
        self,
        ibs_cont: np.ndarray,
        ibs_thumb: np.ndarray,
        avg_dist: float,
        disable: bool = False
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[float]]:
        """
        Filter contact points by force-closure constraint with thumb.
        
        Returns:
            filtered_cont: Filtered contact points
            filtered_indices: Indices of retained clusters
            all_cluster_indices: All cluster indices
            scores: Force-closure scores
        """
        filtered_indices = []
        all_cluster_indices = []
        scores = []
        
        if ibs_cont.shape[0] == 0 or ibs_thumb.shape[0] == 0:
            return ibs_cont, filtered_indices, all_cluster_indices, scores
        
        # Cluster contact points
        eps = avg_dist * self.config.cluster_eps_multiplier
        clusters, cluster_indices = cluster_points(ibs_cont[:, :3], threshold=eps)
        all_cluster_indices = cluster_indices
        
        # Thumb statistics
        thumb_mean_pos = np.mean(ibs_thumb[:, :3], axis=0)
        thumb_mean_norm = -np.mean(ibs_thumb[:, 3:6], axis=0)  # Flip to point outward
        thumb_mean_norm = thumb_mean_norm / (np.linalg.norm(thumb_mean_norm) + 1e-8)
        
        filtered_points = []
        
        for cluster, indices in zip(clusters, cluster_indices):
            if cluster.shape[0] == 0:
                continue
            
            cluster_mean_pos = np.mean(cluster, axis=0)
            cluster_mean_norm = -np.mean(ibs_cont[indices, 3:6], axis=0)
            cluster_mean_norm = cluster_mean_norm / (np.linalg.norm(cluster_mean_norm) + 1e-8)
            
            # Force-closure check
            in_thumb_cone = is_in_friction_cone(cluster_mean_pos, thumb_mean_norm, thumb_mean_pos, self.mu)
            in_cluster_cone = is_in_friction_cone(thumb_mean_pos, cluster_mean_norm, cluster_mean_pos, self.mu)
            
            score = compute_force_closure_score(cluster_mean_pos, cluster_mean_norm, thumb_mean_pos, thumb_mean_norm)
            scores.append(score)
            
            if disable or (in_thumb_cone and in_cluster_cone):
                filtered_points.append(ibs_cont[indices])
                filtered_indices.append(indices)
        
        if filtered_points:
            filtered_cont = np.concatenate(filtered_points, axis=0)
        else:
            filtered_cont = np.zeros((0, 6))
        
        return filtered_cont, filtered_indices, all_cluster_indices, scores
    
    # ==================== Visualization ====================
    
    def _visualize(
        self,
        ibs_occu: np.ndarray,
        ibs_thumb: np.ndarray,
        ibs_cont_original: np.ndarray,
        cluster_indices: List[np.ndarray],
        all_clusters: List[np.ndarray],
        particle_idx: int
    ):
        """Visualize IBS with friction cones using Plotly."""
        visualize_point_clouds(
            ibs_occu=ibs_occu,
            ibs_thumb_cont=ibs_thumb,
            ibs_cont_original=ibs_cont_original,
            ibs_cont_filtered=None,
            cluster_indices=cluster_indices,
            particle_idx=particle_idx,
            all_cluster_indices=all_clusters,
            mu=self.mu
        )


# ==================== Visualization Functions ====================

def generate_friction_cone_vertices(apex: np.ndarray, normal: np.ndarray, height: float = 0.05, mu: float = 1.0, num_sides: int = 20) -> np.ndarray:
    """Generate vertices for visualizing a friction cone."""
    normal = normal / (np.linalg.norm(normal) + 1e-8)
    theta_cone = np.arctan(mu)
    
    # Find perpendicular vectors
    if abs(normal[0]) > abs(normal[1]):
        v1 = np.array([-normal[2], 0, normal[0]])
    else:
        v1 = np.array([0, -normal[2], normal[1]])
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    v2 = np.cross(normal, v1)
    
    # Generate circle points
    t = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
    radius = height * np.tan(theta_cone)
    circle_points = np.array([
        radius * (np.cos(ti) * v1 + np.sin(ti) * v2)
        for ti in t
    ])
    
    base_points = apex + height * normal + circle_points
    vertices = np.vstack([apex, base_points])
    return vertices


def visualize_point_clouds(
    ibs_occu: np.ndarray,
    ibs_thumb_cont: np.ndarray,
    ibs_cont_original: np.ndarray,
    ibs_cont_filtered: Optional[np.ndarray],
    cluster_indices: List[np.ndarray],
    particle_idx: int,
    all_cluster_indices: List[np.ndarray],
    mu: float
):
    """Visualize IBS point clouds with friction cones using Plotly."""
    fig = go.Figure()

    # IBS occupancy points
    if ibs_occu.shape[0] > 0:
        fig.add_trace(go.Scatter3d(
            x=ibs_occu[:, 0], y=ibs_occu[:, 1], z=ibs_occu[:, 2],
            mode="markers",
            marker=dict(size=2, color="gray", opacity=0.3),
            name="IBS Occupancy",
        ))

    # Thumb contact points
    if ibs_thumb_cont.shape[0] > 0:
        fig.add_trace(go.Scatter3d(
            x=ibs_thumb_cont[:, 0], y=ibs_thumb_cont[:, 1], z=ibs_thumb_cont[:, 2],
            mode="markers",
            marker=dict(size=3, color="blue"),
            name="Thumb Contact",
        ))

    # Contact point visualization
    if ibs_cont_original.shape[0] > 0:
        colors = px.colors.qualitative.Plotly
        filtered_indices = np.concatenate(cluster_indices) if cluster_indices else np.array([])

        # Thumb mean and normal
        if ibs_thumb_cont.shape[0] > 0:
            thumb_mean = np.mean(ibs_thumb_cont[:, :3], axis=0)
            fig.add_trace(go.Scatter3d(
                x=[thumb_mean[0]], y=[thumb_mean[1]], z=[thumb_mean[2]],
                mode="markers",
                marker=dict(size=8, color="blue", symbol="diamond"),
                name="Thumb Mean",
            ))
            
            thumb_normal = -np.mean(ibs_thumb_cont[:, 3:6], axis=0)
            thumb_normal = thumb_normal / (np.linalg.norm(thumb_normal) + 1e-8)
            normal_end = thumb_mean + 0.05 * thumb_normal
            fig.add_trace(go.Scatter3d(
                x=[thumb_mean[0], normal_end[0]],
                y=[thumb_mean[1], normal_end[1]],
                z=[thumb_mean[2], normal_end[2]],
                mode="lines",
                line=dict(color="blue", width=8),
                name="Thumb Normal",
            ))
            
            # Thumb friction cone
            cone_verts = generate_friction_cone_vertices(thumb_mean, thumb_normal, height=0.05, mu=mu)
            triangles = [[0, i, i+1] for i in range(1, len(cone_verts)-1)]
            triangles.append([0, len(cone_verts)-1, 1])
            fig.add_trace(go.Mesh3d(
                x=cone_verts[:, 0], y=cone_verts[:, 1], z=cone_verts[:, 2],
                i=[t[0] for t in triangles],
                j=[t[1] for t in triangles],
                k=[t[2] for t in triangles],
                color="blue", opacity=0.3,
                name="Thumb Friction Cone",
            ))

        # Cluster visualization
        for idx, indices in enumerate(all_cluster_indices):
            if indices.size == 0:
                continue
            cluster_pts = ibs_cont_original[indices]
            color = colors[idx % len(colors)]
            is_retained = np.any(np.isin(indices, filtered_indices))
            prefix = "✓" if is_retained else "✗"
            
            fig.add_trace(go.Scatter3d(
                x=cluster_pts[:, 0], y=cluster_pts[:, 1], z=cluster_pts[:, 2],
                mode="markers",
                marker=dict(size=3, color=color),
                name=f"{prefix} Cluster {idx}",
            ))
            
            mean_pos = np.mean(cluster_pts[:, :3], axis=0)
            fig.add_trace(go.Scatter3d(
                x=[mean_pos[0]], y=[mean_pos[1]], z=[mean_pos[2]],
                mode="markers",
                marker=dict(size=8, color=color, symbol="diamond"),
                name=f"Cluster {idx} Mean",
            ))
            
            mean_normal = -np.mean(ibs_cont_original[indices, 3:6], axis=0)
            mean_normal = mean_normal / (np.linalg.norm(mean_normal) + 1e-8)
            normal_end = mean_pos + 0.05 * mean_normal
            fig.add_trace(go.Scatter3d(
                x=[mean_pos[0], normal_end[0]],
                y=[mean_pos[1], normal_end[1]],
                z=[mean_pos[2], normal_end[2]],
                mode="lines",
                line=dict(color=color, width=8),
                name=f"Cluster {idx} Normal",
            ))
            
            # Cluster friction cone
            cone_verts = generate_friction_cone_vertices(mean_pos, mean_normal, height=0.05, mu=mu)
            triangles = [[0, i, i+1] for i in range(1, len(cone_verts)-1)]
            triangles.append([0, len(cone_verts)-1, 1])
            fig.add_trace(go.Mesh3d(
                x=cone_verts[:, 0], y=cone_verts[:, 1], z=cone_verts[:, 2],
                i=[t[0] for t in triangles],
                j=[t[1] for t in triangles],
                k=[t[2] for t in triangles],
                color=color, opacity=0.3,
                name=f"Cluster {idx} Cone",
            ))

    fig.update_layout(
        title=f"IBS Force-Closure Visualization - Particle {particle_idx}",
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="data",
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        showlegend=True,
    )
    fig.show()


# ==================== Legacy Exports (Backward Compatibility) ====================

# Keep old function names as aliases
cal_min_needed_mu_score = compute_force_closure_score
