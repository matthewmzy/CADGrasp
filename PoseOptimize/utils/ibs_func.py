import numpy as np
from sklearn.cluster import DBSCAN
import plotly.express as px
from plotly import graph_objects as go
import open3d as o3d
from termcolor import cprint
from LASDiffusion.utils.visualize_ibs_vox import voxelize, devoxelize
# from concurrent.futures import ThreadPoolExecutor

def cluster_points(points, threshold=0.01):
    clustering = DBSCAN(eps=threshold, min_samples=1).fit(points)
    labels = clustering.labels_
    unique_labels = np.unique(labels[labels != -1])
    clustered_indices = [np.where(labels == label)[0] for label in unique_labels]
    clustered_points = [points[labels == label] for label in unique_labels]
    return clustered_points, clustered_indices

def is_in_friction_cone(point, normal, ref_point, mu=1):
    vec = point - ref_point
    vec_norm = np.linalg.norm(vec)
    if vec_norm == 0:
        return False
    vec = vec / vec_norm
    cos_theta = np.dot(vec, normal) / (np.linalg.norm(normal) + 1e-8)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    theta_cone = np.arctan(mu)
    return theta <= theta_cone

def cal_min_needed_mu_score(p1,n1,p2,n2):
    n1 = n1 / (np.linalg.norm(n1) + 1e-8)
    n2 = n2 / (np.linalg.norm(n2) + 1e-8)
    v12 = p2 - p1
    v12 = v12 / (np.linalg.norm(v12) + 1e-8)
    v21 = p1 - p2
    v21 = v21 / (np.linalg.norm(v21) + 1e-8)
    cos_theta1 = np.dot(v12, n1)
    cos_theta2 = np.dot(v21, n2)
    return min(cos_theta1, cos_theta2)

def generate_friction_cone_vertices(apex, normal, height=0.05, mu=1, num_sides=20):
    normal = normal / (np.linalg.norm(normal) + 1e-8)
    theta_cone = np.arctan(mu)
    
    if abs(normal[0]) > abs(normal[1]):
        v1 = np.array([-normal[2], 0, normal[0]])
    else:
        v1 = np.array([0, -normal[2], normal[1]])
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    v2 = np.cross(normal, v1)
    
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
    ibs_occu,
    ibs_thumb_cont,
    ibs_cont_original,
    ibs_cont_filtered,
    cluster_indices,
    particle_idx,
    all_cluster_indices,
    mu
):
    fig = go.Figure()

    if ibs_occu.shape[0] > 0:
        fig.add_trace(
            go.Scatter3d(
                x=ibs_occu[:, 0],
                y=ibs_occu[:, 1],
                z=ibs_occu[:, 2],
                mode="markers",
                marker=dict(size=2, color="gray", opacity=0.3),
                name="ibs_occu",
            )
        )

    if ibs_thumb_cont.shape[0] > 0:
        fig.add_trace(
            go.Scatter3d(
                x=ibs_thumb_cont[:, 0],
                y=ibs_thumb_cont[:, 1],
                z=ibs_thumb_cont[:, 2],
                mode="markers",
                marker=dict(size=3, color="blue"),
                name="ibs_thumb_cont",
            )
        )

    if ibs_cont_original.shape[0] > 0:
        colors = px.colors.qualitative.Plotly
        filtered_indices = (
            np.concatenate(cluster_indices) if cluster_indices else np.array([])
        )
        original_indices = np.arange(ibs_cont_original.shape[0])
        deleted_indices = np.setdiff1d(original_indices, filtered_indices)

        if ibs_thumb_cont.shape[0] > 0:
            thumb_mean_pos = np.mean(ibs_thumb_cont[:, :3], axis=0)
            fig.add_trace(
                go.Scatter3d(
                    x=[thumb_mean_pos[0]],
                    y=[thumb_mean_pos[1]],
                    z=[thumb_mean_pos[2]],
                    mode="markers",
                    marker=dict(size=8, color="blue", symbol="diamond"),
                    name="ibs_thumb_cont_mean",
                )
            )
            thumb_normal = np.mean(ibs_thumb_cont[:, 3:6], axis=0)
            thumb_normal = -thumb_normal / (np.linalg.norm(thumb_normal) + 1e-8)
            normal_end = thumb_mean_pos + 0.05 * thumb_normal
            fig.add_trace(
                go.Scatter3d(
                    x=[thumb_mean_pos[0], normal_end[0]],
                    y=[thumb_mean_pos[1], normal_end[1]],
                    z=[thumb_mean_pos[2], normal_end[2]],
                    mode="lines",
                    line=dict(color="blue", width=8),
                    name="ibs_thumb_cont_normal",
                )
            )
            cone_vertices = generate_friction_cone_vertices(
                thumb_mean_pos, thumb_normal, height=0.05, mu=mu
            )
            triangles = []
            for i in range(1, len(cone_vertices) - 1):
                triangles.append([0, i, i + 1])
            triangles.append([0, len(cone_vertices) - 1, 1])
            fig.add_trace(
                go.Mesh3d(
                    x=cone_vertices[:, 0],
                    y=cone_vertices[:, 1],
                    z=cone_vertices[:, 2],
                    i=[t[0] for t in triangles],
                    j=[t[1] for t in triangles],
                    k=[t[2] for t in triangles],
                    color="blue",
                    opacity=0.3,
                    name="ibs_thumb_cont_friction_cone",
                )
            )

        for idx, indices in enumerate(all_cluster_indices):
            if indices.size == 0:
                continue
            cluster_points = ibs_cont_original[indices]
            color = colors[idx % len(colors)]
            is_retained = np.any(np.isin(indices, filtered_indices))
            name_prefix = "retained" if is_retained else "deleted"
            fig.add_trace(
                go.Scatter3d(
                    x=cluster_points[:, 0],
                    y=cluster_points[:, 1],
                    z=cluster_points[:, 2],
                    mode="markers",
                    marker=dict(size=3, color=color),
                    name=f"{name_prefix}_cluster_{idx}",
                )
            )
            mean_pos = np.mean(cluster_points[:, :3], axis=0)
            fig.add_trace(
                go.Scatter3d(
                    x=[mean_pos[0]],
                    y=[mean_pos[1]],
                    z=[mean_pos[2]],
                    mode="markers",
                    marker=dict(size=8, color=color, symbol="diamond"),
                    name=f"{name_prefix}_cluster_{idx}_mean",
                )
            )
            mean_normal = np.mean(ibs_cont_original[indices, 3:6], axis=0)
            mean_normal = -mean_normal / (np.linalg.norm(mean_normal) + 1e-8)
            normal_end = mean_pos + 0.05 * mean_normal
            fig.add_trace(
                go.Scatter3d(
                    x=[mean_pos[0], normal_end[0]],
                    y=[mean_pos[1], normal_end[1]],
                    z=[mean_pos[2], normal_end[2]],
                    mode="lines",
                    line=dict(color=color, width=8),
                    name=f"{name_prefix}_cluster_{idx}_normal",
                )
            )
            cone_vertices = generate_friction_cone_vertices(
                mean_pos, mean_normal, height=0.05, mu=mu
            )
            triangles = []
            for i in range(1, len(cone_vertices) - 1):
                triangles.append([0, i, i + 1])
            triangles.append([0, len(cone_vertices) - 1, 1])
            fig.add_trace(
                go.Mesh3d(
                    x=cone_vertices[:, 0],
                    y=cone_vertices[:, 1],
                    z=cone_vertices[:, 2],
                    i=[t[0] for t in triangles],
                    j=[t[1] for t in triangles],
                    k=[t[2] for t in triangles],
                    color=color,
                    opacity=0.3,
                    name=f"{name_prefix}_cluster_{idx}_friction_cone",
                )
            )

    fig.update_layout(
        title=f"Point Cloud Visualization for Particle {particle_idx}",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",  # Maintain original proportions without stretching
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        showlegend=True,
    )
    fig.show()

class IBSFilter:
    def __init__(self, mu=1.0):
        self.mu = mu
        self.processed_data = []
        self.valid_indices = []

    def filter_ibs(self, ibs_goal, top_n, cone_viz_num=0, disable=False, verbose=False, output_score=True):
        self.processed_data = []
        self.valid_indices = []
        self.valid_global_indices = []
        self.ibs_triplets = []

        # 定义每个粒子的处理函数，用于并行执行
        def process_particle(i):
            ibs_triplets_cache = []
            score_cache = []
            for j in range(top_n):
                ibs_occu, ibs_cont, ibs_thumb_cont, score = self._process_ibs(
                    ibs_goal[i * top_n + j], i * top_n + j, cone_viz_num, disable, output_score=output_score
                )
                if ibs_cont is not None and ibs_cont.shape[0] > 10:
                    score_cache.append(score)
                    ibs_triplets_cache.append((ibs_occu, ibs_cont, ibs_thumb_cont, j))
            if len(score_cache) > 0:
                min_mu_idx = np.argmax(score_cache)
                ibs_occu, ibs_cont, ibs_thumb_cont, j = ibs_triplets_cache[min_mu_idx]
                return i, i * top_n + j, (ibs_occu, ibs_cont, ibs_thumb_cont)
            return None

        # 使用 ThreadPoolExecutor 并行处理每个粒子
        # with ThreadPoolExecutor() as executor:
        #     results = list(executor.map(process_particle, range(ibs_goal.shape[0] // top_n)))
        results = [process_particle(i) for i in range(ibs_goal.shape[0] // top_n)]

        # 收集并行处理的结果
        for result in results:
            if result is not None:
                i, global_idx, triplet = result
                self.valid_indices.append(i)
                self.valid_global_indices.append(global_idx)
                self.ibs_triplets.append(triplet)

        if not self.valid_indices:
            return [], [], []

        return self.ibs_triplets, self.valid_indices, self.valid_global_indices

    def _process_ibs(self, ibs, particle_idx, cone_viz_num=0, disable=False, output_score=True):
        ibs_occu, ibs_cont, ibs_thumb_cont = devoxelize(ibs)
        shapes = (ibs_occu.shape[0], ibs_cont.shape[0], ibs_thumb_cont.shape[0])
        if min(shapes)<5:
            return None, None, None, None
        # Estimate normals
        pcd = o3d.geometry.PointCloud()
        pts = np.concatenate((ibs_occu, ibs_cont, ibs_thumb_cont), axis=0)
        pcd.points = o3d.utility.Vector3dVector(pts)
        avg_dist = np.mean(pcd.compute_nearest_neighbor_distance())
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=avg_dist * 2.5, max_nn=15))
        pcd.orient_normals_consistent_tangent_plane(k=30)
        normals = np.asarray(pcd.normals)
        mean_normal = np.mean(normals, axis=0)
        if mean_normal[2] < 0:
            normals = -normals
            pcd.normals = o3d.utility.Vector3dVector(normals)
        ibs_occu_norm = normals[:shapes[0]]
        ibs_cont_norm = normals[shapes[0]:shapes[0]+shapes[1]]
        ibs_thumb_cont_norm = normals[shapes[0]+shapes[1]:]
        ibs_occu = np.concatenate((ibs_occu, ibs_occu_norm), axis=1)
        ibs_cont = np.concatenate((ibs_cont, ibs_cont_norm), axis=1)
        ibs_thumb_cont = np.concatenate((ibs_thumb_cont, ibs_thumb_cont_norm), axis=1)

        ibs_cont_original = ibs_cont.copy()

        filtered_cluster_indices = []
        all_cluster_indices = []
        scores = []
        if ibs_cont.shape[0] > 0 and ibs_thumb_cont.shape[0] > 0:
            cont_clusters, cont_cluster_indices = cluster_points(ibs_cont[:, :3], threshold=avg_dist * 2)
            all_cluster_indices = cont_cluster_indices
            filtered_cont_points = []
            thumb_mean_pos = np.mean(ibs_thumb_cont[:, :3], axis=0)
            thumb_mean_norm = -np.mean(ibs_thumb_cont[:, 3:6], axis=0)
            thumb_mean_norm = thumb_mean_norm / (np.linalg.norm(thumb_mean_norm) + 1e-8)
            
            for cluster, indices in zip(cont_clusters, cont_cluster_indices):
                if cluster.shape[0] == 0:
                    continue
                cluster_mean_pos = np.mean(cluster, axis=0)
                cluster_mean_norm = -np.mean(ibs_cont[indices, 3:6], axis=0)
                cluster_mean_norm = cluster_mean_norm / (np.linalg.norm(cluster_mean_norm) + 1e-8)
                in_thumb_cone = is_in_friction_cone(cluster_mean_pos, thumb_mean_norm, thumb_mean_pos, mu=self.mu)
                in_cluster_cone = is_in_friction_cone(thumb_mean_pos, cluster_mean_norm, cluster_mean_pos, mu=self.mu)
                scores.append(cal_min_needed_mu_score(cluster_mean_pos, cluster_mean_norm, thumb_mean_pos, thumb_mean_norm))
                if disable or (in_thumb_cone and in_cluster_cone):
                    filtered_cont_points.append(ibs_cont[indices])
                    filtered_cluster_indices.append(indices)
            
            if filtered_cont_points:
                ibs_cont = np.concatenate(filtered_cont_points, axis=0)
            else:
                ibs_cont = np.zeros((0, 6))
                cprint(f"[WARNING] No force-closure contact points found for particle {particle_idx}", 'red')

        if particle_idx < cone_viz_num:
            visualize_point_clouds(
                ibs_occu=ibs_occu,
                ibs_thumb_cont=ibs_thumb_cont,
                ibs_cont_original=ibs_cont_original,
                ibs_cont_filtered=ibs_cont,
                cluster_indices=filtered_cluster_indices,
                particle_idx=particle_idx,
                all_cluster_indices=all_cluster_indices,
                mu=self.mu
            )

        return ibs_occu, ibs_cont, ibs_thumb_cont, max(scores) if len(scores)>0 else 0.0
