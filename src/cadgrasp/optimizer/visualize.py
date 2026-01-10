'''
LastEditTime: 2022-05-23 19:24:38
Description: Your description
Date: 2021-11-04 04:54:29
Author: Aiden Li
LastEditors: Aiden Li (i@aidenli.net)
'''
import io
import os
from tkinter.messagebox import NO
import numpy as np
import torch
import torch
import trimesh as tm
from plotly import graph_objects as go
import plotly.offline as pyo
from PIL import Image
import open3d as o3d
from time import time
import k3d
from cadgrasp.optimizer.utils_hand_meta import *
from cadgrasp.optimizer.HandModel import get_handmodel
from LASDiffusion.utils.visualize_ibs_vox import devoxelize
import copy

colors = [
    'blue', 'red', 'yellow', 'pink', 'gray', 'orange'
]

def plot_mesh(mesh, color='lightblue', opacity=1.0):
    return go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        color=color, opacity=opacity)

def plot_hand(verts, faces, color='lightpink', opacity=1.0):
    return go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color, opacity=opacity)

def plot_contact_points(pts, grad, color='lightpink'):
    pts = pts.detach().cpu().numpy()
    grad = grad.detach().cpu().numpy()
    grad = grad / np.linalg.norm(grad, axis=-1, keepdims=True)
    return go.Cone(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], u=-grad[:, 0], v=-grad[:, 1], w=-grad[:, 2], anchor='tip',
                   colorscale=[(0, color), (1, color)], sizemode='absolute', sizeref=0.2, opacity=0.5)

def plot_point_cloud(pts, color='lightblue', mode='markers'):
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode=mode,
        marker=dict(
            color=color,
            size=3.
        )
    )

def dis_cmap(levels, red_thred=0.01, blue_thred=0.1):
    colors = []
    for x in levels.tolist():
        if x < red_thred:  # 小于 0.01 为红色
            colors.append(f"rgb(255, 0, 0)")
        elif x > blue_thred:  # 大于 0.1 为蓝色
            colors.append(f"rgb(0, 0, 255)")
        else:  # 在线性区间 [0.01, 0.1] 插值
            t = (x - red_thred) / (blue_thred - red_thred)  
            r = int(255 * (1 - t)) 
            b = int(255 * t)
            colors.append(f"rgb({r}, 0, {b})") 
    return colors


def plot_point_cloud_dis(pts, color_levels=None, red_thred=0.01, blue_thred=0.1):
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode='markers',
        marker={
            'color': dis_cmap(color_levels, red_thred=red_thred, blue_thred=blue_thred),
            'size': 3.5,
            'opacity': 1
        }
    )

debug_cmap = lambda levels: [
    f"rgb(0, 0, 0)" if x < 0 else f"rgb(255, 255, 255)" for x in levels.tolist()
]
# 0,0,0 black 255,255,255 white

def plot_point_cloud_debug(pts, color_levels=None):
    """
    可视化点云，每个点根据其值的范围设置颜色
    """
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode='markers',
        marker={
            'color': debug_cmap(color_levels),
            'size': 3.5,
            'opacity': 1
        }
    )





occ_cmap = lambda levels, thres=0.: [f"rgb({int(255)},{int(255)},{int(255)})" if x > thres else
                           f"rgb({int(0)},{int(0)},{int(0)})" for x in levels.tolist()]


def plot_point_cloud_occ(pts, color_levels=None):
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode='markers',
        marker={
            'color': occ_cmap(color_levels),
            'size': 3,
            'opacity': 1
        }
    )


contact_cmap = lambda levels, thres=0.: [f"rgb({int(255 * (1 - x))},{int(255 * (1 - x))},{int(255 * (1 - x))})" if x >= thres else
                                         f"rgb({int(0)},{int(0)},{int(0)})" for x in levels.tolist()]

def plot_point_cloud_cmap(pts, color_levels=None):
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode='markers',
        marker={
            'color': contact_cmap(color_levels),
            'size': 3.5,
            'opacity': 1
        }
    )


normal_color_map = lambda levels, thres=0., color_scale=8.: [f"rgb({int(255 * (color_scale * x[0]))},{int(255 * (color_scale * x[1]))},{int(255 * (color_scale * x[2]))})" if x[0] >= thres else
                                                             f"rgb({int(0)},{int(0)},{int(0)})" for x in levels.tolist()]


def plot_normal_map(pts, normal):
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode='markers',
        marker={
            'color': normal_color_map(np.abs(normal)),
            'size': 3.5,
            'opacity': 1
        }
    )


def plot_grasps(directory, tag, uuids, physics_guide, handcodes, contact_idx, ret_plots=False, save_html=True, include_contacts=True):
    handcode = handcodes[:, -1]
    hand_vertices = physics_guide.get_vertices(handcodes)
    hand_faces = physics_guide.hand_model.faces
    
    object_models = physics_guide.object_models
        
    if include_contacts:
        contact_points = []
        for ind in range(contact_idx.shape[1]):
            contact_point_vertices = torch.gather(
                hand_vertices, 1,
                contact_idx[:, ind].unsqueeze(-1).tile((1, 1, 3))
            )
            contact_points.append(contact_point_vertices.detach().cpu().numpy())

    hand_vertices = hand_vertices.detach().cpu().numpy()
    
    plots = []

    for batch_idx in range(hand_vertices.shape[0]):
        to_plot = []

        to_plot.append(plot_hand(hand_vertices[batch_idx], hand_faces))

        for obj_ind, obj in enumerate(object_models):
            to_plot.append(obj.get_plot(batch_idx))
            if include_contacts:
                to_plot.append(plot_point_cloud(contact_points[obj_ind][batch_idx], color=colors[obj_ind]))
        
        fig = go.Figure(to_plot)
        
        if save_html:
            fig.write_html(os.path.join(f"{ directory }", f"fig-{ str(uuids[ batch_idx ]) }-{ batch_idx }-{ tag }.html"))
        if ret_plots:
            plots.append(torch.from_numpy(np.asarray(Image.open(io.BytesIO(fig.to_image(format="png", width=1280, height=720))))))
            
    if ret_plots:
        return plots
    

def plot_mesh_from_name(dataset_object_name, color='lightblue', opacity=1.):
    dataset_name = dataset_object_name.split('+')[0]
    object_name = dataset_object_name.split('+')[1]
    mesh_path = os.path.join('data', 'object', dataset_name, object_name, f'{object_name}.stl')
    object_mesh = tm.load(mesh_path)
    return plot_mesh(object_mesh, color=color, opacity=opacity)

def o3d_visualize_mesh(mesh, color=[0.7, 0.7, 0.7]):
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    o3d.visualization.draw_geometries([mesh])

# def o3d_visualize_mesh_and_ibs(scene_o3d, ibs_data, show=None, hand_trans=None):
#     ibs_data = ibs_data.copy()
#     scene_mesh = scene_o3d
#     scene_mesh.compute_vertex_normals()
#     scene_mesh.paint_uniform_color([0.7, 0.7, 0.7]) 
#     # 提取 IBS 数据
#     ibs_points = ibs_data[:, :3]
#     hand_distances = ibs_data[:, 3]
#     hand_uv = ibs_data[:, 4:7]

#     # center = np.mean(ibs_data['points'],axis=0)
#     # uv = np.mean(ibs_data['hand_unit_vectors'],axis=0)
#     # uv = uv*0.1/np.linalg.norm(uv)
#     # pointer = o3d.geometry.LineSet()
#     # pointer.points = o3d.utility.Vector3dVector([center,center+uv])
#     # pointer.lines = o3d.utility.Vector2iVector([[0,1]])

#     # 初始化颜色数组
#     colors = np.array([[0.0, 0.0, 1.0]]).repeat(ibs_points.shape[0],axis=0) 
#     if show == 'dis' or show == 'hand':
#         if show == 'hand':
#             ibs_points += hand_uv*hand_distances[:,None]
#         contact_min_thre = 0.005
#         contact_max_thre = 0.1
        
#         colors[hand_distances < contact_min_thre] = [1.0, 0.0, 0.0]  # 红色
#         colors[hand_distances > contact_max_thre] = [0.0, 0.0, 1.0]  # 蓝色
#         mask = (hand_distances >= contact_min_thre) & (hand_distances <= contact_max_thre)
#         normalized_distances = (hand_distances[mask] - contact_min_thre) / (contact_max_thre - contact_min_thre)
#         # 线性插值，颜色从红色渐变到蓝色
#         colors[mask, 0] = 1.0 - normalized_distances  # 红色分量
#         colors[mask, 2] = normalized_distances        # 蓝色分量
#     elif show == 'cont':
#         contact_min_thre = 0.005
#         mask1 = hand_distances < contact_min_thre
#         # mask2 = info_arr[:, 1] == IsPalmar.PALMAR
#         # mask = mask1 & mask2
#         colors[mask1] = [1.0, 0.0, 0.0]

#     ibs_pcd = o3d.geometry.PointCloud()
#     ibs_pcd.points = o3d.utility.Vector3dVector(ibs_points)
#     ibs_pcd.colors = o3d.utility.Vector3dVector(colors)
#     ibs_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
#     if hand_trans is None:
#         o3d.visualization.draw_geometries([scene_mesh, ibs_pcd])
#     else:
#         hand_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
#         hand_coord.transform(hand_trans)
#         o3d.visualization.draw_geometries([scene_mesh, ibs_pcd, hand_coord])
    


def update_visualizer(scene_o3d_array, hand_o3d_array, ibs_data_array, time_interval=1, show=None):

    colors = np.array([[0.0, 0.0, 1.0]]).repeat(len(ibs_pcd.points),axis=0)

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    total_num = len(scene_mesh)
    for i in range(total_num):
        scene_mesh = scene_o3d_array[i]
        hand_mesh = hand_o3d_array[i]
        ibs_data = ibs_data_array[i]
        scene_mesh.compute_vertex_normals()
        scene_mesh.paint_uniform_color([0.7, 0.7, 0.7])  # 灰色场景
        hand_mesh.compute_vertex_normals()
        hand_mesh.paint_uniform_color([1.0, 0.6, 0.6])  # 粉色手
        ibs_pcd = o3d.geometry.PointCloud()
        ibs_pcd.points = o3d.utility.Vector3dVector(ibs_data[:, :3])

        if show == 'dis' or show == 'hand':
            if show == 'hand':
                ibs_points = ibs_data[:, :3]+ibs_data[:, 4:7]*ibs_data[:, 3][:,None]
                ibs_pcd.points = o3d.utility.Vector3dVector(ibs_points)
            scene_distances = ibs_data[:, 8]
            hand_distances = ibs_data[:, 3]
            # print(np.min(scene_distances))
            # print(np.min(hand_distances))
            contact_min_thre = 0.005
            contact_max_thre = 0.1
            
            colors[hand_distances < contact_min_thre] = [1.0, 0.0, 0.0]  # 红色
            colors[hand_distances > contact_max_thre] = [0.0, 0.0, 1.0]  # 蓝色
            mask = (hand_distances >= contact_min_thre) & (hand_distances <= contact_max_thre)
            normalized_distances = (hand_distances[mask] - contact_min_thre) / (contact_max_thre - contact_min_thre)
            # 线性插值，颜色从红色渐变到蓝色
            colors[mask, 0] = 1.0 - normalized_distances  # 红色分量
            colors[mask, 2] = normalized_distances        # 蓝色分量
            ibs_pcd.colors = o3d.utility.Vector3dVector(colors)
        elif show == 'finger':
            hand_part = InfoData.from_array(ibs_data)[:,0]
            candidate_colors = np.array([[1.0,0.0,0.0],
                                        [0.0,1.0,0.0],
                                        [0.0,0.0,1.0],
                                        [0.0,1.0,1.0],
                                        [1.0,0.0,1.0],
                                        [1.0,1.0,0.0]])
            for i in range(len(colors)):
                colors[i] = candidate_colors[hand_part[i]]
            ibs_pcd.colors = o3d.utility.Vector3dVector(colors)                

        visualizer.clear_geometries()  
        visualizer.add_geometry(scene_mesh) 
        if show!='reverse':
            visualizer.add_geometry(hand_mesh)
        visualizer.add_geometry(ibs_pcd) 
        # visualizer.add_geometry(pointer) 

        visualizer.poll_events() 
        visualizer.update_renderer() 
        time.sleep(time_interval)

    visualizer.destroy_window()
    
def tm2o3d(mesh):
    vertices = mesh.vertices
    faces = mesh.faces
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    return o3d_mesh

def visualize_plotly_animation(static_plots, dynamic_plots, fps=30):
    print('>>> Generating animation...')
    fig = go.Figure()
    for plot in static_plots:
        fig.add_trace(plot)
    for plot in dynamic_plots[0]:
        fig.add_trace(go.Mesh3d())
 
    print('>>> Adding frames...')

    frames = []
    for i in range(len(dynamic_plots)):
        data = static_plots+ dynamic_plots[i] 
        frames.append(go.Frame(data=data, name=f'frame{i}'))
    fig.frames = frames

    print('>>> Setting layout...')

    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(label="Play",
                        method="animate",
                        args=[None, dict(frame=dict(duration=1000 / fps, redraw=True),
                                        fromcurrent=True, mode='immediate')]),
                    dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                                            mode="immediate")])])],
        sliders=[dict(
            steps=[dict(method='animate',
                        args=[[f'frame{i}'], dict(mode='immediate',
                                                frame=dict(duration=1000 / fps, redraw=True),
                                                transition=dict(duration=0))],
                        label=f'Frame {i}') for i in range(len(frames))],
            active=0,
            transition=dict(duration=0),
            x=0,  
            y=-0.1, 
            currentvalue=dict(font=dict(size=12), prefix='Frame: ', visible=True, xanchor='center'),
            len=1.0  
        )]
    )

    print('>>> Showing plot...')

    # fig.show()

    print('>>> Done!')

def visualize_k3d_animation(static_plots, dynamic_plots_mesh, dynamic_plots_pts, dynamic_plots_pts_thumb, save_path, fps=120):
    plot = k3d.plot()
    
    for static_plot in static_plots:
        plot += static_plot
    
    plot += dynamic_plots_mesh[0]
    dynamic_plots_mesh[0].vertices = {str(i): dynamic_plots_mesh[i].vertices for i in range(len(dynamic_plots_mesh))}

    plot += dynamic_plots_pts[0]  
    dynamic_plots_pts[0].positions = {str(i): dynamic_plots_pts[i].positions for i in range(len(dynamic_plots_pts))}

    plot += dynamic_plots_pts_thumb[0]
    dynamic_plots_pts_thumb[0].positions = {str(i): dynamic_plots_pts_thumb[i].positions for i in range(len(dynamic_plots_pts_thumb))}

    plot.fps = fps

    html_snapshot = plot.get_snapshot()
    with open(save_path, 'w') as f:
        f.write(html_snapshot)

    print(">>> Done!")


def visualize_trajectory(scene, ibs : torch.Tensor,  hand_name : str, q_trajectory : torch.Tensor, energy : torch.Tensor, i : int = 0):
    hand_model = get_handmodel(hand_name, 1, scene.device)
    trajectory = q_trajectory[i]
    energy = energy[i]
    dynamic_plots = []
    for q in trajectory:
        data = hand_model.get_plotly_data(q=q.unsqueeze(0), i=0, opacity=0.5)
        dynamic_plots.append(data)
    static_plots = [plot_mesh(scene.tm_mesh, color='gray'), plot_point_cloud(ibs[:,:3], color='green')]
    visualize_plotly_animation(static_plots, dynamic_plots)

def visualize_traj_k3d(scene, ibs: torch.Tensor, hand_name: str, q_trajectory: torch.Tensor, i: int = 0, save_path: str = None, scene_trans: np.ndarray = np.eye(4)):
    def tm2k3d_mesh(mesh, color):
        vertices = mesh.vertices
        faces = mesh.faces
        return k3d.mesh(vertices, faces, color=color)

    def tm2k3d_points(points, color):
        return k3d.points(points, color=color, point_size=0.003)

    hand_model = get_handmodel(hand_name, 1, scene.device)
    trajectory = q_trajectory[i]
    dynamic_plots_mesh = []
    dynamic_plots_pts = []
    dynamic_plots_pts_thumb = []
    for q in trajectory:
        mesh_data = hand_model.get_k3d_data(q=q.unsqueeze(0), i=0, opacity=0.5, color=0x0000ff, concat=True) # <k3d.mesh>
        pts_data, thumb_pts_data = hand_model.get_palmar_points_k3d(i=0) # <k3d.points>
        dynamic_plots_mesh.append(mesh_data)
        dynamic_plots_pts.append(pts_data)
        dynamic_plots_pts_thumb.append(thumb_pts_data)

    ibs_not_contact, ibs_contact, ibs_thumb_contact = devoxelize(ibs)
    scene_mesh = copy.deepcopy(scene.combined_mesh).apply_transform(scene_trans)
    static_plots = [tm2k3d_mesh(scene_mesh, color=0x808080), 
                    tm2k3d_points(ibs_contact[:,:3], color=0xff0000), 
                    tm2k3d_points(ibs_not_contact[:,:3], color=0xffff00),
                    tm2k3d_points(ibs_thumb_contact[:,:3], color=0x00ffff)]
    visualize_k3d_animation(static_plots, dynamic_plots_mesh, dynamic_plots_pts, dynamic_plots_pts_thumb, save_path=save_path)

def save_energy_curve(scene, energy_dict, index, save_path):
    # remove keys that cause len(energy_dict[key]) = 0
    energy_dict = {k: v for k, v in energy_dict.items() if len(v) > 0}

    fig = go.Figure()

    traj_len = max([energy.size(0) for energy in energy_dict.values()]) 
    

    for key in energy_dict.keys():
        energy = energy_dict[key][:,index]
        if key == 'E_distal' or key == 'E_pen1':
            x = list(range(len(energy)))
        else:
            x = list(range(traj_len-len(energy),traj_len))
        fig.add_trace(go.Scatter(
            x=x,
            y=energy.numpy(),
            mode='lines+markers',
            name=key
        ))

    fig.update_layout(
        title="Energy curve",
        xaxis_title="Iteration",
        yaxis_title="Energy",
        legend_title="Legend",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )

    # 保存图像为HTML文件
    save_file = os.path.join(save_path, f'energy_{index}.html')
    fig.write_html(save_file)
    print(f"Plot saved to {save_file}")