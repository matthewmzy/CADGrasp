import numpy as np
import torch
import plotly.graph_objects as go

def transform_points(pc, trans):
    if type(pc)==torch.Tensor:
        homo = torch.cat((pc,torch.ones((pc.shape[0],1),device=pc.device)),dim=1)
        homo = homo @ trans.T
        pc = homo[:,:3]/homo[:,3].unsqueeze(-1)
    elif type(pc)==np.ndarray:
        homo = np.concatenate([pc,np.ones((pc.shape[0],1))],axis=1)
        homo = homo @ trans.T
        pc = homo[:,:3]/homo[:,3].reshape(-1,1)
    return pc

def batch_transform_points(points, trans_matrix):
    """
    将一批(B个)点云通过变换矩阵进行坐标变换。
    :param points: (B, N, 3) 的点云坐标。
    :param trans_matrix: (B, 4, 4) 的齐次变换矩阵。
    :return: (B, N, 3) 的变换后的点云坐标。
    """
    if isinstance(points, np.ndarray):
        points_homogeneous = np.concatenate([points, np.ones((points.shape[0], points.shape[1], 1))], axis=2)
        transformed_points_homogeneous = np.einsum('bij,bjk->bik', points_homogeneous, trans_matrix.transpose(0,2,1))
        transformed_points = transformed_points_homogeneous[:, :, :3] / transformed_points_homogeneous[:, :, 3][:, :, np.newaxis].repeat(3, axis=2)
        return transformed_points
    elif isinstance(points, torch.Tensor):
        points_homogeneous = torch.cat([points, torch.ones_like(points[:, :, :1])], dim=2)  # 拼接为齐次坐标
        transformed_points_homogeneous = points_homogeneous @ trans_matrix.transpose(1, 2)
        transformed_points = transformed_points_homogeneous[:, :, :3] / transformed_points_homogeneous[:, :, 3].unsqueeze(2).repeat(1, 1, 3)
        return transformed_points
    
def pc_plotly(pc, size=3, color='green'):
    return go.Scatter3d(
            x=pc[:, 0] if isinstance(pc, np.ndarray) else pc[:, 0].numpy(),
            y=pc[:, 1] if isinstance(pc, np.ndarray) else pc[:, 1].numpy(),
            z=pc[:, 2] if isinstance(pc, np.ndarray) else pc[:, 2].numpy(),
            mode='markers',
            marker=dict(size=size, color=color),
        )

def show(plotly_list):
    fig = go.Figure(data=plotly_list, layout=go.Layout(scene=dict(aspectmode='data')))
    fig.show()