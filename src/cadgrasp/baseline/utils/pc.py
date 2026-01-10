import numpy as np

def depth_image_to_point_cloud(depth: np.ndarray, # (H, W)
                               instrincs: np.ndarray, # (3, 3)
                               factor_depth: float
) -> np.ndarray: # (H, W, 3)
    h, w = depth.shape
    cx, fx, cy, fy = instrincs[0, 2], instrincs[0, 0], instrincs[1, 2], instrincs[1, 1]

    xmap = np.arange(w)
    ymap = np.arange(h)
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = depth / factor_depth
    points_x = (xmap - cx) * points_z / fx
    points_y = (ymap - cy) * points_z / fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)

    return cloud

def transform_pc(cloud: np.ndarray, # (N, 3) 
                 transform: np.ndarray, # (4, 4)
):
    ones = np.ones((len(cloud), 1))
    cloud_cat = np.concatenate([cloud, ones], axis=-1)
    cloud_cat = np.einsum('ab,nb->na', transform, cloud_cat)
    return cloud_cat[:, :3]

def get_workspace_mask(cloud: np.ndarray, # (H, W, 3)
                       seg: np.ndarray, # (H, W)
                       trans: np.ndarray, # (4, 4)
                       outlier: float = 0.2
) -> np.ndarray: # (H, W)
    h, w = seg.shape
    cloud = cloud.reshape(-1, 3)
    seg = seg.reshape(-1)
    cloud = transform_pc(cloud, trans)

    foreground = cloud[seg > 0]
    xmin, ymin, zmin = foreground.min(axis=0)
    xmax, ymax, zmax = foreground.max(axis=0)
    mask_x = ((cloud[:, 0] > xmin - outlier) & (cloud[:, 0] < xmax + outlier))
    mask_y = ((cloud[:, 1] > ymin - outlier) & (cloud[:, 1] < ymax + outlier))
    mask_z = ((cloud[:, 2] > zmin - outlier) & (cloud[:, 2] < zmax + outlier))
    mask = mask_x & mask_y & mask_z
    return mask.reshape(h, w)
