import os, trimesh as tm, open3d as o3d, numpy as np
from termcolor import cprint

def tm2o3d(mesh):
    vertices = mesh.vertices
    faces = mesh.faces
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
    return mesh_o3d

def get_ibs_pc(ibs_vox, bound=0.1, resolution=0.005):
    grid_x, grid_y, grid_z = np.mgrid[-bound:bound:resolution, -bound:bound:resolution, -bound:bound:resolution]
    points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
    occ_vox = ibs_vox[...,0].ravel()
    con_vox = ibs_vox[...,1].ravel()
    thu_vox = ibs_vox[...,2].ravel()
    occu_mask = occ_vox>0
    cont_mask = con_vox>0
    thumb_mask = thu_vox>0
    occu_mask = np.logical_and(occu_mask, np.logical_not(cont_mask))
    cont_mask = np.logical_and(cont_mask, np.logical_not(thumb_mask))
    occu_ibs_pc = points[occu_mask]
    cont_ibs_pc = points[cont_mask]
    thumb_cont_pc = points[thumb_mask]
    return occu_ibs_pc, cont_ibs_pc, thumb_cont_pc

def voxelize(occu_ibs_pc, cont_ibs_pc, thumb_cont_pc):
    x_min, x_max = -0.1, 0.1
    y_min, y_max = -0.1, 0.1
    z_min, z_max = -0.1, 0.1
    resolution = 0.005
    voxelized_ibs = np.zeros((int((x_max-x_min)/resolution), int((y_max-y_min)/resolution), int((z_max-z_min)/resolution), 3), dtype=bool)
    occu_ibs_pc = (occu_ibs_pc - np.array([x_min, y_min, z_min])) / resolution
    occu_ibs_pc = np.floor(occu_ibs_pc).astype(int)
    cont_ibs_pc = (cont_ibs_pc - np.array([x_min, y_min, z_min])) / resolution
    cont_ibs_pc = np.floor(cont_ibs_pc).astype(int)
    thumb_cont_pc = (thumb_cont_pc - np.array([x_min, y_min, z_min])) / resolution
    thumb_cont_pc = np.floor(thumb_cont_pc).astype(int)
    # occu_ibs_pc = np.clip(occu_ibs_pc, 0, voxelized_ibs.shape[0]-1)
    # cont_ibs_pc = np.clip(cont_ibs_pc, 0, voxelized_ibs.shape[0]-1)
    # thumb_cont_pc = np.clip(thumb_cont_pc, 0, voxelized_ibs.shape[0]-1)
    voxelized_ibs[occu_ibs_pc[:, 0], occu_ibs_pc[:, 1], occu_ibs_pc[:, 2], 0] = True
    voxelized_ibs[cont_ibs_pc[:, 0], cont_ibs_pc[:, 1], cont_ibs_pc[:, 2], 1] = True
    voxelized_ibs[thumb_cont_pc[:, 0], thumb_cont_pc[:, 1], thumb_cont_pc[:, 2], 2] = True
    return voxelized_ibs

def devoxelize(ibs_vox):
    bound=0.1
    resolution=0.005
    grid_x, grid_y, grid_z = np.mgrid[-bound:bound:resolution, -bound:bound:resolution, -bound:bound:resolution]
    points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
    occ_vox = ibs_vox[...,0].ravel()
    con_vox = ibs_vox[...,1].ravel()
    thu_vox = ibs_vox[...,2].ravel()
    occu_mask = occ_vox>0.5
    cont_mask = con_vox>0.5
    thco_mask = thu_vox>0.5
    # cprint(np.sum(thco_mask),'red')
    occu_mask = np.logical_and(occu_mask, np.logical_not(cont_mask))
    cont_mask = np.logical_and(cont_mask, np.logical_not(thco_mask))
    occu_ibs_pc = points[occu_mask]
    cont_ibs_pc = points[cont_mask]
    thco_ibs_pc = points[thco_mask]
    return occu_ibs_pc, cont_ibs_pc, thco_ibs_pc

def visualize_scene_pc_and_ibs(ibs_vox, scene_pc, vis_hand_pose=False):
    occu_ibs_pc, cont_ibs_pc, thco_ibs_pc = devoxelize(ibs_vox)
    occu_ibs_pc_o3d = o3d.geometry.PointCloud()
    occu_ibs_pc_o3d.points = o3d.utility.Vector3dVector(occu_ibs_pc)
    occu_ibs_pc_o3d.paint_uniform_color([1, 0, 0])
    cont_ibs_pc_o3d = o3d.geometry.PointCloud()
    cont_ibs_pc_o3d.points = o3d.utility.Vector3dVector(cont_ibs_pc)
    cont_ibs_pc_o3d.paint_uniform_color([0, 1, 0])
    scene_pc_o3d = o3d.geometry.PointCloud()
    scene_pc_o3d.points = o3d.utility.Vector3dVector(scene_pc)
    scene_pc_o3d.paint_uniform_color([0, 0, 1])

    # from ipdb import set_trace; set_trace()

    if vis_hand_pose:
        hand_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([scene_pc_o3d, occu_ibs_pc_o3d, cont_ibs_pc_o3d, hand_coord], window_name="scene vs occu ibs")
    else:
        hand_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([scene_pc_o3d, occu_ibs_pc_o3d, cont_ibs_pc_o3d], window_name="scene vs occu ibs")
    return occu_ibs_pc

def visualize_scene_mesh_and_ibs(ibs_vox, scene_mesh, vis_hand_pose=False):
    bound=0.1
    resolution=0.005
    grid_x, grid_y, grid_z = np.mgrid[-bound:bound:resolution, -bound:bound:resolution, -bound:bound:resolution]
    points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
    occ_vox = ibs_vox[0].ravel()
    con_vox = ibs_vox[1].ravel()
    occu_mask = occ_vox>0
    cont_mask = con_vox>0
    occu_mask = np.logical_and(occu_mask, np.logical_not(cont_mask))
    occu_ibs_pc = points[occu_mask]
    cont_ibs_pc = points[cont_mask]
    occu_ibs_pc_o3d = o3d.geometry.PointCloud()
    occu_ibs_pc_o3d.points = o3d.utility.Vector3dVector(occu_ibs_pc)
    occu_ibs_pc_o3d.paint_uniform_color([1, 0, 0])
    cont_ibs_pc_o3d = o3d.geometry.PointCloud()
    cont_ibs_pc_o3d.points = o3d.utility.Vector3dVector(cont_ibs_pc)
    cont_ibs_pc_o3d.paint_uniform_color([0, 1, 0])
    scene_mesh.compute_vertex_normals()
    scene_mesh.paint_uniform_color([0.5,0.5,0.5])

    # from ipdb import set_trace; set_trace()

    if vis_hand_pose:
        hand_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([scene_mesh, occu_ibs_pc_o3d, cont_ibs_pc_o3d, hand_coord], window_name="scene vs occu ibs")
    else:
        hand_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([scene_mesh, occu_ibs_pc_o3d, cont_ibs_pc_o3d], window_name="scene vs occu ibs")


if __name__ == '__main__':
    path_root = "/home/matthew_zy/Documents/summer/dipgrasp/CLEAN_DATASET/ibs_w_thumb/shadow_5/ibs/ibs_08000.npz"
    ibs = np.load(path_root)
    ibs_vox = ibs["graspnet-040"][0]
    occu_ibs_pc, cont_ibs_pc, thumb_cont_pc = get_ibs_pc(ibs_vox)
    print(occu_ibs_pc.shape, cont_ibs_pc.shape, thumb_cont_pc.shape)
    recon_vox = voxelize(occu_ibs_pc, cont_ibs_pc, thumb_cont_pc)
    rec_occu_ibs_pc, rec_cont_ibs_pc, rec_thumb_cont_pc = get_ibs_pc(recon_vox)
    # 检查体素化之后是否和ibs_vox一模一样
    o3d_ibs = o3d.geometry.PointCloud()
    o3d_ibs.points = o3d.utility.Vector3dVector(occu_ibs_pc)
    o3d_ibs.paint_uniform_color([1, 0, 0])
    o3d_cont = o3d.geometry.PointCloud()
    o3d_cont.points = o3d.utility.Vector3dVector(cont_ibs_pc)
    o3d_cont.paint_uniform_color([0, 1, 0])
    o3d_thumb = o3d.geometry.PointCloud()
    o3d_thumb.points = o3d.utility.Vector3dVector(thumb_cont_pc)
    o3d_thumb.paint_uniform_color([0, 0, 1])
    o3d_recon_ibs = o3d.geometry.PointCloud()
    o3d_recon_ibs.points = o3d.utility.Vector3dVector(rec_occu_ibs_pc)
    o3d_recon_ibs.paint_uniform_color([1, 1, 0])
    o3d_recon_cont = o3d.geometry.PointCloud()
    o3d_recon_cont.points = o3d.utility.Vector3dVector(rec_cont_ibs_pc)
    o3d_recon_cont.paint_uniform_color([0, 1, 1])
    o3d_recon_thumb = o3d.geometry.PointCloud()
    o3d_recon_thumb.points = o3d.utility.Vector3dVector(rec_thumb_cont_pc)
    o3d_recon_thumb.paint_uniform_color([1, 0, 1])
    o3d.visualization.draw_geometries([o3d_ibs, o3d_cont, o3d_thumb, o3d_recon_ibs, o3d_recon_cont, o3d_recon_thumb], window_name="recon ibs")
    o3d.visualization.draw_geometries([o3d_ibs, o3d_cont, o3d_thumb], window_name="origin ibs")
    o3d.visualization.draw_geometries([o3d_recon_ibs, o3d_recon_cont, o3d_recon_thumb], window_name="recon ibs")
    


