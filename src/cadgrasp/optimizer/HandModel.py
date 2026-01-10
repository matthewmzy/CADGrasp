import json
import numpy as np
import pytorch_kinematics as pk
import torch
import k3d
from plotly import graph_objects as go
import trimesh as tm
import transforms3d
import urdf_parser_py.urdf as URDF_PARSER
from pytorch_kinematics.urdf_parser_py.urdf import URDF, Box, Cylinder, Mesh, Sphere
from torchsdf import index_vertices_by_faces, compute_sdf
from cadgrasp.optimizer.rot6d import robust_compute_rotation_matrix_from_ortho6d
from xml.dom.minidom import parse
import pytorch3d.ops


def get_handmodel(robot, batch_size, device, hand_scale=1., use_collision=False, urdf_assets_meta_path="robot_models/meta/leap_hand/hand_assets_meta.json", sample_density=4e6):
    urdf_assets_meta = json.load(open(urdf_assets_meta_path))
    hand_model = HandModel(robot, urdf_assets_meta, batch_size=batch_size, device=device, scale=hand_scale, use_collision=use_collision, sample_density=sample_density)
    return hand_model


class HandModel:
    def __init__(self, hand_name, urdf_assets_meta,
                 batch_size=1, sample_density=4e6,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 scale=1., use_collision=True):
        # Device and robot configurations
        self.device = device
        self.hand_name = hand_name
        self.batch_size = batch_size
        self.sample_density = sample_density
        self.use_collision = use_collision
        self.xml_filename = urdf_assets_meta['xml_path'][hand_name]
        self.urdf_path = urdf_assets_meta['urdf_path'][hand_name]
        self.penetration_keypoints_meta = urdf_assets_meta['penetration_keypoints'][hand_name]

        # Build the kinematic chain from the URDF
        self.robot = pk.build_chain_from_urdf(open(self.urdf_path).read()).to(dtype=torch.float, device=self.device)
        self.robot_full = URDF_PARSER.URDF.from_xml_file(self.urdf_path)

        # Initialize revolute joints properties
        # Extract the revolute joint limits from the URDF file
        self.revolute_joints = [joint for joint in self.robot_full.joints if joint.joint_type == 'revolute']
        self.tendon_joints = [joint for joint in self.robot_full.joints if (joint.joint_type == 'revolute' and joint.mimic is None)]
        self.revolute_joints_q_lower = torch.Tensor([joint.limit.lower for joint in self.revolute_joints]).repeat([self.batch_size, 1]).to(device)
        self.revolute_joints_q_upper = torch.Tensor([joint.limit.upper for joint in self.revolute_joints]).repeat([self.batch_size, 1]).to(device)
        self.revolute_tendon_q_lower = torch.Tensor([joint.limit.lower for joint in self.tendon_joints]).repeat([self.batch_size, 1]).to(device)
        self.revolute_tendon_q_upper = torch.Tensor([joint.limit.upper for joint in self.tendon_joints]).repeat([self.batch_size, 1]).to(device)
        self.n_dofs = len(self.revolute_joints)

        # Placeholder for transformations
        self.global_translation = None
        self.global_rotation = None
        self.scale = scale
        
        # Default world-to-hand transformation (identity matrix)
        # This is used by IBSAdam to compute initial pose
        self.w2h_trans_meta = torch.eye(4, dtype=torch.float, device=self.device).unsqueeze(0).repeat(self.batch_size, 1, 1)
        
        self.penetration_keypoints = json.load(open(self.penetration_keypoints_meta, 'r'))

        # sample surface points
        self.surface_points = {}
        self.surface_points_normal = {}
        self.mesh_verts = {}
        self.mesh_faces = {}
        self.mesh_volumes = {}
        self.mesh_face_verts = {}
        self.sample_counts = {}

        visual = URDF.from_xml_string(open(self.urdf_path).read())

        self.links = [link.name for link in visual.links]

        
        for i_link, link in enumerate(visual.links):
            self.mesh_volumes[link.name] = 0
            # Load mesh
            if self.use_collision:
                components = link.collisions
            else:
                components = link.visuals
            if len(components) == 0:
                continue
            meshes = []
            for i in range(len(components)):
                if type(components[i].geometry) == Mesh:
                    filename = components[i].geometry.filename
                    # Handle package:// protocol prefix
                    if filename.startswith('package://'):
                        filename = filename.replace('package://', '')
                    mesh = tm.load(filename, force='mesh', process=False)
                    # print(f'link: {link.name} is volume: {mesh.is_volume}')
                elif type(components[i].geometry) == Cylinder:
                    mesh = tm.primitives.Cylinder(
                        radius=components[i].geometry.radius, height=components[i].geometry.length)
                elif type(components[i].geometry) == Box:
                    mesh = tm.primitives.Box(extents=components[i].geometry.size)
                elif type(components[i].geometry) == Sphere:
                    mesh = tm.primitives.Sphere(radius=components[i].geometry.radius)
                else:
                    print(type(components[i].geometry))
                    raise NotImplementedError
                # Get scale
                try:
                    scale = np.array(components[i].geometry.scale).reshape([1, 3])
                except :
                    scale = np.array([[1, 1, 1]])
                # Get rotation and translation
                try:
                    rotation = transforms3d.euler.euler2mat(*components[i].origin.rpy)
                    # translation = np.reshape(components[i].origin.xyz, [1, 3])
                    translation = np.array(components[i].origin.xyz)
                except AttributeError:
                    rotation = transforms3d.euler.euler2mat(0, 0, 0)
                    translation = np.array([[0, 0, 0]])
                transform = np.eye(4)
                transform[:3, :3] = rotation
                transform[:3, 3] = translation
                mesh.apply_scale(scale)
                mesh.apply_transform(transform)
                meshes.append(mesh)

            mesh = tm.util.concatenate(meshes)

            # Sample surface points
            self.mesh_volumes[link.name] = mesh.volume
            self.sample_counts[link.name] = min(int(mesh.volume*self.sample_density), 200)
            if mesh.volume < 1e-7:
                continue

            # Save mesh vertices and faces
            self.mesh_verts[link.name] = np.array(mesh.vertices)
            self.mesh_faces[link.name] = np.array(mesh.faces)
            self.mesh_face_verts[link.name] = index_vertices_by_faces(torch.tensor(self.mesh_verts[link.name],device=self.device), torch.tensor(self.mesh_faces[link.name],device=self.device))
            
            pts, pts_face_index = tm.sample.sample_surface(mesh=mesh, count=self.sample_counts[link.name]*20)
            pts_normal = np.array([mesh.face_normals[x] for x in pts_face_index], dtype=float)
            pts = torch.tensor(pts, dtype=torch.float, device=self.device).unsqueeze(0)
            pts, indices = pytorch3d.ops.sample_farthest_points(pts, K=self.sample_counts[link.name])
            pts = pts[0].detach().cpu().numpy()
            indices = indices[0].detach().cpu().numpy()
            pts_normal = pts_normal[indices]

            # Apply transformations
            pts *= self.scale
            # pts = np.matmul(rotation, pts.T).T + translation
            pts = np.concatenate([pts, np.ones([len(pts), 1])], axis=-1)
            
            if len(pts_normal.shape) == 1:
                pts_normal = np.expand_dims(pts_normal, axis=0)
            pts_normal = np.concatenate([pts_normal, np.ones([len(pts_normal), 1])], axis=-1)

            # Save surface points and normals
            self.surface_points[link.name] = torch.from_numpy(pts).to(self.device).float().unsqueeze(0).repeat(self.batch_size, 1, 1)
            self.surface_points_normal[link.name] = torch.from_numpy(pts_normal).to(self.device).float().unsqueeze(0).repeat(self.batch_size, 1, 1)
        
        # sample palmar surface points
        self.palmar_surface_points = {}
        self.palmar_surface_points_normal = {}

        DOMTree = parse(self.xml_filename)
        collection = DOMTree.documentElement
        links_data = collection.getElementsByTagName("PointCloudLinkData")

        for link in self.links:
            self.palmar_surface_points[link] = np.asarray([])
            self.palmar_surface_points_normal[link] = np.asarray([])

        for link_data in links_data:
            name = link_data.getElementsByTagName('linkName')[0].childNodes[0].data
            if self.mesh_volumes[name] < 1e-7:
                continue
            assert name in self.links, f"Link {name} not in the URDF file"
            link = name
            point_tmp = []
            normal_tmp = []
            point_data = link_data.getElementsByTagName('points')[0].getElementsByTagName('Vector3')
            normal_data = link_data.getElementsByTagName('normal')[0].getElementsByTagName('Vector3')
            for point in point_data:
                x = point.getElementsByTagName('x')[0].childNodes[0].data
                y = point.getElementsByTagName('y')[0].childNodes[0].data
                z = point.getElementsByTagName('z')[0].childNodes[0].data
                point_tmp.append([float(z), -float(x), float(y)])
            for normal in normal_data:
                x = normal.getElementsByTagName('x')[0].childNodes[0].data
                y = normal.getElementsByTagName('y')[0].childNodes[0].data
                z = normal.getElementsByTagName('z')[0].childNodes[0].data
                normal_tmp.append([float(z), -float(x), float(y)])

            point_tmp = torch.tensor(point_tmp, dtype=torch.float, device=self.device).unsqueeze(0)
            normal_tmp = torch.tensor(normal_tmp, dtype=torch.float, device=self.device).unsqueeze(0)
            tmp_size = point_tmp.shape[1]
            point_tmp, indices = pytorch3d.ops.sample_farthest_points(point_tmp, K=min(self.sample_counts[name], tmp_size))
            normal_tmp = normal_tmp[:,indices[0],:]
            self.palmar_surface_points[link] = torch.cat([point_tmp, torch.ones([1, point_tmp.shape[1], 1], device=self.device)], dim=-1).repeat(self.batch_size, 1, 1)
            self.palmar_surface_points_normal[link] = torch.cat([normal_tmp, torch.ones([1, normal_tmp.shape[1], 1], device=self.device)], dim=-1).repeat(self.batch_size, 1, 1)
             
    def update_kinematics(self, q):
        """Update the kinematic chain with new joint angles"""
        self.global_translation = q[:, :3]
        self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(q[:, 3:9])
        self.current_status = self.robot.forward_kinematics(q[:, 9:])

    def get_surface_points_and_normals(self, q=None, palmar=False):
        if q is not None:
            self.update_kinematics(q=q)
        # if not palmar:
        #     cprint("[WARNING] BUGS FOR ABILITY AND LEAP HAND FOR FULL MESH POINTS", 'yellow')
        points = []
        normals = []

        palmar_finger_points = []
        palmar_finger_normals = []

        thumb_finger_points = []
        thumb_finger_normals = []

        for link_name in self.surface_points:
            trans_matrix = self.current_status[link_name].get_matrix()
            if palmar and link_name in self.palmar_surface_points and len(self.palmar_surface_points[link_name]) > 0:
                points.append(torch.matmul(trans_matrix, self.palmar_surface_points[link_name].transpose(1, 2)).transpose(1, 2)[..., :3])
                normals.append(torch.matmul(trans_matrix, self.palmar_surface_points_normal[link_name].transpose(1, 2)).transpose(1, 2)[..., :3])
                if link_name in ["thumb_pip", "thumb_dip", "thumb_fingertip", "link_13.0", "link_14.0", "link_15.0", "link_15.0_tip"]:
                    thumb_finger_points.append(torch.matmul(trans_matrix, self.palmar_surface_points[link_name].transpose(1, 2)).transpose(1, 2)[..., :3])
                    thumb_finger_normals.append(torch.matmul(trans_matrix, self.palmar_surface_points_normal[link_name].transpose(1, 2)).transpose(1, 2)[..., :3])
                elif link_name not in ["hand_base_link", "pip_4", "palm", "wrist", "link_12.0", "base_link"]:
                    palmar_finger_points.append(torch.matmul(trans_matrix, self.palmar_surface_points[link_name].transpose(1, 2)).transpose(1, 2)[..., :3])
                    palmar_finger_normals.append(torch.matmul(trans_matrix, self.palmar_surface_points_normal[link_name].transpose(1, 2)).transpose(1, 2)[..., :3])
            elif not palmar and link_name in self.surface_points and len(self.surface_points[link_name]) > 0:
                points.append(torch.matmul(trans_matrix, self.surface_points[link_name].transpose(1, 2)).transpose(1, 2)[..., :3])
                normals.append(torch.matmul(trans_matrix, self.surface_points_normal[link_name].transpose(1, 2)).transpose(1, 2)[..., :3])
            else:
                continue
        points = torch.cat(points, 1)
        normals = torch.cat(normals, 1)
        points = torch.matmul(self.global_rotation, points.transpose(1, 2)).transpose(1, 2) + self.global_translation.unsqueeze(1)
        normals = torch.matmul(self.global_rotation, normals.transpose(1, 2)).transpose(1, 2)
        if palmar:
            palmar_finger_points = torch.cat(palmar_finger_points, 1)
            palmar_finger_normals = torch.cat(palmar_finger_normals, 1)
            palmar_finger_points = torch.matmul(self.global_rotation, palmar_finger_points.transpose(1, 2)).transpose(1, 2) + self.global_translation.unsqueeze(1)
            palmar_finger_normals = torch.matmul(self.global_rotation, palmar_finger_normals.transpose(1, 2)).transpose(1, 2)
            thumb_finger_points = torch.cat(thumb_finger_points, 1)
            thumb_finger_normals = torch.cat(thumb_finger_normals, 1)
            thumb_finger_points = torch.matmul(self.global_rotation, thumb_finger_points.transpose(1, 2)).transpose(1, 2) + self.global_translation.unsqueeze(1)
            thumb_finger_normals = torch.matmul(self.global_rotation, thumb_finger_normals.transpose(1, 2)).transpose(1, 2)
            return points * self.scale, normals, palmar_finger_points * self.scale, palmar_finger_normals, thumb_finger_points * self.scale, thumb_finger_normals
        else:
            return points * self.scale, normals

    def get_self_penetration(self, q=None, get_points=False):
        if q is not None:
            self.update_kinematics(q=q)

        batch_size = self.batch_size
        points_list = []
        link_indices_list = []

        for i, link_name in enumerate(self.penetration_keypoints):
            keypoints = self.penetration_keypoints[link_name]
            if not keypoints:  # 跳过空的关键点
                continue
            
            keypoints = torch.tensor(keypoints, dtype=torch.float, device=self.device)
            if keypoints.ndim == 1:
                keypoints = keypoints.unsqueeze(0)
            keypoints = keypoints.unsqueeze(0).expand(batch_size, -1, -1)  # 扩展批量大小，无需重复拼接
            points_list.append(keypoints)
            link_indices_list.append(torch.full((batch_size, keypoints.shape[1]), i, dtype=torch.int, device=self.device))
        
        points = torch.cat(points_list, dim=1)  # (batch_size, total_keypoints, 3)
        link_indices = torch.cat(link_indices_list, dim=1)  # (batch_size, total_keypoints)
        points_homogeneous = torch.cat([points, torch.ones(batch_size, points.shape[1], 1, device=self.device)], dim=2)  # 齐次坐标

        transforms_list = []
        for link_name in self.penetration_keypoints:
            transform_matrix = self.current_status[link_name].get_matrix()
            if transform_matrix.shape[0] != batch_size:
                transform_matrix = transform_matrix.expand(batch_size, -1, -1)
            transforms_list.append(transform_matrix)
            
        transforms = torch.stack(transforms_list, dim=1)  # (batch_size, num_links, 4, 4)

        link_indices = link_indices.long()
        selected_transforms = transforms.gather(1, link_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 4))
        points_transformed = (selected_transforms @ points_homogeneous.unsqueeze(3))[:, :, :3, 0]
        
        points_transformed = points_transformed @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        if get_points:
            return points_transformed

        dis = torch.cdist(points_transformed, points_transformed, p=2) + 1e-13
        dis = torch.where(dis < 1e-6, torch.full_like(dis, 1e6), dis)  # 防止除零误差

        thred = 0.025
        E_spen = torch.clamp(thred - dis, min=0)

        return E_spen.sum(dim=(1, 2))

    def calculate_distances1(self, point_cloud, q=None):
        """
        use pv
        """
        sdf_dis, sdf_grad = self.robot_sdf(point_cloud)
        return sdf_dis[0]

    def calculate_distances(self, point_cloud, q=None):
        """
        Calculate the distance between the hand and the point cloud.
        :param point_cloud: (batch_size, num_points, 3) Tensor of points in the point cloud.
        :param q: (batch_size, num_dofs) Tensor of joint angles.
        :return: (batch_size, num_points) Tensor of distances.
        """
        if q is not None:
            self.update_kinematics(q=q)

        # Transform the point cloud to each link's local coordinate system
        batch_size, num_points, _ = point_cloud.size()
        dis = []

        x = (point_cloud - self.global_translation.unsqueeze(1)) @ self.global_rotation#.transpose(1, 2)

        # Consider each link separately
        for link_name in self.mesh_face_verts:
            '''
            palm,ffknuckle,ffproximal,ffmiddle,ffdistal,
            mfknuckle,mfproximal,mfmiddle,mfdistal,
            rfknuckle,rfproximal,rfmiddle,rfdistal,
            lfmetacarpal,lfknuckle,lfproximal,lfmiddle,lfdistal
            thproximal,thmiddle,thdistal
            '''
            # print(link_name)
            # if link_name not in ['palm','ffknuckle','ffmiddle','ffproximal','ffdistal']:
            #     continue
            matrix = self.current_status[link_name].get_matrix()
            x_local = (x - matrix[:, :3, 3].unsqueeze(1)) @ matrix[:, :3, :3]#.transpose(1, 2)

            face_verts = self.mesh_face_verts[link_name]
            dis_local, dis_signs, _, _ = compute_sdf(x_local.reshape(-1, 3), face_verts.to(torch.float32))
            dis_local = torch.sqrt(dis_local + 1e-8)
            dis_local = dis_local*dis_signs
            dis.append(dis_local.reshape(batch_size, num_points))

        # The final distance is the minimum across all links
        dis = torch.stack(dis, dim=0)
        min_dis = torch.min(dis, dim=0)[0]
        return min_dis

    def get_meshes_from_q(self, q=None, i=0):
        data = []
        if q is not None: self.update_kinematics(q)
        for idx, link_name in enumerate(self.mesh_verts):
            trans_matrix = self.current_status[link_name].get_matrix()
            trans_matrix = trans_matrix[min(len(trans_matrix) - 1, i)].detach().cpu().numpy()
            v = self.mesh_verts[link_name]
            transformed_v = np.concatenate([v, np.ones([len(v), 1])], axis=-1)
            transformed_v = np.matmul(trans_matrix, transformed_v.T).T[..., :3]
            transformed_v = np.matmul(self.global_rotation[i].detach().cpu().numpy(),
                                      transformed_v.T).T + np.expand_dims(
                self.global_translation[i].detach().cpu().numpy(), 0)
            transformed_v = transformed_v * self.scale
            f = self.mesh_faces[link_name]
            data.append(tm.Trimesh(vertices=transformed_v, faces=f))
        return data
    
    def get_plotly_data(self, q=None, i=0, color='lightblue', opacity=1.):
        data = []
        if q is not None: self.update_kinematics(q)
        for link_name in self.mesh_verts:
            trans_matrix = self.current_status[link_name].get_matrix()
            trans_matrix = trans_matrix[min(len(trans_matrix) - 1, i)].detach().cpu().numpy()
            v = self.mesh_verts[link_name]
            transformed_v = np.concatenate([v, np.ones([len(v), 1])], axis=-1)
            transformed_v = np.matmul(trans_matrix, transformed_v.T).T[..., :3]
            transformed_v = np.matmul(self.global_rotation[i].detach().cpu().numpy(),
                                      transformed_v.T).T + np.expand_dims(
                self.global_translation[i].detach().cpu().numpy(), 0)
            transformed_v = transformed_v * self.scale
            f = self.mesh_faces[link_name]
            data.append(
                go.Mesh3d(x=transformed_v[:, 0], y=transformed_v[:, 1], z=transformed_v[:, 2], i=f[:, 0], j=f[:, 1],
                          k=f[:, 2], color=color, opacity=opacity))
        return data
    
    def get_k3d_data(self, q=None, i=0, opacity=1., concat=False, color=0x808080):
        data = []
        if q is not None: self.update_kinematics(q)
        if concat:
            verts = []
            faces = []
            num_v = 0
        for link_name in self.mesh_verts:
            trans_matrix = self.current_status[link_name].get_matrix()
            trans_matrix = trans_matrix[min(len(trans_matrix) - 1, i)].detach().cpu().numpy()
            v = self.mesh_verts[link_name]
            transformed_v = np.concatenate([v, np.ones([len(v), 1])], axis=-1)
            transformed_v = np.matmul(trans_matrix, transformed_v.T).T[..., :3]
            transformed_v = np.matmul(self.global_rotation[i].detach().cpu().numpy(),
                                      transformed_v.T).T + np.expand_dims(
                self.global_translation[i].detach().cpu().numpy(), 0)
            transformed_v = (transformed_v * self.scale).astype(np.float32)
            f = self.mesh_faces[link_name].astype(np.uint32)
            if concat:
                verts.append(transformed_v)
                faces.append(f + num_v)
                num_v += len(transformed_v)
            else:
                data.append(
                    k3d.mesh(vertices=transformed_v[:, :3], indices=f, opacity=opacity, color=color)
                )
        if concat:
            return k3d.mesh(
                vertices=np.concatenate(verts, axis=0).astype(np.float32), 
                indices=np.concatenate(faces, axis=0).astype(np.uint32), 
                opacity=opacity, 
                color=color
            )
        return data
    
    def get_palmar_points_k3d(self, i=0, color1=0x000F0F, color2=0xFFF0F0, opacity=0.5):
        _,_,pts,_,thumb_pts,_ = self.get_surface_points_and_normals(palmar=True)
        return k3d.points(pts[i, :, :3].detach().cpu().numpy().astype(np.float32), point_size=0.003, color=color1, opacity=opacity), \
            k3d.points(thumb_pts[i, :, :3].detach().cpu().numpy().astype(np.float32), point_size=0.003, color=color2, opacity=opacity)


if __name__ == '__main__':
    def plot_point_cloud(pts, color='lightblue', mode='markers', size=3.):
        return go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode=mode,
            marker=dict(
                color=color,
                size=size
            )
        )
    
    device = torch.device('cpu')
    hand_model = get_handmodel(
        'leap_hand', 1, device, 1., 
        use_collision=False, 
        urdf_assets_meta_path="robot_models/meta/leap_hand/hand_assets_meta.json",
        sample_density=2e7
    )
    
    print("Joint names:", hand_model.robot.get_joint_parameter_names())
    print("Link names:", [link.name for link in hand_model.robot_full.links])
    
    joint_lower = np.array(hand_model.revolute_joints_q_lower.cpu().reshape(-1))
    joint_upper = np.array(hand_model.revolute_joints_q_upper.cpu().reshape(-1))
    joint_mid = (joint_lower + joint_upper) / 2
    joints_q = (joint_mid + joint_lower) / 2
    
    q = torch.from_numpy(np.concatenate([
        np.array([0, 0, 0, 1, 0, 0, 0, 1, 0]),  # translation (3) + rotation 6D (6)
        joints_q
    ])).unsqueeze(0).to(device).float()
    
    data = hand_model.get_plotly_data(q=q, opacity=0.5)
    surface_points, _, palmar_points, _, thumb_points, _ = hand_model.get_surface_points_and_normals(q, palmar=True)
    data += [plot_point_cloud(surface_points.cpu().squeeze(0).numpy(), color='yellow')]
    data += [plot_point_cloud(palmar_points.cpu().squeeze(0).numpy(), color='green')]
    data += [plot_point_cloud(thumb_points.cpu().squeeze(0).numpy(), color='blue')]
    data += [plot_point_cloud(hand_model.get_self_penetration(q, get_points=True).cpu().squeeze(0).numpy(), color='black', size=8)]
    
    fig = go.Figure(data=data)
    fig.show()

