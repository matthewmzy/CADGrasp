import os
import yaml

import numpy as np

# from src.utils.data_evaluator.data_evaluator import DataEvaluator
from src.utils.simulator.isaacgym_simulator import IsaacGymSimulator
from src.utils.robot_model import RobotModel
from src.utils.width_mapper import WidthMapper
from src.utils.collision_checker import CollisionChecker
import time
from pytorch3d.transforms import matrix_to_euler_angles
import torch

class SimulationEvaluator():
    """
    class for simulation evaluator
    """
    
    def __init__(
        self, 
        config: dict, 
        device: torch.device,
    ):
        """
        initialize the class
        
        Args:
        - config: dict, config of the simulation evaluator
        - device: torch.device, device
        """
        self._config: dict = config
        self._device: torch.device = torch.device(device)
        
        # create robot model
        self._robot_model = RobotModel(config['urdf_path'], config['meta_path'])
        
        # create width mapper
        self._width_mapper = WidthMapper(self._robot_model, config['width_mapper_meta_path'])
        
        # load collision checker
        collision_checker_config_path = os.path.join(
            'configs/collision_checker', config['robot_name'], 'CollisionChecker.yaml')
        collision_checker_config = yaml.safe_load(open(collision_checker_config_path, 'r'))
        self._collision_checker = CollisionChecker(collision_checker_config, device)
    
    def set_environments(
        self, 
        object_pose_dict: dict, 
        object_surface_points_dict: dict, 
        num_envs: int, 
        dataset = 'graspnet'
    ):
        """
        set environments
        
        Args:
        - object_pose_dict: dict[str, np.ndarray[4, 4]], object code -> pose
        - object_surface_points_dict: dict[str, torch.tensor[num_points, 3]]
        - num_envs: int, number of environments
        """
        
        self._object_pose_dict = object_pose_dict
        self._object_surface_points_dict = object_surface_points_dict
        self._num_envs = num_envs
        
        # create simulator
        simulator_config_path = os.path.join(
            'configs/simulator', f'{self._config["simulator_type"]}.yaml')
        simulator_config = yaml.safe_load(open(simulator_config_path, 'r'))
        simulator_config['table_height'] = 0
        simulator_class = IsaacGymSimulator
        t = time.time()
        self._simulator = simulator_class(
            config=simulator_config,
            num_envs=num_envs,
            headless=self._config['headless'],
            device_id=int(str(self._device).split(':')[-1]))
        
        # register assets
        t = time.time()
        self._robot_info = self._simulator.register_asset(
            asset_name='robot', 
            asset_root='', 
            asset_path=self._config['robot_name'] + '_free', 
            asset_config={}, 
        )
        self._object_info_dict = {}
        if dataset == 'graspnet':
            for object_code in object_pose_dict:
                self._object_info_dict[object_code] = self._simulator.register_asset(
                    asset_name=f'object_{object_code}', 
                    asset_root=os.path.join('data/meshdata', object_code), 
                    asset_path='nontextured_simplified.urdf', 
                    asset_config=None, 
                )
        elif dataset == 'acronym':
            for object_code in object_pose_dict:
                self._object_info_dict[object_code] = self._simulator.register_asset(
                    asset_name=f'object_{object_code}', 
                    asset_root=os.path.join('data/acronym/meshes/models', object_code), 
                    asset_path='collision.urdf', 
                    asset_config=None, 
                )
        
        # create environments
        for i in range(num_envs):
            # create env
            self._simulator.create_env()
            # create actors
            self._simulator.create_actor(
                actor_name='robot',
                asset_name='robot',
                actor_config=self._config['robot_name'] + '_free', 
            )
            for object_index, object_code in enumerate(object_pose_dict):
                self._simulator.create_actor(
                    actor_name=f'object_{object_code}',
                    asset_name=f'object_{object_code}',
                    actor_config=dict(
                        no_collision=False, 
                        filter=2, 
                        segmentation_id=0, 
                        friction=1, 
                        dof_force_sensors=False, 
                        mass=0.1)
                )
        
        # prepare simulator
        self._simulator.prepare_sim()

    def _compute_waypoints(
        self,
        grasps: dict, 
    ):
        """
        compute waypoints: pregrasp, cover, grasp, squeeze, lift
        
        Args:
        - grasps: dict[str, np.ndarray], grasps, format: {
            'translation': np.ndarray[batch_size, 3], translations,
            'rotation': np.ndarray[batch_size, 3, 3], rotations,
            'jointxxx': np.ndarray[batch_size], joint values,
            ...
        }
        """
        
        batch_size = len(grasps['translation'])
        assert batch_size <= self._num_envs, \
            'batch size should be less than or equal to number of environments'
        
        self._waypoint_pose_list = []
        self._waypoint_qpos_dict_list = []
        self._waypoint_qpos_list = []
        dof_names = self._robot_info['dof_names'][6:]
        canonical_frame_rotation = torch.tensor(
            self._config['canonical_frame_rotation'], dtype=torch.float, device=self._device)
        
        # get grasp pose and qpos
        grasp_pose = torch.eye(4, dtype=torch.float, device=self._device
            ).unsqueeze(0).repeat(batch_size, 1, 1)
        grasp_pose[:, :3, 3] = torch.tensor(grasps['translation'],
            dtype=torch.float, device=self._device)
        grasp_pose[:, :3, :3] = torch.tensor(grasps['rotation'], 
            dtype=torch.float, device=self._device)
        grasp_qpos_dict = { 
            joint_name: torch.tensor(grasps[joint_name], dtype=torch.float, device=self._device)
            for joint_name in grasps 
            if joint_name not in ['translation', 'rotation'] }
        grasp_qpos = torch.stack([grasp_qpos_dict[joint_name]
            for joint_name in dof_names], dim=1)
        
        # waypoint 1 (pregrasp): relax fingers and move back along gripper x-axis for 10cm
        pregrasp_qpos_dict = self._width_mapper.squeeze_fingers(
            grasp_qpos_dict, -0.025, -0.025)[0]
        pregrasp_pose_local = torch.eye(4, dtype=torch.float, device=self._device
            ).unsqueeze(0).repeat(batch_size, 1, 1)
        pregrasp_pose_local[:, :3, 3] = canonical_frame_rotation.T @ \
            torch.tensor([-0.1, 0.0, 0.0], dtype=torch.float, device=self._device)
        pregrasp_pose = grasp_pose @ pregrasp_pose_local
        pregrasp_qpos = torch.stack([pregrasp_qpos_dict[joint_name]
            for joint_name in dof_names], dim=1)
        self._waypoint_pose_list.append(pregrasp_pose)
        self._waypoint_qpos_dict_list.append(pregrasp_qpos_dict.copy())
        self._waypoint_qpos_list.append(pregrasp_qpos)
        
        # waypoint 2 (cover): only relax fingers
        self._waypoint_pose_list.append(grasp_pose)
        self._waypoint_qpos_dict_list.append(pregrasp_qpos_dict.copy())
        self._waypoint_qpos_list.append(pregrasp_qpos)
        
        # waypoint 3 (grasp): input grasp pose and qpos
        self._waypoint_pose_list.append(grasp_pose)
        self._waypoint_qpos_dict_list.append(grasp_qpos_dict.copy())
        self._waypoint_qpos_list.append(grasp_qpos)
        
        # waypoint 4 (squeeze): only squeeze fingers
        target_qpos_dict = self._width_mapper.squeeze_fingers(
            grasp_qpos_dict, 0.015, 0.015, keep_z=True)[0]   # TODO: 0.03 previo
        target_qpos = torch.stack([target_qpos_dict[joint_name]
            for joint_name in dof_names], dim=1)
        self._waypoint_pose_list.append(grasp_pose)
        self._waypoint_qpos_dict_list.append(target_qpos_dict.copy())
        self._waypoint_qpos_list.append(target_qpos)
        
        # waypoint 5 (lift): squeeze fingers and lift
        # case 1 (top grasp): if gripper x-axis and gravity direction spans less than 60 degrees,
        # move back along gripper x-axis for 20cm
        # case 2 (side grasp): otherwise, move up along world z-axis for 20cm
        # case 1
        lift_pose_top_local = torch.eye(4, dtype=torch.float, device=self._device
            ).unsqueeze(0).repeat(batch_size, 1, 1)
        lift_pose_top_local[:, :3, 3] = canonical_frame_rotation.T @ \
            torch.tensor([-0.2, 0.0, 0.0], dtype=torch.float, device=self._device)
        lift_pose_top = grasp_pose @ lift_pose_top_local
        # case 2
        lift_pose_side = grasp_pose.clone()
        lift_pose_side[:, :3, 3] += torch.tensor([0.0, 0.0, 0.2], 
            dtype=torch.float, device=self._device)
        # compose
        gripper_x_axis = (grasp_pose[:, :3, :3] @ canonical_frame_rotation.T)[:, :, 0]
        gravity_direction = torch.tensor([0.0, 0.0, -1.0], 
            dtype=torch.float, device=self._device)
        top_mask = (gripper_x_axis * gravity_direction).sum(dim=1) > np.cos(np.pi / 3)
        lift_pose = torch.where(top_mask.unsqueeze(1).unsqueeze(1),
            lift_pose_top, lift_pose_side)
        self._waypoint_pose_list.append(lift_pose)
        self._waypoint_qpos_dict_list.append(target_qpos_dict.copy())
        self._waypoint_qpos_list.append(target_qpos)
        
        # pad waypoints
        if batch_size < self._num_envs:
            for i in range(len(self._waypoint_pose_list)):
                self._waypoint_pose_list[i] = torch.cat([
                    self._waypoint_pose_list[i], 
                    self._waypoint_pose_list[i][-1:].repeat(self._num_envs - batch_size, 1, 1)
                ], dim=0)
                for joint in self._waypoint_qpos_dict_list[i]:
                    self._waypoint_qpos_dict_list[i][joint] = torch.cat([
                        self._waypoint_qpos_dict_list[i][joint], 
                        self._waypoint_qpos_dict_list[i][joint][-1:]\
                            .repeat(self._num_envs - batch_size)
                    ], dim=0)
                self._waypoint_qpos_list[i] = torch.cat([
                    self._waypoint_qpos_list[i], 
                    self._waypoint_qpos_list[i][-1:].repeat(self._num_envs - batch_size, 1)
                ], dim=0)
        
        # compose pose and dof pos
        self._waypoint_qpos_all_list = []
        for i in range(len(self._waypoint_pose_list)):
            root_pos = self._waypoint_pose_list[i][:, :3, 3]
            root_rot = self._waypoint_pose_list[i][:, :3, :3]
            root_rot = matrix_to_euler_angles(root_rot, 'XYZ')  # convention equivalent to 'rxyz'
            dof_qpos = self._waypoint_qpos_list[i]
            dof_qpos_all = torch.cat([root_pos, root_rot, dof_qpos], dim=1)
            self._waypoint_qpos_all_list.append(dof_qpos_all)
    
    def _check_collision(self):
        """
        check collision between pregrasp and scene
        
        Returns:
        - pregrasp_valid: np.ndarray[num_envs, np.bool], pregrasp validity
        """
        grasps = {}
        grasps.update(self._waypoint_qpos_dict_list[0])
        grasps['translation'] = self._waypoint_pose_list[0][:, :3, 3]
        grasps['rotation'] = self._waypoint_pose_list[0][:, :3, :3]
        scene_point_cloud = torch.cat([
            torch.tensor(self._object_surface_points_dict[object_code], 
            dtype=torch.float, device=self._device) 
            for object_code in self._object_pose_dict], dim=0)
        scene_pen_distance, table_pen_distance = self._collision_checker.check_collision_batch(
            grasps, scene_point_cloud)
        scene_pen_distance = scene_pen_distance.cpu().numpy()
        table_pen_distance = table_pen_distance.cpu().numpy()
        scene_pen_valid = scene_pen_distance < self._config['scene_pen_threshold']
        table_pen_valid = table_pen_distance < self._config['table_pen_threshold']
        pregrasp_valid = scene_pen_valid & table_pen_valid
        return pregrasp_valid
    
    def _reset_environments(self):
        """
        reset environments
        """
        
        # set object states
        for object_code in self._object_pose_dict:
            object_pose = self._object_pose_dict[object_code]
            self._simulator.set_actor_states(
                actor_name=f'object_{object_code}',
                actor_states=dict(
                    root_pos=torch.tensor(object_pose[:3, 3], dtype=torch.float, 
                        device=self._device).unsqueeze(0).repeat(self._num_envs, 1),
                    root_rot=torch.tensor(object_pose[:3, :3],dtype=torch.float, 
                        device=self._device).unsqueeze(0).repeat(self._num_envs, 1, 1),
                    root_linvel=torch.zeros([self._num_envs, 3], 
                        dtype=torch.float, device=self._device),
                    root_angvel=torch.zeros([self._num_envs, 3], dtype=torch.float, 
                        device=self._device),
                )
            )
        
        # set robot states
        dof_pos_all = self._waypoint_qpos_all_list[0]
        self._simulator.set_actor_states(
            actor_name='robot',
            actor_states=dict(
                root_pos=torch.zeros([self._num_envs, 3], 
                    dtype=torch.float, device=self._device), 
                root_rot=torch.eye(3, dtype=torch.float, device=self._device
                    ).unsqueeze(0).repeat(self._num_envs, 1, 1), 
                root_linvel=torch.zeros([self._num_envs, 3], 
                    dtype=torch.float, device=self._device),
                root_angvel=torch.zeros([self._num_envs, 3], 
                    dtype=torch.float, device=self._device),
                dof_pos=dof_pos_all, 
                dof_vel=torch.zeros_like(dof_pos_all),
            )
        )
        self._simulator.set_actor_actions(
            actor_name='robot',
            actor_actions=dof_pos_all, 
        )
    
    def _execute_waypoints(self):
        """
        execute waypoints
        """
        
        # disable gravity before the first waypoint (pregrasp)
        for object_code in self._object_pose_dict:
            self._simulator.disable_gravity(f'object_{object_code}')
        # execute waypoints
        for i in range(1, len(self._waypoint_pose_list)):
            start_qpos_all = self._waypoint_qpos_all_list[i - 1]
            end_qpos_all = self._waypoint_qpos_all_list[i]
            steps = self._config['waypoint_steps'][i - 1]
            for step in range(steps):
                target_qpos = start_qpos_all + (end_qpos_all - start_qpos_all) * (step + 1) / steps
                self._simulator.set_actor_actions(
                    actor_name='robot',
                    actor_actions=target_qpos,
                )
                self._simulator.step()
            # enable gravity after the fourth waypoint (squeeze)
            if i == 3:
                for object_code in self._object_pose_dict:
                    self._simulator.enable_gravity(f'object_{object_code}')
    
    def _get_sim_successes(self):
        """
        get successful environments
        
        Returns:
        - successes: np.ndarray[num_envs, np.bool], successes
        """
        
        sim_successes = np.zeros(self._num_envs, dtype=bool)
        
        # successful if any object is lifted by 3cm
        for object_code in self._object_pose_dict:
            object_height_init = self._object_pose_dict[object_code][2, 3]
            object_height_final = self._simulator.get_actor_states(
                f'object_{object_code}')['root_pos'][:, 2].cpu().numpy()
            sim_successes |= (object_height_final > object_height_init + 0.03)
        
        return sim_successes

    def evaluate_data(
        self, 
        grasps: dict,
    ):
        """
        evaluate a batch of grasps on one cluttered scene
        
        Args:
        - grasps: dict[str, np.ndarray], grasps, format: {
            'translation': np.ndarray[batch_size, 3], translations,
            'rotation': np.ndarray[batch_size, 3, 3], rotations,
            'jointxxx': np.ndarray[batch_size], joint values,
            ...
        }
        
        Returns:
        - successes: np.ndarray[batch_size, np.bool], successes
        """
        
        # compute waypoints
        self._compute_waypoints(grasps)
        
        # check collision
        pregrasp_valid = self._check_collision()[:len(grasps['translation'])]
        
        # reset environments
        self._reset_environments()
        self._simulator.step()
        self._simulator.step()
        self._reset_environments()
        
        # execute waypoints
        self._execute_waypoints()
        
        # get sim_successes
        sim_successes = self._get_sim_successes()[:len(grasps['translation'])]
        
        successes = pregrasp_valid & sim_successes
        
        return successes
