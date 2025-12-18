import os
from typing import Union
from isaacgym import gymapi, gymtorch

from urdf_parser_py.urdf import Robot
import yaml
import numpy as np
from pytorch3d.transforms import matrix_to_quaternion
from ipdb import set_trace

from src.utils.simulator.simulator import Simulator
import torch

class IsaacGymSimulator(Simulator):
    """
    class for Isaac Gym simulator
    """
    
    # drive mode dict, for setting dof properties
    # currently only supports positional control
    drive_mode_dict = dict(
        pos=gymapi.DOF_MODE_POS,
        # vel=gymapi.DOF_MODE_VEL,
        # effort=gymapi.DOF_MODE_EFFORT,
        none=gymapi.DOF_MODE_NONE,
    )
    
    def __init__(
        self, 
        config: dict, 
        num_envs: int, 
        device_id: int, 
        headless: bool,
    ):
        """
        initialize the class
        
        Args:
        - config: dict, config of the simulator
        """
        super().__init__(config, num_envs, device_id, headless)
        
        self._device = torch.device(f'cuda:{self._device_id}')

        self._init_sim()
        
        # create empty lists/dicts for the handles of envs, assets, and actors
        self._asset_handle = {}
        self._env_list = []
        self._actor_indices = {}
        
        # create empty dicts for indexing actor dofs, bodies, sensors, and dof sensors
        self._total_dofs = 0
        self._total_rigid_bodies = 0
        self._total_force_sensors = 0
        self._total_dof_force_sensors = 0
        self._actor_dof_indices = {}
        self._actor_rigid_body_indices = {}
        self._actor_force_sensor_indices = {}
        self._actor_dof_force_sensor_indices = {}
        
        # create empty list for rigid body mass
        self._rigid_body_mass = []
        
        # create empty dict for asset collision filters
        self._collision_filters = {}
        
        # create empty cache for state tensors
        self._actor_state_cache = {}
        
        # create dict for gravity toggles
        self._enable_gravity = {}
    
    def _init_sim(self):
        """
        initialize gym, configure sim, and add ground plane
        """
        
        # initialize gym
        self._gym = gymapi.acquire_gym()
        
        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.dt = 1. / self._config['env_hz']
        sim_params.substeps = 2
        sim_params.num_client_threads = 0
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.contact_offset = 0.002
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.bounce_threshold_velocity = 0.2
        sim_params.physx.max_depenetration_velocity = 1000.0
        sim_params.physx.default_buffer_size_multiplier = 5.0
        sim_params.use_gpu_pipeline = True
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = True
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
        device_id = self._device_id
        
        if not self._headless:
            self._sim = self._gym.create_sim(device_id, device_id, gymapi.SIM_PHYSX, sim_params)
        else:
            self._sim = self._gym.create_sim(device_id, -1, gymapi.SIM_PHYSX, sim_params)

        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self._gym.add_ground(self._sim, plane_params)

    def register_asset(
        self, 
        asset_name: str,
        asset_root: str,
        asset_path: str,
        asset_config: Union[dict, None],
    ) -> dict:
        """
        register an asset to the simulator
        
        Args:
        - asset_name: str, name of the asset
        - asset_root: str, root directory of the asset
        - asset_path: str, path to the asset
        - asset_config: Union[dict, None], config of the asset, format: {
            "asset_options": dict[str, str], gymapi.AssetOptions key-value pairs, \n
            "force_sensors": list[str], list of rigid body names for force sensors, \n
            "body_names": Optional[list[str]], list of rigid body names to find indices, \n
        }
        
        Returns:
        - asset_info: dict, info of the asset, format: { \n
            "num_dofs": int, number of dofs, \n
            "dof_lower": torch.Tensor[num_dofs], \n
            "dof_upper": torch.Tensor[num_dofs], \n
            "body_indices": torch.Tensor[num_bodies], \n
        }
        """
        
        assert asset_name not in self._asset_handle, f"asset {asset_name} already exists"
        
        # read everything from config if not provided, index by asset_path
        if asset_root == '':
            asset_root = self._config['asset'][asset_path]['asset_root']
            asset_config = self._config['asset'][asset_path]['asset_config']
            asset_path = self._config['asset'][asset_path]['asset_path']
        
        # read asset config from config if not provided, index by asset_name
        if asset_config is None:
            if asset_name.startswith('object'):
                asset_config = self._config['asset']['object']['asset_config']
            else:
                asset_config = self._config['asset'][asset_name]['asset_config']
        
        # parse asset options
        asset_options = gymapi.AssetOptions()
        for key, value in asset_config['asset_options'].items():
            setattr(asset_options, key, eval(str(value)))
        if asset_name.startswith('object'):
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params = gymapi.VhacdParams()
            asset_options.vhacd_params.resolution = 100000
        
        # load asset
        asset = self._gym.load_asset(self._sim, asset_root, asset_path, asset_options)
        
        # create asset force sensors
        body_handles = []
        for body_name in asset_config.get('force_sensors', []):
            body_handle = self._gym.find_asset_rigid_body_index(asset, body_name)
            body_handles.append(body_handle)
        body_handles.sort()
        for body_handle in body_handles:
            sensor_pose = gymapi.Transform()
            self._gym.create_asset_force_sensor(asset, body_handle, sensor_pose)
        
        # find asset rigid body indices
        body_indices = []
        for body_name in asset_config.get('body_names', []):
            body_indices.append(self._gym.find_asset_rigid_body_index(asset, body_name))
        body_indices.sort()
        body_indices = torch.tensor(body_indices, dtype=torch.long, device=self._device)
        
        # save asset handle
        self._asset_handle[asset_name] = asset
        
        # return asset info
        dof_props = self._gym.get_asset_dof_properties(asset)
        dof_lower = torch.tensor(dof_props['lower'], dtype=torch.float, device=self._device)
        dof_upper = torch.tensor(dof_props['upper'], dtype=torch.float, device=self._device)
        asset_info = dict(
            num_dofs=len(dof_props),
            dof_names=self._gym.get_asset_dof_names(asset),
            dof_lower=dof_lower,
            dof_upper=dof_upper,
            body_indices=body_indices,
        )
        return asset_info
    
    def create_env(self):
        """
        create an environment, called after registering all assets
        """
        
        # create env
        spacing = self._config['env_spacing']
        num_per_row = int(self._num_envs ** 0.5)
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        env = self._gym.create_env(self._sim, lower, upper, num_per_row)
                
        # save env handle
        self._env_list.append(env)
    
    def create_actor(
        self, 
        actor_name: str,
        asset_name: str,
        actor_config: Union[dict, str],
    ):
        """
        create an actor to the last created environment
        
        Args:
        - actor_name: str, name of the actor, same across all environments
        - asset_name: str, name of the asset
        - actor_config: Union[dict, str], config of the actor, format: {
            "no_collision": bool, whether the actor has no collision, 
            "filter": int, collision filter of the actor, 
            "segmentation_id": int, segmentation id of the actor, 
            "friction": float, friction of the actor, 
            "dof_props": Optional[dict[str, dict]], dof properties of the actor,
            "dof_force_sensors": bool, whether to enable dof force sensors
            "mass": Optional[float], mass of the actor
        }
        """
        
        # read actor config from config if not provided, index by actor_config
        if type(actor_config) == str:
            actor_config = self._config['actor'][actor_config]['actor_config']
        
        # get env handle and asset handle
        env = self._env_list[-1]
        asset = self._asset_handle[asset_name]
        
        # create actor
        collision_group = len(self._env_list) - 1
        if actor_config['no_collision']:
            collision_group += self._num_envs
        filter = actor_config['filter']
        segmentation_id = actor_config['segmentation_id']
        actor = self._gym.create_actor(env, asset, gymapi.Transform(), 
            actor_name, collision_group, filter, segmentation_id)
        
        # set rigid shape properties: friction
        shape_props = self._gym.get_actor_rigid_shape_properties(
            env, actor)
        for i in range(len(shape_props)):
            shape_props[i].friction = actor_config['friction']
        self._gym.set_actor_rigid_shape_properties(
            env, actor, shape_props)
        
        # set dof properties: driveMode, stiffness, damping, effort, velocity
        if 'dof_props' in actor_config:
            dof_props = self._gym.get_actor_dof_properties(env, actor)
            dof_names = self._gym.get_actor_dof_names(env, actor)
            if dof_names[0] not in actor_config['dof_props']:
                actor_config['dof_props'] = {
                    dof_name: actor_config['dof_props'] for dof_name in dof_names}
            for i, dof_name in enumerate(dof_names):
                config_dict = actor_config['dof_props'][dof_name]
                if 'driveMode' in config_dict:
                    dof_props['driveMode'][i] = self.drive_mode_dict[config_dict['driveMode']]
                if 'stiffness' in config_dict:
                    dof_props['stiffness'][i] = config_dict['stiffness']
                if 'damping' in config_dict:
                    dof_props['damping'][i] = config_dict['damping']
                if 'effort' in config_dict:
                    dof_props['effort'][i] = config_dict['effort']
                if 'velocity' in config_dict:
                    dof_props['velocity'][i] = config_dict['velocity']
            self._gym.set_actor_dof_properties(env, actor, dof_props)
        
        # set rigid body properties: mass, inertia
        if 'mass' in actor_config:
            rigid_props = self._gym.get_actor_rigid_body_properties(
                env, actor)
            current_mass = sum([rigid_props[i].mass for i in range(len(rigid_props))])
            for i in range(len(rigid_props)):
                rigid_props[i].mass *= actor_config['mass'] / current_mass
                rigid_props[i].invMass = 1 / rigid_props[i].mass
                rigid_props[i].inertia.x *= actor_config['mass'] / current_mass
                rigid_props[i].inertia.y *= actor_config['mass'] / current_mass
                rigid_props[i].inertia.z *= actor_config['mass'] / current_mass
            self._gym.set_actor_rigid_body_properties(
                env, actor, rigid_props)
                
        # enable dof force sensors
        if actor_config['dof_force_sensors']:
            self._gym.enable_actor_dof_force_sensors(env, actor)
        
        # save actor handle and update indices
        
        if actor_name not in self._actor_indices:
            self._actor_indices[actor_name] = []
            self._actor_dof_indices[actor_name] = []
            self._actor_rigid_body_indices[actor_name] = []
            self._actor_force_sensor_indices[actor_name] = []
            self._actor_dof_force_sensor_indices[actor_name] = []
        
        # save actor handle
        actor_index = self._gym.get_actor_index(env, actor, gymapi.DOMAIN_SIM)
        self._actor_indices[actor_name].append(actor_index)
        
        # update dof indices
        dof_count = self._gym.get_actor_dof_count(env, actor)
        self._actor_dof_indices[actor_name].append(list(range(
            self._total_dofs, self._total_dofs + dof_count)))
        self._total_dofs += dof_count
        
        # update rigid body indices
        rigid_body_count = self._gym.get_actor_rigid_body_count(env, actor)
        self._actor_rigid_body_indices[actor_name].append(list(range(
            self._total_rigid_bodies, self._total_rigid_bodies + rigid_body_count)))
        self._total_rigid_bodies += rigid_body_count

        # update force sensor indices
        force_sensor_count = self._gym.get_actor_force_sensor_count(env, actor)
        self._actor_force_sensor_indices[actor_name].append(list(range(
            self._total_force_sensors, self._total_force_sensors + force_sensor_count)))
        self._total_force_sensors += force_sensor_count
        
        # update dof force sensor indices
        dof_force_sensors = dof_count if actor_config['dof_force_sensors'] else 0
        self._actor_dof_force_sensor_indices[actor_name].append(list(range(
            self._total_dof_force_sensors, self._total_dof_force_sensors + dof_force_sensors)))
        self._total_dof_force_sensors += dof_force_sensors
        
        # update rigid body mass
        rigid_body_props = self._gym.get_actor_rigid_body_properties(env, actor)
        rigid_body_mass = [rigid_body_props[i].mass for i in range(len(rigid_body_props))]
        self._rigid_body_mass.extend(rigid_body_mass)
    
    def _tensorfy_actor_indices(self):
        """
        tensorfy actor indices
        """
        
        for actor_name in self._actor_indices:
            
            # [num_envs]
            self._actor_indices[actor_name] = torch.tensor(
                self._actor_indices[actor_name], 
                dtype=torch.long, device=self._device)
            
            # [num_envs, actor_num_dofs]
            self._actor_dof_indices[actor_name] = torch.tensor(
                self._actor_dof_indices[actor_name], 
                dtype=torch.long, device=self._device)
            
            # [num_envs, actor_num_rigid_bodies]
            self._actor_rigid_body_indices[actor_name] = torch.tensor(
                self._actor_rigid_body_indices[actor_name], 
                dtype=torch.long, device=self._device)
            
            # [num_envs, actor_num_force_sensors]
            self._actor_force_sensor_indices[actor_name] = torch.tensor(
                self._actor_force_sensor_indices[actor_name], 
                dtype=torch.long, device=self._device)
            
            # [num_envs, actor_num_dof_force_sensors]
            self._actor_dof_force_sensor_indices[actor_name] = torch.tensor(
                self._actor_dof_force_sensor_indices[actor_name], 
                dtype=torch.long, device=self._device)
        
        # [num_total_rigid_bodies]
        self._rigid_body_mass = torch.tensor(
            self._rigid_body_mass, dtype=torch.float, device=self._device)
    
    def _acquire_state_tensors(self):
        """
        acquire state tensors
        """
        
        # acquire state tensors
        self._root_state_tensor: torch.Tensor = gymtorch.wrap_tensor(
            self._gym.acquire_actor_root_state_tensor(self._sim)
            ).view(-1, 13)      # [total_actors, 13]
        self._dof_state_tensor: torch.Tensor = gymtorch.wrap_tensor(
            self._gym.acquire_dof_state_tensor(self._sim)
            ).view(-1, 2)       # [total_dofs, 2]
        self._rigid_body_state_tensor: torch.Tensor = gymtorch.wrap_tensor(
            self._gym.acquire_rigid_body_state_tensor(self._sim)
            ).view(-1, 13)      # [total_rigid_bodies, 13]
        if self._total_force_sensors > 0:
            self._force_sensor_tensor: torch.Tensor = gymtorch.wrap_tensor(
                self._gym.acquire_force_sensor_tensor(self._sim)
                ).view(-1, 6)   # [total_force_sensors, 6]
        else:
            self._force_sensor_tensor: torch.Tensor = torch.zeros(
                [0, 6], dtype=torch.float32, device=self._device)
        if self._total_dof_force_sensors > 0:
            self._dof_force_tensor: torch.Tensor = gymtorch.wrap_tensor(
                self._gym.acquire_dof_force_tensor(self._sim)
                ).view(-1)      # [total_dof_force_sensors]
        else:
            self._dof_force_tensor: torch.Tensor = torch.zeros(
                [0], dtype=torch.float32, device=self._device)
        
        # check shapes for debugging
        assert self._root_state_tensor.shape[0] == sum([
            len(self._actor_indices[actor_name]) for actor_name in self._actor_indices])
        assert self._dof_state_tensor.shape[0] == self._total_dofs
        assert self._rigid_body_state_tensor.shape[0] == self._total_rigid_bodies
        assert self._force_sensor_tensor.shape[0] == self._total_force_sensors
        assert self._dof_force_tensor.shape[0] == self._total_dof_force_sensors
        
        # impossible to create views because tensor indexing always returns a new tensor
        
        # allocate buffer for dof position targets, since Isaac Gym doesn't expose it
        self._dof_position_target_tensor = torch.zeros(
            self._total_dofs, dtype=torch.float32, device=self._device)
    
    def _create_viewer(self):
        """
        create viewer
        """
        
        # viewer state
        self._enable_viewer_sync = True
        
        # configure camera
        self._viewer = self._gym.create_viewer(
            self._sim, gymapi.CameraProperties())
        if self._viewer is None:
            print("*** Failed to create viewer")
            quit()
        
        # subscribe to keyboard shortcuts
        self._gym.subscribe_viewer_keyboard_event(
            self._viewer, gymapi.KEY_ESCAPE, "QUIT")
        self._gym.subscribe_viewer_keyboard_event(
            self._viewer, gymapi.KEY_V, "toggle_viewer_sync")
        
        # set camera position
        cam_pos = gymapi.Vec3(*self._config['camera_pos'])
        cam_target = gymapi.Vec3(*self._config['camera_target'])
        self._gym.viewer_camera_look_at(
            self._viewer, None, cam_pos, cam_target)
        
        # render first frame
        self._gym.fetch_results(self._sim, True)
        self._gym.step_graphics(self._sim)
        self._gym.draw_viewer(self._viewer, self._sim, False)
    
    def prepare_sim(self):
        """
        prepare the simulator before running, 
        called after creating all environments and actors. 
        - tensorfy actor indices
        - prepare sim
        - acquire state tensors
        - create viewer
        """
        
        # tensorfy actor indices
        self._tensorfy_actor_indices()
        
        # prepare sim
        self._gym.prepare_sim(self._sim)
        
        # acquire state tensors
        self._acquire_state_tensors()
        
        # enable gravity for all actors by default
        for actor_name in self._actor_indices:
            self._enable_gravity[actor_name] = True
        self._forces = torch.zeros([self._total_rigid_bodies, 3],
            dtype=torch.float, device=self._device)
        
        # create viewer
        if not self._headless:
            self._create_viewer()
        else:
            self._viewer = None
    
    def _refresh(self):
        """
        refresh state and force tensors
        """
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_dof_state_tensor(self._sim)
        self._gym.refresh_rigid_body_state_tensor(self._sim)
        self._gym.refresh_force_sensor_tensor(self._sim)
        self._gym.refresh_dof_force_tensor(self._sim)
    
    def get_actor_states(
        self, 
        actor_name: str,
    ) -> dict:
        """
        get the batched states of an actor
        
        Args:
        - actor_name: str, name of the actor
        
        Returns:
        - actor_states: dict, batched states of the actor, format: { \n
            "root_state": torch.Tensor[num_envs, 13], \n
            "root_pos": torch.Tensor[num_envs, 3], \n
            "root_rot": torch.Tensor[num_envs, 4], xyzw, \n
            "root_linvel": torch.Tensor[num_envs, 3], \n
            "root_angvel": torch.Tensor[num_envs, 3], \n
            
            "dof_state": torch.Tensor[num_envs, actor_num_dofs, 2], \n
            "dof_pos": torch.Tensor[num_envs, actor_num_dofs], \n
            "dof_vel": torch.Tensor[num_envs, actor_num_dofs], \n
            
            "body_state": torch.Tensor[num_envs, actor_num_bodies, 13], \n
            "body_pos": torch.Tensor[num_envs, actor_num_bodies, 3], \n
            "body_rot": torch.Tensor[num_envs, actor_num_bodies, 4], xyzw, \n
            "body_linvel": torch.Tensor[num_envs, actor_num_bodies, 3], \n
            "body_angvel": torch.Tensor[num_envs, actor_num_bodies, 3], \n
            
            "sensor_state": torch.Tensor[num_envs, actor_num_force_sensors, 6], \n
            "sensor_force": torch.Tensor[num_envs, actor_num_sensors, 3], \n
            "sensor_torque": torch.Tensor[num_envs, actor_num_sensors, 3], \n
            
            "dof_force": torch.Tensor[num_envs, actor_num_dofs], \n
        }
        """
        
        self._refresh()
        
        if actor_name in self._actor_state_cache:
            return self._actor_state_cache[actor_name]
        
        actor_states = {}
        
        # get root states
        actor_indices = self._actor_indices[actor_name]
        root_states = self._root_state_tensor[actor_indices]
        root_states[:, 2] -= self._config['table_height']
        actor_states.update(dict(
            root_state=root_states,
            root_pos=root_states[:, :3],
            root_rot=root_states[:, 3:7],
            root_linvel=root_states[:, 7:10],
            root_angvel=root_states[:, 10:],
        ))
        
        # get dof states
        dof_indices = self._actor_dof_indices[actor_name]
        dof_states = self._dof_state_tensor[dof_indices]
        actor_states.update(dict(
            dof_state=dof_states,
            dof_pos=dof_states[:, :, 0],
            dof_vel=dof_states[:, :, 1],
        ))
        
        # get body states
        rigid_body_indices = self._actor_rigid_body_indices[actor_name]
        body_states = self._rigid_body_state_tensor[rigid_body_indices]
        actor_states.update(dict(
            body_state=body_states,
            body_pos=body_states[:, :, :3],
            body_rot=body_states[:, :, 3:7],
            body_linvel=body_states[:, :, 7:10],
            body_angvel=body_states[:, :, 10:],
        ))
        
        # get force sensor states
        force_sensor_indices = self._actor_force_sensor_indices[actor_name]
        force_sensor_states = self._force_sensor_tensor[force_sensor_indices]
        actor_states.update(dict(
            sensor_state=force_sensor_states,
            sensor_force=force_sensor_states[:, :, :3],
            sensor_torque=force_sensor_states[:, :, 3:],
        ))
        
        # get dof force sensor states
        dof_force_sensor_indices = self._actor_dof_force_sensor_indices[actor_name]
        dof_force_sensor_states = self._dof_force_tensor[dof_force_sensor_indices]
        actor_states.update(dict(
            dof_force=dof_force_sensor_states,
        ))
        
        # cache the states
        # self._actor_state_cache[actor_name] = actor_states
        
        # no need to clone tensors, since tensor indexing always returns a new tensor
        return actor_states
    
    def _to_tensor(
        self, 
        data: Union[torch.Tensor, np.ndarray],
    ):
        """
        convert data to torch.Tensor
        
        Args:
        - data: Union[torch.Tensor, np.array], data to convert
        
        Returns:
        - tensor: torch.Tensor, converted tensor
        """
        
        if type(data) != torch.Tensor:
            return torch.tensor(data, dtype=torch.float, device=self._device)
        return data
    
    def set_actor_states(
        self,
        actor_name: str,
        actor_states: dict,
        env_ids: Union[torch.Tensor, None] = None,
    ):
        """
        set the batched states of an actor
        
        Args:
        - actor_name: str, name of the actor
        - actor_states: dict, batched states of the actor, format: { \n
            "root_pos": torch.Tensor[set_envs, 3], \n
            "root_rot": torch.Tensor[set_envs, 4], xyzw, \n
            "root_linvel": torch.Tensor[set_envs, 3], \n
            "root_angvel": torch.Tensor[set_envs, 3], \n
            
            "dof_pos": torch.Tensor[set_envs, actor_num_dofs], \n
            "dof_vel": torch.Tensor[set_envs, actor_num_dofs], \n
        }
        - env_ids: torch.Tensor[set_envs], envs to set states, defaults to all
        """
        
        self._refresh()
        
        # set root states
        actor_indices = self._actor_indices[actor_name]
        if env_ids is not None:
            actor_indices = actor_indices[env_ids]
        root_states = self._root_state_tensor[actor_indices]
        if 'root_pos' in actor_states:
            root_states[:, :3] = self._to_tensor(actor_states['root_pos'])
            root_states[:, 2] += self._config['table_height']
        if 'root_rot' in actor_states:
            root_rot = self._to_tensor(actor_states['root_rot'])
            if root_rot.shape[1] == 3:
                root_rot = matrix_to_quaternion(root_rot).roll(-1, dims=1)
            root_states[:, 3:7] = root_rot
        if 'root_linvel' in actor_states:
            root_states[:, 7:10] = self._to_tensor(actor_states['root_linvel'])
        if 'root_angvel' in actor_states:
            root_states[:, 10:] = self._to_tensor(actor_states['root_angvel'])
        self._root_state_tensor[actor_indices] = root_states
        
        # set dof states
        dof_indices = self._actor_dof_indices[actor_name]
        if env_ids is not None:
            dof_indices = dof_indices[env_ids]
        dof_states = self._dof_state_tensor[dof_indices]
        if 'dof_pos' in actor_states:
            dof_states[:, :, 0] = self._to_tensor(actor_states['dof_pos'])
        if 'dof_vel' in actor_states:
            dof_states[:, :, 1] = self._to_tensor(actor_states['dof_vel'])
        self._dof_state_tensor[dof_indices] = dof_states
        
        self._flush_states()
    
    def _flush_states(self):
        """
        flush the states to the simulator
        """
        self._gym.set_actor_root_state_tensor(
            self._sim, gymtorch.unwrap_tensor(self._root_state_tensor))
        self._gym.set_dof_state_tensor(
            self._sim, gymtorch.unwrap_tensor(self._dof_state_tensor))

    def set_actor_actions(
        self,
        actor_name: str,
        actor_actions: torch.Tensor,
    ):
        """
        set the batched actions of an actor
        
        Args:
        - actor_name: str, name of the actor
        - actor_actions: torch.Tensor[num_envs, actor_num_dofs], batched control signals
        """
        
        # set dof position targets
        dof_indices = self._actor_dof_indices[actor_name]
        self._dof_position_target_tensor[dof_indices] = self._to_tensor(actor_actions)
        self._flush_actions()
    
    def _flush_actions(self):
        """
        flush the actions to the simulator
        """
        self._gym.set_dof_position_target_tensor(
            self._sim, gymtorch.unwrap_tensor(self._dof_position_target_tensor))
    
    def disable_gravity(
        self, 
        actor_name: Union[str, None] = None,
    ):
        """
        disable gravity for an actor
        
        Args:
        - actor_name: Union[str, None], name of the actor
        """
        if actor_name is None:
            for actor_name in self._actor_indices:
                self._enable_gravity[actor_name] = False
            self._forces[:, 2] = 9.81 * self._rigid_body_mass
        else:
            self._enable_gravity[actor_name] = False
            rigid_body_indices = self._actor_rigid_body_indices[actor_name]
            rigid_body_mass = self._rigid_body_mass[rigid_body_indices]
            self._forces[rigid_body_indices, 2] = 9.81 * rigid_body_mass
    
    def enable_gravity(
        self, 
        actor_name: Union[str, None] = None,
    ):
        """
        enable gravity for an actor
        
        Args:
        - actor_name: Union[str, None], name of the actor
        """
        if actor_name is None:
            for actor_name in self._actor_indices:
                self._enable_gravity[actor_name] = True
            self._forces[:, 2] = 0.0
        else:
            self._enable_gravity[actor_name] = True
            rigid_body_indices = self._actor_rigid_body_indices[actor_name]
            self._forces[rigid_body_indices, 2] = 0.0

    def step(self):
        """
        step physics and update viewer
        """
        
        # flush states and actions
        # self._flush_states()
        # self._flush_actions()
        
        # take multiple steps
        for i in range(self._config['control_freq_inv']):
            
            # apply canceling forces to actors that disable gravity
            self._gym.apply_rigid_body_force_tensors(
                self._sim, gymtorch.unwrap_tensor(self._forces))
            
            # simulate
            self._gym.simulate(self._sim)
            self._gym.fetch_results(self._sim, True)
            
            # clear the speed of actors that disable gravity
            self._refresh()
            for actor_name in self._actor_indices:
                if not self._enable_gravity[actor_name]:
                    actor_indices = self._actor_indices[actor_name]
                    self._root_state_tensor[actor_indices, 7:] = 0.0
            self._flush_states()
            
            # update the viewer
            if self._viewer is not None:
                
                # check for window closed
                if self._gym.query_viewer_has_closed(self._viewer):
                    print("Viewer has been closed")
                    self.close()
                    exit(0)
                
                # check for keyboard events
                for evt in self._gym.query_viewer_action_events(self._viewer):
                    if evt.action == "QUIT" and evt.value > 0:
                        print("Excape key detected, closing viewer")
                        exit(0)
                    elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                        self._enable_viewer_sync = not self._enable_viewer_sync
                
                # step graphics
                if self._enable_viewer_sync:
                    self._gym.step_graphics(self._sim)
                    self._gym.draw_viewer(self._viewer, self._sim, True)
                    self._gym.sync_frame_time(self._sim)
                else:
                    self._gym.poll_viewer_events(self._viewer)
        
        # refresh state and force tensors
        self._refresh()
        
        # clear cache
        self._actor_state_cache = {}
    
    def close(self):
        """
        close the simulator
        """
        if not self._headless:
            self._gym.destroy_viewer(self._viewer)
        self._gym.destroy_sim(self._sim)