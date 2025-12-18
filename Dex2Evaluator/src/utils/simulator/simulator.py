from typing import Optional
import numpy as np
import transforms3d


class Simulator:
    """
    base class for simulator
    """
    
    gravity_direction_to_scene_rotation = {
        '+x': transforms3d.euler.euler2mat(0, np.pi / 2, 0), 
        '-x': transforms3d.euler.euler2mat(0, -np.pi / 2, 0),
        '+y': transforms3d.euler.euler2mat(-np.pi / 2, 0, 0),
        '-y': transforms3d.euler.euler2mat(np.pi / 2, 0, 0),
        '+z': transforms3d.euler.euler2mat(np.pi, 0, 0), 
        '-z': np.eye(3),
    }   
    
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
        - num_envs: int, number of environments
        - device_id: int, device id
        - headless: bool, whether to run the simulator in headless mode
        """
        self._config: dict = config
        self._num_envs: int = num_envs
        self._device_id: int = device_id
        self._headless: bool = headless
    
    def register_asset(
        self, 
        asset_name: str,
        asset_root: str,
        asset_path: str,
        asset_config: dict,
    ) -> dict:
        """
        register an asset to the simulator
        
        Args:
        - asset_name: str, name of the asset
        - asset_root: str, root directory of the asset
        - asset_path: str, path to the asset
        - asset_config: dict, config of the asset, specific to the simulator
        
        Returns:
        - asset_info: dict, info of the asset, format: { \n
            "num_dofs": int, number of dofs, \n
            "dof_lower": torch.Tensor[num_dofs], \n
            "dof_upper": torch.Tensor[num_dofs], \n
        }
        """
        raise NotImplementedError()
    
    def create_env(self):
        """
        create an environment, called after registering all assets
        """
        raise NotImplementedError()
    
    def create_actor(
        self, 
        actor_name: str,
        asset_name: str,
        actor_config: dict,
    ):
        """
        create an actor to the last created environment
        
        Args:
        - actor_name: str, name of the actor, same across all environments
        - asset_name: str, name of the asset
        - actor_config: dict, config of the actor, specific to the simulator
        """
        raise NotImplementedError()
    
    def prepare_sim(self):
        """
        prepare the simulator before running, 
        called after creating all environments and actors
        """
        raise NotImplementedError()
    
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
            "root_pos": torch.Tensor[num_envs, 3], \n
            "root_rot": torch.Tensor[num_envs, 4], xyzw, \n
            "root_linvel": torch.Tensor[num_envs, 3], \n
            "root_angvel": torch.Tensor[num_envs, 3], \n
            
            "dof_pos": torch.Tensor[num_envs, num_dofs], \n
            "dof_vel": torch.Tensor[num_envs, num_dofs], \n
            
            "body_pos": torch.Tensor[num_envs, num_bodies, 3], \n
            "body_rot": torch.Tensor[num_envs, num_bodies, 4], xyzw, \n
            "body_linvel": torch.Tensor[num_envs, num_bodies, 3], \n
            "body_angvel": torch.Tensor[num_envs, num_bodies, 3], \n
            
            "sensor_force": torch.Tensor[num_envs, num_sensors, 3], \n
            "sensor_torque": torch.Tensor[num_envs, num_sensors, 3], \n
            
            "dof_force": torch.Tensor[num_envs, num_dofs], \n
        }
        """
        raise NotImplementedError()
    
    def set_actor_states(
        self,
        actor_name: str,
        actor_states: dict,
    ):
        """
        set the batched states of an actor
        
        Args:
        - actor_name: str, name of the actor
        - actor_states: dict, batched states of the actor, format: { \n
            "root_pos": torch.Tensor[num_envs, 3], \n
            "root_rot": torch.Tensor[num_envs, 4], xyzw, \n
            "root_linvel": torch.Tensor[num_envs, 3], \n
            "root_angvel": torch.Tensor[num_envs, 3], \n
            
            "dof_pos": torch.Tensor[num_envs, num_dofs], \n
            "dof_vel": torch.Tensor[num_envs, num_dofs], \n
        }
        """
        raise NotImplementedError()
    
    def set_actor_actions(
        self,
        actor_name: str,
        actor_actions,
    ):
        """
        set the batched actions of an actor
        
        Args:
        - actor_name: str, name of the actor
        - actor_actions: torch.Tensor[num_envs, num_dofs], batched control signals
        """
        raise NotImplementedError()
    
    def disable_gravity(
        self,
        actor_name: str,
    ):
        """
        disable gravity for an actor
        
        Args:
        - actor_name: str, name of the actor
        """
        raise NotImplementedError()
    
    def enable_gravity(
        self,
        actor_name: str,
    ):
        """
        enable gravity for an actor
        
        Args:
        - actor_name: str, name of the actor
        """
        raise NotImplementedError()
    
    def close(
        self, 
    ):
        """
        close the simulator
        """
        raise NotImplementedError()


def get_simulator(simulator_type: str):
    """
    get simulator class
    
    Args:
    - simulator_type: str, type of the simulator
    Returns:
    - type[Simulator], simulator class
    """
    if simulator_type == 'IsaacGymSimulator':
        from src.utils.simulator.isaacgym_simulator import IsaacGymSimulator
        return IsaacGymSimulator
    else:
        raise ValueError(f"simulator_type: {simulator_type} is not supported")

def get_squeeze_params(qpos_dict, width_mapper, args):
    if args.robot_name == 'leap_hand' and args.squeeze_mode == 'equal':
        pregrasp_qpos_dict = width_mapper.squeeze_fingers(
            qpos_dict, -0.04, -0.04)[0][:, 9:]
        target_qpos_dict = width_mapper.squeeze_fingers(
            qpos_dict, 0.04, 0.04)[0][:, 9:]
    elif args.robot_name == 'leap_hand' and args.squeeze_mode == 'diff':
        pregrasp_qpos_dict = width_mapper.squeeze_fingers(
            qpos_dict, -0.04, -0.02)[0][:, 9:]
        target_qpos_dict = width_mapper.squeeze_fingers(
            qpos_dict, 0.02, 0.01)[0][:, 9:]
    else:
        raise NotImplementedError
    
    return pregrasp_qpos_dict, target_qpos_dict